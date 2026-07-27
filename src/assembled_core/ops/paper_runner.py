"""OPS-6: Paper daily runner — callable helper for one-day and range runs."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# run_paper_daily_one helpers (_prd_*)
# ---------------------------------------------------------------------------


def _prd_load_paper_state(
    mode: str,
    app_cfg: dict[str, Any],
    prices: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    root: Path,
    start_capital: float,
) -> tuple[
    dict | None, Path | None, float, pd.DataFrame, "pd.Series | None", int | None
]:
    """Load paper ledger state when mode='paper'. Returns 6-tuple of state vars."""
    if mode != "paper":
        empty_pos = pd.DataFrame(columns=["symbol", "qty", "target_qty"])
        return None, None, start_capital, empty_pos, None, None

    from src.assembled_core.ops.paper_ledger import (
        load_ledger_state,
        mark_to_market_equity,
    )

    paper_cfg = app_cfg.get("paper_runner") or {}
    ledger_path_str = (
        paper_cfg.get("ledger_path") or "output/runs/_paper_ledger/ledger_state.json"
    )
    ledger_path = (
        root / ledger_path_str
        if not Path(ledger_path_str).is_absolute()
        else Path(ledger_path_str)
    )
    ledger_state = load_ledger_state(ledger_path, start_capital=start_capital)
    if (
        not prices.empty
        and "timestamp" in prices.columns
        and "symbol" in prices.columns
    ):
        p_ts = pd.to_datetime(prices["timestamp"], utc=True)
        prices_cut = prices.loc[p_ts <= as_of_ts]
        if not prices_cut.empty:
            prices_latest_mtm = (
                prices_cut.groupby("symbol", group_keys=False).last().reset_index()
            )
            equity_before = mark_to_market_equity(ledger_state, prices_latest_mtm)
        else:
            equity_before = ledger_state.get("cash") or start_capital
    else:
        equity_before = ledger_state.get("cash") or start_capital

    pos_list = [
        {"symbol": sym, "qty": p["qty"], "target_qty": p["qty"]}
        for sym, p in (ledger_state.get("positions") or {}).items()
        if float(p.get("qty", 0)) != 0
    ]
    current_positions_df = (
        pd.DataFrame(pos_list)
        if pos_list
        else pd.DataFrame(columns=["symbol", "qty", "target_qty"])
    )
    curve = ledger_state.get("equity_curve") or []
    if curve:
        _series = pd.Series([float(c.get("equity", 0)) for c in curve], dtype=float)
        equity_series: pd.Series | None = _series
        equity_curve_index: int | None = len(_series) - 1
    else:
        equity_series = None
        equity_curve_index = None

    return (
        ledger_state,
        ledger_path,
        equity_before,
        current_positions_df,
        equity_series,
        equity_curve_index,
    )


def _prd_make_strategy_fns(
    strategy_name: str,
    strategy_cfg: dict[str, Any],
    ledger_state: dict[str, Any] | None,
) -> tuple[Any, Any]:
    """Build (signal_fn, position_sizing_fn) closures for the given strategy."""

    def _no_signal_fn(prices_with_features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    def _no_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    if strategy_name == "ema_trend_v0":
        from src.assembled_core.strategies.ema_trend_v0 import (
            check_exit_signals as ema_check_exits,
        )
        from src.assembled_core.strategies.ema_trend_v0 import (
            compute_signals as ema_compute_signals,
        )
        from src.assembled_core.strategies.ema_trend_v0 import (
            compute_target_positions as ema_compute_targets,
        )

        ema_fast = int(strategy_cfg.get("ema_fast") or 20)
        ema_slow = int(strategy_cfg.get("ema_slow") or 60)
        equal_weight = bool(strategy_cfg.get("equal_weight", True))
        max_positions = int(strategy_cfg.get("max_positions") or 0)
        min_position_weight = float(strategy_cfg.get("min_position_weight") or 0.0)
        target_invested_pct = float(strategy_cfg.get("target_invested_pct") or 1.0)

        def _ema_signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            signals = ema_compute_signals(df, ema_fast=ema_fast, ema_slow=ema_slow)
            if ledger_state and (ledger_state.get("positions") or {}):
                prices_latest_exit = None
                if df is not None and not df.empty and "close" in df.columns:
                    prices_latest_exit = (
                        df.groupby("symbol", group_keys=False)["close"]
                        .last()
                        .reset_index()
                    )
                exit_signals = ema_check_exits(
                    ledger_state.get("positions", {}),
                    prices_latest_exit,
                    strategy_cfg,
                )
                if not exit_signals.empty:
                    full_exits = exit_signals[exit_signals["exit_qty_pct"] >= 1.0]
                    if not full_exits.empty:
                        exit_syms = set(full_exits["symbol"])
                        if not signals.empty:
                            signals = signals[~signals["symbol"].isin(exit_syms)]
                        for ex in full_exits.itertuples(index=False):
                            log.info(
                                "[EMA] EXIT signal: %s — %s", ex.symbol, ex.exit_reason
                            )
                    for ex in exit_signals[
                        exit_signals["exit_qty_pct"] < 1.0
                    ].itertuples(index=False):
                        log.info(
                            "[EMA] PARTIAL EXIT signal: %s (%.0f%%) — %s",
                            ex.symbol,
                            ex.exit_qty_pct * 100,
                            ex.exit_reason,
                        )
            return signals

        def _ema_sizing(sig: pd.DataFrame, cap: float) -> pd.DataFrame:
            return ema_compute_targets(
                sig,
                cap,
                equal_weight=equal_weight,
                max_positions=max_positions,
                min_position_weight=min_position_weight,
                target_invested_pct=target_invested_pct,
            )

        return _ema_signal_fn, _ema_sizing

    if strategy_name in ("multifactor_v1", "multifactor_v2"):
        if strategy_name == "multifactor_v2":
            from src.assembled_core.strategies.multifactor_v2 import (
                check_exit_signals as mf_check_exits,
            )
            from src.assembled_core.strategies.multifactor_v2 import (
                compute_signals as mf_compute_signals,
            )
            from src.assembled_core.strategies.multifactor_v2 import (
                compute_target_positions as mf_compute_targets,
            )

            _mf_tag = "[MF-V2]"
        else:
            from src.assembled_core.strategies.multifactor_v1 import (
                check_exit_signals as mf_check_exits,
            )

            # v1's compute_signals lacks the kw-only ``as_of`` param that v2
            # has; the only call site below never passes as_of, so the
            # narrower v1 signature is safe. Type-level conflict only.
            from src.assembled_core.strategies.multifactor_v1 import (  # type: ignore[assignment]
                compute_signals as mf_compute_signals,
            )
            from src.assembled_core.strategies.multifactor_v1 import (
                compute_target_positions as mf_compute_targets,
            )

            _mf_tag = "[MF-V1]"

        max_positions = int(strategy_cfg.get("max_positions") or 10)
        min_position_weight = float(strategy_cfg.get("min_position_weight") or 0.03)
        target_invested_pct = float(strategy_cfg.get("target_invested_pct") or 0.80)
        equal_weight = bool(strategy_cfg.get("equal_weight", False))

        # Shared exit state: signal fn writes, sizing fn reads so that exits
        # force zero-weight targets even when signals are empty (e.g. bear regime).
        _exit_state: dict = {"full": set(), "partial": {}, "prices": {}}  # sym → price

        def _mf_signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            _exit_state["full"] = set()
            _exit_state["partial"] = {}
            _exit_state["prices"] = {}
            signals = mf_compute_signals(df, strategy_cfg=strategy_cfg)
            if ledger_state and (ledger_state.get("positions") or {}):
                prices_latest_exit = None
                if df is not None and not df.empty and "close" in df.columns:
                    prices_latest_exit = (
                        df.groupby("symbol", group_keys=False)["close"]
                        .last()
                        .reset_index()
                    )
                    # Store current prices for accurate hold-target computation
                    _exit_state["prices"] = dict(
                        zip(prices_latest_exit["symbol"], prices_latest_exit["close"])
                    )
                exit_signals = mf_check_exits(
                    ledger_state.get("positions", {}),
                    prices_latest_exit,
                    strategy_cfg,
                )
                if not exit_signals.empty:
                    full_exits = exit_signals[exit_signals["exit_qty_pct"] >= 1.0]
                    if not full_exits.empty:
                        exit_syms = set(full_exits["symbol"])
                        _exit_state["full"] = exit_syms
                        if not signals.empty:
                            signals = signals[~signals["symbol"].isin(exit_syms)]
                        for ex in full_exits.itertuples(index=False):
                            log.info(
                                "%s EXIT signal: %s — %s",
                                _mf_tag,
                                ex.symbol,
                                ex.exit_reason,
                            )
                    for ex in exit_signals[
                        exit_signals["exit_qty_pct"] < 1.0
                    ].itertuples(index=False):
                        _exit_state["partial"][ex.symbol] = float(ex.exit_qty_pct)
                        log.info(
                            "%s PARTIAL EXIT signal: %s (%.0f%%) — %s",
                            _mf_tag,
                            ex.symbol,
                            ex.exit_qty_pct * 100,
                            ex.exit_reason,
                        )
            return signals

        def _mf_sizing(sig: pd.DataFrame, cap: float) -> pd.DataFrame:
            targets = mf_compute_targets(
                sig,
                cap,
                equal_weight=equal_weight,
                max_positions=max_positions,
                min_position_weight=min_position_weight,
                target_invested_pct=target_invested_pct,
            )
            positions = (ledger_state or {}).get("positions", {})
            syms_in_targets = (
                set(targets["symbol"].astype(str)) if not targets.empty else set()
            )
            exit_syms_all = {str(s) for s in _exit_state.get("full", set())} | {
                str(s) for s in _exit_state.get("partial", {})
            }
            extra_rows: list[dict] = []

            # 1. Full exits → target_qty=0 → SELL entire position
            for sym in _exit_state.get("full", set()):
                if str(sym) not in syms_in_targets:
                    extra_rows.append(
                        {"symbol": sym, "target_weight": 0.0, "target_qty": 0.0}
                    )
                    log.debug(
                        "%s injecting zero target for full exit: %s", _mf_tag, sym
                    )

            # 2. Partial exits → target_qty = remaining notional after partial sell
            for sym, exit_pct in _exit_state.get("partial", {}).items():
                if str(sym) in syms_in_targets:
                    continue
                pos = positions.get(sym, {})
                current_qty = float(pos.get("qty", 0.0))
                avg_price = float(pos.get("avg_price", 0.0))
                if current_qty > 0 and avg_price > 0:
                    remaining_qty = current_qty * (1.0 - exit_pct)
                    remaining_notional = remaining_qty * avg_price
                    extra_rows.append(
                        {
                            "symbol": sym,
                            "target_weight": (
                                remaining_notional / cap if cap > 0 else 0.0
                            ),
                            "target_qty": remaining_notional,
                        }
                    )
                    log.debug(
                        "%s injecting partial target for %s: qty=%.2f→%.2f",
                        _mf_tag,
                        sym,
                        current_qty,
                        remaining_qty,
                    )

            # 3. Hold targets → preserve non-exit positions that are absent from targets
            #    using current_price so delta = current_qty * px / px = current_qty → 0 order.
            current_prices = _exit_state.get("prices", {})
            for sym, pos in positions.items():
                current_qty = float(pos.get("qty", 0.0))
                avg_price = float(pos.get("avg_price", 0.0))
                if current_qty <= 1e-6 or avg_price <= 0:
                    continue
                if str(sym) in syms_in_targets or str(sym) in exit_syms_all:
                    continue
                # Use current_price if available; fall back to avg_price
                cur_px = float(current_prices.get(sym, avg_price))
                hold_notional = (
                    current_qty * cur_px
                )  # target_qty / cur_px = current_qty → Δ = 0
                extra_rows.append(
                    {
                        "symbol": sym,
                        "target_weight": hold_notional / cap if cap > 0 else 0.0,
                        "target_qty": hold_notional,
                    }
                )
                log.debug(
                    "%s injecting hold target for %s (qty=%.2f @ %.2f)",
                    _mf_tag,
                    sym,
                    current_qty,
                    cur_px,
                )

            if extra_rows:
                targets = pd.concat(
                    [targets, pd.DataFrame(extra_rows)], ignore_index=True
                )
            return targets

        return _mf_signal_fn, _mf_sizing

    if strategy_name == "trend_baseline":
        # §9.6 (b) Phase 1 shadow-mode (2026-05-21): trend_baseline as a
        # registered paper strategy. Backtest 2025-01..05 showed +43.02% CAGR /
        # 1.44 Sharpe vs mfv2 -7.74% baseline.
        # §9.6 (b) Phase 2 (2026-05-21): promoted from shadow to primary-eligible.
        # Pre-condition (i) closed: tb_check_exits gives stop-loss / trailing /
        # take-profit discipline, not just LONG → FLAT via MA-flip.
        from src.assembled_core.strategies.trend_baseline import (
            check_exit_signals as tb_check_exits,
        )
        from src.assembled_core.strategies.trend_baseline import (
            compute_signals as tb_compute_signals,
        )
        from src.assembled_core.strategies.trend_baseline import (
            compute_target_positions as tb_compute_targets,
        )

        ma_fast = int(strategy_cfg.get("ma_fast") or 20)
        ma_slow = int(strategy_cfg.get("ma_slow") or 60)
        max_positions = int(strategy_cfg.get("max_positions") or 0)
        target_invested_pct = float(strategy_cfg.get("target_invested_pct") or 1.0)

        def _tb_signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            signals = tb_compute_signals(df, ma_fast=ma_fast, ma_slow=ma_slow)
            # Wire exit signals (mirrors ema_trend_v0 / multifactor_v2 pattern
            # in this same dispatch) — without this wire-up the exit gates
            # configured under strategy.stop_loss_pct etc. would be defined
            # but never consulted by the trading cycle.
            if ledger_state and (ledger_state.get("positions") or {}):
                prices_latest_exit = None
                if df is not None and not df.empty and "close" in df.columns:
                    prices_latest_exit = (
                        df.groupby("symbol", group_keys=False)["close"]
                        .last()
                        .reset_index()
                    )
                exit_signals = tb_check_exits(
                    ledger_state.get("positions", {}),
                    prices_latest_exit,
                    strategy_cfg,
                )
                if not exit_signals.empty:
                    full_exits = exit_signals[exit_signals["exit_qty_pct"] >= 1.0]
                    if not full_exits.empty:
                        exit_syms = set(full_exits["symbol"])
                        if not signals.empty:
                            signals = signals[~signals["symbol"].isin(exit_syms)]
                        for ex in full_exits.itertuples(index=False):
                            log.info(
                                "[TB] EXIT signal: %s — %s",
                                ex.symbol,
                                ex.exit_reason,
                            )
                    for ex in exit_signals[
                        exit_signals["exit_qty_pct"] < 1.0
                    ].itertuples(index=False):
                        log.info(
                            "[TB] PARTIAL EXIT signal: %s (%.0f%%) — %s",
                            ex.symbol,
                            ex.exit_qty_pct * 100,
                            ex.exit_reason,
                        )
            return signals

        def _tb_sizing(sig: pd.DataFrame, cap: float) -> pd.DataFrame:
            return tb_compute_targets(
                sig,
                cap,
                max_positions=max_positions,
                target_invested_pct=target_invested_pct,
            )

        return _tb_signal_fn, _tb_sizing

    return _no_signal_fn, _no_sizing_fn


def _prd_intel_summaries(
    result: Any,
    paper_cfg: dict[str, Any],
    root: Path,
) -> None:
    """Load news + disclosures trigger summaries into result.meta (in-place)."""
    import json as _json

    intel_cfg = paper_cfg.get("intel") or {}
    news_out = (intel_cfg.get("news") or {}).get("output_dir") or "output/intel/news"
    discl_out = (intel_cfg.get("disclosures") or {}).get(
        "output_dir"
    ) or "output/intel/disclosures"
    news_path = (
        root / news_out if not Path(news_out).is_absolute() else Path(news_out)
    ) / "triggers_latest.json"
    discl_path = (
        root / discl_out if not Path(discl_out).is_absolute() else Path(discl_out)
    ) / "triggers_latest.json"

    try:
        from src.assembled_core.intel import (
            load_disclosures_triggers,
            load_news_triggers,
        )

        news_snap = load_news_triggers(news_path)
        result.meta["news_triggers_summary"] = {
            "count": len(news_snap.triggers),
            "max_severity": news_snap.summary.get("max_severity", 0),
            "count_sev1plus": news_snap.summary.get("watch_count_sev1plus", 0),
            "count_sev2plus": news_snap.summary.get("active_count_sev2plus", 0),
            "status": "ok",
        }
    except Exception as exc:
        log.warning("[PaperRunner] news triggers unavailable: %s", exc)
        result.meta["news_triggers_summary"] = {
            "count": 0,
            "max_severity": 0,
            "count_sev1plus": 0,
            "count_sev2plus": 0,
            "status": "error",
            "error": str(exc),
        }

    try:
        funnel_path = news_path.parent / "debug_funnel_latest.json"
        if funnel_path.exists():
            funnel_data = _json.loads(funnel_path.read_text(encoding="utf-8"))
            if isinstance(funnel_data, dict) and "counts" in funnel_data:
                compact = dict(funnel_data["counts"])
                exc_reasons = funnel_data.get("normalize_exception_reasons") or {}
                none_reasons = funnel_data.get("normalize_none_reasons") or {}
                compact["normalize_exception_reasons_preview"] = dict(
                    list(exc_reasons.items())[:2]
                )
                compact["normalize_none_reasons_preview"] = dict(
                    list(none_reasons.items())[:2]
                )
                result.meta["news_debug_funnel"] = compact
    except Exception as exc:
        log.warning("[PaperRunner] failed to load news funnel: %s", exc)

    try:
        discl_snap = load_disclosures_triggers(discl_path)
        result.meta["disclosures_triggers_summary"] = {
            "count": len(discl_snap.triggers),
            "max_severity": discl_snap.summary.get("max_severity", 0),
            "count_sev1plus": discl_snap.summary.get("count_sev1plus", 0),
            "count_sev2plus": discl_snap.summary.get("count_sev2plus", 0),
            "status": "ok",
        }
    except Exception as exc:
        log.warning("[PaperRunner] disclosures triggers unavailable: %s", exc)
        result.meta["disclosures_triggers_summary"] = {
            "count": 0,
            "max_severity": 0,
            "count_sev1plus": 0,
            "count_sev2plus": 0,
            "status": "error",
            "error": str(exc),
        }


def _prd_paper_fills_and_ledger(
    *,
    mode: str,
    ledger_state: dict[str, Any],
    ledger_path: Path,
    result: Any,
    execution_mode: str,
    broker_adapter: Any | None,
    app_cfg: dict[str, Any],
    as_of_ts: pd.Timestamp,
    output_dir: Path,
    start_capital: float,
    strategy_name: str = "none",
    pilot_policy: dict[str, Any] | None = None,
) -> str | None:
    """Simulate/execute fills, update ledger, write journal+TCA+post-trade. Returns reconcile_status."""
    import copy

    from src.assembled_core.ops.paper_ledger import (
        append_equity_curve_deduped,
        apply_fills_to_ledger,
        mark_to_market_equity,
        save_ledger_state,
        simulate_fills,
        write_ledger_snapshot,
    )
    from src.assembled_core.ops.reconcile import (
        build_reconcile_report,
        write_reconcile_artifact,
    )

    orders_for_fills = (
        result.orders_filtered if not result.orders_filtered.empty else result.orders
    )
    prices_for_fills = result.prices_filtered
    if prices_for_fills.empty:
        prices_for_fills = result.prices_with_features
    _pilot_policy = (
        pilot_policy
        if pilot_policy is not None
        else _load_pilot_policy_fail_fast("cost_model")
    )
    cost_cfg = _resolve_cost_cfg(app_cfg, _pilot_policy)
    ledger_before = copy.deepcopy(ledger_state)

    # Almgren-Chriss impact + SOR annotation — reuses _pilot_policy (no second load_policy call)
    try:
        from src.assembled_core.ops.execution_cost_meta import annotate_execution_cost

        regime = getattr(result, "regime", None) or "bull"
        orders_for_fills, exec_meta = annotate_execution_cost(
            orders_for_fills,
            prices_for_fills,
            policy=_pilot_policy,
            regime=regime,
        )
        result.meta["execution_cost"] = exec_meta
    except Exception as _exec_exc:
        log.debug("[EXEC-COST] annotate_execution_cost skipped: %s", _exec_exc)

    _VALID_EXEC_MODES = ("sim", "broker", "dry_run")
    if execution_mode not in _VALID_EXEC_MODES:
        log.error("Unknown execution_mode=%r — falling back to sim", execution_mode)
        execution_mode = "sim"
    if execution_mode in ("broker", "dry_run") and broker_adapter is None:
        log.warning(
            "execution_mode=%r but broker_adapter is None — falling back to sim",
            execution_mode,
        )
        execution_mode = "sim"

    broker_exec_meta: dict[str, Any] = {}
    if execution_mode == "broker" and broker_adapter is not None:
        from src.assembled_core.execution.broker_execution import execute_via_broker

        exec_result = execute_via_broker(
            broker_adapter, orders_for_fills, dry_run=False
        )
        fills = exec_result.fills_for_ledger
        broker_exec_meta = {
            "filled": len(exec_result.filled),
            "rejected": len(exec_result.rejected),
            "timed_out": len(exec_result.timed_out),
            "errors": exec_result.errors,
            "execution_time_s": exec_result.execution_time_s,
        }
    elif execution_mode == "dry_run" and broker_adapter is not None:
        from src.assembled_core.execution.broker_execution import execute_via_broker

        exec_result = execute_via_broker(broker_adapter, orders_for_fills, dry_run=True)
        fills = []
        broker_exec_meta = {
            "filled": 0,
            "rejected": 0,
            "timed_out": 0,
            "dry_run": True,
            "would_submit": len(exec_result.submitted),
        }
    else:
        orders_for_sim = orders_for_fills
        partial_cfg = (cost_cfg or {}).get("partial_fill") or {}
        if partial_cfg.get("enabled", False) and not orders_for_fills.empty:
            try:
                from src.assembled_core.execution.fill_model import (
                    PartialFillModel,
                    apply_partial_fills,
                )

                pfm = PartialFillModel(
                    participation_cap=float(partial_cfg.get("participation_cap", 0.1)),
                    min_fill_qty=float(partial_cfg.get("min_fill_qty", 1.0)),
                    adv_window=int(partial_cfg.get("adv_window", 20)),
                    fallback_fill_ratio=float(
                        partial_cfg.get("fallback_fill_ratio", 1.0)
                    ),
                )
                clipped = apply_partial_fills(
                    orders_for_fills, prices=prices_for_fills, partial_fill_model=pfm
                )
                if not clipped.empty and "fill_qty" in clipped.columns:
                    orders_for_sim = orders_for_fills.copy()
                    qty_map = dict(
                        zip(
                            clipped["symbol"].astype(str),
                            clipped["fill_qty"].astype(float),
                        )
                    )
                    orders_for_sim["qty"] = orders_for_sim["symbol"].map(
                        lambda s, _m=qty_map: _m.get(str(s), 0.0)
                    )
                    orders_for_sim = orders_for_sim[orders_for_sim["qty"] > 0]
                    result.meta["partial_fill"] = {
                        "n_orders_in": int(len(orders_for_fills)),
                        "n_orders_out": int(len(orders_for_sim)),
                        "participation_cap": pfm.participation_cap,
                    }
            except Exception as exc:
                log.debug("[PaperRunner] partial_fill skipped: %s", exc)
        fills = simulate_fills(orders_for_sim, prices_for_fills, cost_cfg)

    state_after = apply_fills_to_ledger(ledger_state, fills)
    equity_after = mark_to_market_equity(state_after, prices_for_fills)
    now_iso = pd.Timestamp.now("UTC").isoformat()
    append_equity_curve_deduped(state_after, now_iso, equity_after)

    # Update drawdown damper only when mfv2 is the active strategy — prevents
    # cross-contamination of mfv2 state with equity from other strategies.
    if strategy_name == "multifactor_v2":
        try:
            from src.assembled_core.strategies.multifactor_v2 import (
                update_drawdown_damper,
            )

            update_drawdown_damper(float(equity_after), as_of_ts.date())
        except ImportError as _imp_exc:
            log.debug("[paper] DD-damper import failed: %s", _imp_exc)
        except Exception as _exc:
            log.debug("[paper] update_drawdown_damper raised %s at %s", _exc, as_of_ts)
    report = build_reconcile_report(
        as_of_utc=now_iso,
        ledger_before=ledger_before,
        ledger_after=state_after,
        orders=orders_for_fills,
        fills=fills,
        prices_latest=prices_for_fills,
        cost_model_cfg=cost_cfg,
    )
    write_reconcile_artifact(output_dir, report)
    # K5 (2026-07-21): stable root-level copy for the reconcile-block gate.
    # apply_reconcile_block_gate reads <root>/output/reconcile_latest.json,
    # but the artifact was only ever written into the per-run directory —
    # an ARMED gate could therefore never verify a pass and would block
    # every cycle fail-closed. Derive the stable location from the per-run
    # layout (<root>/output/runs/<run_id>); refuse to guess on unexpected
    # layouts so we never scatter artifacts into arbitrary directories.
    try:
        _stable_dir = output_dir.resolve().parent.parent
        if _stable_dir.name == "output":
            write_reconcile_artifact(_stable_dir, report)
        else:
            log.warning(
                "[reconcile] unexpected output_dir layout %s — stable "
                "reconcile_latest.json NOT written (armed gate would fail closed)",
                output_dir,
            )
    except Exception as _stable_exc:  # noqa: BLE001 — Stage-1 O6: any failure
        # here must not skip the ledger save below; the armed gate fails
        # closed on a stale artifact, which is the visible symptom.
        log.warning(
            "[reconcile] stable reconcile_latest.json write failed: %s",
            _stable_exc,
        )
    reconcile_status = report.get("status") or "OK"
    save_ledger_state(state_after, ledger_path)
    write_ledger_snapshot(output_dir, state_after, equity_after)
    result.meta["paper_ledger"] = {
        "equity": equity_after,
        "equity_curve_point": {"utc": now_iso, "equity": equity_after},
    }
    if broker_exec_meta:
        result.meta["broker_execution"] = broker_exec_meta

    if fills:
        try:
            from src.assembled_core.ops.trade_journal import (
                append_trade_journal_entries,
                write_daily_summary,
            )

            append_trade_journal_entries(
                fills,
                signal_context=None,
                ledger_state=ledger_before,
                run_id=str(output_dir.name) if output_dir else "",
            )
            write_daily_summary(
                date_str=as_of_ts.strftime("%Y-%m-%d"),
                ledger_state=state_after,
                equity=equity_after,
                start_capital=start_capital,
                fills=fills,
            )
        except Exception as exc:
            log.warning("[PaperRunner] trade journal write failed: %s", exc)

        try:
            from src.assembled_core.qa.tca import compute_implementation_shortfall

            fills_df = pd.DataFrame(
                [
                    {
                        "symbol": f["symbol"],
                        "side": f.get("side", "BUY"),
                        "fill_price": float(f.get("price", 0.0)),
                        "fill_qty": float(f.get("qty", 0.0)),
                    }
                    for f in fills
                ]
            )
            if (
                not fills_df.empty
                and orders_for_fills is not None
                and not orders_for_fills.empty
                and "arrival_price" in orders_for_fills.columns
            ):
                arrival_lookup = orders_for_fills[
                    ["symbol", "arrival_price"]
                ].drop_duplicates("symbol")
                fills_df = fills_df.merge(arrival_lookup, on="symbol", how="left")
            is_df = compute_implementation_shortfall(fills_df)
            if not is_df.empty and "is_bps" in is_df.columns:
                is_vals = pd.to_numeric(is_df["is_bps"], errors="coerce").dropna()
                result.meta["tca"] = {
                    "n_fills": int(len(is_df)),
                    "avg_is_bps": float(is_vals.mean()) if not is_vals.empty else 0.0,
                    "max_is_bps": float(is_vals.max()) if not is_vals.empty else 0.0,
                    "min_is_bps": float(is_vals.min()) if not is_vals.empty else 0.0,
                }
        except Exception as exc:
            log.debug("[PaperRunner] TCA IS compute failed: %s", exc)

        try:
            from src.assembled_core.qa.learning_store import append_learning_record
            from src.assembled_core.qa.post_trade_analyzer import (
                build_learning_record,
                compute_forward_returns,
                compute_signal_hit_rate,
            )

            prices_for_analysis = result.prices_with_features
            if not prices_for_analysis.empty and "close" in prices_for_analysis.columns:
                fwd_returns = compute_forward_returns(
                    prices_for_analysis, horizon_days=5
                )
                trades_for_analysis = pd.DataFrame(
                    [
                        {
                            "symbol": f["symbol"],
                            "side": f.get("side", "BUY"),
                            "event_ts": as_of_ts,
                            "qty": f.get("qty", 0),
                            "price": f.get("price", 0),
                        }
                        for f in fills
                    ]
                )
                if not trades_for_analysis.empty and not fwd_returns.empty:
                    hit_rate_df = compute_signal_hit_rate(
                        trades_for_analysis, fwd_returns
                    )
                    if not hit_rate_df.empty:
                        record = build_learning_record(
                            run_id=str(output_dir.name) if output_dir else "",
                            analysis_date=as_of_ts.strftime("%Y-%m-%d"),
                            hit_rate_df=hit_rate_df,
                        )
                        append_learning_record(record)
                        result.meta["post_trade_analysis"] = {
                            "symbols_analyzed": len(hit_rate_df),
                            "avg_hit_rate": float(hit_rate_df["hit_rate"].mean()),
                        }
        except Exception as exc:
            log.warning("[PaperRunner] post-trade analysis failed: %s", exc)

    return reconcile_status


def _prd_write_artifacts(
    *,
    mode: str,
    ctx: Any,
    result: Any,
    output_dir: Path,
    as_of_ts: pd.Timestamp,
    app_cfg: dict[str, Any],
    reconcile_status: str | None,
    execution_mode: str,
) -> None:
    """Write KPIs, targets, orders, reasons, diff, alerts artifacts."""
    import json as _json

    from src.assembled_core.config.policy_loader import load_policy
    from src.assembled_core.ops.alerts import (
        compute_alerts,
        make_reconcile_fail_alert,
        write_alerts_artifact,
    )
    from src.assembled_core.ops.kpi_artifacts import (
        maybe_execute_orders,
        write_diff_vs_prev,
        write_orders_artifact,
        write_reasons_artifact,
        write_run_kpis,
        write_targets_artifact,
    )

    if execution_mode == "sim":
        _ = maybe_execute_orders(
            mode,
            (
                result.orders_filtered
                if not result.orders_filtered.empty
                else result.orders
            ),
        )

    policy = load_policy()
    kpis_path = write_run_kpis(
        output_dir=output_dir, ctx=ctx, result=result, policy=policy, mode=mode
    )
    write_targets_artifact(
        output_dir=output_dir, target_positions=result.target_positions
    )
    orders_df = (
        result.orders_filtered if not result.orders_filtered.empty else result.orders
    )
    write_orders_artifact(output_dir=output_dir, orders=orders_df)
    write_reasons_artifact(
        output_dir=output_dir, ctx=ctx, result=result, policy=policy, mode=mode
    )

    try:
        current_kpis = _json.loads(kpis_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("[PaperRunner] failed to read run_kpis for diff: %s", exc)
        current_kpis = {}
    prev_date = as_of_ts.date() - timedelta(days=1)
    prev_dir = output_dir.parent / prev_date.isoformat()
    write_diff_vs_prev(
        output_dir=output_dir,
        prev_dir=prev_dir,
        current_targets=result.target_positions,
        current_kpis=current_kpis,
    )

    if app_cfg.get("alerts", {}).get("enabled", True):
        out = Path(output_dir)
        reasons_data: dict[str, Any] = {}
        diff_data: dict[str, Any] = {}
        if (out / "reasons_latest.json").exists():
            try:
                reasons_data = _json.loads(
                    (out / "reasons_latest.json").read_text(encoding="utf-8")
                )
            except Exception as exc:
                log.warning("[PaperRunner] failed to read reasons_latest.json: %s", exc)
        if (out / "diff_vs_prev.json").exists():
            try:
                diff_data = _json.loads(
                    (out / "diff_vs_prev.json").read_text(encoding="utf-8")
                )
            except Exception as exc:
                log.warning("[PaperRunner] failed to read diff_vs_prev.json: %s", exc)
        generated_utc = current_kpis.get("generated_utc") or ""
        alerts_list = compute_alerts(current_kpis, reasons_data, diff_data, app_cfg)
        if mode == "paper" and reconcile_status == "FAIL":
            alerts_list = list(alerts_list)
            alerts_list.append(make_reconcile_fail_alert(generated_utc))
            severity_map = app_cfg.get("alerts", {}).get("severity_map") or {
                "info": 0,
                "warn": 1,
                "critical": 2,
            }
            alerts_list.sort(
                key=lambda a: (
                    -severity_map.get(a["level"], 0),
                    a["kind"],
                    a["alert_id"],
                )
            )
        write_alerts_artifact(output_dir, alerts_list, generated_utc, app_cfg)


def _prd_run_shadow_strategy(
    *,
    prices: pd.DataFrame,
    paper_cfg: dict[str, Any],
    output_dir: Path,
    as_of_ts: pd.Timestamp,
    primary_signals: pd.DataFrame | None,
) -> None:
    """§9.6 (b) Phase 1 shadow-mode (2026-05-21): run a secondary strategy
    in parallel for signal comparison, without affecting orders/broker.

    Reads ``paper_cfg.shadow_strategy.{enabled,name,...}`` from app_cfg.
    When ``enabled=True`` (default ``False``), builds the shadow strategy via
    ``_prd_make_strategy_fns`` and computes signals + target positions on the
    same prices the main cycle saw. Writes ``shadow_signals.json`` +
    ``shadow_targets.json`` to ``output_dir`` and logs a one-line comparison
    summary against the primary signals (LONG counts, symbol overlap).

    The shadow path:
        - DOES NOT generate orders
        - DOES NOT touch broker
        - DOES NOT mutate any ledger / position state
        - Pure observation for evaluating strategy switch candidates over N days

    Best-effort: any exception logged at WARNING and execution continues —
    shadow-mode failure must never block the primary pilot cycle.

    Note on input asymmetry (F-S1-M2 §9.6 (b) Phase-2 pre-cond (iii)
    closure 2026-05-21): the shadow signal_fn receives the RAW prices
    panel passed in here. The primary cycle's signal_fn (via
    run_trading_cycle → build_features) receives a feature-enriched
    panel with ta_rsi_14_v1, trend_ema_spread, ta_macd_*, etc.
    Price-only strategies (trend_baseline, ema_trend_v0) compute their
    own MAs and are unaffected. multifactor_v1/v2 as shadows would
    silently degrade to zero-factor signals on raw prices — the
    whitelist below blocks those names from running as shadow until
    a feature-pipeline-aligned shadow path is added.
    """
    import json

    # F-S1-M1 §9.6 (b) Phase-2 pre-cond (ii) closure 2026-05-21:
    # whitelist known price-only shadow-safe strategies. Unknown names log
    # a loud WARNING and are skipped — the previous silent fall-through
    # to _no_signal_fn produced empty payloads + misleading "n_long=0"
    # comparison lines indistinguishable from genuine "no signals today".
    _PRICE_ONLY_SHADOW_WHITELIST = {"trend_baseline", "ema_trend_v0"}

    shadow_cfg = paper_cfg.get("shadow_strategy") or {}
    if not shadow_cfg.get("enabled", False):
        return

    shadow_name = (shadow_cfg.get("name") or "").strip().lower()
    if not shadow_name:
        log.debug("[shadow] shadow_strategy.enabled=True but name missing — skipping")
        return

    if shadow_name not in _PRICE_ONLY_SHADOW_WHITELIST:
        log.warning(
            "[shadow] strategy '%s' not in shadow-safe whitelist %s — "
            "skipping (feature-pipeline-aligned shadow not implemented for "
            "feature-dependent strategies; see KNOWN_ISSUES.md §9.6 (b) "
            "Phase-2 pre-cond (iii))",
            shadow_name,
            sorted(_PRICE_ONLY_SHADOW_WHITELIST),
        )
        return

    try:
        shadow_signal_fn, shadow_sizing_fn = _prd_make_strategy_fns(
            shadow_name, shadow_cfg, ledger_state=None
        )
        shadow_signals_full = shadow_signal_fn(prices)
        if shadow_signals_full is None:
            shadow_signals_full = pd.DataFrame(
                columns=["timestamp", "symbol", "direction", "score"]
            )

        # Comparison + sizing must operate on the LATEST bar per symbol
        # (apples-to-apples vs primary which already comes out of
        # run_trading_cycle at the as_of point). The full signal history
        # remains the persisted artifact for post-hoc analysis.
        if (
            not shadow_signals_full.empty
            and "timestamp" in shadow_signals_full.columns
            and "symbol" in shadow_signals_full.columns
        ):
            shadow_signals = (
                shadow_signals_full.sort_values("timestamp")
                .groupby("symbol", group_keys=False)
                .tail(1)
                .reset_index(drop=True)
            )
        else:
            shadow_signals = shadow_signals_full

        # Compute hypothetical target positions on the latest bar's capital
        # baseline (start_capital — shadow doesn't track its own equity).
        start_capital = float(paper_cfg.get("start_capital", 100000.0))
        try:
            shadow_targets = shadow_sizing_fn(shadow_signals, start_capital)
        except Exception as exc:  # noqa: BLE001
            log.warning("[shadow] sizing_fn failed: %s — targets skipped", exc)
            shadow_targets = pd.DataFrame(
                columns=["symbol", "target_weight", "target_qty"]
            )

        # Persist for post-hoc comparison.
        output_dir.mkdir(parents=True, exist_ok=True)
        signals_path = output_dir / "shadow_signals.json"
        targets_path = output_dir / "shadow_targets.json"
        signals_payload = {
            "as_of_utc": as_of_ts.isoformat(),
            "strategy": shadow_name,
            "config": {k: v for k, v in shadow_cfg.items() if k not in ("enabled",)},
            # n_signals_total counts the LATEST-bar slice (apples-to-apples
            # with primary signals). Full history kept under
            # n_signals_full_history for post-hoc analysis.
            "n_signals_total": int(len(shadow_signals)),
            "n_long": (
                int((shadow_signals["direction"] == "LONG").sum())
                if "direction" in shadow_signals.columns
                else 0
            ),
            "n_signals_full_history": int(len(shadow_signals_full)),
            "signals": (
                shadow_signals.assign(
                    timestamp=shadow_signals["timestamp"].astype(str)
                ).to_dict(orient="records")
                if not shadow_signals.empty
                else []
            ),
        }
        targets_payload = {
            "as_of_utc": as_of_ts.isoformat(),
            "strategy": shadow_name,
            "n_targets": int(len(shadow_targets)),
            "targets": (
                shadow_targets.to_dict(orient="records")
                if not shadow_targets.empty
                else []
            ),
        }
        signals_path.write_text(json.dumps(signals_payload, indent=2), encoding="utf-8")
        targets_path.write_text(json.dumps(targets_payload, indent=2), encoding="utf-8")

        # Comparison summary against primary signals.
        primary_long_count = 0
        overlap_count = 0
        if primary_signals is not None and not primary_signals.empty:
            if "direction" in primary_signals.columns:
                primary_long = primary_signals[primary_signals["direction"] == "LONG"]
                primary_long_count = int(len(primary_long))
                if not shadow_signals.empty and "direction" in shadow_signals.columns:
                    shadow_long = shadow_signals[shadow_signals["direction"] == "LONG"]
                    overlap_count = int(
                        len(
                            set(primary_long["symbol"]).intersection(
                                set(shadow_long["symbol"])
                            )
                        )
                    )
        log.info(
            "[shadow] %s — %d LONGs (primary=%d LONGs, overlap=%d). Persisted to %s.",
            shadow_name,
            signals_payload["n_long"],
            primary_long_count,
            overlap_count,
            output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[shadow] shadow strategy '%s' failed (non-fatal, primary cycle "
            "continues): %s",
            shadow_name,
            exc,
        )


def _load_pilot_policy_fail_fast(context: str) -> dict[str, Any]:
    """Load policy.yaml; re-raise hard parse errors, fall back on soft failures.

    Hard errors (malformed YAML, non-mapping top-level) → log.error + re-raise.
    Soft failures (file absent, import error, etc.) → log.warning + return {}.
    """
    try:
        from src.assembled_core.config.policy_loader import load_policy as _lp

        return _lp()
    except Exception as _exc:  # noqa: BLE001
        _is_hard = isinstance(_exc, ValueError)
        if not _is_hard:
            try:
                import yaml as _yaml

                _is_hard = isinstance(_exc, _yaml.YAMLError)
            except ImportError:
                pass
        if _is_hard:
            log.error(
                "[paper_runner] policy.yaml parse error (%s) — aborting: %s",
                context,
                _exc,
            )
            raise
        log.warning(
            "[paper_runner] policy load failed for %s (fallback to app_cfg): %s",
            context,
            _exc,
        )
        return {}


def _resolve_active_strategy(paper_cfg: dict[str, Any], policy: dict[str, Any]) -> str:
    """Resolve active strategy: policy.paper_pilot.active_strategy > app_cfg strategy."""
    app_name = (
        str((paper_cfg.get("strategy") or {}).get("name") or "none").strip().lower()
    )
    pol_name = (
        str((policy.get("paper_pilot") or {}).get("active_strategy") or "")
        .strip()
        .lower()
    )
    if pol_name and pol_name != "none":
        if pol_name != app_name:
            log.info(
                "[paper_runner] active_strategy overridden by policy: %r → %r",
                app_name,
                pol_name,
            )
        return pol_name
    return app_name


def _resolve_cost_cfg(
    app_cfg: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Resolve cost model: policy.paper_pilot.cost_model > app_cfg.paper_runner.cost_model.

    Fail-CLOSED: if NEITHER source supplies a non-empty cost_model, an empty dict
    would make simulate_fills/paper_ledger treat commission_bps and slippage_bps as
    0 — fills at exact close, silently optimistic paper P&L (A13). Instead, log a
    loud WARNING and return the conservative policy.yaml-documented default.
    """
    app_cost = (app_cfg.get("paper_runner") or {}).get("cost_model") or {}
    pol_cost = (policy.get("paper_pilot") or {}).get("cost_model") or {}
    if pol_cost:
        log.info(
            "[paper_runner] cost_model loaded from policy (paper_pilot.cost_model)"
        )
        return dict(pol_cost)
    if app_cost:
        return dict(app_cost)  # defensive copy — prevent caller mutation of app_cfg
    # Neither policy nor app_cfg supplied a cost_model — never silently bill 0 bps.
    log.warning(
        "[paper_runner] NO cost_model in policy.paper_pilot OR app_cfg.paper_runner "
        "— falling back to conservative default "
        "{commission_bps:10.0, spread_w:0.25, impact_w:0.5}; "
        "paper fills would otherwise be modelled at exact close (0 bps)"
    )
    return {"commission_bps": 10.0, "spread_w": 0.25, "impact_w": 0.5}


def run_paper_daily_one(
    as_of_ts: pd.Timestamp,
    output_dir: Path,
    mode: str,
    app_cfg: dict[str, Any],
    prices: pd.DataFrame,
    *,
    root: Path,
    day_index: int | None = None,
    execution_mode: str = "sim",
    broker_adapter: Any | None = None,
) -> tuple[int, str | None]:
    """Run a single paper/shadow day. Returns (exit_code, reconcile_status or None)."""
    from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
    from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle

    # Normalize the routing-mode label BEFORE it is frozen into the ctx and
    # stamped into KPI/manifest/journal/heartbeat artifacts (F-senior-3):
    # same validation + sim-fallback as _prd_paper_fills_and_ledger applies
    # later on its local variable — an unknown label must never masquerade
    # as a real routing mode in ops artifacts.
    if execution_mode not in ("sim", "broker", "dry_run"):
        log.error(
            "Unknown execution_mode=%r — falling back to sim (label + routing)",
            execution_mode,
        )
        execution_mode = "sim"

    paper_cfg = app_cfg.get("paper_runner") or {}
    start_capital = float(paper_cfg.get("start_capital", 100000.0))

    (
        ledger_state,
        ledger_path,
        equity_before,
        current_positions_df,
        equity_series,
        equity_curve_index,
    ) = _prd_load_paper_state(mode, app_cfg, prices, as_of_ts, root, start_capital)

    strategy_cfg = paper_cfg.get("strategy") or {}
    _pilot_policy = _load_pilot_policy_fail_fast("strategy")
    strategy_name = _resolve_active_strategy(paper_cfg, _pilot_policy)
    signal_fn, position_sizing_fn = _prd_make_strategy_fns(
        strategy_name, strategy_cfg, ledger_state
    )

    intel_cfg = paper_cfg.get("intel") or {}
    intel_mode = (intel_cfg.get("mode") or "sim").strip().lower()
    intel_orchestration: dict[str, Any] = {}
    if intel_mode == "real":
        from src.assembled_core.ops.intel_orchestrator import run_intel_pipelines

        intel_orchestration = run_intel_pipelines(app_cfg, root=root)

    ctx = TradingContext(
        prices=prices,
        as_of=as_of_ts,
        freq="1d",
        mode="eod",
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=start_capital,
        write_outputs=False,
        enable_risk_controls=True,
        # E-059 #2: route the real order-routing mode (sim|broker|dry_run) into
        # the cycle so KPI/manifest/journal/heartbeat consumers see it.
        execution_mode=execution_mode,
    )
    if mode == "paper" and ledger_state is not None:
        ctx.capital = equity_before
        ctx.current_positions = (
            current_positions_df if not current_positions_df.empty else None
        )
        ctx.equity_curve = equity_series
        ctx.equity_curve_index = equity_curve_index

    intel_sim_cfg = paper_cfg.get("intel_sim") or {}
    if (
        intel_mode == "sim"
        and intel_sim_cfg.get("enabled", False)
        and day_index is not None
    ):
        from src.assembled_core.ops.intel_sim import apply_intel_sim

        apply_intel_sim(ctx, day_index, intel_sim_cfg)

    try:
        from src.assembled_core.paper.intel_context import populate_ctx_from_artifacts

        populate_ctx_from_artifacts(ctx, root)
    except Exception as _ctx_exc:
        log.debug("[INTEL-CTX] populate_ctx_from_artifacts failed: %s", _ctx_exc)

    # ------------------------------------------------------------------
    # Pre-cycle gates (Wave 20 — wiring):
    #   * halt_cache_gate — populates ctx.halted_symbols from a file feed
    #     so the existing filter in _tc_sizing.size_positions drops them.
    #   * tilt_gate — runs detect_tilt on the ledger equity curve. With
    #     policy.tilt.block_orders=true a fired rule short-circuits the
    #     cycle and returns 0 with no orders.
    # Both gates are default-off and have no effect unless explicitly
    # enabled in policy.yaml under `paper_runner.halt_cache.enabled` /
    # `paper_runner.tilt.enabled`.
    # ------------------------------------------------------------------
    try:
        from src.assembled_core.ops._paper_runner_gates import (
            apply_halt_cache_gate,
            apply_tilt_gate,
        )

        apply_halt_cache_gate(ctx, paper_cfg=paper_cfg, root=root)
        tilt_decision = apply_tilt_gate(
            ctx, paper_cfg=paper_cfg, ledger_state=ledger_state
        )
        if tilt_decision.blocked:
            log.warning(
                "[tilt] block_orders=true and rules fired %s — skipping cycle",
                tilt_decision.triggered_rules,
            )
            return 0, "tilt_blocked"
    except Exception as _gate_exc:  # noqa: BLE001 — gates are best-effort
        log.warning("[paper-runner-gates] gate failure (continuing): %s", _gate_exc)

    # ------------------------------------------------------------------
    # Reconcile-block gate (SAFETY, default-off): if the last reconcile
    # did not provably pass, refuse to trade this cycle. This is the
    # next-cycle blocking seam scoped out at orchestrator.py:864-868 /
    # FU-1. Lives ONLY in this live/paper driver — backtest uses
    # qa/backtest_engine.py and never reaches here, so a historical
    # replay never reads the live reconcile_latest.json. ARMED behaviour
    # is fail-closed (FAIL / unverified / unreadable -> block). With
    # paper_runner.reconcile_block.enabled=false (default) this is a pure
    # pass-through and behaviour is byte-identical.
    # NOTE: this gate's short-circuit is intentionally OUTSIDE the
    # best-effort try-block above so a real BLOCK is never swallowed.
    # ------------------------------------------------------------------
    try:
        from src.assembled_core.ops._paper_runner_gates import (
            apply_reconcile_block_gate,
        )

        reconcile_decision = apply_reconcile_block_gate(
            ctx, paper_cfg=paper_cfg, root=root
        )
    except Exception as _rec_gate_exc:  # noqa: BLE001 — import/setup only
        # An armed gate that cannot even run must not silently fail open.
        if (paper_cfg.get("reconcile_block") or {}).get("enabled", False):
            log.warning(
                "[RECONCILE-GATE] gate setup failed while ARMED — fail-closed, "
                "skipping cycle: %s",
                _rec_gate_exc,
            )
            return 0, "reconcile_blocked"
        log.warning(
            "[RECONCILE-GATE] gate setup failed (disabled, continuing): %s",
            _rec_gate_exc,
        )
        reconcile_decision = None

    if reconcile_decision is not None and reconcile_decision.blocked:
        log.warning(
            "[RECONCILE-GATE] next cycle blocked: %s", reconcile_decision.reason
        )
        return 0, "reconcile_blocked"

    result = run_trading_cycle(ctx)
    if result.status != "success":
        log.error("Trading cycle failed: %s", result.error_message)
        return 1, None

    # §9.6 (b) Phase 1 shadow-mode: optional parallel strategy run for signal
    # comparison, no order/broker impact. Default off.
    _prd_run_shadow_strategy(
        prices=prices,
        paper_cfg=paper_cfg,
        output_dir=output_dir,
        as_of_ts=as_of_ts,
        primary_signals=result.signals,
    )

    try:
        if (
            result.signals is not None
            and not result.signals.empty
            and "score" in result.signals.columns
        ):
            from src.assembled_core.paper.intel_context import persist_historical_scores

            persist_historical_scores(result.signals["score"], root)
    except Exception as _hs_exc:
        log.debug("[INTEL-CTX] persist_historical_scores failed: %s", _hs_exc)

    result.meta["intel_orchestration"] = intel_orchestration

    _prd_intel_summaries(result, paper_cfg, root)

    reconcile_status: str | None = None
    if mode == "paper" and ledger_state is not None and ledger_path is not None:
        reconcile_status = _prd_paper_fills_and_ledger(
            mode=mode,
            ledger_state=ledger_state,
            ledger_path=ledger_path,
            result=result,
            execution_mode=execution_mode,
            broker_adapter=broker_adapter,
            app_cfg=app_cfg,
            as_of_ts=as_of_ts,
            output_dir=output_dir,
            start_capital=start_capital,
            strategy_name=strategy_name,
            pilot_policy=_pilot_policy,
        )

    _prd_write_artifacts(
        mode=mode,
        ctx=ctx,
        result=result,
        output_dir=output_dir,
        as_of_ts=as_of_ts,
        app_cfg=app_cfg,
        reconcile_status=reconcile_status,
        execution_mode=execution_mode,
    )

    return 0, reconcile_status
