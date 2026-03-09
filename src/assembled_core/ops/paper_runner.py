"""OPS-6: Paper daily runner — callable helper for one-day and range runs."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def run_paper_daily_one(
    as_of_ts: pd.Timestamp,
    output_dir: Path,
    mode: str,
    app_cfg: dict[str, Any],
    prices: pd.DataFrame,
    *,
    root: Path,
    day_index: int | None = None,
) -> tuple[int, str | None]:
    """Run a single paper/shadow day. Returns (exit_code, reconcile_status or None)."""
    from src.assembled_core.config.policy_loader import load_policy
    from src.assembled_core.ops.kpi_artifacts import (
        maybe_execute_orders,
        write_diff_vs_prev,
        write_orders_artifact,
        write_reasons_artifact,
        write_run_kpis,
        write_targets_artifact,
    )
    from src.assembled_core.ops.alerts import compute_alerts, write_alerts_artifact
    from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle

    reconcile_status: str | None = None
    start_capital = 10000.0
    ledger_state: dict[str, Any] | None = None
    ledger_path: Path | None = None
    equity_before = start_capital
    current_positions_df = pd.DataFrame(columns=["symbol", "qty", "target_qty"])
    equity_series: pd.Series | None = None
    equity_curve_index: int | None = None

    if mode == "paper":
        from src.assembled_core.ops.paper_ledger import load_ledger_state, mark_to_market_equity
        paper_cfg = app_cfg.get("paper_runner") or {}
        ledger_path_str = paper_cfg.get("ledger_path") or "output/runs/_paper_ledger/ledger_state.json"
        ledger_path = root / ledger_path_str if not Path(ledger_path_str).is_absolute() else Path(ledger_path_str)
        ledger_state = load_ledger_state(ledger_path, start_capital=start_capital)
        if not prices.empty and "timestamp" in prices.columns and "symbol" in prices.columns:
            p_ts = pd.to_datetime(prices["timestamp"], utc=True)
            prices_cut = prices.loc[p_ts <= as_of_ts]
            if not prices_cut.empty:
                prices_latest_mtm = prices_cut.groupby("symbol", group_keys=False).last().reset_index()
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
        current_positions_df = pd.DataFrame(pos_list) if pos_list else pd.DataFrame(columns=["symbol", "qty", "target_qty"])
        curve = ledger_state.get("equity_curve") or []
        if curve:
            equity_series = pd.Series([float(c.get("equity", 0)) for c in curve], dtype=float)
            equity_curve_index = len(equity_series) - 1
        else:
            equity_series = None
            equity_curve_index = None

    def _no_signal_fn(prices_with_features: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])

    def _no_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    signal_fn = _no_signal_fn
    position_sizing_fn = _no_sizing_fn
    paper_cfg = app_cfg.get("paper_runner") or {}
    strategy_cfg = paper_cfg.get("strategy") or {}

    # OPS-11: Intel mode real | sim | none
    intel_cfg = paper_cfg.get("intel") or {}
    intel_mode = (intel_cfg.get("mode") or "sim").strip().lower()
    intel_orchestration: dict[str, Any] = {}
    if intel_mode == "real":
        from src.assembled_core.ops.intel_orchestrator import run_intel_pipelines
        intel_orchestration = run_intel_pipelines(app_cfg, root=root)

    strategy_name = (strategy_cfg.get("name") or "none").strip().lower()
    if strategy_name == "ema_trend_v0":
        from src.assembled_core.strategies.ema_trend_v0 import (
            compute_signals as ema_compute_signals,
            compute_target_positions as ema_compute_targets,
        )
        ema_fast = int(strategy_cfg.get("ema_fast") or 20)
        ema_slow = int(strategy_cfg.get("ema_slow") or 60)
        equal_weight = bool(strategy_cfg.get("equal_weight", True))

        def _ema_signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            return ema_compute_signals(df, ema_fast=ema_fast, ema_slow=ema_slow)

        signal_fn = _ema_signal_fn
        def _ema_sizing(sig: pd.DataFrame, cap: float) -> pd.DataFrame:
            prices_latest = None
            if ctx.prices is not None and not ctx.prices.empty and ctx.as_of is not None:
                p_ts = pd.to_datetime(ctx.prices["timestamp"], utc=True)
                p_cut = ctx.prices.loc[p_ts <= ctx.as_of]
                if not p_cut.empty:
                    prices_latest = p_cut.groupby("symbol", group_keys=False)["close"].last().reset_index()
            return ema_compute_targets(sig, cap, equal_weight=equal_weight, prices_latest=prices_latest)
        position_sizing_fn = _ema_sizing

    ctx = TradingContext(
        prices=prices,
        as_of=as_of_ts,
        freq="1d",
        mode="eod",
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=start_capital,
        write_outputs=False,
        enable_risk_controls=False,
    )
    if mode == "paper" and ledger_state is not None:
        ctx.capital = equity_before
        ctx.current_positions = current_positions_df if not current_positions_df.empty else None
        ctx.equity_curve = equity_series
        ctx.equity_curve_index = equity_curve_index

    # BENCH-1/BENCH-2: Intel sim harness only when mode is sim
    intel_sim_cfg = paper_cfg.get("intel_sim") or {}
    if intel_mode == "sim" and intel_sim_cfg.get("enabled", False) and day_index is not None:
        from src.assembled_core.ops.intel_sim import apply_intel_sim
        apply_intel_sim(ctx, day_index, intel_sim_cfg)

    result = run_trading_cycle(ctx)
    if result.status != "success":
        log.error("Trading cycle failed: %s", result.error_message)
        return 1, None

    result.meta["intel_orchestration"] = intel_orchestration

    # OPS-14: Per-run intel trigger summaries for run_kpis (enables experiment-level activity without global artifacts)
    paper_cfg = app_cfg.get("paper_runner") or {}
    intel_cfg = paper_cfg.get("intel") or {}
    news_out = (intel_cfg.get("news") or {}).get("output_dir") or "output/intel/news"
    discl_out = (intel_cfg.get("disclosures") or {}).get("output_dir") or "output/intel/disclosures"
    news_path = root / news_out if not Path(news_out).is_absolute() else Path(news_out)
    news_path = news_path / "triggers_latest.json"
    discl_path = root / discl_out if not Path(discl_out).is_absolute() else Path(discl_out)
    discl_path = discl_path / "triggers_latest.json"
    try:
        from src.assembled_core.intel import load_news_triggers, load_disclosures_triggers
        news_snap = load_news_triggers(news_path)
        result.meta["news_triggers_summary"] = {
            "count": len(news_snap.triggers),
            "max_severity": news_snap.summary.get("max_severity", 0),
            "count_sev1plus": news_snap.summary.get("watch_count_sev1plus", 0),
            "count_sev2plus": news_snap.summary.get("active_count_sev2plus", 0),
        }
    except Exception:
        result.meta["news_triggers_summary"] = {"count": 0, "max_severity": 0, "count_sev1plus": 0, "count_sev2plus": 0}
    # NEWS-DEBUG-1/2: load debug funnel summary for run_kpis (counts + compact reason previews, no samples)
    try:
        funnel_path = news_path.parent / "debug_funnel_latest.json"
        if funnel_path.exists():
            funnel_data = json.loads(funnel_path.read_text(encoding="utf-8"))
            if isinstance(funnel_data, dict) and "counts" in funnel_data:
                compact = dict(funnel_data["counts"])
                exc_reasons = funnel_data.get("normalize_exception_reasons") or {}
                none_reasons = funnel_data.get("normalize_none_reasons") or {}
                compact["normalize_exception_reasons_preview"] = dict(list(exc_reasons.items())[:2])
                compact["normalize_none_reasons_preview"] = dict(list(none_reasons.items())[:2])
                result.meta["news_debug_funnel"] = compact
    except Exception:
        pass
    try:
        discl_snap = load_disclosures_triggers(discl_path)
        result.meta["disclosures_triggers_summary"] = {
            "count": len(discl_snap.triggers),
            "max_severity": discl_snap.summary.get("max_severity", 0),
            "count_sev1plus": discl_snap.summary.get("count_sev1plus", 0),
            "count_sev2plus": discl_snap.summary.get("count_sev2plus", 0),
        }
    except Exception:
        result.meta["disclosures_triggers_summary"] = {"count": 0, "max_severity": 0, "count_sev1plus": 0, "count_sev2plus": 0}

    if mode == "paper" and ledger_state is not None and ledger_path is not None:
        import copy
        from src.assembled_core.ops.paper_ledger import (
            apply_fills_to_ledger,
            mark_to_market_equity,
            save_ledger_state,
            simulate_fills,
            write_ledger_snapshot,
        )
        from src.assembled_core.ops.reconcile import build_reconcile_report, write_reconcile_artifact
        orders_for_fills = result.orders_filtered if not result.orders_filtered.empty else result.orders
        prices_for_fills = result.prices_filtered
        if prices_for_fills.empty:
            prices_for_fills = result.prices_with_features
        cost_cfg = (app_cfg.get("paper_runner") or {}).get("cost_model") or {}
        ledger_before = copy.deepcopy(ledger_state)
        fills = simulate_fills(orders_for_fills, prices_for_fills, cost_cfg)
        state_after = apply_fills_to_ledger(ledger_state, fills)
        equity_after = mark_to_market_equity(state_after, prices_for_fills)
        now_iso = pd.Timestamp.utcnow().isoformat()
        state_after["equity_curve"] = list(state_after.get("equity_curve") or []) + [{"utc": now_iso, "equity": equity_after}]
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
        reconcile_status = report.get("status") or "OK"
        save_ledger_state(state_after, ledger_path)
        write_ledger_snapshot(output_dir, state_after, equity_after)
        result.meta["paper_ledger"] = {"equity": equity_after, "equity_curve_point": {"utc": now_iso, "equity": equity_after}}

    _ = maybe_execute_orders(mode, result.orders_filtered if not result.orders_filtered.empty else result.orders)

    policy = load_policy()
    kpis_path = write_run_kpis(output_dir=output_dir, ctx=ctx, result=result, policy=policy, mode=mode)
    write_targets_artifact(output_dir=output_dir, target_positions=result.target_positions)
    orders_df = result.orders_filtered if not result.orders_filtered.empty else result.orders
    write_orders_artifact(output_dir=output_dir, orders=orders_df)
    write_reasons_artifact(output_dir=output_dir, ctx=ctx, result=result, policy=policy, mode=mode)

    try:
        current_kpis = json.loads(kpis_path.read_text(encoding="utf-8"))
    except Exception:
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
        run_kpis_data = current_kpis
        reasons_data = {}
        diff_data = {}
        if (out / "reasons_latest.json").exists():
            try:
                reasons_data = json.loads((out / "reasons_latest.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        if (out / "diff_vs_prev.json").exists():
            try:
                diff_data = json.loads((out / "diff_vs_prev.json").read_text(encoding="utf-8"))
            except Exception:
                pass
        generated_utc = run_kpis_data.get("generated_utc") or ""
        alerts_list = compute_alerts(run_kpis_data, reasons_data, diff_data, app_cfg)
        if mode == "paper" and reconcile_status == "FAIL":
            from src.assembled_core.ops.alerts import make_reconcile_fail_alert
            alerts_list = list(alerts_list)
            alerts_list.append(make_reconcile_fail_alert(generated_utc))
            severity_map = app_cfg.get("alerts", {}).get("severity_map") or {"info": 0, "warn": 1, "critical": 2}
            alerts_list.sort(key=lambda a: (-severity_map.get(a["level"], 0), a["kind"], a["alert_id"]))
        write_alerts_artifact(output_dir, alerts_list, generated_utc, app_cfg)

    return 0, reconcile_status
