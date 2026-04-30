"""trading_cycle_v2 — decomposed trading cycle (Week 4–6 refactor).

The old trading_cycle.py remains the active implementation until Day 9.
This file holds the 7-function target structure.

A step survives only when ALL three hold:
  1. It changes a value that a downstream step or caller reads.
  2. It has a test asserting concrete output values (not just existence).
  3. It does not have the shape  result.meta["x"] = {"available": True}.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from src.assembled_core.config import get_base_dir
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
    _apply_group_exposure_caps,
    _apply_pre_trade_impact,
    _apply_risk_controls_default,
    _build_features_default,
    _estimate_symbol_volatilities,
    _evaluate_auto_dd_kill_switch,
    _evaluate_circuit_breaker,
    _evaluate_circuit_breaker_daily,
    _evaluate_var_gate,
    _filter_prices_for_as_of,
    _generate_orders_default,
    should_rebalance,
)
from src.assembled_core.risk.market_stress import compute_market_stress
from src.assembled_core.risk.state_machine import (
    compute_next_state,
    load_risk_state,
    save_risk_state,
)

# Submodule imports — keep all public names importable from this module
from src.assembled_core.pipeline._tc_features import build_features
from src.assembled_core.pipeline._tc_signals import (
    _apply_evidence_gate,
    _compute_news_triggers,
    generate_signals,
)
from src.assembled_core.pipeline._tc_sizing import (
    _sp_apply_correlation_guard,
    _sp_apply_cost_aware,
    _sp_apply_crash_cap,
    _sp_apply_crisis_alpha_cap,
    _sp_apply_crowding_cap,
    _sp_apply_factor_risk,
    _sp_apply_inverse_etf,
    _sp_apply_liquidity,
    _sp_apply_quantile_asymmetry,
    _sp_apply_trailing_stops,
    _sp_apply_turnover_gate,
    _sp_check_rebalance,
    _sp_compute_final_multiplier,
    _sp_dispatch_sizing,
    size_positions,
)
from src.assembled_core.pipeline._tc_risk import check_risk
from src.assembled_core.pipeline._tc_execution import book_fills, route_orders

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ingest_data — Stage 1
# ---------------------------------------------------------------------------


def ingest_data(
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Validate, prepare context, and filter prices PIT-safely.

    Real steps included (see 3-criteria rule in module docstring):
      - input validation (raises ValueError on bad ctx)
      - risk state machine: sets ctx.risk_state (read by check_risk)
      - intel loading: disclosures triggers (ctx.disclosures_triggers),
        crisis state (ctx.crisis_state_intel, ctx.news_geo),
        market stress (ctx.market_stress)
      - circuit breaker: activates kill switch when daily CB trips
      - disclosures confirm: adjusts ctx.news_geo.geo_confidence
      - price filtering: PIT-safe, returns (prices_filtered, prices_latest)

    Observability-only steps dropped vs the old monolith:
      - result.meta["data_lineage"] (Step 1.8)
      - result.meta["price_quality_check"] (Step 1.9)
      - result.meta["market_breadth"] (Phase 5.2)
      - result.meta["intel_geo_triggers"] (intel crisis sub-block)
      - Steps 1.95, 1.97 (comprehensive QC, macro diffusion)

    Returns:
        (prices_filtered, prices_latest)
    Raises:
        ValueError: on missing/invalid ctx fields.
    """
    if log is None:
        log = logger

    # --- Validation ---
    if ctx.prices is None or ctx.prices.empty:
        raise ValueError("prices DataFrame is None or empty")

    required_cols = ["timestamp", "symbol", "close"]
    missing = [c for c in required_cols if c not in ctx.prices.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")

    if ctx.signal_fn is None:
        raise ValueError("signal_fn is required but not provided")

    if ctx.position_sizing_fn is None:
        raise ValueError("position_sizing_fn is required but not provided")

    # --- Risk state machine setup ---
    try:
        policy = load_policy()
    except Exception as e:
        log.warning("load_policy failed, using empty policy: %s", e)
        policy = {}

    rsm = policy.get("risk_state_machine") or {}
    base_dir = get_base_dir()
    persistence = rsm.get("persistence") or {}
    mode = os.environ.get("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE") or persistence.get(
        "mode", "live"
    )

    if getattr(ctx, "as_of", None) is not None:
        now_utc = pd.to_datetime(ctx.as_of, utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        now_utc = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    if mode == "ephemeral":
        import tempfile

        _ephemeral_path = (
            Path(tempfile.gettempdir())
            / f"assembled_risk_state_ephemeral_{os.getpid()}.json"
        )
        prev = load_risk_state(_ephemeral_path)
        next_rec = compute_next_state(ctx, policy, now_utc, prev)
        ctx.risk_state = next_rec.to_dict()
    else:
        if mode == "per_run":
            run_id = (
                getattr(ctx, "run_id", None)
                or os.environ.get("ASSEMBLED_RUN_ID")
                or f"pid{os.getpid()}"
            )
            per_run_dir = base_dir / str(
                persistence.get("per_run_dir", "output/state/runs")
            )
            state_path = per_run_dir / str(run_id) / "risk_state.json"
        else:
            state_path = base_dir / str(
                rsm.get("state_path", "output/state/risk_state.json")
            )
        prev = load_risk_state(state_path)
        next_rec = compute_next_state(ctx, policy, now_utc, prev)
        if rsm.get("enabled", True):
            save_risk_state(next_rec, state_path, policy)
        ctx.risk_state = next_rec.to_dict()

    # --- Intel loading (skip when paper_runner injected simulated intel) ---
    if not getattr(ctx, "intel_sim_applied", False):
        _load_intel(ctx, policy, base_dir, log)

    # --- Price filtering (Step 1) ---
    prices_filtered, prices_latest = _filter_prices_for_as_of(
        prices=ctx.prices,
        as_of=ctx.as_of,
        universe=ctx.universe,
        mode=ctx.mode,
    )

    if prices_filtered.empty:
        raise ValueError("No prices remaining after filtering (as_of or universe)")

    log.debug(
        "Prices filtered: %d rows, %d symbols (mode=%s, latest=%s)",
        len(prices_filtered),
        prices_filtered["symbol"].nunique(),
        ctx.mode,
        "yes" if prices_latest is not None else "no",
    )

    return prices_filtered, prices_latest


def _load_intel(
    ctx: TradingContext,
    policy: dict[str, Any],
    base_dir: Path,
    log: logging.Logger,
) -> None:
    """Load intel into ctx (disclosures triggers, crisis state, market stress, CB)."""
    import json as _json

    intel_cfg = policy.get("intel") or {}

    # Disclosures triggers
    try:
        disc_tr_cfg = intel_cfg.get("disclosures_triggers") or {}
        if disc_tr_cfg.get("enabled", False):
            from src.assembled_core.intel.disclosures_triggers_loader import (
                load_disclosures_triggers,
            )

            path_raw = disc_tr_cfg.get(
                "path", "output/intel/disclosures/triggers_latest.json"
            )
            path_resolved = (
                (base_dir / path_raw) if not Path(path_raw).is_absolute() else Path(path_raw)
            )
            snap = load_disclosures_triggers(path_resolved)
            ctx.disclosures_triggers = snap if snap.generated_utc else None
            if not snap.generated_utc:
                ctx.intel_health_flags["intel_disclosures_triggers"] = "DEGRADED"
    except Exception as e:
        log.warning("intel disclosures_triggers load failed: %s", e)
        ctx.disclosures_triggers = None
        ctx.intel_health_flags = ctx.intel_health_flags or {}
        ctx.intel_health_flags.setdefault("intel_disclosures_triggers", "DEGRADED")

    # Crisis Alpha state
    try:
        crisis_cfg = intel_cfg.get("crisis_alpha") or {}
        if crisis_cfg.get("enabled", False):
            cs_path_raw = crisis_cfg.get("crisis_state_path", "data/intel/crisis_state.json")
            cs_path = (
                (base_dir / cs_path_raw)
                if not Path(cs_path_raw).is_absolute()
                else Path(cs_path_raw)
            )
            if cs_path.exists():
                cs_data = _json.loads(cs_path.read_text(encoding="utf-8"))
                ctx.crisis_state_intel = cs_data
                geo_score = int(cs_data.get("geo_score", 0))
                mode_str = str(cs_data.get("mode", "NORMAL"))
                ctx.news_geo = {
                    "geo_score": geo_score,
                    "geo_confidence": float(cs_data.get("confidence", 0.0)),
                    "state_hint": mode_str,
                    "crisis_mode": mode_str,
                    "active_triggers": cs_data.get("active_triggers", []),
                    "basket_overrides": cs_data.get("basket_overrides", {}),
                }
                log.info(
                    "CRISIS_ALPHA: mode=%s, geo_score=%d, triggers=%d",
                    mode_str,
                    geo_score,
                    len(cs_data.get("active_triggers", [])),
                )
    except Exception as e:
        log.warning("crisis_alpha intel load failed: %s", e)
        ctx.intel_health_flags["intel_crisis_alpha"] = "DEGRADED"

    # Market stress (INT-5)
    ms_cfg = policy.get("market_stress") or {}
    if ms_cfg.get("enabled", False):
        ctx.market_stress = compute_market_stress(ctx.prices, policy)
    else:
        ctx.market_stress = None

    # Daily circuit breaker
    try:
        cb_trip = _evaluate_circuit_breaker_daily(ctx.prices, policy, ctx.as_of)
        if cb_trip is not None:
            from src.assembled_core.execution.kill_switch import activate_kill_switch

            activate_kill_switch(
                throttle_pct=0.0,
                reason=cb_trip["reason"],
                actor="trading_cycle_circuit_breaker",
            )
            log.critical(
                "CIRCUIT_BREAKER: %s — kill-switch engaged (block all)",
                cb_trip["reason"],
            )
    except Exception as e:
        log.warning(
            "[RISK-SAFETY] circuit_breaker_daily check failed: %s — breaker may not engage", e
        )

    # Disclosures confirm (boosts geo_confidence when disclosure triggers sev >= 1)
    try:
        from src.assembled_core.risk.disclosures_confirm import (
            apply_disclosures_confirm,
        )

        apply_disclosures_confirm(ctx, policy)
    except Exception as e:
        log.warning("disclosures_confirm apply failed: %s", e)



# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_trading_cycle(
    ctx: TradingContext,
    *,
    hooks: dict[str, Any] | None = None,
) -> TradingCycleResult:
    """Run the full trading cycle via the seven stage functions.

    This replaces _run_trading_cycle_inner once all stubs are filled.
    The old trading_cycle.run_trading_cycle() remains active until Day 9.
    """
    log = ctx.logger if ctx.logger is not None else logger

    # E0.1 parity: backtest kill-switch backup/restore
    _ks_state_backup: bool | None = None
    _is_backtest = getattr(ctx, "mode", None) in ("backtest", "bt")
    _ks_persist = bool(getattr(ctx, "kill_switch_persist", True))
    _ks_restore_active = _is_backtest and not _ks_persist
    if _ks_restore_active:
        try:
            from src.assembled_core.execution.kill_switch import is_kill_switch_engaged

            _ks_state_backup = is_kill_switch_engaged()
        except Exception as _e:
            log.warning("[KS-BACKUP] kill-switch state snapshot failed: %s", _e)

    result = TradingCycleResult(
        run_id=ctx.run_id,
        timestamp=pd.Timestamp.now("UTC"),
        status="success",
    )
    hooks = hooks or {}

    # Side-channel event bus — bus stays null (no-op) if REDIS_URL not set in env;
    # publish() calls are fire-and-forget and never block the trading cycle.
    _bus = None
    try:
        from src.assembled_core.pipeline.event_bus import get_null_bus
        _bus = get_null_bus()
        import os as _os

        from src.assembled_core.pipeline.event_bus import EventBus as _EventBus
        _redis_url = _os.environ.get("REDIS_URL", "")
        if _redis_url:
            try:
                _bus = _EventBus(redis_url=_redis_url, connect_timeout=0.5)
            except Exception:
                pass
    except Exception:
        pass

    def _pub(phase: str, **kw: object) -> None:
        if _bus is not None:
            try:
                _bus.publish(phase, {"run_id": ctx.run_id, **kw})
            except Exception:
                pass

    _pub("cycle_start", mode=getattr(ctx, "mode", "unknown"))

    try:
        _pub("ingest_start")
        prices, prices_latest = ingest_data(ctx, log=log)
        result.prices_filtered = prices
        result.prices_latest = prices_latest
        _pub("ingest_end", n_rows=len(prices) if prices is not None else 0)

        _pub("features_start")
        features, pl_update = build_features(prices, ctx, log=log)
        result.prices_with_features = features
        # Backtest snapshot mode can override prices_filtered/prices_latest
        if pl_update is not None:
            result.prices_latest = pl_update
            result.prices_filtered = pl_update
        _pub("features_end")

        _pub("signals_start")
        signals = generate_signals(features, ctx, log=log)
        result.signals = signals
        _pub("signals_end", n_signals=len(signals) if signals is not None else 0)

        _pub("sizing_start")
        targets, do_rebal, sizing_meta = size_positions(
            signals, ctx,
            prices_filtered=result.prices_filtered,
            prices_with_features=result.prices_with_features,
            prices_latest=result.prices_latest,
            log=log,
        )
        result.target_positions = targets
        result.meta.update(sizing_meta)
        _pub("sizing_end", n_targets=len(targets) if targets is not None else 0)

        _pub("routing_start")
        orders = route_orders(
            targets, ctx,
            prices_filtered=result.prices_filtered,
            prices_with_features=result.prices_with_features,
            prices_latest=result.prices_latest,
            do_rebal=do_rebal,
            log=log,
        )
        result.orders = orders
        _pub("routing_end", n_orders=len(orders) if orders is not None else 0)

        result = check_risk(orders, result, ctx, prices_filtered=result.prices_filtered, log=log)
        _pub("risk_checked", status=result.status)

        result = book_fills(result, ctx, log=log)
        _pub("fills_booked")

    except ValueError as exc:
        result.status = "error"
        result.error_message = str(exc)
        _pub("cycle_error", error=str(exc))
    except Exception as exc:
        result.status = "error"
        result.error_message = f"Unexpected error: {exc}"
        log.exception("trading_cycle_v2: unexpected error in run_trading_cycle")
        _pub("cycle_error", error=str(exc))
    else:
        _pub("cycle_end", status=result.status)
    finally:
        if _ks_restore_active and _ks_state_backup is not None and not _ks_state_backup:
            try:
                from src.assembled_core.execution.kill_switch import (
                    deactivate_kill_switch,
                    is_kill_switch_engaged,
                )

                if is_kill_switch_engaged():
                    deactivate_kill_switch(
                        reason="backtest_bar_restore",
                        actor="trading_cycle_v2_backtest_guard",
                    )
            except Exception as _e:
                log.warning("[KS-RESTORE] kill-switch state restore failed: %s", _e)

    return result
