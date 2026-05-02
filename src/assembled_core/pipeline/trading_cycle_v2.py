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
from src.assembled_core.execution.transaction_costs import add_cost_columns_to_trades  # A8 wiring — re-export for tests
from src.assembled_core.risk.georisk_overlay import compute_exposure_multiplier  # re-export for tests

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

    # Options IV skew Z-score — populate ctx.options_iv_skew_z for Phase H triple-confirmation.
    # Uses vix_zscore_252d (Z-score of VIX vs 252d window) as proxy for tail-risk pricing.
    # Tries panel first (pre-computed), then CBOESource direct fetch.
    try:
        _vix_z: float = 0.0
        _prices = getattr(ctx, "prices_filtered", None) or getattr(ctx, "prices", None)
        if _prices is not None and "vix_zscore_252d" in _prices.columns:
            _vix_z_series = _prices["vix_zscore_252d"].dropna()
            if not _vix_z_series.empty:
                _vix_z = float(_vix_z_series.iloc[-1])
        if _vix_z == 0.0:
            from src.assembled_core.data.sources.cboe_source import CBOESource
            from src.assembled_core.features.options_derived_signals import (
                build_options_regime_factors,
            )
            _cboe_df = CBOESource().fetch_options_regime_data()
            if not _cboe_df.empty:
                _opts = build_options_regime_factors(_cboe_df)
                if not _opts.empty:
                    _vix_z = float(_opts.iloc[-1].get("vix_zscore_252d", 0.0) or 0.0)
        ctx.options_iv_skew_z = _vix_z
        if abs(_vix_z) > 1.0:
            log.info("[OPTIONS-IV] vix_z=%.2f → options_iv_skew_z populated", _vix_z)
    except Exception as _e:
        log.debug("options_iv_skew_z population skipped: %s", _e)

    # EDCL — Event-Driven Conviction Layer basket computation
    # Runs even when edcl_conviction_overlay.enabled=false so that ctx.edcl_state
    # is always populated for observability. Multiplier only fires when enabled.
    try:
        _edcl_cfg = (policy.get("edcl_conviction_overlay") or {})
        # Skip entirely in backtest mode unless allow_in_backtest is set
        _edcl_mode = getattr(ctx, "mode", "backtest")
        _allow_bt = _edcl_cfg.get("allow_in_backtest", False)
        if _edcl_mode not in ("backtest", "bt") or _allow_bt:
            from src.assembled_core.intel.trigger_basket import (
                TriggerBasket,
                build_trigger_basket,
            )
            from src.assembled_core.intel.conviction_engine import compute_conviction_score
            from src.assembled_core.intel.models import TriggerType

            _basket: TriggerBasket | None = None

            # Path 1: full keyword scoring from raw NewsEvent objects
            _raw = getattr(ctx, "raw_news_events", None)
            if _raw:
                _basket = build_trigger_basket(_raw)
                _source = "raw_news_events"

            # Path 2: construct TriggerBasket directly from active_triggers in ctx.news_geo
            if _basket is None:
                _geo = ctx.news_geo or {}
                _geo_conf = float(_geo.get("geo_confidence", 0.0))
                _geo_tags: set[str] = set(_geo.get("geo_tags", []))
                _active = _geo.get("active_triggers", [])
                _fired: list[tuple[TriggerType, float]] = []
                for _name in _active:
                    try:
                        _fired.append((TriggerType(_name), _geo_conf))
                    except ValueError:
                        pass
                # Derive sector/asset maps from fired triggers
                if _fired:
                    from src.assembled_core.intel.trigger_basket import (
                        _TRIGGER_SECTOR_MAP,
                    )
                    from src.assembled_core.intel.news_classifier import (
                        COUNTRY_TO_ASSETS,
                        SECTOR_TO_ETFS,
                    )
                    _sector_scores: dict[str, float] = {}
                    for _tt, _sc in _fired:
                        for _sec in _TRIGGER_SECTOR_MAP.get(_tt, []):
                            _sector_scores[_sec] = max(_sector_scores.get(_sec, 0.0), _sc)
                    _seen: set[str] = set()
                    _assets: list[str] = []
                    for _sec in _sector_scores:
                        for _a in SECTOR_TO_ETFS.get(_sec, []):
                            if _a not in _seen:
                                _assets.append(_a)
                                _seen.add(_a)
                    for _iso in _geo_tags:
                        for _a in COUNTRY_TO_ASSETS.get(_iso.upper(), []):
                            if _a not in _seen:
                                _assets.append(_a)
                                _seen.add(_a)
                    _n_high = sum(1 for _, _s in _fired if _s >= 0.6)
                    _basket = TriggerBasket(
                        fired_triggers=_fired,
                        affected_sectors=_sector_scores,
                        affected_assets=_assets,
                        geo_tags=_geo_tags,
                        conviction=_geo_conf,
                        n_events=max(len(_fired), 1),
                        n_high_conviction=_n_high,
                    )
                    _source = f"active_triggers({len(_fired)})"
                else:
                    _basket = TriggerBasket()
                    _source = "no_triggers"

            _conviction = compute_conviction_score(
                _basket,
                as_of=getattr(ctx, "as_of", None),
                policy=policy,
            )
            ctx.edcl_state = {
                "conviction": _conviction,
                "source": _source,
                "basket": _basket.as_dict(),
            }
            if _conviction > 0.0:
                log.info(
                    "[EDCL] conviction=%.3f source=%s triggers=%d sectors=%s",
                    _conviction,
                    _source,
                    len(_basket.fired_triggers),
                    list(_basket.affected_sectors.keys()),
                )
                # Phase C: log fired events to geo_events_historical.parquet
                # Builds training data for compute_event_betas.py over time.
                try:
                    from src.assembled_core.intel.geo_event_logger import log_basket_event
                    _tier = 1 if _source == "raw_news_events" else 2
                    log_basket_event(
                        _basket, _conviction,
                        as_of=getattr(ctx, "as_of", None),
                        source_tier=_tier,
                    )
                except Exception as _log_e:
                    log.debug("geo_event_logger skipped: %s", _log_e)
    except Exception as _e:
        log.debug("edcl_basket computation skipped: %s", _e)

    # EDCL Phase G — Tail-Hunting: match active basket against pre-positioned plans
    try:
        _edcl_state = getattr(ctx, "edcl_state", None) or {}
        _conviction_g = float(_edcl_state.get("conviction", 0.0))
        if _conviction_g > 0.0 and _basket is not None:
            from src.assembled_core.intel.tail_hunting import match_tail_plans
            _tail_signals = match_tail_plans(_basket, _conviction_g)
            if _tail_signals:
                _edcl_state["tail_signals"] = [s.as_dict() for s in _tail_signals]
                ctx.edcl_state = _edcl_state
                log.info(
                    "[TAIL-G] %d plan(s) activated: %s",
                    len(_tail_signals),
                    [s.event_name for s in _tail_signals],
                )
    except Exception as _e:
        log.debug("tail_hunting Phase G skipped: %s", _e)


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
