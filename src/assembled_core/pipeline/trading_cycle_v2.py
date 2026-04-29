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
    _evaluate_auto_dd_kill_switch,
    _evaluate_circuit_breaker,
    _evaluate_circuit_breaker_daily,
    _evaluate_var_gate,
    _estimate_symbol_volatilities,
    _filter_prices_for_as_of,
    _generate_orders_default,
    should_rebalance,
)
from src.assembled_core.risk.correlation_guard import (
    apply_correlation_guard,
    detect_correlation_regime_shift,
)
from src.assembled_core.risk.georisk_overlay import (
    apply_exposure_multiplier_to_targets,
    compute_exposure_multiplier,
)
from src.assembled_core.risk.market_stress import compute_market_stress
from src.assembled_core.risk.profit_lock import compute_profit_lock_multiplier
from src.assembled_core.risk.state_machine import (
    compute_next_state,
    load_risk_state,
    save_risk_state,
)
from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
)
from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

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
        from src.assembled_core.risk.disclosures_confirm import apply_disclosures_confirm

        apply_disclosures_confirm(ctx, policy)
    except Exception as e:
        log.warning("disclosures_confirm apply failed: %s", e)


# ---------------------------------------------------------------------------
# build_features — Stage 2 (stub)
# ---------------------------------------------------------------------------


def build_features(
    prices: pd.DataFrame,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add TA features and real enrichment steps.

    Returns (prices_with_features, prices_latest_update).
    prices_latest_update is non-None only in backtest snapshot mode, when
    the precomputed panel overrides both prices_filtered and prices_latest.
    The orchestrator must apply these overrides to result.

    Real steps kept (vs old monolith):
      - Step 2: core features (build_or_load_factors / add_all_features)
      - Step 2.2: ta_factors_core + cross-sectional normalization (adds columns)
      - Step 2.5 HMM: D3 HMM regime → sets ctx.regime_state (used by size_positions)
      - Step 2.5 behavioral: behavioral_composite column (adds column)
      - Step 2.6: seasonal features (adds calendar columns)
      - Step 2.8: mean-reversion factors (adds mr_* columns)
      - Step 2.9: interaction features (adds computed columns)
      - Step 2.12: weekly alignment filter (adds column)
      - Step 2.10: realized volatility rv_20/rv_60 (adds columns)
      - Step 2.11: fractional differentiation ffd_close (adds column)

    Dropped (meta-only, no column additions):
      - Steps 2.1 (PIT check meta), 2.3 (freshness meta), 2.4 (drift meta),
        2.7 (correlation regime meta), 2.13-2.14 (clustering/IC meta),
        2.16-2.35 (all observability-labelled steps)
    """
    if log is None:
        log = logger

    try:
        policy = load_policy()
    except Exception:
        policy = {}

    prices_latest_update: pd.DataFrame | None = None

    # --- Step 2: Core features ---
    if (
        ctx.mode == "backtest"
        and ctx.precomputed_prices_with_features is not None
        and not ctx.precomputed_prices_with_features.empty
    ):
        precomputed = ctx.precomputed_prices_with_features.copy()
        if precomputed["timestamp"].dtype.tz is None:
            precomputed["timestamp"] = pd.to_datetime(precomputed["timestamp"], utc=True)

        if ctx.backtest_use_snapshot:
            if ctx.precomputed_panel_index is not None and ctx.as_of is not None:
                from src.assembled_core.pipeline.precomputed_index import snapshot_as_of

                snap = snapshot_as_of(
                    df=precomputed,
                    index=ctx.precomputed_panel_index,
                    as_of=ctx.as_of,
                    use_monotonic_optimization=True,
                )
            else:
                precomputed_filtered = (
                    precomputed[precomputed["timestamp"] <= ctx.as_of].copy()
                    if ctx.as_of is not None
                    else precomputed.copy()
                )
                snap = (
                    precomputed_filtered.groupby("symbol", group_keys=False, dropna=False)
                    .last()
                    .reset_index()
                    .sort_values("symbol")
                    .reset_index(drop=True)
                )
            pwf = snap.copy()
            prices_latest_update = snap.copy()
            log.debug(
                "Using precomputed features (snapshot mode): %d rows (index=%s)",
                len(pwf),
                "yes" if ctx.precomputed_panel_index is not None else "no",
            )
        else:
            pwf = (
                precomputed[precomputed["timestamp"] <= ctx.as_of].copy()
                if ctx.as_of is not None
                else precomputed.copy()
            )
            prices_latest_update = (
                pwf.groupby("symbol", group_keys=False, dropna=False)
                .last()
                .reset_index()
                .sort_values("symbol")
                .reset_index(drop=True)
            )
            log.debug(
                "Using precomputed features (history-slice mode): %d rows", len(pwf)
            )
    elif ctx.mode in ("eod", "paper", "live") and ctx.as_of is not None:
        prices_for_features = ctx.prices[ctx.prices["timestamp"] <= ctx.as_of].copy()
        if ctx.universe is not None:
            universe_upper = [s.upper().strip() for s in ctx.universe]
            prices_for_features = prices_for_features[
                prices_for_features["symbol"].str.upper().isin(universe_upper)
            ].copy()
        pwf = _build_features_default(ctx, prices_for_features)
        prices_latest_update = (
            pwf.groupby("symbol", group_keys=False, dropna=False)
            .last()
            .reset_index()
            .sort_values("symbol")
            .reset_index(drop=True)
            if not pwf.empty
            else None
        )
    else:
        pwf = _build_features_default(ctx, prices)

    log.debug(
        "Features: %d columns (was %d)", len(pwf.columns), len(prices.columns)
    )

    # --- Step 2.5 HMM: D3 regime detection → sets ctx.regime_state ---
    try:
        regime_cfg = policy.get("regime_detection", {})
        if regime_cfg.get("method") == "hmm" and getattr(ctx, "regime_state", None) is None:
            from src.assembled_core.risk.regime_models import build_regime_state_hmm

            prices_for_hmm = prices if prices is not None and not prices.empty else ctx.prices
            if prices_for_hmm is not None and not prices_for_hmm.empty:
                hmm_df = build_regime_state_hmm(
                    prices=prices_for_hmm,
                    n_regimes=int(regime_cfg.get("n_regimes", 3)),
                    benchmark_symbol=regime_cfg.get("benchmark_symbol"),
                )
                if not hmm_df.empty:
                    ctx.regime_state = hmm_df.iloc[-1].get("regime_label", "sideways")
                    log.info("REGIME_HMM: detected regime='%s'", ctx.regime_state)
    except Exception as e:
        log.debug("HMM regime detection skipped: %s", e)

    # --- Step 2.2: Enhanced enrichment (ta_factors_core + cross_sectional) ---
    try:
        enh_cfg = (policy.get("features") or {}).get("enhanced_factors") or {}
        if enh_cfg.get("enabled", False) and not pwf.empty:
            if enh_cfg.get("ta_factors_core", True):
                from src.assembled_core.features.ta_factors_core import build_core_ta_factors

                pwf = build_core_ta_factors(
                    pwf, price_col="close", group_col="symbol", timestamp_col="timestamp"
                )
            if enh_cfg.get("cross_sectional_rank", True):
                from src.assembled_core.features.cross_sectional import rank_cross_sectional

                rank_cols = [
                    c
                    for c in enh_cfg.get(
                        "rank_cols",
                        [
                            "trend_ema_spread", "mom_rsi_centered", "mom_12_1",
                            "low_vol_rank", "quality_score",
                            "trend_strength_20", "trend_strength_50",
                            "momentum_12m_excl_1m",
                        ],
                    )
                    if c in pwf.columns
                ]
                if rank_cols:
                    pwf = rank_cross_sectional(
                        pwf,
                        feature_cols=rank_cols,
                        timestamp_col="timestamp",
                        normalize_to=enh_cfg.get("rank_normalize_to", "symmetric"),
                    )
    except Exception as e:
        log.debug("[FEATURE-ENH] enhanced enrichment skipped: %s", e)

    # --- Step 2.5 behavioral: adds behavioral_composite column ---
    try:
        beh_cfg = (policy.get("features") or {}).get("behavioral_features") or {}
        if beh_cfg.get("enabled", False) and not pwf.empty:
            from src.assembled_core.features.behavioral_features import (
                compute_behavioral_composite,
            )

            _req_cols = {"symbol", "close"}
            if _req_cols.issubset(pwf.columns):
                _beh_scores: dict[str, float] = {}
                _beh_min_rows = int(beh_cfg.get("min_rows", 60))
                for _sym in pwf["symbol"].unique()[:50]:
                    _grp = pwf[pwf["symbol"] == _sym]
                    if "timestamp" in _grp.columns:
                        _grp = _grp.sort_values("timestamp")
                    if len(_grp) < _beh_min_rows:
                        continue
                    _bp = _grp["close"].reset_index(drop=True)
                    _bv = (
                        _grp["volume"].reset_index(drop=True)
                        if "volume" in _grp.columns
                        else pd.Series(1.0, index=range(len(_grp)))
                    )
                    _br = _bp.pct_change().fillna(0)
                    try:
                        _bc = compute_behavioral_composite(_bp, _bv, _br)
                        _beh_scores[str(_sym)] = float(_bc.iloc[-1]) if len(_bc) > 0 else 0.0
                    except Exception:
                        pass
                if _beh_scores:
                    pwf = pwf.copy()
                    pwf["behavioral_composite"] = pwf["symbol"].map(_beh_scores)
    except Exception as e:
        log.debug("[BEHAVIORAL] behavioral_features skipped: %s", e)

    # --- Step 2.6: Seasonal features (zero look-ahead calendar columns) ---
    try:
        seas_cfg = (policy.get("features") or {}).get("seasonal_features") or {}
        if seas_cfg.get("enabled", False) and not pwf.empty and "timestamp" in pwf.columns:
            from src.assembled_core.features.seasonal_features import build_seasonal_features

            _seas_ts = pd.DatetimeIndex(pwf["timestamp"])
            _seas_df = build_seasonal_features(_seas_ts)
            pwf = pwf.reset_index(drop=True)
            for col in _seas_df.columns:
                pwf[col] = _seas_df[col].values
    except Exception as e:
        log.debug("[SEASONAL] seasonal_features skipped: %s", e)

    # --- Step 2.8: Mean-reversion factor columns ---
    try:
        mr_cfg = (policy.get("features") or {}).get("mean_reversion_factors") or {}
        if mr_cfg.get("enabled", False) and not pwf.empty:
            _req = {"symbol", "timestamp", "close"}
            if _req.issubset(pwf.columns):
                from src.assembled_core.features.mean_reversion_factors import (
                    compute_mean_reversion_factors,
                )

                _mr_df = compute_mean_reversion_factors(pwf)
                if not _mr_df.empty:
                    _mr_cols = [c for c in _mr_df.columns if c.startswith("mr_")]
                    _keys = [k for k in ["symbol", "timestamp"] if k in _mr_df.columns]
                    pwf = pwf.merge(
                        _mr_df[_keys + _mr_cols], on=_keys, how="left", suffixes=("", "_mrf")
                    )
    except Exception as e:
        log.debug("[MR-FACTORS] mean_reversion_factors skipped: %s", e)

    # --- Step 2.9: Interaction feature columns ---
    try:
        ix_cfg = (policy.get("features") or {}).get("interaction_features") or {}
        if ix_cfg.get("enabled", False) and not pwf.empty:
            from src.assembled_core.features.interaction_features import (
                compute_interaction_features,
            )

            _before = set(pwf.columns)
            _ix_df = compute_interaction_features(pwf)
            if [c for c in _ix_df.columns if c not in _before]:
                pwf = _ix_df
    except Exception as e:
        log.debug("[IX-FEATURES] interaction_features skipped: %s", e)

    # --- Step 2.12: Weekly alignment filter column ---
    try:
        wa_cfg = (policy.get("features") or {}).get("weekly_alignment") or {}
        if wa_cfg.get("enabled", False) and not pwf.empty:
            _req = {"close", "symbol", "timestamp"}
            if _req.issubset(pwf.columns):
                from src.assembled_core.features.weekly_alignment import add_weekly_alignment

                _trend_col = next(
                    (
                        c
                        for c in (
                            "trend_strength_50", "momentum_12m_excl_1m", "trend_strength_200"
                        )
                        if c in pwf.columns
                    ),
                    None,
                )
                if _trend_col:
                    _wa_df = pwf.copy().set_index("timestamp")
                    _wa_df = add_weekly_alignment(_wa_df, daily_trend_col=_trend_col)
                    pwf = _wa_df.reset_index()
    except Exception as e:
        log.debug("[WEEKLY-ALIGN] weekly_alignment skipped: %s", e)

    # --- Step 2.10: Realized volatility rv_20 / rv_60 ---
    try:
        rv_cfg = (policy.get("features") or {}).get("realized_volatility") or {}
        if rv_cfg.get("enabled", False) and not pwf.empty:
            _req = {"close", "symbol", "timestamp"}
            if _req.issubset(pwf.columns):
                from src.assembled_core.features.ta_liquidity_vol_factors import (
                    add_realized_volatility,
                )

                pwf = add_realized_volatility(
                    pwf, windows=[int(w) for w in rv_cfg.get("windows", [20, 60])]
                )
    except Exception as e:
        log.debug("[RV] realized_volatility skipped: %s", e)

    # --- Step 2.11: Fractional differentiation ffd_close ---
    try:
        ffd_cfg = (policy.get("features") or {}).get("fractional_diff") or {}
        if ffd_cfg.get("enabled", False) and not pwf.empty:
            _req = {"close", "symbol", "timestamp"}
            if _req.issubset(pwf.columns):
                from src.assembled_core.features.fractional_diff import apply_ffd_to_panel

                pwf = apply_ffd_to_panel(
                    pwf,
                    price_cols=["close"],
                    d=float(ffd_cfg.get("d", 0.4)),
                )
    except Exception as e:
        log.debug("[FFD] fractional_diff skipped: %s", e)

    return pwf, prices_latest_update


# ---------------------------------------------------------------------------
# generate_signals — Stage 3 (stub)
# ---------------------------------------------------------------------------


def generate_signals(
    features: pd.DataFrame,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply signal_fn + real signal enrichment layers.

    Real steps kept:
      - Step 3: signal_fn (caller-provided) + reduce to latest bar
      - Zombie killer: force-FLAT signals for held positions past hold limit
      - Step 3.1: Intel signal layer (modifies scores for ctx-wired intel dims)
      - Step 3.2: Sector rotation signals (adds new rows)
      - Step 3.3: Earnings guard (suppresses pre-earnings signals)
      - Step 3.35: News→Signal bridge (modifies signals from news data)
      - Step 3.4 Bayesian: confidence scaling on scores
      - Step 3.5: Crash prediction + short signal generation
      - Step 3.4 MR: mean-reversion signal column (mr_signal)
      - Step 3.55: Multi-factor composite mf_score column
      - Step 3.6: Ranking hysteresis (anti-churn filter)
      - Step 3.62: MA-crossover trend_ma_score column

    Dropped (meta-only, no signal data changes):
      - Steps 3.45 (factor timing meta), 3.58 (signal correlation meta),
        3.7 (meta-model meta-only path), 3.75 (FDR meta), 3.8 (conformal meta),
        3.86-3.91 (all observability), 3.9 (adversarial validation meta)

    Returns signals DataFrame (columns: timestamp, symbol, direction, score).
    Raises ValueError if signals are missing required columns after generation.
    """
    if log is None:
        log = logger

    try:
        policy = load_policy()
    except Exception:
        policy = {}

    from src.assembled_core.config.settings import get_settings
    from src.assembled_core.risk.zombie_killer import get_zombie_positions

    # --- Step 3: Call signal_fn ---
    signals = ctx.signal_fn(features)

    # Reduce to latest bar per symbol if configured
    settings = get_settings()
    if (
        settings.reduce_signals_to_latest_bar
        and ctx.mode in ("backtest", "eod", "paper", "live")
        and "timestamp" in signals.columns
        and not signals.empty
    ):
        signals["_ts"] = pd.to_datetime(signals["timestamp"], utc=True)
        signals = (
            signals.sort_values("_ts")
            .groupby("symbol", group_keys=False)
            .last()
            .reset_index()
            .drop(columns=["_ts"])
        )

    # Validate required columns
    required_cols = ["timestamp", "symbol", "direction"]
    missing = [c for c in required_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"signals missing required columns: {', '.join(missing)}")

    log.debug("Signals generated: %d rows", len(signals))

    # --- Zombie killer: force-FLAT for positions held too long ---
    try:
        if ctx.current_positions is not None and not ctx.current_positions.empty and not signals.empty:
            now_utc = pd.Timestamp.now("UTC").to_pydatetime()
            zombies = get_zombie_positions(ctx.current_positions.to_dict("records"), now_utc, policy)
            if zombies:
                from src.assembled_core.ops.shadow_recorder import is_shadow_only, record_shadow

                zk_shadow = is_shadow_only(policy, "zombie_killer")
                zombie_symbols = {pos["symbol"] for pos, _reason in zombies}
                record_shadow(
                    "zombie_killer",
                    {"zombie_symbols": sorted(zombie_symbols), "would_force_flat": sorted(zombie_symbols)},
                    as_of=str(ctx.as_of) if ctx.as_of is not None else None,
                    meta={"zombies_found": len(zombies), "applied": not zk_shadow},
                )
                for _pos, reason in zombies:
                    log.warning(reason)
                if not zk_shadow:
                    mask = signals["symbol"].isin(zombie_symbols)
                    signals.loc[mask, "direction"] = "FLAT"
                    existing_syms = set(signals["symbol"].values)
                    missing_zombies = zombie_symbols - existing_syms
                    if missing_zombies:
                        zombie_rows = pd.DataFrame({
                            "timestamp": [ctx.as_of or pd.Timestamp.now("UTC")] * len(missing_zombies),
                            "symbol": list(missing_zombies),
                            "direction": ["FLAT"] * len(missing_zombies),
                            "score": [0.0] * len(missing_zombies),
                        })
                        signals = pd.concat([signals, zombie_rows], ignore_index=True)
    except Exception as e:
        log.debug("zombie_killer check skipped: %s", e)

    # --- Step 3.1: Intel signal layer ---
    try:
        intel_sig_cfg = (policy.get("intel") or {}).get("signal_layer") or {}
        if intel_sig_cfg.get("enabled", False) and not signals.empty:
            from src.assembled_core.signals.intel_signal_adapter import (
                IntelSignalAdapter,
                compute_symbol_intel_scores,
            )

            sector_impacts = getattr(ctx, "intel_sector_impacts", None)
            supply_vuln = getattr(ctx, "intel_supply_vulnerability", None)
            sanctions_ben = getattr(ctx, "intel_sanctions_beneficiary", None)
            chokepoint_exp = getattr(ctx, "intel_chokepoint_exposure", None)
            intel_conf = getattr(ctx, "intel_confidence", None)

            if any(x is not None for x in [sector_impacts, supply_vuln, sanctions_ben, chokepoint_exp]):
                raw_scores = compute_symbol_intel_scores(
                    sector_impacts=sector_impacts,
                    supply_chain_vulnerability=supply_vuln,
                    sanctions_beneficiary=sanctions_ben,
                    chokepoint_exposure=chokepoint_exp,
                    confidence=intel_conf,
                )
                if raw_scores and "score" in signals.columns:
                    intel_weight = float(intel_sig_cfg.get("weight", 0.15))
                    for idx, row in signals.iterrows():
                        sym = row.get("symbol", "")
                        if sym in raw_scores:
                            signals.at[idx, "score"] = float(row["score"]) + intel_weight * raw_scores[sym]
                    log.info("[INTEL] signal layer applied: %d symbols scored", len(raw_scores))

            active_shocks = getattr(ctx, "intel_active_shocks", None)
            if active_shocks:
                adapter = IntelSignalAdapter(
                    allow_short_signals=intel_sig_cfg.get("allow_short", False),
                    min_confidence=float(intel_sig_cfg.get("min_confidence", 0.50)),
                )
                shock_df = adapter.enrich_signals_with_shock_beneficiaries(
                    active_shocks,
                    base_confidence=float(intel_sig_cfg.get("shock_confidence", 0.60)),
                )
                if not shock_df.empty:
                    existing_syms = set(signals["symbol"].values)
                    new_shock = shock_df[~shock_df["symbol"].isin(existing_syms)].copy()
                    if not new_shock.empty:
                        new_shock["timestamp"] = ctx.as_of or pd.Timestamp.now("UTC")
                        new_shock["direction"] = "LONG"
                        signals = pd.concat(
                            [signals, new_shock[["timestamp", "symbol", "direction", "score"]]],
                            ignore_index=True,
                        )
    except Exception as e:
        log.debug("[INTEL] intel_signal_layer skipped: %s", e)

    # --- Step 3.2: Sector rotation signals ---
    try:
        sr_cfg = (policy.get("signal_generation") or {}).get("sector_rotation") or {}
        if sr_cfg.get("enabled", False):
            from src.assembled_core.signals.sector_rotation import (
                generate_sector_rotation_signals,
                get_sector_weights,
            )

            scores_row = getattr(ctx, "sector_rotation_scores", None)
            if scores_row is not None:
                sr_signals = generate_sector_rotation_signals(scores_row)
                sr_weights = get_sector_weights(
                    sr_signals,
                    long_weight=float(sr_cfg.get("long_weight", 0.12)),
                    short_weight=float(sr_cfg.get("short_weight", 0.08)),
                )
                if sr_weights:
                    ts_now = ctx.as_of or pd.Timestamp.now("UTC")
                    existing_syms = set(signals["symbol"].values) if not signals.empty else set()
                    sr_rows = [
                        {"timestamp": ts_now, "symbol": sym,
                         "direction": "LONG" if w > 0 else "SHORT", "score": round(w, 4)}
                        for sym, w in sr_weights.items()
                        if sym not in existing_syms
                    ]
                    if sr_rows:
                        signals = pd.concat([signals, pd.DataFrame(sr_rows)], ignore_index=True)
    except Exception as e:
        log.debug("[SIGNAL-DIAG] sector_rotation skipped: %s", e)

    # --- Step 3.3: Earnings guard ---
    try:
        eg_cfg = (policy.get("signal_generation") or {}).get("earnings_guard") or {}
        if eg_cfg.get("enabled", False) and not signals.empty:
            from src.assembled_core.signals.earnings_integration import apply_earnings_integration

            earnings_calendar = getattr(ctx, "earnings_calendar", None)
            earnings_events = getattr(ctx, "earnings_events", None)
            if earnings_calendar is not None or earnings_events is not None:
                adjusted_signals, _eg_result = apply_earnings_integration(
                    signals,
                    earnings_calendar=earnings_calendar,
                    earnings_events=earnings_events,
                    as_of=ctx.as_of or pd.Timestamp.now("UTC"),
                    suppress_window=int(eg_cfg.get("suppress_window", 3)),
                    pead_window_days=int(eg_cfg.get("pead_window_days", 60)),
                    pead_weight=float(eg_cfg.get("pead_weight", 0.15)),
                )
                signals = adjusted_signals
    except Exception as e:
        log.debug("[SIGNAL-DIAG] earnings_guard skipped: %s", e)

    # --- Step 3.35: News→Signal bridge ---
    try:
        from src.assembled_core.signals.news_signal_bridge import load_and_apply_news_signals

        root_for_news = Path(ctx.data_root) if getattr(ctx, "data_root", None) else Path.cwd()
        signals, _nsb_meta = load_and_apply_news_signals(
            signals, root=root_for_news, policy=policy, as_of=ctx.as_of
        )
    except Exception as e:
        log.debug("[SIGNAL-DIAG] news_signal_bridge skipped: %s", e)

    # --- Step 3.36: News IC-weights (BACKLOG — Phase 2c) ---
    # news_ml_bridge.get_event_type_ic_weights() was archived to
    # archive/observability_graveyard_2026q2/ml/news_ml_bridge.py.
    # IC weights require a calibrated historical event→alpha dataset.
    # Wire in once news_ground_truth labels (tests/news_gold/) are complete.
    # See: autonome_weiterarbeit/MIGRATION_TRADING_CYCLE_V2.md §Phase 2c

    # --- Step 3.4a: Bayesian signal confidence scoring ---
    try:
        bc_cfg = (policy.get("signal_generation") or {}).get("bayesian_confidence") or {}
        if bc_cfg.get("enabled", False) and not signals.empty and "score" in signals.columns:
            from src.assembled_core.signals.signal_confidence import (
                compute_signal_confidence,
                confidence_position_scaler,
            )

            current_scores = signals.set_index("symbol")["score"].dropna()
            if len(current_scores) >= 2:
                confidences = compute_signal_confidence(
                    current_scores,
                    historical_scores=getattr(ctx, "signal_historical_scores", None),
                    ci_level=float(bc_cfg.get("ci_level", 0.90)),
                )
                for idx, row in signals.iterrows():
                    sym = row.get("symbol", "")
                    if sym in confidences:
                        scaler = confidence_position_scaler(
                            confidences[sym],
                            max_scale=float(bc_cfg.get("max_scale", 1.5)),
                            min_scale=float(bc_cfg.get("min_scale", 0.5)),
                        )
                        signals.at[idx, "score"] = float(row["score"]) * scaler
    except Exception as e:
        log.debug("[SIGNAL-DIAG] bayesian_confidence skipped: %s", e)

    # --- Step 3.5: Crash prediction + short signals ---
    try:
        shorts_policy = policy.get("shorts", {})
        if shorts_policy.get("enabled", False):
            from src.assembled_core.signals.crash_prediction import CrashPredictionEngine
            from src.assembled_core.signals.short_signals import ShortSignalGenerator
            from src.assembled_core.risk.short_risk import ShortRiskManager

            macro_data: dict = {}
            if ctx.prices is not None and not ctx.prices.empty and "VIX" in ctx.prices.columns:
                macro_data["vix"] = float(ctx.prices["VIX"].iloc[-1])

            crash_engine = CrashPredictionEngine()
            crash_signal = crash_engine.predict(
                market_data=ctx.prices,
                regime=getattr(ctx, "regime_state", None),
                intel_state=getattr(ctx, "crisis_state_intel", None),
                macro_data=macro_data or None,
            )
            if crash_signal.crash_probability >= float(shorts_policy.get("min_crash_probability", 0.60)):
                short_gen = ShortSignalGenerator(policy=shorts_policy)
                short_df = short_gen.generate_short_targets(
                    crash_signal=crash_signal,
                    universe=ctx.universe if hasattr(ctx, "universe") and ctx.universe is not None else pd.DataFrame(),
                    prices=ctx.prices,
                    regime=getattr(ctx, "regime_state", None),
                )
                risk_mgr = ShortRiskManager(policy=policy)
                risk_check = risk_mgr.validate_short_targets(short_df, regime=getattr(ctx, "regime_state", None))
                if risk_check.passed and not short_df.empty:
                    existing_syms = set(signals["symbol"].values) if not signals.empty else set()
                    short_rows = [
                        {"timestamp": ctx.as_of or pd.Timestamp.now("UTC"),
                         "symbol": row["symbol"],
                         "direction": row.get("direction", "SHORT"),
                         "score": -abs(row["confidence"])}
                        for _, row in short_df.iterrows()
                        if row["symbol"] not in existing_syms
                    ]
                    if short_rows:
                        signals = pd.concat([signals, pd.DataFrame(short_rows)], ignore_index=True)
    except Exception as e:
        log.debug("crash_prediction step skipped: %s", e)

    # --- Step 3.4b: MR signal column (mr_signal) ---
    try:
        mr_cfg = (policy.get("signals") or {}).get("mean_reversion") or {}
        if mr_cfg.get("enabled", False) and not features.empty:
            from src.assembled_core.signals.mean_reversion import compute_mean_reversion_signals

            _mr_signals = compute_mean_reversion_signals(features, regime=str(getattr(ctx, "regime_state", "bull") or "bull"))
            if not _mr_signals.empty and not signals.empty:
                _mr_map = _mr_signals.set_index("symbol")["reversion_signal"].to_dict()
                signals = signals.copy()
                signals["mr_signal"] = signals["symbol"].map(_mr_map)
    except Exception as e:
        log.debug("[MR-SIGNAL] mean_reversion_signals skipped: %s", e)

    # --- Step 3.55: Multi-factor composite mf_score column ---
    try:
        mf_cfg = policy.get("multifactor_signal") or {}
        if mf_cfg.get("enabled", False) and not features.empty and not signals.empty:
            import pathlib as _pl

            from src.assembled_core.config.factor_bundles import load_factor_bundle
            from src.assembled_core.signals.multifactor_signal import build_multifactor_signal

            _bundle_path = _pl.Path(mf_cfg.get("bundle_path", "configs/factor_bundles/macro_world_etfs_core_bundle.yaml"))
            if _bundle_path.exists():
                _mf_result = build_multifactor_signal(features, load_factor_bundle(_bundle_path))
                if not _mf_result.df.empty and "mf_score" in _mf_result.df.columns:
                    _mf_latest = (
                        _mf_result.df.sort_values("timestamp").groupby("symbol")["mf_score"].last()
                        if "timestamp" in _mf_result.df.columns
                        else _mf_result.df.groupby("symbol")["mf_score"].last()
                    )
                    signals = signals.copy()
                    signals["mf_score"] = signals["symbol"].map(_mf_latest)
    except Exception as e:
        log.debug("[MULTIFACTOR] multifactor_signal skipped: %s", e)

    # --- Step 3.6: Ranking hysteresis (anti-churn) ---
    try:
        anti_churn_cfg = policy.get("anti_churn") or {}
        if anti_churn_cfg.get("ranking_hysteresis_enabled", False) and not signals.empty:
            from src.assembled_core.paper.ranking_hysteresis import apply_ranking_hysteresis

            held_symbols: set[str] = set()
            if (ctx.current_positions is not None and not ctx.current_positions.empty
                    and "symbol" in ctx.current_positions.columns):
                held_symbols = set(ctx.current_positions["symbol"].tolist())
            signals, _rh_meta = apply_ranking_hysteresis(
                signals, held_symbols,
                entry_n=int(anti_churn_cfg.get("entry_n", 5)),
                hold_n=int(anti_churn_cfg.get("hold_n", 7)),
            )
    except Exception as e:
        log.debug("[ANTI-CHURN] ranking_hysteresis skipped: %s", e)

    # --- Step 3.62: MA-crossover trend_ma_score column ---
    try:
        if not features.empty and not signals.empty:
            _req = {"timestamp", "symbol", "close"}
            if _req.issubset(set(features.columns)):
                from src.assembled_core.signals.rules_trend import generate_trend_signals

                _ts_signals = generate_trend_signals(features, ma_fast=20, ma_slow=50)
                if not _ts_signals.empty and "symbol" in _ts_signals.columns:
                    _ts_latest = (
                        _ts_signals.sort_values("timestamp")
                        .groupby("symbol")
                        .last()
                        .reset_index()[["symbol", "score"]]
                        .rename(columns={"score": "trend_ma_score"})
                    )
                    signals = signals.merge(_ts_latest, on="symbol", how="left")
    except Exception as e:
        log.debug("[TREND-MA] rules_trend skipped: %s", e)

    # --- Step 3.9: Evidence-grade gate (optional, filters news-backed signals) ---
    try:
        ev_cfg = policy.get("evidence_gate") or {}
        if ev_cfg.get("enabled", False):
            _news_ev_df = getattr(ctx, "news_events_df", None)
            signals, _ev_audit = _apply_evidence_gate(signals, _news_ev_df, policy)
            result_meta_ref = getattr(ctx, "_evidence_gate_audit", None)
            ctx.__dict__["_evidence_gate_audit"] = _ev_audit
    except Exception as e:
        log.debug("[EVIDENCE-GATE] skipped: %s", e)

    return signals


# ---------------------------------------------------------------------------
# _apply_evidence_gate — Phase 2a helper
# ---------------------------------------------------------------------------


def _apply_evidence_gate(
    signals: "pd.DataFrame",
    news_events: "pd.DataFrame | None",
    policy: dict,
) -> "tuple[pd.DataFrame, dict]":
    """Filter signals through evidence-grade gate. Returns (filtered_signals, audit_info).

    Applies only to signals whose symbol appears in ``news_events`` with
    insufficient evidence quality.  Signals for symbols not in news_events
    (i.e. non-news-driven signals) always pass through.

    Policy key: evidence_gate.require_grade (default "B").
    """
    import pandas as _pd
    from src.assembled_core.events.evidence_engine.grader import grade_evidence
    from src.assembled_core.events.evidence_engine.action_gate import check_evidence_grade_gate
    from src.assembled_core.events.evidence_engine.misinfo_risk import compute_misinfo_risk

    cfg = policy.get("evidence_gate") or {}

    audit: dict = {
        "enabled": bool(cfg.get("enabled", False)),
        "filtered_count": 0,
        "total_signals": len(signals),
        "filtered_symbols": [],
    }

    if not cfg.get("enabled", False):
        return signals, audit

    if news_events is None or not isinstance(news_events, _pd.DataFrame) or news_events.empty:
        audit["reason"] = "no_news_events"
        return signals, audit

    if signals.empty:
        return signals, audit

    require_grade = str(cfg.get("require_grade", "B")).upper()
    if require_grade not in {"A", "B", "C", "D"}:
        require_grade = "B"

    sym_col = "symbol" if "symbol" in news_events.columns else None
    tier_col = "source_tier" if "source_tier" in news_events.columns else None
    if sym_col is None or tier_col is None:
        audit["reason"] = "missing_required_columns"
        return signals, audit

    failing_symbols: set[str] = set()
    symbol_evidence: dict[str, dict] = {}

    for sym, grp in news_events.groupby(sym_col):
        tiers = grp[tier_col].str.upper().fillna("T3")
        n_src_a = int((tiers == "T1").sum())
        n_src_b = int((tiers == "T2").sum())
        n_src_b_ind = int(grp["source_id"].nunique()) if "source_id" in grp.columns else n_src_b
        evidence_summary = {
            "tierA_count": n_src_a,
            "tierB_count": n_src_b,
            "tierB_independent_count": n_src_b_ind,
            "evidence_ok": n_src_a >= 1 or n_src_b_ind >= 2,
        }
        social_only = bool((tiers.isin({"T3", "SOCIAL"})).all()) if len(grp) > 0 else False
        misinfo_score = compute_misinfo_risk(evidence_summary, social_only=social_only, event_count=len(grp))
        grade = grade_evidence(evidence_summary, misinfo_risk_score=misinfo_score)
        ok, reason = check_evidence_grade_gate(grade, require_for_active=require_grade)
        symbol_evidence[str(sym)] = {"grade": grade.value, "ok": ok, "reason": reason}
        if not ok:
            failing_symbols.add(str(sym))

    audit["symbol_evidence"] = symbol_evidence

    if not failing_symbols:
        return signals, audit

    mask_keep = ~signals["symbol"].isin(failing_symbols)
    filtered = signals[mask_keep].reset_index(drop=True)
    audit["filtered_count"] = len(signals) - len(filtered)
    audit["filtered_symbols"] = list(failing_symbols)
    return filtered, audit


# ---------------------------------------------------------------------------
# _compute_news_triggers — Phase 2b helper
# ---------------------------------------------------------------------------


def _compute_news_triggers(
    news_events: "pd.DataFrame | None",
    policy: dict,
    *,
    as_of: "datetime | None" = None,
) -> "pd.DataFrame":
    """Convert processed news events into actionable triggers.

    Pipeline (order matters):
      1. Fingerprint deduplication (simhash64 + hamming distance — title-level)
      2. Greedy TF-IDF cosine clustering (groups semantically similar events)
      3. Rule-based trigger scoring (source_tier weight + burst bonus)

    Note: burst detection via compute_bursts_for_window requires pre-built cluster
    dicts and baseline counts; that integration is handled upstream by run_news_pipeline.
    Here we apply a lightweight burst bonus: events where published_utc is within
    ``burst_window_minutes`` of the most recent event get a +0.2 score boost.

    Returns a DataFrame with columns: symbol, trigger_score, cluster_id, dedup_kept.
    Returns empty DataFrame if news_events is None/empty or any step fails.
    """
    import pandas as _pd
    from src.assembled_core.events.news.fingerprint import simhash64, hamming_distance
    from src.assembled_core.events.news.tfidf import build_tfidf_vectors

    if news_events is None or not isinstance(news_events, _pd.DataFrame) or news_events.empty:
        return _pd.DataFrame()

    cfg = policy.get("news_triggers") or {}
    hamming_threshold = int(cfg.get("dedup_hamming_threshold", 3))
    cosine_threshold = float(cfg.get("cluster_cosine_threshold", 0.75))
    burst_window_minutes = int(cfg.get("burst_window_minutes", 60))

    text_col = next((c for c in ("title", "headline", "text") if c in news_events.columns), None)
    sym_col = "symbol" if "symbol" in news_events.columns else None

    if text_col is None:
        return _pd.DataFrame()

    # Step 1: Fingerprint deduplication
    try:
        hashes = [simhash64(str(t)) for t in news_events[text_col].fillna("")]
        keep_mask = [True] * len(news_events)
        for i in range(len(hashes)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(hashes)):
                if keep_mask[j] and hamming_distance(hashes[i], hashes[j]) <= hamming_threshold:
                    keep_mask[j] = False
        deduped = news_events[keep_mask].copy().reset_index(drop=True)
        deduped["dedup_kept"] = True
    except Exception:
        deduped = news_events.copy()
        deduped["dedup_kept"] = True

    if deduped.empty:
        return _pd.DataFrame()

    # Step 2: TF-IDF cluster grouping (greedy, O(n²) — safe for <1000 events)
    try:
        texts = deduped[text_col].fillna("").tolist()
        vectors = build_tfidf_vectors(texts)
        cluster_ids: list[int] = [-1] * len(vectors)
        next_cid = 0
        for i, v_i in enumerate(vectors):
            if cluster_ids[i] >= 0:
                continue
            cluster_ids[i] = next_cid
            for j in range(i + 1, len(vectors)):
                if cluster_ids[j] >= 0:
                    continue
                v_j = vectors[j]
                dot = sum(v_i.get(k, 0.0) * v_j.get(k, 0.0) for k in v_i)
                norm_i = sum(x * x for x in v_i.values()) ** 0.5
                norm_j = sum(x * x for x in v_j.values()) ** 0.5
                if norm_i > 0 and norm_j > 0 and dot / (norm_i * norm_j) >= cosine_threshold:
                    cluster_ids[j] = next_cid
            next_cid += 1
        deduped["cluster_id"] = cluster_ids
    except Exception:
        deduped["cluster_id"] = list(range(len(deduped)))

    # Step 3: Rule-based trigger scoring
    # Base score by source tier; burst bonus for events near the most-recent timestamp
    tier_weights = {"T1": 1.0, "T2": 0.7, "T3": 0.4}
    tier_col = "source_tier" if "source_tier" in deduped.columns else None
    if tier_col:
        deduped["trigger_score"] = deduped[tier_col].str.upper().map(tier_weights).fillna(0.4)
    else:
        deduped["trigger_score"] = 0.5

    time_col = "published_utc" if "published_utc" in deduped.columns else None
    if time_col:
        try:
            import pandas as _pd2
            times = _pd2.to_datetime(deduped[time_col], utc=True, errors="coerce")
            max_t = times.max()
            if _pd2.notna(max_t):
                burst_cutoff = max_t - _pd2.Timedelta(minutes=burst_window_minutes)
                deduped["trigger_score"] = deduped["trigger_score"] + (times >= burst_cutoff).astype(float) * 0.2
        except Exception:
            pass

    keep_cols = [c for c in (["symbol"] if sym_col else []) + ["trigger_score", "cluster_id", "dedup_kept"] if c in deduped.columns]
    return deduped[keep_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# size_positions — Stage 4
# ---------------------------------------------------------------------------


def _sp_dispatch_sizing(
    signals: pd.DataFrame,
    ctx: "TradingContext",
    prices_for_sizing: pd.DataFrame | None,
    sizing_cfg: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """Dispatch to the configured sizing method; validate and return target_positions."""
    sizing_method = sizing_cfg.get("method", "default")
    target_positions: pd.DataFrame = pd.DataFrame()
    try:
        if sizing_method == "kelly":
            from src.assembled_core.portfolio.position_sizing import compute_kelly_weights
            target_positions = compute_kelly_weights(
                signals,
                fraction=float(sizing_cfg.get("kelly_fraction", 0.5)),
                max_weight=float(sizing_cfg.get("max_weight", 0.25)),
                total_capital=ctx.capital,
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "risk_parity":
            from src.assembled_core.portfolio.position_sizing import compute_risk_parity_weights
            vols = _estimate_symbol_volatilities(prices_for_sizing, lookback=int(sizing_cfg.get("vol_lookback_days", 60)))
            target_positions = compute_risk_parity_weights(
                signals, vols, total_capital=ctx.capital,
                max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "vol_scaled":
            from src.assembled_core.portfolio.position_sizing import compute_vol_scaled_weights
            vols = _estimate_symbol_volatilities(prices_for_sizing, lookback=int(sizing_cfg.get("vol_lookback_days", 60)))
            target_positions = compute_vol_scaled_weights(
                signals, vols,
                target_vol=float(sizing_cfg.get("target_vol", 0.15)),
                total_capital=ctx.capital,
                max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "black_litterman":
            try:
                from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer
                from src.assembled_core.portfolio.covariance import estimate_covariance
                bl = BlackLittermanOptimizer(
                    risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)),
                    tau=float(sizing_cfg.get("tau", 0.05)),
                    max_position=float(sizing_cfg.get("max_weight", 0.15)),
                    min_position=float(sizing_cfg.get("min_position", 0.0)),
                )
                scores_dict: dict[str, float] = {}
                if not signals.empty and "symbol" in signals.columns and "score" in signals.columns:
                    _bl = signals[["symbol", "score"]].copy()
                    _bl["score"] = _bl["score"].astype(float).fillna(0.0)
                    scores_dict = {str(sym): float(s) for sym, s in _bl[_bl["score"].abs() > 0.01].set_index("symbol")["score"].items()}
                if scores_dict and prices_for_sizing is not None and not prices_for_sizing.empty and "close" in prices_for_sizing.columns and "symbol" in prices_for_sizing.columns:
                    _pivot = prices_for_sizing.pivot_table(index="timestamp", columns="symbol", values="close")
                    sigma = estimate_covariance(_pivot.pct_change().dropna(how="all"), method=sizing_cfg.get("cov_method", "ledoit_wolf"))
                    if not sigma.empty:
                        bl_w = bl.optimize_from_scores(
                            scores=pd.Series(scores_dict), sigma=sigma,
                            confidence=float(sizing_cfg.get("bl_confidence", 0.5)),
                        )
                        target_positions = pd.DataFrame([
                            {"symbol": s, "target_weight": round(w, 4), "target_qty": round(w * ctx.capital, 2)}
                            for s, w in bl_w.items()
                        ])
                    else:
                        target_positions = ctx.position_sizing_fn(signals, ctx.capital)
                else:
                    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("Black-Litterman sizing failed, using default: %s", e)
                target_positions = ctx.position_sizing_fn(signals, ctx.capital)
        elif sizing_method == "cost_aware":
            try:
                from src.assembled_core.portfolio.cost_aware_optimizer import OptimizerConfig, optimize_portfolio
                from src.assembled_core.portfolio.covariance import estimate_covariance
                if prices_for_sizing is not None and not prices_for_sizing.empty and not signals.empty and "close" in prices_for_sizing.columns and "symbol" in prices_for_sizing.columns:
                    _pivot_cao = prices_for_sizing.pivot_table(index="timestamp", columns="symbol", values="close")
                    sigma_cao = estimate_covariance(_pivot_cao.pct_change().dropna(how="all"), method="ledoit_wolf")
                    mu_cao = signals.set_index("symbol")["score"].reindex(sigma_cao.index).fillna(0.0) if "score" in signals.columns else pd.Series(dtype=float)
                    _cur_w: dict[str, float] = {}
                    if ctx.current_positions is not None and isinstance(ctx.current_positions, pd.DataFrame) and "symbol" in ctx.current_positions.columns:
                        _cp = ctx.current_positions
                        _wcol_cao = "weight" if "weight" in _cp.columns else "target_weight"
                        if _wcol_cao in _cp.columns:
                            _cur_w = {str(k): float(v) for k, v in _cp.set_index("symbol")[_wcol_cao].fillna(0.0).items()}
                    cao_res = optimize_portfolio(mu_cao, sigma_cao, _cur_w, OptimizerConfig(
                        risk_aversion=float(sizing_cfg.get("risk_aversion", 1.0)),
                        turnover_penalty=float(sizing_cfg.get("turnover_penalty", 0.001)),
                        max_weight=float(sizing_cfg.get("max_weight", 0.10)),
                    ))
                    target_positions = pd.DataFrame([
                        {"symbol": s, "target_weight": round(w, 4), "target_qty": round(w * ctx.capital, 2)}
                        for s, w in cao_res.weights.items() if abs(w) > 1e-6
                    ])
                else:
                    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("cost_aware_optimizer failed, using default: %s", e)
                target_positions = ctx.position_sizing_fn(signals, ctx.capital)
        elif sizing_method == "erc":
            try:
                from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights
                from src.assembled_core.portfolio.covariance import estimate_covariance
                if prices_for_sizing is not None and not prices_for_sizing.empty and not signals.empty and "close" in prices_for_sizing.columns and "symbol" in prices_for_sizing.columns:
                    _sig_syms = [s for s in signals["symbol"].tolist() if s in prices_for_sizing["symbol"].unique()]
                    if len(_sig_syms) >= 2:
                        _pivot_erc = prices_for_sizing[prices_for_sizing["symbol"].isin(_sig_syms)].pivot_table(index="timestamp", columns="symbol", values="close")
                        _rets_erc = _pivot_erc.pct_change().dropna(how="all")
                        if len(_rets_erc) >= 3:
                            sigma_erc = estimate_covariance(_rets_erc, method="ledoit_wolf")
                            erc_res = compute_erc_weights(sigma_erc, symbols=list(sigma_erc.columns), long_only=True, max_weight=float(sizing_cfg.get("max_weight", 0.25)))
                            target_positions = pd.DataFrame([
                                {"symbol": s, "target_weight": round(w, 6), "target_qty": round(w * ctx.capital, 2)}
                                for s, w in erc_res.weights.items() if abs(w) > 1e-6
                            ])
                        else:
                            target_positions = ctx.position_sizing_fn(signals, ctx.capital)
                    else:
                        target_positions = ctx.position_sizing_fn(signals, ctx.capital)
                else:
                    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("erc sizing failed, using default: %s", e)
                target_positions = ctx.position_sizing_fn(signals, ctx.capital)
        elif sizing_method == "bl_blend":
            try:
                from src.assembled_core.portfolio.bl_sizing import apply_bl_sizing
                base_tp = ctx.position_sizing_fn(signals, ctx.capital)
                if base_tp is not None and not base_tp.empty and "target_weight" in base_tp.columns and prices_for_sizing is not None and not prices_for_sizing.empty:
                    _btw = base_tp.dropna(subset=["target_weight"])
                    score_w = {str(k): float(v) for k, v in _btw.set_index("symbol")["target_weight"].items()}
                    bl_w, _ = apply_bl_sizing(score_w, prices_for_sizing, lookback_days=int(sizing_cfg.get("lookback_days", 60)), risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)), tau=float(sizing_cfg.get("tau", 0.05)), max_position=float(sizing_cfg.get("max_weight", 0.15)), confidence=float(sizing_cfg.get("bl_confidence", 0.5)), return_scale=float(sizing_cfg.get("return_scale", 0.10)), target_invested_pct=float(sizing_cfg.get("target_invested_pct", 1.0)))
                    target_positions = pd.DataFrame([{"symbol": s, "target_weight": round(w, 6), "target_qty": round(w * ctx.capital, 2)} for s, w in bl_w.items() if abs(w) > 1e-6])
                else:
                    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("bl_blend sizing failed, using default: %s", e)
                target_positions = ctx.position_sizing_fn(signals, ctx.capital)
        elif sizing_method == "hrp":
            try:
                from src.assembled_core.portfolio.hrp_sizing import apply_hrp_sizing
                base_tp = ctx.position_sizing_fn(signals, ctx.capital)
                if base_tp is not None and not base_tp.empty and "target_weight" in base_tp.columns and prices_for_sizing is not None and not prices_for_sizing.empty:
                    _btw = base_tp.dropna(subset=["target_weight"])
                    score_w = {str(k): float(v) for k, v in _btw.set_index("symbol")["target_weight"].items()}
                    blended, _ = apply_hrp_sizing(score_w, prices_for_sizing, lookback_days=int(sizing_cfg.get("lookback_days", 60)), blend=float(sizing_cfg.get("blend", 0.7)), target_invested_pct=float(sizing_cfg.get("target_invested_pct", 1.0)), min_weight=float(sizing_cfg.get("min_weight", 0.0)), max_weight=float(sizing_cfg.get("max_weight", 1.0)))
                    target_positions = pd.DataFrame([{"symbol": s, "target_weight": round(w, 6), "target_qty": round(w * ctx.capital, 2)} for s, w in blended.items() if abs(w) > 1e-6])
                else:
                    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("hrp sizing failed, using default: %s", e)
                target_positions = ctx.position_sizing_fn(signals, ctx.capital)
        elif sizing_method == "mvo":
            try:
                import numpy as _np
                from src.assembled_core.portfolio.mvo_optimizer import mvo_with_cardinality
                from src.assembled_core.portfolio.covariance import estimate_covariance
                if prices_for_sizing is not None and not prices_for_sizing.empty and not signals.empty and "close" in prices_for_sizing.columns and "symbol" in prices_for_sizing.columns:
                    _sig_syms_mvo = [s for s in signals["symbol"].tolist() if s in prices_for_sizing["symbol"].unique()]
                    if len(_sig_syms_mvo) >= 2:
                        _pivot_mvo = prices_for_sizing[prices_for_sizing["symbol"].isin(_sig_syms_mvo)].pivot_table(index="timestamp", columns="symbol", values="close")
                        _rets_mvo = _pivot_mvo.pct_change().dropna(how="all")
                        if len(_rets_mvo) >= 3:
                            sigma_mvo = estimate_covariance(_rets_mvo, method="ledoit_wolf").values
                            mvo_syms = list(_rets_mvo.columns)
                            mu_series = signals.set_index("symbol")["score"] if "score" in signals.columns else pd.Series(0.0, index=mvo_syms)
                            mu_mvo = _np.asarray(mu_series.reindex(mvo_syms).fillna(0.0).values, dtype=float)
                            w_arr = mvo_with_cardinality(mu_mvo, sigma_mvo, max_positions=int(sizing_cfg.get("max_positions", 20)), risk_aversion=float(sizing_cfg.get("risk_aversion", 1.0)), min_weight=float(sizing_cfg.get("min_weight", 0.01)))
                            target_positions = pd.DataFrame([{"symbol": s, "target_weight": round(float(w_arr[i]), 6), "target_qty": round(float(w_arr[i]) * ctx.capital, 2)} for i, s in enumerate(mvo_syms) if abs(w_arr[i]) > 1e-6])
                        else:
                            target_positions = ctx.position_sizing_fn(signals, ctx.capital)
                    else:
                        target_positions = ctx.position_sizing_fn(signals, ctx.capital)
                else:
                    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("mvo sizing failed, using default: %s", e)
                target_positions = ctx.position_sizing_fn(signals, ctx.capital)
        else:
            target_positions = ctx.position_sizing_fn(signals, ctx.capital)

        if target_positions is None or target_positions.empty:
            target_positions = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
        if not any(c in target_positions.columns for c in ["target_weight", "target_qty"]):
            raise ValueError("target_positions missing target_weight or target_qty")
    except Exception as e:
        raise ValueError(f"Error in size_positions sizing dispatch: {e}") from e
    return target_positions


def _sp_apply_liquidity(
    target_positions: pd.DataFrame,
    prices_for_sizing: pd.DataFrame | None,
    policy: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """Phase 9.5: Liquidity-aware position scaling."""
    try:
        liq_cfg = policy.get("liquidity_scoring") or {}
        if liq_cfg.get("enabled", False) and not target_positions.empty and "target_weight" in target_positions.columns and prices_for_sizing is not None and not prices_for_sizing.empty:
            from src.assembled_core.risk.liquidity_scoring import apply_liquidity_adjusted_sizing, compute_liquidity_scores
            liq_scores = compute_liquidity_scores(prices_for_sizing, lookback_days=int(liq_cfg.get("lookback_days", 60)))
            if liq_scores:
                tw_map = {str(k).upper(): float(v) for k, v in target_positions.set_index("symbol")["target_weight"].items()}
                for s in liq_scores:
                    s.symbol = s.symbol.upper()
                adjusted_tw = apply_liquidity_adjusted_sizing(tw_map, liq_scores, alpha=float(liq_cfg.get("alpha", 0.5)), min_score_threshold=float(liq_cfg.get("min_score_threshold", 0.1)))
                target_positions["target_weight"] = target_positions["symbol"].astype(str).str.upper().map(adjusted_tw).fillna(target_positions["target_weight"])
    except Exception as e:
        log.debug("liquidity_scoring skipped: %s", e)
    return target_positions


def _sp_compute_final_multiplier(
    ctx: "TradingContext",
    policy: dict,
    meta: dict,
    log: logging.Logger,
) -> float:
    """Exposure overlay: geo × profit_lock × vol_targeting × market_stress × crisis_alpha."""
    geo_multiplier = compute_exposure_multiplier(ctx, policy)
    profit_lock_mult = 1.0
    try:
        pl_cfg = policy.get("profit_lock") or {}
        if pl_cfg.get("enabled") and getattr(ctx, "equity_curve", None) is not None and getattr(ctx, "equity_curve_index", None) is not None:
            pl_state = getattr(ctx, "profit_lock_state", None) or {}
            profit_lock_mult, pl_state_out = compute_profit_lock_multiplier(ctx.equity_curve, pl_cfg, ctx.equity_curve_index, state=pl_state)
            ctx.profit_lock_state = pl_state_out
            meta["profit_lock_state"] = pl_state_out
            meta["profit_lock"] = {"multiplier": float(profit_lock_mult)}
    except Exception as e:
        log.debug("profit_lock skipped: %s", e)

    vol_scale_factor = 1.0
    try:
        vt_cfg = policy.get("vol_targeting") or {}
        if vt_cfg.get("enabled", False) and getattr(ctx, "equity_curve", None) is not None and getattr(ctx, "equity_curve_index", None) is not None:
            vol_scale_factor, realized_vol, target_vol = compute_vol_targeting_result(ctx.equity_curve, vt_cfg, now_idx=ctx.equity_curve_index)
            meta["vol_targeting"] = {"scale_factor": vol_scale_factor, "realized_vol": realized_vol, "target_vol": target_vol}
        else:
            meta["vol_targeting"] = {"scale_factor": 1.0, "realized_vol": float("nan"), "target_vol": float("nan")}
    except Exception as e:
        log.debug("vol_targeting skipped: %s", e)
        meta["vol_targeting"] = {"scale_factor": 1.0}

    ms_multiplier = 1.0
    if ctx.market_stress:
        stress_score = int(ctx.market_stress.get("stress_score", 0))
        _ms_scaling = (policy.get("market_stress") or {}).get("exposure_scaling") or {}
        if stress_score >= 2:
            ms_multiplier = float(_ms_scaling.get("stress_score_2", 0.50))
        elif stress_score >= 1:
            ms_multiplier = float(_ms_scaling.get("stress_score_1", 0.75))

    crisis_alpha_multiplier = 1.0
    if getattr(ctx, "crisis_state_intel", None):
        crisis_mode = str(ctx.crisis_state_intel.get("mode", "NORMAL")).upper()
        ca_cfg = (policy.get("crisis_alpha") or policy.get("intel", {}).get("crisis_alpha") or {})
        if crisis_mode == "CRISIS":
            crisis_alpha_multiplier = min(float(ca_cfg.get("crisis_multiplier", 0.25)), 1.0)
        elif crisis_mode == "ELEVATED":
            crisis_alpha_multiplier = min(float(ca_cfg.get("elevated_multiplier", 0.60)), 1.0)

    final_multiplier = geo_multiplier * profit_lock_mult * vol_scale_factor * ms_multiplier * crisis_alpha_multiplier
    _MIN_EXPOSURE_MULT = 0.05
    if final_multiplier < _MIN_EXPOSURE_MULT:
        log.warning("[SIZE] exposure multiplier %.4f below floor %.2f — clamping", final_multiplier, _MIN_EXPOSURE_MULT)
        final_multiplier = _MIN_EXPOSURE_MULT
    return final_multiplier


def _sp_apply_factor_risk(
    target_positions: pd.DataFrame,
    prices_for_sizing: pd.DataFrame | None,
    policy: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """D2: Factor risk model vol check — rescales target_weight if portfolio_vol > limit."""
    try:
        factor_risk_cfg = policy.get("factor_risk", {})
        if factor_risk_cfg.get("enabled", False) and prices_for_sizing is not None and not prices_for_sizing.empty:
            from src.assembled_core.risk.factor_risk_model import FactorRiskModel
            frm = FactorRiskModel()
            frm.fit(prices_for_sizing)
            if "target_weight" in target_positions.columns:
                tw_dict = dict(zip(target_positions["symbol"], target_positions["target_weight"].fillna(0)))
                portfolio_vol = frm.predict_portfolio_vol(tw_dict)
                vol_limit = float(factor_risk_cfg.get("max_portfolio_vol", 0.25))
                if portfolio_vol > vol_limit and portfolio_vol > 0:
                    scale = vol_limit / portfolio_vol
                    target_positions["target_weight"] = target_positions["target_weight"] * scale
                    log.info("FACTOR_RISK: portfolio_vol=%.3f > limit=%.3f → scaled by %.3f", portfolio_vol, vol_limit, scale)
    except Exception as e:
        log.debug("factor_risk_model skipped: %s", e)
    return target_positions


def _sp_apply_trailing_stops(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    prices_filtered: pd.DataFrame | None,
    policy: dict,
    meta: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """Phase 11.5: Trailing stops (regime-adaptive ATR)."""
    try:
        ts_cfg = policy.get("trailing_stops") or {}
        if ts_cfg.get("enabled", False) and not target_positions.empty:
            current_positions_df = ctx.current_positions
            if current_positions_df is not None and not current_positions_df.empty:
                from src.assembled_core.risk.trailing_stops import apply_stop_reductions_to_weights, compute_trailing_stops
                pos_map: dict[str, dict] = {}
                for _, row in current_positions_df.iterrows():
                    sym = str(row.get("symbol", "")).upper()
                    entry = row.get("avg_entry_price") or row.get("entry_price") or row.get("price")
                    if sym and entry is not None:
                        pos_map[sym] = {"entry_price": float(entry), "qty": float(row.get("qty", 0.0) or 0.0), "weight": float(row.get("weight", 0.0) or 0.0)}
                rs_meta = meta.get("risk_state") or {}
                regime_label = str(rs_meta.get("regime", "unknown")).lower()
                vix_level = ctx.market_stress.get("vix_level") if ctx.market_stress else None
                if pos_map and prices_filtered is not None:
                    ts_result = compute_trailing_stops(pos_map, prices_filtered, regime=regime_label, atr_window=int(ts_cfg.get("atr_window", 14)), vix_level=vix_level)
                    if ts_result.triggered_symbols or ts_result.reduction_symbols:
                        tw_col = "target_weight" if "target_weight" in target_positions.columns else "weight"
                        if tw_col in target_positions.columns:
                            weights_map = {str(k).upper(): float(v) for k, v in target_positions.set_index("symbol")[tw_col].items()}
                            adjusted = apply_stop_reductions_to_weights(weights_map, ts_result)
                            target_positions[tw_col] = target_positions["symbol"].astype(str).str.upper().map(adjusted).fillna(target_positions[tw_col])
                            if "target_qty" in target_positions.columns:
                                for sym in ts_result.triggered_symbols:
                                    target_positions.loc[target_positions["symbol"].astype(str).str.upper() == sym, "target_qty"] = 0.0
    except Exception as e:
        log.debug("trailing_stops skipped: %s", e)
    return target_positions


def _sp_apply_turnover_gate(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    prices_for_sizing: pd.DataFrame | None,
    prices_latest: pd.DataFrame | None,
    policy: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """Turnover budget gate (INT-6)."""
    tb = policy.get("turnover_budget") or {}
    if tb.get("enabled", False) and not target_positions.empty:
        try:
            cap = float(tb.get("cap", 0.15) or 0.15)
            prices_for_turnover = prices_latest if prices_latest is not None and not prices_latest.empty else prices_for_sizing
            estimated = estimate_turnover(ctx.current_positions, target_positions, prices_for_turnover, portfolio_value=ctx.capital)
            _invested_pct = None
            if ctx.capital > 0 and ctx.current_positions is not None and not ctx.current_positions.empty and "qty" in ctx.current_positions.columns:
                _price_s = prices_for_turnover.groupby("symbol")["close"].last() if (prices_for_turnover is not None and not prices_for_turnover.empty and "close" in prices_for_turnover.columns) else pd.Series(dtype=float)
                _cp = ctx.current_positions
                _inv = float((_cp["qty"].fillna(0).astype(float) * _cp["symbol"].map(_price_s).fillna(0)).sum())
                _invested_pct = _inv / ctx.capital
            target_positions, _scale = apply_turnover_gate(
                target_positions, ctx.current_positions, cap=cap,
                estimated_turnover=1.0 if estimated == float("inf") else estimated,
                behavior="block" if estimated == float("inf") else str(tb.get("behavior", "scale") or "scale"),
                prices=prices_for_turnover, portfolio_value=ctx.capital,
                invested_pct=_invested_pct, target_invested_pct=float(tb.get("target_invested_pct", 0.80) or 0.80),
            )
        except Exception as e:
            log.debug("turnover_budget gate skipped: %s", e)
    return target_positions


def _sp_apply_correlation_guard(
    target_positions: pd.DataFrame,
    prices_for_sizing: pd.DataFrame | None,
    policy: dict,
    ctx: "TradingContext",
) -> pd.DataFrame:
    """Correlation guard (M6-T07) + regime shift exposure scaling."""
    try:
        if not target_positions.empty and len(target_positions) >= 2 and "target_weight" in target_positions.columns:
            tw_dict_cg = dict(zip(target_positions["symbol"], target_positions["target_weight"]))
            corr_prices = prices_for_sizing
            adjusted_weights, corr_reasons = apply_correlation_guard(tw_dict_cg, corr_prices, policy)
            if corr_reasons:
                from src.assembled_core.ops.shadow_recorder import is_shadow_only, record_shadow
                cg_shadow = is_shadow_only(policy, "correlation_guard")
                record_shadow("correlation_guard", {"adjusted_weights": adjusted_weights}, as_of=str(ctx.as_of) if ctx.as_of else None, meta={"applied": not cg_shadow})
                if not cg_shadow:
                    target_positions["target_weight"] = target_positions["symbol"].map(adjusted_weights)
                    if "target_qty" in target_positions.columns:
                        target_positions["target_qty"] = target_positions["target_weight"] * ctx.capital
            symbols_in_portfolio = list(target_positions["symbol"].unique())
            if len(symbols_in_portfolio) >= 2:
                shift_result = detect_correlation_regime_shift(corr_prices, symbols_in_portfolio)
                if shift_result.get("regime_shift_detected", False):
                    exp_scale = shift_result["exposure_scale"]
                    target_positions["target_weight"] *= exp_scale
                    if "target_qty" in target_positions.columns:
                        target_positions["target_qty"] *= exp_scale
    except Exception as e:
        logger.debug("correlation_guard skipped: %s", e)
    return target_positions


def _sp_apply_crash_cap(
    target_positions: pd.DataFrame,
    policy: dict,
    meta: dict,
    as_of_str: str | None,
) -> pd.DataFrame:
    """D3: Crash prediction equity cap."""
    try:
        cp_meta = meta.get("crash_prediction", {}) or {}
        crash_prob = float(cp_meta.get("crash_probability", 0.0) or 0.0)
        cp_cfg = policy.get("crash_prediction", {}) or {}
        if cp_cfg.get("equity_cap_enabled", False) and not target_positions.empty and "target_weight" in target_positions.columns:
            threshold = float(cp_cfg.get("equity_cap_threshold", 0.4))
            if crash_prob > threshold:
                from src.assembled_core.ops.shadow_recorder import is_shadow_only, record_shadow
                base_long_gross = float(cp_cfg.get("base_long_gross", 1.0))
                cap_val = max(0.5 - crash_prob, 0.0) * base_long_gross
                long_mask = target_positions["target_weight"] > 0
                current_long_gross = float(target_positions.loc[long_mask, "target_weight"].sum())
                scale = min(cap_val / current_long_gross, 1.0) if current_long_gross > 0.0 else 1.0
                cp_shadow = is_shadow_only(policy, "crash_prediction")
                record_shadow("crash_prediction_cap", {"cap": cap_val, "scale": scale}, as_of=as_of_str, meta={"applied": (not cp_shadow) and scale < 1.0})
                if not cp_shadow and scale < 1.0:
                    target_positions.loc[long_mask, "target_weight"] *= scale
                    if "target_qty" in target_positions.columns:
                        target_positions.loc[long_mask, "target_qty"] *= scale
    except Exception as e:
        logger.debug("crash_prediction equity cap skipped: %s", e)
    return target_positions


def _sp_apply_inverse_etf(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    meta: dict,
) -> pd.DataFrame:
    """D4: Inverse-ETF tail hedge."""
    try:
        ie_cfg = policy.get("inverse_etf", {}) or {}
        if ie_cfg.get("enabled", False):
            cp_meta = meta.get("crash_prediction", {}) or {}
            crash_prob = float(cp_meta.get("crash_probability", 0.0) or 0.0)
            vix_val: float | None = None
            if ctx.prices is not None and not ctx.prices.empty and "VIX" in ctx.prices.columns:
                try:
                    vix_val = float(ctx.prices["VIX"].iloc[-1])
                except Exception:
                    pass
            if vix_val is not None and vix_val > float(ie_cfg.get("vix_threshold", 25.0)) and crash_prob > float(ie_cfg.get("crash_prob_threshold", 0.4)):
                from src.assembled_core.ops.shadow_recorder import is_shadow_only, record_shadow
                from src.assembled_core.portfolio.inverse_etf_selector import InverseETFSelector
                selector = InverseETFSelector(allow_2x=False, allow_3x=False)
                hedge_sym = selector.select_best_short_instrument("BROAD", severity=float(cp_meta.get("severity", 0.5) or 0.5), holding_period_days=int(ie_cfg.get("max_holding_days", 5)))
                hedge_ratio = float(ie_cfg.get("hedge_ratio", 0.1))
                ie_shadow = is_shadow_only(policy, "inverse_etf")
                record_shadow("inverse_etf", {"hedge_symbol": hedge_sym, "hedge_weight": hedge_ratio}, as_of=str(ctx.as_of) if ctx.as_of else None, meta={"applied": (not ie_shadow) and hedge_sym is not None})
                if not ie_shadow and hedge_sym and "target_weight" in target_positions.columns:
                    if hedge_sym not in target_positions["symbol"].values:
                        target_positions = pd.concat([target_positions, pd.DataFrame([{"symbol": hedge_sym, "target_weight": hedge_ratio, "target_qty": hedge_ratio * ctx.capital}])], ignore_index=True)
    except Exception as e:
        logger.debug("inverse_etf hedge skipped: %s", e)
    return target_positions


def _sp_apply_quantile_asymmetry(
    target_positions: pd.DataFrame,
    prices_with_features: pd.DataFrame | None,
    policy: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """Quantile asymmetry sizing — reduces high-downside-skew positions."""
    try:
        qm_cfg = (policy.get("risk", {}) or {}).get("quantile_sizing", {}) or {}
        if qm_cfg.get("enabled", False) and not target_positions.empty and "target_weight" in target_positions.columns and prices_with_features is not None and not prices_with_features.empty:
            from src.assembled_core.ml.quantile_models import predict_quantiles
            _feature_cols = qm_cfg.get("feature_cols", [])
            _target_col = qm_cfg.get("target_col", "return_1d")
            if _feature_cols and _target_col in prices_with_features.columns:
                _valid_fcols = [c for c in _feature_cols if c in prices_with_features.columns]
                if _valid_fcols:
                    _qpreds = predict_quantiles(prices_with_features, target_col=_target_col, feature_cols=_valid_fcols)
                    _asym_map = {qp.symbol: qp.asymmetry for qp in _qpreds}
                    _asym_thresh = float(qm_cfg.get("asymmetry_threshold", 1.5))
                    _asym_red = float(qm_cfg.get("asymmetry_reduction", 0.5))
                    mask = target_positions["symbol"].map(lambda s: _asym_map.get(s, 0.0) > _asym_thresh)
                    target_positions.loc[mask, "target_weight"] *= _asym_red
                    if "target_qty" in target_positions.columns:
                        target_positions.loc[mask, "target_qty"] *= _asym_red
    except Exception as e:
        log.debug("quantile_asymmetry skipped: %s", e)
    return target_positions


def _sp_apply_crowding_cap(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    log: logging.Logger,
) -> pd.DataFrame:
    """Crowding detector (HHI cap) — clamps max position when portfolio is too concentrated."""
    try:
        if not target_positions.empty and "target_weight" in target_positions.columns:
            from src.assembled_core.risk.crowding_detector import compute_hhi
            _tw_dict_crowd = dict(zip(target_positions["symbol"], target_positions["target_weight"].fillna(0.0)))
            _hhi = compute_hhi(_tw_dict_crowd)
            if _hhi > 0.25 and len(_tw_dict_crowd) >= 5:
                _max_w = 0.10
                mask = target_positions["target_weight"].abs() > _max_w
                target_positions.loc[mask, "target_weight"] = _max_w
                if "target_qty" in target_positions.columns:
                    target_positions.loc[mask, "target_qty"] = _max_w * ctx.capital
    except Exception as e:
        log.debug("crowding_detector skipped: %s", e)
    return target_positions


def _sp_apply_crisis_alpha_cap(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """T4.1: Crisis Alpha weight cap."""
    if not (policy or {}).get("intel", {}).get("crisis_alpha", {}).get("enabled", False):
        return target_positions
    try:
        from src.assembled_core.events.crisis_alpha.pipeline import run_crisis_alpha_pipeline
        from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
        _ca_ctx = ctx.meta.get("crisis_alpha_ctx") if hasattr(ctx, "meta") else None
        if _ca_ctx is None:
            _as_of_dt = pd.to_datetime(ctx.as_of, utc=True).to_pydatetime() if getattr(ctx, "as_of", None) is not None else None
            _ca_ctx = CrisisAlphaContext.empty(timestamp_utc=_as_of_dt)
        shadow_only = (policy or {}).get("intel", {}).get("crisis_alpha", {}).get("shadow_only", True)
        ca_result = run_crisis_alpha_pipeline(_ca_ctx, policy=policy, dry_run=shadow_only)
        if not shadow_only and ca_result.get("target_weights") and not target_positions.empty and "target_weight" in target_positions.columns:
            ca_tw = ca_result["target_weights"]
            _cap_series = target_positions["symbol"].astype(str).map(ca_tw)
            mask = _cap_series.notna() & (target_positions["target_weight"] > _cap_series.astype(float))
            n_adjusted = int(mask.sum())
            target_positions.loc[mask, "target_weight"] = _cap_series[mask].astype(float)
            if n_adjusted:
                log.info("[T4.1] crisis_alpha: capped %d positions", n_adjusted)
    except Exception as exc:
        log.warning("[T4.1] crisis_alpha_pipeline failed: %s", exc)
    return target_positions


def _sp_check_rebalance(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    meta: dict,
    log: logging.Logger,
) -> tuple[bool, str]:
    """Step 4.5: Rebalance trigger check."""
    vol_regime_changed = bool(meta.get("vol_targeting", {}).get("regime_changed", False))
    corr_spiked = bool(meta.get("correlation_regime_shift", {}).get("exposure_scale", 1.0) < 1.0)
    dd_pct = meta.get("drawdown_pct")
    current_w: dict[str, float] = {}
    if hasattr(ctx, "current_positions") and ctx.current_positions is not None:
        if isinstance(ctx.current_positions, dict):
            current_w = ctx.current_positions
        elif isinstance(ctx.current_positions, pd.DataFrame) and "symbol" in ctx.current_positions.columns:
            _cp = ctx.current_positions
            _wcol = "weight" if "weight" in _cp.columns else "target_weight"
            if _wcol in _cp.columns:
                current_w.update({str(k): float(v) for k, v in _cp.set_index("symbol")[_wcol].fillna(0.0).items()})
    return should_rebalance(
        ctx, target_positions, current_weights=current_w,
        weight_drift_threshold=float(policy.get("rebalancing", {}).get("weight_drift_threshold", 0.05)),
        vol_regime_change=vol_regime_changed, corr_spike=corr_spiked,
        scheduled=True, drawdown_pct=float(dd_pct) if dd_pct is not None else None,
    )


def _sp_apply_cost_aware(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """Step 4.85: Cost-aware weight shrinkage."""
    try:
        caw_cfg = policy.get("cost_aware_wrapper") or {}
        if caw_cfg.get("enabled", False) and not target_positions.empty:
            from src.assembled_core.portfolio.cost_aware_wrapper import apply_cost_aware_from_policy
            w_col = next((c for c in ("target_weight", "weight", "target_pct") if c in target_positions.columns), None)
            if w_col and "symbol" in target_positions.columns:
                _target_w = {str(k): float(v) for k, v in target_positions.dropna(subset=[w_col]).set_index("symbol")[w_col].items()}
                _curr_w_caw: dict[str, float] = {}
                if ctx.current_positions is not None and not ctx.current_positions.empty and "symbol" in ctx.current_positions.columns and w_col in ctx.current_positions.columns:
                    _curr_w_caw = {str(k): float(v) for k, v in ctx.current_positions.dropna(subset=[w_col]).set_index("symbol")[w_col].items()}
                _adj_w, _caw_reasons = apply_cost_aware_from_policy(_target_w, _curr_w_caw, policy, current_invested_pct=float(sum(abs(v) for v in _target_w.values())))
                if _caw_reasons:
                    target_positions = target_positions.copy()
                    target_positions[w_col] = target_positions["symbol"].map(lambda s: _adj_w.get(str(s), _target_w.get(str(s), 0.0)))
    except Exception as e:
        log.debug("cost_aware_wrapper skipped: %s", e)
    return target_positions


def size_positions(
    signals: pd.DataFrame,
    ctx: TradingContext,
    prices_filtered: pd.DataFrame | None = None,
    prices_with_features: pd.DataFrame | None = None,
    prices_latest: pd.DataFrame | None = None,
    *,
    log: logging.Logger | None = None,
) -> tuple[pd.DataFrame, bool, dict]:
    """Convert signals to target positions with all exposure overlays.

    Returns (target_positions, do_rebal, meta). do_rebal=False means route_orders
    should return empty orders for this bar. meta contains profit_lock, vol_targeting,
    and other overlay diagnostics for result.meta.
    """
    if log is None:
        log = logger

    try:
        policy = load_policy()
    except Exception:
        policy = {}

    prices_for_sizing = prices_filtered if prices_filtered is not None else ctx.prices
    meta: dict = {}

    sizing_cfg = policy.get("position_sizing") or {}
    target_positions = _sp_dispatch_sizing(signals, ctx, prices_for_sizing, sizing_cfg, log)
    meta["sizing_method"] = sizing_cfg.get("method", "default")

    target_positions = _sp_apply_liquidity(target_positions, prices_for_sizing, policy, log)

    final_multiplier = _sp_compute_final_multiplier(ctx, policy, meta, log)
    if abs(final_multiplier - 1.0) > 1e-9 and not target_positions.empty:
        target_positions = apply_exposure_multiplier_to_targets(target_positions, multiplier=final_multiplier, cash_symbol="CASH")

    target_positions = _sp_apply_factor_risk(target_positions, prices_for_sizing, policy, log)
    target_positions = _sp_apply_trailing_stops(target_positions, ctx, prices_filtered, policy, meta, log)
    target_positions = _sp_apply_turnover_gate(target_positions, ctx, prices_for_sizing, prices_latest, policy, log)
    target_positions = _sp_apply_correlation_guard(target_positions, prices_for_sizing, policy, ctx)
    target_positions = _sp_apply_crash_cap(target_positions, policy, meta, str(ctx.as_of) if ctx.as_of else None)
    target_positions = _sp_apply_inverse_etf(target_positions, ctx, policy, meta)
    target_positions = _sp_apply_quantile_asymmetry(target_positions, prices_with_features, policy, log)
    target_positions = _sp_apply_crowding_cap(target_positions, ctx, log)
    target_positions = _sp_apply_crisis_alpha_cap(target_positions, ctx, policy, log)

    do_rebal, rebal_reason = _sp_check_rebalance(target_positions, ctx, policy, meta, log)
    if not do_rebal:
        log.info("REBALANCE SKIPPED: %s — no orders generated", rebal_reason)

    target_positions = _sp_apply_cost_aware(target_positions, ctx, policy, log)

    return target_positions, do_rebal, meta


# ---------------------------------------------------------------------------
# check_risk — Stage 5
# ---------------------------------------------------------------------------


def check_risk(
    orders: pd.DataFrame,
    result: TradingCycleResult,
    ctx: TradingContext,
    *,
    prices_filtered: pd.DataFrame | None = None,
    log: logging.Logger | None = None,
) -> TradingCycleResult:
    """Apply EVT/copula/barbell + risk controls to orders; return updated result.

    Steps kept (modify orders or orders_filtered):
      - QA gate: block all orders if ctx.qa_block_trading
      - EVT tail VaR: scale orders qty if EVT VaR > 2× historical VaR
      - Copula tail dep: scale orders qty if avg_lower_tail_dep > 0.5
      - Barbell: scale orders qty when composite tail risk score > 0.30
      - Step 6: _apply_risk_controls_default (kill switch, position limits)
      - Step 6.35: Parametric VaR exposure gate (clears orders_filtered)
      - Step 6.4: Auto-drawdown kill-switch trigger (activates KS, may clear)
      - Step 6.45: Intraday circuit breaker (clears orders_filtered)
      - Step 6.6: Anti-churn deadzone + min-notional filters
      - Step 6.7: Fat-finger guard (hard notional + qty-multiple cap)
      - Step 6.9: Order lifecycle tracking (audit trail)

    Dropped (meta-only):
      - Step 6.5 scenario engine, Step 6.8 borrow cost, Step 6.85 tx costs,
        Steps 5.5-5.14 (all meta-only), Step 5.14 risk escalation
    """
    if log is None:
        log = logger

    try:
        policy = load_policy()
    except Exception:
        policy = {}

    # Fast path: if risk controls are disabled, skip all steps and pass orders through.
    if not getattr(ctx, "enable_risk_controls", True):
        result.orders = orders
        result.orders_filtered = orders.copy()
        return result

    # QA gate
    if ctx.qa_block_trading:
        log.warning("QA Gate: Trading blocked - %s", ctx.qa_block_reason or "no reason")
        result.orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        result.orders_filtered = result.orders.copy()
        result.meta["qa_block_trading"] = True
        result.meta["qa_block_reason"] = ctx.qa_block_reason
        return result

    # EVT tail VaR
    try:
        prices_for_evt = prices_filtered if prices_filtered is not None else ctx.prices
        if not orders.empty and prices_for_evt is not None and not prices_for_evt.empty and "close" in prices_for_evt.columns:
            from src.assembled_core.risk.evt_tail_var import evt_var
            import numpy as _np_evt
            _pivot_evt = prices_for_evt.pivot_table(index="timestamp" if "timestamp" in prices_for_evt.columns else prices_for_evt.columns[0], columns="symbol" if "symbol" in prices_for_evt.columns else None, values="close")
            _rets_evt = _pivot_evt.pct_change().dropna(how="all")
            if len(_rets_evt) >= 60:
                _port_rets = _rets_evt.mean(axis=1).dropna()
                _losses = (-_port_rets).values
                _hist_var_99 = float(_np_evt.quantile(_losses, 0.99))
                try:
                    _evt_var_99 = evt_var(_losses, alpha=0.99, threshold_pct=0.90)
                except Exception:
                    _evt_var_99 = None
                if _evt_var_99 is not None and _hist_var_99 > 1e-8 and _evt_var_99 > 2.0 * _hist_var_99:
                    orders = orders.copy()
                    orders["qty"] = orders["qty"] * 0.80
                    log.warning("[RISK] EVT VaR %.4f > 2× Hist VaR %.4f — reducing qty by 20%%", _evt_var_99, _hist_var_99)
    except Exception as e:
        log.debug("evt_tail_var skipped: %s", e)

    # Copula tail dependence
    try:
        prices_for_cop = prices_filtered if prices_filtered is not None else ctx.prices
        if not orders.empty and prices_for_cop is not None and not prices_for_cop.empty and "close" in prices_for_cop.columns:
            from src.assembled_core.ml.copula_models import compute_portfolio_tail_risk
            _pivot_cop = prices_for_cop.pivot_table(index="timestamp" if "timestamp" in prices_for_cop.columns else prices_for_cop.columns[0], columns="symbol" if "symbol" in prices_for_cop.columns else None, values="close")
            _rets_cop = _pivot_cop.pct_change().dropna(how="all")
            if len(_rets_cop) >= 60 and 1 < _rets_cop.shape[1] <= 30:
                _cop_metrics = compute_portfolio_tail_risk(_rets_cop)
                if float(_cop_metrics.get("avg_lower_tail_dep", 0.0)) > 0.5:
                    orders = orders.copy()
                    orders["qty"] = orders["qty"] * 0.80
                    log.warning("[RISK] Copula avg_lower_tail_dep > 0.5 — reducing qty by 20%%")
    except Exception as e:
        log.debug("copula_tail_risk skipped: %s", e)

    # Barbell strategy
    try:
        from src.assembled_core.portfolio.barbell_strategy import build_barbell_allocation, compute_tail_risk_score
        _evt_var_meta = result.meta.get("evt_var_99", 0.0) or 0.0
        _hist_var_meta = result.meta.get("hist_var_99", 0.0) or 0.0
        _cop_ltd_meta = float((result.meta.get("copula_tail_risk") or {}).get("avg_lower_tail_dep", 0.0))
        _bb_score, _bb_reasons = compute_tail_risk_score(evt_var_99=float(_evt_var_meta), evt_var_99_historical_avg=float(_hist_var_meta), hmm_crisis_prob=0.0, vix_current=0.0, vix_5d_change=0.0, avg_copula_tail_dep=_cop_ltd_meta)
        if _bb_score > 0.30 and not orders.empty:
            _alpha_scores: dict[str, float] = {}
            if not result.signals.empty and "symbol" in result.signals.columns and "score" in result.signals.columns:
                _alpha_scores = dict(zip(result.signals["symbol"], result.signals["score"].fillna(0.0)))
            _bb_alloc = build_barbell_allocation(tail_risk_score=_bb_score, trigger_reasons=_bb_reasons, alpha_scores=_alpha_scores)
            if _bb_alloc.active:
                orders = orders.copy()
                orders["qty"] = orders["qty"] * _bb_alloc.speculative_weight
                log.warning("[RISK] Barbell ACTIVATED: score=%.3f spec_weight=%.2f", _bb_score, _bb_alloc.speculative_weight)
    except Exception as e:
        log.debug("barbell_strategy skipped: %s", e)

    result.orders = orders
    _n_orders_in = len(orders) if orders is not None else 0
    _rej_counts: dict[str, int] = {}

    # Step 6: risk controls default
    try:
        result.orders_filtered = _apply_risk_controls_default(ctx, orders)
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in risk_controls: {e}"
        return result
    _n_after_6 = len(result.orders_filtered)
    if _n_orders_in > _n_after_6:
        _rej_counts["risk_controls"] = _n_orders_in - _n_after_6

    # Step 6.35: VaR gate
    try:
        var_decision = _evaluate_var_gate(ctx, result, policy)
        if var_decision is not None:
            result.meta["var_gate"] = var_decision
            log.warning("[RISK] VaR gate breach: %s", var_decision.get("reason", ""))
            _rej_counts["var_gate"] = len(result.orders_filtered)
            result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        log.warning("[RISK] var_gate evaluation raised — gate no-op: %s", e)
        result.meta["var_gate"] = {"status": "error", "error": str(e)}

    # Step 6.4: Auto-DD kill switch
    try:
        dd_decision = _evaluate_auto_dd_kill_switch(ctx, result, policy)
        if dd_decision is not None:
            from src.assembled_core.execution.kill_switch import activate_kill_switch, is_kill_switch_engaged
            activate_kill_switch(throttle_pct=dd_decision["throttle_allowed_pct"], reason=dd_decision["reason"], actor="trading_cycle_v2_auto_dd")
            result.meta["auto_dd_kill_switch"] = dd_decision
            if dd_decision["level"] == "kill":
                _rej_counts["auto_dd_kill_switch"] = len(result.orders_filtered)
                result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        log.warning("[RISK] auto_dd_kill_switch raised — gate no-op: %s", e)
        result.meta["auto_dd_kill_switch"] = {"status": "error", "error": str(e)}

    # Step 6.45: Circuit breaker
    try:
        cb_decision = _evaluate_circuit_breaker(ctx, result, policy)
        if cb_decision is not None:
            result.meta["circuit_breaker"] = cb_decision
            _rej_counts["circuit_breaker"] = len(result.orders_filtered)
            result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        log.warning("[RISK] circuit_breaker raised — gate no-op: %s", e)
        result.meta["circuit_breaker"] = {"status": "error", "error": str(e)}

    # Step 6.6: Anti-churn deadzone + min-notional
    try:
        anti_churn_cfg = policy.get("anti_churn") or {}
        if not result.orders_filtered.empty:
            if anti_churn_cfg.get("deadzone_enabled", False):
                from src.assembled_core.paper.deadzone_rebalance import filter_deadzone_orders
                _dz_pos = ctx.current_positions[["symbol", "qty"]].copy() if ctx.current_positions is not None and not ctx.current_positions.empty and "qty" in ctx.current_positions.columns else None
                result.orders_filtered, _dz_meta = filter_deadzone_orders(result.orders_filtered, _dz_pos, deadzone_pct=float(anti_churn_cfg.get("deadzone_pct", 0.05)))
                result.meta["deadzone_rebalance"] = _dz_meta
            if anti_churn_cfg.get("rebalance_filter_enabled", False) and not result.orders_filtered.empty:
                from src.assembled_core.paper.rebalance_filter import filter_small_rebalances
                result.orders_filtered, _rf_meta = filter_small_rebalances(result.orders_filtered, min_notional=float(anti_churn_cfg.get("min_notional", 500.0)), prices=prices_filtered if prices_filtered is not None else ctx.prices)
                result.meta["rebalance_filter"] = _rf_meta
    except Exception as e:
        log.debug("anti_churn filters skipped: %s", e)

    # Step 6.7: Fat-finger guard
    try:
        ffg_cfg = policy.get("fat_finger_guard") or {}
        if ffg_cfg.get("enabled", False) and not result.orders_filtered.empty:
            from src.assembled_core.execution.fat_finger_guard import apply_fat_finger_guard_from_policy
            _ffg_orders, _ffg_reasons = apply_fat_finger_guard_from_policy(result.orders_filtered, policy)
            n_rejected = len(result.orders_filtered) - len(_ffg_orders)
            result.orders_filtered = _ffg_orders
            if n_rejected:
                log.warning("[FAT-FINGER] Rejected %d orders: %s", n_rejected, _ffg_reasons[:3])
                _rej_counts["fat_finger"] = n_rejected
    except Exception as e:
        log.debug("fat_finger_guard skipped: %s", e)

    # Step 6.9: Order lifecycle tracking
    try:
        if not result.orders_filtered.empty:
            from src.assembled_core.execution.order_lifecycle import OrderLifecycleTracker, OrderState
            _olt = OrderLifecycleTracker()
            for _, _ord_row in result.orders_filtered.iterrows():
                _oid = _olt.create(symbol=str(_ord_row.get("symbol", "")), side=str(_ord_row.get("side", "buy")), quantity=float(_ord_row.get("qty", 0)), price=float(_ord_row.get("price", 0)) or None, source="trading_cycle_v2")
                _olt.transition(_oid, OrderState.VALIDATED)
                _olt.transition(_oid, OrderState.SUBMITTED)
            result.meta["order_lifecycle"] = {"n_orders_tracked": len(result.orders_filtered), "state": "SUBMITTED"}
    except Exception as e:
        log.debug("order_lifecycle tracking skipped: %s", e)

    result.meta["rejection_counts"] = _rej_counts
    return result


# ---------------------------------------------------------------------------
# route_orders — Stage 6
# ---------------------------------------------------------------------------


def route_orders(
    targets: pd.DataFrame,
    ctx: TradingContext,
    *,
    prices_filtered: pd.DataFrame | None = None,
    prices_with_features: pd.DataFrame | None = None,
    prices_latest: pd.DataFrame | None = None,
    do_rebal: bool = True,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Generate orders from approved target positions.

    Steps kept:
      - Step 5: _generate_orders_default + price enrichment from prices_latest
      - Phase 17.8: Pre-Trade Impact estimate (modifies order qty)
      - Phase 17.9: Group-Exposure caps (modifies orders)

    Dropped (meta-only):
      - Phase 17.85 TWAP (only writes meta, does not replace orders)
      - Steps 5.5-5.14 (all meta-only)
    """
    if log is None:
        log = logger

    _empty = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    if not do_rebal or targets is None or targets.empty:
        return _empty

    try:
        policy = load_policy()
    except Exception:
        policy = {}

    # Step 5: Generate orders
    try:
        orders = _generate_orders_default(ctx, targets)

        # Enrich with latest prices
        pwf = prices_with_features if prices_with_features is not None else pd.DataFrame()
        if not orders.empty and not pwf.empty:
            if prices_latest is not None and "close" in prices_latest.columns:
                latest_prices = prices_latest[["symbol", "close"]].rename(columns={"close": "price"})
            elif "close" in pwf.columns:
                latest_prices = pwf.groupby("symbol", group_keys=False)["close"].last().reset_index().rename(columns={"close": "price"})
            else:
                latest_prices = None
            if latest_prices is not None:
                orders = orders.merge(latest_prices, on="symbol", how="left", suffixes=("", "_latest"))
                if "price_latest" in orders.columns:
                    orders["price"] = orders["price_latest"].fillna(orders["price"])
                    orders = orders.drop(columns=["price_latest"])
    except Exception as e:
        log.warning("order generation failed: %s", e)
        return _empty

    # Phase 17.8: Pre-trade impact
    try:
        impact_cfg = (policy.get("execution", {}) or {}).get("pre_trade_impact", {}) or {}
        if impact_cfg.get("enabled", False) and not orders.empty:
            orders, _impact_meta = _apply_pre_trade_impact(orders, prices_filtered, impact_cfg)
    except Exception as e:
        log.debug("pre_trade_impact skipped: %s", e)

    # Phase 17.9: Group-exposure caps
    try:
        group_cfg = (policy.get("risk", {}) or {}).get("group_limits", {}) or {}
        if group_cfg.get("enabled", False) and not orders.empty:
            sec_meta = None
            try:
                from src.assembled_core.data.security_master import load_security_master
                sec_meta = load_security_master(group_cfg.get("security_master_path") or None)
            except Exception:
                pass
            if sec_meta is not None:
                orders, _grp_meta = _apply_group_exposure_caps(orders, sec_meta, group_cfg)
    except Exception as e:
        log.debug("group_exposures skipped: %s", e)

    if not orders.empty and "qty" in orders.columns:
        orders = orders.copy()
        orders["qty"] = orders["qty"].abs()

    return orders


# ---------------------------------------------------------------------------
# book_fills — Stage 7
# ---------------------------------------------------------------------------


def book_fills(
    result: TradingCycleResult,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> TradingCycleResult:
    """Write outputs and return the final TradingCycleResult.

    Steps kept (write artifacts read by monitoring / downstream pipelines):
      - Step 7: write_outputs (safe_csv / equity_curve / state)
      - Step 7.6: write_run_kpis
      - Step 7.62: write_run_manifest
      - Step 7.63: append_run_index
      - Step 7.66: trade journal
      - Step 7.68: heartbeat
      - Phase 9: signal diagnostics (write signal_health.json)
      - Phase 11: KPI export (Prometheus metrics)

    Dropped (meta-only, 3-criteria rule):
      - Steps 7.5, 7.64, 7.65, 7.67, 7.69-7.71, 7.8, 7.9, 8.x,
        Phase 10 Monte Carlo, tail_hedge shadow, attribution shadow,
        portfolio_execution shadow, almgren_chriss shadow
    """
    if log is None:
        log = logger

    try:
        policy = load_policy()
    except Exception:
        policy = {}

    # Ensure orders_filtered exists
    if result.orders_filtered is None:
        result.orders_filtered = result.orders.copy() if result.orders is not None else pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])

    # A8: Apply cost annotation for backtest/paper modes
    if ctx.mode in ("backtest", "paper") and result.orders_filtered is not None and not result.orders_filtered.empty:
        try:
            from src.assembled_core.execution.transaction_costs import (
                add_cost_columns_to_trades,
                CommissionModel,
            )
            from src.assembled_core.costs import get_default_cost_model
            cost_model = get_default_cost_model()
            commission_model = CommissionModel(commission_bps=cost_model.commission_bps)
            prices = getattr(ctx, "prices", None)
            result.orders_filtered = add_cost_columns_to_trades(
                result.orders_filtered,
                commission_model=commission_model,
                prices=prices if prices is not None else pd.DataFrame(),
            )
        except Exception as _cost_err:
            log.debug("[book_fills] cost annotation skipped (mode=%s): %s", ctx.mode, _cost_err)

    # A8b: Derive total_cost_bps for Phase 11 slippage histogram
    try:
        if result.orders_filtered is not None and not result.orders_filtered.empty:
            if "total_cost_bps" not in result.orders_filtered.columns:
                if "total_cost_cash" in result.orders_filtered.columns:
                    _notional = (result.orders_filtered["qty"].abs() * result.orders_filtered["price"].abs())
                    result.orders_filtered["total_cost_bps"] = 0.0
                    _mask = _notional > 0
                    result.orders_filtered.loc[_mask, "total_cost_bps"] = (
                        result.orders_filtered.loc[_mask, "total_cost_cash"].abs() / _notional[_mask] * 10_000.0
                    )
                elif "expected_impact_bps" in result.orders_filtered.columns:
                    result.orders_filtered["total_cost_bps"] = result.orders_filtered["expected_impact_bps"]
    except Exception as _bps_err:
        log.debug("[book_fills] total_cost_bps derivation skipped: %s", _bps_err)

    # A8c: Drift monitoring (policy: drift_monitor.enabled + reference_path)
    try:
        dm_cfg = policy.get("drift_monitor") or {}
        if dm_cfg.get("enabled", False):
            ref_path = dm_cfg.get("reference_path")
            current_features = result.prices_with_features
            if ref_path and current_features is not None and not current_features.empty:
                from src.assembled_core.ops.drift_monitor import DriftMonitor
                _ref_df = (
                    pd.read_parquet(ref_path) if str(ref_path).endswith(".parquet")
                    else pd.read_csv(ref_path)
                )
                _monitor = DriftMonitor(
                    reference=_ref_df,
                    output_dir=ctx.output_dir if ctx.write_outputs else None,
                    psi_warn_threshold=float(dm_cfg.get("psi_warn", 0.25)),
                    psi_pause_threshold=float(dm_cfg.get("psi_pause", 0.35)),
                )
                _drift_report = _monitor.check_drift(
                    current=current_features,
                    report_date=ctx.as_of,
                )
                result.meta["drift_monitor"] = {
                    "max_psi": float(_drift_report.max_psi),
                    "action": _drift_report.action,
                    "drifted_features": _drift_report.drifted_features,
                    "n_drifted": len(_drift_report.drifted_features),
                }
                log.info(
                    "[DRIFT] max_psi=%.3f action=%s drifted=%d",
                    _drift_report.max_psi, _drift_report.action, len(_drift_report.drifted_features),
                )
    except Exception as _dm_err:
        log.debug("[book_fills] drift_monitor skipped: %s", _dm_err)

    # Step 7: Write outputs
    try:
        if ctx.write_outputs:
            if ctx.output_format == "safe_csv":
                from src.assembled_core.execution.safe_bridge import write_safe_orders_csv
                ctx.output_dir.mkdir(parents=True, exist_ok=True)
                out_path = write_safe_orders_csv(result.orders_filtered, output_path=ctx.output_dir / "orders_latest.csv")
                result.output_paths = {"safe_csv": out_path}
            else:
                result.output_paths = {}
    except Exception as e:
        result.status = "error"
        result.error_message = f"Error in write_outputs: {e}"
        return result

    # Step 7.6: KPI artifact
    try:
        if ctx.write_outputs:
            from src.assembled_core.ops.kpi_artifacts import write_run_kpis
            write_run_kpis(output_dir=ctx.output_dir, ctx=ctx, result=result, policy=policy, mode=ctx.execution_mode)
    except Exception as e:
        log.debug("[KPI] write_run_kpis skipped: %s", e)

    # Step 7.62: Run manifest
    try:
        if ctx.write_outputs and ctx.as_of is not None:
            from src.assembled_core.ops.run_manifest import write_run_manifest
            write_run_manifest(run_id=str(ctx.as_of.date()), date=str(ctx.as_of.date()), started_at_utc=ctx.as_of.isoformat(), status="success", metrics={"n_orders": len(result.orders_filtered), "n_signals": len(result.signals), "execution_mode": ctx.execution_mode}, manifests_dir=ctx.output_dir / "manifests")
    except Exception as e:
        log.debug("[MANIFEST] run_manifest skipped: %s", e)

    # Step 7.63: Run index
    try:
        if ctx.write_outputs and ctx.as_of is not None:
            from src.assembled_core.ops.run_index import append_run_index
            from src.assembled_core.ops.run_manifest import compute_config_hash
            append_run_index(run_id=str(ctx.as_of.date()), date=str(ctx.as_of.date()), status="success", metrics={"final_equity": float(getattr(ctx, "current_equity", ctx.equity)), "n_fills": len(result.orders_filtered)}, git_sha=result.meta.get("git_sha", ""), config_hash=compute_config_hash(policy) if policy else "", manifest_path=ctx.output_dir / "manifests" / str(ctx.as_of.date()) / "manifest.latest.json", index_path=ctx.output_dir / "manifests" / "index.csv")
    except Exception as e:
        log.debug("[RUN-INDEX] run_index skipped: %s", e)

    # Step 7.66: Trade journal
    try:
        if ctx.write_outputs and not result.orders_filtered.empty and ctx.as_of is not None:
            from src.assembled_core.ops.trade_journal import append_trade_journal_entries
            _of = result.orders_filtered
            _qty_col = "quantity" if "quantity" in _of.columns else "qty"
            _price_col = "price" if "price" in _of.columns else "limit_price"
            _tj_fills = [{"symbol": str(r["symbol"]), "side": str(r["side"]), "qty": float(r[_qty_col] if pd.notna(r[_qty_col]) else 0), "price": float(r[_price_col] if pd.notna(r[_price_col]) else 0)} for r in _of[["symbol", "side", _qty_col, _price_col]].itertuples(index=False)]
            append_trade_journal_entries(_tj_fills, signal_context={"regime": result.meta.get("regime", {}).get("regime", ""), "execution_mode": ctx.execution_mode}, run_id=str(ctx.as_of.date()), journal_path=ctx.output_dir / "trade_journal.jsonl")
    except Exception as e:
        log.debug("[TRADE-JOURNAL] trade_journal skipped: %s", e)

    # Step 7.68: Heartbeat
    try:
        from src.assembled_core.ops.heartbeat import write_heartbeat
        _hb_path = ctx.output_dir / "state" / "heartbeat.json"
        write_heartbeat(path=_hb_path, status="ok", details={"cycle_date": str(ctx.as_of.date()) if ctx.as_of else "", "n_orders": len(result.orders_filtered), "execution_mode": str(ctx.execution_mode)})
        result.meta["heartbeat"] = {"status": "ok", "path": str(_hb_path)}
    except Exception as e:
        log.debug("[HEARTBEAT] heartbeat skipped: %s", e)

    # Phase 9: Signal diagnostics
    try:
        sd_cfg = (policy.get("signal_generation") or {}).get("signal_diagnostics") or {}
        if sd_cfg.get("enabled", False) and result.prices_with_features is not None and not result.prices_with_features.empty:
            from src.assembled_core.signals.signal_diagnostics import compute_signal_health, generate_signal_health_alerts, save_signal_health_artifact
            fwd_col = sd_cfg.get("forward_returns_col", "return_1d")
            if fwd_col in result.prices_with_features.columns and "timestamp" in result.prices_with_features.columns:
                factor_cols = [c for c in result.prices_with_features.columns if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume", fwd_col} and result.prices_with_features[c].dtype in ("float64", "float32")][:20]
                if factor_cols:
                    health_df = compute_signal_health(result.prices_with_features, forward_returns_col=fwd_col, factor_cols=factor_cols)
                    alerts = generate_signal_health_alerts(health_df, ic_alert_threshold=float(sd_cfg.get("ic_alert_threshold", 0.0)))
                    save_signal_health_artifact(health_df, alerts, output_dir=str(ctx.output_dir / "diagnostics") if ctx.write_outputs else sd_cfg.get("output_dir", "output/diagnostics"), run_date=ctx.as_of.strftime("%Y-%m-%d") if ctx.as_of else None)
    except Exception as e:
        log.debug("[SIGNAL-DIAG] signal_diagnostics skipped: %s", e)

    # Phase 11: KPI export (Prometheus)
    try:
        kpi_cfg = policy.get("kpi_export") or {}
        if kpi_cfg.get("enabled", False):
            from src.assembled_core.ops.metrics_exporter import export_metrics, slippage_histogram
            kpi_metrics: dict[str, float] = {
                "assembled_orders_generated_total": float(len(result.orders_filtered)),
                "assembled_targets_count": float(len(result.target_positions)),
                "assembled_signals_count": float(len(result.signals)),
            }
            tb_meta = result.meta.get("turnover_budget") or {}
            if "estimated_turnover" in tb_meta and tb_meta["estimated_turnover"] != float("inf"):
                kpi_metrics["assembled_turnover_estimated"] = float(tb_meta["estimated_turnover"])
            vt_meta = result.meta.get("vol_targeting") or {}
            if "realized_vol" in vt_meta:
                kpi_metrics["assembled_realized_vol"] = float(vt_meta["realized_vol"])
            # Slippage histogram: use cost-annotated orders if total_cost_bps column present
            kpi_histograms = None
            if result.orders_filtered is not None and "total_cost_bps" in result.orders_filtered.columns:
                _slip_obs = result.orders_filtered["total_cost_bps"].dropna().tolist()
                if _slip_obs:
                    kpi_histograms = {"assembled_slippage_bps": slippage_histogram(_slip_obs)}
            # Kill-switch state gauge (1 = engaged, 0 = inactive)
            try:
                from src.assembled_core.execution.kill_switch import is_kill_switch_engaged
                kpi_metrics["assembled_kill_switch_engaged"] = 1.0 if is_kill_switch_engaged() else 0.0
            except Exception:
                pass
            # Drift-PSI gauge from drift_monitor meta (if present)
            _drift_meta = result.meta.get("drift_monitor") or {}
            if "max_psi" in _drift_meta:
                kpi_metrics["assembled_drift_max_psi"] = float(_drift_meta["max_psi"])
            # Rejection counters (per reason)
            _rej_meta = result.meta.get("rejection_counts") or {}
            for _reason, _cnt in _rej_meta.items():
                _safe = str(_reason).replace("-", "_").replace(" ", "_").upper()
                kpi_metrics[f"assembled_rejections_{_safe}_total"] = float(_cnt)
            metrics_dir = ctx.output_dir / "metrics" if ctx.write_outputs else None
            export_metrics(kpi_metrics, histograms=kpi_histograms, labels={"strategy": ctx.strategy_name or "unknown", "mode": ctx.mode}, path=metrics_dir / "assembled.prom" if metrics_dir else None)
    except Exception as e:
        log.debug("[KPI] kpi_export skipped: %s", e)

    result.status = "success"
    return result


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

    try:
        prices, prices_latest = ingest_data(ctx, log=log)
        result.prices_filtered = prices
        result.prices_latest = prices_latest

        features, pl_update = build_features(prices, ctx, log=log)
        result.prices_with_features = features
        # Backtest snapshot mode can override prices_filtered/prices_latest
        if pl_update is not None:
            result.prices_latest = pl_update
            result.prices_filtered = pl_update

        signals = generate_signals(features, ctx, log=log)
        result.signals = signals

        targets, do_rebal, sizing_meta = size_positions(
            signals, ctx,
            prices_filtered=result.prices_filtered,
            prices_with_features=result.prices_with_features,
            prices_latest=result.prices_latest,
            log=log,
        )
        result.target_positions = targets
        result.meta.update(sizing_meta)

        orders = route_orders(
            targets, ctx,
            prices_filtered=result.prices_filtered,
            prices_with_features=result.prices_with_features,
            prices_latest=result.prices_latest,
            do_rebal=do_rebal,
            log=log,
        )
        result.orders = orders

        result = check_risk(orders, result, ctx, prices_filtered=result.prices_filtered, log=log)

        result = book_fills(result, ctx, log=log)

    except ValueError as exc:
        result.status = "error"
        result.error_message = str(exc)
    except Exception as exc:
        result.status = "error"
        result.error_message = f"Unexpected error: {exc}"
        log.exception("trading_cycle_v2: unexpected error in run_trading_cycle")
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
