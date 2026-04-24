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
from src.assembled_core.pipeline.trading_cycle import (
    TradingContext,
    TradingCycleResult,
    _build_features_default,
    _evaluate_circuit_breaker_daily,
    _filter_prices_for_as_of,
)
from src.assembled_core.risk.market_stress import compute_market_stress
from src.assembled_core.risk.state_machine import (
    compute_next_state,
    load_risk_state,
    save_risk_state,
)

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

            _bundle_path = _pl.Path(mf_cfg.get("bundle_path", "config/factor_bundles/macro_world_etfs_core_bundle.yaml"))
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

    return signals


# ---------------------------------------------------------------------------
# size_positions — Stage 4 (stub)
# ---------------------------------------------------------------------------


def size_positions(
    signals: pd.DataFrame,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Convert signals to target positions with vol-targeting and real risk overlays.

    Real steps to fill in during Day 4:
      - Step 4: position_sizing_fn (caller-provided)
      - Phase 11.5: Trailing stops (regime-adaptive ATR)
      - Step 4.5: Rebalancing trigger check (skip order generation if no trigger)
      - Step 4.85: Cost-aware weight shrinkage
      - Step 4.9: Long-short balance enforcement

    Observability-only steps to DROP:
      - Steps 4.86, 4.87, 4.88, 4.93, 4.94 (all meta-only)
      - Step 4.9: ML training snapshot (meta-only)

    Returns target_positions DataFrame (columns: symbol, target_weight, target_qty).
    """
    raise NotImplementedError("size_positions — stub, not yet filled in")


# ---------------------------------------------------------------------------
# check_risk — Stage 5 (stub)
# ---------------------------------------------------------------------------


def check_risk(
    targets: pd.DataFrame,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Apply pre-trade risk controls and return approved positions.

    Real steps to fill in during Day 5:
      - Step 6: filter_orders_with_risk_controls (kill switch, position limits)
      - Step 6.35: Parametric VaR exposure gate
      - Step 6.4: Auto-drawdown kill-switch trigger
      - Step 6.45: Intraday circuit breaker
      - Step 6.5: Scenario engine stress tests
      - Step 6.6: Anti-churn order filters (deadzone + min-notional)
      - Step 6.7: Fat-finger guard (hard notional + qty-multiple cap)
      - Step 6.9: Order lifecycle tracking (audit trail)
      - QA Gate: block orders if ctx.qa_block_trading

    Observability-only steps to DROP:
      - Step 6.8 (borrow cost meta-only snapshot)

    Returns filtered_targets DataFrame (same schema as target_positions).
    """
    raise NotImplementedError("check_risk — stub, not yet filled in")


# ---------------------------------------------------------------------------
# route_orders — Stage 6 (stub)
# ---------------------------------------------------------------------------


def route_orders(
    checked: pd.DataFrame,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Generate orders from approved target positions.

    Real steps to fill in during Day 5:
      - Step 5: generate_orders_from_targets + align_current_and_target
      - Phase 17.8: Pre-Trade Impact estimate (Almgren-Chriss)
      - Phase 17.85: Optional TWAP order slicing
      - Phase 17.9: Group-Exposure caps (sector/region/currency)

    Observability-only steps to DROP:
      - Steps 5.5-5.14 (portfolio risk metrics, tail dependence, HRP shadow,
        systemic risk, param stability, TCA, execution cost, risk escalation — all meta-only)

    Returns orders DataFrame (columns: timestamp, symbol, side, qty, price).
    """
    raise NotImplementedError("route_orders — stub, not yet filled in")


# ---------------------------------------------------------------------------
# book_fills — Stage 7 (stub)
# ---------------------------------------------------------------------------


def book_fills(
    orders: pd.DataFrame,
    result: TradingCycleResult,
    ctx: TradingContext,
    *,
    log: logging.Logger | None = None,
) -> TradingCycleResult:
    """Write outputs and return the final TradingCycleResult.

    Real steps to fill in during Day 5:
      - Step 7: write_outputs (safe_csv, equity_curve, state)
      - Step 7.6: Write run KPIs artifact
      - Step 7.62: Write run manifest
      - Step 7.63: Append run index CSV
      - Step 7.66: Trade journal (policy-gated on write_outputs)
      - Step 7.68: Heartbeat (write cycle completion heartbeat)
      - Phase 9: Signal diagnostics (write signal_health.json)
      - Phase 11: KPI export

    Observability-only steps to DROP (all Step 7.5, 7.64, 7.65, 7.67, 7.69-7.71,
    7.8, 7.9, Steps 8.x — none of these affect trading decisions):

    Returns a fully-populated TradingCycleResult.
    """
    raise NotImplementedError("book_fills — stub, not yet filled in")


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

        targets = size_positions(signals, ctx, log=log)
        result.target_positions = targets

        checked = check_risk(targets, ctx, log=log)

        orders = route_orders(checked, ctx, log=log)
        result.orders = orders

        result = book_fills(orders, result, ctx, log=log)

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
