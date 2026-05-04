"""_tc_features — build_features() extracted from trading_cycle_v2."""

from __future__ import annotations

import logging

import pandas as pd
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    _build_features_default,
)

logger = logging.getLogger(__name__)


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

    policy = getattr(ctx, "_policy_cache", None)
    if policy is None:
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
            precomputed["timestamp"] = pd.to_datetime(
                precomputed["timestamp"], utc=True
            )

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
                    precomputed_filtered.groupby(
                        "symbol", group_keys=False, dropna=False
                    )
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

    log.debug("Features: %d columns (was %d)", len(pwf.columns), len(prices.columns))

    # --- Step 2.5 HMM: D3 regime detection → sets ctx.regime_state ---
    try:
        regime_cfg = policy.get("regime_detection", {})
        if (
            regime_cfg.get("method") == "hmm"
            and getattr(ctx, "regime_state", None) is None
        ):
            from src.assembled_core.risk.regime_models import build_regime_state_hmm

            prices_for_hmm = (
                prices if prices is not None and not prices.empty else ctx.prices
            )
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
    # Skip in backtest mode when precomputed features are already present — re-running
    # build_core_ta_factors on a short slice would overwrite valid precomputed factor
    # values (returns_12m, momentum_12m_excl_1m, trend_strength_50, rv_20) with NaN.
    _using_precomputed = (
        ctx.mode == "backtest"
        and ctx.precomputed_prices_with_features is not None
        and not ctx.precomputed_prices_with_features.empty
    )
    try:
        enh_cfg = (policy.get("features") or {}).get("enhanced_factors") or {}
        if enh_cfg.get("enabled", False) and not pwf.empty and not _using_precomputed:
            if enh_cfg.get("ta_factors_core", True):
                from src.assembled_core.features.ta_factors_core import (
                    build_core_ta_factors,
                )

                pwf = build_core_ta_factors(
                    pwf,
                    price_col="close",
                    group_col="symbol",
                    timestamp_col="timestamp",
                )
            if enh_cfg.get("cross_sectional_rank", True):
                from src.assembled_core.features.cross_sectional import (
                    rank_cross_sectional,
                )

                rank_cols = [
                    c
                    for c in enh_cfg.get(
                        "rank_cols",
                        [
                            "trend_ema_spread",
                            "mom_rsi_centered",
                            "mom_12_1",
                            "low_vol_rank",
                            "quality_score",
                            "trend_strength_20",
                            "trend_strength_50",
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
                _syms_cap = set(pwf["symbol"].unique()[:50])
                _pwf_sub = pwf[pwf["symbol"].isin(_syms_cap)]
                if "timestamp" in _pwf_sub.columns:
                    _pwf_sub = _pwf_sub.sort_values("timestamp")
                _has_volume = "volume" in _pwf_sub.columns
                for _sym, _grp in _pwf_sub.groupby("symbol", sort=False):
                    if len(_grp) < _beh_min_rows:
                        continue
                    _bp = _grp["close"].reset_index(drop=True)
                    _bv = (
                        _grp["volume"].reset_index(drop=True)
                        if _has_volume
                        else pd.Series(1.0, index=range(len(_grp)))
                    )
                    _br = _bp.pct_change(fill_method=None).fillna(0)
                    try:
                        _bc = compute_behavioral_composite(_bp, _bv, _br)
                        _beh_scores[str(_sym)] = (
                            float(_bc.iloc[-1]) if len(_bc) > 0 else 0.0
                        )
                    except Exception as _exc:
                        logger.debug(
                            "[behavioral_composite] %s skipped: %s", _sym, _exc
                        )
                if _beh_scores:
                    pwf = pwf.copy()
                    pwf["behavioral_composite"] = pwf["symbol"].map(_beh_scores)
    except Exception as e:
        log.debug("[BEHAVIORAL] behavioral_features skipped: %s", e)

    # --- Step 2.6: Seasonal features (zero look-ahead calendar columns) ---
    try:
        seas_cfg = (policy.get("features") or {}).get("seasonal_features") or {}
        if (
            seas_cfg.get("enabled", False)
            and not pwf.empty
            and "timestamp" in pwf.columns
        ):
            from src.assembled_core.features.seasonal_features import (
                build_seasonal_features,
            )

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
                        _mr_df[_keys + _mr_cols],
                        on=_keys,
                        how="left",
                        suffixes=("", "_mrf"),
                    )
                    _null_frac = pwf[_mr_cols].isna().mean().max() if _mr_cols else 0.0
                    if _null_frac > 0.5:
                        log.warning(
                            "[MR-FACTORS] %.0f%% NaN after merge — possible key misalignment",
                            _null_frac * 100,
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
                from src.assembled_core.features.weekly_alignment import (
                    add_weekly_alignment,
                )

                _trend_col = next(
                    (
                        c
                        for c in (
                            "trend_strength_50",
                            "momentum_12m_excl_1m",
                            "trend_strength_200",
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
                from src.assembled_core.features.fractional_diff import (
                    apply_ffd_to_panel,
                )

                pwf = apply_ffd_to_panel(
                    pwf,
                    price_cols=["close"],
                    d=float(ffd_cfg.get("d", 0.4)),
                )
    except Exception as e:
        log.debug("[FFD] fractional_diff skipped: %s", e)

    # --- Step 2.15: Order-book imbalance features (optional, requires L2 snapshot data) ---
    try:
        ob_cfg = (policy.get("features") or {}).get("order_book_imbalance") or {}
        if ob_cfg.get("enabled", False) and not pwf.empty:
            snapshots = getattr(ctx, "order_book_snapshots", None)
            if snapshots:
                from src.assembled_core.features.order_book_imbalance import (
                    rolling_imbalance_signal,
                )

                ob_signals = rolling_imbalance_signal(
                    snapshots,
                    lookback=int(ob_cfg.get("lookback", 10)),
                )
                if ob_signals:
                    import pandas as _pd

                    ob_df = _pd.DataFrame(
                        [
                            {
                                "symbol": sym,
                                "ob_imbalance": sig.l1_imbalance,
                                "ob_vw_imbalance": sig.vw_imbalance,
                            }
                            for sym, sig in ob_signals.items()
                        ]
                    )
                    if not ob_df.empty and "symbol" in pwf.columns:
                        pwf = pwf.merge(ob_df, on="symbol", how="left")
    except Exception as e:
        log.debug("[OB-IMBALANCE] order_book_imbalance skipped: %s", e)

    # --- Step 2.16: News features (EWM sentiment, event count, velocity, confidence) ---
    # Requires ctx.news_events (DataFrame with event_date/symbol/direction/confidence).
    # Silently skipped when no news events are available — no degradation for backtests
    # that don't feed news data.
    try:
        nf_cfg = (policy.get("features") or {}).get("news_features") or {}
        if nf_cfg.get("enabled", False) and not pwf.empty:
            _news_events = getattr(ctx, "news_events", None)
            if _news_events is not None and not _news_events.empty:
                from src.assembled_core.features.news_features import add_news_features

                _ts_col = "timestamp" if "timestamp" in pwf.columns else None
                _prices_dates = pwf[_ts_col].unique() if _ts_col else None
                pwf = add_news_features(
                    prices=pwf,
                    events=_news_events,
                    short_window=int(nf_cfg.get("short_window", 7)),
                    long_window=int(nf_cfg.get("long_window", 30)),
                )
                _nf_cols = [c for c in pwf.columns if c.startswith("news_")]
                log.debug(
                    "[NEWS-FEATURES] added %d news columns: %s", len(_nf_cols), _nf_cols
                )
            else:
                log.debug("[NEWS-FEATURES] enabled but no ctx.news_events — skipped")
    except Exception as e:
        log.debug("[NEWS-FEATURES] news_features skipped: %s", e)

    return pwf, prices_latest_update
