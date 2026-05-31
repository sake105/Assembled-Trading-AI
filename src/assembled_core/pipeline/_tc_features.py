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

# Warn-once registry for bundle-critical feature-build failures. Without dedup
# a persistent error in any of these except blocks emits one WARN per bar in
# backtests (1260+/5y) and buries the real signal. Same E-018 pattern as
# multifactor_v2._GEO_RISK_ZERO_FILL_WARNED. Reset on process restart.
#
# Bundle-critical = features that strategy bundles reference by name (rv_20,
# trend_strength_*, behavioral_composite, news/macro panel cols, HMM regime).
# Optional = decorative columns (seasonal, mean-reversion, interaction, ffd,
# order-book-imbalance) where silent skip is acceptable noise control.
_FEATURE_BUILD_WARN_KEYS: set[tuple[str, str]] = set()
_FEATURE_ENH_WARN_KEYS = _FEATURE_BUILD_WARN_KEYS  # backwards-compat alias

# 200 chars per exception message keeps near-identical errors that diverge late
# in the string still distinguishable (Stage-1 F-1). 1024 keys is well past any
# realistic per-process exception vocabulary (~12-20 prefixes × typical
# distinct messages) — cap prevents monotonic growth in long-running daemons.
_WARN_KEY_MAX_CHARS = 200
_WARN_REGISTRY_MAX_KEYS = 1024


def _warn_once_feature_skip(
    prefix: str,
    exc: BaseException,
    log_obj: logging.Logger | None = None,
) -> None:
    """Emit a WARN on first occurrence per (prefix, exc-signature) per process,
    DEBUG on subsequent repeats. Caller is responsible for keeping the except
    graceful (no raise) — this only handles the log level. ``log_obj`` lets
    callers route through ctx-specific loggers; falls back to module logger.

    Registry is bounded at _WARN_REGISTRY_MAX_KEYS entries; once full, further
    distinct keys are still logged at WARN (we don't silently demote) but are
    not added to the registry — net effect: more WARN noise instead of stale
    state. This is the explicit failure mode for the edge case."""
    lg = log_obj or logger
    key = (prefix, f"{type(exc).__name__}:{str(exc)[:_WARN_KEY_MAX_CHARS]}")
    if key not in _FEATURE_BUILD_WARN_KEYS:
        if len(_FEATURE_BUILD_WARN_KEYS) < _WARN_REGISTRY_MAX_KEYS:
            _FEATURE_BUILD_WARN_KEYS.add(key)
        lg.warning("%s skipped (first occurrence, repeats at DEBUG): %s", prefix, exc)
    else:
        lg.debug("%s skipped (repeat): %s", prefix, exc)


def _strip_forward_label_cols(
    rank_cols: list[str],
    log_obj: logging.Logger | None = None,
) -> list[str]:
    """PIT guard (STR-001): drop forward-looking LABEL columns from a
    cross-sectional rank list so they can never be materialised as a live
    ``*_xrank`` feature.

    The forward labels (``returns_1m/3m/6m/12m``, ``momentum_12m_excl_1m``) are
    built with ``shift(-N)`` and encode the FUTURE return; some read like
    trailing factors (name collision with the causal ``trailing_*`` twins),
    which is exactly how a forward label slips into a rank list — whether via
    the in-code default or a ``policy.yaml`` ``rank_cols`` entry. The drop is
    LOUD (WARN): silently stripping would hide a real config defect. Returns
    the filtered list with order preserved.

    Contract: ``rank_cols`` is a non-None list[str] (the sole call site builds
    one). Not None-guarded on purpose — a None here is a caller bug worth a
    TypeError, not a silent empty-rank degradation.
    """
    from src.assembled_core.features.ta_factors_core import (
        FORWARD_LOOKING_LABEL_COLS,
    )

    leaking = [c for c in rank_cols if c in FORWARD_LOOKING_LABEL_COLS]
    if not leaking:
        return list(rank_cols)
    lg = log_obj or logger
    lg.warning(
        "[FEATURE-ENH] PIT guard dropped forward-looking label column(s) from "
        "cross-sectional rank: %s — these encode future returns and must not "
        "become features (STR-001)",
        sorted(leaking),
    )
    return [c for c in rank_cols if c not in FORWARD_LOOKING_LABEL_COLS]


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

    # --- Step 2.3: Data freshness check (Backlog Item 26) ---
    # Skip in backtest mode — historical data is always "stale" by design.
    if ctx.mode in ("eod", "paper", "live") and not pwf.empty:
        try:
            from datetime import datetime, timezone as _tz

            _stale_threshold_h = 8
            _ts_col = "timestamp" if "timestamp" in pwf.columns else None
            if _ts_col:
                _latest_ts = pd.to_datetime(pwf[_ts_col]).max()
                if _latest_ts.tzinfo is None:
                    _latest_ts = _latest_ts.replace(tzinfo=_tz.utc)
                _now = datetime.now(_tz.utc)
                _age_h = (_now - _latest_ts).total_seconds() / 3600
                _n_syms = pwf["symbol"].nunique() if "symbol" in pwf.columns else 0
                if _age_h > _stale_threshold_h:
                    log.warning(
                        "[DATA-FRESHNESS] Stale data detected: latest timestamp %s is %.1fh old "
                        "(threshold %dh). Trading on stale data — signals may be wrong.",
                        _latest_ts.isoformat(),
                        _age_h,
                        _stale_threshold_h,
                    )
                else:
                    log.info(
                        "[DATA-FRESHNESS] OK: latest %s (%.1fh old, %d symbols)",
                        _latest_ts.date(),
                        _age_h,
                        _n_syms,
                    )
        except Exception as _fe:
            log.debug("[DATA-FRESHNESS] freshness check skipped: %s", _fe)

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
                    # F-A-4 MAJOR fix: sort by date BEFORE iloc[-1]. Without
                    # this, "last row" can correspond to the wrong day
                    # (e.g. if build_regime_state_hmm returns symbol/timestamp
                    # ordered rows). Wrong regime contaminates sizing + risk.
                    if "date" in hmm_df.columns:
                        hmm_df = hmm_df.sort_values("date", kind="mergesort")
                    elif "timestamp" in hmm_df.columns:
                        hmm_df = hmm_df.sort_values("timestamp", kind="mergesort")
                    ctx.regime_state = hmm_df.iloc[-1].get("regime_label", "sideways")
                    log.info("REGIME_HMM: detected regime='%s'", ctx.regime_state)
    except Exception as e:
        # Bundle-critical: ctx.regime_state drives size_positions + risk gating.
        _warn_once_feature_skip("[HMM-REGIME] regime detection", e, log)

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
                        ],
                    )
                    if c in pwf.columns
                ]
                # PIT guard (STR-001): never cross-sectionally rank a
                # forward-looking label into a live ``*_xrank`` feature, no
                # matter whether it came from the default or policy.yaml.
                rank_cols = _strip_forward_label_cols(rank_cols, log)
                if rank_cols:
                    pwf = rank_cross_sectional(
                        pwf,
                        feature_cols=rank_cols,
                        timestamp_col="timestamp",
                        normalize_to=enh_cfg.get("rank_normalize_to", "symmetric"),
                    )
    except Exception as e:
        # Bundle-critical: trend_strength_*, trailing_momentum_*, returns_* —
        # all consumed by macro_world_etfs_core_bundle and multifactor_v2.
        _warn_once_feature_skip("[FEATURE-ENH] enhanced enrichment", e, log)

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
                    _br = _bp.pct_change().fillna(0)
                    try:
                        _bc = compute_behavioral_composite(_bp, _bv, _br)
                        _beh_scores[str(_sym)] = (
                            float(_bc.iloc[-1]) if len(_bc) > 0 else 0.0
                        )
                    except Exception as _exc:
                        # F-tc-3 (2026-05-19): use `log` (param-aware) not
                        # `logger` (module-only) for consistency. Per-symbol
                        # debug — keep at DEBUG (not bundle-critical here).
                        log.debug("[behavioral_composite] %s skipped: %s", _sym, _exc)
                if _beh_scores:
                    pwf = pwf.copy()
                    pwf["behavioral_composite"] = pwf["symbol"].map(_beh_scores)
    except Exception as e:
        # Bundle-critical: behavioral_composite is referenced by composite bundles.
        _warn_once_feature_skip("[BEHAVIORAL] behavioral_features", e, log)

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
        # Bundle-critical: rv_20 listed explicitly in macro_world_etfs_core_bundle.
        _warn_once_feature_skip("[RV] realized_volatility", e, log)

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
        # Bundle-critical: news_sentiment_z + news_volume_spike are
        # consumed by composite news bundles + meta-model features.
        _warn_once_feature_skip("[NEWS-FEATURES] news_features", e, log)

    # --- Step 2.17: Macro intermarket enrichment from local macro.parquet ---
    # Broadcasts yield_curve_slope, credit_spread_change_5d,
    # bond_equity_divergence_flag, and vix into the per-symbol panel so that
    # _compute_intermarket_factors() and _compute_options_factors() can use
    # Path 1 (pre-merged columns) without a live FRED API key.
    try:
        macro_cfg = (policy.get("features") or {}).get("macro_panel") or {}
        macro_path = macro_cfg.get("path", "output/macro.parquet")
        import pathlib as _pl

        if (
            macro_cfg.get("enabled", True)
            and _pl.Path(macro_path).exists()
            and not pwf.empty
            and "timestamp" in pwf.columns
        ):
            _macro = pd.read_parquet(macro_path)
            _macro["timestamp"] = pd.to_datetime(
                _macro["timestamp"], utc=True
            ).dt.normalize()
            _macro = _macro.sort_values("timestamp").drop_duplicates(
                "timestamp", keep="last"
            )

            # Derive intermarket factor columns
            _im = pd.DataFrame({"timestamp": _macro["timestamp"]})

            # yield_curve_slope: 10y - 2y (or pre-computed yield_curve_spread)
            if "yield_curve_spread" in _macro.columns:
                _im["yield_curve_slope"] = _macro["yield_curve_spread"].values
            elif "treasury_10y" in _macro.columns and "treasury_2y" in _macro.columns:
                _im["yield_curve_slope"] = pd.to_numeric(
                    _macro["treasury_10y"], errors="coerce"
                ) - pd.to_numeric(_macro["treasury_2y"], errors="coerce")

            # credit_spread_change_5d: 5-day pct-change of hy_spread proxy
            if "hy_spread" in _macro.columns:
                _hy = pd.to_numeric(_macro["hy_spread"], errors="coerce")
                _im["credit_spread_change_5d"] = _hy.pct_change(5).fillna(0.0).values

            # bond_equity_divergence_flag: bonds up + equities down in last 5 days
            if (
                "sp500_close" in _macro.columns
                and "treasury_bond_futures" in _macro.columns
            ):
                _sp = pd.to_numeric(_macro["sp500_close"], errors="coerce")
                _tb = pd.to_numeric(_macro["treasury_bond_futures"], errors="coerce")
                _sp_ret = _sp.pct_change(5).fillna(0.0)
                _tb_ret = _tb.pct_change(5).fillna(0.0)
                _im["bond_equity_divergence_flag"] = (
                    ((_sp_ret < -0.01) & (_tb_ret > 0.01)).astype(float).values
                )

            # vix: used by VIX cap in multifactor_v2 and options factors
            if "vix" in _macro.columns:
                _im["vix"] = pd.to_numeric(_macro["vix"], errors="coerce").values
            elif "vix_close" in _macro.columns:
                _im["vix"] = pd.to_numeric(_macro["vix_close"], errors="coerce").values

            _im = _im.dropna(subset=["timestamp"])

            # Normalize pwf timestamps to date for merge
            _pwf_ts = pd.to_datetime(pwf["timestamp"], utc=True).dt.normalize()
            _new_cols = [
                c for c in _im.columns if c != "timestamp" and c not in pwf.columns
            ]
            if _new_cols:
                _im_sub = _im[["timestamp"] + _new_cols].copy()
                _pwf_idx = pd.DataFrame(
                    {"timestamp": _pwf_ts, "_row_idx": range(len(pwf))}
                )
                _merged = _pwf_idx.merge(_im_sub, on="timestamp", how="left")
                for col in _new_cols:
                    if col in _merged.columns:
                        pwf = pwf.copy()
                        pwf[col] = _merged[col].values
                log.debug(
                    "[MACRO-PANEL] merged %d macro cols into panel (%d rows)",
                    len(_new_cols),
                    len(pwf),
                )
    except Exception as e:
        # Bundle-critical: macro_growth/inflation/risk_aversion regime cols.
        _warn_once_feature_skip("[MACRO-PANEL] macro_panel", e, log)

    # --- Step 2.17b: Caldara-Iacoviello GPR enrichment (§9.9 wiring) ---
    # Populates the gpr_index column that _compute_geo_risk_composite (mfv2
    # factor 31) reads via Path 1 (pre-merged panel columns) — replaces the
    # dead FRED GPRC fetch removed in commit 6be8ce3 (the FRED series does
    # not exist). Source data is produced by
    # scripts/ops/fetch_caldara_iacoviello_gpr.py into output/macro_gpr.parquet
    # (monthly, ~1516 rows 1900..present). Default-on but graceful: missing
    # file → debug log + no panel mutation (Path 2 zero-fill continues).
    try:
        gpr_cfg = (policy.get("features") or {}).get("macro_gpr") or {}
        if gpr_cfg.get("enabled", True):
            from src.assembled_core.data.macro.gpr import merge_gpr_index_into_panel

            gpr_path = gpr_cfg.get("path", "output/macro_gpr.parquet")
            pwf = merge_gpr_index_into_panel(pwf, gpr_path)
    except Exception as e:
        # Non-critical: failure here just leaves Path 2 zero-fill active.
        _warn_once_feature_skip("[MACRO-GPR] macro_gpr", e, log)

    # ------------------------------------------------------------------
    # Step 2.18: News sentiment enrichment — per-symbol per-date features
    # Source: output/news_sentiment_daily.parquet (Finnhub/NewsAPI/Polygon/RSS)
    # Columns added: fetched_sentiment_score, fetched_sentiment_volume
    # ------------------------------------------------------------------
    try:
        import pathlib as _pl

        _news_path = _pl.Path("output") / "news_sentiment_daily.parquet"
        if _news_path.exists():
            _ns = pd.read_parquet(_news_path)
            # Normalize timestamp to date (UTC midnight)
            _ns["timestamp"] = pd.to_datetime(_ns["timestamp"], utc=True).dt.normalize()
            _ns = _ns.rename(
                columns={
                    "sentiment_score": "fetched_sentiment_score",
                    "sentiment_volume": "fetched_sentiment_volume",
                }
            )
            _ns_cols = ["fetched_sentiment_score", "fetched_sentiment_volume"]
            _ns_sub = _ns[["timestamp", "symbol"] + _ns_cols].drop_duplicates(
                subset=["timestamp", "symbol"], keep="last"
            )

            # Merge per (timestamp, symbol)
            _pwf_ts2 = pd.to_datetime(pwf["timestamp"], utc=True).dt.normalize()
            _pwf_idx2 = pd.DataFrame(
                {
                    "timestamp": _pwf_ts2,
                    "symbol": pwf["symbol"],
                    "_row_idx2": range(len(pwf)),
                }
            )
            _merged2 = _pwf_idx2.merge(_ns_sub, on=["timestamp", "symbol"], how="left")
            for col in _ns_cols:
                if col not in pwf.columns and col in _merged2.columns:
                    pwf = pwf.copy()
                    pwf[col] = _merged2[col].values
            log.debug(
                "[NEWS-PANEL] merged news sentiment cols into panel (%d rows)",
                len(pwf),
            )
    except Exception as e:
        # Bundle-critical: news_sentiment columns merged into panel.
        _warn_once_feature_skip("[NEWS-PANEL] news_sentiment", e, log)

    return pwf, prices_latest_update
