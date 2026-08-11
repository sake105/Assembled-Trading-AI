"""_tc_sizing — size_positions() and all _sp_* helpers extracted from trading_cycle_v2."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd

# Module-level cache: market-return series derived from panel (keyed by panel path).
# Item 3: bounded to _MAX_HMM_CACHE_ENTRIES to prevent memory leak during walk-forward runs.
_HMM_MKT_RET_CACHE: dict[str, pd.Series] = {}
_MAX_HMM_CACHE_ENTRIES = 4

# GPR parquet cache: keyed by file path; loaded once per process (monthly data, ~1500 rows).
# Avoids re-reading on every backtest cycle while keeping the direct-read fallback cheap.
_GPR_PARQUET_CACHE: dict[str, "pd.DataFrame | None"] = {}
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    _estimate_symbol_volatilities,
    _record_degraded_step,
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
from src.assembled_core.risk.profit_lock import compute_profit_lock_multiplier
from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
)
from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

logger = logging.getLogger(__name__)

# E-059 follow-up (Stage-2 F-senior-6): warn-once dedup for D-site except
# swallows whose ONLY visibility is a `log.debug(...)` — the exact mask that
# hid the archived shadow_recorder for months (E-059). D1/D3 already surface
# via _record_degraded_step (WARN + meta trail, not doubled here); D4
# (inverse_etf) had DEBUG-only visibility. WARN must be deduped, or a
# persistent error emits one WARN per bar in backtests (1260+ bars/5y). Same
# E-018 pattern as _tc_features._warn_once_feature_skip; module-local registry
# so pipeline stages stay independent. Reset on process restart.
_SIZING_STEP_WARN_KEYS: set[tuple[str, str]] = set()
_WARN_KEY_MAX_CHARS = 200
_WARN_REGISTRY_MAX_KEYS = 1024


def _warn_once_sizing_skip(
    prefix: str,
    exc: BaseException,
    log_obj: logging.Logger | None = None,
) -> None:
    """Emit WARN on first occurrence per (prefix, exc-signature) per process,
    DEBUG on repeats. Log-level only — the caller keeps its graceful except
    (no raise). Mirror of ``_tc_features._warn_once_feature_skip`` incl. the
    bounded-registry failure mode (once full, further distinct keys still WARN
    but are not registered — more noise instead of silent demotion)."""
    lg = log_obj or logger
    key = (prefix, f"{type(exc).__name__}:{str(exc)[:_WARN_KEY_MAX_CHARS]}")
    if key not in _SIZING_STEP_WARN_KEYS:
        if len(_SIZING_STEP_WARN_KEYS) < _WARN_REGISTRY_MAX_KEYS:
            _SIZING_STEP_WARN_KEYS.add(key)
        lg.warning("%s skipped (first occurrence, repeats at DEBUG): %s", prefix, exc)
    else:
        lg.debug("%s skipped (repeat): %s", prefix, exc)


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
    # position_sizing_fn is Optional on TradingContext only for late binding;
    # every orchestrator entry path assigns it before sizing. cast() keeps
    # runtime behaviour byte-identical (a None would raise TypeError at the
    # same call sites as before).
    _sizing_fn = cast(
        "Callable[[pd.DataFrame, float], pd.DataFrame]", ctx.position_sizing_fn
    )
    try:
        if sizing_method == "kelly":
            from src.assembled_core.portfolio.position_sizing import (
                compute_kelly_weights,
            )

            target_positions = compute_kelly_weights(
                signals,
                fraction=float(sizing_cfg.get("kelly_fraction", 0.5)),
                max_weight=float(sizing_cfg.get("max_weight", 0.25)),
                total_capital=ctx.capital,
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "risk_parity":
            from src.assembled_core.portfolio.position_sizing import (
                compute_risk_parity_weights,
            )

            vols = _estimate_symbol_volatilities(
                prices_for_sizing, lookback=int(sizing_cfg.get("vol_lookback_days", 60))
            )
            target_positions = compute_risk_parity_weights(
                signals,
                vols,
                total_capital=ctx.capital,
                max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "vol_scaled":
            from src.assembled_core.portfolio.position_sizing import (
                compute_vol_scaled_weights,
            )

            vols = _estimate_symbol_volatilities(
                prices_for_sizing, lookback=int(sizing_cfg.get("vol_lookback_days", 60))
            )
            target_positions = compute_vol_scaled_weights(
                signals,
                vols,
                target_vol=float(sizing_cfg.get("target_vol", 0.15)),
                total_capital=ctx.capital,
                max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "black_litterman":
            try:
                from src.assembled_core.portfolio.black_litterman import (
                    BlackLittermanOptimizer,
                )
                from src.assembled_core.portfolio.covariance import estimate_covariance

                bl = BlackLittermanOptimizer(
                    risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)),
                    tau=float(sizing_cfg.get("tau", 0.05)),
                    max_position=float(sizing_cfg.get("max_weight", 0.15)),
                    min_position=float(sizing_cfg.get("min_position", 0.0)),
                )
                scores_dict: dict[str, float] = {}
                if (
                    not signals.empty
                    and "symbol" in signals.columns
                    and "score" in signals.columns
                ):
                    _bl = signals[["symbol", "score"]].copy()
                    _bl["score"] = _bl["score"].astype(float).fillna(0.0)
                    scores_dict = {
                        str(sym): float(s)
                        for sym, s in _bl[_bl["score"].abs() > 0.01]
                        .set_index("symbol")["score"]
                        .items()
                    }
                if (
                    scores_dict
                    and prices_for_sizing is not None
                    and not prices_for_sizing.empty
                    and "close" in prices_for_sizing.columns
                    and "symbol" in prices_for_sizing.columns
                ):
                    _pivot = prices_for_sizing.pivot_table(
                        index="timestamp", columns="symbol", values="close"
                    )
                    sigma = estimate_covariance(
                        _pivot.pct_change().dropna(how="all"),
                        method=sizing_cfg.get("cov_method", "ledoit_wolf"),
                    )
                    if not sigma.empty:
                        bl_w = bl.optimize_from_scores(
                            scores=pd.Series(scores_dict),
                            sigma=sigma,
                            confidence=float(sizing_cfg.get("bl_confidence", 0.5)),
                        )
                        target_positions = pd.DataFrame(
                            [
                                {
                                    "symbol": s,
                                    "target_weight": round(w, 4),
                                    "target_qty": round(w * ctx.capital, 2),
                                }
                                for s, w in bl_w.items()
                            ]
                        )
                    else:
                        target_positions = _sizing_fn(signals, ctx.capital)
                else:
                    target_positions = _sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("Black-Litterman sizing failed, using default: %s", e)
                target_positions = _sizing_fn(signals, ctx.capital)
        elif sizing_method == "cost_aware":
            try:
                from src.assembled_core.portfolio.cost_aware_optimizer import (
                    OptimizerConfig,
                    optimize_portfolio,
                )
                from src.assembled_core.portfolio.covariance import estimate_covariance

                if (
                    prices_for_sizing is not None
                    and not prices_for_sizing.empty
                    and not signals.empty
                    and "close" in prices_for_sizing.columns
                    and "symbol" in prices_for_sizing.columns
                ):
                    _pivot_cao = prices_for_sizing.pivot_table(
                        index="timestamp", columns="symbol", values="close"
                    )
                    sigma_cao = estimate_covariance(
                        _pivot_cao.pct_change().dropna(how="all"),
                        method="ledoit_wolf",
                    )
                    mu_cao = (
                        signals.set_index("symbol")["score"]
                        .reindex(sigma_cao.index)
                        .fillna(0.0)
                        if "score" in signals.columns
                        else pd.Series(dtype=float)
                    )
                    _cur_w: dict[str, float] = {}
                    if (
                        ctx.current_positions is not None
                        and isinstance(ctx.current_positions, pd.DataFrame)
                        and "symbol" in ctx.current_positions.columns
                    ):
                        _cp = ctx.current_positions
                        _wcol_cao = (
                            "weight" if "weight" in _cp.columns else "target_weight"
                        )
                        if _wcol_cao in _cp.columns:
                            _cur_w = {
                                str(k): float(v)
                                for k, v in _cp.set_index("symbol")[_wcol_cao]
                                .fillna(0.0)
                                .items()
                            }
                    cao_res = optimize_portfolio(
                        mu_cao,
                        sigma_cao,
                        _cur_w,
                        config=OptimizerConfig(
                            risk_aversion=float(sizing_cfg.get("risk_aversion", 1.0)),
                            turnover_penalty=float(
                                sizing_cfg.get("turnover_penalty", 0.001)
                            ),
                            max_weight=float(sizing_cfg.get("max_weight", 0.10)),
                        ),
                    )
                    target_positions = pd.DataFrame(
                        [
                            {
                                "symbol": s,
                                "target_weight": round(w, 4),
                                "target_qty": round(w * ctx.capital, 2),
                            }
                            for s, w in cao_res.weights.items()
                            if abs(w) > 1e-6
                        ]
                    )
                else:
                    target_positions = _sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("cost_aware_optimizer failed, using default: %s", e)
                target_positions = _sizing_fn(signals, ctx.capital)
        elif sizing_method == "erc":
            try:
                from src.assembled_core.portfolio.covariance import estimate_covariance

                # FIXME(mypy-sweep): module does not exist — this sizing method
                # always falls back to default sizing via the except below.
                from src.assembled_core.portfolio.risk_budgeting import (  # type: ignore[import-not-found]
                    compute_erc_weights,
                )

                if (
                    prices_for_sizing is not None
                    and not prices_for_sizing.empty
                    and not signals.empty
                    and "close" in prices_for_sizing.columns
                    and "symbol" in prices_for_sizing.columns
                ):
                    _sig_syms = [
                        s
                        for s in signals["symbol"].tolist()
                        if s in prices_for_sizing["symbol"].unique()
                    ]
                    if len(_sig_syms) >= 2:
                        _pivot_erc = prices_for_sizing[
                            prices_for_sizing["symbol"].isin(_sig_syms)
                        ].pivot_table(
                            index="timestamp", columns="symbol", values="close"
                        )
                        _rets_erc = _pivot_erc.pct_change().dropna(how="all")
                        if len(_rets_erc) >= 3:
                            sigma_erc = estimate_covariance(
                                _rets_erc, method="ledoit_wolf"
                            )
                            erc_res = compute_erc_weights(
                                sigma_erc,
                                symbols=list(sigma_erc.columns),
                                long_only=True,
                                max_weight=float(sizing_cfg.get("max_weight", 0.25)),
                            )
                            target_positions = pd.DataFrame(
                                [
                                    {
                                        "symbol": s,
                                        "target_weight": round(w, 6),
                                        "target_qty": round(w * ctx.capital, 2),
                                    }
                                    for s, w in erc_res.weights.items()
                                    if abs(w) > 1e-6
                                ]
                            )
                        else:
                            target_positions = _sizing_fn(signals, ctx.capital)
                    else:
                        target_positions = _sizing_fn(signals, ctx.capital)
                else:
                    target_positions = _sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("erc sizing failed, using default: %s", e)
                target_positions = _sizing_fn(signals, ctx.capital)
        elif sizing_method == "bl_blend":
            try:
                # FIXME(mypy-sweep): module does not exist — this sizing method
                # always falls back to default sizing via the except below.
                from src.assembled_core.portfolio.bl_sizing import (  # type: ignore[import-not-found]
                    apply_bl_sizing,
                )

                base_tp = _sizing_fn(signals, ctx.capital)
                if (
                    base_tp is not None
                    and not base_tp.empty
                    and "target_weight" in base_tp.columns
                    and prices_for_sizing is not None
                    and not prices_for_sizing.empty
                ):
                    _btw = base_tp.dropna(subset=["target_weight"])
                    score_w = {
                        str(k): float(v)
                        for k, v in _btw.set_index("symbol")["target_weight"].items()
                    }
                    bl_w, _ = apply_bl_sizing(
                        score_w,
                        prices_for_sizing,
                        lookback_days=int(sizing_cfg.get("lookback_days", 60)),
                        risk_aversion=float(sizing_cfg.get("risk_aversion", 2.5)),
                        tau=float(sizing_cfg.get("tau", 0.05)),
                        max_position=float(sizing_cfg.get("max_weight", 0.15)),
                        confidence=float(sizing_cfg.get("bl_confidence", 0.5)),
                        return_scale=float(sizing_cfg.get("return_scale", 0.10)),
                        target_invested_pct=float(
                            sizing_cfg.get("target_invested_pct", 1.0)
                        ),
                    )
                    target_positions = pd.DataFrame(
                        [
                            {
                                "symbol": s,
                                "target_weight": round(w, 6),
                                "target_qty": round(w * ctx.capital, 2),
                            }
                            for s, w in bl_w.items()
                            if abs(w) > 1e-6
                        ]
                    )
                else:
                    target_positions = _sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("bl_blend sizing failed, using default: %s", e)
                target_positions = _sizing_fn(signals, ctx.capital)
        elif sizing_method == "hrp":
            try:
                from src.assembled_core.portfolio.hrp_sizing import apply_hrp_sizing

                base_tp = _sizing_fn(signals, ctx.capital)
                if (
                    base_tp is not None
                    and not base_tp.empty
                    and "target_weight" in base_tp.columns
                    and prices_for_sizing is not None
                    and not prices_for_sizing.empty
                ):
                    _btw = base_tp.dropna(subset=["target_weight"])
                    score_w = {
                        str(k): float(v)
                        for k, v in _btw.set_index("symbol")["target_weight"].items()
                    }
                    blended, _ = apply_hrp_sizing(
                        score_w,
                        prices_for_sizing,
                        lookback_days=int(sizing_cfg.get("lookback_days", 60)),
                        blend=float(sizing_cfg.get("blend", 0.7)),
                        target_invested_pct=float(
                            sizing_cfg.get("target_invested_pct", 1.0)
                        ),
                        min_weight=float(sizing_cfg.get("min_weight", 0.0)),
                        max_weight=float(sizing_cfg.get("max_weight", 1.0)),
                    )
                    target_positions = pd.DataFrame(
                        [
                            {
                                "symbol": s,
                                "target_weight": round(w, 6),
                                "target_qty": round(w * ctx.capital, 2),
                            }
                            for s, w in blended.items()
                            if abs(w) > 1e-6
                        ]
                    )
                else:
                    target_positions = _sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("hrp sizing failed, using default: %s", e)
                target_positions = _sizing_fn(signals, ctx.capital)
        elif sizing_method == "mvo":
            try:
                import numpy as _np
                from src.assembled_core.portfolio.covariance import estimate_covariance

                # FIXME(mypy-sweep): module does not exist — this sizing method
                # always falls back to default sizing via the except below.
                from src.assembled_core.portfolio.mvo_optimizer import (  # type: ignore[import-not-found]
                    mvo_with_cardinality,
                )

                if (
                    prices_for_sizing is not None
                    and not prices_for_sizing.empty
                    and not signals.empty
                    and "close" in prices_for_sizing.columns
                    and "symbol" in prices_for_sizing.columns
                ):
                    _sig_syms_mvo = [
                        s
                        for s in signals["symbol"].tolist()
                        if s in prices_for_sizing["symbol"].unique()
                    ]
                    if len(_sig_syms_mvo) >= 2:
                        _pivot_mvo = prices_for_sizing[
                            prices_for_sizing["symbol"].isin(_sig_syms_mvo)
                        ].pivot_table(
                            index="timestamp", columns="symbol", values="close"
                        )
                        _rets_mvo = _pivot_mvo.pct_change().dropna(how="all")
                        if len(_rets_mvo) >= 3:
                            sigma_mvo = estimate_covariance(
                                _rets_mvo, method="ledoit_wolf"
                            ).values
                            mvo_syms = list(_rets_mvo.columns)
                            mu_series = (
                                signals.set_index("symbol")["score"]
                                if "score" in signals.columns
                                else pd.Series(0.0, index=mvo_syms)
                            )
                            mu_mvo = _np.asarray(
                                mu_series.reindex(mvo_syms).fillna(0.0).values,
                                dtype=float,
                            )
                            w_arr = mvo_with_cardinality(
                                mu_mvo,
                                sigma_mvo,
                                max_positions=int(sizing_cfg.get("max_positions", 20)),
                                risk_aversion=float(
                                    sizing_cfg.get("risk_aversion", 1.0)
                                ),
                                min_weight=float(sizing_cfg.get("min_weight", 0.01)),
                            )
                            target_positions = pd.DataFrame(
                                [
                                    {
                                        "symbol": s,
                                        "target_weight": round(float(w_arr[i]), 6),
                                        "target_qty": round(
                                            float(w_arr[i]) * ctx.capital, 2
                                        ),
                                    }
                                    for i, s in enumerate(mvo_syms)
                                    if abs(w_arr[i]) > 1e-6
                                ]
                            )
                        else:
                            target_positions = _sizing_fn(signals, ctx.capital)
                    else:
                        target_positions = _sizing_fn(signals, ctx.capital)
                else:
                    target_positions = _sizing_fn(signals, ctx.capital)
            except Exception as e:
                log.warning("mvo sizing failed, using default: %s", e)
                target_positions = _sizing_fn(signals, ctx.capital)
        else:
            target_positions = _sizing_fn(signals, ctx.capital)

        if target_positions is None or target_positions.empty:
            target_positions = pd.DataFrame(
                columns=["symbol", "target_weight", "target_qty"]
            )
        if not any(
            c in target_positions.columns for c in ["target_weight", "target_qty"]
        ):
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
        if (
            liq_cfg.get("enabled", False)
            and not target_positions.empty
            and "target_weight" in target_positions.columns
            and prices_for_sizing is not None
            and not prices_for_sizing.empty
        ):
            from src.assembled_core.risk.liquidity_scoring import (
                apply_liquidity_adjusted_sizing,
                compute_liquidity_scores,
            )

            liq_scores = compute_liquidity_scores(
                prices_for_sizing, lookback_days=int(liq_cfg.get("lookback_days", 60))
            )
            if liq_scores:
                tw_map = {
                    str(k).upper(): float(v)
                    for k, v in target_positions.set_index("symbol")[
                        "target_weight"
                    ].items()
                }
                for s in liq_scores:
                    s.symbol = s.symbol.upper()
                adjusted_tw = apply_liquidity_adjusted_sizing(
                    tw_map,
                    liq_scores,
                    alpha=float(liq_cfg.get("alpha", 0.5)),
                    min_score_threshold=float(liq_cfg.get("min_score_threshold", 0.1)),
                )
                target_positions["target_weight"] = (
                    target_positions["symbol"]
                    .astype(str)
                    .str.upper()
                    .map(adjusted_tw)
                    .fillna(target_positions["target_weight"])
                )
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
        if (
            pl_cfg.get("enabled")
            and getattr(ctx, "equity_curve", None) is not None
            and getattr(ctx, "equity_curve_index", None) is not None
        ):
            pl_state = getattr(ctx, "profit_lock_state", None) or {}
            profit_lock_mult, pl_state_out = compute_profit_lock_multiplier(
                ctx.equity_curve,
                pl_cfg,
                # cast: non-None guaranteed by the getattr-guard above (mypy
                # cannot narrow attribute access through getattr()).
                cast(int, ctx.equity_curve_index),
                state=pl_state,
            )
            ctx.profit_lock_state = pl_state_out
            meta["profit_lock_state"] = pl_state_out
            meta["profit_lock"] = {"multiplier": float(profit_lock_mult)}
    except Exception as e:
        log.debug("profit_lock skipped: %s", e)

    vol_scale_factor = 1.0
    try:
        vt_cfg = policy.get("vol_targeting") or {}
        if (
            vt_cfg.get("enabled", False)
            and getattr(ctx, "equity_curve", None) is not None
            and getattr(ctx, "equity_curve_index", None) is not None
        ):
            vol_scale_factor, realized_vol, target_vol = compute_vol_targeting_result(
                ctx.equity_curve,
                vt_cfg,
                # cast: non-None guaranteed by the getattr-guard above.
                now_idx=cast(int, ctx.equity_curve_index),
            )
            meta["vol_targeting"] = {
                "scale_factor": vol_scale_factor,
                "realized_vol": realized_vol,
                "target_vol": target_vol,
            }
        else:
            meta["vol_targeting"] = {
                "scale_factor": 1.0,
                "realized_vol": float("nan"),
                "target_vol": float("nan"),
            }
    except Exception as e:
        _record_degraded_step("vol_targeting", e, meta=meta, log_obj=log)
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
    # Local binding (same object) so mypy sees the truthiness guard; runtime
    # behaviour identical to the previous getattr-based check.
    _crisis_intel = getattr(ctx, "crisis_state_intel", None)
    if _crisis_intel and os.environ.get("ASSEMBLED_NO_CRISIS_OVERLAY") != "1":
        crisis_mode = str(_crisis_intel.get("mode", "NORMAL")).upper()
        ca_cfg = (
            policy.get("crisis_alpha")
            or policy.get("intel", {}).get("crisis_alpha")
            or {}
        )
        if crisis_mode == "CRISIS":
            crisis_alpha_multiplier = min(
                float(ca_cfg.get("crisis_multiplier", 0.25)), 1.0
            )
        elif crisis_mode == "ELEVATED":
            crisis_alpha_multiplier = min(
                float(ca_cfg.get("elevated_multiplier", 0.60)), 1.0
            )

    # Prediction-market overlay (live/paper only — never fetches live API in backtest)
    pm_multiplier = 1.0
    try:
        pm_cfg = policy.get("prediction_market_overlay") or {}
        if pm_cfg.get("enabled", False) and getattr(ctx, "mode", "") in (
            "live",
            "paper",
        ):
            from src.assembled_core.risk.georisk_overlay import (
                get_market_implied_geo_signal,
            )

            pm_signal = get_market_implied_geo_signal(policy=policy)
            raw_pm = float(pm_signal.get("signal", 0.0))
            pm_threshold = float(pm_cfg.get("threshold", 0.25))
            if raw_pm > pm_threshold:
                reduction = float(pm_cfg.get("reduction_factor", 0.50))
                pm_multiplier = max(0.0, 1.0 - reduction * raw_pm)
            meta["prediction_market"] = {
                "signal": raw_pm,
                "multiplier": pm_multiplier,
                "n_sources": pm_signal.get("n_sources", 0),
            }
    except Exception as e:
        log.debug("prediction_market_overlay skipped: %s", e)

    # HMM regime overlay — reduce/increase exposure based on detected market regime
    hmm_regime_multiplier = 1.0
    try:
        hmm_cfg = policy.get("hmm_regime_overlay") or {}
        if hmm_cfg.get("enabled", False):
            from pathlib import Path as _Path
            from src.assembled_core.ml.regime_hmm import MultiFeatureRegimeHMM

            _root = _Path(__file__).parents[3]
            _model_path = hmm_cfg.get(
                "model_path", "models/regime_hmm_4state_spy.joblib"
            )
            _hmm_path = _root / _model_path
            if _hmm_path.exists():
                _hmm = MultiFeatureRegimeHMM.load(_hmm_path)
                # Build market-return series: prefer full panel (for snapshot mode where
                # ctx.prices may only contain a single day), fall back to ctx.prices.
                _prices_src = getattr(ctx, "prices", None)
                _mkt_ret: pd.Series | None = None
                # Try full panel first — gives rolling history needed for vol estimate.
                # Cached per process (keyed by path string) to avoid re-reading 210k rows
                # at every cycle.
                _panel_path = (
                    _root / "output" / "factor_panels" / "full_panel_7y.parquet"
                )
                _panel_key = str(_panel_path)
                if _panel_path.exists():
                    try:
                        if _panel_key not in _HMM_MKT_RET_CACHE:
                            if len(_HMM_MKT_RET_CACHE) >= _MAX_HMM_CACHE_ENTRIES:
                                _HMM_MKT_RET_CACHE.pop(next(iter(_HMM_MKT_RET_CACHE)))
                            _panel = pd.read_parquet(_panel_path)
                            _close_col = (
                                "close" if "close" in _panel.columns else "adj_close"
                            )
                            _panel["date"] = pd.to_datetime(
                                _panel["date"]
                            ).dt.tz_localize(None)
                            _px_full = _panel.pivot_table(
                                index="date",
                                columns="symbol",
                                values=_close_col,
                                aggfunc="last",
                            )
                            _px_mean = _px_full.mean(axis=1)
                            _px_mean_lag = _px_mean.shift(1).replace(0, np.nan)
                            _HMM_MKT_RET_CACHE[_panel_key] = np.log(
                                (_px_mean / _px_mean_lag).clip(lower=1e-10)
                            ).dropna()
                            log.info(
                                "[HMM-REGIME] Panel market-return series cached (%d days)",
                                len(_HMM_MKT_RET_CACHE[_panel_key]),
                            )
                        _full_mkt_ret = _HMM_MKT_RET_CACHE[_panel_key]
                        _as_of_raw = (
                            pd.Timestamp(
                                getattr(ctx, "as_of", _full_mkt_ret.index.max())
                            )
                            if getattr(ctx, "as_of", None)
                            else _full_mkt_ret.index.max()
                        )
                        _as_of = (
                            _as_of_raw.tz_convert(None)
                            if _as_of_raw.tzinfo is not None
                            else _as_of_raw
                        )
                        _mkt_ret = _full_mkt_ret[_full_mkt_ret.index <= _as_of].iloc[
                            -60:
                        ]
                    except Exception as _exc:
                        log.debug(
                            "[HMM-REGIME] market-return panel load failed: %s", _exc
                        )
                # Fall back to ctx.prices if panel load failed
                if _mkt_ret is None and _prices_src is not None:
                    if (
                        "close" in _prices_src.columns
                        and "symbol" in _prices_src.columns
                        and "timestamp" in _prices_src.columns
                    ):
                        _px = _prices_src.pivot_table(
                            index="timestamp",
                            columns="symbol",
                            values="close",
                            aggfunc="last",
                        )
                        _mkt_ret = np.log(
                            (_px.mean(axis=1) / _px.mean(axis=1).shift(1)).clip(
                                lower=1e-10
                            )
                        ).dropna()
                if _mkt_ret is not None and len(_mkt_ret) >= 20:
                    _mkt_vol = _mkt_ret.rolling(20, min_periods=20).std().dropna()
                    _mkt_ret = _mkt_ret.loc[_mkt_vol.index]
                    if len(_mkt_ret) >= 20:
                        _feat = pd.DataFrame(
                            {
                                "daily_return": _mkt_ret.values,
                                "realized_vol": _mkt_vol.values,
                            },
                            index=_mkt_ret.index,
                        )
                        _regimes = _hmm.predict_regime(_feat)
                        _regime = (
                            str(_regimes.iloc[-1]) if len(_regimes) > 0 else "sideways"
                        )
                        _mult_map = hmm_cfg.get("multipliers") or {
                            "bull": 1.15,
                            "sideways": 1.0,
                            "bear": 0.75,
                            "crisis": 0.40,
                        }
                        hmm_regime_multiplier = float(_mult_map.get(_regime, 1.0))
                        meta["hmm_regime"] = {
                            "regime": _regime,
                            "multiplier": hmm_regime_multiplier,
                        }
                        log.info(
                            "[HMM-REGIME] regime=%s multiplier=%.3f",
                            _regime,
                            hmm_regime_multiplier,
                        )
    except Exception as e:
        log.warning("[HMM-REGIME] overlay skipped unexpectedly: %r", e)

    # EDCL conviction overlay — Phase H triple-confirmation (EDCL + regime + IV skew)
    edcl_multiplier = 1.0
    # Derive composite regime from crisis_state_intel once — reused for both
    # the EDCL multiplier lookup and the suppression guard below.
    _crisis_intel_edcl = getattr(ctx, "crisis_state_intel", None) or {}
    _crisis_mode_edcl = str(_crisis_intel_edcl.get("mode", "NORMAL")).upper()
    _composite_regime = (
        "crisis"
        if _crisis_mode_edcl == "CRISIS"
        else "elevated"
        if _crisis_mode_edcl == "ELEVATED"
        else "normal"
    )

    try:
        edcl_cfg = (policy or {}).get("edcl_conviction_overlay") or {}
        if edcl_cfg.get("enabled", False):
            _mode = getattr(ctx, "mode", "backtest")
            if _mode in ("live", "paper") or edcl_cfg.get("allow_in_backtest", False):
                _edcl_state = getattr(ctx, "edcl_state", None) or {}
                _edcl_conviction = float(_edcl_state.get("conviction", 0.0))
                # IV skew Z-score — optional field, defaults to 0.0 (no IV data)
                _iv_skew_z = float(getattr(ctx, "options_iv_skew_z", 0.0) or 0.0)
                from src.assembled_core.signals.composite_score import (
                    compute_edcl_conviction_multiplier as _phase_h_mult,
                )

                edcl_multiplier = _phase_h_mult(
                    _edcl_conviction, _composite_regime, _iv_skew_z, policy
                )
                if edcl_multiplier != 1.0:
                    meta["edcl_conviction"] = {
                        "multiplier": edcl_multiplier,
                        "conviction": _edcl_conviction,
                        "regime": _composite_regime,
                        "iv_skew_z": _iv_skew_z,
                    }
                    log.info(
                        "[EDCL-H] triple_confirm: conviction=%.3f regime=%s iv_z=%.2f → mult=%.3f",
                        _edcl_conviction,
                        _composite_regime,
                        _iv_skew_z,
                        edcl_multiplier,
                    )
    except Exception as e:
        _edcl_mode_check = getattr(ctx, "mode", "backtest")
        _log_fn = log.warning if _edcl_mode_check in ("live", "paper") else log.debug
        _log_fn("edcl_conviction_overlay raised — multiplier stays 1.0: %s", e)

    # Suppress EDCL conviction boosts when crisis-alpha is CRISIS or ELEVATED.
    # The crisis overlay adds defensive ETFs via _sp_apply_crisis_alpha_cap;
    # boosting long-equity sizing into a confirmed/pre-crisis state would
    # work against the drawdown-protection objective.
    # Values ≤ 1.0 are intentionally not suppressed — they already reduce exposure.
    # ELEVATED is treated the same as CRISIS (matching composite_score.py semantics
    # where both regimes produce 1.5–2.0 EDCL multipliers).
    if edcl_multiplier > 1.0 and _crisis_mode_edcl in ("CRISIS", "ELEVATED"):
        _suppress_log = (
            log.info
            if getattr(ctx, "mode", "backtest") in ("live", "paper")
            else log.debug
        )
        _suppress_log(
            "[EDCL-H] suppressed (crisis_alpha %s): mult %.2f → 1.0",
            _crisis_mode_edcl,
            edcl_multiplier,
        )
        edcl_multiplier = 1.0

    final_multiplier = (
        geo_multiplier
        * profit_lock_mult
        * vol_scale_factor
        * ms_multiplier
        * crisis_alpha_multiplier
        * pm_multiplier
        * hmm_regime_multiplier
        * edcl_multiplier
    )
    _MIN_EXPOSURE_MULT = 0.05
    _MAX_EXPOSURE_MULT = 3.0
    if final_multiplier < _MIN_EXPOSURE_MULT:
        log.warning(
            "[SIZE] exposure multiplier %.4f below floor %.2f — clamping",
            final_multiplier,
            _MIN_EXPOSURE_MULT,
        )
        final_multiplier = _MIN_EXPOSURE_MULT
    if final_multiplier > _MAX_EXPOSURE_MULT:
        log.warning(
            "[SIZE] exposure multiplier %.4f above ceiling %.1f — clamping",
            final_multiplier,
            _MAX_EXPOSURE_MULT,
        )
        final_multiplier = _MAX_EXPOSURE_MULT
    if final_multiplier > 1.5:
        log.warning(
            "[SIZE] exposure multiplier %.4f > 1.5 — confirm EDCL/HMM overlay is intentional",
            final_multiplier,
        )
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
        if (
            factor_risk_cfg.get("enabled", False)
            and prices_for_sizing is not None
            and not prices_for_sizing.empty
        ):
            # FIXME(mypy-sweep): module does not exist — factor-risk overlay is
            # a silent no-op via the enclosing except.
            from src.assembled_core.risk.factor_risk_model import (  # type: ignore[import-not-found]
                FactorRiskModel,
            )

            frm = FactorRiskModel()
            frm.fit(prices_for_sizing)
            if "target_weight" in target_positions.columns:
                tw_dict = dict(
                    zip(
                        target_positions["symbol"],
                        target_positions["target_weight"].fillna(0),
                    )
                )
                portfolio_vol = frm.predict_portfolio_vol(tw_dict)
                vol_limit = float(factor_risk_cfg.get("max_portfolio_vol", 0.25))
                if portfolio_vol > vol_limit and portfolio_vol > 0:
                    scale = vol_limit / portfolio_vol
                    target_positions["target_weight"] = (
                        target_positions["target_weight"] * scale
                    )
                    log.info(
                        "FACTOR_RISK: portfolio_vol=%.3f > limit=%.3f → scaled by %.3f",
                        portfolio_vol,
                        vol_limit,
                        scale,
                    )
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
                from src.assembled_core.risk.trailing_stops import (
                    apply_stop_reductions_to_weights,
                    compute_trailing_stops,
                )

                _price_priority = [
                    c
                    for c in ("avg_entry_price", "entry_price", "price")
                    if c in current_positions_df.columns
                ]
                pos_map: dict[str, dict] = {}
                if _price_priority:
                    _cpd = current_positions_df.copy()
                    # KEINE Unterstrich-Spaltennamen: itertuples benennt
                    # Bezeichner mit fuehrendem "_" positional um (_0, _1, ...)
                    # — row._sym/row._entry warfen deshalb IMMER
                    # AttributeError und der DEGRADED-Pfad uebersprang die
                    # Trailing-Stops in jeder Runde, in jedem Env (Fund via
                    # CI-Log 2026-08-11; in beiden pandas-Versionen
                    # reproduziert).
                    _cpd["sym_norm"] = _cpd["symbol"].astype(str).str.upper()
                    # Coalesce price columns in priority order (first non-null wins)
                    _cpd["entry_px"] = pd.to_numeric(
                        _cpd[_price_priority].bfill(axis=1).iloc[:, 0], errors="coerce"
                    )
                    _cpd = _cpd[_cpd["sym_norm"].str.len() > 0].dropna(
                        subset=["entry_px"]
                    )
                    _qty_col = "qty" if "qty" in _cpd.columns else None
                    _wt_col = "weight" if "weight" in _cpd.columns else None
                    for row in _cpd.itertuples(index=False):
                        pos_map[row.sym_norm] = {
                            "entry_price": float(row.entry_px),
                            "qty": (
                                float(getattr(row, _qty_col, 0.0) or 0.0)
                                if _qty_col
                                else 0.0
                            ),
                            "weight": (
                                float(getattr(row, _wt_col, 0.0) or 0.0)
                                if _wt_col
                                else 0.0
                            ),
                        }
                rs_meta = meta.get("risk_state") or {}
                regime_label = str(rs_meta.get("regime", "unknown")).lower()
                vix_level = (
                    ctx.market_stress.get("vix_level") if ctx.market_stress else None
                )
                if pos_map and prices_filtered is not None:
                    ts_result = compute_trailing_stops(
                        pos_map,
                        prices_filtered,
                        regime=regime_label,
                        atr_window=int(ts_cfg.get("atr_window", 14)),
                        vix_level=vix_level,
                    )
                    if ts_result.triggered_symbols or ts_result.reduction_symbols:
                        tw_col = (
                            "target_weight"
                            if "target_weight" in target_positions.columns
                            else "weight"
                        )
                        if tw_col in target_positions.columns:
                            weights_map = {
                                str(k).upper(): float(v)
                                for k, v in target_positions.set_index("symbol")[
                                    tw_col
                                ].items()
                            }
                            adjusted = apply_stop_reductions_to_weights(
                                weights_map, ts_result
                            )
                            target_positions[tw_col] = (
                                target_positions["symbol"]
                                .astype(str)
                                .str.upper()
                                .map(adjusted)
                                .fillna(target_positions[tw_col])
                            )
                            if "target_qty" in target_positions.columns:
                                for sym in ts_result.triggered_symbols:
                                    target_positions.loc[
                                        target_positions["symbol"]
                                        .astype(str)
                                        .str.upper()
                                        == sym,
                                        "target_qty",
                                    ] = 0.0
    except Exception as e:
        _record_degraded_step("trailing_stops", e, meta=meta, log_obj=log)
    return target_positions


def _sp_apply_turnover_gate(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    prices_for_sizing: pd.DataFrame | None,
    prices_latest: pd.DataFrame | None,
    policy: dict,
    log: logging.Logger,
    meta: dict,
) -> pd.DataFrame:
    """Turnover budget gate (INT-6)."""
    tb = policy.get("turnover_budget") or {}
    if tb.get("enabled", False) and not target_positions.empty:
        try:
            cap = float(tb.get("cap", 0.15) or 0.15)
            prices_for_turnover = (
                prices_latest
                if prices_latest is not None and not prices_latest.empty
                else prices_for_sizing
            )
            estimated = estimate_turnover(
                ctx.current_positions,
                target_positions,
                prices_for_turnover,
                portfolio_value=ctx.capital,
            )
            _invested_pct = None
            if (
                ctx.capital > 0
                and ctx.current_positions is not None
                and not ctx.current_positions.empty
                and "qty" in ctx.current_positions.columns
            ):
                _price_s = (
                    prices_for_turnover.groupby("symbol")["close"].last()
                    if (
                        prices_for_turnover is not None
                        and not prices_for_turnover.empty
                        and "close" in prices_for_turnover.columns
                    )
                    else pd.Series(dtype=float)
                )
                _cp = ctx.current_positions
                _inv = float(
                    (
                        _cp["qty"].fillna(0).astype(float)
                        * _cp["symbol"].map(_price_s).fillna(0)
                    ).sum()
                )
                _invested_pct = _inv / ctx.capital if ctx.capital else 0.0
            target_positions, _scale = apply_turnover_gate(
                target_positions,
                ctx.current_positions,
                cap=cap,
                estimated_turnover=1.0 if estimated == float("inf") else estimated,
                behavior=(
                    "block"
                    if estimated == float("inf")
                    else str(tb.get("behavior", "scale") or "scale")
                ),
                prices=prices_for_turnover,
                portfolio_value=ctx.capital,
                invested_pct=_invested_pct,
                target_invested_pct=float(tb.get("target_invested_pct", 0.80) or 0.80),
            )
        except Exception as e:
            # W11b (2026-07-22, GESAMTBEWERTUNG; Stage-1 B1 fix: meta is now
            # a real parameter — the first version referenced an undefined
            # name and inverted fail-open into fail-crash): a crashed
            # turnover gate previously vanished at DEBUG while the cycle
            # proceeded UNCAPPED — the CLAUDE.md "stille except-Pfadlogik"
            # class. Degraded-step marker carries the UNCAPPED context; the
            # fail-open direction itself is kept deliberately (flipping to
            # fail-closed here would need an explicit risk sign-off).
            _record_degraded_step(
                "turnover_budget_gate",
                e,
                meta=meta,
                log_obj=log,
                detail="cycle proceeds UNCAPPED (turnover budget not applied)",
            )
    return target_positions


def _sp_apply_correlation_guard(
    target_positions: pd.DataFrame,
    prices_for_sizing: pd.DataFrame | None,
    policy: dict,
    ctx: "TradingContext",
    meta: dict,
) -> pd.DataFrame:
    """Correlation guard (M6-T07) + regime shift exposure scaling."""
    try:
        if (
            not target_positions.empty
            and len(target_positions) >= 2
            and "target_weight" in target_positions.columns
        ):
            tw_dict_cg = dict(
                zip(target_positions["symbol"], target_positions["target_weight"])
            )
            corr_prices = prices_for_sizing
            adjusted_weights, corr_reasons = apply_correlation_guard(
                tw_dict_cg, corr_prices, policy
            )
            if corr_reasons:
                # shadow_recorder restored 2026-07-27 (was archived 13a97b54
                # while the D1-D4 call sites stayed behind; see E-059).
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )

                cg_shadow = is_shadow_only(policy, "correlation_guard")
                record_shadow(
                    "correlation_guard",
                    {"adjusted_weights": adjusted_weights},
                    as_of=str(ctx.as_of) if ctx.as_of else None,
                    meta={"applied": not cg_shadow},
                )
                if not cg_shadow:
                    target_positions["target_weight"] = target_positions["symbol"].map(
                        adjusted_weights
                    )
                    if "target_qty" in target_positions.columns:
                        target_positions["target_qty"] = (
                            target_positions["target_weight"] * ctx.capital
                        )
            symbols_in_portfolio = list(target_positions["symbol"].unique())
            if len(symbols_in_portfolio) >= 2:
                shift_result = detect_correlation_regime_shift(
                    corr_prices, symbols_in_portfolio
                )
                if shift_result.get("regime_shift_detected", False):
                    exp_scale = shift_result["exposure_scale"]
                    target_positions["target_weight"] *= exp_scale
                    if "target_qty" in target_positions.columns:
                        target_positions["target_qty"] *= exp_scale
    except Exception as e:
        _record_degraded_step("correlation_guard", e, meta=meta, log_obj=logger)
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
        if (
            cp_cfg.get("equity_cap_enabled", False)
            and not target_positions.empty
            and "target_weight" in target_positions.columns
        ):
            threshold = float(cp_cfg.get("equity_cap_threshold", 0.4))
            if crash_prob > threshold:
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )

                base_long_gross = float(cp_cfg.get("base_long_gross", 1.0))
                cap_val = max(0.5 - crash_prob, 0.0) * base_long_gross
                long_mask = target_positions["target_weight"] > 0
                current_long_gross = float(
                    target_positions.loc[long_mask, "target_weight"].sum()
                )
                scale = (
                    min(cap_val / current_long_gross, 1.0)
                    if current_long_gross > 1e-9  # Item 47: guard ZeroDivisionError
                    else 1.0
                )
                cp_shadow = is_shadow_only(policy, "crash_prediction")
                record_shadow(
                    "crash_prediction_cap",
                    {"cap": cap_val, "scale": scale},
                    as_of=as_of_str,
                    meta={"applied": (not cp_shadow) and scale < 1.0},
                )
                if not cp_shadow and scale < 1.0:
                    target_positions.loc[long_mask, "target_weight"] *= scale
                    if "target_qty" in target_positions.columns:
                        target_positions.loc[long_mask, "target_qty"] *= scale
    except Exception as e:
        _record_degraded_step("crash_prediction_cap", e, meta=meta, log_obj=logger)
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
            if (
                ctx.prices is not None
                and not ctx.prices.empty
                and "VIX" in ctx.prices.columns
            ):
                try:
                    # B-pipe-1 (latent/defensive): production ctx.prices is
                    # long-format, so this wide "VIX"-column branch does not fire
                    # in production. Slice to the as_of window before the tail
                    # read so a wide panel reaching here in backtest/replay does
                    # not leak future VIX. Live/eod (tail == as_of) byte-identical.
                    _vix_src = ctx.prices
                    _as_of = getattr(ctx, "as_of", None)
                    if _as_of is not None and "timestamp" in _vix_src.columns:
                        _ts = pd.to_datetime(_vix_src["timestamp"], utc=True)
                        _as_of_utc = pd.Timestamp(_as_of)
                        if _as_of_utc.tzinfo is None:
                            _as_of_utc = _as_of_utc.tz_localize("UTC")
                        _vix_src = _vix_src.loc[_ts <= _as_of_utc]
                    if not _vix_src.empty:
                        vix_val = float(_vix_src["VIX"].iloc[-1])
                except Exception as _exc:
                    # E-059 follow-up: a persistent VIX-parse failure silently
                    # disables the whole D4 hedge (outer except never fires) —
                    # same masked-failure class, so WARN once, then DEBUG.
                    _warn_once_sizing_skip(
                        "inverse_etf VIX value parse (D4 gate)", _exc, logger
                    )
            if (
                vix_val is not None
                and vix_val > float(ie_cfg.get("vix_threshold", 25.0))
                and crash_prob > float(ie_cfg.get("crash_prob_threshold", 0.4))
            ):
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )
                from src.assembled_core.portfolio.inverse_etf_selector import (
                    InverseETFSelector,
                )

                selector = InverseETFSelector(allow_2x=False, allow_3x=False)
                hedge_sym = selector.select_best_short_instrument(
                    "BROAD",
                    severity=float(cp_meta.get("severity", 0.5) or 0.5),
                    holding_period_days=int(ie_cfg.get("max_holding_days", 5)),
                )
                hedge_ratio = float(ie_cfg.get("hedge_ratio", 0.1))
                ie_shadow = is_shadow_only(policy, "inverse_etf")
                record_shadow(
                    "inverse_etf",
                    {"hedge_symbol": hedge_sym, "hedge_weight": hedge_ratio},
                    as_of=str(ctx.as_of) if ctx.as_of else None,
                    meta={"applied": (not ie_shadow) and hedge_sym is not None},
                )
                if (
                    not ie_shadow
                    and hedge_sym
                    and "target_weight" in target_positions.columns
                ):
                    # R2-7 (B2-01) scope note: this hedge entry is ADDED after the
                    # global exposure multiplier was applied to the base book, so it
                    # escapes the geo/vol/stress/HMM de-risk chain like crisis_alpha /
                    # news_alpha did. B2-01 names only those two overlays; this third
                    # ADD-overlay is a defensive crash hedge (exempt-by-default is the
                    # correct posture, same as crisis_alpha) and is a documented
                    # follow-up, NOT a silent escape — wiring _apply_overlay_global_derisk
                    # here is deferred to keep this commit scoped to B2-01.
                    if hedge_sym not in target_positions["symbol"].values:
                        target_positions = pd.concat(
                            [
                                target_positions,
                                pd.DataFrame(
                                    [
                                        {
                                            "symbol": hedge_sym,
                                            "target_weight": hedge_ratio,
                                            "target_qty": hedge_ratio * ctx.capital,
                                        }
                                    ]
                                ),
                            ],
                            ignore_index=True,
                        )
    except Exception as e:
        # E-059 follow-up: D4 shadow recording is live — DEBUG-only swallow
        # was its sole visibility. WARN once per signature, stays graceful.
        _warn_once_sizing_skip("inverse_etf hedge (D4)", e, logger)
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
        if (
            qm_cfg.get("enabled", False)
            and not target_positions.empty
            and "target_weight" in target_positions.columns
            and prices_with_features is not None
            and not prices_with_features.empty
        ):
            from src.assembled_core.ml.quantile_models import predict_quantiles

            _feature_cols = qm_cfg.get("feature_cols", [])
            _target_col = qm_cfg.get("target_col", "return_1d")
            if _feature_cols and _target_col in prices_with_features.columns:
                _valid_fcols = [
                    c for c in _feature_cols if c in prices_with_features.columns
                ]
                if _valid_fcols:
                    _qpreds = predict_quantiles(
                        prices_with_features,
                        target_col=_target_col,
                        feature_cols=_valid_fcols,
                    )
                    _asym_map = {qp.symbol: qp.asymmetry for qp in _qpreds}
                    _asym_thresh = float(qm_cfg.get("asymmetry_threshold", 1.5))
                    _asym_red = float(qm_cfg.get("asymmetry_reduction", 0.5))
                    mask = target_positions["symbol"].map(
                        lambda s: _asym_map.get(s, 0.0) > _asym_thresh
                    )
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

            _tw_dict_crowd = dict(
                zip(
                    target_positions["symbol"],
                    target_positions["target_weight"].fillna(0.0),
                )
            )
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


def _apply_overlay_global_derisk(
    overlay_weights: dict,
    overlay_name: str,
    overlay_cfg: dict | None,
    meta: dict | None,
    log: logging.Logger,
) -> dict:
    """R2-7 (audit B2-01): compose the crisis_alpha / news_alpha overlays with the
    global exposure multiplier deterministically and observably.

    ``size_positions`` applies the global multiplier
    (geo × profit_lock × vol_targeting × market_stress × crisis × pm × hmm, clamp
    [0.05, 3.0]) to the BASE book BEFORE these overlays append their entries, so
    the overlay entries escape the whole de-risk/leverage chain. That escape is
    INTENTIONAL for crisis_alpha (a defensive hedge must not be de-risked away
    exactly when it is needed — the de-risk frees the capital the hedge then uses)
    and DEBATABLE for news_alpha (a directional event bet). Before R2-7 the escape
    was silent.

    This helper makes it explicit and opt-in configurable:

    * default (``apply_global_derisk`` absent/false) — behaviour-preserving: the
      weights are returned UNCHANGED, but the escape is recorded in
      ``meta['overlay_exposure']`` and logged at INFO so it is auditable, not
      silent.
    * opt-in (``overlay_cfg['apply_global_derisk'] = true``) — the overlay weights
      are scaled by the same global multiplier so the sub-portfolio composes with
      the system-wide risk appetite. Off by default; deterministic when on.

    ``meta=None`` (ad-hoc / unit-test callers without a cycle meta) → the
    multiplier defaults to 1.0 → no-op, no recording. Never raises; a non-dict
    ``meta`` is treated as absent.
    """
    if not overlay_weights:
        return overlay_weights
    mult = 1.0
    if isinstance(meta, dict):
        try:
            mult = float(meta.get("final_exposure_multiplier", 1.0))
        except (TypeError, ValueError):
            mult = 1.0
    # No global scaling in effect → nothing to compose, stay silent.
    if abs(mult - 1.0) <= 1e-9:
        return overlay_weights
    apply = bool((overlay_cfg or {}).get("apply_global_derisk", False))
    n = len(overlay_weights)
    if apply:
        overlay_weights = {str(k): float(v) * mult for k, v in overlay_weights.items()}
        log.info(
            "[R2-7] %s: applied global exposure multiplier %.3f to %d overlay "
            "entr%s (apply_global_derisk=true)",
            overlay_name,
            mult,
            n,
            "y" if n == 1 else "ies",
        )
    else:
        log.info(
            "[R2-7] %s: %d overlay entr%s sized independently of the global "
            "exposure multiplier %.3f (apply_global_derisk=false) — overlay "
            "exempt from geo/vol/stress/HMM scaling",
            overlay_name,
            n,
            "y" if n == 1 else "ies",
            mult,
        )
    if isinstance(meta, dict):
        meta.setdefault("overlay_exposure", []).append(
            {
                "overlay": overlay_name,
                "global_multiplier": mult,
                "derisk_applied": apply,
                "n_entries": n,
            }
        )
    return overlay_weights


def _sp_apply_crisis_alpha_cap(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    log: logging.Logger,
    *,
    meta: dict | None = None,
) -> pd.DataFrame:
    """T4.1: Crisis Alpha — add defensive/inverse entries + cap existing on overlap.

    When crisis state is ACTIVE and shadow_only=False:
    - New instruments (GLD, TLT, SH, VIXY …) are ADDED to target_positions so
      the execution layer can fill them alongside mfv2 longs.
    - Existing symbols that appear in both ca_tw and target_positions are capped
      (never boosted) to the crisis weight — preserving the original safety contract.
    - The georisk_overlay (run earlier in the pipeline) already reduced long
      weights proportionally, freeing capital for crisis entries.

    Context building (fills CrisisAlphaContext from TradingContext):
    - geo_score: from ctx.news_geo (live intel) or GPR index fallback (backtest).
    - market_stress_ok / market_stress_score: from ctx.market_stress.
    - health_ok: derived from ctx.intel_health_flags.
    - GPR fallback: when news intel geo_score == 0, derives score from the
      gpr_index panel column (GPR > 200 → 2, GPR > 150 → 1). geo_sources is set
      to 2 for GPR-triggered activation (GPR is an institutional-quality index,
      not a social signal).
    """
    if (
        not (policy or {})
        .get("intel", {})
        .get("crisis_alpha", {})
        .get("enabled", False)
    ):
        return target_positions
    try:
        from datetime import datetime, timezone

        from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
        from src.assembled_core.events.crisis_alpha.pipeline import (
            run_crisis_alpha_pipeline,
        )

        # Use pre-built context from ctx.meta if provided (e.g. by tests or workers)
        _ca_ctx = ctx.meta.get("crisis_alpha_ctx") if hasattr(ctx, "meta") else None
        if _ca_ctx is None:
            _as_of_dt = (
                pd.to_datetime(ctx.as_of, utc=True).to_pydatetime()
                if getattr(ctx, "as_of", None) is not None
                else datetime.now(timezone.utc)
            )

            # --- geo_score + triggers from live intel ---
            _news_geo = getattr(ctx, "news_geo", None) or {}
            _geo_score = float(_news_geo.get("geo_score", 0.0))
            _geo_sources = int(
                len(_news_geo.get("active_triggers", []))
                or (1 if _geo_score > 0 else 0)
            )
            _social_only = bool(_news_geo.get("social_only", False))
            # Wire news_trigger_items for evidence gate: try multiple key names from live intel
            _news_trigger_items: list[dict] = list(
                _news_geo.get("news_trigger_items")
                or _news_geo.get("triggers")
                or _news_geo.get("active_triggers")
                or []
            )

            # --- GPR fallback for backtests (no live intel) ---
            # F-CA-005: only activate when BOTH geo_score and live triggers are absent.
            # Prevents GPR from overriding a live intel "no crisis" judgment.
            if _geo_score == 0.0 and not _news_trigger_items:
                # Source the GPR time series.  Two paths:
                # 1. ctx.features (live/paper — TradingContext exposes enriched panel)
                # 2. Direct parquet read (backtest — TradingContext does NOT store
                #    prices_with_features as ctx.features; this is the primary path).
                _gpr_s: pd.Series = pd.Series(dtype=float)
                _feat = getattr(ctx, "features", None)
                if _feat is not None and "gpr_index" in _feat.columns:
                    _gpr_s = pd.to_numeric(_feat["gpr_index"], errors="coerce").dropna()
                    # PIT guard: when features has a DatetimeIndex, restrict to rows
                    # at or before as_of so we never read future GPR data.
                    if isinstance(_feat.index, pd.DatetimeIndex) and len(_gpr_s) > 0:
                        try:
                            _cutoff = pd.Timestamp(_as_of_dt)
                            _idx_tz = _gpr_s.index.tz
                            if _cutoff.tzinfo is None and _idx_tz is not None:
                                _cutoff = _cutoff.tz_localize("UTC").tz_convert(_idx_tz)
                            elif _cutoff.tzinfo is not None and _idx_tz is None:
                                _cutoff = _cutoff.tz_localize(None)
                            elif _cutoff.tzinfo is not None and _idx_tz is not None:
                                _cutoff = _cutoff.tz_convert(_idx_tz)
                            _gpr_s = _gpr_s[_gpr_s.index <= _cutoff]
                        except Exception as _tz_exc:
                            log.warning(
                                "[T4.1] GPR PIT tz-guard failed (%s) — "
                                "zeroing GPR series to prevent look-ahead",
                                _tz_exc,
                            )
                            _gpr_s = pd.Series(dtype=float)
                else:
                    # Backtest direct-read: load GPR parquet once (module-level cache),
                    # then filter to rows at or before as_of (PIT-safe).
                    try:
                        _gpr_cfg = (policy or {}).get("features", {}).get(
                            "macro_gpr"
                        ) or {}
                        if _gpr_cfg.get("enabled", True):
                            _gpr_path = str(
                                _gpr_cfg.get("path", "output/macro_gpr.parquet")
                            )
                            if _gpr_path not in _GPR_PARQUET_CACHE:
                                try:
                                    _gpr_df_loaded = pd.read_parquet(
                                        _gpr_path, columns=["timestamp", "gpr_index"]
                                    )
                                    _GPR_PARQUET_CACHE[_gpr_path] = _gpr_df_loaded
                                    log.debug(
                                        "[T4.1] GPR cache loaded: %s (%d rows)",
                                        _gpr_path,
                                        len(_gpr_df_loaded),
                                    )
                                except Exception as _load_exc:
                                    log.warning(
                                        "[T4.1] GPR parquet load failed path=%s err=%s",
                                        _gpr_path,
                                        _load_exc,
                                    )
                                    _GPR_PARQUET_CACHE[_gpr_path] = None
                            _gpr_df = _GPR_PARQUET_CACHE.get(_gpr_path)
                            if _gpr_df is not None and not _gpr_df.empty:
                                _gpr_ts = pd.to_datetime(_gpr_df["timestamp"], utc=True)
                                _cutoff = pd.Timestamp(_as_of_dt)
                                if _cutoff.tzinfo is None:
                                    _cutoff = _cutoff.tz_localize("UTC")
                                else:
                                    _cutoff = _cutoff.tz_convert("UTC")
                                # Apply same release lag as merge_gpr_index_into_panel
                                # (monthly index published ~32 days after period end).
                                _release_lag = int(_gpr_cfg.get("release_lag_days", 32))
                                _available_cutoff = _cutoff - pd.Timedelta(
                                    days=_release_lag
                                )
                                _gpr_s = pd.to_numeric(
                                    _gpr_df.loc[
                                        _gpr_ts <= _available_cutoff, "gpr_index"
                                    ],
                                    errors="coerce",
                                ).dropna()
                    except Exception as _gpr_exc:
                        log.debug("[T4.1] GPR direct read skipped: %s", _gpr_exc)

                # Use most recent available value (not mean — avoids cross-row averaging).
                _gpr_val = float(_gpr_s.iloc[-1] if len(_gpr_s) > 0 else 0.0)
                if _gpr_val > 200:
                    _geo_score = 2.0
                    # GPR is an institutional-quality single-source index. Setting
                    # geo_sources=2 lets the min_sources=2 gate pass by design —
                    # GPR has confirmed cross-crisis reliability unlike social signals.
                    _geo_sources = 2
                    # CR-001 fix: provide synthetic trigger so evidence gate passes.
                    # GPR index IS derived from news-article counts — the trigger is real.
                    _news_trigger_items = [
                        {
                            "severity": 2,
                            "topic": "gpr_index",
                            "source": "Caldara-Iacoviello",
                        }
                    ]
                    log.debug(
                        "[T4.1] GPR fallback: gpr=%.1f -> geo_score=2.0", _gpr_val
                    )
                elif _gpr_val > 150:
                    _geo_score = 1.0
                    _geo_sources = 2
                    _news_trigger_items = [
                        {
                            "severity": 1,
                            "topic": "gpr_index",
                            "source": "Caldara-Iacoviello",
                        }
                    ]
                    log.debug(
                        "[T4.1] GPR fallback: gpr=%.1f -> geo_score=1.0", _gpr_val
                    )

            # --- market stress ---
            # Default True (pass-through) when absent — mirrors health_ok convention.
            # Live/paper will supply a real stress dict; backtest has none, so we must
            # not veto GPR-driven crisis activation when no stress signal exists.
            _ms = getattr(ctx, "market_stress", None) or {}
            _market_stress_ok = bool(_ms.get("stress_ok", True))
            _market_stress_score = int(_ms.get("stress_score", 0))

            # --- health ---
            _hflags = getattr(ctx, "intel_health_flags", {}) or {}
            _health_ok = not any(v == "ERROR" for v in _hflags.values())

            # --- daily loss guard and open positions ---
            # daily_pnl wired from ctx.meta["crisis_daily_pnl"] if available.
            # Defaults to 0.0 (guard never fires) — safe-side until full wiring.
            _meta = getattr(ctx, "meta", {}) or {}
            _daily_pnl = float(_meta.get("crisis_daily_pnl", 0.0))
            # F-CA-001: policy nests crisis_alpha under intel, not at top level.
            _ca_cfg = (policy or {}).get("intel", {}).get("crisis_alpha") or {}
            _daily_loss_limit = float(
                (_ca_cfg.get("daily_loss") or {}).get("limit", 0.02)
            )
            _open_positions = list(_meta.get("crisis_open_positions", []) or [])

            _ca_ctx = CrisisAlphaContext(
                timestamp_utc=_as_of_dt,
                geo_score=_geo_score,
                geo_sources=_geo_sources,
                social_only=_social_only,
                market_stress_ok=_market_stress_ok,
                market_stress_score=_market_stress_score,
                health_ok=_health_ok,
                news_trigger_items=_news_trigger_items,
                daily_pnl=_daily_pnl,
                daily_loss_limit=_daily_loss_limit,
                open_positions=_open_positions,
            )

        shadow_only = (
            (policy or {})
            .get("intel", {})
            .get("crisis_alpha", {})
            .get("shadow_only", True)
        ) or (os.environ.get("ASSEMBLED_NO_CRISIS_OVERLAY") == "1")
        # Pass PIT-filtered prices for regime-aware basket selection.
        # ctx.prices is the full dataset; as_of filtering mirrors the _load_intel PIT guard.
        _ca_prices_df: object = None
        _ca_as_of = getattr(ctx, "as_of", None)
        _ctx_prices = getattr(ctx, "prices", None)
        if (
            _ca_as_of is not None
            and _ctx_prices is not None
            and not getattr(_ctx_prices, "empty", True)
        ):
            try:
                import pandas as _pd

                _ca_prices_df = _ctx_prices[
                    _pd.to_datetime(_ctx_prices["timestamp"], utc=True)
                    <= _pd.to_datetime(_ca_as_of, utc=True)
                ]
            except Exception as _regime_exc:
                log.warning(
                    "[T4.1] PIT price filter for regime detection failed (%s) — regime detection disabled",
                    _regime_exc,
                )
                _ca_prices_df = None
        ca_result = run_crisis_alpha_pipeline(
            _ca_ctx, policy=policy, dry_run=shadow_only, prices_df=_ca_prices_df
        )
        # §9.13 visibility: log flatten/exit commands so silent discard is observable.
        # Gated on not shadow_only — backtest/paper-shadow noise would mask real alerts.
        # Actual flatten execution is NOT wired anywhere — §9.13 deferred.
        if not shadow_only:
            if ca_result.get("should_flatten_all") and _open_positions:
                log.warning(
                    "[T4.1] crisis_alpha: should_flatten_all=True, %d open positions will NOT "
                    "be flattened (§9.13 deferred; no consumer exists yet).",
                    len(_open_positions),
                )
            _exits = ca_result.get("positions_to_exit") or []
            if _exits:
                _exit_syms = [
                    p[0].get("symbol", "?")
                    if isinstance(p, (list, tuple)) and p and isinstance(p[0], dict)
                    else (p.get("symbol", "?") if isinstance(p, dict) else str(p))
                    for p in _exits[:20]
                ]
                log.warning(
                    "[T4.1] crisis_alpha: %d positions_to_exit flagged (not consumed) — symbols=%s%s",
                    len(_exits),
                    _exit_syms,
                    " …" if len(_exits) > 20 else "",
                )
        _errs = ca_result.get("errors") or []
        if _errs:
            log.warning(
                "[T4.1] crisis_alpha: pipeline returned %d error(s): %s",
                len(_errs),
                _errs[:5],
            )
        if not shadow_only and ca_result.get("target_weights"):
            ca_tw = ca_result["target_weights"]
            # F-NEW-002: capital needed before cap loop to keep target_qty in sync.
            _capital = float(getattr(ctx, "capital", 0.0))

            # R2-7 (B2-01): compose with the global exposure multiplier. Default
            # is behaviour-preserving (crisis hedges exempt + recorded); opt-in
            # apply_global_derisk folds the de-risk in. Scaling ca_tw here keeps
            # both the min-merge cap and the ADD entries (+ their target_qty,
            # derived from these weights below) consistent.
            ca_tw = _apply_overlay_global_derisk(
                ca_tw,
                "crisis_alpha",
                (policy or {}).get("intel", {}).get("crisis_alpha", {}),
                meta,
                log,
            )

            # Cap overlapping symbols (never increase — original safety contract).
            # F-NEW-001: only when target_positions is non-empty; ADD-entries block
            # runs unconditionally so crisis instruments reach orders on flat days.
            existing_syms: set[str] = set()
            n_capped = 0
            if (
                not target_positions.empty
                and "target_weight" in target_positions.columns
            ):
                existing_syms = set(target_positions["symbol"].astype(str).str.upper())
                for idx, row in target_positions.iterrows():
                    sym = str(row["symbol"]).upper()
                    if sym in ca_tw:
                        old_w = float(row["target_weight"])
                        new_w = min(old_w, float(ca_tw[sym]))
                        if new_w < old_w:
                            target_positions.at[idx, "target_weight"] = new_w
                            # F-NEW-002: sync target_qty with capped weight.
                            # round(..., 2) matches the convention used by upstream
                            # position-sizing functions throughout this pipeline.
                            if "target_qty" in target_positions.columns:
                                target_positions.at[idx, "target_qty"] = round(
                                    new_w * _capital, 2
                                )
                            n_capped += 1

            # ADD crisis entries not already in target_positions (core new behavior).
            # target_qty is set here so order_generation never sees NaN → 0 delta.
            # round(..., 2) matches the target_qty convention used elsewhere.
            new_rows = [
                {
                    "symbol": sym,
                    "target_weight": float(w),
                    "target_qty": round(float(w) * _capital, 2),
                }
                for sym, w in ca_tw.items()
                if sym.upper() not in existing_syms
            ]
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                # When target_positions is empty, replace it with a bare DataFrame
                # (no columns) before concat. This avoids the FutureWarning about
                # all-NA column dtype inference while preserving any non-empty rows
                # exactly as-is (the `else` branch keeps the original frame intact,
                # including any metadata columns like "side" or "signal_score").
                base = pd.DataFrame() if target_positions.empty else target_positions
                target_positions = pd.concat(
                    [base, new_df],
                    ignore_index=True,
                )
                log.info(
                    "[T4.1] crisis_alpha ACTIVE: added %d crisis positions %s | capped %d",
                    len(new_rows),
                    [r["symbol"] for r in new_rows],
                    n_capped,
                )
            elif n_capped:
                log.info("[T4.1] crisis_alpha: capped %d existing positions", n_capped)

            # Gross-exposure guard: renormalize if crisis entries push combined
            # portfolio above risk_limits.max_gross_exposure.
            _max_gross = float(
                (policy or {}).get("risk_limits", {}).get("max_gross_exposure", 1.20)
            )
            _total_abs = target_positions["target_weight"].abs().sum()
            if _total_abs > _max_gross and _total_abs > 0:
                _scale = _max_gross / _total_abs
                target_positions["target_weight"] = (
                    target_positions["target_weight"] * _scale
                )
                # F-CA-002: recompute target_qty from the already-scaled weight
                # rather than scaling the old value. This handles rows where
                # target_qty was NaN (fillna(0.0)*scale would silently zero them).
                if "target_qty" in target_positions.columns:
                    target_positions["target_qty"] = (
                        target_positions["target_weight"] * _capital
                    ).round(2)
                log.info(
                    "[T4.1] gross-exposure guard: %.2f > max %.2f — scaled by %.3f",
                    _total_abs,
                    _max_gross,
                    _scale,
                )

    except Exception as exc:
        log.error(
            "[T4.1] crisis_alpha_pipeline failed — returning unmodified targets: %s",
            exc,
            exc_info=True,
        )
    return target_positions


def _sp_apply_news_alpha(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    log: logging.Logger,
    *,
    meta: dict | None = None,
) -> pd.DataFrame:
    """T4.2: News Alpha — add directional event-driven entries to target positions.

    Mirrors _sp_apply_crisis_alpha_cap in structure but serves a different purpose:
    - crisis_alpha: slow defensive MDD-reduction basket (weeks, sector hedges)
    - news_alpha: fast directional alpha (days, event-specific ETFs e.g. XLE on Hormuz)

    Architecture note — state ownership:
    The intraday runner (scripts/run_news_alpha_intraday.py) owns the open signal
    lifecycle: it polls every 5-10 min, fires entries at market price, and monitors
    exits against live prices. open_signals=[] here is intentional — the EOD sizing
    cycle uses this function to apply any *new* trigger items arriving at EOD (e.g.
    after-hours RSS), NOT to duplicate the intraday runner's position management.
    open_signals is always empty here. The intraday runner manages position lifecycle
    exclusively via output/news_alpha_state.json; there is no state hand-off to the
    EOD cycle. In backtest mode there are no intraday runner positions by definition.

    Trigger items: sourced from ctx.news_geo["news_trigger_items"] — same live-intel
    path as crisis_alpha. No GPR fallback: news_alpha requires a concrete news event.

    Cap semantics (DESIGN DECISION — conservative first-pass):
    Overlapping symbols use min-merge (same as crisis_alpha). If mfv2 holds XLE at
    0.30 and news_alpha generates XLE at 0.08, the combined weight is 0.08 (capped
    down, not boosted). Rationale: news_alpha signal may confirm the same thesis as
    mfv2 but was sized for the news_alpha sub-portfolio (max 0.40 gross). Boosting
    to max() would silently exceed the intended sub-portfolio exposure. Once paper
    validation confirms the signal quality, switch to max()-merge for same-direction
    confirming signals (update this function and the test in
    test_trading_cycle_news_alpha.py::TestT42CapNeverBoosts).

    Policy path: policy["news_alpha"] (top-level) vs. crisis_alpha's nested
    policy["intel"]["crisis_alpha"]. Deliberate: news_alpha is a standalone module,
    not part of the broader intel/crisis overlay hierarchy.

    When shadow_only=True: signals generated and logged, NOT applied to positions.
    shadow_only=False is now active (set in policy.yaml 2026-05-26). To disable,
    flip policy.yaml news_alpha.shadow_only back to true without code changes.
    """
    if not (policy or {}).get("news_alpha", {}).get("enabled", False):
        return target_positions
    try:
        from datetime import datetime, timezone

        from src.assembled_core.events.news_alpha.pipeline import (
            run_news_alpha_pipeline,
        )

        _as_of_dt = (
            pd.to_datetime(ctx.as_of, utc=True).to_pydatetime()
            if getattr(ctx, "as_of", None) is not None
            else datetime.now(timezone.utc)
        )

        # --- trigger_items from live intel ---
        _news_geo = getattr(ctx, "news_geo", None) or {}
        _trigger_items: list[dict] = list(
            _news_geo.get("news_trigger_items")
            or _news_geo.get("triggers")
            or _news_geo.get("active_triggers")
            or []
        )

        # --- price lookup for stop/tp evaluation (PIT-safe: <= as_of) ---
        # Failure → WARNING so operator can see the degradation; exits fall back
        # to time-based only (check_exits() handles missing prices gracefully).
        _prices: dict[str, float] = {}
        _ctx_prices = getattr(ctx, "prices", None)
        if _ctx_prices is not None and not getattr(_ctx_prices, "empty", True):
            try:
                _cutoff = pd.to_datetime(_as_of_dt, utc=True)
                _pit = _ctx_prices[
                    pd.to_datetime(_ctx_prices["timestamp"], utc=True) <= _cutoff
                ]
                if "close" in _pit.columns and "symbol" in _pit.columns:
                    _prices = (
                        _pit.sort_values("timestamp")
                        .groupby("symbol")["close"]
                        .last()
                        .to_dict()
                    )
            except Exception as _price_exc:
                log.warning(
                    "[T4.2] price lookup failed — stop/tp exits will use time-based only: %s",
                    _price_exc,
                )

        # --- open_signals and day counter from ctx.meta ---
        # Populated by the intraday runner or EOD worker; empty in backtest/shadow.
        _meta = getattr(ctx, "meta", {}) or {}
        _open_signals = list(_meta.get("news_alpha_open_signals", []) or [])
        _day_counter = int(_meta.get("news_alpha_day_counter", 0))

        shadow_only = bool(
            (policy or {}).get("news_alpha", {}).get("shadow_only", True)
        )

        na_result = run_news_alpha_pipeline(
            trigger_items=_trigger_items,
            open_signals=_open_signals,
            current_day=_day_counter,
            prices=_prices,
            policy=policy,
            shadow_only=shadow_only,
            timestamp_utc=_as_of_dt,
        )

        _errs = na_result.errors or []
        if _errs:
            log.warning(
                "[T4.2] news_alpha: pipeline returned %d error(s): %s",
                len(_errs),
                _errs[:5],
            )

        # Log exits for visibility (same pattern as crisis_alpha §9.13).
        # Actual exit order generation for intraday positions is the intraday
        # runner's responsibility; the EOD cycle logs but does not consume exits.
        _exits = na_result.positions_to_exit or []
        if _exits:
            _exit_syms = [
                sig.symbol for sig, _ in _exits[:20] if hasattr(sig, "symbol")
            ]
            if not shadow_only:
                log.warning(
                    "[T4.2] news_alpha: %d position(s) flagged for exit (not consumed by EOD cycle) "
                    "— intraday runner must process: %s%s",
                    len(_exits),
                    _exit_syms,
                    " …" if len(_exits) > 20 else "",
                )
            else:
                log.debug(
                    "[T4.2] news_alpha shadow exits (not consumed): %d — %s",
                    len(_exits),
                    _exit_syms,
                )

        if not shadow_only and na_result.target_weights:
            # Normalize keys to uppercase to match target_positions symbol convention.
            na_tw = {k.upper(): v for k, v in na_result.target_weights.items()}

            # R2-7 (B2-01): compose with the global exposure multiplier. Default is
            # behaviour-preserving (event entries sized independently of the de-risk
            # chain + recorded in meta); opt-in apply_global_derisk folds it in.
            # Scaling na_tw here keeps both the cap loop and the ADD entries (+ their
            # target_qty, derived from these weights below) consistent.
            na_tw = _apply_overlay_global_derisk(
                na_tw,
                "news_alpha",
                (policy or {}).get("news_alpha", {}),
                meta,
                log,
            )

            _capital = float(getattr(ctx, "capital", 0.0))
            if _capital <= 0.0:
                log.warning(
                    "[T4.2] capital=%.2f — target_qty will be 0 for all news_alpha entries; "
                    "check ctx.capital wiring",
                    _capital,
                )

            existing_syms: set[str] = set()
            n_capped = 0
            if (
                not target_positions.empty
                and "target_weight" in target_positions.columns
            ):
                existing_syms = set(target_positions["symbol"].astype(str).str.upper())
                for idx, row in target_positions.iterrows():
                    sym = str(row["symbol"]).upper()
                    if sym in na_tw:
                        old_w = float(row["target_weight"])
                        # min-merge: conservative cap (see DESIGN DECISION in docstring)
                        new_w = min(old_w, float(na_tw[sym]))
                        if new_w < old_w:
                            target_positions.at[idx, "target_weight"] = new_w
                            if "target_qty" in target_positions.columns:
                                target_positions.at[idx, "target_qty"] = round(
                                    new_w * _capital, 2
                                )
                            n_capped += 1

            new_rows = [
                {
                    "symbol": sym,
                    "target_weight": float(w),
                    "target_qty": round(float(w) * _capital, 2),
                }
                for sym, w in na_tw.items()
                if sym.upper() not in existing_syms
            ]
            if new_rows:
                new_df = pd.DataFrame(new_rows)
                base = pd.DataFrame() if target_positions.empty else target_positions
                target_positions = pd.concat([base, new_df], ignore_index=True)
                log.info(
                    "[T4.2] news_alpha ACTIVE: added %d positions %s | capped %d",
                    len(new_rows),
                    [r["symbol"] for r in new_rows],
                    n_capped,
                )
            elif n_capped:
                log.info("[T4.2] news_alpha: capped %d existing positions", n_capped)

            # Gross-exposure guard (combined portfolio cap).
            # NOTE: signals_to_weights() already enforces policy["news_alpha"]["max_gross_exposure"]
            # (0.40 default) on the news_alpha sub-portfolio before this point.
            # This guard enforces the global risk_limits cap on the COMBINED portfolio.
            # The two caps are complementary: sub-portfolio cap limits news_alpha slice;
            # global cap limits aggregate exposure including mfv2 + crisis_alpha + news_alpha.
            _max_gross = float(
                (policy or {}).get("risk_limits", {}).get("max_gross_exposure", 1.20)
            )
            _total_abs = target_positions["target_weight"].abs().sum()
            if _total_abs > _max_gross and _total_abs > 0:
                _scale = _max_gross / _total_abs
                target_positions["target_weight"] = (
                    target_positions["target_weight"] * _scale
                )
                if "target_qty" in target_positions.columns:
                    target_positions["target_qty"] = (
                        target_positions["target_weight"] * _capital
                    ).round(2)
                log.info(
                    "[T4.2] gross-exposure guard: %.2f > max %.2f — scaled by %.3f",
                    _total_abs,
                    _max_gross,
                    _scale,
                )
        elif shadow_only and na_result.target_weights:
            log.info(
                "[T4.2] news_alpha shadow_only=True — %d targets NOT applied: %s",
                len(na_result.target_weights),
                {s: f"{w:+.3f}" for s, w in na_result.target_weights.items()},
            )

    except Exception as exc:
        log.error(
            "[T4.2] news_alpha_pipeline failed — returning unmodified targets: %s",
            exc,
            exc_info=True,
        )
    return target_positions


def _sp_check_rebalance(
    target_positions: pd.DataFrame,
    ctx: "TradingContext",
    policy: dict,
    meta: dict,
    log: logging.Logger,
) -> tuple[bool, str]:
    """Step 4.5: Rebalance trigger check."""
    vol_regime_changed = bool(
        meta.get("vol_targeting", {}).get("regime_changed", False)
    )
    corr_spiked = bool(
        meta.get("correlation_regime_shift", {}).get("exposure_scale", 1.0) < 1.0
    )
    dd_pct = meta.get("drawdown_pct")
    current_w: dict[str, float] = {}
    if hasattr(ctx, "current_positions") and ctx.current_positions is not None:
        if isinstance(ctx.current_positions, dict):
            current_w = ctx.current_positions
        elif (
            isinstance(ctx.current_positions, pd.DataFrame)
            and "symbol" in ctx.current_positions.columns
        ):
            _cp = ctx.current_positions
            _wcol = "weight" if "weight" in _cp.columns else "target_weight"
            if _wcol in _cp.columns:
                current_w.update(
                    {
                        str(k): float(v)
                        for k, v in _cp.set_index("symbol")[_wcol].fillna(0.0).items()
                    }
                )
    return should_rebalance(
        ctx,
        target_positions,
        current_weights=current_w,
        weight_drift_threshold=float(
            policy.get("rebalancing", {}).get("weight_drift_threshold", 0.05)
        ),
        vol_regime_change=vol_regime_changed,
        corr_spike=corr_spiked,
        scheduled=True,
        drawdown_pct=float(dd_pct) if dd_pct is not None else None,
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
            from src.assembled_core.portfolio.cost_aware_wrapper import (
                apply_cost_aware_from_policy,
            )

            w_col = next(
                (
                    c
                    for c in ("target_weight", "weight", "target_pct")
                    if c in target_positions.columns
                ),
                None,
            )
            if w_col and "symbol" in target_positions.columns:
                _target_w = {
                    str(k): float(v)
                    for k, v in target_positions.dropna(subset=[w_col])
                    .set_index("symbol")[w_col]
                    .items()
                }
                _curr_w_caw: dict[str, float] = {}
                if (
                    ctx.current_positions is not None
                    and not ctx.current_positions.empty
                    and "symbol" in ctx.current_positions.columns
                    and w_col in ctx.current_positions.columns
                ):
                    _curr_w_caw = {
                        str(k): float(v)
                        for k, v in ctx.current_positions.dropna(subset=[w_col])
                        .set_index("symbol")[w_col]
                        .items()
                    }
                _adj_w, _caw_reasons = apply_cost_aware_from_policy(
                    _target_w,
                    _curr_w_caw,
                    policy,
                    current_invested_pct=float(sum(abs(v) for v in _target_w.values())),
                )
                if _caw_reasons:
                    target_positions = target_positions.copy()
                    target_positions[w_col] = target_positions["symbol"].map(
                        lambda s: _adj_w.get(str(s), _target_w.get(str(s), 0.0))
                    )
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

    meta: dict = {}
    try:
        policy = load_policy()
    except Exception as _policy_exc:
        policy = {}
        _record_degraded_step(
            "size_positions_policy_load", _policy_exc, meta=meta, log_obj=log
        )

    prices_for_sizing = prices_filtered if prices_filtered is not None else ctx.prices

    sizing_cfg = policy.get("position_sizing") or {}
    target_positions = _sp_dispatch_sizing(
        signals, ctx, prices_for_sizing, sizing_cfg, log
    )
    meta["sizing_method"] = sizing_cfg.get("method", "default")

    target_positions = _sp_apply_liquidity(
        target_positions, prices_for_sizing, policy, log
    )

    final_multiplier = _sp_compute_final_multiplier(ctx, policy, meta, log)
    # R2-7 (B2-01): publish the global exposure multiplier so the crisis_alpha /
    # news_alpha overlays (applied AFTER the base book below) can compose with it
    # deterministically and observably instead of silently escaping the de-risk chain.
    meta["final_exposure_multiplier"] = float(final_multiplier)
    if abs(final_multiplier - 1.0) > 1e-9 and not target_positions.empty:
        _max_gross = policy.get("risk_limits", {}).get("max_gross_exposure", 1.0)
        target_positions = apply_exposure_multiplier_to_targets(
            target_positions,
            multiplier=final_multiplier,
            cash_symbol="CASH",
            max_gross_exposure=_max_gross,
        )
        # Per-symbol weight re-clamp: only when upscaling (multiplier > 1) since upscaling can push
        # individual weights above max_position_weight; downscaling is already safe.
        if (
            final_multiplier > 1.0
            and not target_positions.empty
            and "target_weight" in target_positions.columns
        ):
            _max_pos_w = float(
                policy.get("risk_limits", {}).get("max_position_weight", 0.20)
            )
            _is_cash = target_positions.get("symbol", target_positions.index) == "CASH"
            target_positions.loc[~_is_cash, "target_weight"] = target_positions.loc[
                ~_is_cash, "target_weight"
            ].clip(lower=-_max_pos_w, upper=_max_pos_w)

    target_positions = _sp_apply_factor_risk(
        target_positions, prices_for_sizing, policy, log
    )
    target_positions = _sp_apply_trailing_stops(
        target_positions, ctx, prices_filtered, policy, meta, log
    )
    target_positions = _sp_apply_turnover_gate(
        target_positions, ctx, prices_for_sizing, prices_latest, policy, log, meta
    )
    target_positions = _sp_apply_correlation_guard(
        target_positions, prices_for_sizing, policy, ctx, meta
    )
    target_positions = _sp_apply_crash_cap(
        target_positions, policy, meta, str(ctx.as_of) if ctx.as_of else None
    )
    target_positions = _sp_apply_inverse_etf(target_positions, ctx, policy, meta)
    target_positions = _sp_apply_quantile_asymmetry(
        target_positions, prices_with_features, policy, log
    )
    target_positions = _sp_apply_crowding_cap(target_positions, ctx, log)
    target_positions = _sp_apply_crisis_alpha_cap(
        target_positions, ctx, policy, log, meta=meta
    )
    target_positions = _sp_apply_news_alpha(
        target_positions, ctx, policy, log, meta=meta
    )

    do_rebal, rebal_reason = _sp_check_rebalance(
        target_positions, ctx, policy, meta, log
    )
    if not do_rebal:
        log.info("REBALANCE SKIPPED: %s — no orders generated", rebal_reason)
        # Override: force rebalance if current positions have symbols not in target
        # (stop-loss / take-profit exits must execute regardless of drift threshold)
        if (
            ctx.current_positions is not None
            and not ctx.current_positions.empty
            and "symbol" in ctx.current_positions.columns
        ):
            try:
                _cur_qty = (
                    ctx.current_positions.get("qty")
                    if hasattr(ctx.current_positions, "get")
                    else ctx.current_positions.get("qty", None)
                )
                _cur_df = ctx.current_positions
                if "qty" in _cur_df.columns:
                    _held = set(
                        _cur_df.loc[_cur_df["qty"].abs() > 1e-6, "symbol"].astype(str)
                    )
                else:
                    _held = set(_cur_df["symbol"].astype(str))
                _tgt_syms = (
                    set(target_positions["symbol"].astype(str))
                    if not target_positions.empty
                    else set()
                )
                _exits_needed = _held - _tgt_syms
                if _exits_needed:
                    do_rebal = True
                    log.info(
                        "[size_positions] exit override: rebalance forced for %d position(s) not in target: %s",
                        len(_exits_needed),
                        sorted(_exits_needed),
                    )
            except Exception as _e:
                log.debug("[size_positions] exit override check failed: %s", _e)

    target_positions = _sp_apply_cost_aware(target_positions, ctx, policy, log)

    # --- Quantile-interval position sizing overlay (LightGBM q10/q90 models) ---
    try:
        conf_cfg = (policy.get("position_sizing") or {}).get("conformal") or {}
        if (
            conf_cfg.get("enabled", False)
            and not target_positions.empty
            and prices_with_features is not None
        ):
            import joblib as _jl
            from pathlib import Path as _Path

            _default_conf = "models/conformal_position_v3.joblib"
            _conf_rel = conf_cfg.get("model_path", _default_conf)
            _conf_path = _Path(__file__).parents[3] / _conf_rel
            if not _conf_path.exists():
                # fall back to v2 if v3 not present
                _conf_path = (
                    _Path(__file__).parents[3]
                    / "models"
                    / "conformal_position_v3.joblib"
                )
            if not _conf_path.exists():
                _conf_path = (
                    _Path(__file__).parents[3]
                    / "models"
                    / "conformal_position_v2.joblib"
                )
            if _conf_path.exists():
                _conf_bundle = _jl.load(_conf_path)
                _conf_feat_cols = _conf_bundle["feature_cols"]
                _med_width = float(_conf_bundle.get("median_interval_width", 0.05))
                _latest_f = (
                    (prices_with_features.groupby("symbol").last().reset_index())
                    if "symbol" in prices_with_features.columns
                    else prices_with_features
                )

                # Translate legacy conformal feature names to panel-native names.
                # Models trained before panel naming was standardised used short names;
                # panel uses prefixed names (ta_*, rv_*). Map where semantics match exactly.
                _CONF_NAME_MAP = {
                    "rsi_14": "ta_rsi_14_v1",
                    "macd_hist": "ta_macd_hist_v1",
                    "bb_pos": "ta_bb_pctb_v1",
                    "bb_width": "ta_bb_bandwidth_v1",
                    "atr_norm": "ta_atr_14_v1",
                    "vol_20d": "rv_20",
                    "vol_zscore_20": "volume_zscore",
                }
                _lf_work = _latest_f.copy()
                for _old, _new in _CONF_NAME_MAP.items():
                    if _old not in _lf_work.columns and _new in _lf_work.columns:
                        _lf_work[_old] = _lf_work[_new]

                _avail = [c for c in _conf_feat_cols if c in _lf_work.columns]
                if len(_avail) >= 5 and "symbol" in _lf_work.columns:
                    # Keep DataFrame (not ndarray) so LightGBM skips feature-name warning
                    _X_conf_df = (
                        _lf_work.set_index("symbol")[_avail]
                        .reindex(columns=_conf_feat_cols, fill_value=0.0)
                        .astype(float)
                    )
                    _X_conf = _X_conf_df  # pass DataFrame to preserve feature names
                    _mtype = _conf_bundle.get("model_type", "")
                    if _mtype == "QuantileRegressionInterval_v2":
                        # v2/v3: q05/q95 models for >=87% coverage
                        _q_lo = _conf_bundle["q05_model"].predict(_X_conf)
                        _q_hi = _conf_bundle["q95_model"].predict(_X_conf)
                        _raw = _q_hi - _q_lo
                        _n_inv = int((_raw < 0).sum())
                        if _n_inv > 0:
                            log.warning(
                                "[CONFORMAL] %d/%d symbols have inverted quantile intervals (q05>q95) — model may have distribution shift",
                                _n_inv,
                                len(_raw),
                            )
                        _widths = _raw.clip(1e-8)
                    elif _mtype == "QuantileRegressionInterval":
                        # v1 legacy: q10/q90 models
                        _q_lo = _conf_bundle["q10_model"].predict(_X_conf)
                        _q_hi = _conf_bundle["q90_model"].predict(_X_conf)
                        _raw = _q_hi - _q_lo
                        _n_inv = int((_raw < 0).sum())
                        if _n_inv > 0:
                            log.warning(
                                "[CONFORMAL] %d/%d symbols have inverted quantile intervals (q10>q90) — model may have distribution shift",
                                _n_inv,
                                len(_raw),
                            )
                        _widths = _raw.clip(1e-8)
                    else:
                        _, _intervals = _conf_bundle["model"].predict_interval(
                            _X_conf_df.values
                        )
                        _widths = (_intervals[:, 1, 0] - _intervals[:, 0, 0]).clip(1e-8)
                    # Anchor to runtime cross-section median to avoid train/test
                    # distribution shift (test widths often differ from train).
                    _runtime_med = (
                        float(np.median(_widths)) if len(_widths) >= 3 else _med_width
                    )
                    _anchor = _runtime_med if _runtime_med > 0 else _med_width
                    _size_mult = (_anchor / _widths).clip(0.25, 2.0)
                    _mult_map = dict(
                        zip(_lf_work["symbol"].values, _size_mult.tolist())
                    )
                    _weight_col = next(
                        (
                            c
                            for c in ("target_pct", "target_weight")
                            if c in target_positions.columns
                        ),
                        None,
                    )
                    if _weight_col and "symbol" in target_positions.columns:
                        target_positions = target_positions.copy()
                        _sym_mult = (
                            target_positions["symbol"].map(_mult_map).fillna(1.0)
                        )
                        target_positions[_weight_col] = (
                            target_positions[_weight_col] * _sym_mult
                        ).clip(-1.0, 1.0)
                        if "target_qty" in target_positions.columns:
                            target_positions["target_qty"] = (
                                target_positions["target_qty"] * _sym_mult
                            )
                        log.debug(
                            "[CONFORMAL] Applied quantile-interval size multipliers to %d positions via %s",
                            len(target_positions),
                            _weight_col,
                        )
    except Exception as e:
        log.debug("[CONFORMAL] quantile sizing skipped: %s", e)

    # --- Item 43: Halt-check — remove halted symbols from final target positions ---
    # halt-cache is wired in ops/_paper_runner_gates.apply_halt_cache_gate
    # (utils.halt_cache.HaltCache, TTL default 60s via policy.halt_cache.ttl_seconds,
    # default-off — requires policy.halt_cache.enabled=true). Results land in ctx.halted_symbols.
    try:
        _raw_halted = getattr(ctx, "halted_symbols", None)
        if _raw_halted is None:
            log.debug(
                "[size_positions] halt-check: ctx.halted_symbols absent (halt feed not wired)"
            )
        _halted: set[str] = set(_raw_halted or ())
        if (
            _halted
            and not target_positions.empty
            and "symbol" in target_positions.columns
        ):
            _before_syms: set[str] = set(target_positions["symbol"].tolist())
            _halted_in_target = _halted & _before_syms
            target_positions = target_positions[
                ~target_positions["symbol"].isin(_halted)
            ].copy()
            if _halted_in_target:
                log.warning(
                    "[size_positions] halt-check: dropped %d halted symbol(s): %s",
                    len(_halted_in_target),
                    sorted(_halted_in_target),
                )
    except Exception as _halt_err:
        _record_degraded_step("halt_check", _halt_err, meta=meta, log_obj=log)

    # --- Item 69: Buying-power pre-check — skip orders that exceed 95% of available capital ---
    # Only activates when ctx.buying_power is explicitly provided (live broker value).
    # Falls back to capital only when the policy flag buying_power_from_capital=true is set.
    # Without explicit buying_power, the check is a no-op (avoids scaling backtest weights).
    try:
        if not target_positions.empty and "target_weight" in target_positions.columns:
            _raw_bp = getattr(ctx, "buying_power", None)
            _bp_from_capital: bool = bool(
                (policy or {})
                .get("risk_limits", {})
                .get("buying_power_from_capital", False)
            )
            _buying_power: float = float(
                _raw_bp
                if _raw_bp is not None
                else (ctx.capital if _bp_from_capital else 0.0)
            )
            if _buying_power > 0:
                _BP_LIMIT: float = float(
                    (policy or {})
                    .get("risk_limits", {})
                    .get("buying_power_utilization_limit", 0.95)
                )
                _gross_weight = float(target_positions["target_weight"].abs().sum())
                if _gross_weight > _BP_LIMIT:
                    _scale_bp = _BP_LIMIT / _gross_weight
                    target_positions = target_positions.copy()
                    target_positions["target_weight"] = (
                        target_positions["target_weight"] * _scale_bp
                    )
                    if "target_qty" in target_positions.columns:
                        target_positions["target_qty"] = (
                            target_positions["target_qty"] * _scale_bp
                        )
                    log.warning(
                        "[size_positions] buying-power pre-check: gross weight %.3f exceeds %.0f%% "
                        "of capital — scaled down by %.4f",
                        _gross_weight,
                        _BP_LIMIT * 100,
                        _scale_bp,
                    )
    except Exception as _bp_err:
        _record_degraded_step("buying_power_precheck", _bp_err, meta=meta, log_obj=log)

    # --- Item 81: Pre-earnings size reduction (50% for symbols with earnings tomorrow) ---
    # AAPL/NVDA earnings = 5-15% gap at next open. Daily EOD cycle can't hedge intraday.
    # Reduce position to 50% for symbols with earnings within 1 trading day.
    # Uses panel column 'earnings_next_day' (bool) if available — no live API call.
    try:
        if not target_positions.empty and "symbol" in target_positions.columns:
            _earnings_col = "earnings_next_day"  # pre-computed in altdata_loader
            _pre_earnings_scale: float = float(
                (policy or {})
                .get("risk_limits", {})
                .get("pre_earnings_size_factor", 0.50)
            )
            if _earnings_col in target_positions.columns:
                _near_earnings = (
                    target_positions[_earnings_col].fillna(False).astype(bool)
                )
                if _near_earnings.any():
                    _earnings_syms = target_positions.loc[
                        _near_earnings, "symbol"
                    ].tolist()
                    target_positions = target_positions.copy()
                    _wt_col = (
                        "target_weight"
                        if "target_weight" in target_positions.columns
                        else None
                    )
                    _qty_col = (
                        "target_qty"
                        if "target_qty" in target_positions.columns
                        else None
                    )
                    if _wt_col:
                        target_positions.loc[_near_earnings, _wt_col] *= (
                            _pre_earnings_scale
                        )
                    if _qty_col:
                        target_positions.loc[_near_earnings, _qty_col] *= (
                            _pre_earnings_scale
                        )
                    log.info(
                        "[size_positions] pre-earnings: reduced %d symbol(s) by %.0f%%: %s",
                        len(_earnings_syms),
                        (1 - _pre_earnings_scale) * 100,
                        _earnings_syms[:10],
                    )
    except Exception as _earn_err:
        _record_degraded_step("pre_earnings_cut", _earn_err, meta=meta, log_obj=log)

    # Item 85: M&A exclusion — drop symbols with active M&A events from target positions.
    # When a Cash-deal M&A is announced the target stock moves to deal price and stays there;
    # factor signals no longer apply.  Symbols flagged via ctx.ma_symbols or the
    # `ma_activity` news category are dropped to avoid holding stagnant cash-deal targets.
    try:
        _ma_syms: set[str] = set()
        _raw_ma = getattr(ctx, "ma_symbols", None)
        if _raw_ma:
            _ma_syms = {str(s).upper() for s in _raw_ma}

        if not _ma_syms and "symbol" in target_positions.columns:
            # Opportunistically detect from news_category column if present
            _nc_col = next(
                (
                    c
                    for c in target_positions.columns
                    if "news_cat" in c.lower() or "category" in c.lower()
                ),
                None,
            )
            if _nc_col is not None:
                _ma_mask = (
                    target_positions[_nc_col]
                    .astype(str)
                    .str.contains(
                        "ma_activity|merger|acquisition", case=False, na=False
                    )
                )
                _ma_syms = set(target_positions.loc[_ma_mask, "symbol"].tolist())

        if (
            _ma_syms
            and not target_positions.empty
            and "symbol" in target_positions.columns
        ):
            _before_ma = set(target_positions["symbol"].tolist())
            _ma_in_target = _ma_syms & _before_ma
            if _ma_in_target:
                target_positions = target_positions[
                    ~target_positions["symbol"].isin(_ma_in_target)
                ].copy()
                log.warning(
                    "[size_positions] M&A exclusion: dropped %d symbol(s) with active M&A event: %s",
                    len(_ma_in_target),
                    sorted(_ma_in_target)[:10],
                )
    except Exception as _ma_err:
        log.debug("[size_positions] M&A filter skipped: %s", _ma_err)

    return target_positions, do_rebal, meta
