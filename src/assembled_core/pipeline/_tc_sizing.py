"""_tc_sizing — size_positions() and all _sp_* helpers extracted from trading_cycle_v2."""

from __future__ import annotations

import logging

import pandas as pd
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    _estimate_symbol_volatilities,
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
from src.assembled_core.risk.turnover_budget import (
    apply_turnover_gate,
    estimate_turnover,
)
from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

logger = logging.getLogger(__name__)


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
            vols = _estimate_symbol_volatilities(prices_for_sizing, lookback=int(sizing_cfg.get("vol_lookback_days", 60)))
            target_positions = compute_risk_parity_weights(
                signals, vols, total_capital=ctx.capital,
                max_weight=float(sizing_cfg.get("max_weight", 0.30)),
                top_n=sizing_cfg.get("top_n"),
            )
        elif sizing_method == "vol_scaled":
            from src.assembled_core.portfolio.position_sizing import (
                compute_vol_scaled_weights,
            )
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
                from src.assembled_core.portfolio.cost_aware_optimizer import (
                    OptimizerConfig,
                    optimize_portfolio,
                )
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
                from src.assembled_core.portfolio.covariance import estimate_covariance
                from src.assembled_core.portfolio.risk_budgeting import (
                    compute_erc_weights,
                )
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
                from src.assembled_core.portfolio.covariance import estimate_covariance
                from src.assembled_core.portfolio.mvo_optimizer import (
                    mvo_with_cardinality,
                )
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
            from src.assembled_core.risk.liquidity_scoring import (
                apply_liquidity_adjusted_sizing,
                compute_liquidity_scores,
            )
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

    # Prediction-market overlay (live/paper only — never fetches live API in backtest)
    pm_multiplier = 1.0
    try:
        pm_cfg = (policy.get("prediction_market_overlay") or {})
        if pm_cfg.get("enabled", False) and getattr(ctx, "mode", "") in ("live", "paper"):
            from src.assembled_core.risk.georisk_overlay import (
                get_market_implied_geo_signal,
            )
            pm_signal = get_market_implied_geo_signal(policy=policy)
            raw_pm = float(pm_signal.get("signal", 0.0))
            pm_threshold = float(pm_cfg.get("threshold", 0.25))
            if raw_pm > pm_threshold:
                reduction = float(pm_cfg.get("reduction_factor", 0.50))
                pm_multiplier = max(0.0, 1.0 - reduction * raw_pm)
            meta["prediction_market"] = {"signal": raw_pm, "multiplier": pm_multiplier, "n_sources": pm_signal.get("n_sources", 0)}
    except Exception as e:
        log.debug("prediction_market_overlay skipped: %s", e)

    final_multiplier = geo_multiplier * profit_lock_mult * vol_scale_factor * ms_multiplier * crisis_alpha_multiplier * pm_multiplier
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
                from src.assembled_core.risk.trailing_stops import (
                    apply_stop_reductions_to_weights,
                    compute_trailing_stops,
                )
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
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )
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
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )
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
                from src.assembled_core.ops.shadow_recorder import (
                    is_shadow_only,
                    record_shadow,
                )
                from src.assembled_core.portfolio.inverse_etf_selector import (
                    InverseETFSelector,
                )
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
        from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
        from src.assembled_core.events.crisis_alpha.pipeline import (
            run_crisis_alpha_pipeline,
        )
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
            from src.assembled_core.portfolio.cost_aware_wrapper import (
                apply_cost_aware_from_policy,
            )
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
