"""_tc_risk — check_risk() extracted from trading_cycle_v2."""

from __future__ import annotations

import logging

import pandas as pd
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
    _apply_risk_controls_default,
    _evaluate_auto_dd_kill_switch,
    _evaluate_circuit_breaker,
    _evaluate_var_gate,
)

logger = logging.getLogger(__name__)


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
            import numpy as _np_evt
            from src.assembled_core.risk.evt_tail_var import evt_var
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
        from src.assembled_core.portfolio.barbell_strategy import (
            build_barbell_allocation,
            compute_tail_risk_score,
        )
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
            from src.assembled_core.execution.kill_switch import (
                activate_kill_switch,
            )
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
                from src.assembled_core.paper.deadzone_rebalance import (
                    filter_deadzone_orders,
                )
                _dz_pos = ctx.current_positions[["symbol", "qty"]].copy() if ctx.current_positions is not None and not ctx.current_positions.empty and "qty" in ctx.current_positions.columns else None
                result.orders_filtered, _dz_meta = filter_deadzone_orders(result.orders_filtered, _dz_pos, deadzone_pct=float(anti_churn_cfg.get("deadzone_pct", 0.05)))
                result.meta["deadzone_rebalance"] = _dz_meta
            if anti_churn_cfg.get("rebalance_filter_enabled", False) and not result.orders_filtered.empty:
                from src.assembled_core.paper.rebalance_filter import (
                    filter_small_rebalances,
                )
                result.orders_filtered, _rf_meta = filter_small_rebalances(result.orders_filtered, min_notional=float(anti_churn_cfg.get("min_notional", 500.0)), prices=prices_filtered if prices_filtered is not None else ctx.prices)
                result.meta["rebalance_filter"] = _rf_meta
    except Exception as e:
        log.debug("anti_churn filters skipped: %s", e)

    # Step 6.7: Fat-finger guard
    try:
        ffg_cfg = policy.get("fat_finger_guard") or {}
        if ffg_cfg.get("enabled", False) and not result.orders_filtered.empty:
            from src.assembled_core.execution.fat_finger_guard import (
                apply_fat_finger_guard_from_policy,
            )
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
            from src.assembled_core.execution.order_lifecycle import (
                OrderLifecycleTracker,
                OrderState,
            )
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
