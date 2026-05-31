"""_tc_risk — check_risk() extracted from trading_cycle_v2."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from src.assembled_core.config.policy_loader import load_policy
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
    _apply_risk_controls_default,
    _evaluate_auto_dd_kill_switch,
    _evaluate_circuit_breaker,
    _evaluate_var_gate,
    _record_degraded_step,
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
    """Apply copula tail-risk + risk controls to orders; return updated result.

    Steps kept (modify orders or orders_filtered):
      - QA gate: block all orders if ctx.qa_block_trading
      - Copula tail dep: scale orders qty if avg_lower_tail_dep > 0.5
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

    Retired: the former EVT-tail-VaR and barbell qty-reduction overlays were
    moved to archive/observability_graveyard_2026q2/ (observability-only, never
    influenced trading) — their try-blocks were dead (ModuleNotFoundError every
    cycle), so they are not present here.
    """
    if log is None:
        log = logger

    policy = getattr(ctx, "_policy_cache", None)
    if policy is None:
        try:
            policy = load_policy()
        except Exception as _policy_exc:
            log.warning(
                "[RISK] load_policy() failed — all policy-gated guards disabled: %s",
                _policy_exc,
            )
            policy = {}

    # Fast path: the generic ``enable_risk_controls`` flag gates the numeric /
    # churn gate stack (set =False for backtest parity). R2-2 DECOUPLING: the
    # OPERATOR kill switch is an INDEPENDENT safety layer — an operator HALT must
    # not be silently disarmable by this generic flag in a real trading mode. So
    # in any non-backtest mode we still route orders through the standalone
    # operator kill-switch guard before passing them through. Backtest keeps the
    # pure pass-through on purpose: reading the *current* live HALT state during a
    # historical replay would be a wrong-context read and break replay
    # determinism. (Note: trading_cycle_shared.py:1534-1535 couples
    # enable_kill_switch to the same flag, but that branch is only reachable when
    # the flag is True — i.e. KS correctly on — so no decoupling is needed there.)
    if not getattr(ctx, "enable_risk_controls", True):
        result.orders = orders
        if getattr(ctx, "mode", "eod") != "backtest" and not orders.empty:
            try:
                from src.assembled_core.execution.kill_switch import (
                    guard_orders_with_kill_switch,
                )

                guarded = guard_orders_with_kill_switch(orders)
            except Exception as e:
                # Fail-CLOSED (consistent with R2-1): if the operator kill-switch
                # state cannot be evaluated, we cannot prove trading is permitted
                # — block this cycle's orders rather than pass them through.
                log.error(
                    "[RISK] operator kill-switch check raised under "
                    "enable_risk_controls=False — FAIL-CLOSED, blocking orders: %s",
                    e,
                )
                result.meta["risk_gate_error"] = True
                result.meta["operator_kill_switch"] = {
                    "status": "error",
                    "error": str(e),
                }
                result.orders_filtered = orders.iloc[0:0].copy()
                return result
            if len(guarded) < len(orders):
                result.meta["operator_kill_switch_blocked"] = True
            result.orders_filtered = guarded.copy()
            return result
        result.orders_filtered = orders.copy()
        return result

    # QA gate
    if ctx.qa_block_trading:
        log.warning("QA Gate: Trading blocked - %s", ctx.qa_block_reason or "no reason")
        result.orders = pd.DataFrame(
            columns=["timestamp", "symbol", "side", "qty", "price"]
        )
        result.orders_filtered = result.orders.copy()
        result.meta["qa_block_trading"] = True
        result.meta["qa_block_reason"] = ctx.qa_block_reason
        return result

    # Shared pivot for Copula tail-risk (compute once, reuse)
    _prices_for_risk = prices_filtered if prices_filtered is not None else ctx.prices
    _shared_rets = None
    if (
        not orders.empty
        and _prices_for_risk is not None
        and not _prices_for_risk.empty
        and "close" in _prices_for_risk.columns
    ):
        try:
            _ts_col = (
                "timestamp"
                if "timestamp" in _prices_for_risk.columns
                else _prices_for_risk.columns[0]
            )
            _pivot_risk = _prices_for_risk.pivot_table(
                index=_ts_col,
                columns="symbol" if "symbol" in _prices_for_risk.columns else None,
                values="close",
            )
            _shared_rets = _pivot_risk.pct_change().dropna(how="all")
        except Exception as e:
            # Fail-SOFT but OBSERVABLE (QUAL/Zensus-1): a failure here leaves
            # _shared_rets=None, which silently short-circuits the copula
            # tail-risk qty reduction below. Surface it so the disabled
            # protection is visible (the hard VaR/DD/CB gates further down are
            # independently fail-CLOSED and still run).
            _shared_rets = None
            _record_degraded_step(
                "risk_shared_returns_pivot", e, meta=result.meta, log_obj=log
            )

    # Copula tail dependence
    try:
        if (
            _shared_rets is not None
            and len(_shared_rets) >= 60
            and 1 < _shared_rets.shape[1] <= 30
        ):
            from src.assembled_core.ml.copula_models import compute_portfolio_tail_risk

            _cop_metrics = compute_portfolio_tail_risk(_shared_rets)
            if float(_cop_metrics.get("avg_lower_tail_dep", 0.0)) > 0.5:
                orders = orders.copy()
                orders["qty"] = orders["qty"] * 0.80
                log.warning(
                    "[RISK] Copula avg_lower_tail_dep > 0.5 — reducing qty by 20%%"
                )
    except Exception as e:
        _record_degraded_step("copula_tail_risk", e, meta=result.meta, log_obj=log)

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
        # Fail-CLOSED (R2-1): a VaR-gate evaluation error means we CANNOT prove
        # the portfolio is within its VaR limit. The safe action is to block this
        # cycle's orders, not let them pass. Surfaced loudly + flagged in meta.
        log.error(
            "[RISK] var_gate evaluation raised — FAIL-CLOSED, blocking orders: %s", e
        )
        result.meta["var_gate"] = {"status": "error", "error": str(e)}
        result.meta["risk_gate_error"] = True
        _rej_counts["var_gate_error"] = len(result.orders_filtered)
        result.orders_filtered = result.orders_filtered.iloc[0:0].copy()

    # Step 6.4: Auto-DD kill switch
    try:
        dd_decision = _evaluate_auto_dd_kill_switch(ctx, result, policy)
        if dd_decision is not None:
            from src.assembled_core.execution.kill_switch import (
                activate_kill_switch,
            )

            activate_kill_switch(
                throttle_pct=dd_decision["throttle_allowed_pct"],
                reason=dd_decision["reason"],
                actor="trading_cycle_v2_auto_dd",
            )
            result.meta["auto_dd_kill_switch"] = dd_decision
            if dd_decision["level"] == "kill":
                _rej_counts["auto_dd_kill_switch"] = len(result.orders_filtered)
                result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        # Fail-CLOSED (R2-1): if the drawdown kill-switch evaluation raised we
        # cannot confirm the account is under its drawdown limit. Block this
        # cycle's orders. We deliberately do NOT activate the PERSISTENT kill
        # switch on a transient evaluation error (it requires an operator token
        # to clear, per OPS-04/4b) — blocking the batch is the proportionate
        # safe action; a real drawdown breach trips on the next clean evaluation.
        log.error(
            "[RISK] auto_dd_kill_switch raised — FAIL-CLOSED, blocking orders: %s", e
        )
        result.meta["auto_dd_kill_switch"] = {"status": "error", "error": str(e)}
        result.meta["risk_gate_error"] = True
        _rej_counts["auto_dd_error"] = len(result.orders_filtered)
        result.orders_filtered = result.orders_filtered.iloc[0:0].copy()

    # Step 6.45: Circuit breaker
    try:
        cb_decision = _evaluate_circuit_breaker(ctx, result, policy)
        if cb_decision is not None:
            result.meta["circuit_breaker"] = cb_decision
            _rej_counts["circuit_breaker"] = len(result.orders_filtered)
            result.orders_filtered = result.orders_filtered.iloc[0:0].copy()
    except Exception as e:
        # Fail-CLOSED (R2-1): an unknown circuit-breaker state must block, not pass.
        log.error("[RISK] circuit_breaker raised — FAIL-CLOSED, blocking orders: %s", e)
        result.meta["circuit_breaker"] = {"status": "error", "error": str(e)}
        result.meta["risk_gate_error"] = True
        _rej_counts["circuit_breaker_error"] = len(result.orders_filtered)
        result.orders_filtered = result.orders_filtered.iloc[0:0].copy()

    # Step 6.6: Anti-churn deadzone + min-notional
    try:
        anti_churn_cfg = policy.get("anti_churn") or {}
        if not result.orders_filtered.empty:
            if anti_churn_cfg.get("deadzone_enabled", False):
                from src.assembled_core.paper.deadzone_rebalance import (
                    filter_deadzone_orders,
                )

                _dz_pos = (
                    ctx.current_positions[["symbol", "qty"]].copy()
                    if ctx.current_positions is not None
                    and not ctx.current_positions.empty
                    and "qty" in ctx.current_positions.columns
                    else None
                )
                result.orders_filtered, _dz_meta = filter_deadzone_orders(
                    result.orders_filtered,
                    _dz_pos,
                    deadzone_pct=float(anti_churn_cfg.get("deadzone_pct", 0.05)),
                )
                result.meta["deadzone_rebalance"] = _dz_meta
            if (
                anti_churn_cfg.get("rebalance_filter_enabled", False)
                and not result.orders_filtered.empty
            ):
                from src.assembled_core.paper.rebalance_filter import (
                    filter_small_rebalances,
                )

                result.orders_filtered, _rf_meta = filter_small_rebalances(
                    result.orders_filtered,
                    min_notional=float(anti_churn_cfg.get("min_notional", 500.0)),
                    prices=ctx.prices,
                )
                result.meta["rebalance_filter"] = _rf_meta
    except Exception as e:
        # Fail-SOFT but OBSERVABLE: the anti-churn deadzone + min-notional
        # filters are a turnover/cost protection; a silent skip means uncapped
        # churn. Not order-blocking (cost control, not a hard risk gate), so
        # surface + record rather than fail-closed (QUAL-04 / Zensus-1).
        _record_degraded_step("anti_churn_filters", e, meta=result.meta, log_obj=log)

    # Step 6.7: Fat-finger guard
    try:
        ffg_cfg = policy.get("fat_finger_guard") or {}
        if ffg_cfg.get("enabled", False) and not result.orders_filtered.empty:
            from src.assembled_core.execution.fat_finger_guard import (
                apply_fat_finger_guard_from_policy,
            )

            _ffg_orders, _ffg_reasons = apply_fat_finger_guard_from_policy(
                result.orders_filtered, policy
            )
            n_rejected = len(result.orders_filtered) - len(_ffg_orders)
            result.orders_filtered = _ffg_orders
            if n_rejected:
                log.warning(
                    "[FAT-FINGER] Rejected %d orders: %s", n_rejected, _ffg_reasons[:3]
                )
                _rej_counts["fat_finger"] = n_rejected
    except Exception as e:
        # Fail-CLOSED (R2-1): if the fat-finger hard cap could not be applied we
        # cannot guarantee no oversized/erroneous order escapes — block the batch.
        log.error(
            "[RISK] fat_finger_guard raised — FAIL-CLOSED, blocking orders: %s", e
        )
        result.meta["fat_finger_guard"] = {"status": "error", "error": str(e)}
        result.meta["risk_gate_error"] = True
        _rej_counts["fat_finger_error"] = len(result.orders_filtered)
        result.orders_filtered = result.orders_filtered.iloc[0:0].copy()

    # Step 6.9: Order lifecycle tracking
    try:
        if not result.orders_filtered.empty:
            from src.assembled_core.execution.order_lifecycle import (
                OrderLifecycleTracker,
                OrderState,
            )

            _olt = OrderLifecycleTracker()
            for _ord_row in result.orders_filtered.itertuples(index=False):
                _oid = _olt.create(
                    symbol=str(getattr(_ord_row, "symbol", "")),
                    side=str(getattr(_ord_row, "side", "buy")),
                    quantity=float(getattr(_ord_row, "qty", 0) or 0),
                    price=float(getattr(_ord_row, "price", 0) or 0) or None,
                    source="trading_cycle_v2",
                )
                _olt.transition(_oid, OrderState.VALIDATED)
                _olt.transition(_oid, OrderState.SUBMITTED)
            result.meta["order_lifecycle"] = {
                "n_orders_tracked": len(result.orders_filtered),
                "state": "SUBMITTED",
            }
            # Lifecycle log hook — one SUBMITTED entry per order (fire-and-forget)
            try:
                from src.assembled_core.ops.order_lifecycle_log import (
                    append_lifecycle_event,
                )

                _strategy = str(result.meta.get("strategy", ""))
                _run_id = str(result.meta.get("run_id", ""))
                _lc_path = (
                    Path(ctx.output_dir) / "order_lifecycle.jsonl"
                    if getattr(ctx, "output_dir", None)
                    else None
                )
                for _ord_row in result.orders_filtered.itertuples(index=False):
                    _sym = str(getattr(_ord_row, "symbol", ""))
                    _sid = str(getattr(_ord_row, "side", "BUY")).upper()
                    append_lifecycle_event(
                        "SUBMITTED",
                        order_id=str(getattr(_ord_row, "order_id", ""))
                        or f"{_sym}_{_sid}_{_run_id}",
                        symbol=_sym,
                        side=_sid,
                        qty=float(getattr(_ord_row, "qty", 0) or 0),
                        price=float(getattr(_ord_row, "price", 0) or 0) or None,
                        strategy=_strategy,
                        actor="trading_cycle_v2",
                        run_id=_run_id,
                        log_path=_lc_path,
                    )
            except Exception as _lce:
                log.debug("[LIFECYCLE-LOG] SUBMITTED hook skipped: %s", _lce)
    except Exception as e:
        log.debug("order_lifecycle tracking skipped: %s", e)

    result.meta["rejection_counts"] = _rej_counts
    return result
