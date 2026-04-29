"""Risk controls integration for order filtering.

This module provides a centralized function to apply all risk controls (pre-trade checks
and kill switch) to orders before execution. It combines pre-trade checks and kill switch
into a single, easy-to-use interface.

Usage:
    >>> from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls
    >>> import pandas as pd
    >>>
    >>> orders = pd.DataFrame({
    ...     "symbol": ["AAPL", "GOOGL"],
    ...     "side": ["BUY", "BUY"],
    ...     "qty": [100, 50],
    ...     "price": [150.0, 2500.0]
    ... })
    >>>
    >>> filtered, result, kill_switch_engaged = filter_orders_with_risk_controls(
    ...     orders,
    ...     portfolio=None,
    ...     enable_pre_trade_checks=True,
    ...     enable_kill_switch=True
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from src.assembled_core.execution.kill_switch import (
    check_drawdown_kill_switch,
    guard_orders_with_kill_switch,
    is_kill_switch_engaged,
)
from src.assembled_core.execution.pre_trade_checks import (
    PreTradeCheckResult,
    PreTradeConfig,
    run_pre_trade_checks,
)
from src.assembled_core.logging_utils import get_logger

if TYPE_CHECKING:
    from src.assembled_core.qa.qa_gates import QAGatesSummary

logger = get_logger("assembled_core.execution.risk_controls")

# ---------------------------------------------------------------------------
# Crisis-Alpha PAUSE Kill-Switch (T4.3)
# ---------------------------------------------------------------------------

_CRISIS_ALPHA_PASS_STATES = {"NORMAL", "WATCH", "ACTIVE", "COOLDOWN"}
_CRISIS_ALPHA_BLOCK_STATE = "PAUSE"


def check_crisis_alpha_kill_switch(ctx: Any) -> tuple[bool, str]:
    """Return (allowed, reason) for the crisis-alpha PAUSE kill-switch.

    Reads the current crisis-alpha state from *ctx* using three fallbacks:

    1. ``ctx.crisis_alpha_state`` — direct attribute (string or CrisisStateRecord).
    2. ``ctx.meta.get("crisis_alpha_state")`` — generic metadata dict.
    3. File at ``output/ops/crisis_alpha_state.json`` (default state-file path).

    Blocking rule:
    - Returns ``(False, "BLOCKED: crisis_alpha state=PAUSE")`` only when state is
      exactly "PAUSE".
    - Returns ``(True, "OK")`` for NORMAL / WATCH / ACTIVE / COOLDOWN.
    - Returns ``(True, "OK — no crisis state available")`` when state cannot be
      read (fail-open — crisis_alpha is shadow-only by default).

    Args:
        ctx: Any object carrying trading context.  The function duck-types the
             access — a missing attribute is caught, not raised.

    Returns:
        ``(True, reason)`` when orders are allowed, ``(False, reason)`` when
        the PAUSE kill-switch is active.
    """
    state: str | None = None

    # Fallback 1: direct attribute
    raw = getattr(ctx, "crisis_alpha_state", None)
    if raw is not None:
        # Accept CrisisStateRecord (has .state) or plain string
        if hasattr(raw, "state"):
            state = str(raw.state)
        else:
            state = str(raw)

    # Fallback 2: metadata dict
    if state is None:
        meta = getattr(ctx, "meta", None)
        if isinstance(meta, dict):
            val = meta.get("crisis_alpha_state")
            if val is not None:
                state = str(val)

    # Fallback 3: default state file
    if state is None:
        _state_file = Path("output") / "ops" / "crisis_alpha_state.json"
        policy_path: str | None = None
        # Try to get path from ctx.policy if available
        policy = getattr(ctx, "policy", None)
        if isinstance(policy, dict):
            try:
                policy_path = (
                    policy.get("intel", {})
                    .get("crisis_alpha", {})
                    .get("state_path")
                )
            except Exception as _policy_err:
                logger.debug("[risk_controls] policy.intel.crisis_alpha.state_path lookup failed: %s", _policy_err)
        resolved_path = Path(policy_path) if policy_path else _state_file
        if resolved_path.exists():
            try:
                import json
                data = json.loads(resolved_path.read_text(encoding="utf-8"))
                state = str(data.get("state", ""))
            except Exception as exc:
                logger.warning(
                    "[crisis_alpha] Could not read state file %s: %s", resolved_path, exc
                )

    if state is None or state == "":
        return True, "OK — no crisis state available"

    state_upper = state.upper()
    if state_upper == _CRISIS_ALPHA_BLOCK_STATE:
        logger.warning("[WARN] crisis_alpha PAUSE — blocking order")
        return False, "BLOCKED: crisis_alpha state=PAUSE"

    return True, "OK"


@dataclass
class RiskControlResult:
    """Result of risk control filtering.

    Attributes:
        filtered_orders: Orders DataFrame after all risk controls
        pre_trade_result: PreTradeCheckResult (or None if checks disabled)
        kill_switch_engaged: True if kill switch blocked orders
        total_orders_before: Number of orders before filtering
        total_orders_after: Number of orders after filtering
    """

    filtered_orders: pd.DataFrame
    pre_trade_result: PreTradeCheckResult | None
    kill_switch_engaged: bool
    total_orders_before: int
    total_orders_after: int


def filter_orders_with_risk_controls(
    orders: pd.DataFrame,
    portfolio: pd.DataFrame | None = None,
    qa_status: QAGatesSummary | None = None,
    risk_summary: dict[str, Any] | None = None,
    pre_trade_config: PreTradeConfig | None = None,
    enable_pre_trade_checks: bool = True,
    enable_kill_switch: bool = True,
    *,
    current_positions: pd.DataFrame | None = None,
    prices_latest: pd.DataFrame | None = None,
    equity: float | None = None,
    current_equity: float | None = None,
    peak_equity: float | None = None,
    security_meta_df: pd.DataFrame | None = None,
    policy: dict[str, Any] | None = None,
    crisis_alpha_ctx: Any | None = None,
) -> tuple[pd.DataFrame, RiskControlResult]:
    """Apply all risk controls to orders and return filtered orders.

    This function applies pre-trade checks and kill switch in sequence:
    1. Pre-trade checks (if enabled): Position size limits, gross exposure, QA gates
    2. Kill switch (if enabled): Emergency block via environment variable

    Args:
        orders: DataFrame with columns: symbol, side, qty, price (optional: timestamp)
        portfolio: Optional DataFrame with current portfolio snapshot
        qa_status: Optional QA gates summary (for QA_BLOCK check)
        risk_summary: Optional risk summary dictionary (for future use)
        pre_trade_config: Optional PreTradeConfig (default: no limits if None)
        enable_pre_trade_checks: Enable pre-trade checks (default: True)
        enable_kill_switch: Enable kill switch check (default: True)

    Returns:
        Tuple of (filtered_orders DataFrame, RiskControlResult):
        - filtered_orders: Orders that passed all enabled risk controls
        - result: RiskControlResult with details about filtering

    Example:
        >>> import pandas as pd
        >>> from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls
        >>>
        >>> orders = pd.DataFrame({
        ...     "symbol": ["AAPL", "GOOGL"],
        ...     "side": ["BUY", "BUY"],
        ...     "qty": [100, 50],
        ...     "price": [150.0, 2500.0]
        ... })
        >>>
        >>> filtered, result = filter_orders_with_risk_controls(orders)
        >>>
        >>> if len(filtered) < len(orders):
        ...     print(f"Orders filtered: {result.total_orders_before - result.total_orders_after} blocked")
    """
    total_orders_before = len(orders)
    filtered_orders = orders.copy()
    pre_trade_result: PreTradeCheckResult | None = None
    kill_switch_engaged = False

    # Step -1: Crisis-Alpha PAUSE kill-switch (T4.3)
    # Gated under policy.intel.crisis_alpha.enabled — skip entirely if disabled.
    _crisis_alpha_enabled = False
    if policy is not None:
        try:
            _crisis_alpha_enabled = bool(
                policy.get("intel", {})
                .get("crisis_alpha", {})
                .get("enabled", False)
            )
        except Exception:
            _crisis_alpha_enabled = False

    if not _crisis_alpha_enabled:
        logger.debug("[SKIP] crisis_alpha kill_switch disabled")
    elif crisis_alpha_ctx is not None:
        _ca_allowed, _ca_reason = check_crisis_alpha_kill_switch(crisis_alpha_ctx)
        if not _ca_allowed:
            # PAUSE state — block all orders immediately, do not proceed further.
            logger.warning("[WARN] crisis_alpha PAUSE — blocking order")
            empty_orders = pd.DataFrame(columns=list(orders.columns))
            result = RiskControlResult(
                filtered_orders=empty_orders,
                pre_trade_result=None,
                kill_switch_engaged=True,
                total_orders_before=total_orders_before,
                total_orders_after=0,
            )
            return empty_orders, result

    # Step 0: Graduated drawdown exposure caps (from risk state machine)
    # Applies BEFORE pre-trade checks so all downstream logic sees reduced orders.
    if current_equity is not None and peak_equity is not None and peak_equity > 0:
        current_dd_pct = ((current_equity / peak_equity) - 1.0) * 100.0  # e.g. -8.0
        try:
            from src.assembled_core.risk.state_machine import (
                compute_drawdown_risk_level,
            )

            risk_level, exposure_cap = compute_drawdown_risk_level(current_dd_pct)
        except ImportError:
            risk_level, exposure_cap = "NORMAL", 1.0

        if exposure_cap < 1.0 and not filtered_orders.empty and "qty" in filtered_orders.columns:
            logger.warning(
                "[RiskControls] Drawdown %.1f%% -> risk_level=%s, exposure_cap=%.2f. "
                "Scaling all order quantities by %.0f%%.",
                current_dd_pct,
                risk_level,
                exposure_cap,
                exposure_cap * 100,
            )
            filtered_orders = filtered_orders.copy()
            filtered_orders["qty"] = filtered_orders["qty"] * exposure_cap
            # Drop orders that became negligible after scaling
            filtered_orders = filtered_orders[filtered_orders["qty"].abs() >= 1e-10].copy()

    # Step 1: Pre-trade checks
    if enable_pre_trade_checks:
        logger.debug("Applying pre-trade checks...")
        pre_trade_result, filtered_orders = run_pre_trade_checks(
            filtered_orders,
            portfolio=portfolio,
            qa_status=qa_status,
            risk_summary=risk_summary,
            config=pre_trade_config,
            current_positions=current_positions,
            prices_latest=prices_latest,
            equity=equity,
            current_equity=current_equity,
            peak_equity=peak_equity,
            security_meta_df=security_meta_df,
        )

        if not pre_trade_result.is_ok:
            logger.warning(
                f"Pre-trade checks failed: {len(pre_trade_result.blocked_reasons)} reason(s). "
                f"Orders before: {total_orders_before}, after: {len(filtered_orders)}"
            )
            for reason in pre_trade_result.blocked_reasons:
                logger.warning(f"  - {reason}")
        elif len(filtered_orders) < total_orders_before:
            logger.info(
                f"Pre-trade checks filtered orders: {total_orders_before} -> {len(filtered_orders)} "
                f"({len(pre_trade_result.blocked_reasons)} reason(s))"
            )
            for reason in pre_trade_result.blocked_reasons:
                logger.info(f"  - {reason}")
        else:
            logger.debug("Pre-trade checks passed - all orders accepted")
    else:
        logger.debug("Pre-trade checks disabled - skipping")

    # Step 2: Kill switch
    if enable_kill_switch:
        logger.debug("Checking kill switch...")

        # 2a: Drawdown-based kill switch check (CRITICAL-2.1)
        if current_equity is not None and peak_equity is not None and peak_equity > 0:
            if check_drawdown_kill_switch(current_equity, peak_equity):
                kill_switch_engaged = True
                filtered_orders = pd.DataFrame(columns=list(filtered_orders.columns))

        # 2b: Standard kill switch check (env var + sentinel file)
        if not kill_switch_engaged:
            kill_switch_engaged = is_kill_switch_engaged()
            if kill_switch_engaged:
                filtered_orders = guard_orders_with_kill_switch(filtered_orders)

        if kill_switch_engaged:
            logger.warning("KILL_SWITCH engaged - blocking all remaining orders")
        else:
            logger.debug("Kill switch not engaged - orders pass through")
    else:
        logger.debug("Kill switch disabled - skipping")

    total_orders_after = len(filtered_orders)

    # Log summary
    if total_orders_after < total_orders_before:
        logger.info(
            f"Risk controls summary: {total_orders_before} orders -> {total_orders_after} orders "
            f"({total_orders_before - total_orders_after} blocked)"
        )
    elif total_orders_after == 0 and total_orders_before > 0:
        logger.warning("All orders blocked by risk controls")
    elif total_orders_after == total_orders_before:
        logger.debug("All orders passed risk controls")

    result = RiskControlResult(
        filtered_orders=filtered_orders,
        pre_trade_result=pre_trade_result,
        kill_switch_engaged=kill_switch_engaged,
        total_orders_before=total_orders_before,
        total_orders_after=total_orders_after,
    )

    return filtered_orders, result
