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
from typing import Any

import pandas as pd

from src.assembled_core.execution.kill_switch import (
    guard_orders_with_kill_switch,
    is_kill_switch_engaged,
)
from src.assembled_core.execution.pre_trade_checks import (
    PreTradeCheckResult,
    PreTradeConfig,
    run_pre_trade_checks,
)
from src.assembled_core.logging_utils import setup_logging
from src.assembled_core.qa.qa_gates import QAGatesSummary

logger = setup_logging(level="INFO")


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

    # Step 1: Pre-trade checks
    if enable_pre_trade_checks:
        logger.debug("Applying pre-trade checks...")
        pre_trade_result, filtered_orders = run_pre_trade_checks(
            filtered_orders,
            portfolio=portfolio,
            qa_status=qa_status,
            risk_summary=risk_summary,
            config=pre_trade_config,
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
        kill_switch_engaged = is_kill_switch_engaged()

        if kill_switch_engaged:
            logger.warning(
                f"KILL_SWITCH engaged - blocking all {len(filtered_orders)} remaining orders"
            )
            filtered_orders = guard_orders_with_kill_switch(filtered_orders)
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
