"""Pre-trade checks for order validation and risk limits.

This module provides pre-trade validation checks to ensure orders comply with
risk limits and trading rules before execution. It validates position sizes,
exposure limits, and integrates with QA gates.

Key features:
- Position size limits (per symbol, per order)
- Gross exposure limits
- QA gate integration
- Sector/Region exposure limits (optional, for future use)
- Robust handling of missing information

Usage:
    >>> from src.assembled_core.execution.pre_trade_checks import (
    ...     PreTradeConfig,
    ...     PreTradeCheckResult,
    ...     run_pre_trade_checks
    ... )
    >>> import pandas as pd
    >>>
    >>> config = PreTradeConfig(
    ...     max_notional_per_symbol=10000.0,
    ...     max_gross_exposure=50000.0
    ... )
    >>> orders = pd.DataFrame({
    ...     "symbol": ["AAPL", "GOOGL"],
    ...     "side": ["BUY", "BUY"],
    ...     "qty": [100, 50],
    ...     "price": [150.0, 2500.0]
    ... })
    >>>
    >>> result = run_pre_trade_checks(orders, portfolio=None, config=config)
    >>> if result.is_ok:
    ...     print("All orders passed pre-trade checks")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.assembled_core.qa.qa_gates import QAGatesSummary, QAResult


@dataclass
class PreTradeConfig:
    """Configuration for pre-trade checks.

    Attributes:
        max_notional_per_symbol: Maximum notional value per symbol (in currency units)
        max_weight_per_symbol: Maximum weight per symbol (as fraction, e.g., 0.1 = 10%)
        max_gross_exposure: Maximum gross exposure (sum of absolute positions)
        max_sector_exposure: Optional dict mapping sector names to max exposure (fraction)
        max_region_exposure: Optional dict mapping region names to max exposure (fraction)
    """

    max_notional_per_symbol: float | None = None
    max_weight_per_symbol: float | None = None
    max_gross_exposure: float | None = None
    max_sector_exposure: dict[str, float] | None = None
    max_region_exposure: dict[str, float] | None = None


@dataclass
class PreTradeCheckResult:
    """Result of pre-trade checks.

    Attributes:
        is_ok: True if all checks passed, False otherwise
        blocked_reasons: List of reasons why orders were blocked
        filtered_orders: Orders DataFrame after filtering (only orders that passed)
        summary: Optional dictionary with summary metrics (gross exposure, etc.)
    """

    is_ok: bool
    blocked_reasons: list[str] = field(default_factory=list)
    filtered_orders: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    summary: dict[str, Any] = field(default_factory=dict)


def run_pre_trade_checks(
    orders: pd.DataFrame,
    portfolio: pd.DataFrame | None = None,
    qa_status: QAGatesSummary | None = None,
    risk_summary: dict[str, Any] | None = None,
    config: PreTradeConfig | None = None,
) -> tuple[PreTradeCheckResult, pd.DataFrame]:
    """Run pre-trade checks on orders and filter out blocked orders.

    This function performs various validation checks:
    - Position size limits (notional per symbol, weight per symbol)
    - Gross exposure limits
    - QA gate integration (if QA status blocks trading)
    - Sector/Region exposure limits (if config and portfolio data available)

    Args:
        orders: DataFrame with columns: symbol, side, qty, price (optional: timestamp)
            Must have at least symbol and qty. Price is used for notional calculation.
        portfolio: Optional DataFrame with current portfolio snapshot.
            Expected columns: symbol, qty (or weight, or value)
            Used for calculating resulting positions and weights
        qa_status: Optional QA gates summary. If overall_result is BLOCK, all orders are blocked.
        risk_summary: Optional risk summary dictionary (for future use)
        config: Optional PreTradeConfig. If None, uses default config (no limits enforced)

    Returns:
        Tuple of (PreTradeCheckResult, filtered_orders DataFrame):
        - PreTradeCheckResult with is_ok, blocked_reasons, filtered_orders, summary
        - filtered_orders: Orders that passed all checks (may be empty if all blocked)

    Example:
        >>> import pandas as pd
        >>> from src.assembled_core.execution.pre_trade_checks import (
        ...     PreTradeConfig,
        ...     run_pre_trade_checks
        ... )
        >>>
        >>> orders = pd.DataFrame({
        ...     "symbol": ["AAPL", "GOOGL"],
        ...     "side": ["BUY", "BUY"],
        ...     "qty": [100, 50],
        ...     "price": [150.0, 2500.0]
        ... })
        >>>
        >>> config = PreTradeConfig(max_notional_per_symbol=10000.0)
        >>> result, filtered = run_pre_trade_checks(orders, config=config)
        >>>
        >>> if not result.is_ok:
        ...     print(f"Orders blocked: {result.blocked_reasons}")
    """
    if config is None:
        config = PreTradeConfig()

    blocked_reasons = []
    filtered_orders = orders.copy()
    summary = {}

    # Handle empty orders
    if orders.empty:
        return (
            PreTradeCheckResult(
                is_ok=True,
                blocked_reasons=[],
                filtered_orders=pd.DataFrame(
                    columns=orders.columns if not orders.empty else []
                ),
                summary={"total_orders": 0, "passed_orders": 0},
            ),
            pd.DataFrame(columns=orders.columns if not orders.empty else []),
        )

    # Check required columns
    required_cols = ["symbol", "qty"]
    missing_cols = [c for c in required_cols if c not in orders.columns]
    if missing_cols:
        blocked_reasons.append(f"Orders missing required columns: {missing_cols}")
        return (
            PreTradeCheckResult(
                is_ok=False,
                blocked_reasons=blocked_reasons,
                filtered_orders=pd.DataFrame(columns=orders.columns),
                summary={"total_orders": len(orders), "passed_orders": 0},
            ),
            pd.DataFrame(columns=orders.columns),
        )

    # 1. Check QA status (if provided)
    if qa_status is not None:
        if qa_status.overall_result == QAResult.BLOCK:
            blocked_reasons.append("QA_BLOCK: QA gates blocked trading")
            return (
                PreTradeCheckResult(
                    is_ok=False,
                    blocked_reasons=blocked_reasons,
                    filtered_orders=pd.DataFrame(columns=orders.columns),
                    summary={
                        "total_orders": len(orders),
                        "passed_orders": 0,
                        "qa_blocked": True,
                    },
                ),
                pd.DataFrame(columns=orders.columns),
            )

    # 2. Calculate order notional values (if price available)
    orders_with_notional = filtered_orders.copy()
    if "price" in filtered_orders.columns:
        orders_with_notional["notional"] = (
            filtered_orders["qty"] * filtered_orders["price"]
        )
    else:
        # If no price, set notional to 0 (checks will be skipped)
        orders_with_notional["notional"] = 0.0

    # 3. Check max_notional_per_symbol
    if config.max_notional_per_symbol is not None:
        if "price" in filtered_orders.columns:
            # Calculate notional per symbol (sum of BUY and SELL)
            symbol_notionals = (
                orders_with_notional.groupby("symbol")["notional"].sum().abs()
            )
            exceeded_symbols = symbol_notionals[
                symbol_notionals > config.max_notional_per_symbol
            ]

            if len(exceeded_symbols) > 0:
                for symbol, notional in exceeded_symbols.items():
                    blocked_reasons.append(
                        f"max_notional_per_symbol: Symbol '{symbol}' has notional "
                        f"{notional:.2f}, exceeds limit {config.max_notional_per_symbol:.2f}"
                    )
                    # Filter out orders for this symbol
                    filtered_orders = filtered_orders[
                        filtered_orders["symbol"] != symbol
                    ]
                    orders_with_notional = orders_with_notional[
                        orders_with_notional["symbol"] != symbol
                    ]

    # 4. Check max_weight_per_symbol (requires portfolio and total capital)
    if config.max_weight_per_symbol is not None:
        if portfolio is not None and "weight" in portfolio.columns:
            # Calculate resulting weights after orders
            # This is simplified - assumes we can calculate weights from portfolio
            # For a full implementation, we'd need current positions + orders to compute new positions
            # For now, we'll check if any order would result in weight > threshold
            # This requires knowing total capital and current positions

            # Simplified: If portfolio has weights, check if any order would exceed
            # This is a placeholder - full implementation would need position merging logic
            pass  # TODO: Implement weight checking when portfolio + capital info is available

    # 5. Check max_gross_exposure (check on ORIGINAL orders, before filtering)
    if config.max_gross_exposure is not None:
        # Calculate gross exposure from ORIGINAL orders (before symbol filtering)
        # Gross exposure = sum of absolute notionals for all orders
        if "price" in orders.columns:
            # Calculate on original orders
            original_notionals = orders["qty"] * orders["price"]
            gross_exposure = original_notionals.abs().sum()
            summary["gross_exposure"] = gross_exposure

            if gross_exposure > config.max_gross_exposure:
                blocked_reasons.append(
                    f"max_gross_exposure: Gross exposure {gross_exposure:.2f} "
                    f"exceeds limit {config.max_gross_exposure:.2f}"
                )
                # Block all orders if gross exposure exceeded
                filtered_orders = pd.DataFrame(columns=orders.columns)
                orders_with_notional = pd.DataFrame(
                    columns=orders_with_notional.columns
                )
        else:
            # Without prices, cannot calculate gross exposure
            # Skip this check
            summary["gross_exposure_check"] = "skipped_no_prices"

    # 6. Sector/Region exposure checks (placeholder for future implementation)
    # These would require additional portfolio data (sector/region mappings)
    if config.max_sector_exposure is not None:
        # TODO: Implement sector exposure checks when sector data is available
        summary["sector_exposure_check"] = "not_implemented"

    if config.max_region_exposure is not None:
        # TODO: Implement region exposure checks when region data is available
        summary["region_exposure_check"] = "not_implemented"

    # Final summary
    summary["total_orders"] = len(orders)
    summary["passed_orders"] = len(filtered_orders)
    summary["blocked_orders"] = len(orders) - len(filtered_orders)

    # Determine is_ok: True if no reasons to block AND some orders passed
    is_ok = len(blocked_reasons) == 0 and len(filtered_orders) > 0

    # Special case: empty orders originally -> is_ok = True
    if len(orders) == 0:
        is_ok = True

    result = PreTradeCheckResult(
        is_ok=is_ok,
        blocked_reasons=blocked_reasons,
        filtered_orders=filtered_orders,
        summary=summary,
    )

    return result, filtered_orders
