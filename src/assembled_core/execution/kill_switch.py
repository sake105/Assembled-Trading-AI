"""Kill switch for emergency order blocking.

This module provides a simple kill switch mechanism to immediately block all orders
in emergency situations. The kill switch is controlled via environment variable
ASSEMBLED_KILL_SWITCH.

Key features:
- Environment variable-based activation
- Zero I/O side effects (no files, no DB)
- Simple, testable, and fast
- Clear logging when engaged

Usage:
    >>> from src.assembled_core.execution.kill_switch import (
    ...     is_kill_switch_engaged,
    ...     guard_orders_with_kill_switch
    ... )
    >>> import pandas as pd
    >>>
    >>> orders = pd.DataFrame({
    ...     "symbol": ["AAPL"],
    ...     "side": ["BUY"],
    ...     "qty": [100]
    ... })
    >>>
    >>> filtered_orders = guard_orders_with_kill_switch(orders)
    >>> if filtered_orders.empty and not orders.empty:
    ...     print("Kill switch is engaged - all orders blocked")
"""

from __future__ import annotations

import os

import pandas as pd

from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")


def is_kill_switch_engaged() -> bool:
    """Check if kill switch is engaged via environment variable.

    Reads environment variable ASSEMBLED_KILL_SWITCH and returns True if:
    - Variable is set to "1", "true", "True", "TRUE", "yes", "Yes", "YES", "on", "On", "ON"
    - Case-insensitive comparison

    Returns:
        True if kill switch is engaged, False otherwise

    Example:
        >>> import os
        >>> os.environ["ASSEMBLED_KILL_SWITCH"] = "1"
        >>> is_kill_switch_engaged()
        True
        >>>
        >>> os.environ.pop("ASSEMBLED_KILL_SWITCH", None)
        >>> is_kill_switch_engaged()
        False
    """
    kill_switch_env = os.environ.get("ASSEMBLED_KILL_SWITCH", "").strip().lower()

    # Accepted values for "engaged"
    engaged_values = {"1", "true", "yes", "on"}

    return kill_switch_env in engaged_values


def guard_orders_with_kill_switch(orders: pd.DataFrame) -> pd.DataFrame:
    """Guard orders with kill switch - return empty DataFrame if kill switch is engaged.

    If kill switch is engaged, all orders are blocked and an empty DataFrame is returned.
    A warning is logged to indicate that orders were blocked due to kill switch.

    Args:
        orders: DataFrame with orders (any structure)

    Returns:
        Original orders DataFrame if kill switch is not engaged,
        Empty DataFrame with same columns if kill switch is engaged

    Example:
        >>> import pandas as pd
        >>> import os
        >>>
        >>> orders = pd.DataFrame({
        ...     "symbol": ["AAPL", "GOOGL"],
        ...     "side": ["BUY", "SELL"],
        ...     "qty": [100, 50]
        ... })
        >>>
        >>> # Normal operation
        >>> filtered = guard_orders_with_kill_switch(orders)
        >>> assert len(filtered) == 2
        >>>
        >>> # Kill switch engaged
        >>> os.environ["ASSEMBLED_KILL_SWITCH"] = "1"
        >>> filtered = guard_orders_with_kill_switch(orders)
        >>> assert len(filtered) == 0
    """
    if is_kill_switch_engaged():
        logger.warning(
            "KILL_SWITCH: All orders blocked - ASSEMBLED_KILL_SWITCH environment variable is set"
        )
        # Return empty DataFrame with same columns as original
        return pd.DataFrame(columns=list(orders.columns))

    # Kill switch not engaged - return orders unchanged
    return orders
