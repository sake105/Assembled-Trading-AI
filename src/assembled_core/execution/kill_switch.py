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
from pathlib import Path

import pandas as pd

from src.assembled_core.logging_utils import setup_logging

logger = setup_logging(level="INFO")

# Default sentinel file path (relative to project root)
_DEFAULT_SENTINEL = Path("output/ops/.kill_switch_active")


def _sentinel_path() -> Path:
    """Return the kill switch sentinel file path."""
    override = os.environ.get("ASSEMBLED_KILL_SWITCH_SENTINEL", "")
    if override:
        return Path(override)
    return _DEFAULT_SENTINEL


def is_kill_switch_engaged() -> bool:
    """Check if kill switch is engaged via environment variable OR sentinel file.

    Returns True if any of the following conditions are met:
    1. Environment variable ASSEMBLED_KILL_SWITCH is set to a truthy value
       ("1", "true", "yes", "on" — case-insensitive)
    2. Sentinel file ``output/ops/.kill_switch_active`` exists (written by
       run_kill_switch_worker.py)

    Returns:
        True if kill switch is engaged, False otherwise
    """
    # --- Check 1: environment variable ---
    kill_switch_env = os.environ.get("ASSEMBLED_KILL_SWITCH", "").strip().lower()
    if kill_switch_env in {"1", "true", "yes", "on"}:
        return True

    # --- Check 2: sentinel file (written by run_kill_switch_worker.py) ---
    if _sentinel_path().exists():
        logger.warning(
            "KILL_SWITCH: sentinel file detected at %s", _sentinel_path()
        )
        return True

    return False


def check_drawdown_kill_switch(
    current_equity: float,
    peak_equity: float,
    kill_threshold: float = 0.30,
) -> bool:
    """Check whether current drawdown breaches the kill-switch threshold.

    If the drawdown exceeds kill_threshold, logs a CRITICAL message.
    Does NOT engage the kill switch automatically — call
    ``guard_orders_with_kill_switch`` after this to block orders.

    Args:
        current_equity: Current portfolio equity value.
        peak_equity: Highest equity value observed (high-water mark).
        kill_threshold: Drawdown fraction that triggers the kill flag (default 0.30 = 30%).

    Returns:
        True if drawdown >= kill_threshold.
    """
    if peak_equity <= 0 or current_equity <= 0:
        return False
    drawdown = (peak_equity - current_equity) / peak_equity
    if drawdown >= kill_threshold:
        logger.critical(
            "KILL_SWITCH: drawdown %.1f%% >= kill threshold %.1f%% "
            "(current=%.2f, peak=%.2f) — orders should be blocked",
            drawdown * 100,
            kill_threshold * 100,
            current_equity,
            peak_equity,
        )
        return True
    return False


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
