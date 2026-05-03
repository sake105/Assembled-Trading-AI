"""Dead-zone rebalance filter for paper track runner.

Suppresses orders where the change relative to the current position
is below a configurable threshold, eliminating micro-rebalancing churn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_deadzone_orders(
    orders: pd.DataFrame,
    current_positions: pd.DataFrame | None = None,
    *,
    deadzone_pct: float = 0.05,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop orders where qty is small relative to the existing position.

    For each order, the relative change is:
        ratio = order_qty / max(current_qty, order_qty)

    If ratio < deadzone_pct the order is suppressed.
    New positions (no current holding) always pass through.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price.
        current_positions: DataFrame with columns: symbol, qty (current holdings).
        deadzone_pct: Minimum relative change to keep an order (default 5%).

    Returns:
        Tuple of (filtered_orders, stats_dict).
    """
    stats: Dict[str, Any] = {
        "orders_before": len(orders),
        "orders_after": len(orders),
        "orders_dropped": 0,
        "deadzone_pct": deadzone_pct,
    }

    if orders.empty or deadzone_pct <= 0:
        return orders, stats

    if current_positions is None or current_positions.empty:
        return orders, stats

    pos_s = pd.to_numeric(current_positions["qty"], errors="coerce").abs().fillna(0.0)
    valid = pos_s > 0
    pos_map: Dict[str, float] = dict(zip(current_positions.loc[valid, "symbol"], pos_s[valid]))

    order_qty = pd.to_numeric(orders["qty"], errors="coerce").abs().fillna(0.0)
    current_qty = orders["symbol"].map(pos_map).fillna(0.0)
    has_pos = current_qty > 0
    denom = np.maximum(current_qty.values, order_qty.values)
    ratio = np.where(denom > 0, order_qty.values / denom, 0.0)
    keep_mask = ~has_pos.values | (ratio >= deadzone_pct)

    filtered = orders[keep_mask].reset_index(drop=True)

    stats["orders_after"] = len(filtered)
    stats["orders_dropped"] = len(orders) - len(filtered)

    return filtered, stats
