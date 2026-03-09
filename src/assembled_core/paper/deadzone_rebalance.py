"""Dead-zone rebalance filter for paper track runner.

Suppresses orders where the change relative to the current position
is below a configurable threshold, eliminating micro-rebalancing churn.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

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

    pos_map: Dict[str, float] = {}
    for _, row in current_positions.iterrows():
        sym = row["symbol"]
        qty = abs(float(row.get("qty", 0)))
        if qty > 0:
            pos_map[sym] = qty

    keep_mask = []
    for _, order in orders.iterrows():
        sym = order["symbol"]
        order_qty = abs(float(order["qty"]))
        current_qty = pos_map.get(sym, 0.0)

        if current_qty == 0:
            keep_mask.append(True)
            continue

        denominator = max(current_qty, order_qty)
        ratio = order_qty / denominator if denominator > 0 else 0.0

        keep_mask.append(ratio >= deadzone_pct)

    filtered = orders[keep_mask].reset_index(drop=True)

    stats["orders_after"] = len(filtered)
    stats["orders_dropped"] = len(orders) - len(filtered)

    return filtered, stats
