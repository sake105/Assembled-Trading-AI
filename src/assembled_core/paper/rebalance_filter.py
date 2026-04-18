"""Minimal rebalance filter to suppress small order churn.

Drops orders where the quantity is too small relative to a threshold,
reducing unnecessary turnover from daily micro-rebalances.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def filter_small_rebalances(
    orders: pd.DataFrame,
    *,
    min_notional: float = 500.0,
    prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Drop orders with notional value below threshold.

    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price.
        min_notional: Minimum order notional (qty × price) to keep.
        prices: Optional latest prices (symbol, close) for notional calc
                when order price is 0.

    Returns:
        Tuple of (filtered_orders, stats_dict).
    """
    stats: Dict[str, Any] = {
        "orders_before": len(orders),
        "orders_after": len(orders),
        "orders_dropped": 0,
        "min_notional": min_notional,
    }

    if orders.empty or min_notional <= 0:
        return orders, stats

    df = orders.copy()

    if "price" in df.columns:
        order_price = df["price"].fillna(0.0)
    else:
        order_price = pd.Series(0.0, index=df.index)

    if prices is not None and not prices.empty:
        price_map = dict(zip(prices["symbol"], prices["close"]))
        # Symbols missing from the feed used to silently resolve to price=0.0
        # and therefore notional=0, which dropped the order below min_notional
        # with no trace — exactly the names mid-event (halts, data-feed
        # partial delivery) were most likely to be suppressed. Warn on
        # fallback so the operator sees which symbols are mid-event.
        missing = set(df["symbol"]) - set(price_map.keys())
        if missing:
            logger.warning(
                "[rebalance_filter] %d symbol(s) absent from price feed — "
                "notional defaults to 0 and order may be dropped: %s",
                len(missing),
                sorted(missing)[:10],
            )
        order_price = df.apply(
            lambda r: (
                r.get("price", 0.0)
                if r.get("price", 0.0) > 0
                else price_map.get(r["symbol"], 0.0)
            ),
            axis=1,
        )

    notional = df["qty"].abs() * order_price.abs()
    keep = notional >= min_notional
    filtered = df[keep].reset_index(drop=True)

    stats["orders_after"] = len(filtered)
    stats["orders_dropped"] = len(df) - len(filtered)

    return filtered, stats
