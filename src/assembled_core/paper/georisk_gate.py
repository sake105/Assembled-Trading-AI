"""Minimal GeoRisk gate for paper track runner.

Scales target order quantities based on news_geo state:
- WATCH or missing → multiplier 1.0 (no change)
- ACTIVE → multiplier < 1.0 (reduce exposure)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def compute_georisk_multiplier(
    news_geo: Dict[str, Any] | None,
    *,
    active_multiplier: float = 0.70,
) -> float:
    """Derive exposure multiplier from news_geo state.

    Args:
        news_geo: Dict with at least 'state_hint' key, or None.
        active_multiplier: Multiplier when state_hint == "ACTIVE".

    Returns:
        Float in [0.0, 1.0].
    """
    if not news_geo or not isinstance(news_geo, dict):
        return 1.0

    state_hint = str(news_geo.get("state_hint", "WATCH")).upper()
    if state_hint == "ACTIVE":
        return max(0.0, min(float(active_multiplier), 1.0))

    return 1.0


def apply_georisk_to_orders(
    orders: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    """Scale order quantities by multiplier, preserving sign and structure.

    Args:
        orders: DataFrame with 'qty' column.
        multiplier: Float in [0.0, 1.0].

    Returns:
        Copy of orders with scaled qty. Rows with qty rounded to 0 are dropped.
    """
    if orders.empty or multiplier >= 1.0:
        return orders

    out = orders.copy()
    if "qty" in out.columns:
        out["qty"] = out["qty"] * multiplier
        out["qty"] = out["qty"].apply(lambda q: int(q) if abs(q) >= 0.5 else 0)
        out = out[out["qty"] != 0].reset_index(drop=True)
    return out
