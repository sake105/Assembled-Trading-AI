"""Corporate actions (splits, dividends) stub module."""

from __future__ import annotations

import pandas as pd


def load_corporate_actions(path: str | None = None) -> pd.DataFrame:
    """Load corporate actions from CSV. Returns empty DataFrame if missing."""
    if path is None:
        return pd.DataFrame(columns=["symbol", "date", "action_type", "factor"])
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "date", "action_type", "factor"])


def adjust_prices_for_splits(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Adjust prices for stock splits (no-op if no actions)."""
    if actions.empty:
        return prices
    return prices.copy()
