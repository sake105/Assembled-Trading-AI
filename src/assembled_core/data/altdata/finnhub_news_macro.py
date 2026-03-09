"""Finnhub news/macro data client stub."""

from __future__ import annotations

import pandas as pd


def fetch_finnhub_news(symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch news from Finnhub API (stub)."""
    return pd.DataFrame(columns=["timestamp", "symbol", "headline", "sentiment"])


def fetch_finnhub_macro() -> pd.DataFrame:
    """Fetch macro data from Finnhub (stub)."""
    return pd.DataFrame(columns=["timestamp", "indicator", "value"])
