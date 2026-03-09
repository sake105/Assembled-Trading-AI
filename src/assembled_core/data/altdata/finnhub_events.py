"""Finnhub corporate events client stub."""

from __future__ import annotations

import pandas as pd


def fetch_finnhub_events(symbols: list[str] | None = None) -> pd.DataFrame:
    """Fetch corporate events from Finnhub API (stub)."""
    return pd.DataFrame(
        columns=["timestamp", "symbol", "event_type", "description"]
    )
