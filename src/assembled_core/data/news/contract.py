"""News data contract stub."""

from __future__ import annotations

import pandas as pd


def normalize_news_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise news events."""
    df = events.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
