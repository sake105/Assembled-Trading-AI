"""Alt-data contract: normalisation and PIT filtering for events."""

from __future__ import annotations

import pandas as pd


def normalize_alt_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise alt-data events to a common schema.

    Ensures columns: timestamp (UTC), symbol, event_type, value.
    """
    df = events.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "event_type" not in df.columns:
        df["event_type"] = "unknown"
    if "value" not in df.columns:
        df["value"] = 0.0
    return df


def filter_events_pit(
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    latency_days: int = 0,
) -> pd.DataFrame:
    """Filter events to those known at *as_of* minus publication latency."""
    df = normalize_alt_events(events)
    cutoff = as_of - pd.Timedelta(days=latency_days)
    return df[df["timestamp"] <= cutoff].copy()
