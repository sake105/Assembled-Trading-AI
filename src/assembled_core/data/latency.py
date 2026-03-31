"""Point-in-time latency helpers for alt-data events.

Ensures alt-data events respect filing/publication delays
so backtests don't suffer from look-ahead bias.
"""

from __future__ import annotations

import pandas as pd


def ensure_event_schema(events: pd.DataFrame) -> pd.DataFrame:
    """Ensure events have the required columns (timestamp, symbol, event_type)."""
    required = ["timestamp", "symbol"]
    missing = [c for c in required if c not in events.columns]
    if missing:
        raise ValueError(f"Missing event columns: {missing}")
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    return events


def apply_source_latency(
    events: pd.DataFrame,
    latency_days: int = 1,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Shift event timestamps forward by *latency_days* to model publication delay."""
    df = events.copy()
    df[timestamp_col] = df[timestamp_col] + pd.Timedelta(days=latency_days)
    return df


def filter_events_as_of(
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Return only events known at *as_of* (point-in-time safe)."""
    return events[events[timestamp_col] <= as_of].copy()
