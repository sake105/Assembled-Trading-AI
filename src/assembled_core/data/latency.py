"""Point-in-time latency helpers for alt-data events.

Ensures alt-data events respect filing/publication delays
so backtests don't suffer from look-ahead bias.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_event_schema(
    events: pd.DataFrame,
    required_cols: list[str] | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Ensure events have the required columns.

    Parameters
    ----------
    events:
        Input DataFrame.
    required_cols:
        Columns that must (or will be created if) exist.
        Defaults to ``["timestamp", "symbol"]`` when *None*.
    strict:
        If True, raise ValueError on missing columns.
        If False, create missing columns with sensible defaults.
    """
    if required_cols is None:
        required_cols = ["timestamp", "symbol"]

    # Handle empty DataFrame: just add missing columns with correct dtypes.
    if events.empty:
        df = events.copy()
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df

    df = events.copy()
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        if strict:
            raise ValueError(f"Missing required columns: {missing}")
        # Non-strict: create missing columns with defaults.
        for col in missing:
            if col == "disclosure_date" and "timestamp" in df.columns:
                # Derive disclosure_date from timestamp (no extra latency).
                df["disclosure_date"] = pd.to_datetime(
                    df["timestamp"], utc=True
                ).dt.normalize()
            else:
                df[col] = None

    # Normalize timestamp if present.
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    return df


def apply_source_latency(
    events: pd.DataFrame,
    latency_days: int | None = None,
    days: int | None = None,
    timestamp_col: str = "timestamp",
    event_date_col: str | None = None,
    mode: str = "derive",
) -> pd.DataFrame:
    """Create or shift *disclosure_date* to model publication delay.

    Parameters
    ----------
    events:
        Input DataFrame.
    latency_days:
        Alias for *days* (kept for backwards compatibility).
    days:
        Number of days to add.  Takes precedence over *latency_days*.
    timestamp_col:
        Fallback source column when *event_date_col* is absent.
    event_date_col:
        Primary source column for deriving *disclosure_date*.
    mode:
        ``"derive"`` (default) — compute *disclosure_date* from source column.
        ``"shift"`` — shift an existing *disclosure_date* column forward.
    """
    # Resolve latency value: `days` wins over `latency_days`.
    if days is not None:
        n_days = days
    elif latency_days is not None:
        n_days = latency_days
    else:
        n_days = 1

    df = events.copy()
    delta = pd.Timedelta(days=n_days)

    if mode == "shift":
        # Shift existing disclosure_date in place.
        df["disclosure_date"] = pd.to_datetime(df["disclosure_date"], utc=True) + delta
        return df

    # mode == "derive": build disclosure_date from source column.
    if event_date_col is not None and event_date_col in df.columns:
        source = pd.to_datetime(df[event_date_col], utc=True)
    elif timestamp_col in df.columns:
        source = pd.to_datetime(df[timestamp_col], utc=True)
    else:
        raise ValueError(
            f"Neither event_date_col={event_date_col!r} nor timestamp_col={timestamp_col!r} "
            "found in DataFrame."
        )

    # Normalize to midnight UTC then add latency.
    df["disclosure_date"] = source.dt.normalize() + delta
    return df


def filter_events_as_of(
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    timestamp_col: str = "timestamp",
    disclosure_col: str | None = None,
    event_date_col: str | None = None,
    fallback_to_event_date: bool = True,
) -> pd.DataFrame:
    """Return only events known at *as_of* (point-in-time safe).

    Parameters
    ----------
    events:
        Input DataFrame.
    as_of:
        Cut-off timestamp (inclusive).
    timestamp_col:
        Legacy default filter column when *disclosure_col* is not given.
    disclosure_col:
        Preferred column to filter on (e.g. ``"disclosure_date"``).
    event_date_col:
        Fallback column when *disclosure_col* is absent from the DataFrame.
    fallback_to_event_date:
        If True and *disclosure_col* is missing, try *event_date_col*.
        If False, raise ValueError when *disclosure_col* is missing.
    """
    # Ensure as_of is timezone-aware (UTC).
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    # Determine which column to filter on.
    if disclosure_col is not None:
        if disclosure_col in events.columns:
            filter_col = disclosure_col
        elif (
            fallback_to_event_date
            and event_date_col is not None
            and event_date_col in events.columns
        ):
            filter_col = event_date_col
        elif not fallback_to_event_date:
            raise ValueError(
                f"Cannot filter events: disclosure column '{disclosure_col}' not found "
                "and fallback_to_event_date=False."
            )
        else:
            raise ValueError(
                f"Cannot filter events: disclosure column '{disclosure_col}' not found "
                "and no valid fallback column available."
            )
    else:
        filter_col = timestamp_col

    col_series = pd.to_datetime(events[filter_col], utc=True)
    # Normalize as_of to midnight for date-level comparison.
    as_of_norm = as_of.normalize()
    return events[col_series.dt.normalize() <= as_of_norm].copy()
