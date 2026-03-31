"""Alt-data contract: normalisation and PIT filtering for events (Sprint 10.A)."""

from __future__ import annotations

from typing import Final, List

import pandas as pd

# Public schema constants (mirrored in tests)
REQUIRED_COLUMNS: Final[List[str]] = [
    "symbol",
    "event_date",
    "disclosure_date",
    "effective_date",
]
OPTIONAL_COLUMNS: Final[List[str]] = ["event_type", "source", "value", "is_public"]


def _ensure_utc(dt: pd.Series) -> pd.Series:
    """Normalize a datetime-like series to UTC tz-aware."""
    if dt.empty:
        return dt
    return pd.to_datetime(dt, utc=True)


def normalize_alt_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise alt-data events to a common schema.

    Contract (see tests/test_alt_data_contract.py):
    - Schema: REQUIRED_COLUMNS always present after normalization.
    - Timestamps: event_date, disclosure_date, effective_date are UTC tz-aware.
    - Fallbacks: effective_date falls back to disclosure_date when missing/NaT.
    - Constraints: disclosure_date >= event_date and effective_date >= disclosure_date.
    - Deterministic sorting and deduplication by (symbol, event_date, disclosure_date, effective_date).
    - Optional columns preserved; string columns trimmed; is_public policy enforced.
    """
    df = events.copy()

    # Empty input: return empty frame with required + optional columns
    if df.empty and len(df.columns) == 0:
        cols = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        return pd.DataFrame(columns=cols)

    # Basic required columns for normalization
    missing_symbol_event = [c for c in ("symbol", "event_date") if c not in df.columns]
    if missing_symbol_event:
        raise ValueError(f"Missing required columns: {missing_symbol_event}")

    has_disclosure = "disclosure_date" in df.columns
    has_effective = "effective_date" in df.columns

    if not has_disclosure and not has_effective:
        # Neither disclosure_date nor effective_date present
        # Message must satisfy both "Missing required columns" and
        # "disclosure_date is mandatory for PIT-safe filtering" tests.
        raise ValueError(
            "Missing required columns; disclosure_date is mandatory for PIT-safe filtering"
        )

    if not has_disclosure and has_effective:
        # PIT filtering requires explicit disclosure_date
        raise ValueError("disclosure_date is mandatory for PIT-safe filtering")

    # Ensure disclosure_date column exists (for later use); at this point it must
    # be present, otherwise previous branch would have raised.
    # Effective_date may be missing and is handled by fallback below.

    # Datetime normalization to UTC
    for col in ("event_date", "disclosure_date", "effective_date"):
        if col in df.columns:
            df[col] = _ensure_utc(df[col])

    # effective_date fallback: create/patch from disclosure_date
    if "effective_date" not in df.columns:
        df["effective_date"] = df["disclosure_date"]
    else:
        # Fill NaT values from disclosure_date
        mask_nat = df["effective_date"].isna()
        if mask_nat.any():
            df.loc[mask_nat, "effective_date"] = df.loc[mask_nat, "disclosure_date"]

    # Enforce ordering constraints
    # disclosure_date >= event_date
    mask_bad_disclosure = df["disclosure_date"] < df["event_date"]
    if mask_bad_disclosure.any():
        raise ValueError("disclosure_date < event_date")

    # effective_date >= disclosure_date
    mask_bad_effective = df["effective_date"] < df["disclosure_date"]
    if mask_bad_effective.any():
        raise ValueError("effective_date < disclosure_date")

    # String trimming for well-known string columns
    for col in ("symbol", "event_type", "source"):
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # Public Disclosures Only policy via is_public
    if "is_public" in df.columns:
        col = df["is_public"]
        # Any explicit False -> violation
        if (col == False).any():  # noqa: E712
            raise ValueError("Public Disclosures Only policy violated")

    # Deterministic sort
    sort_keys = ["symbol", "event_date", "disclosure_date", "effective_date"]
    df = df.sort_values(sort_keys, kind="mergesort")

    # Deterministic deduplication: keep first occurrence for identical (symbol, dates)
    df = df.drop_duplicates(subset=sort_keys, keep="first")

    # Ensure all required columns exist in final frame
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            # Should not normally happen given the logic above, but keep contract explicit
            df[col] = pd.NaT if col.endswith("_date") else None

    # Preserve optional columns; add missing optional ones with appropriate dtypes
    if "event_type" not in df.columns:
        df["event_type"] = "unknown"
    if "source" not in df.columns:
        df["source"] = pd.Series([None] * len(df), index=df.index, dtype="object")
    if "value" not in df.columns:
        df["value"] = 0.0
    if "is_public" not in df.columns:
        df["is_public"] = pd.Series([True] * len(df), index=df.index, dtype="boolean")

    # Reorder columns: required first, then the rest (stable order)
    remaining = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[REQUIRED_COLUMNS + remaining]

    return df


def filter_events_pit(
    events: pd.DataFrame,
    as_of: pd.Timestamp,
    latency_days: int = 0,
) -> pd.DataFrame:
    """Filter events to those known at *as_of* minus publication latency.

    Uses disclosure_date as the publication date boundary. Input can be raw or
    already normalized; normalize_alt_events is applied first to enforce schema.
    """
    if "disclosure_date" not in events.columns:
        raise ValueError("Missing required column 'disclosure_date'")

    df = normalize_alt_events(events)
    if df.empty:
        return df

    cutoff = as_of - pd.Timedelta(days=latency_days)
    mask = df["disclosure_date"] <= cutoff
    return df[mask].copy()


__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "normalize_alt_events",
    "filter_events_pit",
]
