"""Shipping release data contract — normalisation and PIT filtering."""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"release_ts", "available_ts", "value"}
_OPTIONAL_COLS = ["region", "source", "revision_id", "metric"]
_TS_COLS = ["release_ts", "available_ts"]


def normalize_shipping_releases(
    df: pd.DataFrame,
    dedupe_keep: str = "last",
) -> pd.DataFrame:
    """Normalise shipping release events to contract schema.

    Required columns: series_id (or metric as alias), release_ts, available_ts, value.

    Raises:
        ValueError: If required columns are missing or temporal sanity fails.
    """
    out = df.copy()

    # Accept 'metric' as alias for 'series_id'
    if "series_id" not in out.columns and "metric" in out.columns:
        out["series_id"] = out["metric"]

    all_required = _REQUIRED_COLS | {"series_id"}
    missing = all_required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Add missing optional columns
    for col in _OPTIONAL_COLS:
        if col not in out.columns:
            out[col] = None

    # Trim strings
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Normalise timestamps
    for col in _TS_COLS:
        if col in out.columns:
            ts = pd.to_datetime(out[col], errors="coerce")
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            else:
                ts = ts.dt.tz_convert("UTC")
            out[col] = ts

    if out.empty:
        return out[
            ["series_id", "release_ts", "available_ts", "value"] + _OPTIONAL_COLS
        ]

    # Temporal sanity: available_ts must not precede release_ts
    if (out["available_ts"] < out["release_ts"]).any():
        raise ValueError("available_ts < release_ts")

    # Deduplication key
    if "revision_id" in out.columns and not out["revision_id"].isna().all():
        dedupe_key = ["series_id", "release_ts", "available_ts", "revision_id"]
    else:
        dedupe_key = ["series_id", "release_ts", "available_ts"]

    out = out.drop_duplicates(subset=dedupe_key, keep=dedupe_keep)
    out = out.sort_values("release_ts").reset_index(drop=True)

    return out[["series_id", "release_ts", "available_ts", "value"] + _OPTIONAL_COLS]


def filter_shipping_pit(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Return shipping releases with available_ts <= as_of.

    Raises:
        ValueError: If available_ts column is missing.
    """
    if "available_ts" not in df.columns:
        raise ValueError("available_ts column required for PIT filtering")

    ts = pd.to_datetime(df["available_ts"], utc=True)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    return df[ts <= as_of].reset_index(drop=True)


# Keep backward-compatible stub function
def normalize_shipping_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise shipping events (legacy interface)."""
    out = events.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    return out
