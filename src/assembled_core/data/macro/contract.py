"""Macro data contract: normalisation and PIT filtering."""

from __future__ import annotations

import pandas as pd

_REQUIRED_COLS = {"series_id", "release_ts", "available_ts", "value"}
_STR_COLS = ["series_id", "country", "currency", "source", "metric"]
_STANDARD_COLS = [
    "series_id",
    "release_ts",
    "available_ts",
    "value",
    "country",
    "currency",
    "source",
    "revision_id",
    "metric",
]


def normalize_macro_releases(
    releases: pd.DataFrame,
    dedupe_keep: str | None = "first",
) -> pd.DataFrame:
    """Normalise macro releases to common schema.

    Args:
        releases: Raw macro releases DataFrame.
        dedupe_keep: How to handle duplicate rows. Passed to
            ``DataFrame.drop_duplicates(keep=...)``. Deduplication key is
            ``(series_id, release_ts)``; ``revision_id`` is included when
            the column exists so distinct revisions are preserved.
            Set to ``None`` to skip deduplication.

    Raises:
        ValueError: If required columns are missing.
        ValueError: If any row has available_ts < release_ts.
    """
    df = releases.copy()

    # Promote 'metric' → 'series_id' if series_id is absent
    if "series_id" not in df.columns and "metric" in df.columns:
        df["series_id"] = df["metric"]
    elif "series_id" in df.columns and "metric" not in df.columns:
        df["metric"] = df["series_id"]

    # Validate required columns
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Parse timestamps to UTC
    for col in ["release_ts", "available_ts", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Sanity check: available_ts must be >= release_ts
    if not df.empty:
        mask = df["available_ts"].notna() & df["release_ts"].notna()
        bad = mask & (df["available_ts"] < df["release_ts"])
        if bad.any():
            raise ValueError(
                "available_ts < release_ts for some rows — data integrity violation"
            )

    # Trim string columns
    for col in _STR_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()

    # Add missing standard columns as None
    for col in _STANDARD_COLS:
        if col not in df.columns:
            df[col] = None

    # Deduplication
    if dedupe_keep is not None:
        key_cols = [
            c for c in ["series_id", "release_ts", "revision_id"] if c in df.columns
        ]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep=dedupe_keep).reset_index(
                drop=True
            )

    # Deterministic sort
    sort_cols = [c for c in ["series_id", "release_ts"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def filter_macro_pit(
    releases: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Return macro releases known at *as_of* (PIT filter on available_ts).

    Does minimal timestamp parsing only — does not apply the full
    normalize_macro_releases sanity checks, so partial/raw data is accepted.

    Raises:
        ValueError: If available_ts column is missing.
    """
    if "available_ts" not in releases.columns:
        raise ValueError(
            "available_ts column is required for PIT filtering but was not found"
        )
    df = releases.copy()
    df["available_ts"] = pd.to_datetime(df["available_ts"], utc=True, errors="coerce")
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    return df[df["available_ts"] <= as_of].copy()
