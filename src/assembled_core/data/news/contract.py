"""News data contract — normalisation and PIT filtering."""

from __future__ import annotations

import hashlib

import pandas as pd

_REQUIRED_COLS = {"publish_ts", "source"}
_IDENTIFIER_COLS = {"headline", "url", "provider_id"}
_OPTIONAL_COLS = [
    "symbol",
    "symbols",
    "headline",
    "url",
    "provider_id",
    "ingest_ts",
    "revised_ts",
    "sentiment",
    "raw_url",
]
_TS_COLS = ["publish_ts", "ingest_ts", "revised_ts"]


def normalize_news_events(
    events: pd.DataFrame,
    dedupe_keep: str = "last",
) -> pd.DataFrame:
    """Normalise news events to contract schema.

    Required columns: publish_ts, source, and at least one of headline/url/provider_id.

    Raises:
        ValueError: If required columns or identifiers are missing,
                    or if temporal sanity checks fail.
    """
    missing = _REQUIRED_COLS - set(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not (_IDENTIFIER_COLS & set(events.columns)):
        raise ValueError(
            "Missing identifier: at least one of headline, url, or provider_id required"
        )

    df = events.copy()

    # Add missing optional columns
    for col in _OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None

    # Trim strings
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Normalise timestamps
    for col in _TS_COLS:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            if ts.dt.tz is None:
                ts = ts.dt.tz_localize("UTC")
            else:
                ts = ts.dt.tz_convert("UTC")
            df[col] = ts

    if df.empty:
        return df[["publish_ts", "source"] + _OPTIONAL_COLS]

    # Temporal sanity: ingest_ts must not precede publish_ts
    if "ingest_ts" in df.columns and not df["ingest_ts"].isna().all():
        if (df["ingest_ts"] < df["publish_ts"]).any():
            raise ValueError("publish_ts in future relative to ingest_ts")

    # Temporal sanity: revised_ts must not precede publish_ts
    if "revised_ts" in df.columns and not df["revised_ts"].isna().all():
        if (df["revised_ts"] < df["publish_ts"]).any():
            raise ValueError("revised_ts < publish_ts")

    # Deduplication key
    if "provider_id" in df.columns and not df["provider_id"].isna().all():
        dedupe_key = ["publish_ts", "source", "provider_id"]
    else:
        df["_hash"] = df.apply(
            lambda r: hashlib.md5(
                f"{r.get('publish_ts')}{r.get('source')}{r.get('headline')}".encode(),
                usedforsecurity=False,
            ).hexdigest(),
            axis=1,
        )
        dedupe_key = ["_hash"]

    df = df.drop_duplicates(subset=dedupe_key, keep=dedupe_keep)
    if "_hash" in df.columns:
        df = df.drop(columns=["_hash"])

    df = df.sort_values("publish_ts").reset_index(drop=True)
    return df[["publish_ts", "source"] + _OPTIONAL_COLS]


def filter_news_pit(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Return news events with publish_ts <= as_of.

    Raises:
        ValueError: If publish_ts column is missing.
    """
    if "publish_ts" not in df.columns:
        raise ValueError("publish_ts column required for PIT filtering")

    ts = pd.to_datetime(df["publish_ts"], utc=True)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")

    return df[ts <= as_of].reset_index(drop=True)
