"""Macro data contract: normalisation and PIT filtering."""

from __future__ import annotations

import pandas as pd


def normalize_macro_releases(releases: pd.DataFrame) -> pd.DataFrame:
    """Normalise macro releases to common schema."""
    df = releases.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def filter_macro_pit(
    releases: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Return macro releases known at *as_of*."""
    df = normalize_macro_releases(releases)
    if "timestamp" in df.columns:
        return df[df["timestamp"] <= as_of].copy()
    return df
