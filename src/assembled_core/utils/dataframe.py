"""DataFrame utility functions (shared across layers)."""

from __future__ import annotations

import pandas as pd


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ensure required columns exist in DataFrame.

    Args:
        df: Input DataFrame
        cols: List of required column names

    Returns:
        DataFrame with validated columns

    Raises:
        KeyError: If any required column is missing
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Fehlende Spalten: {missing} | vorhanden={df.columns.tolist()}")
    return df


def coerce_price_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce price DataFrame to correct types.

    Args:
        df: DataFrame with price data

    Returns:
        DataFrame with coerced types (timestamp UTC, close float64, symbol string)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string")
    df = df.dropna(subset=["timestamp", "close"])
    return df
