"""DataFrame utility functions (shared across layers)."""

from __future__ import annotations

from typing import Union

import numpy as np
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
    n_before = len(df)
    df = df.dropna(subset=["timestamp", "close"])
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "[coerce_price_types] dropped %d/%d rows with invalid timestamp or close",
            n_dropped,
            n_before,
        )
    return df


# ---------------------------------------------------------------------------
# Item 47: safe division helpers
# ---------------------------------------------------------------------------

_Numeric = Union[float, int, "np.floating", "np.integer"]


def safe_divide(
    numerator: Union["_Numeric", "np.ndarray", "pd.Series"],
    denominator: Union["_Numeric", "np.ndarray", "pd.Series"],
    default: float = 0.0,
) -> Union[float, "np.ndarray", "pd.Series"]:
    """Divide *numerator* by *denominator*, returning *default* when denom is 0 or NaN.

    Works element-wise for scalars, numpy arrays, and pandas Series.

    Args:
        numerator:   Dividend — scalar, ndarray, or Series.
        denominator: Divisor  — scalar, ndarray, or Series.
        default:     Value to return wherever denominator is 0 or NaN (default 0.0).

    Returns:
        Scalar float, ndarray, or Series depending on the input types.
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        denom = np.asarray(denominator, dtype=float)
        numer = np.asarray(numerator, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(
                (denom == 0) | ~np.isfinite(denom),
                default,
                numer / np.where(denom == 0, 1.0, denom),
            )
        if isinstance(denominator, pd.Series):
            return pd.Series(result, index=denominator.index)
        return result

    # Scalar path
    den = float(denominator) if denominator is not None else float("nan")
    if den == 0.0 or not np.isfinite(den):
        return default
    num = float(numerator) if numerator is not None else float("nan")
    return num / den
