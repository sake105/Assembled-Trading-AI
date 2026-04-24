"""Fractional Differentiation (Lopez de Prado, AIFML Chapter 5).

Preserves memory while achieving (near) stationarity.
Traditional diff (d=1) destroys most memory; Fractional d ∈ (0, 1) keeps it.

Algorithm — Fixed-Window Fractional Differentiation (FFD):
1. Compute weights w_k = (-1)^k * Γ(d+1) / (k! * Γ(d-k+1))
2. Truncate at threshold (|w_k| < tau) for fixed window
3. Apply weights to rolling window

Usage:
    # Test stationarity: find minimum d that makes series stationary
    d_opt = find_optimal_d(prices)  # e.g., 0.37
    prices_ffd = frac_diff_ffd(prices, d=d_opt)
    # prices_ffd is stationary AND retains memory of long-term levels

PIT-Invariante:
- FFD verwendet nur vergangene Werte → PIT-safe per Design
- Erste `window_size` Zeilen sind NaN (Initialisierungsfenster)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def frac_diff_weights(d: float, threshold: float = 1e-4, max_size: int = 500) -> np.ndarray:
    """Berechnet Fractional-Diff-Gewichte w_k bis |w_k| < threshold.

    w_0 = 1
    w_k = -w_{k-1} * (d - k + 1) / k

    Args:
        d: Fractional-Diff-Ordnung (typ. 0.1–0.9)
        threshold: Abbruch-Threshold für kleine Gewichte
        max_size: Hard-Max um unendliche Loops zu verhindern

    Returns:
        Array der Gewichte, sortiert [w_0, w_1, w_2, …]
    """
    weights = [1.0]
    for k in range(1, max_size):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights, dtype=np.float64)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    threshold: float = 1e-4,
) -> pd.Series:
    """Fixed-Window Fractional Differentiation.

    Fix-Window = gleiches Gewichts-Array für alle Zeilen, kein expanding window.
    Nur Werte wo vollständiges Fenster verfügbar ist → Rest NaN.

    Args:
        series: Zeitreihe (sortiert)
        d: Fractional-Diff-Ordnung (0 < d < 1 typisch)
        threshold: Gewichts-Threshold für Window-Größe

    Returns:
        pd.Series gleicher Länge, erste `window_size - 1` Zeilen NaN.
    """
    weights = frac_diff_weights(d, threshold=threshold)
    window_size = len(weights)

    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)

    # Flip weights for dot-product convention
    w_flipped = weights[::-1]

    for i in range(window_size - 1, n):
        window = values[i - window_size + 1 : i + 1]
        if np.isnan(window).any():
            continue
        result[i] = float(w_flipped @ window)

    return pd.Series(result, index=series.index, name=f"{series.name}_ffd_{d:.2f}")


def adf_stationarity_test(series: pd.Series) -> dict:
    """ADF-Test für Stationarität. Graceful degradation wenn statsmodels fehlt.

    Returns:
        dict mit {"statistic", "pvalue", "is_stationary"} oder {"error": ...}
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except ImportError:
        return {"error": "statsmodels not installed"}

    clean = series.dropna()
    if len(clean) < 20:
        return {"error": "too few observations"}

    try:
        stat, pvalue, *_ = adfuller(clean, autolag="AIC")
        return {
            "statistic": float(stat),
            "pvalue": float(pvalue),
            "is_stationary": bool(pvalue < 0.05),
        }
    except Exception as exc:
        return {"error": str(exc)}


def find_optimal_d(
    series: pd.Series,
    d_values: list[float] | None = None,
    threshold: float = 1e-4,
    target_pvalue: float = 0.05,
) -> dict:
    """Findet minimales d, bei dem Serie stationär wird (ADF-Test).

    Iteriert über d-Werte und gibt das kleinste d zurück mit p-Wert < target.
    Kleineres d = mehr Memory behalten.

    Returns:
        dict mit {"optimal_d", "results": [{d, pvalue, is_stationary}, …]}
    """
    if d_values is None:
        d_values = [round(0.1 * i, 1) for i in range(1, 11)]  # 0.1..1.0

    results = []
    optimal_d = None

    for d in d_values:
        diffed = frac_diff_ffd(series, d=d, threshold=threshold)
        adf = adf_stationarity_test(diffed)
        record = {"d": d, **adf}
        results.append(record)

        if (
            optimal_d is None
            and adf.get("is_stationary") is True
            and adf.get("pvalue", 1.0) < target_pvalue
        ):
            optimal_d = d

    return {
        "optimal_d": optimal_d,
        "results": results,
    }


def apply_ffd_to_panel(
    panel_df: pd.DataFrame,
    price_cols: list[str],
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    d: float = 0.4,
    threshold: float = 1e-4,
) -> pd.DataFrame:
    """Wendet FFD auf mehrere Preisspalten in einem Panel an.

    Für jedes Symbol separat (groupby) → pro Symbol eigene Memory-Serie.

    Neue Spalten: `{col}_ffd_{d:.2f}`.
    """
    result = panel_df.sort_values([symbol_col, timestamp_col]).copy()
    for col in price_cols:
        if col not in result.columns:
            logger.warning("[FFD] Spalte %s nicht im Panel — übersprungen", col)
            continue
        new_col = f"{col}_ffd_{d:.2f}"
        result[new_col] = result.groupby(symbol_col, group_keys=False)[col].transform(
            lambda s: frac_diff_ffd(s, d=d, threshold=threshold)
        )
        logger.info(
            "[FFD] %s → %s (d=%.2f, %d non-null)",
            col, new_col, d, int(result[new_col].notna().sum()),
        )
    return result


__all__ = [
    "frac_diff_weights",
    "frac_diff_ffd",
    "adf_stationarity_test",
    "find_optimal_d",
    "apply_ffd_to_panel",
]
