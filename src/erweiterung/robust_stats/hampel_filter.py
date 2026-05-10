"""Hampel-Filter — robuste Outlier-Detection in Time-Series.

Reference
---------
Hampel, F. (1974). The Influence Curve and Its Role in Robust Estimation.
*JASA* 69.
Pearson, R. (2002). Outliers in Process Modeling.

Idee
----
Klassisches 3σ-Filter ist nicht robust (Outlier selbst beeinflussen σ).
Hampel-Filter ersetzt mean+std durch median+MAD im Rolling-Window:

    |x_t - median(x_{t-k..t+k})| > n_sigma × MAD  ⇒  Outlier

Standard n_sigma = 3 (entspricht 3-σ unter Gaussian, da MAD-Scale ≈ σ/1.4826).

Anwendung
---------
- Spike-Removal in Returns / Prices
- Pre-Processing für ML-Modelle
- Outlier-Detection für Earnings-Surprises
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hampel_filter(
    series: pd.Series,
    window: int = 11,
    n_sigma: float = 3.0,
    return_mask: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """Apply Hampel-Filter to a series.

    Args:
        series: 1-D input.
        window: must be odd; rolling-window size (centered).
        n_sigma: threshold for outlier-detection (in MAD-σ units).
        return_mask: if True, return (filtered, outlier_mask).

    Returns:
        Filtered series (outliers replaced by rolling-median) or tuple.
    """
    if window % 2 == 0:
        window += 1
    s = pd.Series(series).copy()
    half = window // 2
    rolling = s.rolling(window, center=True, min_periods=half + 1)
    med = rolling.median()
    mad = rolling.apply(
        lambda x: (
            1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 0 else np.nan
        ),
        raw=True,
    )
    diff = (s - med).abs()
    threshold = n_sigma * mad
    outliers = (diff > threshold) & (mad > 0)
    filtered = s.copy()
    filtered[outliers] = med[outliers]
    if return_mask:
        return filtered, outliers.fillna(False)
    return filtered


def winsorize_series(
    series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99
) -> pd.Series:
    """Cross-section-level Winsorization."""
    s = pd.Series(series).copy()
    if s.empty:
        return s
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lo, hi)


def rolling_zscore_outliers(
    series: pd.Series, window: int = 60, threshold: float = 3.0
) -> pd.Series:
    """Identify outliers via rolling Z-Score (boolean mask)."""
    s = pd.Series(series)
    mean = s.rolling(window, min_periods=window // 2).mean()
    std = s.rolling(window, min_periods=window // 2).std()
    z = (s - mean) / std.replace(0, np.nan)
    return (z.abs() > threshold).fillna(False)


__all__ = ["hampel_filter", "winsorize_series", "rolling_zscore_outliers"]
