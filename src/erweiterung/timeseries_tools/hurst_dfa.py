"""Hurst-Exponent + Detrended-Fluctuation-Analysis.

Hurst-Exponent
--------------
H ∈ [0, 1] charakterisiert Memory-Strukturen einer Time-Series:
- H < 0.5: Mean-Reverting (anti-persistent)
- H = 0.5: Random-Walk (Brownian)
- H > 0.5: Trending (persistent)

Methoden
--------
1. **R/S-Analysis** (Hurst 1951): rescaled range vs window size.
2. **DFA — Detrended Fluctuation Analysis** (Peng et al. 1994):
   robuster gegen Non-Stationarität.

Anwendung
---------
- Filter für Trend-/Mean-Rev-Strategien (H<0.4 = pure mean-rev candidate).
- Crisis-Indicator (H drift over time).
- Ergänzung zu Variance-Ratio-Test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hurst_rs(series: np.ndarray, max_window: int = 200) -> float:
    """Hurst exponent via Rescaled-Range-Analysis.

    Args:
        series: 1-D array (returns or log-prices).
        max_window: maximum subseries length.

    Returns:
        Hurst exponent estimate.
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 50:
        return float("nan")
    max_window = min(max_window, len(s) // 2)
    lags = np.unique(np.logspace(0.5, np.log10(max_window), 20).astype(int))
    rs_list = []
    for lag in lags:
        if lag < 2:
            continue
        n_chunks = len(s) // lag
        if n_chunks == 0:
            continue
        rs_chunk = []
        for i in range(n_chunks):
            chunk = s[i * lag : (i + 1) * lag]
            mean = chunk.mean()
            cum = (chunk - mean).cumsum()
            R = cum.max() - cum.min()
            S = chunk.std(ddof=0)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_list.append((lag, float(np.mean(rs_chunk))))
    if len(rs_list) < 3:
        return float("nan")
    arr = np.array(rs_list)
    log_n = np.log(arr[:, 0])
    log_rs = np.log(arr[:, 1])
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(slope)


def detrended_fluctuation_analysis(series: np.ndarray, max_window: int = 200) -> float:
    """DFA-Hurst-Schätzung.

    Robuster als R/S bei Non-Stationarität.
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 50:
        return float("nan")
    y = (s - s.mean()).cumsum()
    max_window = min(max_window, len(y) // 4)
    scales = np.unique(np.logspace(1, np.log10(max_window), 20).astype(int))
    flucts = []
    for scale in scales:
        if scale < 4:
            continue
        n_seg = len(y) // scale
        if n_seg < 2:
            continue
        sq_devs = []
        for i in range(n_seg):
            seg = y[i * scale : (i + 1) * scale]
            x = np.arange(len(seg))
            beta = np.polyfit(x, seg, 1)
            trend = np.polyval(beta, x)
            sq_devs.append(((seg - trend) ** 2).mean())
        flucts.append((scale, np.sqrt(np.mean(sq_devs))))
    if len(flucts) < 3:
        return float("nan")
    arr = np.array(flucts)
    slope, _ = np.polyfit(np.log(arr[:, 0]), np.log(arr[:, 1]), 1)
    return float(slope)


def variance_ratio_test(
    series: np.ndarray, lags: tuple[int, ...] = (2, 4, 8, 16)
) -> dict:
    """Lo/MacKinlay (1988) Variance-Ratio-Test.

    Returns:
        Dict with VR(lag) for each lag. VR < 1 = mean-reverting; VR > 1 = trending.
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    out = {}
    for lag in lags:
        if len(s) < lag * 5:
            out[f"vr_{lag}"] = float("nan")
            continue
        var1 = float(s.var(ddof=0))
        rolled = pd.Series(s).rolling(lag).sum().dropna().values
        var_l = float(rolled.var(ddof=0))
        out[f"vr_{lag}"] = var_l / (lag * var1) if var1 > 0 else float("nan")
    return out


__all__ = ["hurst_rs", "detrended_fluctuation_analysis", "variance_ratio_test"]
