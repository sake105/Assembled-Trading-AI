"""Change-Point Detection — CUSUM + Binary-Segmentation.

Anwendung
---------
- Regime-Change-Detection
- Volatility-Break-Detection
- Strategy-Switching-Trigger
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_filter(series: pd.Series, threshold: float = 0.05) -> list[pd.Timestamp]:
    """Symmetric CUSUM filter — returns list of detected change points.

    Reference: Lopez de Prado (2018) §2.5.2.

    Args:
        series: returns or log-returns.
        threshold: cumulative-deviation threshold for triggering.

    Returns:
        Index timestamps of detected events.
    """
    s = pd.Series(series).dropna()
    s_pos = 0.0
    s_neg = 0.0
    events = []
    diff = s.diff().fillna(0)
    for t, d in diff.items():
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -threshold:
            s_neg = 0.0
            events.append(t)
        elif s_pos > threshold:
            s_pos = 0.0
            events.append(t)
    return events


def binary_segmentation(
    series: np.ndarray, n_breakpoints: int = 3, min_segment: int = 30
) -> list[int]:
    """Binary-Segmentation für Mean-Change-Points (greedy).

    Args:
        series: 1-D array.
        n_breakpoints: max breakpoints to find.
        min_segment: minimum segment length.

    Returns:
        List of breakpoint indices (sorted).
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 2 * min_segment:
        return []

    def _best_split(arr, off):
        n = len(arr)
        if n < 2 * min_segment:
            return None, -np.inf
        cs = arr.cumsum()
        css = (arr * arr).cumsum()
        best_t = None
        best_gain = -np.inf
        for t in range(min_segment, n - min_segment):
            mu_l = cs[t - 1] / t
            ssr_l = css[t - 1] - t * mu_l * mu_l
            mu_r = (cs[-1] - cs[t - 1]) / (n - t)
            ssr_r = (css[-1] - css[t - 1]) - (n - t) * mu_r * mu_r
            ssr_total = css[-1] - n * (cs[-1] / n) ** 2
            gain = ssr_total - (ssr_l + ssr_r)
            if gain > best_gain:
                best_gain = gain
                best_t = t + off
        return best_t, best_gain

    breakpoints: list[int] = []
    segments = [(0, n)]
    for _ in range(n_breakpoints):
        best_overall = -np.inf
        best_idx = None
        best_seg = None
        for a, b in segments:
            arr = s[a:b]
            idx, gain = _best_split(arr, a)
            if idx is not None and gain > best_overall:
                best_overall = gain
                best_idx = idx
                best_seg = (a, b)
        if best_idx is None:
            break
        breakpoints.append(best_idx)
        a, b = best_seg
        segments.remove((a, b))
        segments.extend([(a, best_idx), (best_idx, b)])
    return sorted(breakpoints)


__all__ = ["cusum_filter", "binary_segmentation"]
