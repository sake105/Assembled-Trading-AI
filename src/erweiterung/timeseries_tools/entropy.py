"""Entropy-Measures: Sample-Entropy, Approximate-Entropy, Shannon-Entropy.

Anwendung
---------
- Komplexitätsmaß für Zeitreihen
- Detection von Regime-Changes (entropy spike)
- Filter für mean-reverting (low entropy = predictable)

References
----------
- Pincus, S. (1991). Approximate entropy as a measure of system complexity.
- Richman/Moorman (2000). Physiological time-series analysis using approximate
  entropy and sample entropy.
"""

from __future__ import annotations

import numpy as np


def sample_entropy(series: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Sample entropy (Richman/Moorman 2000).

    Args:
        series: 1-D array.
        m: pattern length.
        r: tolerance, default 0.2 * std.

    Returns:
        SampEn (higher = more random/complex).
    """
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 50:
        return float("nan")
    if r is None:
        r = 0.2 * float(s.std())
    if r <= 0:
        return 0.0

    def _phi(m_):
        templates = np.array([s[i : i + m_] for i in range(n - m_ + 1)])
        # Chebyshev distance
        count = 0
        total = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += int(np.sum(d <= r) - 1)  # exclude self
            total += len(templates) - 1
        return count / total if total > 0 else 0.0

    A = _phi(m + 1)
    B = _phi(m)
    if A == 0 or B == 0:
        return float("inf")
    return float(-np.log(A / B))


def approximate_entropy(
    series: np.ndarray, m: int = 2, r: float | None = None
) -> float:
    """Approximate entropy (Pincus 1991)."""
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    if n < 50:
        return float("nan")
    if r is None:
        r = 0.2 * float(s.std())
    if r <= 0:
        return 0.0

    def _phi(m_):
        templates = np.array([s[i : i + m_] for i in range(n - m_ + 1)])
        c_i = []
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            c_i.append(np.sum(d <= r) / len(templates))
        return float(np.mean(np.log(np.clip(c_i, 1e-12, 1.0))))

    return _phi(m) - _phi(m + 1)


def shannon_entropy(series: np.ndarray, n_bins: int = 20) -> float:
    """Discrete Shannon entropy via histogram."""
    s = np.asarray(series, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < 5:
        return float("nan")
    hist, _ = np.histogram(s, bins=n_bins, density=False)
    p = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


__all__ = ["sample_entropy", "approximate_entropy", "shannon_entropy"]
