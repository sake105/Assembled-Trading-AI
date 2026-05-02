"""Chart Pattern Recognition via Matrix Profile (stumpy) and DTW (dtaidistance).

From 11_FREE_MODELLE.md §11.13 and 31_COMPOSITE_SCORE.md Dim-5.
Matrix Profile beats Head-and-Shoulders detection in most benchmarks.

stumpy.stumpi(): incremental O(n) update per new bar — ideal for live FastAPI.

Install: pip install stumpy==1.14.0 dtaidistance==2.3.13
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _try_stumpy():
    try:
        import stumpy
        return stumpy
    except ImportError:
        logger.warning("stumpy not installed — pip install stumpy==1.14.0")
        return None


def _try_dtai():
    try:
        from dtaidistance import dtw
        return dtw
    except ImportError:
        logger.warning("dtaidistance not installed — pip install dtaidistance==2.3.13")
        return None


class MatrixProfileResult(NamedTuple):
    mp: np.ndarray       # matrix profile distances
    mpi: np.ndarray      # matrix profile indices (nearest neighbor)
    motif_idx: int       # index of best motif (most repeated pattern)
    discord_idx: int     # index of anomaly (most unusual pattern)
    anomaly_score: float # normalized discord distance


def compute_matrix_profile(
    series: pd.Series | np.ndarray,
    window: int = 20,
) -> MatrixProfileResult | None:
    """Compute Matrix Profile for a price/return series.

    Args:
        series: 1D price or return series (or log-prices for better stationarity)
        window: Subsequence length (default 20 bars = ~1 month EOD)

    Returns:
        MatrixProfileResult or None if stumpy unavailable.
    """
    stumpy = _try_stumpy()
    if stumpy is None:
        return None

    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < window * 2:
        logger.debug("Insufficient data for matrix profile: %d bars (need %d)", len(arr), window * 2)
        return None

    mp_result = stumpy.stump(arr, m=window)
    mp = mp_result[:, 0].astype(float)  # profile distances
    mpi = mp_result[:, 1].astype(int)   # nearest neighbor indices

    motif_idx = int(np.argmin(mp))
    discord_idx = int(np.argmax(mp))
    max_distance = float(np.nanmax(mp))
    anomaly_score = float(mp[discord_idx] / (max_distance + 1e-9))

    return MatrixProfileResult(
        mp=mp, mpi=mpi,
        motif_idx=motif_idx, discord_idx=discord_idx,
        anomaly_score=anomaly_score,
    )


def discord_anomaly_feature(
    prices: pd.Series,
    window: int = 20,
) -> float:
    """Return normalized anomaly score for the most recent window.

    Useful as Chart-Pattern-Dimension feature in the Composite Score.
    Higher = current pattern is more unusual relative to historical patterns.
    """
    result = compute_matrix_profile(prices, window=window)
    if result is None or len(result.mp) == 0:
        return 0.0

    # Check if most recent subsequence is near a discord
    recent_idx = len(result.mp) - 1
    recent_mp = result.mp[recent_idx]
    max_mp = float(np.nanmax(result.mp))
    return float(recent_mp / (max_mp + 1e-9))


def motif_similarity_feature(
    prices: pd.Series,
    pattern_prices: pd.Series,
    window: int = 20,
) -> float:
    """DTW distance between current window and a target pattern.

    Args:
        prices: Full price history
        pattern_prices: Reference pattern (e.g., a head-and-shoulders template)
        window: Window size for comparison

    Returns:
        Normalized DTW distance (0 = identical, 1 = maximally dissimilar).
    """
    dtw = _try_dtai()
    if dtw is None:
        return 0.0

    if len(prices) < window:
        return 0.0

    current = np.asarray(prices.iloc[-window:], dtype=float)
    pattern = np.asarray(pattern_prices.iloc[:window], dtype=float)

    if len(pattern) < window:
        pattern = np.pad(pattern, (0, window - len(pattern)), mode="edge")

    # Normalize both
    def _norm(arr: np.ndarray) -> np.ndarray:
        std = arr.std()
        return (arr - arr.mean()) / (std if std > 0 else 1.0)

    try:
        dist = dtw.distance(_norm(current), _norm(pattern))
        max_dist = np.sqrt(window * 4.0)  # rough upper bound
        return float(np.clip(dist / max_dist, 0, 1))
    except Exception as exc:
        logger.debug("DTW distance failed: %s", exc)
        return 0.0


def chart_pattern_score(
    prices: pd.Series,
    window: int = 20,
) -> dict[str, float]:
    """Composite chart-pattern features for Composite Score Dim-5.

    Returns dict with:
      anomaly_score: how unusual current pattern is (0-1)
      trend_strength: matrix profile trend signal (proxy for momentum)
    """
    result = compute_matrix_profile(np.log(prices.clip(lower=1e-10)), window=window)

    features: dict[str, float] = {
        "anomaly_score": 0.0,
        "trend_strength": 0.0,
    }

    if result is None:
        return features

    features["anomaly_score"] = result.anomaly_score

    # Trend strength: recent mean-reversion vs momentum
    # High discord near recent = potential reversal
    mp = result.mp
    if len(mp) >= 5:
        recent_trend = float(np.mean(mp[-5:]) / (np.mean(mp) + 1e-9))
        features["trend_strength"] = float(np.clip(recent_trend - 1.0, -1, 1))

    return features


__all__ = [
    "MatrixProfileResult",
    "compute_matrix_profile",
    "discord_anomaly_feature",
    "motif_similarity_feature",
    "chart_pattern_score",
]
