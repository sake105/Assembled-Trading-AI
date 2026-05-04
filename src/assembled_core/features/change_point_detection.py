"""Change-Point Detection via ruptures library.

From 11_FREE_MODELLE.md §11.10.
Offline regime-break detection as complement to HMM.
PELT (Pruned Exact Linear Time) is the fastest exact algorithm.

Install: pip install ruptures==1.1.10
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChangePointResult(NamedTuple):
    breakpoints: list[int]
    n_segments: int
    algorithm: str
    signal_length: int


def _try_ruptures():
    try:
        import ruptures as rpt

        return rpt
    except ImportError:
        logger.warning("ruptures not installed — pip install ruptures==1.1.10")
        return None


def detect_change_points_pelt(
    signal: np.ndarray | pd.Series,
    model: str = "rbf",
    penalty: float = 10.0,
    min_size: int = 5,
) -> ChangePointResult:
    """Detect change points using PELT (Pruned Exact Linear Time).

    Args:
        signal: 1D signal array (e.g. returns, volatility series)
        model: Cost function — 'rbf' (default), 'l1', 'l2', 'normal', 'ar'
        penalty: Penalty for adding a new breakpoint (BIC-like). Higher = fewer breaks.
        min_size: Minimum segment length.

    Returns:
        ChangePointResult with breakpoint indices (end of each segment, excluding last).
        Returns empty result if ruptures unavailable.
    """
    rpt = _try_ruptures()
    if rpt is None:
        return ChangePointResult([], 0, "pelt", 0)

    arr = signal.values if isinstance(signal, pd.Series) else np.asarray(signal)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    try:
        algo = rpt.Pelt(model=model, min_size=min_size).fit(arr)
        result = algo.predict(pen=penalty)
        # ruptures returns end indices; last is len(arr) (remove it)
        breakpoints = result[:-1]
        return ChangePointResult(
            breakpoints=breakpoints,
            n_segments=len(result),
            algorithm="pelt",
            signal_length=len(arr),
        )
    except Exception as exc:
        logger.debug("PELT change-point detection failed: %s", exc)
        return ChangePointResult([], 0, "pelt", len(arr))


def detect_change_points_binseg(
    signal: np.ndarray | pd.Series,
    model: str = "rbf",
    n_breaks: int = 5,
) -> ChangePointResult:
    """Detect change points using Binary Segmentation.

    Args:
        signal: 1D signal array
        model: Cost function — 'rbf', 'l1', 'l2', 'normal'
        n_breaks: Maximum number of breakpoints to find.

    Returns:
        ChangePointResult with breakpoint indices.
    """
    rpt = _try_ruptures()
    if rpt is None:
        return ChangePointResult([], 0, "binseg", 0)

    arr = signal.values if isinstance(signal, pd.Series) else np.asarray(signal)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    try:
        algo = rpt.Binseg(model=model).fit(arr)
        result = algo.predict(n_bkps=n_breaks)
        breakpoints = result[:-1]
        return ChangePointResult(
            breakpoints=breakpoints,
            n_segments=len(result),
            algorithm="binseg",
            signal_length=len(arr),
        )
    except Exception as exc:
        logger.debug("BinSeg change-point detection failed: %s", exc)
        return ChangePointResult([], 0, "binseg", len(arr))


def change_point_regime_feature(
    returns: pd.Series,
    penalty: float = 10.0,
    min_size: int = 10,
) -> pd.Series:
    """Assign regime labels based on PELT change-point segmentation.

    Returns integer segment labels (0, 1, 2, ...) aligned to returns index.
    Useful as a categorical feature for ML models.
    """
    result = detect_change_points_pelt(
        returns, model="rbf", penalty=penalty, min_size=min_size
    )

    labels = np.zeros(len(returns), dtype=int)
    prev = 0
    for seg_idx, bkpt in enumerate(result.breakpoints):
        labels[prev:bkpt] = seg_idx
        prev = bkpt
    labels[prev:] = len(result.breakpoints)

    return pd.Series(labels, index=returns.index, name="cp_regime")


def recent_break_flag(
    returns: pd.Series,
    lookback_bars: int = 60,
    penalty: float = 5.0,
) -> bool:
    """Return True if a regime change was detected in the last N bars.

    Useful as a binary signal for increasing uncertainty / reducing position.
    """
    if len(returns) < lookback_bars:
        return False

    recent = returns.iloc[-lookback_bars:]
    result = detect_change_points_pelt(recent, penalty=penalty, min_size=5)
    return len(result.breakpoints) > 0


__all__ = [
    "ChangePointResult",
    "detect_change_points_pelt",
    "detect_change_points_binseg",
    "change_point_regime_feature",
    "recent_break_flag",
]
