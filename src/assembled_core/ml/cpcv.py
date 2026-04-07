"""Combinatorial Purged Cross-Validation (V12).

Implements López de Prado's CPCV: generates C(N, K) backtests from N groups,
producing a distribution of Sharpe ratios for rigorous overfitting detection.

Reference: López de Prado — "Advances in Financial Machine Learning" (Ch. 12).
"""

from __future__ import annotations

import logging
from itertools import combinations
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class CPCVResult:
    """Results from Combinatorial Purged Cross-Validation."""

    n_paths: int  # Total number of backtest paths
    sharpe_distribution: list[float]  # Sharpe ratio per path
    mean_sharpe: float
    std_sharpe: float
    median_sharpe: float
    prob_positive_sharpe: float  # P(Sharpe > 0)
    prob_sharpe_above_1: float  # P(Sharpe > 1.0)
    deflated_sharpe: float | None  # Adjusted for multiple testing
    is_likely_overfit: bool  # True if high risk of overfitting


def generate_cpcv_splits(
    n_timestamps: int,
    n_groups: int = 6,
    k_test_groups: int = 2,
    purge_length: int = 5,
    embargo_length: int = 3,
) -> list[tuple[list[int], list[int]]]:
    """Generate all combinatorial purged train/test splits.

    Args:
        n_timestamps: Total number of time periods.
        n_groups: Number of non-overlapping groups (N).
        k_test_groups: Number of groups used as test per combination (K).
        purge_length: Number of observations to purge between train and test.
        embargo_length: Number of observations to embargo after test.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    group_size = n_timestamps // n_groups
    if group_size < 10:
        _log.warning("Group size too small (%d) for meaningful CPCV", group_size)
        return []

    # Define group boundaries
    groups: list[tuple[int, int]] = []
    for i in range(n_groups):
        start = i * group_size
        end = min((i + 1) * group_size, n_timestamps)
        groups.append((start, end))

    splits = []
    for test_group_indices in combinations(range(n_groups), k_test_groups):
        # Test indices
        test_idx: list[int] = []
        for gi in test_group_indices:
            start, end = groups[gi]
            test_idx.extend(range(start, end))

        # Train indices: all groups NOT in test, minus purge/embargo
        train_idx: list[int] = []
        test_set = set(test_idx)
        purge_embargo_set: set[int] = set()

        for gi in test_group_indices:
            start, end = groups[gi]
            # Purge before test
            for p in range(max(0, start - purge_length), start):
                purge_embargo_set.add(p)
            # Embargo after test
            for e in range(end, min(n_timestamps, end + embargo_length)):
                purge_embargo_set.add(e)

        for i in range(n_timestamps):
            if i not in test_set and i not in purge_embargo_set:
                train_idx.append(i)

        if len(train_idx) >= 20 and len(test_idx) >= 10:
            splits.append((train_idx, test_idx))

    _log.info(
        "CPCV: %d paths from C(%d,%d) = %d combinations",
        len(splits), n_groups, k_test_groups,
        len(list(combinations(range(n_groups), k_test_groups))),
    )
    return splits


def compute_cpcv_sharpe_distribution(
    returns_per_path: list[np.ndarray],
    periods_per_year: int = 252,
) -> CPCVResult:
    """Compute Sharpe distribution across CPCV paths and assess overfitting.

    Args:
        returns_per_path: List of return arrays (one per CPCV path).
        periods_per_year: Annualization factor.

    Returns:
        CPCVResult with distribution statistics and overfitting assessment.
    """
    sharpes = []
    for rets in returns_per_path:
        if len(rets) < 5:
            continue
        mean_r = float(np.mean(rets))
        std_r = float(np.std(rets))
        if std_r > 1e-12:
            sharpe = mean_r / std_r * np.sqrt(periods_per_year)
            sharpes.append(float(sharpe))

    if len(sharpes) < 3:
        return CPCVResult(
            n_paths=len(sharpes), sharpe_distribution=sharpes,
            mean_sharpe=0.0, std_sharpe=0.0, median_sharpe=0.0,
            prob_positive_sharpe=0.0, prob_sharpe_above_1=0.0,
            deflated_sharpe=None, is_likely_overfit=True,
        )

    sharpe_arr = np.array(sharpes)
    mean_s = float(np.mean(sharpe_arr))
    std_s = float(np.std(sharpe_arr))
    median_s = float(np.median(sharpe_arr))
    prob_pos = float(np.mean(sharpe_arr > 0))
    prob_above_1 = float(np.mean(sharpe_arr > 1.0))

    # Deflated Sharpe: adjust for multiple testing
    # DSR = (observed_sharpe - E[max(Sharpe)]) / SE(Sharpe)
    n_tests = len(sharpes)
    n_obs = max(len(r) for r in returns_per_path) if returns_per_path else 252

    try:
        from scipy.stats import norm
        # Expected max Sharpe under null (i.i.d. tests)
        e_max_sharpe = std_s * norm.ppf(1 - 1 / n_tests) if n_tests > 1 else 0.0
        se_sharpe = std_s / max(np.sqrt(n_tests), 1)
        dsr = (mean_s - e_max_sharpe) / max(se_sharpe, 1e-10)
        deflated_sharpe = float(dsr)
    except ImportError:
        deflated_sharpe = None

    # Overfitting assessment
    is_overfit = (
        prob_pos < 0.5  # Less than half of paths have positive Sharpe
        or (deflated_sharpe is not None and deflated_sharpe < 0)
        or std_s > abs(mean_s) * 2  # Very high variance in Sharpe
    )

    return CPCVResult(
        n_paths=len(sharpes),
        sharpe_distribution=sharpes,
        mean_sharpe=round(mean_s, 4),
        std_sharpe=round(std_s, 4),
        median_sharpe=round(median_s, 4),
        prob_positive_sharpe=round(prob_pos, 4),
        prob_sharpe_above_1=round(prob_above_1, 4),
        deflated_sharpe=round(deflated_sharpe, 4) if deflated_sharpe is not None else None,
        is_likely_overfit=is_overfit,
    )


__all__ = [
    "CPCVResult",
    "generate_cpcv_splits",
    "compute_cpcv_sharpe_distribution",
]
