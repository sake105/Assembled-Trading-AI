"""Multiple Testing Corrections for Factor Screening (M16.3).

Provides FDR (Benjamini-Hochberg) and FWER (Holm-Bonferroni) corrections
to prevent false-positive factor discoveries when screening many candidates.

Usage:
    from src.assembled_core.qa.multiple_testing import (
        benjamini_hochberg_fdr,
        holm_bonferroni_fwer,
        screen_factors_with_fdr,
    )
    # Reject factors whose IC p-values don't survive correction
    survived = benjamini_hochberg_fdr(p_values, alpha=0.05)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MultipleTestingResult:
    """Result of a multiple testing correction."""

    method: str
    alpha: float
    n_tests: int
    n_rejected: int
    rejected: list[bool]
    adjusted_threshold: float | None


def benjamini_hochberg_fdr(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Benjamini-Hochberg FDR control.

    Controls the expected proportion of false discoveries among rejections.
    Less conservative than FWER — preferred for factor screening.

    Args:
        p_values: Raw p-values from IC significance tests.
        alpha: Target FDR level (default: 0.05).

    Returns:
        MultipleTestingResult with rejected flags per test.
    """
    pv = np.asarray(p_values, dtype=float)
    n = len(pv)
    if n == 0:
        return MultipleTestingResult("BH-FDR", alpha, 0, 0, [], None)

    sorted_idx = np.argsort(pv)
    threshold_rank = 0

    for rank_minus_1, idx in enumerate(sorted_idx):
        rank = rank_minus_1 + 1
        if pv[idx] <= alpha * rank / n:
            threshold_rank = rank

    if threshold_rank == 0:
        rejected = [False] * n
        adj_thresh = None
    else:
        adj_thresh = alpha * threshold_rank / n
        rejected = [pv[i] <= adj_thresh for i in range(n)]

    result = MultipleTestingResult(
        method="BH-FDR",
        alpha=alpha,
        n_tests=n,
        n_rejected=sum(rejected),
        rejected=rejected,
        adjusted_threshold=adj_thresh,
    )
    logger.info(
        "[FDR] BH alpha=%.3f: %d/%d factors rejected (threshold=%.4f)",
        alpha, result.n_rejected, n,
        adj_thresh if adj_thresh is not None else 0.0,
    )
    return result


def holm_bonferroni_fwer(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Holm-Bonferroni FWER control.

    Controls the probability of ANY false positive. More conservative than
    FDR — use when false positives are very costly.

    Args:
        p_values: Raw p-values.
        alpha: Target FWER level (default: 0.05).

    Returns:
        MultipleTestingResult with rejected flags per test.
    """
    pv = np.asarray(p_values, dtype=float)
    n = len(pv)
    if n == 0:
        return MultipleTestingResult("Holm-Bonferroni", alpha, 0, 0, [], None)

    sorted_idx = np.argsort(pv)
    rejected_set: set[int] = set()
    adj_thresh = None

    for rank_minus_1, idx in enumerate(sorted_idx):
        corrected_alpha = alpha / (n - rank_minus_1)
        if pv[idx] > corrected_alpha:
            break
        rejected_set.add(idx)
        adj_thresh = corrected_alpha

    rejected = [i in rejected_set for i in range(n)]
    result = MultipleTestingResult(
        method="Holm-Bonferroni",
        alpha=alpha,
        n_tests=n,
        n_rejected=len(rejected_set),
        rejected=rejected,
        adjusted_threshold=adj_thresh,
    )
    logger.info(
        "[FWER] Holm-Bonferroni alpha=%.3f: %d/%d factors rejected",
        alpha, result.n_rejected, n,
    )
    return result


def screen_factors_with_fdr(
    factor_ic_df: pd.DataFrame,
    *,
    ic_col: str = "mean_ic",
    pvalue_col: str = "p_value",
    factor_col: str = "factor",
    alpha: float = 0.05,
    method: str = "bh",
) -> pd.DataFrame:
    """Screen a factor table and keep only FDR-surviving factors.

    Args:
        factor_ic_df: DataFrame with factor names, mean IC, and p-values.
        ic_col: Column with mean IC values.
        pvalue_col: Column with p-values from IC significance test.
        factor_col: Column with factor names.
        alpha: FDR level.
        method: "bh" for Benjamini-Hochberg, "holm" for Holm-Bonferroni.

    Returns:
        Filtered DataFrame with only factors surviving the correction.
    """
    if factor_ic_df.empty or pvalue_col not in factor_ic_df.columns:
        return factor_ic_df

    pvals = factor_ic_df[pvalue_col].values

    if method == "holm":
        result = holm_bonferroni_fwer(pvals, alpha=alpha)
    else:
        result = benjamini_hochberg_fdr(pvals, alpha=alpha)

    df = factor_ic_df.copy()
    df["fdr_rejected"] = result.rejected
    survived = df[df["fdr_rejected"]].drop(columns=["fdr_rejected"])

    logger.info(
        "[Screen] %d/%d factors survive %s (alpha=%.3f)",
        len(survived), len(df), result.method, alpha,
    )
    return survived
