"""A/B Testing Framework for Strategy Variants (M40 Task 40.2).

Provides statistical comparison of strategy variants:
1. Paired t-test on daily returns
2. Bootstrap confidence intervals for Sharpe difference
3. Multiple testing correction (Bonferroni, BH)
4. Minimum detectable effect size estimation
5. Sequential testing with optional stopping

Reference:
    Harvey et al. (2016) "...and the Cross-Section of Expected Returns"
    White (2000) "A Reality Check for Data Snooping"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of A/B test between two strategies."""
    strategy_a: str
    strategy_b: str
    mean_diff: float              # Mean daily return difference (A - B)
    t_stat: float                 # Paired t-statistic
    p_value: float                # Two-sided p-value
    sharpe_a: float               # Annualized Sharpe of A
    sharpe_b: float               # Annualized Sharpe of B
    sharpe_diff: float            # Sharpe difference
    sharpe_diff_ci: tuple[float, float]  # 95% CI for Sharpe difference
    n_days: int
    significant: bool             # p < 0.05 after correction
    winner: str                   # "A", "B", or "inconclusive"


@dataclass
class MultipleTestResult:
    """Result of multiple A/B tests with correction."""
    results: list[ABTestResult]
    correction_method: str
    adjusted_p_values: list[float]
    n_significant: int
    best_strategy: str | None


def paired_ab_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    name_a: str = "A",
    name_b: str = "B",
) -> ABTestResult:
    """Run paired A/B test on daily returns.

    Args:
        returns_a: Daily returns of strategy A.
        returns_b: Daily returns of strategy B.
        name_a: Name of strategy A.
        name_b: Name of strategy B.

    Returns:
        ABTestResult.
    """
    # Align
    common = returns_a.dropna().index.intersection(returns_b.dropna().index)
    if len(common) < 30:
        return ABTestResult(
            strategy_a=name_a, strategy_b=name_b,
            mean_diff=0.0, t_stat=0.0, p_value=1.0,
            sharpe_a=0.0, sharpe_b=0.0, sharpe_diff=0.0,
            sharpe_diff_ci=(0.0, 0.0), n_days=len(common),
            significant=False, winner="inconclusive",
        )

    a = returns_a.loc[common].values
    b = returns_b.loc[common].values
    diff = a - b
    n = len(diff)

    # Paired t-test
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se = std_diff / np.sqrt(n)
    t_stat = mean_diff / max(se, 1e-10)

    # p-value (two-sided, using normal approximation for large n)
    try:
        from scipy.stats import t as t_dist
        p_value = float(2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)))
    except ImportError:
        # Normal approximation
        p_value = float(2 * _normal_sf(abs(t_stat)))

    # Sharpe ratios
    sharpe_a = float(np.mean(a) / max(np.std(a, ddof=1), 1e-10)) * np.sqrt(252)
    sharpe_b = float(np.mean(b) / max(np.std(b, ddof=1), 1e-10)) * np.sqrt(252)
    sharpe_diff = sharpe_a - sharpe_b

    # Bootstrap CI for Sharpe difference
    ci = _bootstrap_sharpe_diff_ci(a, b, n_bootstrap=1000)

    # Winner
    if p_value < 0.05:
        winner = name_a if mean_diff > 0 else name_b
    else:
        winner = "inconclusive"

    return ABTestResult(
        strategy_a=name_a,
        strategy_b=name_b,
        mean_diff=round(mean_diff, 8),
        t_stat=round(t_stat, 4),
        p_value=round(p_value, 6),
        sharpe_a=round(sharpe_a, 4),
        sharpe_b=round(sharpe_b, 4),
        sharpe_diff=round(sharpe_diff, 4),
        sharpe_diff_ci=(round(ci[0], 4), round(ci[1], 4)),
        n_days=n,
        significant=p_value < 0.05,
        winner=winner,
    )


def _normal_sf(x: float) -> float:
    """Survival function for standard normal (1 - CDF)."""
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804 * np.exp(-x * x / 2.0)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return p if x > 0 else 1.0 - p


def _bootstrap_sharpe_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap confidence interval for Sharpe ratio difference."""
    rng = np.random.RandomState(42)
    n = len(a)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        sa = np.mean(a[idx]) / max(np.std(a[idx], ddof=1), 1e-10) * np.sqrt(252)
        sb = np.mean(b[idx]) / max(np.std(b[idx], ddof=1), 1e-10) * np.sqrt(252)
        diffs.append(sa - sb)

    alpha = (1 - ci_level) / 2
    lower = float(np.percentile(diffs, alpha * 100))
    upper = float(np.percentile(diffs, (1 - alpha) * 100))
    return lower, upper


def run_multiple_ab_tests(
    returns_dict: dict[str, pd.Series],
    baseline: str,
    correction: str = "bonferroni",
) -> MultipleTestResult:
    """Run A/B tests of multiple strategies against a baseline.

    Args:
        returns_dict: {strategy_name: daily_returns}.
        baseline: Name of baseline strategy.
        correction: Multiple testing correction ("bonferroni", "bh", "none").

    Returns:
        MultipleTestResult with corrected p-values.
    """
    if baseline not in returns_dict:
        raise ValueError(f"Baseline '{baseline}' not in returns_dict")

    baseline_returns = returns_dict[baseline]
    variants = [k for k in returns_dict if k != baseline]
    n_tests = len(variants)

    results = []
    raw_p_values = []
    for variant in variants:
        result = paired_ab_test(baseline_returns, returns_dict[variant], baseline, variant)
        results.append(result)
        raw_p_values.append(result.p_value)

    # Multiple testing correction
    adjusted_p = _correct_p_values(raw_p_values, correction)

    # Update significance based on corrected p-values
    for i, result in enumerate(results):
        result.significant = adjusted_p[i] < 0.05
        if not result.significant:
            result.winner = "inconclusive"

    n_sig = sum(1 for p in adjusted_p if p < 0.05)

    # Find best strategy
    best = None
    best_sharpe = -np.inf
    for result in results:
        if result.significant and result.sharpe_b > best_sharpe:
            best = result.strategy_b
            best_sharpe = result.sharpe_b
    if best is None:
        # Check if baseline is best
        baseline_sharpe = results[0].sharpe_a if results else 0
        if baseline_sharpe > best_sharpe:
            best = baseline

    logger.info("[A/B] %d tests, %d significant (%s correction), best: %s",
                n_tests, n_sig, correction, best)

    return MultipleTestResult(
        results=results,
        correction_method=correction,
        adjusted_p_values=[round(p, 6) for p in adjusted_p],
        n_significant=n_sig,
        best_strategy=best,
    )


def _correct_p_values(p_values: list[float], method: str) -> list[float]:
    """Apply multiple testing correction to p-values."""
    n = len(p_values)
    if n == 0:
        return []

    if method == "bonferroni":
        return [min(1.0, p * n) for p in p_values]

    elif method == "bh":
        # Benjamini-Hochberg
        sorted_indices = np.argsort(p_values)
        adjusted = [0.0] * n
        for rank, idx in enumerate(sorted_indices, 1):
            adjusted[idx] = p_values[idx] * n / rank
        # Enforce monotonicity
        prev = 1.0
        for idx in reversed(sorted_indices):
            adjusted[idx] = min(prev, adjusted[idx])
            adjusted[idx] = min(1.0, adjusted[idx])
            prev = adjusted[idx]
        return adjusted

    else:
        return list(p_values)


def minimum_detectable_effect(
    n_days: int,
    baseline_vol: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Estimate minimum detectable daily return difference.

    Args:
        n_days: Number of trading days in test.
        baseline_vol: Daily return standard deviation.
        alpha: Significance level.
        power: Statistical power.

    Returns:
        Minimum detectable daily return difference.
    """
    # z-values for alpha/2 and power
    try:
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - alpha / 2)
        z_power = norm.ppf(power)
    except ImportError:
        z_alpha = 1.96  # ~0.025
        z_power = 0.842  # ~0.80

    mde = (z_alpha + z_power) * baseline_vol * np.sqrt(2 / n_days)
    return round(float(mde), 8)


__all__ = [
    "ABTestResult",
    "MultipleTestResult",
    "paired_ab_test",
    "run_multiple_ab_tests",
    "minimum_detectable_effect",
]
