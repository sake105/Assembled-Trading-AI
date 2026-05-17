"""Tests for src/assembled_core/signals/pairs_diagnostics.py — C4-084.

Covers:
- ou_half_life on synthetic AR(1) processes with known half-life
- ou_half_life on random walk (no mean reversion → inf)
- ou_half_life edge cases (empty, short, constant)
- engle_granger_cointegration on known-cointegrated series
- engle_granger_cointegration rejecting independent random walks
- input validation
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from src.assembled_core.signals.pairs_diagnostics import (
    CointegrationResult,
    engle_granger_cointegration,
    ou_half_life,
)


# ---------------------------------------------------------------------------
# OU half-life
# ---------------------------------------------------------------------------


def _synthetic_ou(
    n: int, lambda_true: float, mu: float = 0.0, sigma: float = 0.01, seed: int = 0
) -> pd.Series:
    """Generate a synthetic OU process: Δs = -λ(s-μ) + σ·ε."""
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = s[t - 1] + lambda_true * (mu - s[t - 1]) + sigma * rng.standard_normal()
    return pd.Series(s)


def test_ou_half_life_recovers_known_lambda():
    """For λ_true=0.1, half-life ≈ ln(2)/0.1 ≈ 6.93 periods."""
    s = _synthetic_ou(n=2000, lambda_true=0.1, seed=42)
    hl = ou_half_life(s)
    expected = math.log(2.0) / 0.1
    # Allow 20% tolerance — OLS on finite sample
    msg = f"expected ≈{expected:.2f}, got {hl:.2f}"
    assert 0.8 * expected < hl < 1.2 * expected, msg


def test_ou_half_life_fast_mean_reversion():
    """λ_true=0.5 → half-life ≈ 1.39 periods (very fast)."""
    s = _synthetic_ou(n=2000, lambda_true=0.5, seed=1)
    hl = ou_half_life(s)
    expected = math.log(2.0) / 0.5
    assert 0.7 * expected < hl < 1.3 * expected


def test_ou_half_life_random_walk_is_inf_or_very_large():
    """Random walk has no real mean reversion. Finite-sample OLS may produce a
    slightly-negative slope by chance → spuriously finite half-life. Acceptance:
    half-life is either inf OR much larger than what's tradeable (≥ 50 periods).

    This is honest about the finite-sample limitation — practitioners filter
    pairs by half-life threshold (e.g., 1-30 periods) anyway, so a noisy 250
    on a true random walk is correctly excluded.
    """
    rng = np.random.default_rng(7)
    s = pd.Series(np.cumsum(rng.standard_normal(1000)) * 0.01)
    hl = ou_half_life(s)
    msg = f"Random walk should have inf or very large half-life, got {hl:.2f}"
    assert hl == float("inf") or hl >= 50, msg


def test_ou_half_life_short_input_returns_nan():
    s = pd.Series([1.0, 1.1, 1.0, 0.9, 1.0])
    assert math.isnan(ou_half_life(s))


def test_ou_half_life_empty_returns_nan():
    assert math.isnan(ou_half_life(pd.Series([], dtype=float)))


def test_ou_half_life_all_constant_returns_nan():
    """All-constant input has std=0 → undefined half-life."""
    s = pd.Series([5.0] * 100)
    assert math.isnan(ou_half_life(s))


def test_ou_half_life_accepts_numpy_array():
    """Should accept np.ndarray (not just pd.Series)."""
    arr = _synthetic_ou(n=500, lambda_true=0.2, seed=99).to_numpy()
    hl = ou_half_life(arr)
    expected = math.log(2.0) / 0.2
    assert 0.5 * expected < hl < 2.0 * expected


# ---------------------------------------------------------------------------
# Engle-Granger cointegration
# ---------------------------------------------------------------------------


def test_engle_granger_cointegrated_pair_rejects_unit_root():
    """y = x + stationary AR(1) → strongly cointegrated, p < 0.05."""
    rng = np.random.default_rng(123)
    n = 500
    x = np.cumsum(rng.standard_normal(n)) * 0.01  # I(1)
    eps = _synthetic_ou(n=n, lambda_true=0.1, seed=234).to_numpy()
    y = x + eps  # cointegrated

    result = engle_granger_cointegration(y, x)
    assert isinstance(result, CointegrationResult)
    msg1 = f"Expected cointegration at 5%, got p={result.pvalue:.4f}"
    assert result.is_cointegrated_at_5pct, msg1
    msg2 = f"ADF stat {result.statistic} should be below 5% crit {result.crit_values['5%']}"
    assert result.statistic < result.crit_values["5%"], msg2


def test_engle_granger_independent_random_walks_not_cointegrated():
    """Two independent random walks → typically NOT cointegrated."""
    rng = np.random.default_rng(456)
    n = 500
    x = np.cumsum(rng.standard_normal(n)) * 0.01
    y = np.cumsum(rng.standard_normal(n)) * 0.01

    result = engle_granger_cointegration(y, x)
    # Independent random walks → typically p > 0.05 (cannot reject H0 = no cointegration)
    # Allow occasional false positives by sampling — but with n=500 and seed=456 we expect p > 0.05
    msg = f"Two independent random walks should NOT be cointegrated, but p={result.pvalue:.4f}"
    assert not result.is_cointegrated_at_5pct, msg


def test_engle_granger_critical_values_present():
    """Result must expose 1%/5%/10% critical values."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(200)) * 0.01
    y = x + rng.standard_normal(200) * 0.01

    result = engle_granger_cointegration(y, x)
    assert "1%" in result.crit_values
    assert "5%" in result.crit_values
    assert "10%" in result.crit_values
    # ADF critical values are negative (left-tail)
    assert (
        result.crit_values["1%"] < result.crit_values["5%"] < result.crit_values["10%"]
    )


def test_engle_granger_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        engle_granger_cointegration(np.array([]), np.array([]))


def test_engle_granger_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        engle_granger_cointegration(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_engle_granger_rejects_nan_input():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="NaN/inf"):
        engle_granger_cointegration(y, x)
