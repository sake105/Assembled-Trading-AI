"""Tests for deflated Sharpe ratio computation (RB5).

These tests verify that deflated Sharpe ratio is computed correctly:
- Monotonicity: more trials -> lower deflated Sharpe (for same observed Sharpe)
- Deterministic outputs
- Edge cases (n_trials=1, invalid inputs)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from src.assembled_core.qa.robustness import compute_deflated_sharpe


def test_compute_deflated_sharpe_monotonicity():
    """Test that more trials -> lower deflated Sharpe (monotonicity)."""
    sharpe = 1.5
    n_obs = 252  # 1 year daily

    # Compute for different n_trials
    ds_1 = compute_deflated_sharpe(sharpe, n_obs, n_trials=1)
    ds_10 = compute_deflated_sharpe(sharpe, n_obs, n_trials=10)
    ds_100 = compute_deflated_sharpe(sharpe, n_obs, n_trials=100)

    # All should be valid
    assert ds_1 is not None
    assert ds_10 is not None
    assert ds_100 is not None

    # Monotonicity: more trials -> lower deflated Sharpe
    assert ds_1 >= ds_10
    assert ds_10 >= ds_100

    # For n_trials=1, should equal original Sharpe (no adjustment)
    assert abs(ds_1 - sharpe) < 1e-6


def test_compute_deflated_sharpe_deterministic():
    """Test that same inputs produce identical outputs."""
    sharpe = 1.0
    n_obs = 500
    n_trials = 20

    # Run twice
    ds1 = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)
    ds2 = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)

    # Should be identical
    assert ds1 == ds2

    # Should be valid
    assert ds1 is not None
    assert ds2 is not None


def test_compute_deflated_sharpe_n_trials_one():
    """Test that n_trials=1 returns original Sharpe (no adjustment)."""
    sharpe = 2.0
    n_obs = 1000

    ds = compute_deflated_sharpe(sharpe, n_obs, n_trials=1)

    assert ds is not None
    assert abs(ds - sharpe) < 1e-6


def test_compute_deflated_sharpe_invalid_n_obs():
    """Test that invalid n_obs returns None."""
    sharpe = 1.0
    n_obs = 1  # Too small
    n_trials = 10

    ds = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)

    assert ds is None


def test_compute_deflated_sharpe_invalid_n_trials():
    """Test that invalid n_trials returns None."""
    sharpe = 1.0
    n_obs = 500
    n_trials = 0  # Invalid

    ds = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)

    assert ds is None


def test_compute_deflated_sharpe_invalid_alpha():
    """Test that invalid alpha returns None."""
    sharpe = 1.0
    n_obs = 500
    n_trials = 10
    alpha = 1.5  # Invalid (not in (0, 1))

    ds = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials, alpha=alpha)

    assert ds is None


def test_compute_deflated_sharpe_with_skew_kurt():
    """Test that skew and kurtosis affect the result."""
    sharpe = 1.0
    n_obs = 500
    n_trials = 20

    # Normal distribution (skew=0, kurt=3)
    ds_normal = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials, skew=0.0, kurt=3.0)

    # Skewed distribution
    ds_skewed = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials, skew=1.0, kurt=3.0)

    # High kurtosis
    ds_high_kurt = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials, skew=0.0, kurt=5.0)

    # All should be valid
    assert ds_normal is not None
    assert ds_skewed is not None
    assert ds_high_kurt is not None

    # Results should differ (non-normality affects deflated Sharpe)
    # Note: exact relationship depends on formula, but they should be different
    assert ds_normal != ds_skewed or ds_normal != ds_high_kurt


def test_compute_deflated_sharpe_negative_sharpe():
    """Test that negative Sharpe ratios are handled."""
    sharpe = -0.5
    n_obs = 500
    n_trials = 10

    ds = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)

    # Should be valid (negative Sharpe is valid)
    assert ds is not None
    # Deflated Sharpe should be <= original (penalty for multiple testing)
    assert ds <= sharpe


def test_compute_deflated_sharpe_large_n_trials():
    """Test that very large n_trials produces lower deflated Sharpe."""
    sharpe = 2.0
    n_obs = 1000

    ds_10 = compute_deflated_sharpe(sharpe, n_obs, n_trials=10)
    ds_1000 = compute_deflated_sharpe(sharpe, n_obs, n_trials=1000)

    assert ds_10 is not None
    assert ds_1000 is not None

    # More trials -> lower deflated Sharpe
    assert ds_10 >= ds_1000


def test_compute_deflated_sharpe_floating_point_tolerance():
    """Test that floating point comparisons use reasonable tolerance."""
    sharpe = 1.5
    n_obs = 252
    n_trials = 50

    ds1 = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)
    ds2 = compute_deflated_sharpe(sharpe, n_obs, n_trials=n_trials)

    assert ds1 is not None
    assert ds2 is not None

    # Should be identical (deterministic)
    assert abs(ds1 - ds2) < 1e-10

    # Should be reasonable value (not NaN, not Inf)
    assert not math.isnan(ds1)
    assert not math.isinf(ds1)
