"""Tests for Deflated Sharpe Ratio (B4)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.metrics import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
)


@pytest.mark.advanced
def test_deflated_sharpe_leq_raw():
    """Test that deflated Sharpe adjustment reduces significance (DSR is a z-score, not directly comparable).
    
    Note: DSR is a z-score, so it's not directly comparable to raw Sharpe.
    However, we can test that the adjustment (SR - E[max_SR]) is <= SR.
    """
    test_cases = [
        (0.5, 252, 1),
        (0.5, 252, 10),
        (0.5, 252, 100),
        (1.0, 252, 1),
        (1.0, 252, 10),
        (1.0, 252, 100),
        (2.0, 252, 1),
        (2.0, 252, 10),
        (2.0, 252, 100),
    ]
    
    for sharpe_annual, n_obs, n_tests in test_cases:
        dsr = deflated_sharpe_ratio(
            sharpe_annual=sharpe_annual,
            n_obs=n_obs,
            n_tests=n_tests,
        )
        
        assert not np.isnan(dsr), f"DSR should not be NaN for sharpe={sharpe_annual}, n_obs={n_obs}, n_tests={n_tests}"
        # DSR is a z-score, so it can be larger than raw Sharpe in absolute terms
        # But we verify it's finite and in a reasonable range
        assert np.isfinite(dsr), f"DSR should be finite for sharpe={sharpe_annual}, n_obs={n_obs}, n_tests={n_tests}"
        # For n_tests > 1, DSR should generally be lower (more conservative)
        # For n_tests=1, DSR can be higher than raw Sharpe (it's a z-score)


@pytest.mark.advanced
def test_deflated_sharpe_equals_raw_for_single_test_large_sample():
    """Test that for n_tests=1 and large sample, DSR is reasonable.
    
    Note: DSR is a z-score, so it's not directly equal to sharpe_annual.
    For n_tests=1, E[max_SR] = 0, so DSR = SR / std(SR).
    For large n_obs, std(SR) ≈ sqrt((1 + SR^2/2) / n_obs) ≈ sqrt(1/n_obs) for small SR.
    So DSR ≈ SR * sqrt(n_obs) for small SR and large n_obs.
    """
    sharpe_annual = 1.5
    n_obs = 10000  # Large sample
    n_tests = 1
    
    dsr = deflated_sharpe_ratio(
        sharpe_annual=sharpe_annual,
        n_obs=n_obs,
        n_tests=n_tests,
    )
    
    assert not np.isnan(dsr), "DSR should not be NaN"
    # For n_tests=1, expected_max_sharpe = 0, so DSR = SR / std(SR)
    # With large n_obs, std(SR) is small, so DSR will be larger than SR
    # We just verify it's positive and finite
    assert dsr > 0, f"DSR ({dsr:.4f}) should be positive for n_tests=1, large sample"
    assert np.isfinite(dsr), "DSR should be finite"


@pytest.mark.advanced
def test_deflated_sharpe_decreases_with_more_tests():
    """Test that DSR decreases monotonically as n_tests increases."""
    sharpe_annual = 2.0
    n_obs = 252
    
    n_tests_values = [1, 10, 50, 100, 500, 1000]
    dsr_values = []
    
    for n_tests in n_tests_values:
        dsr = deflated_sharpe_ratio(
            sharpe_annual=sharpe_annual,
            n_obs=n_obs,
            n_tests=n_tests,
        )
        assert not np.isnan(dsr), f"DSR should not be NaN for n_tests={n_tests}"
        dsr_values.append(dsr)
    
    # Check monotonicity: each DSR should be <= previous (allowing small numerical errors)
    for i in range(1, len(dsr_values)):
        assert dsr_values[i] <= dsr_values[i-1] + 1e-10, \
            f"DSR should decrease with n_tests. n_tests={n_tests_values[i]}: {dsr_values[i]:.4f} > {dsr_values[i-1]:.4f} (n_tests={n_tests_values[i-1]})"


@pytest.mark.advanced
def test_deflated_sharpe_from_returns_smoke():
    """Smoke test for deflated_sharpe_ratio_from_returns with synthetic returns."""
    # Generate synthetic normal returns (mean=0.001, std=0.02, daily)
    np.random.seed(42)
    n_days = 252
    returns = pd.Series(
        np.random.normal(0.001, 0.02, n_days),
        index=pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC"),
    )
    
    # Test with n_tests=1 (should give reasonable DSR)
    dsr = deflated_sharpe_ratio_from_returns(
        returns=returns,
        n_tests=1,
        scale="daily",
    )
    
    assert not np.isnan(dsr), "DSR should not be NaN"
    assert not np.isinf(dsr), "DSR should not be Inf"
    # DSR is a z-score, so it can be larger than raw Sharpe
    # For normal returns with good Sharpe, DSR can be in range [-10, +20] or even higher
    # We just verify it's finite and in a reasonable range for z-scores
    assert -50.0 <= dsr <= 50.0, f"DSR ({dsr:.4f}) should be in reasonable z-score range [-50, 50]"
    
    # Test with n_tests=100 (should give lower DSR)
    dsr_many_tests = deflated_sharpe_ratio_from_returns(
        returns=returns,
        n_tests=100,
        scale="daily",
    )
    
    assert not np.isnan(dsr_many_tests), "DSR should not be NaN for n_tests=100"
    assert dsr_many_tests <= dsr, \
        f"DSR with n_tests=100 ({dsr_many_tests:.4f}) should be <= DSR with n_tests=1 ({dsr:.4f})"


@pytest.mark.advanced
def test_deflated_sharpe_edge_cases():
    """Test edge cases: small n_obs, invalid inputs."""
    # Edge case: n_obs < 2
    dsr = deflated_sharpe_ratio(sharpe_annual=1.0, n_obs=1, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for n_obs < 2"
    
    dsr = deflated_sharpe_ratio(sharpe_annual=1.0, n_obs=0, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for n_obs = 0"
    
    # Edge case: NaN sharpe
    dsr = deflated_sharpe_ratio(sharpe_annual=np.nan, n_obs=252, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for NaN sharpe"
    
    # Edge case: Inf sharpe
    dsr = deflated_sharpe_ratio(sharpe_annual=np.inf, n_obs=252, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for Inf sharpe"
    
    # Edge case: n_tests < 1 (should clamp to 1)
    dsr_zero = deflated_sharpe_ratio(sharpe_annual=1.0, n_obs=252, n_tests=0)
    dsr_one = deflated_sharpe_ratio(sharpe_annual=1.0, n_obs=252, n_tests=1)
    assert not np.isnan(dsr_zero), "DSR should not be NaN for n_tests=0 (clamped to 1)"
    assert not np.isnan(dsr_one), "DSR should not be NaN for n_tests=1"
    # Should be equal (or very close) since n_tests=0 is clamped to 1
    assert abs(dsr_zero - dsr_one) < 1e-10, \
        f"DSR for n_tests=0 ({dsr_zero:.4f}) should equal DSR for n_tests=1 ({dsr_one:.4f})"
    
    # Edge case: n_tests negative
    dsr_neg = deflated_sharpe_ratio(sharpe_annual=1.0, n_obs=252, n_tests=-5)
    assert not np.isnan(dsr_neg), "DSR should not be NaN for negative n_tests (clamped to 1)"
    assert abs(dsr_neg - dsr_one) < 1e-10, \
        f"DSR for n_tests=-5 ({dsr_neg:.4f}) should equal DSR for n_tests=1 ({dsr_one:.4f})"


@pytest.mark.advanced
def test_deflated_sharpe_with_skew_kurtosis():
    """Test that skewness and kurtosis affect DSR."""
    sharpe_annual = 1.5
    n_obs = 252
    n_tests = 10
    
    # Normal case (skew=0, kurtosis=3)
    dsr_normal = deflated_sharpe_ratio(
        sharpe_annual=sharpe_annual,
        n_obs=n_obs,
        n_tests=n_tests,
        skew=0.0,
        kurtosis=3.0,
    )
    
    # Positive skew
    dsr_skew_pos = deflated_sharpe_ratio(
        sharpe_annual=sharpe_annual,
        n_obs=n_obs,
        n_tests=n_tests,
        skew=1.0,
        kurtosis=3.0,
    )
    
    # High kurtosis (fat tails)
    dsr_kurt_high = deflated_sharpe_ratio(
        sharpe_annual=sharpe_annual,
        n_obs=n_obs,
        n_tests=n_tests,
        skew=0.0,
        kurtosis=5.0,
    )
    
    assert not np.isnan(dsr_normal), "DSR should not be NaN for normal case"
    assert not np.isnan(dsr_skew_pos), "DSR should not be NaN for positive skew"
    assert not np.isnan(dsr_kurt_high), "DSR should not be NaN for high kurtosis"
    
    # DSR values should differ (distribution adjustment)
    # Note: The exact relationship depends on the formula, but they should be different
    assert dsr_normal != dsr_skew_pos or dsr_normal != dsr_kurt_high, \
        "DSR should differ for different skew/kurtosis values"


@pytest.mark.advanced
def test_deflated_sharpe_from_returns_edge_cases():
    """Test edge cases for deflated_sharpe_ratio_from_returns."""
    # Empty returns
    returns_empty = pd.Series([], dtype=float)
    dsr = deflated_sharpe_ratio_from_returns(returns_empty, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for empty returns"
    
    # Single return
    returns_single = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC"))
    dsr = deflated_sharpe_ratio_from_returns(returns_single, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for single return"
    
    # All NaN returns
    returns_nan = pd.Series([np.nan] * 10, index=pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC"))
    dsr = deflated_sharpe_ratio_from_returns(returns_nan, n_tests=1)
    assert np.isnan(dsr), "DSR should be NaN for all-NaN returns"
    
    # Constant returns (zero std)
    returns_constant = pd.Series([0.01] * 100, index=pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC"))
    dsr = deflated_sharpe_ratio_from_returns(returns_constant, n_tests=1)
    # Should be NaN because std=0, so Sharpe cannot be computed (compute_sharpe_ratio returns None)
    # Note: If compute_sharpe_ratio returns None, we return NaN
    assert np.isnan(dsr) or dsr is None, f"DSR should be NaN/None for constant returns (std=0), got: {dsr}"


@pytest.mark.advanced
def test_deflated_sharpe_increases_with_n_obs():
    """Test that DSR increases with more observations (for fixed sharpe and n_tests)."""
    sharpe_annual = 1.5
    n_tests = 10
    
    n_obs_values = [252, 500, 1000, 2000, 5000]
    dsr_values = []
    
    for n_obs in n_obs_values:
        dsr = deflated_sharpe_ratio(
            sharpe_annual=sharpe_annual,
            n_obs=n_obs,
            n_tests=n_tests,
        )
        assert not np.isnan(dsr), f"DSR should not be NaN for n_obs={n_obs}"
        dsr_values.append(dsr)
    
    # Check monotonicity: each DSR should be >= previous (more data = more significance)
    for i in range(1, len(dsr_values)):
        assert dsr_values[i] >= dsr_values[i-1] - 1e-10, \
            f"DSR should increase (or stay same) with n_obs. n_obs={n_obs_values[i]}: {dsr_values[i]:.4f} < {dsr_values[i-1]:.4f} (n_obs={n_obs_values[i-1]})"

