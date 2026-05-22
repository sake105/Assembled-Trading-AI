"""Tests for C2-006 differential testing module.

Verifies that the three Sharpe implementations (NumPy, Polars, Numba) agree
within ε = 1e-9 and that the DiffTestResult dataclass is correctly populated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.assembled_core.qa.differential_testing import (
    DiffTestResult,
    HAS_NUMBA,
    HAS_POLARS,
    diff_test_sharpe,
    sharpe_numpy,
    sharpe_numba,
    sharpe_polars,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG_SEED = 42


@pytest.fixture()
def random_returns() -> np.ndarray:
    """100 random daily returns drawn from N(0.001, 0.02)."""
    rng = np.random.default_rng(RNG_SEED)
    return rng.normal(0.001, 0.02, size=100)


@pytest.fixture()
def positive_returns() -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    return np.abs(rng.normal(0.002, 0.005, size=100)) + 0.0001


@pytest.fixture()
def negative_returns() -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED)
    return -np.abs(rng.normal(0.002, 0.005, size=100)) - 0.0001


# ---------------------------------------------------------------------------
# Test 1: NumPy / Polars / Numba agree within 1e-9 on random returns
# ---------------------------------------------------------------------------


def test_all_implementations_agree_on_random(random_returns):
    """All three implementations must agree within 1e-9."""
    result = diff_test_sharpe(random_returns, epsilon=1e-9)
    assert result.passed, (
        f"max_abs_diff={result.max_abs_diff} > epsilon={result.epsilon}; "
        f"numpy={result.numpy_val}, polars={result.polars_val}, numba={result.numba_val}"
    )


# ---------------------------------------------------------------------------
# Test 2: All-positive returns → positive Sharpe
# ---------------------------------------------------------------------------


def test_positive_returns_positive_sharpe(positive_returns):
    """Sharpe must be positive for uniformly positive returns."""
    nv = sharpe_numpy(positive_returns)
    assert nv > 0, f"Expected positive Sharpe, got {nv}"


# ---------------------------------------------------------------------------
# Test 3: All-negative returns → negative Sharpe
# ---------------------------------------------------------------------------


def test_negative_returns_negative_sharpe(negative_returns):
    """Sharpe must be negative for uniformly negative returns."""
    nv = sharpe_numpy(negative_returns)
    assert nv < 0, f"Expected negative Sharpe, got {nv}"


# ---------------------------------------------------------------------------
# Test 4: rf subtraction is reflected correctly
# ---------------------------------------------------------------------------


def test_rf_subtraction(random_returns):
    """Higher rf → lower (or equal) Sharpe."""
    sharpe_no_rf = sharpe_numpy(random_returns, rf=0.0)
    sharpe_with_rf = sharpe_numpy(random_returns, rf=0.001)
    # mean decreases by rf; std stays the same → Sharpe decreases
    assert sharpe_with_rf <= sharpe_no_rf


# ---------------------------------------------------------------------------
# Test 5: DiffTestResult.passed=True for identical implementations
# ---------------------------------------------------------------------------


def test_diff_test_result_passed_true(random_returns):
    """diff_test_sharpe should pass with default epsilon on any normal series."""
    result = diff_test_sharpe(random_returns)
    assert isinstance(result.passed, bool)
    assert result.passed


# ---------------------------------------------------------------------------
# Test 6: Injected arithmetic drift triggers failure at tight epsilon
# ---------------------------------------------------------------------------


def test_inject_drift_triggers_failure(random_returns):
    """Manually introduce a tiny numeric discrepancy and detect it at ε=1e-15."""
    # Monkeypatch sharpe_polars to return slightly shifted value
    import src.assembled_core.qa.differential_testing as dt_mod

    original = dt_mod.sharpe_polars

    def drifted(returns, rf=0.0):
        return original(returns, rf) + 1e-13  # deliberate drift

    dt_mod.sharpe_polars = drifted
    try:
        result = diff_test_sharpe(random_returns, epsilon=1e-15)
        # If polars is available the drift must be detected
        if HAS_POLARS:
            assert not result.passed, "Expected failure due to injected drift"
            assert result.max_abs_diff > 1e-15
    finally:
        dt_mod.sharpe_polars = original


# ---------------------------------------------------------------------------
# Test 7: Numba result equals NumPy on synthetic data
# ---------------------------------------------------------------------------


def test_numba_equals_numpy(random_returns):
    """sharpe_numba and sharpe_numpy must agree within 1e-9."""
    if not HAS_NUMBA:
        pytest.skip("numba not available")
    nv = sharpe_numpy(random_returns)
    bv = sharpe_numba(random_returns)
    assert abs(nv - bv) < 1e-9, f"Numba {bv} != numpy {nv}"


# ---------------------------------------------------------------------------
# Test 8: Polars result equals NumPy on synthetic data
# ---------------------------------------------------------------------------


def test_polars_equals_numpy(random_returns):
    """sharpe_polars and sharpe_numpy must agree within 1e-9."""
    if not HAS_POLARS:
        pytest.skip("polars not available")
    nv = sharpe_numpy(random_returns)
    pv = sharpe_polars(random_returns)
    assert abs(nv - pv) < 1e-9, f"Polars {pv} != numpy {nv}"


# ---------------------------------------------------------------------------
# Test 9: Edge — single-element series handled without crash
# ---------------------------------------------------------------------------


def test_single_element_no_crash():
    """A single-element array must return NaN, not raise."""
    arr = np.array([0.01])
    nv = sharpe_numpy(arr)
    assert math.isnan(nv)
    if HAS_POLARS:
        pv = sharpe_polars(arr)
        assert math.isnan(pv)
    if HAS_NUMBA:
        bv = sharpe_numba(arr)
        assert math.isnan(bv)


# ---------------------------------------------------------------------------
# Test 10: Edge — zero-variance series returns NaN or inf, no crash
# ---------------------------------------------------------------------------


def test_zero_variance_no_crash():
    """Constant returns must produce NaN (zero std), not raise."""
    arr = np.full(50, 0.001)
    nv = sharpe_numpy(arr)
    assert math.isnan(nv) or math.isinf(nv)
    if HAS_POLARS:
        pv = sharpe_polars(arr)
        assert math.isnan(pv) or math.isinf(pv)


# ---------------------------------------------------------------------------
# Test 11: Deterministic — same seed → same DiffTestResult
# ---------------------------------------------------------------------------


def test_deterministic_same_seed():
    """Two calls with the same array must produce identical DiffTestResult."""
    rng = np.random.default_rng(123)
    arr = rng.normal(0.001, 0.02, 200)
    r1 = diff_test_sharpe(arr)
    r2 = diff_test_sharpe(arr)
    assert r1.numpy_val == r2.numpy_val
    assert r1.max_abs_diff == r2.max_abs_diff
    assert r1.passed == r2.passed


# ---------------------------------------------------------------------------
# Test 12: DiffTestResult fields all present with correct types
# ---------------------------------------------------------------------------


def test_diff_test_result_fields(random_returns):
    """DiffTestResult must expose all required fields with correct types."""
    result = diff_test_sharpe(random_returns)
    assert isinstance(result, DiffTestResult)
    assert hasattr(result, "numpy_val")
    assert hasattr(result, "polars_val")
    assert hasattr(result, "numba_val")
    assert hasattr(result, "max_abs_diff")
    assert hasattr(result, "passed")
    assert hasattr(result, "epsilon")
    assert isinstance(result.numpy_val, float)
    assert isinstance(result.max_abs_diff, float)
    assert isinstance(result.passed, bool)
    assert result.max_abs_diff >= 0.0


# ---------------------------------------------------------------------------
# Test 13: rf=0 and rf!=0 produce different Sharpe values (regression guard)
# ---------------------------------------------------------------------------


def test_rf_nonzero_changes_result(random_returns):
    """Changing rf must change the Sharpe value (unless mean is exactly rf)."""
    r1 = diff_test_sharpe(random_returns, rf=0.0)
    r2 = diff_test_sharpe(random_returns, rf=0.005)
    # Different rf → different numpy_val (mean changes, std does not)
    assert r1.numpy_val != r2.numpy_val


# ---------------------------------------------------------------------------
# Test 14: HAS_NUMBA and HAS_POLARS are boolean flags
# ---------------------------------------------------------------------------


def test_availability_flags():
    """HAS_NUMBA and HAS_POLARS must be booleans."""
    assert isinstance(HAS_NUMBA, bool)
    assert isinstance(HAS_POLARS, bool)
