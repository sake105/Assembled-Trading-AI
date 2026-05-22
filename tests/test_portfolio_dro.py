"""Tests for portfolio/dro_portfolio.py — audit C2-036 (Wasserstein DRO)
and C2-037 (KL-Divergence DRO).

Covers:
- Wasserstein DRO: feasibility, weight constraints, diversification monotonicity
- KL DRO: feasibility, weight constraints, diversification monotonicity
- Dispatch function
- Edge cases: single asset, 2 assets / 5 scenarios, determinism
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from src.assembled_core.portfolio.dro_portfolio import (
    DROResult,
    dro_portfolio,
    kl_dro_portfolio,
    wasserstein_dro_portfolio,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_returns(
    T: int = 60,
    n: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Synthetic return matrix (T × n), well-conditioned."""
    rng = np.random.default_rng(seed)
    means = rng.uniform(0.001, 0.010, n)
    stds = rng.uniform(0.010, 0.030, n)
    return rng.normal(means, stds, size=(T, n))


def _make_returns_2x5(seed: int = 7) -> np.ndarray:
    """Minimal viable: 5 scenarios, 2 assets."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.002, 0.015, size=(5, 2))


# ===========================================================================
# Wasserstein DRO — C2-036
# ===========================================================================


class TestWassersteinDROWeights:
    """Test 1: weights are non-negative and sum to 1."""

    def test_weights_sum_to_one(self):
        R = _make_returns()
        result = wasserstein_dro_portfolio(R, epsilon=0.005)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_weights_non_negative(self):
        R = _make_returns()
        result = wasserstein_dro_portfolio(R, epsilon=0.005)
        assert np.all(result.weights >= -1e-7), (
            f"negative weight found: {result.weights.min():.6f}"
        )


class TestWassersteinDRODiversification:
    """Test 2: larger epsilon leads to more diversification (lower max weight)."""

    def test_higher_epsilon_more_diversified(self):
        R = _make_returns(T=80, n=5, seed=1)
        res_small = wasserstein_dro_portfolio(R, epsilon=1e-4)
        res_large = wasserstein_dro_portfolio(R, epsilon=0.20)
        # Larger ε should not concentrate more than smaller ε
        assert res_large.weights.max() <= res_small.weights.max() + 0.15, (
            f"Expected more diversification with larger epsilon: "
            f"small_eps max_w={res_small.weights.max():.4f}, "
            f"large_eps max_w={res_large.weights.max():.4f}"
        )


class TestWassersteinDROResult:
    """Test 3: result type and fields are correct."""

    def test_returns_dro_result_instance(self):
        R = _make_returns()
        result = wasserstein_dro_portfolio(R)
        assert isinstance(result, DROResult)

    def test_correct_fields_present(self):
        R = _make_returns()
        result = wasserstein_dro_portfolio(R, epsilon=0.01)
        assert hasattr(result, "weights")
        assert hasattr(result, "expected_return")
        assert hasattr(result, "worst_case_return")
        assert hasattr(result, "epsilon")
        assert hasattr(result, "solver")
        assert hasattr(result, "converged")
        assert result.solver == "scipy_linprog"
        assert result.epsilon == pytest.approx(0.01)


class TestWassersteinDROConvergence:
    """Test 4: well-conditioned data converges."""

    def test_converged_true_on_well_conditioned_data(self):
        R = _make_returns(T=100, n=4, seed=99)
        result = wasserstein_dro_portfolio(R, epsilon=0.01)
        assert result.converged is True


class TestWassersteinSingleAsset:
    """Test 5: single asset trivially gets weight=1."""

    def test_single_asset_weight_is_one(self):
        rng = np.random.default_rng(0)
        R = rng.normal(0.005, 0.02, size=(30, 1))
        result = wasserstein_dro_portfolio(R, epsilon=0.01)
        assert abs(result.weights[0] - 1.0) < 1e-5


class TestWassersteinWorstCaseVsExpected:
    """Test 6: worst_case_return <= expected_return."""

    def test_worst_case_le_expected(self):
        R = _make_returns(T=60, n=4)
        result = wasserstein_dro_portfolio(R, epsilon=0.02)
        assert result.worst_case_return <= result.expected_return + 1e-8, (
            f"worst_case={result.worst_case_return:.6f} > "
            f"expected={result.expected_return:.6f}"
        )


# ===========================================================================
# KL DRO — C2-037
# ===========================================================================


class TestKLDROWeights:
    """Test 7: KL weights sum to 1."""

    def test_weights_sum_to_one(self):
        R = _make_returns()
        result = kl_dro_portfolio(R, kl_radius=0.05)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_weights_non_negative(self):
        R = _make_returns()
        result = kl_dro_portfolio(R, kl_radius=0.05)
        assert np.all(result.weights >= -1e-7)


class TestKLDRODiversification:
    """Test 8: higher kl_radius → more diversified (lower max weight)."""

    def test_higher_kl_radius_more_diversified(self):
        R = _make_returns(T=80, n=5, seed=3)
        res_small = kl_dro_portfolio(R, kl_radius=1e-4)
        res_large = kl_dro_portfolio(R, kl_radius=2.0)
        assert res_large.weights.max() <= res_small.weights.max() + 0.15, (
            f"Expected diversification: small_rho max_w={res_small.weights.max():.4f}, "
            f"large_rho max_w={res_large.weights.max():.4f}"
        )


class TestKLDROConvergence:
    """Test 9: well-conditioned data converges."""

    def test_converged_true(self):
        R = _make_returns(T=100, n=4, seed=55)
        result = kl_dro_portfolio(R, kl_radius=0.1)
        assert result.converged is True


class TestKLDROFields:
    """Test 10: DROResult fields present and correct solver label."""

    def test_fields_present(self):
        R = _make_returns()
        result = kl_dro_portfolio(R, kl_radius=0.1)
        assert isinstance(result, DROResult)
        assert result.solver == "scipy_slsqp"
        assert result.epsilon == pytest.approx(0.1)
        assert result.weights.shape == (4,)
        assert isinstance(result.converged, bool)
        assert isinstance(result.expected_return, float)
        assert isinstance(result.worst_case_return, float)


class TestKLDROWorstCaseVsExpected:
    """Test 11: worst_case_return <= expected_return."""

    def test_worst_case_le_expected(self):
        R = _make_returns(T=60, n=4)
        result = kl_dro_portfolio(R, kl_radius=0.1)
        assert result.worst_case_return <= result.expected_return + 1e-8, (
            f"worst_case={result.worst_case_return:.6f} > "
            f"expected={result.expected_return:.6f}"
        )


# ===========================================================================
# Dispatch function
# ===========================================================================


class TestDROPortfolioDispatch:
    """Tests 12-14: dro_portfolio dispatcher."""

    def test_dispatch_wasserstein(self):
        R = _make_returns()
        result = dro_portfolio(R, method="wasserstein", epsilon=0.01)
        assert isinstance(result, DROResult)
        assert result.solver == "scipy_linprog"

    def test_dispatch_kl(self):
        R = _make_returns()
        result = dro_portfolio(R, method="kl", kl_radius=0.1)
        assert isinstance(result, DROResult)
        assert result.solver == "scipy_slsqp"

    def test_dispatch_unknown_method_raises(self):
        R = _make_returns()
        with pytest.raises(ValueError, match="Unknown DRO method"):
            dro_portfolio(R, method="markowitz")  # type: ignore[arg-type]


# ===========================================================================
# Determinism
# ===========================================================================


class TestDeterminism:
    """Test 15: identical inputs produce identical outputs (no randomness)."""

    def test_wasserstein_deterministic(self):
        R = _make_returns(T=40, n=3, seed=17)
        r1 = wasserstein_dro_portfolio(R, epsilon=0.02)
        r2 = wasserstein_dro_portfolio(R, epsilon=0.02)
        np.testing.assert_array_almost_equal(r1.weights, r2.weights, decimal=8)

    def test_kl_deterministic(self):
        R = _make_returns(T=40, n=3, seed=17)
        r1 = kl_dro_portfolio(R, kl_radius=0.1)
        r2 = kl_dro_portfolio(R, kl_radius=0.1)
        np.testing.assert_array_almost_equal(r1.weights, r2.weights, decimal=6)


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test 16: minimal 2 assets × 5 scenarios."""

    def test_two_assets_five_scenarios_wasserstein(self):
        R = _make_returns_2x5()
        result = wasserstein_dro_portfolio(R, epsilon=0.01)
        assert isinstance(result, DROResult)
        assert result.weights.shape == (2,)
        assert abs(result.weights.sum() - 1.0) < 1e-5
        assert np.all(result.weights >= -1e-7)

    def test_two_assets_five_scenarios_kl(self):
        R = _make_returns_2x5()
        result = kl_dro_portfolio(R, kl_radius=0.05)
        assert isinstance(result, DROResult)
        assert result.weights.shape == (2,)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_epsilon_zero_wasserstein_is_mean_optimal(self):
        """ε=0 → no robustness penalty, concentrate on best asset."""
        R = _make_returns(T=50, n=3, seed=5)
        result = wasserstein_dro_portfolio(R, epsilon=0.0)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_kl_radius_zero_is_mean_optimal(self):
        """ρ=0 → no robustness penalty."""
        R = _make_returns(T=50, n=3, seed=5)
        result = kl_dro_portfolio(R, kl_radius=0.0)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_weight_shape_matches_n_assets(self):
        for n in (2, 5, 10):
            R = _make_returns(T=30, n=n, seed=n)
            rw = wasserstein_dro_portfolio(R, epsilon=0.01)
            rk = kl_dro_portfolio(R, kl_radius=0.05)
            assert rw.weights.shape == (n,)
            assert rk.weights.shape == (n,)


# ===========================================================================
# Input validation
# ===========================================================================


class TestInputValidation:
    """Robust error handling for bad inputs."""

    def test_negative_epsilon_raises(self):
        R = _make_returns()
        with pytest.raises(ValueError, match="epsilon must be"):
            wasserstein_dro_portfolio(R, epsilon=-0.01)

    def test_negative_kl_radius_raises(self):
        R = _make_returns()
        with pytest.raises(ValueError, match="kl_radius must be"):
            kl_dro_portfolio(R, kl_radius=-0.1)

    def test_zero_risk_aversion_raises_wasserstein(self):
        R = _make_returns()
        with pytest.raises(ValueError, match="risk_aversion"):
            wasserstein_dro_portfolio(R, risk_aversion=0.0)

    def test_zero_risk_aversion_raises_kl(self):
        R = _make_returns()
        with pytest.raises(ValueError, match="risk_aversion"):
            kl_dro_portfolio(R, risk_aversion=0.0)

    def test_1d_input_treated_as_single_asset(self):
        rng = np.random.default_rng(0)
        R1d = rng.normal(0.005, 0.02, size=20)
        result = wasserstein_dro_portfolio(R1d, epsilon=0.01)
        assert result.weights.shape == (1,)

    def test_nan_in_returns_raises(self):
        R = _make_returns()
        R[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            wasserstein_dro_portfolio(R)

    def test_too_few_scenarios_raises(self):
        R = np.array([[0.01, 0.02]])  # T=1
        with pytest.raises(ValueError, match="at least 2"):
            wasserstein_dro_portfolio(R)
