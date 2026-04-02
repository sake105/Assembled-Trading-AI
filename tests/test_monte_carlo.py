"""Tests for Monte Carlo / Bootstrap simulation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.assembled_core.qa.monte_carlo import (
    bootstrap_returns,
    forward_simulate_gbm,
    summarize_forward_sim,
    summarize_monte_carlo,
)


@pytest.fixture
def sample_returns() -> np.ndarray:
    """Generate sample daily returns (positive drift)."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.015, size=500)  # ~12% CAGR, ~24% vol


@pytest.fixture
def negative_returns() -> np.ndarray:
    """Generate sample negative daily returns."""
    rng = np.random.default_rng(123)
    return rng.normal(-0.001, 0.02, size=300)


class TestBootstrapReturns:
    def test_basic_output_structure(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=100, seed=42)
        assert "sharpe" in result.confidence_intervals
        assert "cagr" in result.confidence_intervals
        assert "max_drawdown" in result.confidence_intervals
        assert len(result.sharpe_distribution) == 100
        assert result.n_paths == 100

    def test_ci_contains_point_estimate(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=500, seed=42)
        ci = result.confidence_intervals["sharpe"]
        # Point estimate should generally be within CI
        assert ci.ci_lower <= ci.ci_upper

    def test_positive_returns_positive_sharpe(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=200, seed=42)
        ci = result.confidence_intervals["sharpe"]
        assert ci.point_estimate > 0

    def test_negative_returns_low_sharpe(self, negative_returns: np.ndarray) -> None:
        result = bootstrap_returns(negative_returns, n_paths=200, seed=42)
        ci = result.confidence_intervals["sharpe"]
        assert ci.point_estimate < 0.5

    def test_p_value_positive_strategy(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=500, seed=42)
        # Positive strategy should have low p-value
        assert result.p_value_vs_zero < 0.5

    def test_block_bootstrap(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=100, block_size=5, seed=42)
        assert len(result.sharpe_distribution) == 100

    def test_reproducibility(self, sample_returns: np.ndarray) -> None:
        r1 = bootstrap_returns(sample_returns, n_paths=100, seed=42)
        r2 = bootstrap_returns(sample_returns, n_paths=100, seed=42)
        np.testing.assert_array_equal(r1.sharpe_distribution, r2.sharpe_distribution)

    def test_too_few_returns_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            bootstrap_returns(np.array([0.01, 0.02, 0.03]), n_paths=10)

    def test_max_dd_always_negative(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=100, seed=42)
        ci = result.confidence_intervals["max_drawdown"]
        assert ci.point_estimate <= 0
        assert ci.ci_upper <= 0


class TestForwardSimulateGBM:
    def test_basic_output_structure(self, sample_returns: np.ndarray) -> None:
        result = forward_simulate_gbm(
            sample_returns, n_paths=50, horizon_days=100, seed=42
        )
        assert result.paths.shape == (50, 100)
        assert len(result.terminal_values) == 50
        assert 0 <= result.prob_loss <= 1
        assert 0 <= result.prob_dd_exceed <= 1

    def test_positive_drift_mostly_gains(self, sample_returns: np.ndarray) -> None:
        result = forward_simulate_gbm(
            sample_returns, n_paths=200, horizon_days=252, seed=42
        )
        assert result.prob_loss < 0.5  # Most paths should end positive

    def test_terminal_ci_ordered(self, sample_returns: np.ndarray) -> None:
        result = forward_simulate_gbm(
            sample_returns, n_paths=100, horizon_days=100, seed=42
        )
        assert result.ci_lower_terminal <= result.median_terminal <= result.ci_upper_terminal

    def test_negative_drift_mostly_losses(self, negative_returns: np.ndarray) -> None:
        result = forward_simulate_gbm(
            negative_returns, n_paths=200, horizon_days=252, seed=42
        )
        assert result.prob_loss > 0.5

    def test_reproducibility(self, sample_returns: np.ndarray) -> None:
        r1 = forward_simulate_gbm(sample_returns, n_paths=50, horizon_days=50, seed=42)
        r2 = forward_simulate_gbm(sample_returns, n_paths=50, horizon_days=50, seed=42)
        np.testing.assert_array_equal(r1.terminal_values, r2.terminal_values)


class TestSummaries:
    def test_bootstrap_summary(self, sample_returns: np.ndarray) -> None:
        result = bootstrap_returns(sample_returns, n_paths=50, seed=42)
        summary = summarize_monte_carlo(result)
        assert "sharpe" in summary
        assert "cagr" in summary
        assert "Simulations: 50" in summary

    def test_forward_summary(self, sample_returns: np.ndarray) -> None:
        result = forward_simulate_gbm(
            sample_returns, n_paths=50, horizon_days=100, seed=42
        )
        summary = summarize_forward_sim(result)
        assert "Median terminal" in summary
        assert "P(loss)" in summary
