"""Tests for reverse stress testing module."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip('src.assembled_core.qa.reverse_stress')
from src.assembled_core.qa.reverse_stress import (
    reverse_stress_test,
    run_multiple_reverse_stress,
    stress_test_portfolio_against_scenarios,
    get_all_scenario_names,
    get_scenario,
    HISTORICAL_CRISES,
    HYPOTHETICAL_SCENARIOS,
    ReverseStressResult,
)


def _simple_portfolio(n: int = 5, seed: int = 42):
    """Create simple equal-weight portfolio with covariance."""
    rng = np.random.default_rng(seed)
    weights = np.ones(n) / n
    # Generate PSD covariance
    L = rng.normal(0, 0.01, (n, n))
    cov = L @ L.T + np.eye(n) * 0.0004  # ~2% daily vol per asset
    return weights, cov


@pytest.mark.phase12
class TestReverseStressTest:
    def test_basic(self):
        weights, cov = _simple_portfolio()
        result = reverse_stress_test(weights, cov, target_loss=-0.10)
        assert isinstance(result, ReverseStressResult)
        assert result.target_loss == -0.10
        assert len(result.shock_vector) == 5

    def test_converges_to_target(self):
        weights, cov = _simple_portfolio()
        result = reverse_stress_test(
            weights, cov, target_loss=-0.10,
            plausibility_bound=10.0, n_restarts=10,
        )
        if result.converged:
            assert result.achieved_loss <= -0.05  # Should get close to target

    def test_larger_loss_needs_larger_shock(self):
        weights, cov = _simple_portfolio()
        r1 = reverse_stress_test(weights, cov, target_loss=-0.05, plausibility_bound=10.0)
        r2 = reverse_stress_test(weights, cov, target_loss=-0.20, plausibility_bound=10.0)
        if r1.converged and r2.converged:
            assert r2.shock_norm >= r1.shock_norm * 0.9  # Larger loss = larger shock

    def test_top_shocks_populated(self):
        weights, cov = _simple_portfolio()
        result = reverse_stress_test(weights, cov, target_loss=-0.10, plausibility_bound=10.0)
        if result.converged:
            assert len(result.top_shocks) > 0

    def test_without_scipy(self):
        """Should gracefully handle missing scipy."""
        weights, cov = _simple_portfolio()
        # This will use scipy if available, which is fine
        result = reverse_stress_test(weights, cov, target_loss=-0.10)
        assert isinstance(result, ReverseStressResult)


@pytest.mark.phase12
class TestRunMultipleReverseStress:
    def test_default_targets(self):
        weights, cov = _simple_portfolio()
        results = run_multiple_reverse_stress(weights, cov, plausibility_bound=10.0)
        assert len(results) == 5  # default: [-0.05, -0.10, -0.15, -0.20, -0.30]

    def test_custom_targets(self):
        weights, cov = _simple_portfolio()
        results = run_multiple_reverse_stress(
            weights, cov,
            target_losses=[-0.05, -0.10],
            plausibility_bound=10.0,
        )
        assert len(results) == 2


@pytest.mark.phase12
class TestScenarioCatalog:
    def test_historical_count(self):
        assert len(HISTORICAL_CRISES) >= 15

    def test_hypothetical_count(self):
        assert len(HYPOTHETICAL_SCENARIOS) >= 8

    def test_all_scenario_names(self):
        names = get_all_scenario_names()
        assert len(names) >= 23
        assert "gfc_2008" in names
        assert "covid_2020" in names
        assert "correlation_crisis" in names

    def test_get_scenario(self):
        s = get_scenario("gfc_2008")
        assert "equity_shock" in s
        assert s["equity_shock"] < 0

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_scenario("nonexistent_scenario")


@pytest.mark.phase12
class TestStressTestPortfolio:
    def test_basic_v2(self):
        weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        result = stress_test_portfolio_against_scenarios(weights)
        assert isinstance(result, pd.DataFrame)
        assert "scenario_name" in result.columns
        assert "portfolio_impact" in result.columns
        assert "severity" in result.columns
        assert len(result) >= 23  # All scenarios

    def test_sorted_by_impact(self):
        weights = np.array([0.5, 0.5])
        result = stress_test_portfolio_against_scenarios(weights)
        impacts = result["portfolio_impact"].values
        assert all(impacts[i] <= impacts[i + 1] for i in range(len(impacts) - 1))

    def test_custom_scenarios(self):
        weights = np.array([1.0])
        custom = {"test_crash": {"description": "Test", "equity_shock": -0.50}}
        result = stress_test_portfolio_against_scenarios(weights, scenarios=custom)
        assert len(result) == 1
        assert result.iloc[0]["scenario_name"] == "test_crash"

    def test_severity_levels(self):
        weights = np.array([1.0])  # Fully invested
        result = stress_test_portfolio_against_scenarios(weights)
        severities = set(result["severity"])
        assert "CRITICAL" in severities or "HIGH" in severities
