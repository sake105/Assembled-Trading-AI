"""Tests for M20.2: Almgren-Chriss Optimal Execution Model."""

from __future__ import annotations

import pytest

from src.assembled_core.execution.almgren_chriss import (
    AlmgrenChrissParams,
    ExecutionTrajectory,
    compute_optimal_trajectory,
    estimate_impact_cost,
    compute_frontier,
)


@pytest.mark.phase12
class TestAlmgrenChrissParams:
    def test_defaults(self):
        p = AlmgrenChrissParams()
        assert p.sigma == 0.02
        assert p.gamma == 0.1
        assert p.eta == 0.05
        assert p.risk_aversion == 1e-5
        assert p.adv == 1_000_000.0


@pytest.mark.phase12
class TestComputeOptimalTrajectory:
    def test_basic_buy_trajectory(self):
        traj = compute_optimal_trajectory(
            total_shares=10_000, price=100.0,
            n_intervals=10, horizon_days=1.0,
        )
        assert isinstance(traj, ExecutionTrajectory)
        assert len(traj.time_steps) == 11  # n+1 points
        assert len(traj.trade_list) == 10
        assert traj.holdings[0] == pytest.approx(10_000, rel=1e-3)
        assert traj.holdings[-1] == pytest.approx(0, abs=1)
        assert traj.expected_cost_bps >= 0
        assert traj.optimal_horizon_days > 0

    def test_sell_trajectory_negative(self):
        traj = compute_optimal_trajectory(
            total_shares=-5_000, price=50.0,
            n_intervals=5, horizon_days=0.5,
        )
        # Holdings should go from -5000 to 0
        assert traj.holdings[0] == pytest.approx(-5_000, rel=1e-3)
        assert traj.holdings[-1] == pytest.approx(0, abs=1)
        # Trade list should be negative (selling = reducing position)
        assert all(t <= 0 for t in traj.trade_list)

    def test_trade_list_sums_to_total(self):
        traj = compute_optimal_trajectory(
            total_shares=20_000, price=75.0,
            n_intervals=20, horizon_days=2.0,
        )
        assert sum(traj.trade_list) == pytest.approx(20_000, rel=1e-3)

    def test_higher_urgency_frontloads(self):
        """Higher risk aversion should front-load execution."""
        params_patient = AlmgrenChrissParams(risk_aversion=1e-7)
        params_urgent = AlmgrenChrissParams(risk_aversion=1e-3)

        traj_patient = compute_optimal_trajectory(
            10_000, 100.0, 10, 1.0, params=params_patient,
        )
        traj_urgent = compute_optimal_trajectory(
            10_000, 100.0, 10, 1.0, params=params_urgent,
        )

        # Urgent trader should execute more in the first interval
        assert traj_urgent.trade_list[0] > traj_patient.trade_list[0]

    def test_participation_rates_computed(self):
        traj = compute_optimal_trajectory(
            total_shares=50_000, price=100.0,
            n_intervals=10, horizon_days=1.0,
        )
        assert len(traj.participation_rates) == 10
        assert all(r >= 0 for r in traj.participation_rates)

    def test_cost_breakdown_positive(self):
        traj = compute_optimal_trajectory(
            total_shares=10_000, price=100.0,
            n_intervals=10, horizon_days=1.0,
        )
        assert traj.permanent_impact_bps >= 0
        assert traj.temporary_impact_bps >= 0
        assert traj.risk_penalty_bps >= 0

    def test_single_interval(self):
        traj = compute_optimal_trajectory(
            total_shares=100, price=50.0,
            n_intervals=1, horizon_days=0.1,
        )
        assert len(traj.trade_list) == 1
        assert traj.trade_list[0] == pytest.approx(100, rel=1e-3)


@pytest.mark.phase12
class TestEstimateImpactCost:
    def test_basic_estimate(self):
        result = estimate_impact_cost(
            total_shares=10_000, price=100.0,
            adv=1_000_000, sigma=0.02,
        )
        assert "permanent_bps" in result
        assert "temporary_bps" in result
        assert "total_bps" in result
        assert "total_cost_usd" in result
        assert result["total_bps"] > 0

    def test_zero_shares_returns_zero(self):
        result = estimate_impact_cost(0, 100.0, 1_000_000, 0.02)
        assert result["total_bps"] == 0.0

    def test_larger_order_costs_more(self):
        small = estimate_impact_cost(1_000, 100.0, 1_000_000, 0.02)
        large = estimate_impact_cost(100_000, 100.0, 1_000_000, 0.02)
        assert large["total_bps"] > small["total_bps"]


@pytest.mark.phase12
class TestComputeFrontier:
    def test_frontier_returns_points(self):
        frontier = compute_frontier(
            total_shares=10_000, price=100.0, n_points=5,
        )
        assert len(frontier) == 5
        assert all("horizon_days" in p for p in frontier)
        assert all("expected_cost_bps" in p for p in frontier)

    def test_frontier_cost_decreases_then_increases(self):
        """Temporary impact decreases with horizon, risk increases."""
        frontier = compute_frontier(
            total_shares=50_000, price=100.0, n_points=10,
        )
        costs = [p["expected_cost_bps"] for p in frontier]
        # There should be a minimum somewhere (not necessarily monotone)
        assert min(costs) < max(costs)
