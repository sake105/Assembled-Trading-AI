"""Tests for M7-T03: policy-driven cost model wrapper.

Covers:
- estimate_rebalance_cost_fraction: normal, no change, empty weights, disabled
- compute_cost_drag_per_period: normal, disabled, empty
- get_effective_cost_params: defaults, policy overrides, disabled
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.data.cost_model_policy import (
    compute_cost_drag_per_period,
    estimate_rebalance_cost_fraction,
    get_effective_cost_params,
)


# ---------------------------------------------------------------------------
# estimate_rebalance_cost_fraction
# ---------------------------------------------------------------------------


class TestEstimateRebalanceCostFraction:
    def test_no_change_zero_cost(self):
        w = {"A": 0.5, "B": 0.5}
        result = estimate_rebalance_cost_fraction(w, w)
        assert result == pytest.approx(0.0)

    def test_full_turnover_applies_cost(self):
        # 100% turnover: sell everything, buy everything new
        old = {"A": 0.5, "B": 0.5}
        new = {"C": 0.5, "D": 0.5}
        result = estimate_rebalance_cost_fraction(
            old, new, commission_bps=0.5, half_spread_bps=2.5, slippage_bps=3.0
        )
        # turnover = 2.0 (1.0 per old symbol), one_way = 6 bps
        expected = 2.0 * 6.0 / 10_000.0
        assert result == pytest.approx(expected)

    def test_partial_rebalance(self):
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.6, "B": 0.4}
        result = estimate_rebalance_cost_fraction(
            old, new, commission_bps=0.5, half_spread_bps=2.5, slippage_bps=3.0
        )
        # turnover = |0.6-0.5| + |0.4-0.5| = 0.2
        expected = 0.2 * 6.0 / 10_000.0
        assert result == pytest.approx(expected)

    def test_policy_overrides_defaults(self):
        old = {"A": 0.0}
        new = {"A": 1.0}
        policy = {
            "cost_model": {
                "commission_bps": 1.0,
                "half_spread_bps": 1.0,
                "slippage_bps": 1.0,
            }
        }
        result = estimate_rebalance_cost_fraction(old, new, policy=policy)
        # turnover = 1.0, one_way = 3 bps
        assert result == pytest.approx(3.0 / 10_000.0)

    def test_disabled_in_policy_returns_zero(self):
        old = {"A": 0.0}
        new = {"A": 1.0}
        policy = {"cost_model": {"enabled": False}}
        result = estimate_rebalance_cost_fraction(old, new, policy=policy)
        assert result == 0.0

    def test_empty_weights_returns_zero(self):
        result = estimate_rebalance_cost_fraction({}, {})
        assert result == 0.0

    def test_none_old_weights(self):
        result = estimate_rebalance_cost_fraction(None, {"A": 1.0})  # type: ignore[arg-type]
        assert result > 0.0  # buys from zero

    def test_cost_is_non_negative(self):
        old = {"A": 0.3, "B": 0.7}
        new = {"A": 0.7, "B": 0.3}
        result = estimate_rebalance_cost_fraction(old, new)
        assert result >= 0.0

    def test_symmetric_rebalance_same_cost(self):
        old = {"A": 0.3, "B": 0.7}
        new = {"A": 0.7, "B": 0.3}
        cost_fwd = estimate_rebalance_cost_fraction(old, new)
        cost_rev = estimate_rebalance_cost_fraction(new, old)
        assert cost_fwd == pytest.approx(cost_rev)


# ---------------------------------------------------------------------------
# compute_cost_drag_per_period
# ---------------------------------------------------------------------------


class TestCostDragPerPeriod:
    def test_normal_case(self):
        turnovers = [0.1, 0.2, 0.15]
        result = compute_cost_drag_per_period(
            turnovers, commission_bps=0.5, half_spread_bps=2.5, slippage_bps=3.0
        )
        one_way = 6.0 / 10_000.0
        assert result[0] == pytest.approx(0.1 * one_way)
        assert result[1] == pytest.approx(0.2 * one_way)
        assert result[2] == pytest.approx(0.15 * one_way)

    def test_disabled_returns_zeros(self):
        policy = {"cost_model": {"enabled": False}}
        result = compute_cost_drag_per_period([0.1, 0.2], policy=policy)
        assert result == [0.0, 0.0]

    def test_empty_series_returns_empty(self):
        result = compute_cost_drag_per_period([])
        assert result == []

    def test_length_matches_input(self):
        turnovers = [0.05] * 20
        result = compute_cost_drag_per_period(turnovers)
        assert len(result) == 20


# ---------------------------------------------------------------------------
# get_effective_cost_params
# ---------------------------------------------------------------------------


class TestGetEffectiveCostParams:
    def test_defaults_without_policy(self):
        params = get_effective_cost_params()
        assert params["commission_bps"] == pytest.approx(0.5)
        assert params["half_spread_bps"] == pytest.approx(2.5)
        assert params["slippage_bps"] == pytest.approx(3.0)
        assert params["one_way_cost_bps"] == pytest.approx(6.0)
        assert params["enabled"] is True

    def test_policy_overrides_commission(self):
        policy = {"cost_model": {"commission_bps": 2.0}}
        params = get_effective_cost_params(policy=policy)
        assert params["commission_bps"] == pytest.approx(2.0)
        assert params["one_way_cost_bps"] == pytest.approx(2.0 + 2.5 + 3.0)

    def test_disabled_in_policy(self):
        policy = {"cost_model": {"enabled": False}}
        params = get_effective_cost_params(policy=policy)
        assert params["enabled"] is False

    def test_returns_all_required_keys(self):
        params = get_effective_cost_params()
        required = {
            "commission_bps",
            "half_spread_bps",
            "slippage_bps",
            "one_way_cost_bps",
            "enabled",
        }
        assert required.issubset(params.keys())
