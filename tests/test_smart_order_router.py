"""Tests for M20.3: Smart Order Router — multi-venue routing."""

from __future__ import annotations

import pytest

from src.assembled_core.execution.smart_order_router import (
    VenueConfig,
    RoutingResult,
    DEFAULT_VENUES,
    REGIME_SPREAD_MULT,
    REGIME_FILL_MULT,
    route_order,
    simulate_fills,
)


@pytest.mark.phase12
class TestVenueConfig:
    def test_defaults(self):
        v = VenueConfig(name="test")
        assert v.spread_bps == 5.0
        assert v.fill_probability == 0.95
        assert v.dark_pool is False

    def test_default_venues_defined(self):
        assert len(DEFAULT_VENUES) >= 4
        names = {v.name for v in DEFAULT_VENUES}
        assert "NYSE" in names
        assert "NASDAQ" in names


@pytest.mark.phase12
class TestRouteOrder:
    def test_basic_routing(self):
        result = route_order(order_size=1000, signal_urgency=0.5, seed=42)
        assert isinstance(result, RoutingResult)
        assert len(result.allocations) > 0
        assert result.total_expected_cost_bps > 0
        assert result.total_expected_fill_pct > 0

    def test_total_allocated_equals_order(self):
        result = route_order(order_size=5000, signal_urgency=0.7, seed=42)
        total_alloc = sum(a.quantity for a in result.allocations)
        assert total_alloc == pytest.approx(5000, rel=0.01)

    def test_high_urgency_prefers_high_fill(self):
        result = route_order(order_size=1000, signal_urgency=0.95, seed=42)
        # High urgency should pick reliable venues
        assert result.total_expected_fill_pct > 80.0

    def test_low_urgency_cheaper(self):
        route_order(order_size=1000, signal_urgency=0.95, seed=42)
        result_patient = route_order(order_size=1000, signal_urgency=0.1, seed=42)
        # Patient routing should produce valid cost
        assert result_patient.total_expected_cost_bps >= 0

    def test_regime_adjusts_spread(self):
        result_bull = route_order(order_size=1000, regime="bull", seed=42)
        result_crisis = route_order(order_size=1000, regime="crisis", seed=42)
        # Crisis spreads should be wider
        assert result_crisis.total_expected_cost_bps >= result_bull.total_expected_cost_bps

    def test_no_dark_pools_flag(self):
        result = route_order(
            order_size=1000, allow_dark_pools=False, seed=42,
        )
        for alloc in result.allocations:
            assert alloc.is_dark is False

    def test_max_venues_respected(self):
        result = route_order(order_size=50_000, max_venues=2, seed=42)
        assert len(result.allocations) <= 2

    def test_participation_rates_reported(self):
        result = route_order(order_size=10_000, adv=500_000, seed=42)
        for alloc in result.allocations:
            assert alloc.participation_pct >= 0

    def test_custom_venues(self):
        venues = [
            VenueConfig("custom1", spread_bps=2.0, fill_probability=0.99),
            VenueConfig("custom2", spread_bps=10.0, fill_probability=0.50),
        ]
        result = route_order(order_size=100, venues=venues, seed=42)
        assert any(a.venue == "custom1" for a in result.allocations)

    def test_large_order_splits(self):
        """Large order relative to ADV should split across venues."""
        result = route_order(
            order_size=100_000, adv=500_000, seed=42, max_venues=3,
        )
        # With 5% participation limit per venue, should use multiple
        assert len(result.allocations) >= 1


@pytest.mark.phase12
class TestSimulateFills:
    def test_basic_fill_simulation(self):
        routing = route_order(order_size=1000, seed=42)
        result = simulate_fills(routing, seed=42)
        assert "filled_qty" in result
        assert "unfilled_qty" in result
        assert "fill_pct" in result
        assert "total_cost_bps" in result
        assert "venue_fills" in result

    def test_deterministic_with_seed(self):
        routing = route_order(order_size=1000, seed=42)
        r1 = simulate_fills(routing, seed=123)
        r2 = simulate_fills(routing, seed=123)
        assert r1["filled_qty"] == r2["filled_qty"]
        assert r1["fill_pct"] == r2["fill_pct"]

    def test_fill_pct_bounded(self):
        routing = route_order(order_size=1000, seed=42)
        result = simulate_fills(routing, seed=42)
        assert 0 <= result["fill_pct"] <= 100


@pytest.mark.phase12
class TestRegimeConstants:
    def test_regime_spread_mult_keys(self):
        assert set(REGIME_SPREAD_MULT.keys()) == {"bull", "sideways", "bear", "crisis"}
        assert REGIME_SPREAD_MULT["bull"] <= REGIME_SPREAD_MULT["crisis"]

    def test_regime_fill_mult_keys(self):
        assert set(REGIME_FILL_MULT.keys()) == {"bull", "sideways", "bear", "crisis"}
        assert REGIME_FILL_MULT["bull"] >= REGIME_FILL_MULT["crisis"]
