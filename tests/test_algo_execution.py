"""Tests for algorithmic execution module (TWAP/VWAP)."""

from __future__ import annotations

from datetime import datetime

import pytest
import pandas as pd

from src.assembled_core.execution.algo_execution import (
    SlicedOrder,
    TWAPScheduler,
    VWAPScheduler,
    ImplementationShortfallModel,
    compute_implementation_shortfall,
)

_START = datetime(2024, 6, 3, 9, 30)
_END = datetime(2024, 6, 3, 16, 0)


@pytest.mark.phase12
class TestSlicedOrder:
    def test_creation(self):
        order = SlicedOrder(
            symbol="AAPL", side="BUY", quantity=100.0,
            scheduled_time=_START, slice_idx=0, total_slices=10,
        )
        assert order.symbol == "AAPL"
        assert order.quantity == 100.0

    def test_to_dict(self):
        order = SlicedOrder(
            symbol="MSFT", side="SELL", quantity=50.0,
            scheduled_time=_START, slice_idx=3, total_slices=10,
        )
        d = order.to_dict()
        assert d["symbol"] == "MSFT"
        assert d["side"] == "SELL"


@pytest.mark.phase12
class TestTWAPScheduler:
    def test_basic_schedule(self):
        scheduler = TWAPScheduler(n_slices=5, randomize=False)
        slices = scheduler.schedule(
            symbol="AAPL", total_qty=1000, side="BUY",
            start_time=_START, end_time=_END,
        )
        assert len(slices) == 5
        total = sum(s.quantity for s in slices)
        assert total == pytest.approx(1000, abs=1)

    def test_randomized_schedule(self):
        scheduler = TWAPScheduler(n_slices=10, randomize=True)
        slices = scheduler.schedule(
            symbol="GOOG", total_qty=500, side="SELL",
            start_time=_START, end_time=_END, random_seed=42,
        )
        assert len(slices) == 10
        total = sum(s.quantity for s in slices)
        assert total == pytest.approx(500, abs=1)

    def test_single_slice(self):
        scheduler = TWAPScheduler(n_slices=1, randomize=False)
        slices = scheduler.schedule("AAPL", 100, "BUY", _START, _END)
        assert len(slices) == 1
        assert slices[0].quantity == pytest.approx(100, abs=1)


@pytest.mark.phase12
class TestVWAPScheduler:
    def test_basic_schedule(self):
        scheduler = VWAPScheduler(n_slices=5)
        volume_profile = pd.DataFrame({
            "time_bucket": ["09:30", "10:00", "10:30", "11:00", "11:30"],
            "volume_fraction": [0.15, 0.20, 0.25, 0.20, 0.20],
        })
        slices = scheduler.schedule(
            symbol="AAPL", total_qty=1000, side="BUY",
            start_time=_START, end_time=_END,
            volume_profile=volume_profile,
        )
        assert len(slices) > 0
        total = sum(s.quantity for s in slices)
        assert total == pytest.approx(1000, abs=10)

    def test_no_profile_fallback(self):
        scheduler = VWAPScheduler(n_slices=5)
        slices = scheduler.schedule("AAPL", 1000, "BUY", _START, _END)
        assert len(slices) > 0


@pytest.mark.phase12
class TestImplementationShortfall:
    def test_estimate_cost(self):
        model = ImplementationShortfallModel()
        cost = model.estimate_cost(
            quantity=1000, adv=1000000,
            daily_vol=0.02, price=150.0,
        )
        assert isinstance(cost, dict)
        assert "total_cost_bps" in cost
        assert cost["total_cost_bps"] > 0

    def test_larger_order_higher_cost(self):
        model = ImplementationShortfallModel()
        cost_small = model.estimate_cost(100, 1000000, 0.02, 150.0)
        cost_large = model.estimate_cost(50000, 1000000, 0.02, 150.0)
        assert cost_large["total_cost_bps"] > cost_small["total_cost_bps"]

    def test_compute_shortfall(self):
        result = compute_implementation_shortfall(
            decision_price=100.0, avg_fill_price=100.05,
            side="BUY",
        )
        assert result > 0  # bought above arrival = positive shortfall
