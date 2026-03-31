"""Tests for dead-zone rebalance filter."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.paper.deadzone_rebalance import filter_deadzone_orders

pytestmark = [pytest.mark.unit]


def _make_orders(entries: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"timestamp": "2025-10-15", "symbol": s, "side": sd, "qty": q, "price": p}
        for s, sd, q, p in entries
    ])


def _make_positions(entries: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": s, "qty": q} for s, q in entries])


class TestFilterDeadzoneOrders:
    def test_small_delta_dropped(self):
        """Order qty 2 vs current 100 = 2% < 5% threshold → drop."""
        orders = _make_orders([("AAPL", "BUY", 2, 150.0)])
        positions = _make_positions([("AAPL", 100)])
        filtered, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
        assert len(filtered) == 0
        assert stats["orders_dropped"] == 1

    def test_large_delta_kept(self):
        """Order qty 20 vs current 100 = 20% > 5% → keep."""
        orders = _make_orders([("AAPL", "BUY", 20, 150.0)])
        positions = _make_positions([("AAPL", 100)])
        filtered, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
        assert len(filtered) == 1
        assert stats["orders_dropped"] == 0

    def test_new_position_always_passes(self):
        """Symbol not in current positions → always keep."""
        orders = _make_orders([("NVDA", "BUY", 1, 300.0)])
        positions = _make_positions([("AAPL", 100)])
        filtered, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
        assert len(filtered) == 1

    def test_no_positions_passes_all(self):
        """No current positions → all orders pass."""
        orders = _make_orders([("AAPL", "BUY", 1, 150.0)])
        filtered, stats = filter_deadzone_orders(orders, None, deadzone_pct=0.05)
        assert len(filtered) == 1

    def test_empty_orders(self):
        orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        filtered, stats = filter_deadzone_orders(orders, None, deadzone_pct=0.05)
        assert len(filtered) == 0
        assert stats["orders_dropped"] == 0

    def test_zero_threshold_keeps_all(self):
        orders = _make_orders([("AAPL", "BUY", 1, 150.0)])
        positions = _make_positions([("AAPL", 1000)])
        filtered, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.0)
        assert len(filtered) == 1

    def test_mixed_small_and_large(self):
        """2 orders: one small (drop), one large (keep)."""
        orders = _make_orders([
            ("AAPL", "SELL", 3, 150.0),
            ("MSFT", "BUY", 50, 300.0),
        ])
        positions = _make_positions([("AAPL", 100), ("MSFT", 100)])
        filtered, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
        assert len(filtered) == 1
        assert filtered.iloc[0]["symbol"] == "MSFT"
        assert stats["orders_dropped"] == 1

    def test_boundary_exactly_at_threshold(self):
        """Order qty 5 vs current 100 = 5% = threshold → keep (>=)."""
        orders = _make_orders([("AAPL", "BUY", 5, 150.0)])
        positions = _make_positions([("AAPL", 100)])
        filtered, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
        assert len(filtered) == 1

    def test_stats_fields_complete(self):
        orders = _make_orders([("A", "BUY", 1, 10.0)])
        positions = _make_positions([("A", 100)])
        _, stats = filter_deadzone_orders(orders, positions, deadzone_pct=0.05)
        assert "orders_before" in stats
        assert "orders_after" in stats
        assert "orders_dropped" in stats
        assert "deadzone_pct" in stats


class TestConfigDefault:
    def test_default_deadzone_disabled(self):
        from src.assembled_core.paper.paper_track import PaperTrackConfig
        from pathlib import Path

        cfg = PaperTrackConfig(
            strategy_name="test",
            strategy_type="trend_baseline",
            universe_file=Path("watchlist.txt"),
            freq="1d",
        )
        assert cfg.deadzone_enabled is False
        assert cfg.deadzone_pct == 0.05
