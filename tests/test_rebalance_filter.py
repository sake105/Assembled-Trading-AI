"""Tests for rebalance filter (small order churn reduction)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.paper.rebalance_filter import filter_small_rebalances

pytestmark = [pytest.mark.unit]


def _make_orders(entries: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    rows = []
    for symbol, side, qty, price in entries:
        rows.append({"timestamp": "2025-10-15", "symbol": symbol, "side": side, "qty": qty, "price": price})
    return pd.DataFrame(rows)


class TestFilterSmallRebalances:
    def test_drops_small_notional(self):
        orders = _make_orders([
            ("AAPL", "BUY", 2, 150.0),
            ("MSFT", "BUY", 100, 300.0),
        ])
        filtered, stats = filter_small_rebalances(orders, min_notional=500.0)
        assert len(filtered) == 1
        assert filtered.iloc[0]["symbol"] == "MSFT"
        assert stats["orders_dropped"] == 1

    def test_keeps_large_orders(self):
        orders = _make_orders([
            ("AAPL", "BUY", 10, 150.0),
            ("MSFT", "SELL", 5, 300.0),
        ])
        filtered, stats = filter_small_rebalances(orders, min_notional=500.0)
        assert len(filtered) == 2
        assert stats["orders_dropped"] == 0

    def test_empty_orders(self):
        orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        filtered, stats = filter_small_rebalances(orders, min_notional=500.0)
        assert len(filtered) == 0
        assert stats["orders_dropped"] == 0

    def test_zero_threshold_keeps_all(self):
        orders = _make_orders([
            ("XYZ", "BUY", 1, 0.01),
        ])
        filtered, stats = filter_small_rebalances(orders, min_notional=0.0)
        assert len(filtered) == 1
        assert stats["orders_dropped"] == 0

    def test_all_below_threshold(self):
        orders = _make_orders([
            ("A", "BUY", 1, 10.0),
            ("B", "SELL", 2, 20.0),
        ])
        filtered, stats = filter_small_rebalances(orders, min_notional=1000.0)
        assert len(filtered) == 0
        assert stats["orders_dropped"] == 2

    def test_stats_fields(self):
        orders = _make_orders([
            ("A", "BUY", 1, 100.0),
            ("B", "BUY", 10, 100.0),
        ])
        _, stats = filter_small_rebalances(orders, min_notional=500.0)
        assert stats["orders_before"] == 2
        assert stats["orders_after"] == 1
        assert stats["orders_dropped"] == 1
        assert stats["min_notional"] == 500.0

    def test_preserves_columns(self):
        orders = _make_orders([("AAPL", "BUY", 100, 150.0)])
        filtered, _ = filter_small_rebalances(orders, min_notional=100.0)
        assert list(filtered.columns) == list(orders.columns)


class TestConfigDefault:
    def test_default_config_filter_disabled(self):
        from src.assembled_core.paper.paper_track import PaperTrackConfig
        from pathlib import Path

        cfg = PaperTrackConfig(
            strategy_name="test",
            strategy_type="trend_baseline",
            universe_file=Path("watchlist.txt"),
            freq="1d",
        )
        assert cfg.rebalance_filter_enabled is False
        assert cfg.rebalance_min_notional == 500.0
