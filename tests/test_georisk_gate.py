"""Tests for GeoRisk gate (exposure scaling based on news_geo)."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.paper.georisk_gate import (
    compute_georisk_multiplier,
    apply_georisk_to_orders,
)

pytestmark = [pytest.mark.unit]


class TestComputeGeoRiskMultiplier:
    def test_watch_returns_1(self):
        assert compute_georisk_multiplier({"state_hint": "WATCH"}) == 1.0

    def test_active_returns_configured(self):
        assert (
            compute_georisk_multiplier({"state_hint": "ACTIVE"}, active_multiplier=0.70)
            == 0.70
        )

    def test_active_custom_multiplier(self):
        assert (
            compute_georisk_multiplier({"state_hint": "ACTIVE"}, active_multiplier=0.50)
            == 0.50
        )

    def test_none_geo_returns_1(self):
        assert compute_georisk_multiplier(None) == 1.0

    def test_empty_dict_returns_1(self):
        assert compute_georisk_multiplier({}) == 1.0

    def test_missing_state_hint_returns_1(self):
        assert compute_georisk_multiplier({"geo_score": 2}) == 1.0

    def test_clamp_below_zero(self):
        assert (
            compute_georisk_multiplier({"state_hint": "ACTIVE"}, active_multiplier=-0.5)
            == 0.0
        )

    def test_clamp_above_one(self):
        assert (
            compute_georisk_multiplier({"state_hint": "ACTIVE"}, active_multiplier=1.5)
            == 1.0
        )


class TestApplyGeoRiskToOrders:
    def _make_orders(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": ["2025-01-15", "2025-01-15"],
                "symbol": ["AAPL", "MSFT"],
                "side": ["BUY", "BUY"],
                "qty": [100, 200],
                "price": [150.0, 300.0],
            }
        )

    def test_multiplier_1_no_change(self):
        orders = self._make_orders()
        result = apply_georisk_to_orders(orders, 1.0)
        assert list(result["qty"]) == [100, 200]

    def test_multiplier_half_scales(self):
        orders = self._make_orders()
        result = apply_georisk_to_orders(orders, 0.5)
        assert list(result["qty"]) == [50, 100]

    def test_multiplier_070(self):
        orders = self._make_orders()
        result = apply_georisk_to_orders(orders, 0.70)
        assert list(result["qty"]) == [70, 140]

    def test_small_qty_dropped(self):
        orders = pd.DataFrame(
            {
                "timestamp": ["2025-01-15"],
                "symbol": ["XYZ"],
                "side": ["BUY"],
                "qty": [1],
                "price": [10.0],
            }
        )
        result = apply_georisk_to_orders(orders, 0.3)
        assert len(result) == 0

    def test_empty_orders(self):
        orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
        result = apply_georisk_to_orders(orders, 0.5)
        assert len(result) == 0

    def test_no_sign_change(self):
        orders = pd.DataFrame(
            {
                "timestamp": ["2025-01-15", "2025-01-15"],
                "symbol": ["A", "B"],
                "side": ["BUY", "SELL"],
                "qty": [100, 100],
                "price": [50.0, 60.0],
            }
        )
        result = apply_georisk_to_orders(orders, 0.5)
        assert all(q > 0 for q in result["qty"])

    def test_preserves_columns(self):
        orders = self._make_orders()
        result = apply_georisk_to_orders(orders, 0.7)
        assert list(result.columns) == list(orders.columns)


class TestGeoRiskGateConfig:
    def test_default_config_has_gate_disabled(self):
        from src.assembled_core.paper.paper_track import PaperTrackConfig
        from pathlib import Path

        cfg = PaperTrackConfig(
            strategy_name="test",
            strategy_type="trend_baseline",
            universe_file=Path("watchlist.txt"),
            freq="1d",
        )
        assert cfg.georisk_gate_enabled is False
        assert cfg.georisk_active_multiplier == 0.70
        assert cfg.intel_mode == "none"
