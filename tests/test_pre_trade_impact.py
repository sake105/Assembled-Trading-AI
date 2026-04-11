"""Tests for pre-trade impact wiring (Sprint 2 / C10)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.trading_cycle import _apply_pre_trade_impact


def _mk_prices(sym: str = "AAA", n: int = 80, vol: float = 1e6) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": [sym] * n,
            "close": closes,
            "volume": [vol] * n,
        }
    )


def _mk_orders(sym: str = "AAA", qty: float = 100.0, price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2025-03-01", tz="UTC"),
                "symbol": sym,
                "side": "BUY",
                "qty": qty,
                "price": price,
            }
        ]
    )


def test_small_order_low_impact_no_scale() -> None:
    prices = _mk_prices(vol=1_000_000.0)  # ADV ~1M shares
    orders = _mk_orders(qty=100.0)  # 0.01% of ADV
    cfg = {"kyle_lambda": 0.1, "max_total_cost_bps": 50.0}
    new_orders, meta = _apply_pre_trade_impact(orders, prices, cfg)
    assert new_orders.iloc[0]["qty"] == 100.0  # not scaled
    assert new_orders.iloc[0]["expected_impact_bps"] < 50.0
    assert meta["n_orders"] == 1
    assert meta["scaled_symbols"] == []


def test_large_order_triggers_scale() -> None:
    prices = _mk_prices(vol=1_000.0)  # tiny ADV
    orders = _mk_orders(qty=10_000.0)  # 10x ADV — massive impact
    cfg = {"kyle_lambda": 0.5, "max_total_cost_bps": 10.0}
    new_orders, meta = _apply_pre_trade_impact(orders, prices, cfg)
    assert new_orders.iloc[0]["qty"] < 10_000.0
    assert "AAA" in meta["scaled_symbols"]
    assert new_orders.iloc[0]["expected_impact_bps"] > 10.0


def test_missing_adv_falls_back_to_opportunity_cost() -> None:
    prices = pd.DataFrame(columns=["timestamp", "symbol", "close", "volume"])
    orders = _mk_orders(qty=100.0)
    cfg = {"opportunity_cost_bps": 5.0, "max_total_cost_bps": 50.0}
    new_orders, meta = _apply_pre_trade_impact(orders, prices, cfg)
    # With adv=0 the model returns just opportunity_cost_bps
    assert new_orders.iloc[0]["expected_impact_bps"] == 5.0
    assert meta["scaled_symbols"] == []


def test_negative_qty_short_uses_abs_for_impact() -> None:
    prices = _mk_prices(vol=1_000.0)
    orders = _mk_orders(qty=-10_000.0)
    cfg = {"kyle_lambda": 0.5, "max_total_cost_bps": 10.0}
    new_orders, meta = _apply_pre_trade_impact(orders, prices, cfg)
    # Sign preserved but magnitude scaled
    assert new_orders.iloc[0]["qty"] < 0
    assert abs(new_orders.iloc[0]["qty"]) < 10_000.0
    assert "AAA" in meta["scaled_symbols"]


def test_empty_orders_returns_empty() -> None:
    prices = _mk_prices()
    orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    new_orders, meta = _apply_pre_trade_impact(orders, prices, {})
    assert new_orders.empty
    assert meta["n_orders"] == 0
    assert meta["avg_bps"] == 0.0


def test_meta_contains_aggregate_stats() -> None:
    prices = pd.concat(
        [_mk_prices("AAA", vol=1_000_000.0), _mk_prices("BBB", vol=500_000.0)],
        ignore_index=True,
    )
    orders = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "AAA", "side": "BUY", "qty": 50.0, "price": 100.0},
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "BBB", "side": "BUY", "qty": 50.0, "price": 100.0},
        ]
    )
    new_orders, meta = _apply_pre_trade_impact(orders, prices, {"max_total_cost_bps": 500.0})
    assert meta["n_orders"] == 2
    assert "avg_bps" in meta and "max_bps" in meta
    assert meta["max_bps"] >= meta["avg_bps"]
    assert "expected_impact_bps" in new_orders.columns
