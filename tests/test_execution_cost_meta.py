"""Tests for Part B pre-trade execution cost annotation."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.ops.execution_cost_meta import annotate_execution_cost


def _orders(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "side": "BUY", "qty": 100},
            {"symbol": "MSFT", "side": "BUY", "qty": 200},
            {"symbol": "NVDA", "side": "SELL", "qty": 50},
        ][:n]
    )


def _prices() -> pd.DataFrame:
    # Long-format with close + volume for ADV estimation
    rows = []
    for sym, px, vol in [("AAPL", 180.0, 5_000_000), ("MSFT", 420.0, 4_000_000), ("NVDA", 890.0, 8_000_000)]:
        for i in range(25):
            rows.append({"symbol": sym, "close": px + i * 0.1, "volume": vol, "timestamp": pd.Timestamp("2026-04-01") + pd.Timedelta(days=i)})
    return pd.DataFrame(rows)


def test_disabled_returns_unchanged():
    orders = _orders()
    policy = {"execution": {"cost_meta": {"enabled": False}, "smart_order_router": {"enabled": False}}}
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    assert len(out) == len(orders)
    assert meta["enabled"] is False


def test_empty_orders_returns_empty():
    policy = {"execution": {"cost_meta": {"enabled": True}}}
    out, meta = annotate_execution_cost(pd.DataFrame(), _prices(), policy)
    assert out.empty
    assert meta["per_order"] == []


def test_impact_cost_annotation():
    orders = _orders()
    policy = {"execution": {"cost_meta": {"enabled": True}}}
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    assert len(out) == 3
    assert len(meta["per_order"]) == 3
    assert all("impact_bps" in p for p in meta["per_order"])
    assert meta["total_est_cost_bps"] >= 0


def test_sor_annotation():
    orders = _orders()
    policy = {"execution": {"smart_order_router": {"enabled": True}}}
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    assert len(meta["per_order"]) == 3
    assert all("venues" in p for p in meta["per_order"])
    assert all("sor_cost_bps" in p for p in meta["per_order"])


def test_high_impact_flag_without_enforce():
    """Large order vs tiny ADV → high impact, but enforce=false keeps it."""
    orders = pd.DataFrame([{"symbol": "AAPL", "side": "BUY", "qty": 10_000_000}])
    policy = {
        "execution": {
            "cost_meta": {"enabled": True, "impact_limit_bps": 5.0, "enforce": False},
        }
    }
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    assert len(out) == 1  # not dropped
    assert meta["high_impact_count"] >= 1
    assert meta["dropped_high_impact"] == 0


def test_enforce_drops_high_impact():
    orders = pd.DataFrame([
        {"symbol": "AAPL", "side": "BUY", "qty": 10_000_000},  # huge
        {"symbol": "MSFT", "side": "BUY", "qty": 10},          # tiny
    ])
    policy = {
        "execution": {
            "cost_meta": {"enabled": True, "impact_limit_bps": 5.0, "enforce": True},
        }
    }
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    assert meta["dropped_high_impact"] >= 1
    assert len(out) < 2


def test_missing_price_skipped_gracefully():
    orders = pd.DataFrame([{"symbol": "UNKNOWN", "side": "BUY", "qty": 100}])
    policy = {"execution": {"cost_meta": {"enabled": True}}}
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    # Unknown symbol: no price → skipped from per_order (no crash)
    assert all(p["symbol"] != "UNKNOWN" for p in meta["per_order"])


def test_invalid_qty_skipped():
    orders = pd.DataFrame([
        {"symbol": "AAPL", "side": "BUY", "qty": "not_a_number"},
        {"symbol": "MSFT", "side": "BUY", "qty": 100},
    ])
    policy = {"execution": {"cost_meta": {"enabled": True}}}
    out, meta = annotate_execution_cost(orders, _prices(), policy)
    assert len(meta["per_order"]) == 1
    assert meta["per_order"][0]["symbol"] == "MSFT"
