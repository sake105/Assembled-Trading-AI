"""Tests for Implementation Shortfall (Sprint 2 / C11)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.tca import compute_implementation_shortfall


def test_empty_fills_returns_empty_with_schema() -> None:
    fills = pd.DataFrame(columns=["symbol", "side", "fill_price", "fill_qty"])
    out = compute_implementation_shortfall(fills)
    assert out.empty
    assert "arrival_price" in out.columns
    assert "is_bps" in out.columns


def test_buy_fill_above_arrival_is_positive_cost() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 101.0, "arrival_price": 100.0}]
    )
    out = compute_implementation_shortfall(fills)
    # (101 - 100) / 100 * 10000 = 100 bps
    assert abs(out.iloc[0]["is_bps"] - 100.0) < 1e-9


def test_sell_fill_below_arrival_is_positive_cost() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "SELL", "fill_price": 99.0, "arrival_price": 100.0}]
    )
    out = compute_implementation_shortfall(fills)
    # sign=-1, (99-100)/100 = -0.01, * -1 * 10000 = 100
    assert abs(out.iloc[0]["is_bps"] - 100.0) < 1e-9


def test_buy_fill_below_arrival_is_negative_cost() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 99.0, "arrival_price": 100.0}]
    )
    out = compute_implementation_shortfall(fills)
    assert abs(out.iloc[0]["is_bps"] - (-100.0)) < 1e-9


def test_missing_arrival_price_yields_zero() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 101.0, "arrival_price": np.nan}]
    )
    out = compute_implementation_shortfall(fills)
    assert out.iloc[0]["is_bps"] == 0.0


def test_zero_arrival_price_yields_zero() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "fill_price": 101.0, "arrival_price": 0.0}]
    )
    out = compute_implementation_shortfall(fills)
    assert out.iloc[0]["is_bps"] == 0.0


def test_merge_arrival_prices_by_symbol() -> None:
    fills = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "AAA", "side": "BUY", "fill_price": 101.0},
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "BBB", "side": "SELL", "fill_price": 49.5},
        ]
    )
    arrival = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "AAA", "arrival_price": 100.0},
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "BBB", "arrival_price": 50.0},
        ]
    )
    out = compute_implementation_shortfall(fills, arrival)
    row_a = out[out["symbol"] == "AAA"].iloc[0]
    row_b = out[out["symbol"] == "BBB"].iloc[0]
    assert abs(row_a["is_bps"] - 100.0) < 1e-9
    # (49.5 - 50)/50 * -1 * 10000 = 100
    assert abs(row_b["is_bps"] - 100.0) < 1e-9


def test_price_column_used_when_fill_price_missing() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "price": 101.0, "arrival_price": 100.0}]
    )
    out = compute_implementation_shortfall(fills)
    assert abs(out.iloc[0]["is_bps"] - 100.0) < 1e-9


def test_no_side_defaults_to_buy_sign() -> None:
    fills = pd.DataFrame(
        [{"symbol": "AAA", "fill_price": 101.0, "arrival_price": 100.0}]
    )
    out = compute_implementation_shortfall(fills)
    assert out.iloc[0]["is_bps"] > 0  # positive = BUY above arrival
