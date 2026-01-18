# tests/test_fill_model_limit_orders.py
"""Tests for limit order support (Sprint 7 / C4).

Tests verify:
1. Buy limit reachable/unreachable
2. Sell limit reachable/unreachable
3. OHLC missing fallback behavior
4. Combined with partial cap (limit reachable but partial fill)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.fill_model import (
    PartialFillModel,
    apply_limit_order_fills,
)


def test_buy_limit_reachable() -> None:
    """Test that BUY limit orders are filled when limit is reachable."""
    # Create BUY limit order with limit_price = 150
    # Bar low = 145 <= 150 -> should be filled
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
        "order_type": ["limit"],
        "limit_price": [150.0],
    })

    # Create prices with OHLC
    prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "open": [148.0],
        "high": [152.0],
        "low": [145.0],  # Low <= limit_price -> fill
        "close": [150.0],
    })

    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be filled
    assert fills["fill_qty"].iloc[0] == 100.0, "BUY limit should be filled when low <= limit"
    assert fills["status"].iloc[0] == "filled", "Status should be 'filled'"
    assert fills["fill_price"].iloc[0] == 150.0, "Fill price should be limit_price"


def test_buy_limit_unreachable() -> None:
    """Test that BUY limit orders are rejected when limit is not reachable."""
    # Create BUY limit order with limit_price = 150
    # Bar low = 155 > 150 -> should be rejected
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
        "order_type": ["limit"],
        "limit_price": [150.0],
    })

    # Create prices with OHLC
    prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "open": [148.0],
        "high": [160.0],
        "low": [155.0],  # Low > limit_price -> reject
        "close": [158.0],
    })

    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be rejected
    assert fills["fill_qty"].iloc[0] == 0.0, "BUY limit should be rejected when low > limit"
    assert fills["status"].iloc[0] == "rejected", "Status should be 'rejected'"
    assert fills["remaining_qty"].iloc[0] == 100.0, "Remaining qty should equal order qty"


def test_sell_limit_reachable() -> None:
    """Test that SELL limit orders are filled when limit is reachable."""
    # Create SELL limit order with limit_price = 150
    # Bar high = 155 >= 150 -> should be filled
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [100.0],
        "price": [150.0],
        "order_type": ["limit"],
        "limit_price": [150.0],
    })

    # Create prices with OHLC
    prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "open": [148.0],
        "high": [155.0],  # High >= limit_price -> fill
        "low": [145.0],
        "close": [150.0],
    })

    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be filled
    assert fills["fill_qty"].iloc[0] == 100.0, "SELL limit should be filled when high >= limit"
    assert fills["status"].iloc[0] == "filled", "Status should be 'filled'"
    assert fills["fill_price"].iloc[0] == 150.0, "Fill price should be limit_price"


def test_sell_limit_unreachable() -> None:
    """Test that SELL limit orders are rejected when limit is not reachable."""
    # Create SELL limit order with limit_price = 150
    # Bar high = 145 < 150 -> should be rejected
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["SELL"],
        "qty": [100.0],
        "price": [150.0],
        "order_type": ["limit"],
        "limit_price": [150.0],
    })

    # Create prices with OHLC
    prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "open": [148.0],
        "high": [145.0],  # High < limit_price -> reject
        "low": [140.0],
        "close": [143.0],
    })

    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be rejected
    assert fills["fill_qty"].iloc[0] == 0.0, "SELL limit should be rejected when high < limit"
    assert fills["status"].iloc[0] == "rejected", "Status should be 'rejected'"
    assert fills["remaining_qty"].iloc[0] == 100.0, "Remaining qty should equal order qty"


def test_ohlc_missing_fallback() -> None:
    """Test that missing OHLC uses close as fallback (deterministic)."""
    # Create BUY limit order
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
        "order_type": ["limit"],
        "limit_price": [150.0],
    })

    # Create prices with only close (no high/low)
    prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "close": [150.0],  # close == limit_price -> should fill (using close as low)
    })

    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be filled (close used as both high and low)
    assert fills["fill_qty"].iloc[0] == 100.0, "Should fill when close == limit_price (fallback)"
    assert fills["status"].iloc[0] == "filled", "Status should be 'filled'"


def test_limit_with_partial_cap() -> None:
    """Test that limit orders can be partially filled when combined with partial cap."""
    # Create BUY limit order with large qty
    # Use a timestamp after the rolling window (so ADV is available)
    trade_timestamp = pd.Timestamp("2024-01-25", tz="UTC")  # After 20+ days
    trades = pd.DataFrame({
        "timestamp": [trade_timestamp],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100000.0],  # Large order
        "price": [150.0],
        "order_type": ["limit"],
        "limit_price": [150.0],
    })

    # Create prices with OHLC (limit reachable)
    # Need at least 20 days before trade_timestamp for ADV calculation
    timestamps = pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC")
    prices_list = []
    for ts in timestamps:
        prices_list.append({
            "timestamp": ts,
            "symbol": "AAPL",
            "open": 148.0,
            "high": 152.0,
            "low": 145.0,  # Low <= limit_price -> eligible
            "close": 150.0,
            "volume": 1e6,  # 1M shares per day
        })
    prices = pd.DataFrame(prices_list)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Apply with partial fill model (participation_cap = 0.05 = 5% of ADV)
    partial_model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)
    
    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=partial_model)

    # Limit is reachable, but ADV cap may limit fill_qty
    # ADV ~ 1e8 USD (1M shares * 150), participation_cap = 0.05 -> max_notional = 5e6
    # max_qty = 5e6 / 150 = 33333.33
    # So fill_qty should be ~33333 (partial fill)
    assert fills["fill_qty"].iloc[0] > 0.0, "Should be filled (limit reachable)"
    assert fills["fill_qty"].iloc[0] < 100000.0, "Should be partial fill (ADV cap)"
    assert fills["status"].iloc[0] == "partial", "Status should be 'partial'"
    assert fills["fill_price"].iloc[0] == 150.0, "Fill price should be limit_price"


def test_market_order_ignores_limit() -> None:
    """Test that market orders ignore limit_price (if present)."""
    # Create market order (order_type="market")
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
        "order_type": ["market"],
        "limit_price": [140.0],  # Limit price present but ignored
    })

    # Create prices
    prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "close": [150.0],
    })

    fills = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be filled (market order, limit ignored)
    assert fills["fill_qty"].iloc[0] == 100.0, "Market order should be filled"
    assert fills["status"].iloc[0] == "filled", "Status should be 'filled'"


def test_limit_order_deterministic() -> None:
    """Test that limit order fills are deterministic."""
    # Create limit orders
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 75.0],
        "price": [150.0, 200.0, 100.0],
        "order_type": ["limit", "limit", "market"],
        "limit_price": [150.0, 200.0, np.nan],
    })

    # Create prices
    timestamps = pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC")
    prices_list = []
    for i, ts in enumerate(timestamps):
        prices_list.append({
            "timestamp": ts,
            "symbol": "AAPL",
            "open": 148.0,
            "high": 152.0,
            "low": 145.0,
            "close": 150.0,
        })
        prices_list.append({
            "timestamp": ts,
            "symbol": "MSFT",
            "open": 198.0,
            "high": 205.0,
            "low": 195.0,
            "close": 200.0,
        })
        prices_list.append({
            "timestamp": ts,
            "symbol": "GOOGL",
            "open": 98.0,
            "high": 102.0,
            "low": 95.0,
            "close": 100.0,
        })
    prices = pd.DataFrame(prices_list)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Apply limit order fills twice
    fills1 = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)
    fills2 = apply_limit_order_fills(trades, prices=prices, partial_fill_model=None)

    # Should be identical
    pd.testing.assert_frame_equal(
        fills1[["timestamp", "symbol", "qty", "fill_qty", "status"]],
        fills2[["timestamp", "symbol", "qty", "fill_qty", "status"]],
        "Limit order fills should be deterministic",
    )
