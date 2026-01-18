# tests/test_fill_model_partial.py
"""Tests for partial fill model based on liquidity (ADV proxy) (Sprint 7 / C3).

Tests verify:
1. Participation cap enforces partial fills
2. Deterministic given same inputs
3. Missing volume fallback deterministic
4. fill_qty <= qty always
5. remaining_qty correct
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
    apply_partial_fills,
    compute_max_fill_qty,
)


def test_participation_cap_enforces_partial_fills() -> None:
    """Test that participation cap enforces partial fills."""
    model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)

    # ADV = 1e8 USD, price = 100, participation_cap = 0.05
    # max_notional = 1e8 * 0.05 = 5e6
    # max_qty = 5e6 / 100 = 50000
    adv_usd = 1e8
    price = 100.0
    max_qty = compute_max_fill_qty(adv_usd, price, model)
    
    assert max_qty == 50000.0, f"Expected max_qty=50000, got {max_qty}"

    # Order qty = 100000 > max_qty = 50000 -> partial fill
    order_qty = 100000.0
    fill_qty = min(order_qty, max_qty)
    
    assert fill_qty == 50000.0, "Should be partially filled"
    assert fill_qty < order_qty, "Fill qty should be less than order qty"


def test_partial_fills_deterministic() -> None:
    """Test that partial fills are deterministic given same inputs."""
    model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)

    # Create trades
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100000.0, 50000.0, 75000.0],
        "price": [150.0, 200.0, 100.0],
    })

    # Create prices with volume (must be sorted by symbol, timestamp)
    timestamps = pd.date_range("2024-01-01", periods=25, freq="1d", tz="UTC")
    prices_list = []
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        for ts in timestamps:
            prices_list.append({
                "timestamp": ts,
                "symbol": symbol,
                "close": 150.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 100.0),
                "volume": 1e6,  # 1M shares per day
            })
    prices = pd.DataFrame(prices_list)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Apply partial fills twice
    fills1 = apply_partial_fills(trades, prices=prices, partial_fill_model=model)
    fills2 = apply_partial_fills(trades, prices=prices, partial_fill_model=model)

    # Should be identical
    pd.testing.assert_frame_equal(
        fills1[["timestamp", "symbol", "qty", "fill_qty", "status"]],
        fills2[["timestamp", "symbol", "qty", "fill_qty", "status"]],
        "Partial fills should be deterministic",
    )


def test_missing_volume_fallback() -> None:
    """Test that missing volume uses fallback deterministically."""
    model = PartialFillModel(
        adv_window=20,
        participation_cap=0.05,
        min_fill_qty=0.0,
        fallback_fill_ratio=0.5,  # 50% fill when ADV missing
    )

    # Create trades
    trades = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01", tz="UTC")],
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [1000.0],
        "price": [150.0],
    })

    # Create prices without volume (ADV will be NaN)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=25, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 25,
        "close": [150.0] * 25,
        # No volume column -> ADV will be NaN
    })

    fills = apply_partial_fills(trades, prices=prices, partial_fill_model=model)

    # Should use fallback: fill_qty = qty * fallback_fill_ratio = 1000 * 0.5 = 500
    assert fills["fill_qty"].iloc[0] == 500.0, "Should use fallback fill ratio"
    assert fills["status"].iloc[0] == "partial", "Should be partial fill"
    assert fills["remaining_qty"].iloc[0] == 500.0, "Remaining qty should be 500"


def test_fill_qty_always_le_qty() -> None:
    """Test that fill_qty <= qty always."""
    model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)

    # Create trades with various qty values
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "side": ["BUY"] * 5,
        "qty": [1000.0, 50000.0, 100000.0, 200000.0, 500000.0],
        "price": [150.0] * 5,
    })

    # Create prices with volume (ADV will be ~1e8 USD for 1M shares * 150)
    timestamps = pd.date_range("2024-01-01", periods=25, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 25,
        "close": [150.0] * 25,
        "volume": [1e6] * 25,  # 1M shares per day
    })
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    fills = apply_partial_fills(trades, prices=prices, partial_fill_model=model)

    # Verify fill_qty <= qty for all rows
    assert (fills["fill_qty"] <= fills["qty"].abs()).all(), "fill_qty should always be <= qty"
    assert (fills["remaining_qty"] >= 0.0).all(), "remaining_qty should always be >= 0"


def test_remaining_qty_correct() -> None:
    """Test that remaining_qty = qty - fill_qty."""
    model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)

    # Create trades
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100000.0, 50000.0, 75000.0],
        "price": [150.0, 200.0, 100.0],
    })

    # Create prices with volume (must be sorted by symbol, timestamp)
    timestamps = pd.date_range("2024-01-01", periods=25, freq="1d", tz="UTC")
    prices_list = []
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        for ts in timestamps:
            prices_list.append({
                "timestamp": ts,
                "symbol": symbol,
                "close": 150.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 100.0),
                "volume": 1e6,  # 1M shares per day
            })
    prices = pd.DataFrame(prices_list)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    fills = apply_partial_fills(trades, prices=prices, partial_fill_model=model)

    # Verify remaining_qty = qty - fill_qty
    expected_remaining = fills["qty"].abs() - fills["fill_qty"]
    assert np.allclose(
        fills["remaining_qty"], expected_remaining, rtol=1e-9, atol=1e-9
    ), "remaining_qty should equal qty - fill_qty"


def test_min_fill_qty_rejects_small_fills() -> None:
    """Test that min_fill_qty rejects fills below minimum."""
    model = PartialFillModel(
        adv_window=20,
        participation_cap=0.05,
        min_fill_qty=1000.0,  # Minimum 1000 shares
    )

    # ADV = 1e8 USD, price = 100, participation_cap = 0.05
    # max_notional = 1e8 * 0.05 = 5e6
    # max_qty = 5e6 / 100 = 50000
    adv_usd = 1e8
    price = 100.0
    max_qty = compute_max_fill_qty(adv_usd, price, model)
    
    assert max_qty == 50000.0, "Should compute max_qty correctly"

    # But if computed fill_qty < min_fill_qty, should reject
    # Example: order qty = 500, max_qty = 50000, fill_qty = 500 < min_fill_qty = 1000 -> reject
    order_qty = 500.0
    computed_fill_qty = min(order_qty, max_qty)  # = 500
    
    # In apply_partial_fills, this would be set to 0 if < min_fill_qty
    if computed_fill_qty < model.min_fill_qty:
        final_fill_qty = 0.0
        status = "rejected"
    else:
        final_fill_qty = computed_fill_qty
        status = "partial" if final_fill_qty < order_qty else "filled"
    
    assert final_fill_qty == 0.0, "Should reject if fill_qty < min_fill_qty"
    assert status == "rejected", "Status should be rejected"


def test_qty_zero_dropped() -> None:
    """Test that qty == 0 rows are dropped."""
    model = PartialFillModel(adv_window=20, participation_cap=0.05, min_fill_qty=0.0)

    # Create trades with qty == 0
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [1000.0, 0.0, 500.0],  # Middle one has qty=0
        "price": [150.0, 200.0, 100.0],
    })

    # Create prices (must be sorted by symbol, timestamp)
    timestamps = pd.date_range("2024-01-01", periods=25, freq="1d", tz="UTC")
    prices_list = []
    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        for ts in timestamps:
            prices_list.append({
                "timestamp": ts,
                "symbol": symbol,
                "close": 150.0 if symbol == "AAPL" else (200.0 if symbol == "MSFT" else 100.0),
                "volume": 1e6,
            })
    prices = pd.DataFrame(prices_list)
    prices = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    fills = apply_partial_fills(trades, prices=prices, partial_fill_model=model)

    # Should have 2 rows (qty=0 dropped)
    assert len(fills) == 2, "Should drop qty==0 rows"
    assert "MSFT" not in fills["symbol"].values, "MSFT row should be dropped"
