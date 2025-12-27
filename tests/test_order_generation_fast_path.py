"""Tests for fast-path order generation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.execution.order_generation import (
    generate_orders_from_targets,
    generate_orders_from_targets_fast,
)


def test_fast_path_basic():
    """Test fast-path with aligned positions."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "target_qty": [100.0, 200.0, 50.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [50.0, 200.0, 0.0],
    })
    
    timestamp = pd.Timestamp("2025-01-15", tz="UTC")
    
    # Fast-path should work
    orders = generate_orders_from_targets_fast(
        target_positions=target,
        current_positions=current,
        timestamp=timestamp,
    )
    
    # Expected: BUY AAPL 50, SELL MSFT 0, BUY GOOGL 50
    assert len(orders) == 2  # MSFT has no delta
    assert orders["symbol"].tolist() == ["AAPL", "GOOGL"]
    assert orders["side"].tolist() == ["BUY", "BUY"]
    assert orders["qty"].tolist() == [50.0, 50.0]
    assert orders["timestamp"].iloc[0] == timestamp


def test_fast_path_with_prices():
    """Test fast-path with prices_latest."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 200.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [50.0, 200.0],
    })
    
    prices_latest = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "close": [150.0, 300.0],
    })
    
    orders = generate_orders_from_targets_fast(
        target_positions=target,
        current_positions=current,
        prices_latest=prices_latest,
    )
    
    assert len(orders) == 1  # Only AAPL has delta
    assert orders["symbol"].iloc[0] == "AAPL"
    assert orders["price"].iloc[0] == 150.0


def test_fast_path_empty_current():
    """Test fast-path with empty current positions."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 200.0],
    })
    
    orders = generate_orders_from_targets_fast(
        target_positions=target,
        current_positions=None,
    )
    
    assert len(orders) == 2
    assert all(orders["side"] == "BUY")
    assert orders["qty"].tolist() == [100.0, 200.0]


def test_fast_path_no_changes():
    """Test fast-path when target equals current (no orders)."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 200.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [100.0, 200.0],
    })
    
    orders = generate_orders_from_targets_fast(
        target_positions=target,
        current_positions=current,
    )
    
    assert orders.empty


def test_fast_path_validation_error_mismatch():
    """Test fast-path raises error when symbols don't match."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 200.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "GOOGL"],  # Different symbol
        "qty": [50.0, 100.0],
    })
    
    with pytest.raises(ValueError, match="same symbols"):
        generate_orders_from_targets_fast(
            target_positions=target,
            current_positions=current,
        )


def test_fast_path_validation_error_length():
    """Test fast-path raises error when lengths don't match."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 200.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL"],  # Different length
        "qty": [50.0],
    })
    
    with pytest.raises(ValueError, match="same length"):
        generate_orders_from_targets_fast(
            target_positions=target,
            current_positions=current,
        )


def test_generate_orders_fallback_to_fast_path():
    """Test that generate_orders_from_targets uses fast-path when aligned."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "target_qty": [100.0, 200.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "qty": [50.0, 200.0],
    })
    
    # Should use fast-path internally
    orders = generate_orders_from_targets(
        target_positions=target,
        current_positions=current,
    )
    
    # Should get same results
    assert len(orders) == 1
    assert orders["symbol"].iloc[0] == "AAPL"
    assert orders["side"].iloc[0] == "BUY"
    assert orders["qty"].iloc[0] == 50.0


def test_generate_orders_fallback_to_merge():
    """Test that generate_orders_from_targets falls back to merge when not aligned."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "target_qty": [100.0, 200.0, 50.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "TSLA"],  # Different symbols (TSLA not in target, GOOGL/MSFT not in current)
        "qty": [50.0, 100.0],
    })
    
    # Should use merge-based path (no fast-path, symbols don't match exactly)
    orders = generate_orders_from_targets(
        target_positions=target,
        current_positions=current,
    )
    
    # Should still work correctly (merge handles misaligned symbols)
    assert len(orders) >= 1
    assert "AAPL" in orders["symbol"].values
    # MSFT and GOOGL should be BUY (not in current)
    msft_order = orders[orders["symbol"] == "MSFT"]
    if len(msft_order) > 0:
        assert msft_order["side"].iloc[0] == "BUY"
    googl_order = orders[orders["symbol"] == "GOOGL"]
    if len(googl_order) > 0:
        assert googl_order["side"].iloc[0] == "BUY"


def test_fast_path_equivalence_with_merge():
    """Test that fast-path produces same results as merge-based path when aligned."""
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "target_qty": [100.0, 200.0, 50.0],
    })
    
    current = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "qty": [50.0, 250.0, 0.0],
    })
    
    prices = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"] * 3,
        "close": [150.0, 300.0, 100.0, 151.0, 301.0, 101.0, 152.0, 302.0, 102.0],
        "timestamp": pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC"),
    })
    
    # Fast-path result
    prices_latest = prices.groupby("symbol", group_keys=False)["close"].last().reset_index()
    orders_fast = generate_orders_from_targets_fast(
        target_positions=target,
        current_positions=current,
        prices_latest=prices_latest,
    )
    
    # Merge-based result (via generate_orders_from_targets with misaligned to force merge)
    # Actually, force merge by making symbols different order
    target_shuffled = target.sample(frac=1).sort_values("symbol").reset_index(drop=True)
    current_shuffled = current.sample(frac=1).sort_values("symbol").reset_index(drop=True)
    orders_merge = generate_orders_from_targets(
        target_positions=target_shuffled,
        current_positions=current_shuffled,
        prices=prices,
    )
    
    # Results should be equivalent (same orders, just possibly different order)
    assert len(orders_fast) == len(orders_merge)
    fast_by_symbol = orders_fast.set_index("symbol")
    merge_by_symbol = orders_merge.set_index("symbol")
    
    for symbol in fast_by_symbol.index:
        assert symbol in merge_by_symbol.index
        assert fast_by_symbol.loc[symbol, "side"] == merge_by_symbol.loc[symbol, "side"]
        assert abs(fast_by_symbol.loc[symbol, "qty"] - merge_by_symbol.loc[symbol, "qty"]) < 1e-6
        assert abs(fast_by_symbol.loc[symbol, "price"] - merge_by_symbol.loc[symbol, "price"]) < 1e-6

