# tests/test_fill_model_schema.py
"""Tests for fill model schema and contract (Sprint 7 / C1).

Tests verify:
1. Required columns exist
2. UTC policy
3. Deterministic ordering
4. rejected/partial rules valid
5. No NaNs in key columns
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.fill_model import (
    create_partial_fill_from_order,
    create_rejected_fill_from_order,
    ensure_fill_schema,
)


def test_fill_schema_required_columns() -> None:
    """Test that required columns exist after ensure_fill_schema."""
    # Create minimal trades DataFrame (orders without fill columns)
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 75.0],
        "price": [150.0, 200.0, 100.0],
    })
    
    # Ensure fill schema (should add fill_qty, fill_price, status, remaining_qty)
    fills = ensure_fill_schema(trades, default_full_fill=True)
    
    # Verify required columns exist
    required_cols = [
        "timestamp", "symbol", "side", "qty", "price",
        "fill_qty", "fill_price", "status", "remaining_qty",
    ]
    for col in required_cols:
        assert col in fills.columns, f"Required column {col} should exist"
    
    # Verify full fills (default)
    assert (fills["fill_qty"] == fills["qty"]).all(), "Default should be full fills"
    assert (fills["fill_price"] == fills["price"]).all(), "Default fill_price should equal price"
    assert (fills["status"] == "filled").all(), "Default status should be 'filled'"
    assert (fills["remaining_qty"] == 0.0).all(), "Default remaining_qty should be 0"


def test_fill_schema_utc_policy() -> None:
    """Test that timestamps are UTC-aware."""
    # Create trades with UTC timestamps
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })
    
    fills = ensure_fill_schema(trades, default_full_fill=True)
    
    # Verify UTC-aware timestamps
    assert pd.api.types.is_datetime64_any_dtype(fills["timestamp"]), "timestamp should be datetime"
    assert all(
        hasattr(ts, "tz") and (str(ts.tz) == "UTC" or (hasattr(ts.tz, "zone") and ts.tz.zone == "UTC"))
        for ts in fills["timestamp"]
    ), "All timestamps should be UTC-aware"


def test_fill_schema_deterministic_ordering() -> None:
    """Test that fills are sorted deterministically (timestamp, symbol)."""
    # Create trades in random order
    trades = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-03", tz="UTC"),
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-02", tz="UTC"),
        ],
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "side": ["SELL", "BUY", "BUY"],
        "qty": [50.0, 100.0, 75.0],
        "price": [200.0, 150.0, 100.0],
    })
    
    fills = ensure_fill_schema(trades, default_full_fill=True)
    
    # Verify sorted by timestamp, then symbol
    assert fills["timestamp"].is_monotonic_increasing, "Should be sorted by timestamp"
    # Check secondary sort by symbol (within same timestamp)
    for ts in fills["timestamp"].unique():
        ts_fills = fills[fills["timestamp"] == ts]
        assert ts_fills["symbol"].is_monotonic_increasing, f"Should be sorted by symbol within timestamp {ts}"


def test_fill_schema_rejected_rules() -> None:
    """Test that rejected fills follow rules: fill_qty=0, remaining_qty=qty, costs=0."""
    # Create rejected fill
    order = {
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 100.0,
        "price": 150.0,
    }
    
    rejected_fill = create_rejected_fill_from_order(order)
    
    # Verify rejected rules
    assert rejected_fill["fill_qty"] == 0.0, "Rejected fill should have fill_qty=0"
    assert rejected_fill["remaining_qty"] == order["qty"], "Rejected fill should have remaining_qty=qty"
    assert rejected_fill["status"] == "rejected", "Status should be 'rejected'"
    assert rejected_fill["fill_price"] == order["price"], "fill_price should equal price for rejected"


def test_fill_schema_partial_rules() -> None:
    """Test that partial fills follow rules: 0 < fill_qty < qty, remaining_qty = qty - fill_qty."""
    # Create partial fill
    order = {
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 100.0,
        "price": 150.0,
    }
    
    partial_fill = create_partial_fill_from_order(order, fill_qty=60.0, fill_price=151.0)
    
    # Verify partial rules
    assert 0 < partial_fill["fill_qty"] < order["qty"], "Partial fill should have 0 < fill_qty < qty"
    assert partial_fill["remaining_qty"] == order["qty"] - partial_fill["fill_qty"], "remaining_qty should equal qty - fill_qty"
    assert partial_fill["status"] == "partial", "Status should be 'partial'"
    assert partial_fill["fill_price"] == 151.0, "fill_price should be set"


def test_fill_schema_no_nans_in_key_columns() -> None:
    """Test that key columns have no NaNs."""
    # Create trades with fill columns
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 75.0],
        "price": [150.0, 200.0, 100.0],
        "fill_qty": [100.0, 50.0, 75.0],
        "fill_price": [150.0, 200.0, 100.0],
        "status": ["filled", "filled", "filled"],
        "remaining_qty": [0.0, 0.0, 0.0],
    })
    
    fills = ensure_fill_schema(trades, default_full_fill=False)
    
    # Verify no NaNs in key columns
    key_cols = [
        "timestamp", "symbol", "side", "qty", "price",
        "fill_qty", "fill_price", "status", "remaining_qty",
    ]
    for col in key_cols:
        assert not fills[col].isna().any(), f"Column {col} should not contain NaNs"


def test_fill_schema_full_fill_default() -> None:
    """Test that ensure_fill_schema adds full fill columns by default (backward compatibility)."""
    # Create orders without fill columns
    orders = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })
    
    # Ensure fill schema (should add fill columns with full fill assumption)
    fills = ensure_fill_schema(orders, default_full_fill=True)
    
    # Verify full fill assumption
    assert (fills["fill_qty"] == fills["qty"]).all(), "Should assume full fills"
    assert (fills["fill_price"] == fills["price"]).all(), "Should assume fill_price = price"
    assert (fills["status"] == "filled").all(), "Should assume status = 'filled'"
    assert (fills["remaining_qty"] == 0.0).all(), "Should assume remaining_qty = 0"


def test_fill_schema_constraints_validation() -> None:
    """Test that ensure_fill_schema validates constraints."""
    # Create invalid fill (fill_qty > qty)
    invalid_trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=1, freq="1d", tz="UTC"),
        "symbol": ["AAPL"],
        "side": ["BUY"],
        "qty": [100.0],
        "price": [150.0],
        "fill_qty": [150.0],  # Invalid: > qty
        "fill_price": [150.0],
        "status": ["filled"],
        "remaining_qty": [-50.0],  # Invalid
    })
    
    # Should raise ValueError
    try:
        ensure_fill_schema(invalid_trades, default_full_fill=False)
        assert False, "Should raise ValueError for invalid fill_qty > qty"
    except ValueError:
        pass  # Expected


def test_create_partial_fill_invalid_qty() -> None:
    """Test that create_partial_fill_from_order raises ValueError for invalid fill_qty."""
    order = {
        "timestamp": pd.Timestamp("2024-01-01", tz="UTC"),
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 100.0,
        "price": 150.0,
    }
    
    # fill_qty >= qty should raise ValueError
    try:
        create_partial_fill_from_order(order, fill_qty=100.0)  # Should be < qty
        assert False, "Should raise ValueError for fill_qty >= qty"
    except ValueError:
        pass  # Expected
    
    # fill_qty <= 0 should raise ValueError
    try:
        create_partial_fill_from_order(order, fill_qty=0.0)
        assert False, "Should raise ValueError for fill_qty <= 0"
    except ValueError:
        pass  # Expected
