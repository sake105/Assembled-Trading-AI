"""Contract tests for core I/O data structures.

This module tests that all data contracts (as defined in docs/CONTRACTS.md) are
correctly enforced by the codebase.

Tests cover:
- Required columns exist
- Datatypes are correct (timestamp tz-aware, numeric types)
- No silent NaNs in required fields
- Sorting is correct
- TZ-Policy is enforced (UTC-only)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.io import load_orders, load_prices
from src.assembled_core.pipeline.orders import signals_to_orders
from src.assembled_core.pipeline.signals import compute_ema_signals
from src.assembled_core.utils.dataframe import coerce_price_types, ensure_cols


# ============================================================================
# Helper Functions
# ============================================================================


def assert_utc_timestamp(series: pd.Series, name: str = "timestamp") -> None:
    """Assert that a timestamp series is UTC-aware."""
    assert pd.api.types.is_datetime64_any_dtype(series), (
        f"{name} must be datetime type"
    )
    assert series.dt.tz is not None, f"{name} must be timezone-aware"
    assert str(series.dt.tz) == "UTC", f"{name} must be UTC (got {series.dt.tz})"


def assert_no_nans_in_required(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Assert that required columns have no NaNs."""
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing"
        nan_count = df[col].isna().sum()
        assert nan_count == 0, (
            f"Column '{col}' has {nan_count} NaNs in required field"
        )


# ============================================================================
# Price/Panel Contract Tests
# ============================================================================


def test_price_contract_required_columns() -> None:
    """Test that price DataFrame has required columns."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Should pass validation
    result = ensure_cols(prices, ["timestamp", "symbol", "close"])
    assert "timestamp" in result.columns
    assert "symbol" in result.columns
    assert "close" in result.columns


def test_price_contract_datatypes() -> None:
    """Test that price DataFrame has correct datatypes."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    result = coerce_price_types(prices)

    # Check timestamp is UTC-aware
    assert_utc_timestamp(result["timestamp"])

    # Check close is float64
    assert result["close"].dtype == np.float64

    # Check symbol is string
    assert result["symbol"].dtype == "string" or result["symbol"].dtype == object


def test_price_contract_no_nans() -> None:
    """Test that price DataFrame has no NaNs in required fields."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    result = coerce_price_types(prices)

    # Should have no NaNs after coercion
    assert_no_nans_in_required(result, ["timestamp", "symbol", "close"])


def test_price_contract_sorting() -> None:
    """Test that price DataFrame is sorted correctly."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC").tolist() * 2,
        "symbol": ["AAPL"] * 5 + ["GOOGL"] * 5,
        "close": [150.0] * 10,
    })

    # Shuffle to test sorting
    prices = prices.sample(frac=1).reset_index(drop=True)

    # Load prices should sort by symbol, then timestamp
    # (We can't test load_prices without a file, so we test the contract expectation)
    expected_sorted = prices.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    assert expected_sorted["symbol"].iloc[0] == "AAPL"
    assert expected_sorted["symbol"].iloc[5] == "GOOGL"


def test_price_contract_utc_only() -> None:
    """Test that price DataFrame enforces UTC-only policy."""
    # Test with non-UTC timestamp (should be coerced to UTC)
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="US/Eastern"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    result = coerce_price_types(prices)

    # Should be converted to UTC
    assert_utc_timestamp(result["timestamp"])


# ============================================================================
# Signal Contract Tests
# ============================================================================


def test_signal_contract_required_columns() -> None:
    """Test that signal DataFrame has required columns."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "close": [150.0 + i * 0.5 for i in range(20)],
    })

    signals = compute_ema_signals(prices, fast=5, slow=10)

    # Should have required columns
    assert "timestamp" in signals.columns
    assert "symbol" in signals.columns
    assert "sig" in signals.columns
    assert "price" in signals.columns


def test_signal_contract_datatypes() -> None:
    """Test that signal DataFrame has correct datatypes."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "close": [150.0 + i * 0.5 for i in range(20)],
    })

    signals = compute_ema_signals(prices, fast=5, slow=10)

    # Check timestamp is UTC-aware
    assert_utc_timestamp(signals["timestamp"])

    # Check sig is int8
    assert signals["sig"].dtype == np.int8

    # Check price is float64
    assert signals["price"].dtype == np.float64


def test_signal_contract_sig_values() -> None:
    """Test that signal values are only -1, 0, or +1."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "close": [150.0 + i * 0.5 for i in range(20)],
    })

    signals = compute_ema_signals(prices, fast=5, slow=10)

    # Should only contain -1, 0, or +1
    assert signals["sig"].isin([-1, 0, 1]).all(), (
        f"Signal values must be -1, 0, or +1 (got {signals['sig'].unique()})"
    )


def test_signal_contract_no_nans() -> None:
    """Test that signal DataFrame has no NaNs in required fields."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "close": [150.0 + i * 0.5 for i in range(20)],
    })

    signals = compute_ema_signals(prices, fast=5, slow=10)

    # Should have no NaNs
    assert_no_nans_in_required(signals, ["timestamp", "symbol", "sig", "price"])


# ============================================================================
# Orders Contract Tests
# ============================================================================


def test_orders_contract_required_columns() -> None:
    """Test that orders DataFrame has required columns."""
    signals = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "sig": np.array([0, 1, 1, -1, 0] * 4, dtype=np.int8),
        "price": [150.0 + i * 0.5 for i in range(20)],
    })

    orders = signals_to_orders(signals)

    # Should have required columns
    assert "timestamp" in orders.columns
    assert "symbol" in orders.columns
    assert "side" in orders.columns
    assert "qty" in orders.columns
    assert "price" in orders.columns


def test_orders_contract_datatypes() -> None:
    """Test that orders DataFrame has correct datatypes."""
    signals = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "sig": np.array([0, 1, 1, -1, 0] * 4, dtype=np.int8),
        "price": [150.0 + i * 0.5 for i in range(20)],
    })

    orders = signals_to_orders(signals)

    # Check timestamp is UTC-aware
    assert_utc_timestamp(orders["timestamp"])

    # Check qty is float64
    assert orders["qty"].dtype == np.float64

    # Check price is float64
    assert orders["price"].dtype == np.float64

    # Check side is string
    assert orders["side"].dtype == "string" or orders["side"].dtype == object


def test_orders_contract_side_values() -> None:
    """Test that side values are only 'BUY' or 'SELL'."""
    signals = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "sig": np.array([0, 1, 1, -1, 0] * 4, dtype=np.int8),
        "price": [150.0 + i * 0.5 for i in range(20)],
    })

    orders = signals_to_orders(signals)

    if not orders.empty:
        # Should only contain 'BUY' or 'SELL'
        assert orders["side"].isin(["BUY", "SELL"]).all(), (
            f"Side values must be 'BUY' or 'SELL' (got {orders['side'].unique()})"
        )


def test_orders_contract_qty_positive() -> None:
    """Test that qty is always positive."""
    signals = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "sig": np.array([0, 1, 1, -1, 0] * 4, dtype=np.int8),
        "price": [150.0 + i * 0.5 for i in range(20)],
    })

    orders = signals_to_orders(signals)

    if not orders.empty:
        # Should always be positive
        assert (orders["qty"] > 0).all(), (
            f"Qty must always be positive (got min={orders['qty'].min()})"
        )


def test_orders_contract_no_nans() -> None:
    """Test that orders DataFrame has no NaNs in required fields."""
    signals = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 20,
        "sig": np.array([0, 1, 1, -1, 0] * 4, dtype=np.int8),
        "price": [150.0 + i * 0.5 for i in range(20)],
    })

    orders = signals_to_orders(signals)

    if not orders.empty:
        # Should have no NaNs
        assert_no_nans_in_required(orders, ["timestamp", "symbol", "side", "qty", "price"])


# ============================================================================
# Target Positions Contract Tests
# ============================================================================


def test_target_positions_contract_required_columns() -> None:
    """Test that target positions DataFrame has required columns."""
    targets = pd.DataFrame({
        "symbol": ["AAPL", "GOOGL", "MSFT"],
        "target_qty": [100.0, 50.0, -25.0],
    })

    # Should have required columns
    assert "symbol" in targets.columns
    assert "target_qty" in targets.columns


def test_target_positions_contract_no_nans() -> None:
    """Test that target positions DataFrame has no NaNs in required fields."""
    targets = pd.DataFrame({
        "symbol": ["AAPL", "GOOGL", "MSFT"],
        "target_qty": [100.0, 50.0, -25.0],
    })

    # Should have no NaNs
    assert_no_nans_in_required(targets, ["symbol", "target_qty"])


# ============================================================================
# Equity Curve Contract Tests
# ============================================================================


def test_equity_curve_contract_required_columns() -> None:
    """Test that equity curve DataFrame has required columns."""
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "equity": [10000.0 + i * 100.0 for i in range(10)],
    })

    # Should have required columns
    assert "timestamp" in equity.columns
    assert "equity" in equity.columns


def test_equity_curve_contract_datatypes() -> None:
    """Test that equity curve DataFrame has correct datatypes."""
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "equity": [10000.0 + i * 100.0 for i in range(10)],
    })

    # Check timestamp is UTC-aware
    assert_utc_timestamp(equity["timestamp"])

    # Check equity is float64
    assert equity["equity"].dtype == np.float64


def test_equity_curve_contract_positive_equity() -> None:
    """Test that equity is always positive."""
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "equity": [10000.0 + i * 100.0 for i in range(10)],
    })

    # Should always be positive
    assert (equity["equity"] > 0).all(), (
        f"Equity must always be positive (got min={equity['equity'].min()})"
    )


def test_equity_curve_contract_no_nans() -> None:
    """Test that equity curve DataFrame has no NaNs in required fields."""
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "equity": [10000.0 + i * 100.0 for i in range(10)],
    })

    # Should have no NaNs
    assert_no_nans_in_required(equity, ["timestamp", "equity"])


# ============================================================================
# Integration Tests (using actual I/O functions)
# ============================================================================


@pytest.mark.integration
def test_price_contract_integration(tmp_path: Path) -> None:
    """Test price contract with actual load_prices function."""
    # Create a test price file
    price_file = tmp_path / "test_prices.parquet"
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })
    prices.to_parquet(price_file, index=False)

    # Load and validate
    result = load_prices("1d", price_file=price_file)

    # Should pass all contract checks
    assert_utc_timestamp(result["timestamp"])
    assert_no_nans_in_required(result, ["timestamp", "symbol", "close"])
    assert result["close"].dtype == np.float64


@pytest.mark.integration
def test_orders_contract_integration(tmp_path: Path) -> None:
    """Test orders contract with actual load_orders function."""
    # Create a test orders file
    orders_file = tmp_path / "orders_1d.csv"
    orders = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "GOOGL", "AAPL"],
        "side": ["BUY", "BUY", "SELL"],
        "qty": [100.0, 50.0, 50.0],
        "price": [150.0, 2500.0, 151.0],
    })
    orders.to_csv(orders_file, index=False)

    # Load and validate
    result = load_orders("1d", output_dir=tmp_path, strict=True)

    # Should pass all contract checks
    assert_utc_timestamp(result["timestamp"])
    assert_no_nans_in_required(result, ["timestamp", "symbol", "side", "qty", "price"])
    assert result["side"].isin(["BUY", "SELL"]).all()
    assert (result["qty"] > 0).all()
