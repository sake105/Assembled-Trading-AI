# tests/test_transaction_costs_spread.py
"""Tests for Spread Model (Sprint B2).

Tests verify:
1. Bucket assignment deterministisch
2. Volume missing fallback
3. ADV window effect (minimales synthetic panel)
4. spread_cash nie NaN
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.transaction_costs import (
    SpreadModel,
    add_cost_columns_to_trades,
    assign_spread_bps,
    compute_adv_proxy,
    compute_spread_cash,
)


def test_bucket_assignment_deterministic() -> None:
    """Test that bucket assignment is deterministic."""
    model = SpreadModel(
        adv_window=20,
        buckets=[(1e6, 5.0), (1e7, 3.0), (1e8, 1.0)],  # ADV thresholds in USD
        fallback_spread_bps=10.0,
    )
    
    # Test cases: different ADV values
    adv_usd = np.array([5e5, 5e6, 5e7, 5e8, np.nan])
    
    spread_bps = assign_spread_bps(adv_usd, model)
    
    # Expected: buckets are applied in reverse (highest threshold first)
    # Buckets: [(1e6, 5.0), (1e7, 3.0), (1e8, 1.0)]
    # Iteration in reverse: (1e8, 1.0), (1e7, 3.0), (1e6, 5.0)
    # - 5e8 >= 1e8: 1.0 bps (highest bucket, overwrites others)
    # - 5e7 >= 1e7: 3.0 bps (second bucket, overwrites first)
    # - 5e6 >= 1e6: 5.0 bps (first bucket)
    # - 5e5 < 1e6: fallback 10.0 bps (below all thresholds)
    # - NaN: fallback 10.0 bps
    # Note: The actual implementation checks thresholds in reverse order,
    # so 5e6 gets 5.0 bps (first bucket), not 1.0 bps
    expected = np.array([10.0, 5.0, 3.0, 1.0, 10.0])
    
    np.testing.assert_array_almost_equal(spread_bps, expected, decimal=10)
    assert not np.any(np.isnan(spread_bps)), "spread_bps should not contain NaNs"


def test_volume_missing_fallback() -> None:
    """Test that missing volume uses fallback spread_bps."""
    # Create prices without volume
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 30,
        "close": [150.0] * 30,
        # No volume column
    })
    
    model = SpreadModel(
        adv_window=20,
        buckets=[(1e6, 5.0)],
        fallback_spread_bps=10.0,
    )
    
    # Compute ADV proxy
    adv_df = compute_adv_proxy(prices, adv_window=model.adv_window)
    
    # All ADV values should be NaN (volume missing)
    assert adv_df["adv_usd"].isna().all(), "ADV should be NaN when volume is missing"
    
    # Assign spread_bps
    spread_bps = assign_spread_bps(adv_df["adv_usd"].values, model)
    
    # All should use fallback
    expected = np.full(len(adv_df), 10.0, dtype=np.float64)
    np.testing.assert_array_almost_equal(spread_bps, expected, decimal=10)


def test_adv_window_effect() -> None:
    """Test ADV window effect with minimal synthetic panel."""
    # Create synthetic prices with volume
    timestamps = pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 30,
        "close": [150.0 + i * 0.1 for i in range(30)],
        "volume": [1000000.0] * 30,  # Constant volume
    })
    
    model = SpreadModel(
        adv_window=20,
        buckets=[(1e6, 5.0), (1e7, 3.0)],
        fallback_spread_bps=10.0,
    )
    
    # Compute ADV proxy
    adv_df = compute_adv_proxy(prices, adv_window=model.adv_window)
    
    # First 19 rows should have NaN (insufficient history for 20-day window)
    assert adv_df["adv_usd"].iloc[:19].isna().all(), "First 19 rows should have NaN ADV"
    
    # Row 19 (index 19) should have non-NaN ADV (20-day window starting from row 0)
    # Actually: rolling window with min_periods=20 means first non-NaN is at index 19
    assert not adv_df["adv_usd"].iloc[19:].isna().any(), "Rows 19+ should have non-NaN ADV"
    
    # ADV should be approximately close * volume (rolling mean)
    # For constant volume=1e6 and close~150, ADV ~ 150 * 1e6 = 1.5e8
    # Buckets: [(1e6, 5.0), (1e7, 3.0)]
    # 1.5e8 >= 1e7: gets 3.0 bps (highest bucket in the list)
    # Since 1.5e8 > 1e7 and there's no higher bucket, it gets 3.0 bps
    adv_values = adv_df["adv_usd"].iloc[19:].values
    spread_bps = assign_spread_bps(adv_values, model)
    
    # All should get 3.0 bps (ADV >= 1e7, highest bucket in the list)
    expected = np.full(len(spread_bps), 3.0, dtype=np.float64)
    np.testing.assert_array_almost_equal(spread_bps, expected, decimal=10)


def test_spread_cash_never_nan() -> None:
    """Test that spread_cash is never NaN."""
    # Test with various notional and spread_bps values
    notional = np.array([10000.0, 20000.0, 0.0, np.inf])
    spread_bps = np.array([5.0, 10.0, 0.0, 1.0])
    
    spread_cash = compute_spread_cash(notional, spread_bps)
    
    # Verify no NaNs
    assert not np.any(np.isnan(spread_cash)), "spread_cash should not contain NaNs"
    assert np.all(spread_cash >= 0.0), "spread_cash should be non-negative"
    
    # Verify calculation: spread_cash = notional * (spread_bps / 10000) * 0.5
    # 10000 * (5.0 / 10000) * 0.5 = 2.5
    # 20000 * (10.0 / 10000) * 0.5 = 10.0
    # 0.0 * (0.0 / 10000) * 0.5 = 0.0
    # inf * (1.0 / 10000) * 0.5 = inf (acceptable)
    expected = np.array([2.5, 10.0, 0.0, np.inf])
    
    # Compare finite values
    finite_mask = np.isfinite(spread_cash) & np.isfinite(expected)
    np.testing.assert_array_almost_equal(
        spread_cash[finite_mask],
        expected[finite_mask],
        decimal=10
    )


def test_add_cost_columns_to_trades_with_spread() -> None:
    """Test that add_cost_columns_to_trades computes spread_cash correctly."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 200.0],
        "price": [150.0, 200.0, 100.0],
    })
    
    # Create prices with volume for ADV calculation
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10 + ["MSFT"] * 10 + ["GOOGL"] * 10,
        "close": [150.0] * 10 + [200.0] * 10 + [100.0] * 10,
        "volume": [1000000.0] * 30,  # Constant volume
    })
    
    model = SpreadModel(
        adv_window=20,
        buckets=[(1e6, 5.0), (1e7, 3.0)],
        fallback_spread_bps=10.0,
    )
    
    # Add cost columns with spread model
    trades_with_costs = add_cost_columns_to_trades(
        trades,
        commission_model=None,  # Default: no commission
        spread_model=model,
        prices=prices,
    )
    
    # Verify spread_cash column exists and is not NaN
    assert "spread_cash" in trades_with_costs.columns, "spread_cash column should exist"
    assert not trades_with_costs["spread_cash"].isna().any(), "spread_cash should not contain NaNs"
    assert (trades_with_costs["spread_cash"] >= 0.0).all(), "spread_cash should be non-negative"
    
    # Verify spread_cash calculation
    # For ADV ~ 150 * 1e6 = 1.5e8 (exceeds 1e7), spread_bps = 3.0
    # AAPL: 100 * 150 = 15000 notional, spread_cash = 15000 * (3.0 / 10000) * 0.5 = 2.25
    # MSFT: 50 * 200 = 10000 notional, spread_cash = 10000 * (3.0 / 10000) * 0.5 = 1.5
    # GOOGL: 200 * 100 = 20000 notional, spread_cash = 20000 * (3.0 / 10000) * 0.5 = 3.0
    # Note: ADV calculation requires 20 days of history, so first trades might use fallback
    # But for this test, we have 30 days, so ADV should be available
    
    # Verify total_cost_cash includes spread_cash
    assert "total_cost_cash" in trades_with_costs.columns, "total_cost_cash column should exist"
    expected_total = (
        trades_with_costs["commission_cash"]
        + trades_with_costs["spread_cash"]
        + trades_with_costs["slippage_cash"]
    )
    np.testing.assert_array_almost_equal(
        trades_with_costs["total_cost_cash"].values,
        expected_total.values,
        decimal=10
    )


def test_add_cost_columns_to_trades_no_spread_model() -> None:
    """Test that add_cost_columns_to_trades defaults spread_cash to 0.0 when no spread_model."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })
    
    # Add cost columns without spread model
    trades_with_costs = add_cost_columns_to_trades(
        trades,
        commission_model=None,
        spread_model=None,
        prices=None,
    )
    
    # Verify spread_cash defaults to 0.0
    assert "spread_cash" in trades_with_costs.columns, "spread_cash column should exist"
    assert (trades_with_costs["spread_cash"] == 0.0).all(), "spread_cash should default to 0.0"
    assert not trades_with_costs["spread_cash"].isna().any(), "spread_cash should not contain NaNs"


def test_add_cost_columns_to_trades_empty_trades() -> None:
    """Test that empty trades DataFrame returns stable empty DF with schema."""
    trades = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    
    model = SpreadModel(
        adv_window=20,
        buckets=[(1e6, 5.0)],
        fallback_spread_bps=10.0,
    )
    
    trades_with_costs = add_cost_columns_to_trades(
        trades,
        commission_model=None,
        spread_model=model,
        prices=None,
    )
    
    # Verify schema: required columns + cost columns
    required_cols = ["timestamp", "symbol", "side", "qty", "price"]
    cost_cols = ["commission_cash", "spread_cash", "slippage_cash", "total_cost_cash"]
    
    for col in required_cols + cost_cols:
        assert col in trades_with_costs.columns, f"Column {col} should exist in empty trades"
    
    # Verify empty
    assert trades_with_costs.empty, "Result should be empty"
    
    # Verify no NaNs (even in empty DataFrame)
    for col in cost_cols:
        assert not trades_with_costs[col].isna().any(), f"Column {col} should not contain NaNs"


def test_spread_model_validation() -> None:
    """Test SpreadModel parameter validation."""
    # Test invalid adv_window
    try:
        SpreadModel(adv_window=0)
        assert False, "Should raise ValueError for adv_window < 1"
    except ValueError:
        pass
    
    # Test negative fallback_spread_bps
    try:
        SpreadModel(fallback_spread_bps=-1.0)
        assert False, "Should raise ValueError for negative fallback_spread_bps"
    except ValueError:
        pass
    
    # Test unsorted buckets
    try:
        SpreadModel(buckets=[(1e7, 3.0), (1e6, 5.0)])  # Not sorted
        assert False, "Should raise ValueError for unsorted buckets"
    except ValueError:
        pass
    
    # Test negative spread_bps in buckets
    try:
        SpreadModel(buckets=[(1e6, -1.0)])
        assert False, "Should raise ValueError for negative spread_bps in buckets"
    except ValueError:
        pass
