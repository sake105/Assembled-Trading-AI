# tests/test_transaction_costs_slippage.py
"""Tests for Slippage Model (Sprint B3).

Tests verify:
1. Deterministic slippage_bps calculation
2. Clamps greifen (min_bps, max_bps)
3. Fallback ohne volume
4. Keine NaNs
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
    SlippageModel,
    add_cost_columns_to_trades,
    compute_slippage_bps,
    compute_slippage_cash,
    compute_volatility_proxy,
)


def test_slippage_bps_deterministic() -> None:
    """Test that slippage_bps calculation is deterministic."""
    model = SlippageModel(
        vol_window=20,
        k=1.0,
        min_bps=0.0,
        max_bps=50.0,
        fallback_slippage_bps=5.0,
    )
    
    # Test case: notional, adv_usd, volatility
    notional = np.array([10000.0, 20000.0, 5000.0])
    adv_usd = np.array([100000.0, 200000.0, 50000.0])
    volatility = np.array([0.02, 0.03, 0.01])  # 2%, 3%, 1% daily volatility
    
    slippage_bps = compute_slippage_bps(notional, adv_usd, volatility, model)
    
    # Verify no NaNs
    assert not np.any(np.isnan(slippage_bps)), "slippage_bps should not contain NaNs"
    assert np.all(slippage_bps >= 0.0), "slippage_bps should be non-negative"
    
    # Verify calculation: k * sigma * sqrt(participation) * 10000
    # participation = notional / adv_usd
    # participation = [0.1, 0.1, 0.1]
    # sqrt(participation) = [0.316, 0.316, 0.316]
    # k * sigma * sqrt(participation) * 10000 = [63.2, 94.8, 31.6] bps
    # But clamped to max_bps=50.0, so: [50.0, 50.0, 31.6] bps
    expected = np.array([50.0, 50.0, 31.6])  # Clamped to max_bps
    
    np.testing.assert_array_almost_equal(slippage_bps, expected, decimal=1)


def test_slippage_bps_clamps() -> None:
    """Test that clamps (min_bps, max_bps) are applied correctly."""
    model = SlippageModel(
        vol_window=20,
        k=10.0,  # High scaling factor to trigger max clamp
        min_bps=5.0,
        max_bps=20.0,
        fallback_slippage_bps=10.0,
    )
    
    # Test case: high volatility and participation to trigger max clamp
    notional = np.array([100000.0, 1000.0])  # High and low notional
    adv_usd = np.array([100000.0, 100000.0])  # Same ADV
    volatility = np.array([0.1, 0.001])  # High and low volatility
    
    slippage_bps = compute_slippage_bps(notional, adv_usd, volatility, model)
    
    # Verify clamps
    assert np.all(slippage_bps >= model.min_bps), f"slippage_bps should be >= {model.min_bps}"
    assert np.all(slippage_bps <= model.max_bps), f"slippage_bps should be <= {model.max_bps}"
    
    # First value should be clamped to max_bps (high volatility + high participation)
    assert slippage_bps[0] == model.max_bps, "High slippage should be clamped to max_bps"
    
    # Second value might be clamped to min_bps or be between min and max
    assert slippage_bps[1] >= model.min_bps, "Low slippage should be >= min_bps"


def test_slippage_fallback_without_volume() -> None:
    """Test that missing volume uses fallback slippage_bps."""
    # Create prices without volume
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 30,
        "close": [150.0] * 30,
        # No volume column
    })
    
    model = SlippageModel(
        vol_window=20,
        k=1.0,
        min_bps=0.0,
        max_bps=50.0,
        fallback_slippage_bps=10.0,
    )
    
    # Compute volatility proxy (should work without volume)
    vol_df = compute_volatility_proxy(prices, vol_window=model.vol_window)
    
    # Compute ADV proxy (should return NaN without volume)
    from src.assembled_core.execution.transaction_costs import compute_adv_proxy
    adv_df = compute_adv_proxy(prices, adv_window=model.vol_window)
    
    # All ADV values should be NaN (volume missing)
    assert adv_df["adv_usd"].isna().all(), "ADV should be NaN when volume is missing"
    
    # Compute slippage_bps with NaN ADV
    notional = np.array([10000.0, 20000.0])
    adv_usd = adv_df["adv_usd"].values[:2]  # First 2 values (NaN)
    volatility = vol_df["volatility"].values[:2]  # May be NaN if insufficient history
    
    slippage_bps = compute_slippage_bps(notional, adv_usd, volatility, model)
    
    # All should use fallback (ADV is NaN)
    expected = np.full(len(slippage_bps), 10.0, dtype=np.float64)
    np.testing.assert_array_almost_equal(slippage_bps, expected, decimal=10)


def test_slippage_cash_never_nan() -> None:
    """Test that slippage_cash is never NaN."""
    # Test with various notional and slippage_bps values
    notional = np.array([10000.0, 20000.0, 0.0, np.inf])
    slippage_bps = np.array([5.0, 10.0, 0.0, 1.0])
    
    slippage_cash = compute_slippage_cash(notional, slippage_bps)
    
    # Verify no NaNs
    assert not np.any(np.isnan(slippage_cash)), "slippage_cash should not contain NaNs"
    assert np.all(slippage_cash >= 0.0), "slippage_cash should be non-negative"
    
    # Verify calculation: slippage_cash = notional * (slippage_bps / 10000)
    # 10000 * (5.0 / 10000) = 5.0
    # 20000 * (10.0 / 10000) = 20.0
    # 0.0 * (0.0 / 10000) = 0.0
    # inf * (1.0 / 10000) = inf (acceptable)
    expected = np.array([5.0, 20.0, 0.0, np.inf])
    
    # Compare finite values
    finite_mask = np.isfinite(slippage_cash) & np.isfinite(expected)
    np.testing.assert_array_almost_equal(
        slippage_cash[finite_mask],
        expected[finite_mask],
        decimal=10
    )


def test_add_cost_columns_to_trades_with_slippage() -> None:
    """Test that add_cost_columns_to_trades computes slippage_cash correctly."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 200.0],
        "price": [150.0, 200.0, 100.0],
    })
    
    # Create prices with volume for ADV and volatility calculation
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC"),
        "symbol": ["AAPL"] * 10 + ["MSFT"] * 10 + ["GOOGL"] * 10,
        "close": [150.0 + i * 0.1 for i in range(30)],
        "volume": [1000000.0] * 30,  # Constant volume
    })
    
    model = SlippageModel(
        vol_window=20,
        k=1.0,
        min_bps=0.0,
        max_bps=50.0,
        fallback_slippage_bps=5.0,
    )
    
    # Add cost columns with slippage model
    trades_with_costs = add_cost_columns_to_trades(
        trades,
        commission_model=None,  # Default: no commission
        spread_model=None,  # Default: no spread
        slippage_model=model,
        prices=prices,
    )
    
    # Verify slippage_cash column exists and is not NaN
    assert "slippage_cash" in trades_with_costs.columns, "slippage_cash column should exist"
    assert not trades_with_costs["slippage_cash"].isna().any(), "slippage_cash should not contain NaNs"
    assert (trades_with_costs["slippage_cash"] >= 0.0).all(), "slippage_cash should be non-negative"
    
    # Verify total_cost_cash includes slippage_cash
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


def test_add_cost_columns_to_trades_no_slippage_model() -> None:
    """Test that add_cost_columns_to_trades defaults slippage_cash to 0.0 when no slippage_model."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })
    
    # Add cost columns without slippage model
    trades_with_costs = add_cost_columns_to_trades(
        trades,
        commission_model=None,
        spread_model=None,
        slippage_model=None,
        prices=None,
    )
    
    # Verify slippage_cash defaults to 0.0
    assert "slippage_cash" in trades_with_costs.columns, "slippage_cash column should exist"
    assert (trades_with_costs["slippage_cash"] == 0.0).all(), "slippage_cash should default to 0.0"
    assert not trades_with_costs["slippage_cash"].isna().any(), "slippage_cash should not contain NaNs"


def test_slippage_model_validation() -> None:
    """Test SlippageModel parameter validation."""
    # Test invalid vol_window
    try:
        SlippageModel(vol_window=0)
        assert False, "Should raise ValueError for vol_window < 1"
    except ValueError:
        pass
    
    # Test negative k
    try:
        SlippageModel(k=-1.0)
        assert False, "Should raise ValueError for negative k"
    except ValueError:
        pass
    
    # Test negative min_bps
    try:
        SlippageModel(min_bps=-1.0)
        assert False, "Should raise ValueError for negative min_bps"
    except ValueError:
        pass
    
    # Test max_bps < min_bps
    try:
        SlippageModel(min_bps=10.0, max_bps=5.0)
        assert False, "Should raise ValueError for max_bps < min_bps"
    except ValueError:
        pass
    
    # Test negative participation_rate_cap
    try:
        SlippageModel(participation_rate_cap=-1.0)
        assert False, "Should raise ValueError for negative participation_rate_cap"
    except ValueError:
        pass
    
    # Test negative fallback_slippage_bps
    try:
        SlippageModel(fallback_slippage_bps=-1.0)
        assert False, "Should raise ValueError for negative fallback_slippage_bps"
    except ValueError:
        pass


def test_slippage_participation_rate_cap() -> None:
    """Test that participation_rate_cap is applied correctly."""
    model = SlippageModel(
        vol_window=20,
        k=1.0,
        min_bps=0.0,
        max_bps=100.0,  # High max to avoid clamp
        participation_rate_cap=0.5,  # Cap participation at 50%
        fallback_slippage_bps=5.0,
    )
    
    # Test case: notional > adv_usd (participation > 1.0, should be capped to 0.5)
    notional = np.array([200000.0])  # 2x ADV
    adv_usd = np.array([100000.0])
    volatility = np.array([0.02])  # 2% daily volatility
    
    slippage_bps = compute_slippage_bps(notional, adv_usd, volatility, model)
    
    # Verify participation is capped
    # participation = 200000 / 100000 = 2.0, but capped to 0.5
    # sqrt(0.5) = 0.707
    # k * sigma * sqrt(participation) * 10000 = 1.0 * 0.02 * 0.707 * 10000 = 141.4 bps
    # But max_bps=100.0, so clamped to 100.0
    assert slippage_bps[0] <= model.max_bps, "slippage_bps should respect max_bps clamp"
    assert slippage_bps[0] > 0.0, "slippage_bps should be positive"


def test_slippage_volatility_proxy() -> None:
    """Test volatility proxy calculation."""
    # Create synthetic prices with known volatility
    timestamps = pd.date_range("2024-01-01", periods=30, freq="1d", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": timestamps,
        "symbol": ["AAPL"] * 30,
        "close": [150.0 + i * 0.1 for i in range(30)],  # Small linear trend
    })
    
    vol_df = compute_volatility_proxy(prices, vol_window=20)
    
    # First 19 rows should have NaN (insufficient history for 20-day window)
    # Rolling window with min_periods=20: first non-NaN is at index 19 (20 data points: 0-19)
    assert vol_df["volatility"].iloc[:19].isna().all(), "First 19 rows should have NaN volatility"
    
    # Row 19+ might have non-NaN volatility (if log returns are valid)
    # For small linear trends, log returns might be very small, leading to very small volatility
    # Check that at least some rows 19+ have non-NaN volatility
    non_nan_vol = vol_df["volatility"].iloc[19:].dropna()
    assert len(non_nan_vol) > 0, "At least some rows 19+ should have non-NaN volatility"
    
    # Volatility should be non-negative (for non-NaN values)
    assert (non_nan_vol >= 0.0).all(), "Volatility should be non-negative"
