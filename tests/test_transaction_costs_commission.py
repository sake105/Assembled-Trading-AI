# tests/test_transaction_costs_commission.py
"""Tests for Commission Model (Sprint B1).

Tests verify:
1. bps-only commission calculation
2. fixed-only commission calculation
3. bps+fixed commission calculation
4. Deterministic behavior (no NaNs)
5. Cost columns added to trades DataFrame
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
    CommissionModel,
    add_cost_columns_to_trades,
    commission_model_from_cost_params,
    compute_commission_cash,
)


def test_commission_bps_only() -> None:
    """Test bps-only commission calculation."""
    model = CommissionModel(mode="bps", commission_bps=10.0, fixed_per_trade=0.0)
    
    # Test case: 10000 notional, 10 bps = 10.0 commission
    notional = np.array([10000.0, 20000.0, 5000.0])
    n_trades = len(notional)
    
    commission_cash = compute_commission_cash(notional, n_trades, model)
    
    # Expected: 10000 * 10 / 10000 = 10.0, 20000 * 10 / 10000 = 20.0, 5000 * 10 / 10000 = 5.0
    expected = np.array([10.0, 20.0, 5.0])
    
    np.testing.assert_array_almost_equal(commission_cash, expected, decimal=10)
    assert not np.any(np.isnan(commission_cash)), "Commission cash should not contain NaNs"
    assert np.all(commission_cash >= 0.0), "Commission cash should be non-negative"


def test_commission_fixed_only() -> None:
    """Test fixed-only commission calculation."""
    model = CommissionModel(mode="fixed", commission_bps=0.0, fixed_per_trade=1.5)
    
    # Test case: fixed 1.5 per trade
    notional = np.array([10000.0, 20000.0, 5000.0])
    n_trades = len(notional)
    
    commission_cash = compute_commission_cash(notional, n_trades, model)
    
    # Expected: 1.5 for each trade (regardless of notional)
    expected = np.array([1.5, 1.5, 1.5])
    
    np.testing.assert_array_almost_equal(commission_cash, expected, decimal=10)
    assert not np.any(np.isnan(commission_cash)), "Commission cash should not contain NaNs"
    assert np.all(commission_cash >= 0.0), "Commission cash should be non-negative"


def test_commission_bps_plus_fixed() -> None:
    """Test bps+fixed commission calculation."""
    model = CommissionModel(mode="bps_plus_fixed", commission_bps=5.0, fixed_per_trade=1.0)
    
    # Test case: 10000 notional, 5 bps + 1.0 fixed = 5.0 + 1.0 = 6.0
    notional = np.array([10000.0, 20000.0, 5000.0])
    n_trades = len(notional)
    
    commission_cash = compute_commission_cash(notional, n_trades, model)
    
    # Expected: 
    # 10000 * 5 / 10000 + 1.0 = 5.0 + 1.0 = 6.0
    # 20000 * 5 / 10000 + 1.0 = 10.0 + 1.0 = 11.0
    # 5000 * 5 / 10000 + 1.0 = 2.5 + 1.0 = 3.5
    expected = np.array([6.0, 11.0, 3.5])
    
    np.testing.assert_array_almost_equal(commission_cash, expected, decimal=10)
    assert not np.any(np.isnan(commission_cash)), "Commission cash should not contain NaNs"
    assert np.all(commission_cash >= 0.0), "Commission cash should be non-negative"


def test_commission_deterministic_no_nans() -> None:
    """Test that commission calculation is deterministic and produces no NaNs."""
    model = CommissionModel(mode="bps", commission_bps=1.0, fixed_per_trade=0.0)
    
    # Test with various edge cases
    notional = np.array([0.0, 1.0, 100.0, 1000000.0, np.inf])
    n_trades = len(notional)
    
    # Filter out inf for comparison (inf handling is implementation-dependent)
    notional_finite = notional[np.isfinite(notional)]
    n_trades_finite = len(notional_finite)
    
    commission_cash_finite = compute_commission_cash(notional_finite, n_trades_finite, model)
    
    # All finite values should produce finite commission
    assert not np.any(np.isnan(commission_cash_finite)), "Commission cash should not contain NaNs for finite notional"
    assert np.all(commission_cash_finite >= 0.0), "Commission cash should be non-negative"
    
    # Test with inf (should handle gracefully)
    commission_cash_all = compute_commission_cash(notional, n_trades, model)
    # inf * bps = inf, which is acceptable (costs are always positive)
    assert np.all(commission_cash_all >= 0.0), "Commission cash should be non-negative (even with inf)"


def test_add_cost_columns_to_trades() -> None:
    """Test that cost columns are added to trades DataFrame."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "side": ["BUY", "SELL", "BUY"],
        "qty": [100.0, 50.0, 200.0],
        "price": [150.0, 200.0, 100.0],
    })
    
    # Test with bps-only model
    model = CommissionModel(mode="bps", commission_bps=10.0, fixed_per_trade=0.0)
    trades_with_costs = add_cost_columns_to_trades(trades, commission_model=model)
    
    # Verify required columns exist
    required_cost_cols = ["commission_cash", "spread_cash", "slippage_cash", "total_cost_cash"]
    for col in required_cost_cols:
        assert col in trades_with_costs.columns, f"Cost column {col} should be present"
    
    # Verify no NaNs
    for col in required_cost_cols:
        assert not trades_with_costs[col].isna().any(), f"Cost column {col} should not contain NaNs"
        assert (trades_with_costs[col] >= 0.0).all(), f"Cost column {col} should be non-negative"
    
    # Verify commission_cash calculation
    # AAPL: 100 * 150 = 15000 notional, 10 bps = 15.0
    # MSFT: 50 * 200 = 10000 notional, 10 bps = 10.0
    # GOOGL: 200 * 100 = 20000 notional, 10 bps = 20.0
    expected_commission = np.array([15.0, 10.0, 20.0])
    np.testing.assert_array_almost_equal(
        trades_with_costs["commission_cash"].values,
        expected_commission,
        decimal=10
    )
    
    # Verify spread_cash and slippage_cash are zero (default)
    assert (trades_with_costs["spread_cash"] == 0.0).all(), "spread_cash should default to 0.0"
    assert (trades_with_costs["slippage_cash"] == 0.0).all(), "slippage_cash should default to 0.0"
    
    # Verify total_cost_cash = commission_cash (since spread and slippage are 0)
    np.testing.assert_array_almost_equal(
        trades_with_costs["total_cost_cash"].values,
        trades_with_costs["commission_cash"].values,
        decimal=10
    )
    
    # Verify deterministic sorting (by timestamp, symbol)
    assert trades_with_costs.equals(
        trades_with_costs.sort_values(["timestamp", "symbol"], ignore_index=True)
    ), "Trades should be sorted by timestamp, symbol"


def test_add_cost_columns_to_trades_fixed_mode() -> None:
    """Test cost columns with fixed commission mode."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })
    
    # Test with fixed-only model
    model = CommissionModel(mode="fixed", commission_bps=0.0, fixed_per_trade=2.5)
    trades_with_costs = add_cost_columns_to_trades(trades, commission_model=model)
    
    # Verify commission_cash is fixed for all trades
    expected_commission = np.array([2.5, 2.5])
    np.testing.assert_array_almost_equal(
        trades_with_costs["commission_cash"].values,
        expected_commission,
        decimal=10
    )


def test_add_cost_columns_to_trades_bps_plus_fixed() -> None:
    """Test cost columns with bps+fixed commission mode."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
        "symbol": ["AAPL", "MSFT"],
        "side": ["BUY", "SELL"],
        "qty": [100.0, 50.0],
        "price": [150.0, 200.0],
    })
    
    # Test with bps+fixed model
    model = CommissionModel(mode="bps_plus_fixed", commission_bps=5.0, fixed_per_trade=1.0)
    trades_with_costs = add_cost_columns_to_trades(trades, commission_model=model)
    
    # Verify commission_cash calculation
    # AAPL: 100 * 150 = 15000 notional, 5 bps = 7.5, + 1.0 fixed = 8.5
    # MSFT: 50 * 200 = 10000 notional, 5 bps = 5.0, + 1.0 fixed = 6.0
    expected_commission = np.array([8.5, 6.0])
    np.testing.assert_array_almost_equal(
        trades_with_costs["commission_cash"].values,
        expected_commission,
        decimal=10
    )


def test_commission_model_from_cost_params() -> None:
    """Test creating CommissionModel from legacy cost parameters."""
    # Test bps-only
    model = commission_model_from_cost_params(commission_bps=10.0, fixed_per_trade=0.0)
    assert model.mode == "bps"
    assert model.commission_bps == 10.0
    assert model.fixed_per_trade == 0.0
    
    # Test fixed-only
    model = commission_model_from_cost_params(commission_bps=0.0, fixed_per_trade=2.5)
    assert model.mode == "fixed"
    assert model.commission_bps == 0.0
    assert model.fixed_per_trade == 2.5
    
    # Test bps+fixed
    model = commission_model_from_cost_params(commission_bps=5.0, fixed_per_trade=1.0)
    assert model.mode == "bps_plus_fixed"
    assert model.commission_bps == 5.0
    assert model.fixed_per_trade == 1.0
    
    # Test explicit mode override
    model = commission_model_from_cost_params(
        commission_bps=5.0, fixed_per_trade=1.0, mode="bps"
    )
    assert model.mode == "bps"
    assert model.commission_bps == 5.0
    assert model.fixed_per_trade == 1.0  # Still set, but not used in bps mode


def test_commission_model_validation() -> None:
    """Test CommissionModel parameter validation."""
    # Test invalid mode
    try:
        CommissionModel(mode="invalid", commission_bps=0.0, fixed_per_trade=0.0)
        assert False, "Should raise ValueError for invalid mode"
    except ValueError:
        pass
    
    # Test negative commission_bps
    try:
        CommissionModel(mode="bps", commission_bps=-1.0, fixed_per_trade=0.0)
        assert False, "Should raise ValueError for negative commission_bps"
    except ValueError:
        pass
    
    # Test negative fixed_per_trade
    try:
        CommissionModel(mode="fixed", commission_bps=0.0, fixed_per_trade=-1.0)
        assert False, "Should raise ValueError for negative fixed_per_trade"
    except ValueError:
        pass
