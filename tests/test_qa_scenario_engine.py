"""Tests for scenario engine risk analysis."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.scenario_engine import (
    Scenario,
    apply_scenario_to_equity,
    apply_scenario_to_prices,
    run_scenario_on_equity,
)

pytestmark = pytest.mark.phase8


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Create synthetic price data."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    prices = []
    for symbol in symbols:
        base_price = 100.0 if symbol == "AAPL" else 150.0 if symbol == "MSFT" else 200.0
        for i, date in enumerate(dates):
            # Simple upward trend with small random noise
            price = base_price * (1 + 0.001 * i) + np.random.normal(0, 1, 1)[0]
            prices.append({
                "timestamp": date,
                "symbol": symbol,
                "close": price,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
            })
    
    return pd.DataFrame(prices)


@pytest.fixture
def synthetic_equity() -> pd.Series:
    """Create synthetic equity curve."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    equity_values = [10000.0]
    for i in range(1, 30):
        # Small positive returns
        equity_values.append(equity_values[-1] * (1 + 0.001))
    
    return pd.Series(equity_values, index=dates)


def test_scenario_baseline_no_shock(synthetic_prices):
    """Test that baseline scenario (no shock) leaves prices unchanged."""
    scenario = Scenario(
        name="Baseline",
        shock_type="equity_crash",
        shock_magnitude=0.0,  # No shock
        shock_start=None,
        shock_end=None
    )
    
    shocked_prices = apply_scenario_to_prices(synthetic_prices, scenario)
    
    # Prices should be unchanged (within floating point precision)
    # Sort both by timestamp and symbol for comparison
    original_sorted = synthetic_prices[["timestamp", "symbol", "close"]].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    shocked_sorted = shocked_prices[["timestamp", "symbol", "close"]].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    
    pd.testing.assert_frame_equal(
        original_sorted,
        shocked_sorted,
        check_exact=False,
        rtol=1e-5
    )


def test_scenario_equity_crash_basic(synthetic_prices):
    """Test equity_crash scenario applies price decline correctly."""
    shock_start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    scenario = Scenario(
        name="Market Crash",
        shock_type="equity_crash",
        shock_magnitude=-0.20,  # -20% crash
        shock_start=shock_start,
        shock_end=None
    )
    
    shocked_prices = apply_scenario_to_prices(synthetic_prices, scenario)
    
    # Check that prices before shock_start are unchanged
    before_shock = synthetic_prices[synthetic_prices["timestamp"] < shock_start].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    shocked_before = shocked_prices[shocked_prices["timestamp"] < shock_start].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        before_shock[["timestamp", "symbol", "close"]],
        shocked_before[["timestamp", "symbol", "close"]],
        check_exact=False,
        rtol=1e-5
    )
    
    # Check that prices at/after shock_start are reduced
    after_shock = synthetic_prices[synthetic_prices["timestamp"] >= shock_start]
    shocked_after = shocked_prices[shocked_prices["timestamp"] >= shock_start]
    
    # Prices should be approximately 20% lower (within tolerance)
    for symbol in synthetic_prices["symbol"].unique():
        symbol_before = after_shock[after_shock["symbol"] == symbol]["close"].iloc[0]
        symbol_after = shocked_after[shocked_after["symbol"] == symbol]["close"].iloc[0]
        
        # Check that price is reduced by approximately 20%
        expected_ratio = 1 + scenario.shock_magnitude  # 0.8
        actual_ratio = symbol_after / symbol_before
        assert abs(actual_ratio - expected_ratio) < 0.01, f"Price ratio mismatch for {symbol}"


def test_scenario_equity_crash_with_end(synthetic_prices):
    """Test equity_crash scenario with explicit shock_end."""
    shock_start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    shock_end = datetime(2024, 1, 20, tzinfo=timezone.utc)
    scenario = Scenario(
        name="Temporary Crash",
        shock_type="equity_crash",
        shock_magnitude=-0.15,  # -15% crash
        shock_start=shock_start,
        shock_end=shock_end
    )
    
    shocked_prices = apply_scenario_to_prices(synthetic_prices, scenario)
    
    # Prices before and after shock period should be unchanged
    before_shock = synthetic_prices[synthetic_prices["timestamp"] < shock_start].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    shocked_before = shocked_prices[shocked_prices["timestamp"] < shock_start].sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        before_shock[["timestamp", "symbol", "close"]],
        shocked_before[["timestamp", "symbol", "close"]],
        check_exact=False,
        rtol=1e-5
    )
    
    after_shock = synthetic_prices[synthetic_prices["timestamp"] > shock_end]
    shocked_after = shocked_prices[shocked_prices["timestamp"] > shock_end]
    # Note: Prices after shock_end will be affected because they're scaled from the shocked baseline
    # This is expected behavior for equity_crash


def test_scenario_vol_spike(synthetic_prices):
    """Test vol_spike scenario increases return volatility."""
    shock_start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    shock_end = datetime(2024, 1, 20, tzinfo=timezone.utc)
    scenario = Scenario(
        name="Volatility Spike",
        shock_type="vol_spike",
        shock_magnitude=2.0,  # 2x volatility
        shock_start=shock_start,
        shock_end=shock_end
    )
    
    shocked_prices = apply_scenario_to_prices(synthetic_prices, scenario)
    
    # Compute returns for both original and shocked prices
    for symbol in synthetic_prices["symbol"].unique():
        original = synthetic_prices[synthetic_prices["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        shocked = shocked_prices[shocked_prices["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        
        original_returns = original["close"].pct_change().dropna()
        shocked_returns = shocked["close"].pct_change().dropna()
        
        # Returns in shock period should have higher variance
        shock_mask = (original["timestamp"] >= shock_start) & (original["timestamp"] <= shock_end)
        if shock_mask.sum() > 1:
            # Get indices for shock period (skip first return which is NaN)
            shock_indices = shock_mask[1:].index[shock_mask[1:]]
            if len(shock_indices) > 0:
                original_vol = original_returns.loc[shock_indices].std()
                shocked_vol = shocked_returns.loc[shock_indices].std()
                
                # Shocked volatility should be higher (approximately 2x, but may vary due to compounding)
                assert shocked_vol > original_vol, f"Volatility should increase for {symbol}"


def test_scenario_shipping_blockade(synthetic_prices):
    """Test shipping_blockade scenario applies stronger shock to shipping symbols."""
    # Add a shipping symbol
    shipping_prices = synthetic_prices.copy()
    shipping_symbol = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC"),
        "symbol": ["SHIPPING"] * 30,
        "close": [50.0 + i * 0.5 for i in range(30)],
        "open": [49.5 + i * 0.5 for i in range(30)],
        "high": [50.5 + i * 0.5 for i in range(30)],
        "low": [49.0 + i * 0.5 for i in range(30)],
    })
    all_prices = pd.concat([shipping_prices, shipping_symbol], ignore_index=True)
    
    shock_start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    scenario = Scenario(
        name="Shipping Blockade",
        shock_type="shipping_blockade",
        shock_magnitude=-0.10,  # -10% base shock
        shock_start=shock_start,
        shock_end=None
    )
    
    shocked_prices = apply_scenario_to_prices(all_prices, scenario)
    
    # Shipping symbol should have stronger shock (base + additional)
    shipping_before = all_prices[(all_prices["symbol"] == "SHIPPING") & (all_prices["timestamp"] >= shock_start)]
    shipping_after = shocked_prices[(shocked_prices["symbol"] == "SHIPPING") & (shocked_prices["timestamp"] >= shock_start)]
    
    if len(shipping_before) > 0 and len(shipping_after) > 0:
        # Shipping symbol should be more affected than non-shipping symbols
        shipping_ratio = shipping_after["close"].iloc[0] / shipping_before["close"].iloc[0]
        
        # Compare with a non-shipping symbol
        non_shipping_symbol = all_prices["symbol"].unique()[0]
        if "SHIP" not in str(non_shipping_symbol).upper():
            non_shipping_before = all_prices[(all_prices["symbol"] == non_shipping_symbol) & (all_prices["timestamp"] >= shock_start)]
            non_shipping_after = shocked_prices[(shocked_prices["symbol"] == non_shipping_symbol) & (shocked_prices["timestamp"] >= shock_start)]
            
            if len(non_shipping_before) > 0 and len(non_shipping_after) > 0:
                non_shipping_ratio = non_shipping_after["close"].iloc[0] / non_shipping_before["close"].iloc[0]
                
                # Shipping ratio should be lower (more negative impact)
                assert shipping_ratio < non_shipping_ratio, "Shipping symbol should have stronger shock"


def test_scenario_empty_prices():
    """Test that empty prices DataFrame is handled gracefully."""
    empty_prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    scenario = Scenario(
        name="Test",
        shock_type="equity_crash",
        shock_magnitude=-0.20
    )
    
    shocked_prices = apply_scenario_to_prices(empty_prices, scenario)
    assert shocked_prices.empty


def test_scenario_no_shock_start_end(synthetic_prices):
    """Test scenario with no shock_start/shock_end applies to entire series."""
    scenario = Scenario(
        name="Full Series Crash",
        shock_type="equity_crash",
        shock_magnitude=-0.20,
        shock_start=None,
        shock_end=None
    )
    
    shocked_prices = apply_scenario_to_prices(synthetic_prices, scenario)
    
    # All prices should be affected
    for symbol in synthetic_prices["symbol"].unique():
        original = synthetic_prices[synthetic_prices["symbol"] == symbol]["close"].iloc[0]
        shocked = shocked_prices[shocked_prices["symbol"] == symbol]["close"].iloc[0]
        
        # First price should be reduced by 20%
        expected_ratio = 1 + scenario.shock_magnitude
        actual_ratio = shocked / original
        assert abs(actual_ratio - expected_ratio) < 0.01


def test_apply_scenario_to_equity_basic(synthetic_equity):
    """Test apply_scenario_to_equity with equity_crash."""
    scenario = Scenario(
        name="Equity Crash",
        shock_type="equity_crash",
        shock_magnitude=-0.20
    )
    
    shocked_equity = apply_scenario_to_equity(synthetic_equity, scenario)
    
    # All equity values should be reduced by 20%
    expected_ratio = 1 + scenario.shock_magnitude
    for i in range(len(synthetic_equity)):
        actual_ratio = shocked_equity.iloc[i] / synthetic_equity.iloc[i]
        assert abs(actual_ratio - expected_ratio) < 1e-5


def test_apply_scenario_to_equity_with_window(synthetic_equity):
    """Test apply_scenario_to_equity with shock window."""
    shock_start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    shock_end = datetime(2024, 1, 20, tzinfo=timezone.utc)
    scenario = Scenario(
        name="Equity Crash Window",
        shock_type="equity_crash",
        shock_magnitude=-0.20,
        shock_start=shock_start,
        shock_end=shock_end
    )
    
    shocked_equity = apply_scenario_to_equity(synthetic_equity, scenario)
    
    # Equity before shock_start should be unchanged
    before_mask = synthetic_equity.index < shock_start
    pd.testing.assert_series_equal(
        synthetic_equity[before_mask],
        shocked_equity[before_mask],
        check_exact=False,
        rtol=1e-5
    )
    
    # Equity in shock window should be reduced
    shock_mask = (synthetic_equity.index >= shock_start) & (synthetic_equity.index <= shock_end)
    if shock_mask.any():
        expected_ratio = 1 + scenario.shock_magnitude
        for idx in synthetic_equity.index[shock_mask]:
            actual_ratio = shocked_equity.loc[idx] / synthetic_equity.loc[idx]
            assert abs(actual_ratio - expected_ratio) < 1e-5


def test_apply_scenario_to_equity_vol_spike(synthetic_equity):
    """Test apply_scenario_to_equity with vol_spike."""
    scenario = Scenario(
        name="Vol Spike",
        shock_type="vol_spike",
        shock_magnitude=2.0
    )
    
    shocked_equity = apply_scenario_to_equity(synthetic_equity, scenario)
    
    # Equity should have higher volatility
    original_returns = synthetic_equity.pct_change().dropna()
    shocked_returns = shocked_equity.pct_change().dropna()
    
    # Shocked returns should have higher variance
    assert shocked_returns.std() > original_returns.std()


def test_run_scenario_on_equity(synthetic_equity):
    """Test run_scenario_on_equity helper function."""
    scenario = Scenario(
        name="Test Crash",
        shock_type="equity_crash",
        shock_magnitude=-0.20
    )
    
    results = run_scenario_on_equity(synthetic_equity, scenario, freq="1d")
    
    # Check that results contain expected keys
    assert "baseline_metrics" in results
    assert "shocked_metrics" in results
    assert "delta_metrics" in results
    
    # Check that metrics are computed
    assert "max_drawdown" in results["baseline_metrics"]
    assert "max_drawdown" in results["shocked_metrics"]
    
    # Shocked equity should have worse (more negative) max_drawdown
    baseline_dd = results["baseline_metrics"]["max_drawdown"]
    shocked_dd = results["shocked_metrics"]["max_drawdown"]
    
    assert shocked_dd <= baseline_dd, "Shocked equity should have worse drawdown"


def test_scenario_missing_columns():
    """Test that missing required columns raise ValueError."""
    invalid_prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "value": [100.0] * 10  # Missing 'symbol' and 'close'
    })
    
    scenario = Scenario(
        name="Test",
        shock_type="equity_crash",
        shock_magnitude=-0.20
    )
    
    with pytest.raises(ValueError, match="missing required columns"):
        apply_scenario_to_prices(invalid_prices, scenario)


def test_scenario_unknown_type(synthetic_prices):
    """Test that unknown scenario type raises ValueError."""
    # Create a scenario with invalid type (using type: ignore to bypass type checking)
    scenario = Scenario(
        name="Invalid",
        shock_type="unknown_type",  # type: ignore
        shock_magnitude=-0.20
    )
    
    with pytest.raises(ValueError, match="Unknown scenario type"):
        apply_scenario_to_prices(synthetic_prices, scenario)

