"""Smoke tests for unified trading cycle implementation (B1).

These tests verify that run_trading_cycle works end-to-end with real
existing modules, producing deterministic orders.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle
from src.assembled_core.portfolio.position_sizing import (
    compute_target_positions_from_trend_signals,
)
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices


def test_eod_cycle_returns_orders_deterministic() -> None:
    """Test that EOD cycle returns orders deterministically."""
    # Create synthetic price data (2 symbols, 20 days)
    dates = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 2,
        "symbol": ["AAPL"] * 20 + ["MSFT"] * 20,
        "close": [100.0 + i * 0.5 for i in range(20)] + [200.0 + i * 0.3 for i in range(20)],
        "high": [101.0 + i * 0.5 for i in range(20)] + [201.0 + i * 0.3 for i in range(20)],
        "low": [99.0 + i * 0.5 for i in range(20)] + [199.0 + i * 0.3 for i in range(20)],
        "open": [100.5 + i * 0.5 for i in range(20)] + [200.5 + i * 0.3 for i in range(20)],
        "volume": [1000000.0] * 40,
    })
    
    # Set as_of to last date (PIT-safe)
    as_of = dates[-1]
    
    # Define signal function (trend signals)
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    # Define position sizing function
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals,
            total_capital=capital,
            top_n=None,
            min_score=0.0,
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=as_of,
        universe=["AAPL", "MSFT"],
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,  # Disable risk controls for deterministic test
        write_outputs=False,  # Pure function, no file writes
    )
    
    result = run_trading_cycle(ctx)
    
    # Verify success
    assert result.status == "success"
    assert result.error_message is None
    
    # Verify prices filtered
    assert not result.prices_filtered.empty
    assert result.prices_filtered["symbol"].nunique() == 2
    # Should have one row per symbol (last available <= as_of)
    assert len(result.prices_filtered) == 2
    
    # Verify features built
    assert not result.prices_with_features.empty
    # Should have more columns than input (features added)
    assert len(result.prices_with_features.columns) > len(result.prices_filtered.columns)
    
    # Verify signals generated
    assert not result.signals.empty
    assert "timestamp" in result.signals.columns
    assert "symbol" in result.signals.columns
    assert "direction" in result.signals.columns
    
    # Verify target positions computed
    assert not result.target_positions.empty
    assert "symbol" in result.target_positions.columns
    assert "target_qty" in result.target_positions.columns
    
    # Verify orders generated
    assert not result.orders.empty
    assert "timestamp" in result.orders.columns
    assert "symbol" in result.orders.columns
    assert "side" in result.orders.columns
    assert "qty" in result.orders.columns
    assert "price" in result.orders.columns
    
    # Verify orders are deterministic (same inputs -> same outputs)
    result2 = run_trading_cycle(ctx)
    assert result2.status == "success"
    pd.testing.assert_frame_equal(result.orders, result2.orders, check_dtype=False)
    
    # Verify orders have prices filled
    assert (result.orders["price"] > 0.0).all()


def test_eod_cycle_pit_safe_filtering() -> None:
    """Test that PIT-safe filtering works correctly."""
    # Create price data with future dates
    dates = pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC")
    
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 2,
        "symbol": ["AAPL"] * 30 + ["MSFT"] * 30,
        "close": [100.0 + i * 0.5 for i in range(30)] + [200.0 + i * 0.3 for i in range(30)],
        "high": [101.0 + i * 0.5 for i in range(30)] + [201.0 + i * 0.3 for i in range(30)],
        "low": [99.0 + i * 0.5 for i in range(30)] + [199.0 + i * 0.3 for i in range(30)],
        "open": [100.5 + i * 0.5 for i in range(30)] + [200.5 + i * 0.3 for i in range(30)],
        "volume": [1000000.0] * 60,
    })
    
    # Set as_of to day 20 (should filter out days 21-30)
    as_of = dates[19]  # Day 20 (0-indexed)
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=as_of,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    
    assert result.status == "success"
    
    # Verify all filtered prices are <= as_of
    assert (result.prices_filtered["timestamp"] <= as_of).all()
    
    # Verify we have one row per symbol (last available <= as_of)
    assert len(result.prices_filtered) == 2
    assert result.prices_filtered["timestamp"].max() <= as_of


def test_eod_cycle_universe_filtering() -> None:
    """Test that universe filtering works correctly."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 3,
        "symbol": ["AAPL"] * 20 + ["MSFT"] * 20 + ["GOOGL"] * 20,
        "close": [100.0] * 60,
        "high": [101.0] * 60,
        "low": [99.0] * 60,
        "open": [100.5] * 60,
        "volume": [1000000.0] * 60,
    })
    
    # Filter to only AAPL and MSFT
    universe = ["AAPL", "MSFT"]
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=dates[-1],
        universe=universe,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    
    assert result.status == "success"
    
    # Verify only universe symbols are present
    symbols_in_result = set(result.prices_filtered["symbol"].unique())
    assert symbols_in_result == {"AAPL", "MSFT"}
    assert "GOOGL" not in symbols_in_result


def test_eod_cycle_factor_store_integration() -> None:
    """Test that factor store integration works (if enabled)."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 2,
        "symbol": ["AAPL"] * 20 + ["MSFT"] * 20,
        "close": [100.0 + i * 0.5 for i in range(20)] + [200.0 + i * 0.3 for i in range(20)],
        "high": [101.0 + i * 0.5 for i in range(20)] + [201.0 + i * 0.3 for i in range(20)],
        "low": [99.0 + i * 0.5 for i in range(20)] + [199.0 + i * 0.3 for i in range(20)],
        "open": [100.5 + i * 0.5 for i in range(20)] + [200.5 + i * 0.3 for i in range(20)],
        "volume": [1000000.0] * 40,
    })
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    # Test with factor store enabled (but may not have cache, so will compute)
    ctx = TradingContext(
        prices=prices,
        as_of=dates[-1],
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        use_factor_store=True,
        factor_group="core_ta",
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    
    # Should still work (factor store will compute if cache miss)
    assert result.status == "success"
    assert not result.prices_with_features.empty


def test_eod_cycle_with_current_positions() -> None:
    """Test that order generation handles current positions correctly."""
    dates = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 2,
        "symbol": ["AAPL"] * 20 + ["MSFT"] * 20,
        "close": [100.0 + i * 0.5 for i in range(20)] + [200.0 + i * 0.3 for i in range(20)],
        "high": [101.0 + i * 0.5 for i in range(20)] + [201.0 + i * 0.3 for i in range(20)],
        "low": [99.0 + i * 0.5 for i in range(20)] + [199.0 + i * 0.3 for i in range(20)],
        "open": [100.5 + i * 0.5 for i in range(20)] + [200.5 + i * 0.3 for i in range(20)],
        "volume": [1000000.0] * 40,
    })
    
    # Current positions: already have some AAPL
    current_positions = pd.DataFrame({
        "symbol": ["AAPL"],
        "qty": [10.0],
    })
    
    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        return generate_trend_signals_from_prices(df, ma_fast=5, ma_slow=10)
    
    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return compute_target_positions_from_trend_signals(
            signals, total_capital=capital, top_n=None, min_score=0.0
        )
    
    ctx = TradingContext(
        prices=prices,
        as_of=dates[-1],
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=10000.0,
        current_positions=current_positions,
        enable_risk_controls=False,
        write_outputs=False,
    )
    
    result = run_trading_cycle(ctx)
    
    assert result.status == "success"
    assert not result.orders.empty
    
    # Orders should account for current positions (may reduce or increase AAPL position)
    assert "AAPL" in result.orders["symbol"].values or len(result.orders) == 0

