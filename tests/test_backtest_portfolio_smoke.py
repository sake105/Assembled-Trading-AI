# tests/test_backtest_portfolio_smoke.py
"""Smoke tests for backtest and portfolio simulation with synthetic data."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.backtest import simulate_equity
from src.assembled_core.pipeline.portfolio import simulate_with_costs


def create_synthetic_prices() -> pd.DataFrame:
    """Create synthetic price data for testing."""
    base_time = datetime(2025, 11, 28, 14, 0, 0)
    symbols = ["AAPL"]
    data = []
    
    for sym in symbols:
        for i in range(20):
            ts = base_time + timedelta(minutes=5 * i)
            price = 100.0 + i * 0.1  # Slight upward trend
            data.append({
                "timestamp": ts,
                "symbol": sym,
                "close": price
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def create_synthetic_orders() -> pd.DataFrame:
    """Create synthetic orders for testing."""
    base_time = datetime(2025, 11, 28, 14, 0, 0)
    data = [
        {
            "timestamp": base_time + timedelta(minutes=5),
            "symbol": "AAPL",
            "side": "BUY",
            "qty": 1.0,
            "price": 100.5
        },
        {
            "timestamp": base_time + timedelta(minutes=15),
            "symbol": "AAPL",
            "side": "SELL",
            "qty": 1.0,
            "price": 101.5
        }
    ]
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def test_simulate_equity_basic():
    """Test basic equity simulation."""
    prices = create_synthetic_prices()
    orders = create_synthetic_orders()
    start_capital = 10000.0
    
    equity = simulate_equity(prices, orders, start_capital)
    
    assert isinstance(equity, pd.DataFrame)
    assert not equity.empty, "Equity DataFrame should not be empty"
    assert "timestamp" in equity.columns
    assert "equity" in equity.columns
    
    # Equity curve should have same length as price timeline (unique timestamps)
    unique_price_timestamps = prices["timestamp"].unique()
    assert len(equity) == len(unique_price_timestamps), \
        f"Equity length ({len(equity)}) should match price timeline length ({len(unique_price_timestamps)})"
    
    # Equity should always be positive
    assert (equity["equity"] > 0).all(), "Equity should always be positive"
    
    # First equity should equal start capital
    assert abs(equity["equity"].iloc[0] - start_capital) < 1e-6, \
        f"First equity should equal start capital ({start_capital})"


def test_simulate_equity_with_trades():
    """Test equity simulation with actual trades."""
    prices = create_synthetic_prices()
    orders = create_synthetic_orders()
    start_capital = 10000.0
    
    equity = simulate_equity(prices, orders, start_capital)
    
    # After BUY at 100.5, cash should decrease
    # After SELL at 101.5, cash should increase
    # Final equity should reflect the profit/loss
    
    # Check that equity changes after trades
    equity_values = equity["equity"].values
    assert len(set(equity_values)) > 1, "Equity should change over time with trades"


def test_simulate_with_costs_basic():
    """Test portfolio simulation with costs."""
    orders = create_synthetic_orders()
    start_capital = 10000.0
    commission_bps = 0.0
    spread_w = 0.25
    impact_w = 0.5
    freq = "5min"
    
    equity, metrics = simulate_with_costs(
        orders, start_capital, commission_bps, spread_w, impact_w, freq
    )
    
    assert isinstance(equity, pd.DataFrame)
    assert not equity.empty, "Equity DataFrame should not be empty"
    assert "timestamp" in equity.columns
    assert "equity" in equity.columns
    
    # Equity should always be positive
    assert (equity["equity"] > 0).all(), "Equity should always be positive"
    
    # Check metrics
    assert isinstance(metrics, dict)
    assert "final_pf" in metrics
    assert "sharpe" in metrics
    assert "trades" in metrics
    
    assert metrics["trades"] == len(orders), "Trades count should match orders count"
    assert metrics["final_pf"] > 0, "Final PF should be positive"


def test_simulate_with_costs_equity_positive():
    """Test that portfolio simulation always produces positive equity."""
    orders = create_synthetic_orders()
    start_capital = 10000.0
    
    # Test with different cost parameters
    for commission_bps in [0.0, 0.5, 1.0]:
        for spread_w in [0.0, 0.25, 0.5]:
            for impact_w in [0.0, 0.5, 1.0]:
                equity, metrics = simulate_with_costs(
                    orders, start_capital, commission_bps, spread_w, impact_w, "5min"
                )
                
                assert (equity["equity"] > 0).all(), \
                    f"Equity should be positive with costs: comm={commission_bps}, spread={spread_w}, impact={impact_w}"

