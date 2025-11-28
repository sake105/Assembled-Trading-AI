# tests/test_signals_ema.py
"""Tests for EMA signal generation with synthetic data."""
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

from src.assembled_core.pipeline.signals import compute_ema_signals


def create_synthetic_prices() -> pd.DataFrame:
    """Create a small synthetic price DataFrame for testing.
    
    Returns:
        DataFrame with columns: timestamp, symbol, close
        Two symbols (AAPL, MSFT) with 10 timestamps each
    """
    base_time = datetime(2025, 11, 28, 14, 0, 0)
    symbols = ["AAPL", "MSFT"]
    data = []
    
    for sym in symbols:
        # Create a simple trend: AAPL goes up, MSFT goes down
        base_price = 100.0 if sym == "AAPL" else 200.0
        for i in range(10):
            ts = base_time + timedelta(minutes=5 * i)
            # Trend: AAPL increases, MSFT decreases
            if sym == "AAPL":
                price = base_price + i * 0.5  # Upward trend
            else:
                price = base_price - i * 0.3  # Downward trend
            data.append({
                "timestamp": ts,
                "symbol": sym,
                "close": price
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_compute_ema_signals_produces_signals():
    """Test that EMA signal generation produces at least one signal."""
    prices = create_synthetic_prices()
    
    # Use small EMA periods for quick crossover
    fast = 2
    slow = 4
    
    signals = compute_ema_signals(prices, fast, slow)
    
    assert isinstance(signals, pd.DataFrame)
    assert not signals.empty, "Signals DataFrame should not be empty"
    assert "timestamp" in signals.columns
    assert "symbol" in signals.columns
    assert "sig" in signals.columns
    assert "price" in signals.columns
    
    # Check that we have signals for both symbols
    assert signals["symbol"].nunique() == 2, "Should have signals for both symbols"
    
    # Check that we have at least one BUY or SELL signal (sig != 0)
    non_zero_signals = signals[signals["sig"] != 0]
    assert len(non_zero_signals) > 0, "Should have at least one non-zero signal"
    
    # Check signal values are valid (-1, 0, or +1)
    assert signals["sig"].isin([-1, 0, 1]).all(), "All signals should be -1, 0, or +1"


def test_compute_ema_signals_no_nans():
    """Test that EMA signals contain no NaNs in numeric columns."""
    prices = create_synthetic_prices()
    
    fast = 2
    slow = 4
    
    signals = compute_ema_signals(prices, fast, slow)
    
    # Check for NaNs in numeric columns
    assert not signals["price"].isna().any(), "Price column should not contain NaNs"
    assert not signals["sig"].isna().any(), "Signal column should not contain NaNs"
    
    # Timestamp should also not be NaT
    assert not signals["timestamp"].isna().any(), "Timestamp column should not contain NaTs"


def test_compute_ema_signals_crossovers():
    """Test that EMA signals detect crossovers correctly."""
    # Create a price series that will definitely cause a crossover
    base_time = datetime(2025, 11, 28, 14, 0, 0)
    data = []
    
    # Create prices that go from low to high (will cause fast EMA to cross above slow EMA)
    for i in range(10):
        ts = base_time + timedelta(minutes=5 * i)
        price = 100.0 + i * 2.0  # Strong upward trend
        data.append({
            "timestamp": ts,
            "symbol": "TEST",
            "close": price
        })
    
    prices = pd.DataFrame(data)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    
    fast = 2
    slow = 4
    
    signals = compute_ema_signals(prices, fast, slow)
    
    # With strong upward trend, we should eventually get a BUY signal (sig = +1)
    buy_signals = signals[signals["sig"] == 1]
    assert len(buy_signals) > 0, "Should have at least one BUY signal with upward trend"

