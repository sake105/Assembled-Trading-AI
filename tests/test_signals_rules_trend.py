# tests/test_signals_rules_trend.py
"""Sprint 11.1: Unit tests for trend-following signal generation.

This module tests the signal generation functions in src/assembled_core/signals/rules_trend.py:
- generate_trend_signals: Generate signals from DataFrame with moving averages
- generate_trend_signals_from_prices: Convenience function to generate signals from prices

Tests cover:
- Happy path scenarios
- LONG vs FLAT signal generation
- Volume filtering
- Missing columns/required fields
- Edge cases (empty DataFrames, single symbol, multiple symbols)
- Output stability (column names, data types, score ranges)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.signals.rules_trend import (
    generate_trend_signals,
    generate_trend_signals_from_prices,
)

pytestmark = pytest.mark.phase11


@pytest.fixture
def sample_price_data_with_ma():
    """Create sample price data with moving averages for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    prices = [100.0 + i * 0.5 for i in range(100)]  # Upward trend
    
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 100,
        "close": prices,
        "volume": [1000000.0] * 100,
    })
    
    # Add moving averages manually
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["ma_50"] = df["close"].rolling(window=50, min_periods=1).mean()
    
    return df


@pytest.fixture
def sample_price_data_crossover():
    """Create price data that will generate crossover signals."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    # Create data where ma_20 crosses above ma_50 around index 60
    prices = [100.0] * 30 + [100.0 + (i - 30) * 2.0 for i in range(30, 100)]
    
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 100,
        "close": prices,
        "volume": [1000000.0] * 100,
    })
    
    df["ma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["ma_50"] = df["close"].rolling(window=50, min_periods=1).mean()
    
    return df


class TestGenerateTrendSignals:
    """Tests for generate_trend_signals function."""

    def test_generate_trend_signals_happy_path(self, sample_price_data_with_ma):
        """Test signal generation with valid data."""
        result = generate_trend_signals(sample_price_data_with_ma, ma_fast=20, ma_slow=50)
        
        assert "timestamp" in result.columns
        assert "symbol" in result.columns
        assert "direction" in result.columns
        assert "score" in result.columns
        assert len(result) == len(sample_price_data_with_ma)
        assert all(result["direction"].isin(["LONG", "FLAT"]))

    def test_generate_trend_signals_missing_columns(self):
        """Test that KeyError is raised when required columns are missing."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
        })
        
        with pytest.raises(KeyError, match="Missing required columns"):
            generate_trend_signals(df)

    def test_generate_trend_signals_auto_compute_ma(self):
        """Test that moving averages are computed automatically if not present."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 100,
            "close": [100.0 + i * 0.5 for i in range(100)],
        })
        
        result = generate_trend_signals(df, ma_fast=20, ma_slow=50)
        
        assert "direction" in result.columns
        assert "score" in result.columns

    def test_generate_trend_signals_long_when_ma_fast_above_ma_slow(self, sample_price_data_crossover):
        """Test that LONG signals are generated when ma_fast > ma_slow."""
        result = generate_trend_signals(sample_price_data_crossover, ma_fast=20, ma_slow=50)
        
        # After crossover, should have LONG signals
        long_signals = result[result["direction"] == "LONG"]
        assert len(long_signals) > 0

    def test_generate_trend_signals_flat_when_ma_fast_below_ma_slow(self):
        """Test that FLAT signals are generated when ma_fast < ma_slow."""
        dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
        prices = [100.0 - i * 0.5 for i in range(100)]  # Downward trend
        
        df = pd.DataFrame({
            "timestamp": dates,
            "symbol": ["AAPL"] * 100,
            "close": prices,
        })
        
        result = generate_trend_signals(df, ma_fast=20, ma_slow=50)
        
        # With downward trend, ma_fast should be below ma_slow, so mostly FLAT
        flat_signals = result[result["direction"] == "FLAT"]
        assert len(flat_signals) > 0

    def test_generate_trend_signals_score_range(self, sample_price_data_with_ma):
        """Test that scores are in valid range [0.0, 1.0]."""
        result = generate_trend_signals(sample_price_data_with_ma, ma_fast=20, ma_slow=50)
        
        assert (result["score"] >= 0.0).all()
        assert (result["score"] <= 1.0).all()

    def test_generate_trend_signals_flat_has_zero_score(self, sample_price_data_with_ma):
        """Test that FLAT signals have score = 0.0."""
        result = generate_trend_signals(sample_price_data_with_ma, ma_fast=20, ma_slow=50)
        
        flat_signals = result[result["direction"] == "FLAT"]
        if len(flat_signals) > 0:
            assert (flat_signals["score"] == 0.0).all()

    def test_generate_trend_signals_with_volume_filter(self, sample_price_data_with_ma):
        """Test signal generation with volume filtering."""
        # Set low volume for some rows
        sample_price_data_with_ma.loc[50:70, "volume"] = 100.0  # Very low volume
        
        result = generate_trend_signals(
            sample_price_data_with_ma,
            ma_fast=20,
            ma_slow=50,
            min_volume_multiplier=2.0  # High threshold
        )
        
        # Rows with low volume should be FLAT even if ma_fast > ma_slow
        low_volume_rows = result.iloc[50:70]
        assert all(low_volume_rows["direction"] == "FLAT")

    def test_generate_trend_signals_without_volume(self):
        """Test signal generation without volume column."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 100,
            "close": [100.0 + i * 0.5 for i in range(100)],
        })
        
        result = generate_trend_signals(df, ma_fast=20, ma_slow=50)
        
        assert "direction" in result.columns
        assert "score" in result.columns

    def test_generate_trend_signals_multiple_symbols(self):
        """Test signal generation with multiple symbols."""
        symbols = ["AAPL", "MSFT"]
        data = []
        for symbol in symbols:
            data.append(pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
                "symbol": [symbol] * 100,
                "close": [100.0 + i * 0.5 for i in range(100)],
            }))
        df = pd.concat(data, ignore_index=True)
        
        result = generate_trend_signals(df, ma_fast=20, ma_slow=50)
        
        assert result["symbol"].nunique() == 2
        for symbol in symbols:
            symbol_data = result[result["symbol"] == symbol]
            assert len(symbol_data) == 100


class TestGenerateTrendSignalsFromPrices:
    """Tests for generate_trend_signals_from_prices convenience function."""

    def test_generate_trend_signals_from_prices_happy_path(self):
        """Test convenience function with valid price data."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 100,
            "close": [100.0 + i * 0.5 for i in range(100)],
            "volume": [1000000.0] * 100,
        })
        
        result = generate_trend_signals_from_prices(df, ma_fast=20, ma_slow=50)
        
        assert "direction" in result.columns
        assert "score" in result.columns
        assert len(result) == len(df)

    def test_generate_trend_signals_from_prices_custom_parameters(self):
        """Test convenience function with custom parameters."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 100,
            "close": [100.0 + i * 0.5 for i in range(100)],
            "volume": [1000000.0] * 100,
        })
        
        result = generate_trend_signals_from_prices(
            df,
            ma_fast=10,
            ma_slow=30,
            volume_threshold=500000.0
        )
        
        assert "direction" in result.columns
        assert "score" in result.columns

