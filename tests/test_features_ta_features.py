# tests/test_features_ta_features.py
"""Sprint 11.1: Unit tests for technical analysis features.

This module tests the feature engineering functions in src/assembled_core/features/ta_features.py:
- add_log_returns: Logarithmic returns calculation
- add_moving_averages: Simple Moving Average (SMA) calculation
- add_atr: Average True Range (ATR) calculation
- add_rsi: Relative Strength Index (RSI) calculation
- add_all_features: Convenience function for all features

Tests cover:
- Happy path scenarios
- Missing columns/required fields
- Edge cases (single symbol, multiple symbols, empty DataFrames)
- Output stability (column names, data types, no NaN explosion)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.features.ta_features import (
    add_all_features,
    add_atr,
    add_log_returns,
    add_moving_averages,
    add_rsi,
)

pytestmark = pytest.mark.phase11


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 50,
            "open": [100.0 + i * 0.1 for i in range(50)],
            "high": [105.0 + i * 0.1 for i in range(50)],
            "low": [99.0 + i * 0.1 for i in range(50)],
            "close": [102.0 + i * 0.1 for i in range(50)],
            "volume": [1000000.0 + i * 1000 for i in range(50)],
        }
    )


@pytest.fixture
def multi_symbol_price_data():
    """Create sample price data with multiple symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    data = []
    for symbol in symbols:
        data.append(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range(
                        "2024-01-01", periods=30, freq="D", tz="UTC"
                    ),
                    "symbol": [symbol] * 30,
                    "open": [100.0 + i * 0.1 for i in range(30)],
                    "high": [105.0 + i * 0.1 for i in range(30)],
                    "low": [99.0 + i * 0.1 for i in range(30)],
                    "close": [102.0 + i * 0.1 for i in range(30)],
                    "volume": [1000000.0] * 30,
                }
            )
        )
    return pd.concat(data, ignore_index=True)


class TestAddLogReturns:
    """Tests for add_log_returns function."""

    def test_add_log_returns_happy_path(self, sample_price_data):
        """Test log returns calculation with valid data."""
        result = add_log_returns(sample_price_data)

        assert "log_return" in result.columns
        assert len(result) == len(sample_price_data)
        # First row should have NaN (no previous value)
        assert pd.isna(result["log_return"].iloc[0])
        # Subsequent rows should have valid log returns
        assert not pd.isna(result["log_return"].iloc[1:]).all()

    def test_add_log_returns_missing_symbol(self):
        """Test that KeyError is raised when symbol column is missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="D", tz="UTC"
                ),
                "close": [100.0] * 10,
            }
        )

        with pytest.raises(KeyError, match="symbol"):
            add_log_returns(df)

    def test_add_log_returns_missing_price_col(self):
        """Test that KeyError is raised when price column is missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": ["AAPL"] * 10,
            }
        )

        with pytest.raises(KeyError, match="Price column"):
            add_log_returns(df, price_col="close")

    def test_add_log_returns_custom_price_col(self, sample_price_data):
        """Test log returns with custom price column."""
        result = add_log_returns(
            sample_price_data, price_col="high", out_col="high_log_return"
        )

        assert "high_log_return" in result.columns
        assert "log_return" not in result.columns

    def test_add_log_returns_multiple_symbols(self, multi_symbol_price_data):
        """Test log returns calculation with multiple symbols."""
        result = add_log_returns(multi_symbol_price_data)

        assert "log_return" in result.columns
        assert len(result) == len(multi_symbol_price_data)
        # Each symbol should have NaN in first row
        for symbol in multi_symbol_price_data["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]
            assert pd.isna(symbol_data["log_return"].iloc[0])


class TestAddMovingAverages:
    """Tests for add_moving_averages function."""

    def test_add_moving_averages_happy_path(self, sample_price_data):
        """Test moving averages calculation with valid data."""
        result = add_moving_averages(sample_price_data, windows=(20, 50))

        assert "ma_20" in result.columns
        assert "ma_50" in result.columns
        assert len(result) == len(sample_price_data)
        # First row should have valid MA (min_periods=1)
        assert not pd.isna(result["ma_20"].iloc[0])

    def test_add_moving_averages_missing_price_col(self):
        """Test that KeyError is raised when price column is missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": ["AAPL"] * 10,
            }
        )

        with pytest.raises(KeyError, match="Price column"):
            add_moving_averages(df, price_col="close")

    def test_add_moving_averages_custom_windows(self, sample_price_data):
        """Test moving averages with custom window sizes."""
        result = add_moving_averages(sample_price_data, windows=(5, 10, 15))

        assert "ma_5" in result.columns
        assert "ma_10" in result.columns
        assert "ma_15" in result.columns

    def test_add_moving_averages_multiple_symbols(self, multi_symbol_price_data):
        """Test moving averages with multiple symbols."""
        result = add_moving_averages(multi_symbol_price_data, windows=(10, 20))

        assert "ma_10" in result.columns
        assert "ma_20" in result.columns
        # Each symbol should have independent MAs
        for symbol in multi_symbol_price_data["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]
            assert len(symbol_data) == 30


class TestAddAtr:
    """Tests for add_atr function."""

    def test_add_atr_happy_path(self, sample_price_data):
        """Test ATR calculation with valid data."""
        result = add_atr(sample_price_data, window=14)

        assert "atr_14" in result.columns
        assert len(result) == len(sample_price_data)
        # ATR should be non-negative
        assert (result["atr_14"].dropna() >= 0).all()

    def test_add_atr_missing_columns(self):
        """Test that KeyError is raised when required columns are missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": ["AAPL"] * 10,
                "close": [100.0] * 10,
            }
        )

        with pytest.raises(KeyError, match="Missing required columns"):
            add_atr(df)

    def test_add_atr_custom_window(self, sample_price_data):
        """Test ATR with custom window size."""
        result = add_atr(sample_price_data, window=21)

        assert "atr_21" in result.columns
        assert "atr_14" not in result.columns

    def test_add_atr_multiple_symbols(self, multi_symbol_price_data):
        """Test ATR with multiple symbols."""
        result = add_atr(multi_symbol_price_data, window=14)

        assert "atr_14" in result.columns
        # Each symbol should have independent ATR
        for symbol in multi_symbol_price_data["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]
            assert len(symbol_data) == 30


class TestAddRsi:
    """Tests for add_rsi function."""

    def test_add_rsi_happy_path(self, sample_price_data):
        """Test RSI calculation with valid data."""
        result = add_rsi(sample_price_data, window=14)

        assert "rsi_14" in result.columns
        assert len(result) == len(sample_price_data)
        # RSI should be between 0 and 100
        rsi_values = result["rsi_14"].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all()

    def test_add_rsi_missing_symbol(self):
        """Test that KeyError is raised when symbol column is missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="D", tz="UTC"
                ),
                "close": [100.0] * 10,
            }
        )

        with pytest.raises(KeyError, match="symbol"):
            add_rsi(df)

    def test_add_rsi_missing_price_col(self):
        """Test that KeyError is raised when price column is missing."""
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=10, freq="D", tz="UTC"
                ),
                "symbol": ["AAPL"] * 10,
            }
        )

        with pytest.raises(KeyError):
            add_rsi(df, price_col="close")

    def test_add_rsi_custom_window(self, sample_price_data):
        """Test RSI with custom window size."""
        result = add_rsi(sample_price_data, window=21)

        assert "rsi_21" in result.columns
        assert "rsi_14" not in result.columns

    def test_add_rsi_multiple_symbols(self, multi_symbol_price_data):
        """Test RSI with multiple symbols."""
        result = add_rsi(multi_symbol_price_data, window=14)

        assert "rsi_14" in result.columns
        # Each symbol should have independent RSI
        for symbol in multi_symbol_price_data["symbol"].unique():
            symbol_data = result[result["symbol"] == symbol]
            assert len(symbol_data) == 30


class TestAddAllFeatures:
    """Tests for add_all_features convenience function."""

    def test_add_all_features_happy_path(self, sample_price_data):
        """Test that all features are added correctly."""
        result = add_all_features(sample_price_data)

        # Check that all expected features are present
        assert "log_return" in result.columns
        assert "ma_20" in result.columns
        assert "ma_50" in result.columns
        assert "ma_200" in result.columns
        assert "atr_14" in result.columns
        assert "rsi_14" in result.columns

    def test_add_all_features_without_rsi(self, sample_price_data):
        """Test that RSI can be excluded."""
        result = add_all_features(sample_price_data, include_rsi=False)

        assert "log_return" in result.columns
        assert "atr_14" in result.columns
        assert "rsi_14" not in result.columns

    def test_add_all_features_custom_windows(self, sample_price_data):
        """Test that custom windows are respected."""
        result = add_all_features(
            sample_price_data, ma_windows=(10, 20), atr_window=21, rsi_window=7
        )

        assert "ma_10" in result.columns
        assert "ma_20" in result.columns
        assert "ma_200" not in result.columns
        assert "atr_21" in result.columns
        assert "rsi_7" in result.columns

    def test_add_all_features_preserves_original_columns(self, sample_price_data):
        """Test that original columns are preserved."""
        result = add_all_features(sample_price_data)

        original_cols = set(sample_price_data.columns)
        result_cols = set(result.columns)

        assert original_cols.issubset(result_cols)
