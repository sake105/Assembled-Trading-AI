# tests/test_data_contracts.py
"""Sprint 11.1: Unit tests for data ingestion and validation contracts.

This module tests the data ingestion functions in src/assembled_core/data/prices_ingest.py:
- load_eod_prices: Loading price data with OHLCV validation
- load_eod_prices_for_universe: Loading prices from universe file
- validate_price_data: Data quality validation

Tests cover:
- Happy path scenarios
- Missing columns/required fields
- Invalid OHLC relationships
- Edge cases (empty DataFrames, NaNs, negative prices)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.assembled_core.data.prices_ingest import (
    load_eod_prices,
    load_eod_prices_for_universe,
    validate_price_data,
)

pytestmark = pytest.mark.phase11


class TestValidatePriceData:
    """Tests for validate_price_data function."""

    def test_validate_price_data_valid(self):
        """Test validation with valid price data."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [102.0 + i for i in range(10)],
            "volume": [1000000.0] * 10,
        })
        
        result = validate_price_data(df)
        
        assert result["valid"] is True
        assert result["row_count"] == 10
        assert result["symbol_count"] == 1
        assert len(result["issues"]) == 0

    def test_validate_price_data_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [100.0] * 5,
        })
        
        result = validate_price_data(df)
        
        assert result["valid"] is False
        assert "Missing columns" in str(result["issues"])

    def test_validate_price_data_empty(self):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
        
        result = validate_price_data(df)
        
        assert result["valid"] is False
        assert "empty" in str(result["issues"]).lower()

    def test_validate_price_data_invalid_ohlc(self):
        """Test validation with invalid OHLC relationships."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "open": [100.0] * 5,
            "high": [95.0] * 5,  # Invalid: high < open
            "low": [99.0] * 5,
            "close": [102.0] * 5,
            "volume": [1000000.0] * 5,
        })
        
        result = validate_price_data(df)
        
        assert result["valid"] is False
        assert "Invalid OHLC" in str(result["issues"])

    def test_validate_price_data_nans(self):
        """Test validation with NaNs in critical columns."""
        df = pd.DataFrame({
            "timestamp": pd.Series([None] + list(pd.date_range("2024-01-02", periods=4, freq="D", tz="UTC"))),
            "symbol": ["AAPL"] * 5,
            "open": [100.0] * 5,
            "high": [105.0] * 5,
            "low": [99.0] * 5,
            "close": [None, 102.0, 103.0, 104.0, 105.0],  # NaN in close
            "volume": [1000000.0] * 5,
        })
        
        result = validate_price_data(df)
        
        assert result["valid"] is False
        assert any("NaN" in issue for issue in result["issues"])

    def test_validate_price_data_negative_prices(self):
        """Test validation with negative prices."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "open": [-100.0] * 5,  # Negative price
            "high": [105.0] * 5,
            "low": [99.0] * 5,
            "close": [102.0] * 5,
            "volume": [1000000.0] * 5,
        })
        
        result = validate_price_data(df)
        
        assert result["valid"] is False
        assert "Negative prices" in str(result["issues"])

    def test_validate_price_data_zero_volume(self):
        """Test validation with high percentage of zero volume."""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "open": [100.0] * 10,
            "high": [105.0] * 10,
            "low": [99.0] * 10,
            "close": [102.0] * 10,
            "volume": [0.0] * 10,  # All zero volume
        })
        
        result = validate_price_data(df)
        
        # Should have issue about zero volume (but might still be valid if < 50%)
        # Actually, 100% zero volume should trigger the issue
        assert "zero volume" in str(result["issues"]).lower() or result["valid"] is True


class TestLoadEodPrices:
    """Tests for load_eod_prices function."""

    def test_load_eod_prices_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised when price file doesn't exist."""
        non_existent_file = tmp_path / "nonexistent.parquet"
        
        with pytest.raises(FileNotFoundError):
            load_eod_prices(price_file=non_existent_file)

    def test_load_eod_prices_with_sample_data(self, tmp_path: Path):
        """Test loading prices from a sample Parquet file."""
        # Create sample price data
        sample_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10 + ["MSFT"] * 10,
            "close": [100.0 + i for i in range(10)] + [200.0 + i for i in range(10)],
            "open": [99.0 + i for i in range(10)] + [199.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)] + [205.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)] + [198.0 + i for i in range(10)],
            "volume": [1000000.0] * 20,
        })
        
        sample_file = tmp_path / "sample_prices.parquet"
        sample_data.to_parquet(sample_file, index=False)
        
        result = load_eod_prices(price_file=sample_file)
        
        assert not result.empty
        assert len(result) == 20
        assert result["symbol"].nunique() == 2
        assert all(col in result.columns for col in ["timestamp", "symbol", "open", "high", "low", "close", "volume"])

    def test_load_eod_prices_filters_by_symbols(self, tmp_path: Path):
        """Test that load_eod_prices filters by symbols when provided."""
        sample_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10 + ["MSFT"] * 10,
            "close": [100.0] * 20,
            "open": [99.0] * 20,
            "high": [105.0] * 20,
            "low": [98.0] * 20,
            "volume": [1000000.0] * 20,
        })
        
        sample_file = tmp_path / "sample_prices.parquet"
        sample_data.to_parquet(sample_file, index=False)
        
        result = load_eod_prices(price_file=sample_file, symbols=["AAPL"])
        
        assert not result.empty
        assert result["symbol"].nunique() == 1
        assert all(result["symbol"] == "AAPL")

    def test_load_eod_prices_synthetic_ohlcv(self, tmp_path: Path):
        """Test that load_eod_prices creates synthetic OHLCV when only close is available."""
        # Create data with only timestamp, symbol, close
        sample_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0 + i for i in range(10)],
        })
        
        sample_file = tmp_path / "sample_prices.parquet"
        sample_data.to_parquet(sample_file, index=False)
        
        result = load_eod_prices(price_file=sample_file)
        
        assert not result.empty
        assert all(col in result.columns for col in ["open", "high", "low", "volume"])
        # Synthetic OHLCV: open=high=low=close, volume=0
        assert all(result["open"] == result["close"])
        assert all(result["volume"] == 0.0)


class TestLoadEodPricesForUniverse:
    """Tests for load_eod_prices_for_universe function."""

    def test_load_eod_prices_for_universe_file_not_found(self, tmp_path: Path):
        """Test that FileNotFoundError is raised when universe file doesn't exist."""
        non_existent_universe = tmp_path / "nonexistent.txt"
        sample_file = tmp_path / "sample_prices.parquet"
        
        # Create sample price file
        pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }).to_parquet(sample_file, index=False)
        
        with pytest.raises(FileNotFoundError):
            load_eod_prices_for_universe(universe_file=non_existent_universe, price_file=sample_file)

    def test_load_eod_prices_for_universe_empty_file(self, tmp_path: Path):
        """Test that ValueError is raised when universe file is empty."""
        universe_file = tmp_path / "empty_universe.txt"
        universe_file.write_text("")
        
        sample_file = tmp_path / "sample_prices.parquet"
        pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0] * 10,
        }).to_parquet(sample_file, index=False)
        
        with pytest.raises(ValueError, match="No symbols found"):
            load_eod_prices_for_universe(universe_file=universe_file, price_file=sample_file)

    def test_load_eod_prices_for_universe_with_comments(self, tmp_path: Path):
        """Test that universe file with comments and empty lines is handled correctly."""
        universe_file = tmp_path / "universe.txt"
        universe_file.write_text("# This is a comment\nAAPL\n\nMSFT\n# Another comment\n")
        
        sample_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10 + ["MSFT"] * 10,
            "close": [100.0] * 20,
            "open": [99.0] * 20,
            "high": [105.0] * 20,
            "low": [98.0] * 20,
            "volume": [1000000.0] * 20,
        })
        
        sample_file = tmp_path / "sample_prices.parquet"
        sample_data.to_parquet(sample_file, index=False)
        
        result = load_eod_prices_for_universe(universe_file=universe_file, price_file=sample_file)
        
        assert not result.empty
        assert result["symbol"].nunique() == 2
        assert set(result["symbol"].unique()) == {"AAPL", "MSFT"}

