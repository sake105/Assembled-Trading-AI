"""Tests for live data sources (PriceDataSource implementations).

This module tests the data source abstraction layer that allows the pipeline
to work with both local files and online providers (e.g., Yahoo Finance).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.config.settings import Settings, reset_settings
from src.assembled_core.data.data_source import (
    LocalParquetPriceDataSource,
    YahooPriceDataSource,
    get_price_data_source,
)

pytestmark = pytest.mark.phase11


class TestLocalParquetPriceDataSource:
    """Tests for LocalParquetPriceDataSource."""
    
    def test_local_source_loads_from_file(self, tmp_path: Path):
        """Test that local source loads from existing Parquet file."""
        # Create sample price data
        sample_data = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [100.0 + i for i in range(10)],
        })
        
        price_file = tmp_path / "prices.parquet"
        sample_data.to_parquet(price_file, index=False)
        
        # Create settings with custom output_dir
        settings = Settings(output_dir=tmp_path)
        
        # Create data source
        source = LocalParquetPriceDataSource(settings, price_file=str(price_file))
        
        # Load data
        df = source.get_history(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-10",
            freq="1d"
        )
        
        assert not df.empty
        assert len(df) == 10
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns
        assert df["symbol"].iloc[0] == "AAPL"
    
    def test_local_source_filters_by_symbols(self, tmp_path: Path):
        """Test that local source filters by symbols."""
        # Create sample data with multiple symbols
        dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
        sample_data = pd.DataFrame({
            "timestamp": list(dates) * 2,
            "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
            "close": [100.0 + i for i in range(5)] * 2,
        })
        
        price_file = tmp_path / "prices.parquet"
        sample_data.to_parquet(price_file, index=False)
        
        settings = Settings(output_dir=tmp_path)
        source = LocalParquetPriceDataSource(settings, price_file=str(price_file))
        
        # Load only AAPL
        df = source.get_history(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            freq="1d"
        )
        
        assert not df.empty
        assert len(df) == 5
        assert all(df["symbol"] == "AAPL")
    
    def test_local_source_filters_by_date_range(self, tmp_path: Path):
        """Test that local source filters by date range."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
        sample_data = pd.DataFrame({
            "timestamp": dates,
            "symbol": ["AAPL"] * 10,
            "close": [100.0 + i for i in range(10)],
        })
        
        price_file = tmp_path / "prices.parquet"
        sample_data.to_parquet(price_file, index=False)
        
        settings = Settings(output_dir=tmp_path)
        source = LocalParquetPriceDataSource(settings, price_file=str(price_file))
        
        # Load subset of dates
        df = source.get_history(
            symbols=["AAPL"],
            start_date="2024-01-03",
            end_date="2024-01-07",
            freq="1d"
        )
        
        assert not df.empty
        assert len(df) == 5  # 3, 4, 5, 6, 7
        assert df["timestamp"].min() >= pd.Timestamp("2024-01-03", tz="UTC")
        assert df["timestamp"].max() <= pd.Timestamp("2024-01-07", tz="UTC")
    
    def test_local_source_handles_today(self, tmp_path: Path):
        """Test that local source handles 'today' in date range."""
        # Create data up to today
        today = pd.Timestamp.now(tz="UTC").normalize()
        dates = pd.date_range(today - pd.Timedelta(days=5), periods=6, freq="D", tz="UTC")
        sample_data = pd.DataFrame({
            "timestamp": dates,
            "symbol": ["AAPL"] * 6,
            "close": [100.0 + i for i in range(6)],
        })
        
        price_file = tmp_path / "prices.parquet"
        sample_data.to_parquet(price_file, index=False)
        
        settings = Settings(output_dir=tmp_path)
        source = LocalParquetPriceDataSource(settings, price_file=str(price_file))
        
        # Load with end_date="today"
        df = source.get_history(
            symbols=["AAPL"],
            start_date="2020-01-01",  # Wide range
            end_date="today",
            freq="1d"
        )
        
        assert not df.empty
        assert df["timestamp"].max() <= today


class TestYahooPriceDataSource:
    """Tests for YahooPriceDataSource.
    
    Note: These tests use a fake/mock implementation to avoid real API calls.
    Real Yahoo API calls should be marked with @pytest.mark.slow and excluded from default runs.
    """
    
    def test_yahoo_source_requires_yfinance(self):
        """Test that Yahoo source requires yfinance package."""
        settings = Settings()
        
        # Try to create source (should work if yfinance is installed)
        try:
            source = YahooPriceDataSource(settings)
            assert source is not None
        except ImportError:
            # yfinance not installed - this is expected in some test environments
            pytest.skip("yfinance not installed - skipping Yahoo data source tests")
    
    def test_yahoo_source_raises_on_empty_symbols(self):
        """Test that Yahoo source raises error on empty symbols list."""
        settings = Settings()
        
        try:
            source = YahooPriceDataSource(settings)
        except ImportError:
            pytest.skip("yfinance not installed")
        
        with pytest.raises(ValueError, match="Symbols list cannot be empty"):
            source.get_history(
                symbols=[],
                start_date="2024-01-01",
                end_date="2024-01-10",
                freq="1d"
            )
    
    def test_yahoo_source_handles_today(self):
        """Test that Yahoo source handles 'today' in date range."""
        settings = Settings()
        
        try:
            source = YahooPriceDataSource(settings)
        except ImportError:
            pytest.skip("yfinance not installed")
        
        # This would make a real API call - skip in standard tests
        pytest.skip("Skipping real Yahoo API call - use @pytest.mark.slow for manual testing")


class TestGetPriceDataSource:
    """Tests for get_price_data_source factory function."""
    
    def test_get_local_source(self, tmp_path: Path):
        """Test factory returns LocalParquetPriceDataSource for 'local'."""
        settings = Settings(output_dir=tmp_path, data_source="local")
        
        source = get_price_data_source(settings)
        
        assert isinstance(source, LocalParquetPriceDataSource)
    
    def test_get_yahoo_source(self):
        """Test factory returns YahooPriceDataSource for 'yahoo'."""
        settings = Settings(data_source="yahoo")
        
        try:
            source = get_price_data_source(settings)
            assert isinstance(source, YahooPriceDataSource)
        except ImportError:
            pytest.skip("yfinance not installed")
    
    def test_get_source_with_override(self, tmp_path: Path):
        """Test factory respects data_source override parameter."""
        settings = Settings(output_dir=tmp_path, data_source="local")
        
        # Override to yahoo
        try:
            source = get_price_data_source(settings, data_source="yahoo")
            assert isinstance(source, YahooPriceDataSource)
        except ImportError:
            pytest.skip("yfinance not installed")
    
    def test_get_source_with_price_file(self, tmp_path: Path):
        """Test factory passes price_file to local source."""
        price_file = tmp_path / "custom.parquet"
        settings = Settings(output_dir=tmp_path, data_source="local")
        
        source = get_price_data_source(settings, price_file=str(price_file))
        
        assert isinstance(source, LocalParquetPriceDataSource)
        assert source.price_file == str(price_file)
    
    def test_get_source_raises_on_unknown(self):
        """Test factory raises error on unknown data source."""
        settings = Settings(data_source="local")  # Valid default
        
        # Test with invalid data_source parameter (not in settings)
        with pytest.raises(ValueError, match="Unknown data_source"):
            get_price_data_source(settings, data_source="unknown")
    
    def teardown_method(self):
        """Reset settings after each test."""
        reset_settings()


class TestFakePriceDataSource:
    """Tests with a fake/mock data source for unit testing."""
    
    def test_fake_source_interface(self):
        """Test that a fake source implements the PriceDataSource protocol."""
        # Create a simple fake implementation
        class FakePriceDataSource:
            def get_history(
                self,
                symbols: list[str],
                start_date: str,
                end_date: str,
                freq: str = "1d",
            ) -> pd.DataFrame:
                # Return fake data
                dates = pd.date_range(start_date, end_date, freq="D", tz="UTC")
                data = []
                for symbol in symbols:
                    for date in dates:
                        data.append({
                            "timestamp": date,
                            "symbol": symbol,
                            "close": 100.0,
                        })
                return pd.DataFrame(data)
        
        fake_source = FakePriceDataSource()
        
        # Test that it works
        df = fake_source.get_history(
            symbols=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            freq="1d"
        )
        
        assert not df.empty
        assert len(df) == 10  # 2 symbols * 5 days
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert "close" in df.columns

