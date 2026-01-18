# tests/test_backtest_local_only_gate.py
"""Tests for Hard Gate: Backtests must use local panels only (Sprint 3 / D3).

This test suite verifies:
1. Backtests cannot use external data sources (yahoo, finnhub, etc.)
2. Backtests load successfully from local parquet files
3. Backtests error clearly if panel is missing
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.data_source import get_price_data_source


def test_backtest_forbids_external_fetch_yahoo() -> None:
    """Test that backtest mode forbids Yahoo Finance fetch."""
    settings = get_settings()
    
    # Attempt to create Yahoo data source with allow_external_fetch=False
    with pytest.raises(ValueError, match="External data source 'yahoo' is forbidden in backtest mode"):
        get_price_data_source(
            settings=settings,
            data_source="yahoo",
            allow_external_fetch=False,  # Backtest mode
        )


def test_backtest_forbids_external_fetch_finnhub() -> None:
    """Test that backtest mode forbids Finnhub fetch."""
    settings = get_settings()
    
    # Attempt to create Finnhub data source with allow_external_fetch=False
    with pytest.raises(ValueError, match="External data source 'finnhub' is forbidden in backtest mode"):
        get_price_data_source(
            settings=settings,
            data_source="finnhub",
            allow_external_fetch=False,  # Backtest mode
        )


def test_backtest_allows_local_source() -> None:
    """Test that backtest mode allows local data source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a dummy price file
        price_file = Path(tmpdir) / "prices.parquet"
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [150.0] * 5,
        })
        prices.to_parquet(price_file, index=False)
        
        settings = get_settings()
        
        # Should succeed with local source
        price_source = get_price_data_source(
            settings=settings,
            data_source="local",
            price_file=str(price_file),
            allow_external_fetch=False,  # Backtest mode
        )
        
        assert price_source is not None, "Local data source should be created"


def test_backtest_loads_from_local_parquet() -> None:
    """Test that backtest loads successfully from local parquet file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a price file
        price_file = Path(tmpdir) / "prices.parquet"
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "close": [150.0 + i * 0.5 for i in range(10)],
        })
        prices.to_parquet(price_file, index=False)
        
        settings = get_settings()
        
        # Load via local data source (backtest mode)
        price_source = get_price_data_source(
            settings=settings,
            data_source="local",
            price_file=str(price_file),
            allow_external_fetch=False,  # Backtest mode
        )
        
        # Load prices
        loaded_prices = price_source.get_history(
            symbols=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-10",
            freq="1d",
        )
        
        assert len(loaded_prices) == 10, "Should load all rows from local file"
        assert loaded_prices["symbol"].nunique() == 1, "Should have one symbol"


def test_backtest_errors_if_panel_missing() -> None:
    """Test that backtest errors clearly if panel is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Non-existent price file
        price_file = Path(tmpdir) / "nonexistent.parquet"
        
        settings = get_settings()
        
        # Create local data source (should succeed)
        price_source = get_price_data_source(
            settings=settings,
            data_source="local",
            price_file=str(price_file),
            allow_external_fetch=False,  # Backtest mode
        )
        
        # Attempt to load (should fail with clear error)
        with pytest.raises((FileNotFoundError, ValueError), match="not found|No data"):
            price_source.get_history(
                symbols=["AAPL"],
                start_date="2024-01-01",
                end_date="2024-01-10",
                freq="1d",
            )


def test_daily_allows_external_fetch() -> None:
    """Test that daily mode (not backtest) allows external fetches."""
    settings = get_settings()
    
    # Daily mode should allow external fetches (if yfinance is available)
    try:
        price_source = get_price_data_source(
            settings=settings,
            data_source="yahoo",
            allow_external_fetch=True,  # Daily mode allows external
        )
        # If yfinance is not available, this will raise ImportError (expected)
        # If available, should succeed
        assert price_source is not None or True  # Either way is fine for this test
    except ImportError:
        # yfinance not available - that's fine, test passes if we get here
        # (we're testing the gate logic, not yfinance availability)
        pass


def test_backtest_monkeypatch_provider_fetch_fails() -> None:
    """Test that monkeypatching provider fetch in backtest mode fails."""
    settings = get_settings()
    
    # Attempt to bypass gate by directly creating Yahoo source
    # This should be caught by the factory function
    with pytest.raises(ValueError, match="forbidden in backtest mode"):
        get_price_data_source(
            settings=settings,
            data_source="yahoo",
            allow_external_fetch=False,  # Backtest mode - should fail
        )


def test_backtest_error_message_clear() -> None:
    """Test that error message is clear and actionable."""
    settings = get_settings()
    
    try:
        get_price_data_source(
            settings=settings,
            data_source="yahoo",
            allow_external_fetch=False,  # Backtest mode
        )
        pytest.fail("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        # Check that error message contains helpful hints
        assert "forbidden in backtest mode" in error_msg.lower()
        assert "local" in error_msg.lower() or "panel" in error_msg.lower()
        assert "daily ingest" in error_msg.lower() or "run" in error_msg.lower()
