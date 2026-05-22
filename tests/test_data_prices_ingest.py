# tests/test_data_prices_ingest.py
"""Tests for EOD price data ingestion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.prices_ingest import (
    load_eod_prices,
    load_eod_prices_for_universe,
    validate_price_data,
)


def test_load_eod_prices_from_sample_file():
    """Test loading EOD prices from sample file."""
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"

    if not sample_file.exists():
        pytest.skip("Sample data file not found. Run data ingestion first.")

    df = load_eod_prices(price_file=sample_file)

    # Assert DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "DataFrame should not be empty"

    # Assert required columns
    required_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        assert col in df.columns, f"Column '{col}' should be present"

    # Assert data types
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), (
        "Timestamp should be datetime"
    )
    assert df["timestamp"].dt.tz is not None, "Timestamp should be timezone-aware (UTC)"

    for col in ["open", "high", "low", "close", "volume"]:
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"Column '{col}' should be numeric"
        )

    # Assert minimum data
    assert len(df) >= 30, "Should have at least 30 rows (30 days * 1 symbol minimum)"
    assert df["symbol"].nunique() >= 1, "Should have at least 1 symbol"

    # Assert sorting
    assert df.equals(df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)), (
        "DataFrame should be sorted by symbol, then timestamp"
    )


def test_load_eod_prices_filter_by_symbols():
    """Test loading EOD prices filtered by specific symbols."""
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"

    if not sample_file.exists():
        pytest.skip("Sample data file not found. Run data ingestion first.")

    # Load only AAPL
    df = load_eod_prices(symbols=["AAPL"], price_file=sample_file)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df["symbol"].nunique() == 1
    assert (df["symbol"] == "AAPL").all()

    # Load multiple symbols
    df_multi = load_eod_prices(symbols=["AAPL", "MSFT"], price_file=sample_file)

    assert isinstance(df_multi, pd.DataFrame)
    assert not df_multi.empty
    assert df_multi["symbol"].nunique() == 2
    assert set(df_multi["symbol"].unique()) == {"AAPL", "MSFT"}


def test_load_eod_prices_file_not_found():
    """Test that FileNotFoundError is raised when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_eod_prices(price_file=Path("nonexistent_file.parquet"))


def test_load_eod_prices_invalid_symbols():
    """Test that ValueError is raised when no data found for symbols."""
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"

    if not sample_file.exists():
        pytest.skip("Sample data file not found. Run data ingestion first.")

    with pytest.raises(ValueError, match="No data found for symbols"):
        load_eod_prices(symbols=["INVALID_SYMBOL"], price_file=sample_file)


def test_validate_price_data_valid():
    """Test price data validation with valid data."""
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"

    if not sample_file.exists():
        pytest.skip("Sample data file not found. Run data ingestion first.")

    df = load_eod_prices(price_file=sample_file)
    result = validate_price_data(df)

    assert isinstance(result, dict)
    assert "valid" in result
    assert "row_count" in result
    assert "symbol_count" in result
    assert "date_range" in result
    assert "issues" in result

    # With valid sample data, should be valid
    assert result["valid"] is True or len(result["issues"]) == 0, (
        f"Sample data should be valid, but got issues: {result['issues']}"
    )
    assert result["row_count"] > 0
    assert result["symbol_count"] > 0


def test_validate_price_data_invalid():
    """Test price data validation with invalid data."""
    # Create invalid DataFrame (high < low)
    invalid_data = {
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True),
        "symbol": ["AAPL", "AAPL"],
        "open": [100.0, 101.0],
        "high": [99.0, 100.0],  # Invalid: high < low
        "low": [100.0, 101.0],
        "close": [100.5, 100.5],
        "volume": [1000000.0, 1000000.0],
    }
    df = pd.DataFrame(invalid_data)

    result = validate_price_data(df)

    assert isinstance(result, dict)
    assert "valid" in result
    assert "issues" in result

    # Should detect invalid OHLC relationships
    assert len(result["issues"]) > 0, "Should detect invalid OHLC relationships"


def test_load_eod_prices_for_universe(tmp_path: Path, monkeypatch):
    """Test loading EOD prices from universe file."""
    # Create temporary universe file
    universe_file = tmp_path / "test_universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")

    # Create sample price file
    sample_file = tmp_path / "sample.parquet"
    from datetime import datetime, timedelta

    base = datetime(2025, 1, 1, 0, 0, 0)
    data = []
    for sym in ["AAPL", "MSFT", "GOOGL"]:
        for i in range(10):
            ts = base + timedelta(days=i)
            price_base = 100.0
            data.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": price_base + i * 0.1,
                    "high": price_base + i * 0.1 + 0.5,
                    "low": price_base + i * 0.1 - 0.3,
                    "close": price_base + i * 0.1 + 0.2,
                    "volume": 1000000.0,
                }
            )
    df_sample = pd.DataFrame(data)
    df_sample["timestamp"] = pd.to_datetime(df_sample["timestamp"], utc=True)
    df_sample.to_parquet(sample_file, index=False)

    # Load prices for universe
    df = load_eod_prices_for_universe(
        universe_file=universe_file, price_file=sample_file
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df["symbol"].nunique() == 3
    assert set(df["symbol"].unique()) == {"AAPL", "MSFT", "GOOGL"}


def test_load_eod_prices_ohlcv_columns():
    """Test that load_eod_prices returns all OHLCV columns."""
    sample_file = ROOT / "data" / "sample" / "eod_sample.parquet"

    if not sample_file.exists():
        pytest.skip("Sample data file not found. Run data ingestion first.")

    df = load_eod_prices(price_file=sample_file)

    # Assert all OHLCV columns are present
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        assert col in df.columns, f"OHLCV column '{col}' should be present"
        assert not df[col].isna().all(), f"Column '{col}' should have non-NaN values"

    # Assert OHLC relationships (high >= low, high >= open, high >= close, etc.)
    assert (df["high"] >= df["low"]).all(), "High should be >= Low"
    assert (df["high"] >= df["open"]).all(), "High should be >= Open"
    assert (df["high"] >= df["close"]).all(), "High should be >= Close"
    assert (df["low"] <= df["open"]).all(), "Low should be <= Open"
    assert (df["low"] <= df["close"]).all(), "Low should be <= Close"
