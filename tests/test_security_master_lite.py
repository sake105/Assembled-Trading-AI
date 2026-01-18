# tests/test_security_master_lite.py
"""Tests for Security Master Lite (Sprint 9).

Tests verify:
1. Load/store roundtrip (csv/parquet)
2. Validation: missing required cols -> ValueError
3. Resolve: missing symbols -> ValueError (with missing list)
4. Deterministic ordering (symbol asc)
5. Atomic write (no tmp left behind)
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

from src.assembled_core.data.security_master import (
    load_security_master,
    resolve_security_meta,
    store_security_master,
)


def test_load_store_roundtrip_parquet() -> None:
    """Test load/store roundtrip with parquet format."""
    # Create sample data
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "sector": ["Technology", "Technology", "Technology"],
        "region": ["US", "US", "US"],
        "currency": ["USD", "USD", "USD"],
        "asset_type": ["EQUITY", "EQUITY", "EQUITY"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"

        # Store
        store_security_master(df, path)

        # Load
        loaded_df = load_security_master(path)

        # Verify: same data (ignoring index)
        pd.testing.assert_frame_equal(
            df.sort_values("symbol").reset_index(drop=True),
            loaded_df.sort_values("symbol").reset_index(drop=True),
            check_dtype=False,
        )


def test_load_store_roundtrip_csv() -> None:
    """Test load/store roundtrip with csv format."""
    # Create sample data
    df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "sector": ["Technology", "Technology", "Technology"],
        "region": ["US", "US", "US"],
        "currency": ["USD", "USD", "USD"],
        "asset_type": ["EQUITY", "EQUITY", "EQUITY"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"

        # Store
        store_security_master(df, path)

        # Load
        loaded_df = load_security_master(path)

        # Verify: same data (ignoring index)
        pd.testing.assert_frame_equal(
            df.sort_values("symbol").reset_index(drop=True),
            loaded_df.sort_values("symbol").reset_index(drop=True),
            check_dtype=False,
        )


def test_load_missing_required_columns_raises_valueerror() -> None:
    """Test that loading with missing required columns raises ValueError."""
    # Create data with missing column
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
        "region": ["US"],
        # Missing: currency, asset_type
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        df.to_parquet(path, index=False)

        # Load should raise ValueError
        with pytest.raises(ValueError, match="missing required columns"):
            load_security_master(path)


def test_store_missing_required_columns_raises_valueerror() -> None:
    """Test that storing with missing required columns raises ValueError."""
    # Create data with missing column
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
        "region": ["US"],
        # Missing: currency, asset_type
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"

        # Store should raise ValueError
        with pytest.raises(ValueError, match="missing required columns"):
            store_security_master(df, path)


def test_resolve_missing_symbols_raises_valueerror_with_list() -> None:
    """Test that resolving missing symbols raises ValueError with missing list."""
    # Create master data
    master_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "US"],
        "currency": ["USD", "USD"],
        "asset_type": ["EQUITY", "EQUITY"],
    })

    # Try to resolve with missing symbol
    with pytest.raises(ValueError) as exc_info:
        resolve_security_meta(["AAPL", "INVALID", "MISSING"], master_df)

    # Verify error message contains missing symbols
    error_msg = str(exc_info.value)
    assert "INVALID" in error_msg or "Missing symbols" in error_msg
    assert "MISSING" in error_msg or "Missing symbols" in error_msg


def test_resolve_missing_symbols_with_default_policy() -> None:
    """Test that resolving missing symbols with default policy uses defaults."""
    # Create master data
    master_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
        "region": ["US"],
        "currency": ["USD"],
        "asset_type": ["EQUITY"],
    })

    # Resolve with missing symbol (default policy)
    result_df = resolve_security_meta(
        ["AAPL", "INVALID"],
        master_df,
        missing_policy="default",
        default_sector="UNKNOWN",
        default_region="UNKNOWN",
        default_currency="UNKNOWN",
        default_asset_type="UNKNOWN",
    )

    # Verify: both symbols present
    assert len(result_df) == 2
    assert set(result_df["symbol"]) == {"AAPL", "INVALID"}

    # Verify: AAPL has original values
    aapl_row = result_df[result_df["symbol"] == "AAPL"].iloc[0]
    assert aapl_row["sector"] == "Technology"

    # Verify: INVALID has default values
    invalid_row = result_df[result_df["symbol"] == "INVALID"].iloc[0]
    assert invalid_row["sector"] == "UNKNOWN"
    assert invalid_row["region"] == "UNKNOWN"
    assert invalid_row["currency"] == "UNKNOWN"
    assert invalid_row["asset_type"] == "UNKNOWN"


def test_deterministic_ordering_symbol_asc() -> None:
    """Test that results are deterministically sorted by symbol (ascending)."""
    # Create unsorted master data
    master_df = pd.DataFrame({
        "symbol": ["MSFT", "AAPL", "GOOGL"],
        "sector": ["Technology", "Technology", "Technology"],
        "region": ["US", "US", "US"],
        "currency": ["USD", "USD", "USD"],
        "asset_type": ["EQUITY", "EQUITY", "EQUITY"],
    })

    # Load should sort by symbol
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        store_security_master(master_df, path)
        loaded_df = load_security_master(path)

        # Verify: sorted by symbol ascending
        assert loaded_df["symbol"].tolist() == ["AAPL", "GOOGL", "MSFT"]

    # Resolve should also sort
    result_df = resolve_security_meta(["MSFT", "AAPL", "GOOGL"], master_df)
    assert result_df["symbol"].tolist() == ["AAPL", "GOOGL", "MSFT"]


def test_atomic_write_no_tmp_left_behind() -> None:
    """Test that atomic write doesn't leave temp files behind."""
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
        "region": ["US"],
        "currency": ["USD"],
        "asset_type": ["EQUITY"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"

        # Store
        store_security_master(df, path)

        # Verify: only target file exists, no temp files
        files = list(Path(tmpdir).glob("*"))
        assert len(files) == 1
        assert files[0] == path

        # Verify: file is readable
        loaded_df = load_security_master(path)
        assert len(loaded_df) == 1


def test_string_normalization_strip() -> None:
    """Test that strings are normalized (strip whitespace)."""
    # Create data with whitespace
    df = pd.DataFrame({
        "symbol": [" AAPL ", " MSFT "],
        "sector": [" Technology ", " Technology "],
        "region": [" US ", " US "],
        "currency": [" USD ", " USD "],
        "asset_type": [" EQUITY ", " EQUITY "],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        store_security_master(df, path)
        loaded_df = load_security_master(path)

        # Verify: strings are stripped
        assert loaded_df["symbol"].iloc[0] == "AAPL"
        assert loaded_df["sector"].iloc[0] == "Technology"
        assert loaded_df["region"].iloc[0] == "US"
        assert loaded_df["currency"].iloc[0] == "USD"
        assert loaded_df["asset_type"].iloc[0] == "EQUITY"


def test_optional_columns_preserved() -> None:
    """Test that optional columns are preserved if present."""
    # Create data with optional columns
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
        "region": ["US"],
        "currency": ["USD"],
        "asset_type": ["EQUITY"],
        "exchange": ["NASDAQ"],
        "timezone": ["America/New_York"],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.parquet"
        store_security_master(df, path)
        loaded_df = load_security_master(path)

        # Verify: optional columns preserved
        assert "exchange" in loaded_df.columns
        assert "timezone" in loaded_df.columns
        assert loaded_df["exchange"].iloc[0] == "NASDAQ"
        assert loaded_df["timezone"].iloc[0] == "America/New_York"


def test_resolve_all_symbols_present() -> None:
    """Test that resolve returns all requested symbols when all are present."""
    master_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "sector": ["Technology", "Technology", "Technology"],
        "region": ["US", "US", "US"],
        "currency": ["USD", "USD", "USD"],
        "asset_type": ["EQUITY", "EQUITY", "EQUITY"],
    })

    result_df = resolve_security_meta(["AAPL", "MSFT", "GOOGL"], master_df)

    # Verify: all symbols present
    assert len(result_df) == 3
    assert set(result_df["symbol"]) == {"AAPL", "MSFT", "GOOGL"}


def test_resolve_subset_of_symbols() -> None:
    """Test that resolve returns only requested symbols."""
    master_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA"],
        "sector": ["Technology", "Technology", "Technology", "Consumer"],
        "region": ["US", "US", "US", "US"],
        "currency": ["USD", "USD", "USD", "USD"],
        "asset_type": ["EQUITY", "EQUITY", "EQUITY", "EQUITY"],
    })

    result_df = resolve_security_meta(["AAPL", "GOOGL"], master_df)

    # Verify: only requested symbols
    assert len(result_df) == 2
    assert set(result_df["symbol"]) == {"AAPL", "GOOGL"}


def test_load_file_not_found_raises_filenotfounderror() -> None:
    """Test that loading non-existent file raises FileNotFoundError."""
    path = Path("/nonexistent/path/security_master.parquet")

    with pytest.raises(FileNotFoundError):
        load_security_master(path)


def test_load_unsupported_format_raises_valueerror() -> None:
    """Test that loading unsupported format raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="Unsupported file format"):
            load_security_master(path)
