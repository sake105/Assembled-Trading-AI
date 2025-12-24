"""Unit tests for factor store paths, manifest, and atomic writes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.factor_store import (
    _write_parquet_atomic,
    compute_universe_key,
    get_factor_store_root,
    list_available_panels,
    load_factors,
    panel_path,
    store_factors,
)

pytestmark = pytest.mark.advanced


def test_get_factor_store_root() -> None:
    """Test that get_factor_store_root returns a valid path."""
    root = get_factor_store_root()
    assert isinstance(root, Path)
    assert "factors" in str(root) or root.name == "factors"


def test_compute_universe_key_deterministic() -> None:
    """Test that universe key is deterministic for same symbols."""
    symbols1 = ["AAPL", "MSFT", "GOOGL"]
    symbols2 = ["AAPL", "MSFT", "GOOGL"]
    symbols3 = ["MSFT", "AAPL", "GOOGL"]  # Different order

    key1 = compute_universe_key(symbols=symbols1)
    key2 = compute_universe_key(symbols=symbols2)
    key3 = compute_universe_key(symbols=symbols3)

    assert key1 == key2, "Universe key should be deterministic"
    assert key1 == key3, "Universe key should be order-independent"
    assert key1.startswith("universe_")


def test_compute_universe_key_from_file(tmp_path: Path) -> None:
    """Test computing universe key from file."""
    universe_file = tmp_path / "universe.txt"
    universe_file.write_text("AAPL\nMSFT\nGOOGL\n", encoding="utf-8")

    key = compute_universe_key(universe_file=universe_file)
    assert key.startswith("universe_")


def test_panel_path_construction() -> None:
    """Test panel path construction."""
    root = get_factor_store_root()
    path = panel_path("core_ta", "1d", "universe_test", year=2024, factors_root=root)

    assert path.parent.name == "universe_test"
    assert path.parent.parent.name == "1d"
    assert path.parent.parent.parent.name == "core_ta"
    assert path.name == "year=2024.parquet"


def test_write_parquet_atomic(tmp_path: Path) -> None:
    """Test atomic Parquet write."""
    target_file = tmp_path / "test.parquet"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    _write_parquet_atomic(target_file, df)

    assert target_file.exists()
    loaded_df = pd.read_parquet(target_file)
    assert len(loaded_df) == 3
    assert list(loaded_df.columns) == ["a", "b"]


def test_store_factors_with_year_partitioning(tmp_path: Path) -> None:
    """Test that store_factors partitions by year."""
    # Create DataFrame spanning 2 years
    dates = pd.date_range("2023-12-15", "2024-01-15", freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "ta_ma_20": [100.0] * len(dates),
        }
    )

    # Add date column (store_factors adds it if missing)
    df["date"] = df["timestamp"].dt.date.astype(str)

    universe_key = compute_universe_key(symbols=["AAPL"])
    panel_dir = store_factors(
        df=df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
        write_manifest=True,
    )

    # Check that year partitions exist
    year_2023 = panel_dir / "year=2023.parquet"
    year_2024 = panel_dir / "year=2024.parquet"

    assert year_2023.exists() or year_2024.exists(), "At least one year partition should exist"

    # Check manifest
    manifest_file = panel_dir / "_metadata.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["factor_group"] == "core_ta"
    assert manifest["freq"] == "1d"
    assert manifest["universe_key"] == universe_key
    assert "years" in manifest
    assert "date_range" in manifest
    assert "factor_columns" in manifest
    assert "config_hash" in manifest


def test_store_factors_manifest_content(tmp_path: Path) -> None:
    """Test that manifest contains expected fields."""
    dates = pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "ta_ma_20": [100.0] * len(dates),
        }
    )

    universe_key = compute_universe_key(symbols=["AAPL"])
    panel_dir = store_factors(
        df=df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
        write_manifest=True,
        metadata={"test_param": "test_value"},
    )

    manifest_file = panel_dir / "_metadata.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check required fields
    assert manifest["factor_group"] == "core_ta"
    assert manifest["freq"] == "1d"
    assert manifest["universe_key"] == universe_key
    assert "computed_at" in manifest
    assert "date_range" in manifest
    assert "years" in manifest
    assert manifest["years"] == [2024]
    assert "factor_columns" in manifest
    assert "ta_ma_20" in manifest["factor_columns"]
    assert "schema" in manifest
    assert "config_hash" in manifest

    # Check custom metadata
    assert manifest["test_param"] == "test_value"


def test_load_factors_with_date_filtering(tmp_path: Path) -> None:
    """Test loading factors with date filtering."""
    # Create data spanning 2 years
    dates = pd.date_range("2023-06-01", "2024-06-01", freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "ta_ma_20": [100.0] * len(dates),
        }
    )

    universe_key = compute_universe_key(symbols=["AAPL"])
    store_factors(
        df=df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        factors_root=tmp_path / "factors",
    )

    # Load only 2024 data
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        start_date="2024-01-01",
        end_date="2024-12-31",
        factors_root=tmp_path / "factors",
    )

    assert loaded_df is not None
    assert len(loaded_df) > 0
    assert loaded_df["timestamp"].min().year == 2024
    assert loaded_df["timestamp"].max().year == 2024


def test_list_available_panels_with_years(tmp_path: Path) -> None:
    """Test that list_available_panels shows year coverage."""
    # Create two factor groups with different year coverage
    dates1 = pd.date_range("2023-01-01", "2023-12-31", freq="D", tz="UTC")
    df1 = pd.DataFrame(
        {
            "timestamp": dates1,
            "symbol": ["AAPL"] * len(dates1),
            "ta_ma_20": [100.0] * len(dates1),
        }
    )

    dates2 = pd.date_range("2024-01-01", "2024-12-31", freq="D", tz="UTC")
    df2 = pd.DataFrame(
        {
            "timestamp": dates2,
            "symbol": ["MSFT"] * len(dates2),
            "vol_rv_20": [0.15] * len(dates2),
        }
    )

    universe_key1 = compute_universe_key(symbols=["AAPL"])
    universe_key2 = compute_universe_key(symbols=["MSFT"])

    _ = store_factors(
        df=df1,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key1,
        factors_root=tmp_path / "factors",
    )

    _ = store_factors(
        df=df2,
        factor_group="vol_liquidity",
        freq="1d",
        universe_key=universe_key2,
        factors_root=tmp_path / "factors",
    )

    panels = list_available_panels(factors_root=tmp_path / "factors")

    assert len(panels) >= 2

    # Check that years are listed
    for panel in panels:
        assert "years" in panel
        assert isinstance(panel["years"], list)
        if panel["factor_group"] == "core_ta":
            assert 2023 in panel["years"]
        elif panel["factor_group"] == "vol_liquidity":
            assert 2024 in panel["years"]


def test_store_factors_append_mode(tmp_path: Path) -> None:
    """Test append mode with deduplication."""
    # Store initial data
    dates1 = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    df1 = pd.DataFrame(
        {
            "timestamp": dates1,
            "symbol": ["AAPL"] * len(dates1),
            "ta_ma_20": [100.0] * len(dates1),
        }
    )

    universe_key = compute_universe_key(symbols=["AAPL"])
    _ = store_factors(
        df=df1,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        mode="overwrite",
        factors_root=tmp_path / "factors",
    )

    # Append overlapping data (should deduplicate)
    dates2 = pd.date_range("2024-01-03", "2024-01-07", freq="D", tz="UTC")
    df2 = pd.DataFrame(
        {
            "timestamp": dates2,
            "symbol": ["AAPL"] * len(dates2),
            "ta_ma_20": [101.0] * len(dates2),  # Different values
        }
    )

    store_factors(
        df=df2,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        mode="append",
        factors_root=tmp_path / "factors",
    )

    # Load and verify deduplication (should have 7 days, not 9)
    loaded_df = load_factors(
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        start_date="2024-01-01",
        end_date="2024-01-10",
        factors_root=tmp_path / "factors",
    )

    assert loaded_df is not None
    assert len(loaded_df) == 7  # 5 days from first + 2 new days (3-5 deduplicated)
    # Check that later values win (101.0 for overlapping dates)
    assert loaded_df[loaded_df["timestamp"] == pd.Timestamp("2024-01-03", tz="UTC")]["ta_ma_20"].iloc[0] == 101.0


def test_store_factors_without_manifest(tmp_path: Path) -> None:
    """Test that store_factors can skip manifest writing."""
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "ta_ma_20": [100.0] * len(dates),
        }
    )

    universe_key = compute_universe_key(symbols=["AAPL"])
    panel_dir = store_factors(
        df=df,
        factor_group="core_ta",
        freq="1d",
        universe_key=universe_key,
        write_manifest=False,
        factors_root=tmp_path / "factors",
    )

    manifest_file = panel_dir / "_metadata.json"
    assert not manifest_file.exists(), "Manifest should not be created when write_manifest=False"


def test_panel_path_without_year() -> None:
    """Test panel_path without year returns directory."""
    root = get_factor_store_root()
    path = panel_path("core_ta", "1d", "universe_test", year=None, factors_root=root)

    assert path.is_dir() or not path.suffix, "Path without year should be directory path"
    assert path.name == "universe_test" or "universe_test" in str(path)

