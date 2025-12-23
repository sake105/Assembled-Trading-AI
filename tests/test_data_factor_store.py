"""Unit tests for factor store module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.factor_store import (
    get_factor_store_root,
    list_available_panels,
    load_factors,
    load_price_panel,
    store_factors,
)


@pytest.mark.advanced
def test_factor_store_roundtrip(tmp_path: Path) -> None:
    """Test roundtrip: store factors, then load them back."""
    # Create synthetic factor DataFrame
    dates_2022 = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    dates_2023 = pd.date_range("2023-01-01", "2023-01-10", freq="D")
    all_dates = dates_2022.tolist() + dates_2023.tolist()

    symbols = ["AAPL", "MSFT"]

    rows = []
    for date in all_dates:
        for symbol in symbols:
            rows.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "factor_mom": 0.5 + hash(f"{date}{symbol}") % 100 / 1000.0,
                    "factor_value": -0.3 + hash(f"{symbol}{date}") % 100 / 1000.0,
                }
            )

    original_df = pd.DataFrame(rows)

    # Store factors
    stored_paths = store_factors(
        freq="1d",
        group="ta",
        df=original_df,
        root=tmp_path,
    )

    # Verify that files were created for both years
    assert "2022" in stored_paths
    assert "2023" in stored_paths
    assert stored_paths["2022"].exists()
    assert stored_paths["2023"].exists()

    # Load factors back
    loaded_df = load_factors(
        freq="1d",
        universe=["AAPL", "MSFT"],
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2023-01-10"),
        groups=["ta"],
        root=tmp_path,
    )

    # Verify loaded DataFrame is not empty
    assert not loaded_df.empty, "Loaded DataFrame should not be empty"

    # Verify columns
    assert "timestamp" in loaded_df.columns
    assert "symbol" in loaded_df.columns
    assert "factor_mom" in loaded_df.columns
    assert "factor_value" in loaded_df.columns

    # Sort both DataFrames for comparison
    original_sorted = original_df.sort_values(["timestamp", "symbol"]).reset_index(
        drop=True
    )
    loaded_sorted = (
        loaded_df[["timestamp", "symbol", "factor_mom", "factor_value"]]
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )

    # Verify same set of (timestamp, symbol) pairs
    original_keys = set(zip(original_sorted["timestamp"], original_sorted["symbol"]))
    loaded_keys = set(zip(loaded_sorted["timestamp"], loaded_sorted["symbol"]))
    assert original_keys == loaded_keys, (
        "Loaded DataFrame should have same (timestamp, symbol) pairs"
    )

    # Verify values roundtrip correctly (allow for small floating point differences)
    pd.testing.assert_frame_equal(
        original_sorted[["timestamp", "symbol", "factor_mom", "factor_value"]],
        loaded_sorted,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.advanced
def test_factor_store_list_available(tmp_path: Path) -> None:
    """Test list_available_panels function."""
    # Create and store data for multiple groups and years
    dates_2022 = pd.date_range("2022-01-01", "2022-01-05", freq="D")
    dates_2023 = pd.date_range("2023-01-01", "2023-01-05", freq="D")

    # Store "ta" group data
    ta_rows = []
    for date in dates_2022.tolist() + dates_2023.tolist():
        ta_rows.append({"timestamp": date, "symbol": "AAPL", "factor_mom": 0.5})

    ta_df = pd.DataFrame(ta_rows)
    store_factors(freq="1d", group="ta", df=ta_df, root=tmp_path)

    # Store "alt_insider" group data
    insider_rows = []
    for date in dates_2022.tolist() + dates_2023.tolist():
        insider_rows.append(
            {"timestamp": date, "symbol": "AAPL", "factor_insider": 0.8}
        )

    insider_df = pd.DataFrame(insider_rows)
    store_factors(freq="1d", group="alt_insider", df=insider_df, root=tmp_path)

    # List available panels
    available = list_available_panels(root=tmp_path)

    # Verify result is a DataFrame
    assert isinstance(available, pd.DataFrame), "Result should be a DataFrame"

    # Verify required columns
    required_cols = {"freq", "group", "year", "path"}
    assert required_cols.issubset(set(available.columns)), (
        f"Result should contain columns: {required_cols}"
    )

    # Verify at least one row with group == "ta"
    ta_rows_df = available[available["group"] == "ta"]
    assert len(ta_rows_df) > 0, "Should have at least one row with group='ta'"

    # Verify (freq, group) combinations include ("1d", "ta")
    freq_group_pairs = set(zip(available["freq"], available["group"]))
    assert ("1d", "ta") in freq_group_pairs, "Should include (freq='1d', group='ta')"


@pytest.mark.advanced
def test_factor_store_point_in_time(tmp_path: Path) -> None:
    """Test point-in-time safety: data beyond end date should not be loaded."""
    # Create synthetic DataFrame with three dates
    dates = [
        pd.Timestamp("2022-01-01"),
        pd.Timestamp("2022-01-02"),
        pd.Timestamp("2022-01-03"),
    ]

    rows = []
    for date in dates:
        rows.append({"timestamp": date, "symbol": "AAPL", "factor_mom": 0.5})

    df = pd.DataFrame(rows)

    # Store factors
    store_factors(freq="1d", group="ta", df=df, root=tmp_path)

    # Load with end date that excludes 2022-01-03
    loaded_df = load_factors(
        freq="1d",
        universe=["AAPL"],
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2022-01-02"),  # Intentionally exclude 2022-01-03
        groups=["ta"],
        root=tmp_path,
    )

    # Verify maximum timestamp is <= end
    assert not loaded_df.empty, "Loaded DataFrame should not be empty"
    max_timestamp = loaded_df["timestamp"].max()
    assert max_timestamp <= pd.Timestamp("2022-01-02"), (
        f"Max timestamp ({max_timestamp}) should be <= 2022-01-02"
    )

    # Verify there is no 2022-01-03 row
    dates_in_result = set(loaded_df["timestamp"].dt.date)
    excluded_date = pd.Timestamp("2022-01-03").date()
    assert excluded_date not in dates_in_result, (
        "Should not contain data for 2022-01-03 (beyond end date)"
    )


@pytest.mark.advanced
def test_factor_store_multiple_groups(tmp_path: Path) -> None:
    """Test loading multiple groups and merging them."""
    dates = pd.date_range("2022-01-01", "2022-01-05", freq="D")

    # Store "ta" group
    ta_rows = []
    for date in dates:
        ta_rows.append({"timestamp": date, "symbol": "AAPL", "factor_mom": 0.5})

    ta_df = pd.DataFrame(ta_rows)
    store_factors(freq="1d", group="ta", df=ta_df, root=tmp_path)

    # Store "alt_insider" group
    insider_rows = []
    for date in dates:
        insider_rows.append(
            {"timestamp": date, "symbol": "AAPL", "factor_insider": 0.8}
        )

    insider_df = pd.DataFrame(insider_rows)
    store_factors(freq="1d", group="alt_insider", df=insider_df, root=tmp_path)

    # Load both groups
    loaded_df = load_factors(
        freq="1d",
        universe=["AAPL"],
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2022-01-05"),
        groups=["ta", "alt_insider"],
        root=tmp_path,
    )

    # Verify both factor columns are present
    assert "factor_mom" in loaded_df.columns, (
        "Should contain factor_mom from 'ta' group"
    )
    assert "factor_insider" in loaded_df.columns, (
        "Should contain factor_insider from 'alt_insider' group"
    )

    # Verify merged correctly (should have same number of rows as input)
    assert len(loaded_df) == len(dates), (
        "Merged DataFrame should have correct number of rows"
    )


@pytest.mark.advanced
def test_factor_store_universe_filtering(tmp_path: Path) -> None:
    """Test that universe filtering works correctly."""
    dates = pd.date_range("2022-01-01", "2022-01-05", freq="D")
    all_symbols = ["AAPL", "MSFT", "GOOGL"]

    rows = []
    for date in dates:
        for symbol in all_symbols:
            rows.append({"timestamp": date, "symbol": symbol, "factor_mom": 0.5})

    df = pd.DataFrame(rows)

    # Store all symbols
    store_factors(freq="1d", group="ta", df=df, root=tmp_path)

    # Load with restricted universe
    loaded_df = load_factors(
        freq="1d",
        universe=["AAPL", "MSFT"],  # Exclude GOOGL
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2022-01-05"),
        groups=["ta"],
        root=tmp_path,
    )

    # Verify only requested symbols are present
    symbols_in_result = set(loaded_df["symbol"].unique())
    assert symbols_in_result == {"AAPL", "MSFT"}, (
        f"Should only contain AAPL and MSFT, got {symbols_in_result}"
    )


@pytest.mark.advanced
def test_factor_store_get_root() -> None:
    """Test get_factor_store_root function."""
    # Test default behavior (should return data/factors)
    root = get_factor_store_root()
    assert isinstance(root, Path), "Should return a Path object"
    assert root.name == "factors", "Default root should end with 'factors'"

    # Test with explicit root
    explicit_root = Path("/tmp/test_factors")
    result = get_factor_store_root(root=explicit_root)
    assert result == explicit_root.resolve(), "Should return the provided root path"


@pytest.mark.advanced
def test_factor_store_empty_dataframe(tmp_path: Path) -> None:
    """Test handling of empty DataFrame in store_factors."""
    # Create empty DataFrame with required columns
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "factor_mom"])

    # Store should not raise error, but return empty dict
    stored_paths = store_factors(freq="1d", group="ta", df=empty_df, root=tmp_path)
    assert stored_paths == {}, "Should return empty dict for empty DataFrame"


@pytest.mark.advanced
def test_factor_store_nonexistent_files(tmp_path: Path) -> None:
    """Test loading when files don't exist (should return empty DataFrame)."""
    loaded_df = load_factors(
        freq="1d",
        universe=["AAPL"],
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2022-01-05"),
        groups=["nonexistent"],
        root=tmp_path,
    )

    # Should return empty DataFrame with timestamp and symbol columns
    assert loaded_df.empty, "Should return empty DataFrame when no files exist"
    assert list(loaded_df.columns) == ["timestamp", "symbol"], (
        "Empty DataFrame should have timestamp and symbol columns"
    )


@pytest.mark.advanced
def test_load_price_panel_not_implemented() -> None:
    """Test that load_price_panel raises NotImplementedError without price_loader."""
    with pytest.raises(NotImplementedError, match="price_loader"):
        load_price_panel(
            freq="1d",
            universe=["AAPL"],
            start=pd.Timestamp("2022-01-01"),
            end=pd.Timestamp("2022-01-05"),
        )


@pytest.mark.advanced
def test_store_factors_missing_columns(tmp_path: Path) -> None:
    """Test that store_factors raises ValueError for missing required columns."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2022-01-01")], "factor_mom": [0.5]})
    # Missing "symbol" column

    with pytest.raises(ValueError, match="missing required columns"):
        store_factors(freq="1d", group="ta", df=df, root=tmp_path)


@pytest.mark.advanced
def test_load_factors_invalid_date_range(tmp_path: Path) -> None:
    """Test that load_factors raises ValueError when start > end."""
    with pytest.raises(ValueError, match="start.*must be <= end"):
        load_factors(
            freq="1d",
            universe=["AAPL"],
            start=pd.Timestamp("2022-01-05"),
            end=pd.Timestamp("2022-01-01"),  # start > end
            groups=["ta"],
            root=tmp_path,
        )
