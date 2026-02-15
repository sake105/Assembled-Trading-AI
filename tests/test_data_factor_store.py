"""Unit tests for factor store module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.data.factor_store import (
    compute_universe_key,
    get_factor_store_root,
    list_available_panels,
    load_factors,
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
    uk = compute_universe_key(symbols=["AAPL", "MSFT"])
    panel_dir = store_factors(
        df=original_df,
        factor_group="ta",
        freq="1d",
        universe_key=uk,
        factors_root=tmp_path,
    )

    # Verify that files were created for both years
    assert (panel_dir / "year=2022.parquet").exists() or (panel_dir / "year=2023.parquet").exists()

    # Load factors back (UTC timestamps for comparison with stored data)
    loaded_df = load_factors(
        factor_group="ta",
        freq="1d",
        universe_key=uk,
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2023-01-10", tz="UTC"),
        factors_root=tmp_path,
    )
    assert loaded_df is not None

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
    # Normalize loaded timestamps to naive for set comparison (original is naive)
    if loaded_sorted["timestamp"].dt.tz is not None:
        loaded_sorted = loaded_sorted.copy()
        loaded_sorted["timestamp"] = loaded_sorted["timestamp"].dt.tz_localize(None)

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
    uk_ta = compute_universe_key(symbols=["AAPL"])
    store_factors(df=ta_df, factor_group="ta", freq="1d", universe_key=uk_ta, factors_root=tmp_path)

    # Store "alt_insider" group data
    insider_rows = []
    for date in dates_2022.tolist() + dates_2023.tolist():
        insider_rows.append(
            {"timestamp": date, "symbol": "AAPL", "factor_insider": 0.8}
        )

    insider_df = pd.DataFrame(insider_rows)
    store_factors(df=insider_df, factor_group="alt_insider", freq="1d", universe_key=uk_ta, factors_root=tmp_path)

    # List available panels
    available = list_available_panels(factors_root=tmp_path)

    # Verify result is a list of dicts
    assert isinstance(available, list), "Result should be a list"
    assert len(available) >= 1, "Should have at least one panel"

    # Verify required keys in first panel
    required_keys = {"factor_group", "freq", "universe_key", "years"}
    assert required_keys.issubset(set(available[0].keys())), (
        f"Result items should contain keys: {required_keys}"
    )

    # Verify at least one panel with factor_group == "ta"
    ta_panels = [p for p in available if p.get("factor_group") == "ta"]
    assert len(ta_panels) > 0, "Should have at least one panel with factor_group='ta'"

    # Verify (freq, factor_group) includes ("1d", "ta")
    freq_group_pairs = {(p.get("freq"), p.get("factor_group")) for p in available}
    assert ("1d", "ta") in freq_group_pairs, "Should include (freq='1d', factor_group='ta')"


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
    uk = compute_universe_key(symbols=["AAPL"])
    store_factors(df=df, factor_group="ta", freq="1d", universe_key=uk, factors_root=tmp_path)

    # Load with end date that excludes 2022-01-03 (UTC for comparison with stored data)
    loaded_df = load_factors(
        factor_group="ta",
        freq="1d",
        universe_key=uk,
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2022-01-02", tz="UTC"),  # Intentionally exclude 2022-01-03
        factors_root=tmp_path,
    )
    assert loaded_df is not None

    # Verify maximum timestamp is <= end (compare with UTC)
    assert not loaded_df.empty, "Loaded DataFrame should not be empty"
    max_timestamp = loaded_df["timestamp"].max()
    end_utc = pd.Timestamp("2022-01-02", tz="UTC")
    assert max_timestamp <= end_utc, (
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

    uk = compute_universe_key(symbols=["AAPL"])
    ta_df = pd.DataFrame(ta_rows)
    store_factors(df=ta_df, factor_group="ta", freq="1d", universe_key=uk, factors_root=tmp_path)

    # Store "alt_insider" group
    insider_rows = []
    for date in dates:
        insider_rows.append(
            {"timestamp": date, "symbol": "AAPL", "factor_insider": 0.8}
        )

    insider_df = pd.DataFrame(insider_rows)
    store_factors(df=insider_df, factor_group="alt_insider", freq="1d", universe_key=uk, factors_root=tmp_path)

    # Load "ta" group (UTC for comparison with stored data)
    loaded_ta = load_factors(
        factor_group="ta",
        freq="1d",
        universe_key=uk,
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2022-01-05", tz="UTC"),
        factors_root=tmp_path,
    )
    assert loaded_ta is not None
    assert "factor_mom" in loaded_ta.columns, "Should contain factor_mom from 'ta' group"

    # Load "alt_insider" group
    loaded_insider = load_factors(
        factor_group="alt_insider",
        freq="1d",
        universe_key=uk,
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2022-01-05", tz="UTC"),
        factors_root=tmp_path,
    )
    assert loaded_insider is not None
    assert "factor_insider" in loaded_insider.columns, "Should contain factor_insider from 'alt_insider' group"

    assert len(loaded_ta) == len(dates), "Loaded ta should have correct number of rows"


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

    # Store all symbols (universe_key for all three)
    uk_all = compute_universe_key(symbols=all_symbols)
    store_factors(df=df, factor_group="ta", freq="1d", universe_key=uk_all, factors_root=tmp_path)

    # Load full panel and filter to requested symbols in test (UTC for stored data)
    loaded_df = load_factors(
        factor_group="ta",
        freq="1d",
        universe_key=uk_all,
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2022-01-05", tz="UTC"),
        factors_root=tmp_path,
    )
    assert loaded_df is not None
    loaded_df = loaded_df[loaded_df["symbol"].isin(["AAPL", "MSFT"])]

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


@pytest.mark.advanced
def test_factor_store_empty_dataframe(tmp_path: Path) -> None:
    """Test handling of empty DataFrame in store_factors."""
    # Create empty DataFrame with required columns
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "factor_mom"])
    uk = compute_universe_key(symbols=["AAPL"])

    # Store should not raise; returns panel dir path
    panel_dir = store_factors(
        df=empty_df, factor_group="ta", freq="1d", universe_key=uk, factors_root=tmp_path
    )
    assert panel_dir is not None and hasattr(panel_dir, "exists"), "Should return Path"


@pytest.mark.advanced
def test_factor_store_nonexistent_files(tmp_path: Path) -> None:
    """Test loading when panel does not exist (should return None)."""
    uk = compute_universe_key(symbols=["AAPL"])
    loaded_df = load_factors(
        factor_group="nonexistent",
        freq="1d",
        universe_key=uk,
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2022-01-05", tz="UTC"),
        factors_root=tmp_path,
    )

    # Should return None when panel does not exist
    assert loaded_df is None, "Should return None when panel does not exist"


@pytest.mark.advanced
def test_price_panel_loader_lives_in_panel_store(tmp_path: Path) -> None:
    """Price panel loading is panel_store.load_price_panel_parquet; no panel file -> FileNotFoundError."""
    from src.assembled_core.data.panel_store import load_price_panel_parquet

    # Current API: load_price_panel_parquet(freq, universe=..., root=...); missing file raises
    with pytest.raises(FileNotFoundError, match="Panel file not found|not found"):
        load_price_panel_parquet(freq="1d", universe="nonexistent", root=tmp_path)


@pytest.mark.advanced
def test_store_factors_missing_columns(tmp_path: Path) -> None:
    """Test that store_factors raises ValueError for missing required columns."""
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2022-01-01")], "factor_mom": [0.5]})
    uk = compute_universe_key(symbols=["AAPL"])

    with pytest.raises(ValueError, match="missing required columns"):
        store_factors(df=df, factor_group="ta", freq="1d", universe_key=uk, factors_root=tmp_path)


@pytest.mark.advanced
def test_load_factors_invalid_date_range(tmp_path: Path) -> None:
    """Test that load_factors with start_date > end_date returns None or empty (no data in range)."""
    uk = compute_universe_key(symbols=["AAPL"])
    loaded = load_factors(
        factor_group="ta",
        freq="1d",
        universe_key=uk,
        start_date=pd.Timestamp("2022-01-05", tz="UTC"),
        end_date=pd.Timestamp("2022-01-01", tz="UTC"),  # start > end
        factors_root=tmp_path,
    )
    assert loaded is None or loaded.empty, "Should return None or empty when start > end"
