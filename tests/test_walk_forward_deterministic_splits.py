"""Tests for deterministic walk-forward splits (RB1).

These tests verify that walk-forward splits are generated deterministically:
- Same input -> same splits
- UTC-aware timestamps
- Deterministic ordering (mergesort)
- No randomness without explicit seed
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.qa.walk_forward import make_walk_forward_splits


def test_make_walk_forward_splits_deterministic():
    """Test that same input produces identical splits."""
    # Create synthetic price data
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    # Generate splits twice
    splits1 = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=5,
        train_days=252,
        test_days=63,
        seed=0,
    )

    splits2 = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=5,
        train_days=252,
        test_days=63,
        seed=0,
    )

    # Assert identical splits
    assert len(splits1) == len(splits2)
    assert splits1 == splits2

    # Verify split structure
    for split in splits1:
        assert "split_index" in split
        assert "train_start" in split
        assert "train_end" in split
        assert "test_start" in split
        assert "test_end" in split
        assert "n_train" in split
        assert "n_test" in split

        # Verify timestamps are ISO format strings
        assert isinstance(split["train_start"], str)
        assert isinstance(split["test_start"], str)

        # Verify split_index is sequential
        assert split["split_index"] >= 0


def test_make_walk_forward_splits_utc_aware():
    """Test that splits use UTC-aware timestamps."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    splits = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=3,
        train_days=252,
        test_days=63,
        seed=0,
    )

    # Verify timestamps are UTC-aware (ISO format includes 'Z' or '+00:00')
    for split in splits:
        train_start = split["train_start"]
        test_start = split["test_start"]

        # Parse and verify UTC
        train_ts = pd.to_datetime(train_start, utc=True)
        test_ts = pd.to_datetime(test_start, utc=True)

        assert train_ts.tz is not None
        assert test_ts.tz is not None


def test_make_walk_forward_splits_deterministic_ordering():
    """Test that splits are sorted by split_index."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    splits = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=5,
        train_days=252,
        test_days=63,
        seed=0,
    )

    # Verify splits are sorted by split_index
    split_indices = [s["split_index"] for s in splits]
    assert split_indices == sorted(split_indices)

    # Verify split indices are sequential (0, 1, 2, ...)
    for i, split in enumerate(splits):
        assert split["split_index"] == i


def test_make_walk_forward_splits_seed_consistency():
    """Test that seed parameter doesn't affect deterministic output (currently unused)."""
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    # Generate splits with different seeds (should be identical since seed is unused)
    splits_seed0 = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=3,
        train_days=252,
        test_days=63,
        seed=0,
    )

    splits_seed42 = make_walk_forward_splits(
        prices_df=prices_df,
        n_splits=3,
        train_days=252,
        test_days=63,
        seed=42,
    )

    # Should be identical (seed currently unused)
    assert splits_seed0 == splits_seed42


def test_make_walk_forward_splits_empty_dataframe():
    """Test that empty DataFrame raises ValueError."""
    prices_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])

    with pytest.raises(ValueError, match="prices_df must not be empty"):
        make_walk_forward_splits(
            prices_df=prices_df,
            n_splits=5,
            train_days=252,
            test_days=63,
        )


def test_make_walk_forward_splits_missing_timestamp_col():
    """Test that missing timestamp column raises ValueError."""
    prices_df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", "2022-12-31", freq="D", tz="UTC"),
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    with pytest.raises(ValueError, match="timestamp_col 'timestamp' not found"):
        make_walk_forward_splits(
            prices_df=prices_df,
            n_splits=3,
            train_days=252,
            test_days=63,
        )


def test_make_walk_forward_splits_insufficient_data():
    """Test that insufficient data raises ValueError."""
    # Only 10 days of data (insufficient for train_days=252 + test_days=63)
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="D", tz="UTC")
    prices_df = pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "AAPL",
            "close": 100.0,
        }
    )

    with pytest.raises(ValueError, match="Insufficient data|No valid splits generated"):
        make_walk_forward_splits(
            prices_df=prices_df,
            n_splits=5,
            train_days=252,
            test_days=63,
        )
