"""Tests for macro release contract (Sprint 11.E3)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.macro.contract import (
    filter_macro_pit,
    normalize_macro_releases,
)


def test_required_columns_validation() -> None:
    """Test that missing required columns raise ValueError."""
    df = pd.DataFrame({"series_id": ["GDP_US"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_macro_releases(df)


def test_metric_alternative_to_series_id() -> None:
    """Test that 'metric' can be used as alternative to 'series_id'."""
    df = pd.DataFrame({
        "metric": ["GDP_US"],
        "release_ts": ["2024-01-15 08:30:00"],
        "available_ts": ["2024-01-15 08:30:00"],
        "value": [2.5],
    })

    result = normalize_macro_releases(df)

    assert "series_id" in result.columns
    assert result["series_id"].iloc[0] == "GDP_US"


def test_utc_normalization_naive_timestamps() -> None:
    """Test that naive timestamps are normalized to UTC."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"],
        "release_ts": ["2024-01-15 08:30:00"],  # Naive
        "available_ts": ["2024-01-15 08:30:00"],  # Naive
        "value": [2.5],
    })

    result = normalize_macro_releases(df)

    assert result["release_ts"].iloc[0].tz is not None
    assert str(result["release_ts"].iloc[0].tz) == "UTC"
    assert result["available_ts"].iloc[0].tz is not None
    assert str(result["available_ts"].iloc[0].tz) == "UTC"


def test_timestamp_sanity_invalid_availability_raises_error() -> None:
    """Test that available_ts < release_ts raises ValueError."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"],
        "release_ts": ["2024-01-15 08:30:00"],
        "available_ts": ["2024-01-15 08:00:00"],  # Earlier than release_ts
        "value": [2.5],
    })

    with pytest.raises(ValueError, match="available_ts < release_ts"):
        normalize_macro_releases(df)


def test_timestamp_sanity_valid_availability_passes() -> None:
    """Test that available_ts >= release_ts passes validation."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"],
        "release_ts": ["2024-01-15 08:00:00"],
        "available_ts": ["2024-01-15 08:30:00"],  # Later than release_ts
        "value": [2.5],
    })

    result = normalize_macro_releases(df)

    assert len(result) == 1


def test_deduplication_deterministic() -> None:
    """Test that deduplication is deterministic."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"] * 2,
        "release_ts": ["2024-01-15 08:30:00"] * 2,
        "available_ts": ["2024-01-15 08:30:00"] * 2,
        "value": [2.5, 2.5],
    })

    result1 = normalize_macro_releases(df, dedupe_keep="first")
    result2 = normalize_macro_releases(df, dedupe_keep="first")

    assert len(result1) == 1
    assert len(result2) == 1
    pd.testing.assert_frame_equal(result1, result2)


def test_deduplication_with_revision_id() -> None:
    """Test that revision_id is included in deduplication key."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"] * 2,
        "release_ts": ["2024-01-15 08:30:00"] * 2,
        "available_ts": ["2024-01-15 08:30:00", "2024-01-15 09:00:00"],
        "value": [2.5, 2.6],
        "revision_id": ["initial", "rev1"],  # Different revisions
    })

    result = normalize_macro_releases(df, dedupe_keep="first")

    assert len(result) == 2  # Both revisions kept


def test_filter_macro_pit_future_availability_filtered() -> None:
    """Test that releases with future available_ts are filtered out."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"] * 3,
        "release_ts": pd.to_datetime(["2024-01-15 08:30:00"] * 3, utc=True),
        "available_ts": pd.to_datetime([
            "2024-01-15 08:30:00",
            "2024-01-20 08:30:00",  # Future
            "2024-01-10 08:30:00",
        ], utc=True),
        "value": [2.5, 2.6, 2.4],
    })

    as_of = pd.Timestamp("2024-01-18", tz="UTC")
    filtered = filter_macro_pit(df, as_of)

    assert len(filtered) == 2
    assert "2024-01-20" not in filtered["available_ts"].astype(str).values


def test_filter_macro_pit_inclusive_boundary() -> None:
    """Test that available_ts == as_of is included."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"],
        "release_ts": pd.to_datetime(["2024-01-15 08:30:00"], utc=True),
        "available_ts": pd.to_datetime(["2024-01-15 08:30:00"], utc=True),
        "value": [2.5],
    })

    as_of = pd.Timestamp("2024-01-15 08:30:00", tz="UTC")
    filtered = filter_macro_pit(df, as_of)

    assert len(filtered) == 1


def test_filter_macro_pit_missing_column_raises_error() -> None:
    """Test that filter_macro_pit raises ValueError if available_ts missing."""
    df = pd.DataFrame({
        "series_id": ["GDP_US"],
        "release_ts": ["2024-01-15 08:30:00"],
        "value": [2.5],
    })

    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    with pytest.raises(ValueError, match="available_ts"):
        filter_macro_pit(df, as_of)


def test_deterministic_sorting() -> None:
    """Test that output is deterministically sorted."""
    df = pd.DataFrame({
        "series_id": ["B", "A", "C"],
        "release_ts": [
            "2024-01-20 08:30:00",
            "2024-01-15 08:30:00",
            "2024-01-18 08:30:00",
        ],
        "available_ts": [
            "2024-01-20 08:30:00",
            "2024-01-15 08:30:00",
            "2024-01-18 08:30:00",
        ],
        "value": [2.7, 2.5, 2.6],
    })

    result1 = normalize_macro_releases(df)
    result2 = normalize_macro_releases(df)

    pd.testing.assert_frame_equal(result1, result2)


def test_string_trimming() -> None:
    """Test that string columns are trimmed."""
    df = pd.DataFrame({
        "series_id": ["  GDP_US  "],
        "release_ts": ["2024-01-15 08:30:00"],
        "available_ts": ["2024-01-15 08:30:00"],
        "value": [2.5],
        "country": ["  US  "],
    })

    result = normalize_macro_releases(df)

    assert result["series_id"].iloc[0] == "GDP_US"
    assert result["country"].iloc[0] == "US"


def test_empty_dataframe_handling() -> None:
    """Test that empty DataFrame is handled gracefully."""
    df = pd.DataFrame(columns=["series_id", "release_ts", "available_ts", "value"])

    result = normalize_macro_releases(df)

    assert result.empty
    expected_cols = set(["series_id", "release_ts", "available_ts", "value"] + [
        "country", "currency", "source", "revision_id", "metric"
    ])
    assert set(result.columns) == expected_cols
