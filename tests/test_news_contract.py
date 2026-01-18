"""Tests for news event contract (Sprint 11.E2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.news.contract import (
    filter_news_pit,
    normalize_news_events,
)


def test_required_columns_validation() -> None:
    """Test that missing required columns raise ValueError."""
    df = pd.DataFrame({"headline": ["Test news"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        normalize_news_events(df)


def test_utc_normalization_naive_timestamps() -> None:
    """Test that naive timestamps are normalized to UTC."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],  # Naive
        "source": ["reuters"],
        "headline": ["Test"],
    })

    result = normalize_news_events(df)

    assert result["publish_ts"].iloc[0].tz is not None
    assert str(result["publish_ts"].iloc[0].tz) == "UTC"


def test_utc_normalization_tz_aware_timestamps() -> None:
    """Test that tz-aware timestamps are converted to UTC."""
    df = pd.DataFrame({
        "publish_ts": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
        "source": ["reuters"],
        "headline": ["Test"],
    })

    result = normalize_news_events(df)

    assert result["publish_ts"].iloc[0].tz is not None
    assert str(result["publish_ts"].iloc[0].tz) == "UTC"


def test_timestamp_sanity_future_publish_raises_error() -> None:
    """Test that publish_ts in future relative to ingest_ts raises ValueError."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-20 10:00:00"],
        "ingest_ts": ["2024-01-15 10:00:00"],  # Earlier than publish_ts
        "source": ["reuters"],
        "headline": ["Test"],
    })

    with pytest.raises(ValueError, match="publish_ts in future"):
        normalize_news_events(df)


def test_timestamp_sanity_invalid_revision_raises_error() -> None:
    """Test that revised_ts < publish_ts raises ValueError."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-20 10:00:00"],
        "revised_ts": ["2024-01-15 10:00:00"],  # Earlier than publish_ts
        "source": ["reuters"],
        "headline": ["Test"],
    })

    with pytest.raises(ValueError, match="revised_ts < publish_ts"):
        normalize_news_events(df)


def test_timestamp_sanity_valid_revision_passes() -> None:
    """Test that revised_ts >= publish_ts passes validation."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "revised_ts": ["2024-01-20 10:00:00"],  # Later than publish_ts
        "source": ["reuters"],
        "headline": ["Test"],
    })

    result = normalize_news_events(df)

    assert len(result) == 1


def test_missing_identifier_raises_error() -> None:
    """Test that missing identifier (headline/url/provider_id) raises ValueError."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        # No headline, url, or provider_id
    })

    with pytest.raises(ValueError, match="identifier"):
        normalize_news_events(df)


def test_deduplication_deterministic() -> None:
    """Test that deduplication is deterministic."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"] * 3,
        "source": ["reuters"] * 3,
        "provider_id": ["123"] * 3,  # Same provider_id -> duplicates
        "headline": ["Test"] * 3,
    })

    result1 = normalize_news_events(df, dedupe_keep="first")
    result2 = normalize_news_events(df, dedupe_keep="first")

    assert len(result1) == 1
    assert len(result2) == 1
    pd.testing.assert_frame_equal(result1, result2)


def test_deduplication_keep_first() -> None:
    """Test that dedupe_keep='first' keeps first occurrence."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"] * 2,
        "source": ["reuters"] * 2,
        "provider_id": ["123"] * 2,
        "headline": ["First", "Second"],
        "ingest_ts": ["2024-01-15 11:00:00", "2024-01-15 12:00:00"],  # After publish_ts
    })

    result = normalize_news_events(df, dedupe_keep="first")

    assert len(result) == 1
    assert result["headline"].iloc[0] == "First"


def test_deduplication_keep_last() -> None:
    """Test that dedupe_keep='last' keeps last occurrence."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"] * 2,
        "source": ["reuters"] * 2,
        "provider_id": ["123"] * 2,
        "headline": ["First", "Second"],
        "ingest_ts": ["2024-01-15 11:00:00", "2024-01-15 12:00:00"],  # After publish_ts
    })

    result = normalize_news_events(df, dedupe_keep="last")

    assert len(result) == 1
    assert result["headline"].iloc[0] == "Second"


def test_deduplication_hash_fallback() -> None:
    """Test that deduplication uses hash fallback when provider_id missing."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"] * 2,
        "source": ["reuters"] * 2,
        "headline": ["Same headline"] * 2,  # Same headline -> should dedupe
    })

    result = normalize_news_events(df, dedupe_keep="first")

    assert len(result) == 1


def test_string_trimming() -> None:
    """Test that string columns are trimmed."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["  reuters  "],  # Has whitespace
        "headline": ["  Test  "],
    })

    result = normalize_news_events(df)

    assert result["source"].iloc[0] == "reuters"
    assert result["headline"].iloc[0] == "Test"


def test_deterministic_sorting() -> None:
    """Test that output is deterministically sorted."""
    df = pd.DataFrame({
        "publish_ts": [
            "2024-01-20 10:00:00",
            "2024-01-15 10:00:00",
            "2024-01-18 10:00:00",
        ],
        "source": ["reuters", "bloomberg", "reuters"],
        "headline": ["C", "A", "B"],
    })

    result1 = normalize_news_events(df)
    result2 = normalize_news_events(df)

    pd.testing.assert_frame_equal(result1, result2)


def test_filter_news_pit_future_publish_filtered() -> None:
    """Test that events with future publish_ts are filtered out."""
    df = pd.DataFrame({
        "publish_ts": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-01-20 10:00:00",  # Future
            "2024-01-10 10:00:00",
        ], utc=True),
        "source": ["reuters"] * 3,
        "headline": ["A", "B", "C"],
    })

    as_of = pd.Timestamp("2024-01-18", tz="UTC")
    filtered = filter_news_pit(df, as_of)

    assert len(filtered) == 2
    assert "2024-01-20" not in filtered["publish_ts"].astype(str).values


def test_filter_news_pit_inclusive_boundary() -> None:
    """Test that publish_ts == as_of is included."""
    df = pd.DataFrame({
        "publish_ts": pd.to_datetime(["2024-01-15 10:00:00"], utc=True),
        "source": ["reuters"],
        "headline": ["Test"],
    })

    as_of = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
    filtered = filter_news_pit(df, as_of)

    assert len(filtered) == 1


def test_filter_news_pit_missing_column_raises_error() -> None:
    """Test that filter_news_pit raises ValueError if publish_ts missing."""
    df = pd.DataFrame({
        "source": ["reuters"],
        "headline": ["Test"],
    })

    as_of = pd.Timestamp("2024-01-15", tz="UTC")

    with pytest.raises(ValueError, match="publish_ts"):
        filter_news_pit(df, as_of)


def test_empty_dataframe_handling() -> None:
    """Test that empty DataFrame is handled gracefully."""
    df = pd.DataFrame(columns=["publish_ts", "source", "headline"])

    result = normalize_news_events(df)

    assert result.empty
    # Should have required columns + optional columns
    expected_cols = set(["publish_ts", "source"] + [
        "symbol", "symbols", "headline", "url", "provider_id",
        "ingest_ts", "revised_ts", "sentiment", "raw_url"
    ])
    assert set(result.columns) == expected_cols


def test_optional_columns_preserved() -> None:
    """Test that optional columns are preserved."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "symbol": ["AAPL"],
        "sentiment": ["positive"],
    })

    result = normalize_news_events(df)

    assert "symbol" in result.columns
    assert "sentiment" in result.columns
    assert result["symbol"].iloc[0] == "AAPL"
