"""Tests for news event storage (Sprint 11.E2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.news.store import (
    load_news_parquet,
    list_news_partitions,
    news_partition_path,
    store_news_parquet,
)


def test_news_partition_path() -> None:
    """Test partition path generation."""
    root = Path("/tmp/news_test")
    path = news_partition_path(root, "reuters", year=2024, month=1)

    assert path.parent.name == "01"
    assert path.parent.parent.name == "2024"
    assert path.parent.parent.parent.name == "reuters"
    assert path.name == "news_reuters_2024_01.parquet"


def test_store_news_parquet_replace_mode(tmp_path: Path) -> None:
    """Test storing news events in replace mode."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test news"],
    })

    partition_path = store_news_parquet(
        df,
        tmp_path,
        "reuters",
        year=2024,
        month=1,
        mode="replace",
    )

    assert partition_path.exists()
    loaded = load_news_parquet(tmp_path, "reuters", year=2024, month=1)
    assert len(loaded) == 1
    assert loaded["headline"].iloc[0] == "Test news"


def test_store_news_parquet_append_mode(tmp_path: Path) -> None:
    """Test storing news events in append mode."""
    df1 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["First"],
        "provider_id": ["1"],
    })

    df2 = pd.DataFrame({
        "publish_ts": ["2024-01-16 10:00:00"],
        "source": ["reuters"],
        "headline": ["Second"],
        "provider_id": ["2"],
    })

    # Store first batch
    store_news_parquet(df1, tmp_path, "reuters", year=2024, month=1, mode="append")

    # Append second batch
    store_news_parquet(df2, tmp_path, "reuters", year=2024, month=1, mode="append")

    # Load and verify
    loaded = load_news_parquet(tmp_path, "reuters", year=2024, month=1)
    assert len(loaded) == 2
    assert set(loaded["headline"].values) == {"First", "Second"}


def test_store_news_parquet_append_deduplicates(tmp_path: Path) -> None:
    """Test that append mode deduplicates existing data."""
    df1 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Same"],
        "provider_id": ["123"],
    })

    df2 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Same"],
        "provider_id": ["123"],  # Same provider_id -> duplicate
    })

    # Store first batch
    store_news_parquet(df1, tmp_path, "reuters", year=2024, month=1, mode="append")

    # Append duplicate
    store_news_parquet(df2, tmp_path, "reuters", year=2024, month=1, mode="append")

    # Load and verify (should be deduplicated)
    loaded = load_news_parquet(tmp_path, "reuters", year=2024, month=1)
    assert len(loaded) == 1


def test_store_news_parquet_empty_raises_error(tmp_path: Path) -> None:
    """Test that storing empty DataFrame raises ValueError."""
    df = pd.DataFrame(columns=["publish_ts", "source", "headline"])

    with pytest.raises(ValueError, match="empty"):
        store_news_parquet(df, tmp_path, "reuters", year=2024, month=1)


def test_load_news_parquet_nonexistent_returns_empty(tmp_path: Path) -> None:
    """Test that loading non-existent partition returns empty DataFrame."""
    loaded = load_news_parquet(tmp_path, "reuters", year=2024, month=1)

    assert loaded.empty


def test_list_news_partitions(tmp_path: Path) -> None:
    """Test listing news partitions."""
    # Store some partitions
    df1 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
    })
    df2 = pd.DataFrame({
        "publish_ts": ["2024-02-15 10:00:00"],
        "source": ["bloomberg"],
        "headline": ["Test"],
    })

    store_news_parquet(df1, tmp_path, "reuters", year=2024, month=1)
    store_news_parquet(df2, tmp_path, "bloomberg", year=2024, month=2)

    # List all partitions
    partitions = list_news_partitions(tmp_path)
    assert len(partitions) == 2

    # List partitions for specific source
    reuters_partitions = list_news_partitions(tmp_path, source="reuters")
    assert len(reuters_partitions) == 1
    assert "reuters" in str(reuters_partitions[0])


def test_atomic_write_no_temp_left_behind(tmp_path: Path) -> None:
    """Test that atomic write doesn't leave temp files behind."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
    })

    partition_path = store_news_parquet(df, tmp_path, "reuters", year=2024, month=1)

    # Check no temp files exist
    temp_files = list(partition_path.parent.glob("*.tmp.parquet"))
    assert len(temp_files) == 0

    # Check final file exists
    assert partition_path.exists()


def test_store_news_parquet_deterministic(tmp_path: Path) -> None:
    """Test that storing same data multiple times produces same result."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "provider_id": ["123"],
    })

    # Store twice
    store_news_parquet(df, tmp_path, "reuters", year=2024, month=1, mode="replace")
    store_news_parquet(df, tmp_path, "reuters", year=2024, month=1, mode="replace")

    # Load and verify deterministic
    loaded1 = load_news_parquet(tmp_path, "reuters", year=2024, month=1)
    loaded2 = load_news_parquet(tmp_path, "reuters", year=2024, month=1)

    pd.testing.assert_frame_equal(loaded1, loaded2)


def test_store_news_parquet_dedupe_keep_first(tmp_path: Path) -> None:
    """Test that dedupe_keep='first' is respected during append."""
    df1 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["First"],
        "provider_id": ["123"],
    })

    df2 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Second"],
        "provider_id": ["123"],  # Same provider_id -> duplicate
    })

    store_news_parquet(df1, tmp_path, "reuters", year=2024, month=1, mode="append", dedupe_keep="first")
    store_news_parquet(df2, tmp_path, "reuters", year=2024, month=1, mode="append", dedupe_keep="first")

    loaded = load_news_parquet(tmp_path, "reuters", year=2024, month=1)
    assert len(loaded) == 1
    assert loaded["headline"].iloc[0] == "First"


def test_store_news_parquet_dedupe_keep_last(tmp_path: Path) -> None:
    """Test that dedupe_keep='last' is respected during append."""
    df1 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["First"],
        "provider_id": ["123"],
    })

    df2 = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Second"],
        "provider_id": ["123"],  # Same provider_id -> duplicate
    })

    store_news_parquet(df1, tmp_path, "reuters", year=2024, month=1, mode="append", dedupe_keep="last")
    store_news_parquet(df2, tmp_path, "reuters", year=2024, month=1, mode="append", dedupe_keep="last")

    loaded = load_news_parquet(tmp_path, "reuters", year=2024, month=1)
    assert len(loaded) == 1
    assert loaded["headline"].iloc[0] == "Second"
