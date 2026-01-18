"""Tests for news entity linking (Sprint 11.E2)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.data.news.entity_linking import link_news_to_symbols


def test_link_news_existing_symbol_passthrough() -> None:
    """Test that existing 'symbol' column is passed through (trimmed)."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "symbol": ["  AAPL  "],  # Has whitespace
    })

    result = link_news_to_symbols(df)

    assert result["symbol"].iloc[0] == "AAPL"


def test_link_news_ticker_mapping_success() -> None:
    """Test that ticker is mapped to symbol via mapping_df."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "ticker": ["AAPL"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL", "MSFT"],
        "symbol": ["AAPL", "MSFT"],
    })

    result = link_news_to_symbols(df, mapping_df=mapping_df)

    assert result["symbol"].iloc[0] == "AAPL"


def test_link_news_entity_mapping_success() -> None:
    """Test that entity is mapped to symbol via mapping_df."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "entity": ["Apple Inc"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["Apple Inc", "Microsoft Corp"],
        "symbol": ["AAPL", "MSFT"],
    })

    result = link_news_to_symbols(df, mapping_df=mapping_df)

    assert result["symbol"].iloc[0] == "AAPL"


def test_link_news_security_master_mapping() -> None:
    """Test that ticker is mapped via security_master_df if it matches a symbol."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "ticker": ["AAPL"],
    })

    security_master_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "sector": ["Technology", "Technology"],
        "region": ["US", "US"],
        "currency": ["USD", "USD"],
        "asset_type": ["Equity", "Equity"],
    })

    result = link_news_to_symbols(df, security_master_df=security_master_df)

    assert result["symbol"].iloc[0] == "AAPL"


def test_link_news_missing_mapping_raise() -> None:
    """Test that missing mapping raises ValueError when missing='raise'."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "ticker": ["UNKNOWN_TICKER"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL"],
        "symbol": ["AAPL"],
    })

    with pytest.raises(ValueError, match="Cannot map"):
        link_news_to_symbols(df, mapping_df=mapping_df, missing="raise")


def test_link_news_missing_mapping_drop() -> None:
    """Test that missing mapping drops rows when missing='drop'."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00", "2024-01-16 10:00:00"],
        "source": ["reuters"] * 2,
        "headline": ["Test1", "Test2"],
        "ticker": ["AAPL", "UNKNOWN_TICKER"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL"],
        "symbol": ["AAPL"],
    })

    result = link_news_to_symbols(df, mapping_df=mapping_df, missing="drop")

    assert len(result) == 1
    assert result["symbol"].iloc[0] == "AAPL"


def test_link_news_missing_mapping_keep_unknown() -> None:
    """Test that missing mapping sets symbol='UNKNOWN' when missing='keep_unknown'."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "ticker": ["UNKNOWN_TICKER"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL"],
        "symbol": ["AAPL"],
    })

    result = link_news_to_symbols(df, mapping_df=mapping_df, missing="keep_unknown")

    assert result["symbol"].iloc[0] == "UNKNOWN"


def test_link_news_deterministic_sorting() -> None:
    """Test that output is deterministically sorted."""
    df = pd.DataFrame({
        "publish_ts": [
            "2024-01-20 10:00:00",
            "2024-01-15 10:00:00",
            "2024-01-18 10:00:00",
        ],
        "source": ["reuters", "bloomberg", "reuters"],
        "headline": ["C", "A", "B"],
        "ticker": ["AAPL", "MSFT", "AAPL"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL", "MSFT"],
        "symbol": ["AAPL", "MSFT"],
    })

    result1 = link_news_to_symbols(df, mapping_df=mapping_df)
    result2 = link_news_to_symbols(df, mapping_df=mapping_df)

    pd.testing.assert_frame_equal(result1, result2)


def test_link_news_multiple_symbols() -> None:
    """Test that multiple news events are correctly mapped."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00", "2024-01-16 10:00:00"],
        "source": ["reuters"] * 2,
        "headline": ["Test1", "Test2"],
        "ticker": ["AAPL", "MSFT"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL", "MSFT"],
        "symbol": ["AAPL", "MSFT"],
    })

    result = link_news_to_symbols(df, mapping_df=mapping_df)

    assert len(result) == 2
    assert set(result["symbol"].values) == {"AAPL", "MSFT"}


def test_link_news_empty_dataframe() -> None:
    """Test that empty DataFrame is handled gracefully."""
    df = pd.DataFrame(columns=["publish_ts", "source", "headline", "ticker"])

    result = link_news_to_symbols(df)

    assert result.empty


def test_link_news_no_entity_columns() -> None:
    """Test that news without ticker/entity columns still works."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        # No ticker or entity column
    })

    result = link_news_to_symbols(df)

    assert "symbol" in result.columns
    assert result["symbol"].iloc[0] is pd.NA or pd.isna(result["symbol"].iloc[0])


def test_link_news_mapping_df_priority() -> None:
    """Test that mapping_df takes priority over security_master_df."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "ticker": ["AAPL"],
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL"],
        "symbol": ["AAPL_MAPPED"],  # Different from security master
    })

    security_master_df = pd.DataFrame({
        "symbol": ["AAPL"],
        "sector": ["Technology"],
        "region": ["US"],
        "currency": ["USD"],
        "asset_type": ["Equity"],
    })

    result = link_news_to_symbols(
        df,
        mapping_df=mapping_df,
        security_master_df=security_master_df,
    )

    assert result["symbol"].iloc[0] == "AAPL_MAPPED"


def test_link_news_ticker_trimmed() -> None:
    """Test that ticker values are trimmed before mapping."""
    df = pd.DataFrame({
        "publish_ts": ["2024-01-15 10:00:00"],
        "source": ["reuters"],
        "headline": ["Test"],
        "ticker": ["  AAPL  "],  # Has whitespace
    })

    mapping_df = pd.DataFrame({
        "entity": ["AAPL"],  # No whitespace
        "symbol": ["AAPL"],
    })

    result = link_news_to_symbols(df, mapping_df=mapping_df)

    assert result["symbol"].iloc[0] == "AAPL"
