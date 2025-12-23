"""Tests for Phase 6 event-based features (Insider, Congress, Shipping, News).

These tests verify that the skeleton implementations for event data ingestion
and feature engineering work correctly with dummy data.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.assembled_core.data.congress_trades_ingest import load_congress_sample
from src.assembled_core.data.insider_ingest import (
    load_insider_sample,
    normalize_insider,
)
from src.assembled_core.data.news_ingest import load_news_sample
from src.assembled_core.data.shipping_routes_ingest import load_shipping_sample
from src.assembled_core.features.congress_features import add_congress_features
from src.assembled_core.features.insider_features import add_insider_features
from src.assembled_core.features.news_features import add_news_features
from src.assembled_core.features.shipping_features import add_shipping_features


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create a sample price DataFrame with 3 symbols and 10 days of data."""
    now = datetime.now(timezone.utc)
    symbols = ["AAPL", "MSFT", "GOOGL"]

    data = []
    for symbol in symbols:
        for i in range(10):
            data.append(
                {
                    "timestamp": now - timedelta(days=10 - i),
                    "symbol": symbol,
                    "close": 100.0 + i * 0.5,
                    "open": 100.0 + i * 0.5,
                    "high": 101.0 + i * 0.5,
                    "low": 99.0 + i * 0.5,
                    "volume": 1000000,
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.mark.phase6
@pytest.mark.unit
def test_insider_ingest_load_sample():
    """Test that load_insider_sample generates dummy data correctly."""
    df = load_insider_sample()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "trades_count" in df.columns
    assert "net_shares" in df.columns
    assert "role" in df.columns

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["net_shares"].dtype in ["float64", "int64"]


@pytest.mark.phase6
@pytest.mark.unit
def test_insider_ingest_normalize():
    """Test that normalize_insider works correctly."""
    df = load_insider_sample()
    normalized = normalize_insider(df)

    assert isinstance(normalized, pd.DataFrame)
    assert len(normalized) == len(df)
    assert "timestamp" in normalized.columns
    assert "symbol" in normalized.columns
    assert "trades_count" in normalized.columns
    assert "net_shares" in normalized.columns
    assert "role" in normalized.columns


@pytest.mark.phase6
@pytest.mark.unit
def test_insider_features_basic(sample_prices):
    """Test that add_insider_features adds features correctly."""
    events = load_insider_sample()
    result = add_insider_features(sample_prices, events)

    # Check that result has same length as input
    assert len(result) == len(sample_prices)

    # Check that feature columns exist
    assert "insider_net_buy_20d" in result.columns
    assert "insider_trade_count_20d" in result.columns
    assert "insider_net_buy_60d" in result.columns
    assert "insider_trade_count_60d" in result.columns

    # Check that features are numeric
    assert pd.api.types.is_numeric_dtype(result["insider_net_buy_20d"])
    assert pd.api.types.is_integer_dtype(result["insider_trade_count_20d"])


@pytest.mark.phase6
@pytest.mark.unit
def test_congress_ingest_load_sample():
    """Test that load_congress_sample generates dummy data correctly."""
    df = load_congress_sample()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "politician" in df.columns
    assert "party" in df.columns
    assert "amount" in df.columns


@pytest.mark.phase6
@pytest.mark.unit
def test_congress_features_basic(sample_prices):
    """Test that add_congress_features adds features correctly."""
    events = load_congress_sample()
    result = add_congress_features(sample_prices, events)

    # Check that result has same length as input
    assert len(result) == len(sample_prices)

    # Check that feature columns exist
    assert "congress_trade_count_60d" in result.columns
    assert "congress_total_amount_60d" in result.columns
    assert "congress_trade_count_90d" in result.columns
    assert "congress_total_amount_90d" in result.columns

    # Check that features are numeric
    assert pd.api.types.is_numeric_dtype(result["congress_total_amount_60d"])


@pytest.mark.phase6
@pytest.mark.unit
def test_shipping_ingest_load_sample():
    """Test that load_shipping_sample generates dummy data correctly."""
    df = load_shipping_sample()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "route_id" in df.columns
    assert "port_from" in df.columns
    assert "port_to" in df.columns
    assert "ships" in df.columns
    assert "congestion_score" in df.columns


@pytest.mark.phase6
@pytest.mark.unit
def test_shipping_features_basic(sample_prices):
    """Test that add_shipping_features adds features correctly."""
    events = load_shipping_sample()
    result = add_shipping_features(sample_prices, events)

    # Check that result has same length as input
    assert len(result) == len(sample_prices)

    # Check that feature columns exist
    assert "shipping_congestion_score" in result.columns
    assert "shipping_ships_count" in result.columns
    assert "shipping_congestion_score_7d" in result.columns
    assert "shipping_ships_count_7d" in result.columns

    # Check that features are numeric (may contain NaN)
    assert (
        pd.api.types.is_numeric_dtype(result["shipping_congestion_score"])
        or result["shipping_congestion_score"].isna().any()
    )


@pytest.mark.phase6
@pytest.mark.unit
def test_news_ingest_load_sample():
    """Test that load_news_sample generates dummy data correctly."""
    df = load_news_sample()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "timestamp" in df.columns
    assert "symbol" in df.columns
    assert "headline" in df.columns
    assert "sentiment_score" in df.columns
    assert "source" in df.columns

    # Check sentiment_score is in range [-1, 1]
    assert df["sentiment_score"].min() >= -1.0
    assert df["sentiment_score"].max() <= 1.0


@pytest.mark.phase6
@pytest.mark.unit
def test_news_features_basic(sample_prices):
    """Test that add_news_features adds features correctly."""
    events = load_news_sample()
    result = add_news_features(sample_prices, events)

    # Check that result has same length as input
    assert len(result) == len(sample_prices)

    # Check that feature columns exist
    assert "news_sentiment_7d" in result.columns
    assert "news_sentiment_30d" in result.columns
    assert "news_count_7d" in result.columns
    assert "news_count_30d" in result.columns

    # Check that features are numeric (may contain NaN for sentiment)
    assert pd.api.types.is_integer_dtype(result["news_count_7d"])


@pytest.mark.phase6
@pytest.mark.unit
def test_insider_features_no_events(sample_prices):
    """Test that add_insider_features handles empty events gracefully."""
    empty_events = pd.DataFrame(
        columns=["timestamp", "symbol", "net_shares", "trades_count"]
    )
    result = add_insider_features(sample_prices, empty_events)

    assert len(result) == len(sample_prices)
    assert (result["insider_net_buy_20d"] == 0.0).all()
    assert (result["insider_trade_count_20d"] == 0).all()


@pytest.mark.phase6
@pytest.mark.unit
def test_news_features_sentiment_range(sample_prices):
    """Test that news sentiment features are in valid range."""
    events = load_news_sample()
    result = add_news_features(sample_prices, events)

    # Check that sentiment values are in range [-1, 1] or NaN
    sentiment_cols = ["news_sentiment_7d", "news_sentiment_30d"]
    for col in sentiment_cols:
        valid_values = result[col].dropna()
        if len(valid_values) > 0:
            assert valid_values.min() >= -1.0
            assert valid_values.max() <= 1.0
