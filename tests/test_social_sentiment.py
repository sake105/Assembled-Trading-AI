"""Tests for M38a: Social Sentiment Aggregation."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.data.altdata.social_sentiment import (
    SentimentConfig,
    aggregate_daily_sentiment,
    add_sentiment_momentum,
    compute_crowd_consensus,
)


def _synthetic_mentions(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = rng.choice(["AAPL", "MSFT", "TSLA"], n)
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    timestamps = rng.choice(dates, n)
    sentiments = rng.uniform(-1.0, 1.0, n)
    return pd.DataFrame({
        "symbol": symbols,
        "timestamp": timestamps,
        "sentiment": sentiments,
    })


@pytest.mark.phase12
class TestAggregateDailySentiment:
    def test_basic_aggregation(self):
        mentions = _synthetic_mentions()
        result = aggregate_daily_sentiment(mentions)
        assert "sentiment_score" in result.columns
        assert "sentiment_volume" in result.columns
        assert "bullish_ratio" in result.columns
        assert "sentiment_dispersion" in result.columns
        assert len(result) > 0

    def test_sentiment_score_range(self):
        mentions = _synthetic_mentions()
        result = aggregate_daily_sentiment(mentions)
        assert all(-1.0 <= s <= 1.0 for s in result["sentiment_score"])

    def test_bullish_ratio_range(self):
        mentions = _synthetic_mentions()
        result = aggregate_daily_sentiment(mentions)
        assert all(0.0 <= r <= 1.0 for r in result["bullish_ratio"])

    def test_min_mentions_filter(self):
        mentions = pd.DataFrame({
            "symbol": ["AAPL"] * 3,
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "sentiment": [0.5, 0.3, 0.2],
        })
        cfg = SentimentConfig(min_mentions=5)
        result = aggregate_daily_sentiment(mentions, config=cfg)
        assert len(result) == 0  # below min threshold

    def test_empty_input(self):
        result = aggregate_daily_sentiment(pd.DataFrame())
        assert len(result) == 0

    def test_multiple_symbols(self):
        mentions = _synthetic_mentions()
        result = aggregate_daily_sentiment(mentions)
        symbols = result["symbol"].unique()
        assert len(symbols) >= 2


@pytest.mark.phase12
class TestSentimentMomentum:
    def test_adds_momentum_column(self):
        mentions = _synthetic_mentions(n=300)
        daily = aggregate_daily_sentiment(mentions, config=SentimentConfig(min_mentions=1))
        result = add_sentiment_momentum(daily, window=3)
        assert "sentiment_momentum_3d" in result.columns

    def test_empty_input(self):
        result = add_sentiment_momentum(pd.DataFrame())
        assert result.empty


@pytest.mark.phase12
class TestCrowdConsensus:
    def test_crowd_signal_values(self):
        mentions = _synthetic_mentions(n=300)
        daily = aggregate_daily_sentiment(mentions, config=SentimentConfig(min_mentions=1))
        result = compute_crowd_consensus(daily)
        assert "crowd_signal" in result.columns
        assert set(result["crowd_signal"].unique()) <= {-1.0, 0.0, 1.0}

    def test_high_bullish_detected(self):
        daily = pd.DataFrame({
            "symbol": ["AAPL"],
            "date": ["2024-01-15"],
            "sentiment_score": [0.8],
            "sentiment_volume": [50],
            "bullish_ratio": [0.9],
            "sentiment_dispersion": [0.1],
        })
        result = compute_crowd_consensus(daily, threshold_bullish=0.7)
        assert result["crowd_signal"].iloc[0] == 1.0

    def test_empty_input(self):
        result = compute_crowd_consensus(pd.DataFrame())
        assert result.empty
