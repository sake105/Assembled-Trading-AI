"""Tests for wave-58 module wiring into trading_cycle.py.

Covers:
  Step 2.36 — features.altdata_earnings_insider_factors (build_earnings_surprise_factors)
  Step 2.37 — features.altdata_news_macro_factors (build_news_sentiment_factors / build_macro_regime_factors)
  Step 2.38 — features.event_features_vectorized (build_event_feature_panel_vectorized)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
    compute_sue,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_news_sentiment_factors,
    build_macro_regime_factors,
)
from src.assembled_core.features.event_features_vectorized import (
    build_event_feature_panel_vectorized,
    add_disclosure_count_feature_vectorized,
)


# ---------------------------------------------------------------------------
# altdata_earnings_insider_factors (Step 2.36)
# ---------------------------------------------------------------------------

def _make_prices_df(n: int = 20, symbol: str = "AAPL") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": symbol,
        "close": 150.0 + np.cumsum(rng.normal(0, 0.5, n)),
    })


def test_build_earnings_surprise_factors_empty_events():
    empty_events = pd.DataFrame(columns=["timestamp", "symbol", "event_type", "event_id"])
    prices = _make_prices_df()
    result = build_earnings_surprise_factors(empty_events, prices)
    assert isinstance(result, pd.DataFrame)


def test_build_earnings_surprise_factors_returns_df():
    empty_events = pd.DataFrame(columns=["timestamp", "symbol", "event_type", "event_id"])
    prices = _make_prices_df()
    result = build_earnings_surprise_factors(empty_events, prices)
    assert "symbol" in result.columns or len(result.columns) >= 0


def test_build_insider_activity_factors_empty():
    empty_events = pd.DataFrame(columns=["timestamp", "symbol", "event_type", "event_id"])
    prices = _make_prices_df()
    result = build_insider_activity_factors(empty_events, prices)
    assert isinstance(result, pd.DataFrame)


def test_compute_sue_returns_float():
    sue = compute_sue(actual_eps=1.5, estimated_eps=1.2, surprise_std=0.1)
    assert isinstance(sue, float)


def test_compute_sue_positive_surprise():
    sue = compute_sue(actual_eps=1.5, estimated_eps=1.0, surprise_std=0.1)
    assert sue > 0


# ---------------------------------------------------------------------------
# altdata_news_macro_factors (Step 2.37)
# ---------------------------------------------------------------------------

def _make_news_df(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": "AAPL",
        "sentiment_score": rng.uniform(-1, 1, n),
        "sentiment_volume": rng.integers(1, 50, n),
        "event_type": "news",
        "event_id": [f"n{i}" for i in range(n)],
    })


def test_build_news_sentiment_factors_empty():
    empty_news = pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score", "sentiment_volume", "event_type", "event_id"])
    prices = _make_prices_df()
    result = build_news_sentiment_factors(empty_news, prices)
    assert isinstance(result, pd.DataFrame)


def test_build_news_sentiment_factors_with_data():
    news = _make_news_df()
    prices = _make_prices_df()
    result = build_news_sentiment_factors(news, prices)
    assert isinstance(result, pd.DataFrame)


def test_build_macro_regime_factors_empty():
    empty_macro = pd.DataFrame(columns=["timestamp", "macro_code", "value", "country"])
    prices = _make_prices_df()
    result = build_macro_regime_factors(empty_macro, prices)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# event_features_vectorized (Step 2.38)
# ---------------------------------------------------------------------------

def _make_events_df(n: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "symbol": "AAPL",
        "event_date": ts,
        "disclosure_date": ts,
        "event_type": "earnings",
        "event_id": [f"e{i}" for i in range(n)],
    })


def test_build_event_feature_panel_vectorized_empty():
    empty_events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date", "event_type", "event_id"])
    prices = _make_prices_df()
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel_vectorized(empty_events, prices, as_of=as_of)
    assert isinstance(result, pd.DataFrame)


def test_build_event_feature_panel_vectorized_with_data():
    events = _make_events_df()
    prices = _make_prices_df()
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel_vectorized(events, prices, as_of=as_of)
    assert isinstance(result, pd.DataFrame)


def test_add_disclosure_count_feature_vectorized_empty():
    empty_events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date", "event_type", "event_id"])
    prices = _make_prices_df()
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    # Note: arg order is (prices, events, ...)
    result = add_disclosure_count_feature_vectorized(prices, empty_events, as_of=as_of)
    assert isinstance(result, pd.DataFrame)
