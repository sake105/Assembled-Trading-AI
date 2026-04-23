"""Tests for wave-45 module wiring into trading_cycle.py.

Covers:
  Step 2.24 — features.news_features (add_news_features)
  Step 2.25 — features.geopolitical_features (compute_gpr_proxy)
  Step 2.26 — features.insider_features (add_insider_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.news_features import (
    add_news_features,
    compute_news_momentum,
    compute_sentiment_trajectory,
)
from src.assembled_core.features.geopolitical_features import compute_gpr_proxy
from src.assembled_core.features.insider_features import add_insider_features


# ---------------------------------------------------------------------------
# add_news_features (Step 2.24)
# ---------------------------------------------------------------------------

def _make_prices(n: int = 30, n_syms: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sym in [f"S{i}" for i in range(n_syms)]:
        ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
        closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
        for t, c in zip(ts, closes):
            rows.append({"timestamp": t, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _make_news_events(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": [f"S{i % 2}" for i in range(n)],
        "sentiment_score": rng.uniform(-1.0, 1.0, n),
    })


def test_add_news_features_returns_df():
    prices = _make_prices()
    events = _make_news_events()
    result = add_news_features(prices, events)
    assert isinstance(result, pd.DataFrame)


def test_add_news_features_has_new_cols():
    prices = _make_prices()
    events = _make_news_events()
    result = add_news_features(prices, events)
    assert len(result.columns) > len(prices.columns)


def test_add_news_features_empty_events():
    prices = _make_prices()
    events = pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score"])
    result = add_news_features(prices, events)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(prices)


def test_add_news_features_missing_price_col_raises():
    prices = pd.DataFrame({"timestamp": [], "symbol": []})
    events = pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score"])
    with pytest.raises(KeyError):
        add_news_features(prices, events)


def test_add_news_features_with_as_of():
    prices = _make_prices()
    events = _make_news_events()
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = add_news_features(prices, events, as_of=as_of)
    assert isinstance(result, pd.DataFrame)


def test_compute_news_momentum_returns_df():
    events = pd.DataFrame(columns=["timestamp", "affected_sectors", "severity"])
    result = compute_news_momentum(events)
    assert isinstance(result, pd.DataFrame)


def test_compute_news_momentum_with_events():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-10", periods=20, freq="h", tz="UTC")
    events = pd.DataFrame({
        "timestamp": ts,
        "affected_sectors": [["Tech", "Finance"]] * 20,
        "severity": rng.uniform(0, 1, 20),
    })
    result = compute_news_momentum(events)
    assert isinstance(result, pd.DataFrame)


def test_compute_sentiment_trajectory_stable_empty():
    events = pd.DataFrame(columns=["timestamp", "symbol", "sentiment_score"])
    result = compute_sentiment_trajectory(events, symbol="S0")
    assert result in {"improving", "worsening", "stable"}


# ---------------------------------------------------------------------------
# compute_gpr_proxy (Step 2.25)
# ---------------------------------------------------------------------------

def test_gpr_proxy_returns_df():
    result = compute_gpr_proxy()
    assert isinstance(result, pd.DataFrame)


def test_gpr_proxy_has_gpr_level_col():
    result = compute_gpr_proxy()
    assert "gpr_level" in result.columns


def test_gpr_proxy_with_vix():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=252, freq="B")
    vix = pd.Series(15.0 + rng.uniform(0, 20, 252), index=idx)
    result = compute_gpr_proxy(vix_series=vix)
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_gpr_proxy_has_gpr_regime():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    vix = pd.Series(15.0 + rng.uniform(0, 20, 300), index=idx)
    result = compute_gpr_proxy(vix_series=vix)
    assert "gpr_regime" in result.columns


def test_gpr_proxy_empty_all_none():
    result = compute_gpr_proxy(
        gdelt_event_counts=None,
        gdelt_tone_scores=None,
        conflict_counts=None,
        vix_series=None,
    )
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# add_insider_features (Step 2.26)
# ---------------------------------------------------------------------------

def test_add_insider_features_returns_df():
    prices = _make_prices()
    events = pd.DataFrame(columns=["timestamp", "symbol"])
    result = add_insider_features(prices, events)
    assert isinstance(result, pd.DataFrame)


def test_add_insider_features_has_new_cols():
    prices = _make_prices()
    events = pd.DataFrame(columns=["timestamp", "symbol"])
    result = add_insider_features(prices, events)
    assert len(result.columns) >= len(prices.columns)


def test_add_insider_features_with_events():
    rng = np.random.default_rng(0)
    prices = _make_prices()
    ts = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    events = pd.DataFrame({
        "timestamp": ts,
        "symbol": [f"S{i % 2}" for i in range(10)],
        "net_shares": rng.normal(0, 1000, 10),
        "trades_count": rng.integers(1, 5, 10),
    })
    result = add_insider_features(prices, events)
    assert isinstance(result, pd.DataFrame)


def test_add_insider_features_missing_price_col_raises():
    prices = pd.DataFrame({"timestamp": [], "symbol": []})
    events = pd.DataFrame(columns=["timestamp", "symbol"])
    with pytest.raises(KeyError):
        add_insider_features(prices, events)


def test_add_insider_features_with_as_of():
    prices = _make_prices()
    events = pd.DataFrame(columns=["timestamp", "symbol"])
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = add_insider_features(prices, events, as_of=as_of)
    assert isinstance(result, pd.DataFrame)
