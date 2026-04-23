"""Tests for wave-46 module wiring into trading_cycle.py.

Covers:
  Step 2.27 — features.event_features (build_event_feature_panel)
  Step 2.28 — features.disclosure_features (compute_fog_index / compute_filing_length_change)
  Step 2.29 — features.buyback_features (build_buyback_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.event_features import build_event_feature_panel
from src.assembled_core.features.disclosure_features import (
    compute_fog_index,
    compute_filing_length_change,
)
from src.assembled_core.features.buyback_features import (
    build_buyback_features,
    compute_buyback_yield,
    detect_buyback_from_shares,
)


# ---------------------------------------------------------------------------
# build_event_feature_panel (Step 2.27)
# ---------------------------------------------------------------------------

def _make_prices(n: int = 20) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": "AAPL",
        "close": 150.0 + np.cumsum(rng.normal(0, 0.5, n)),
    })


def test_event_feature_panel_returns_df():
    prices = _make_prices()
    events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel(events, prices, as_of=as_of)
    assert isinstance(result, pd.DataFrame)


def test_event_feature_panel_same_row_count():
    prices = _make_prices(15)
    events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel(events, prices, as_of=as_of)
    assert len(result) == len(prices)


def test_event_feature_panel_adds_count_col():
    prices = _make_prices()
    events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel(events, prices, as_of=as_of)
    assert any("count" in c for c in result.columns)


def test_event_feature_panel_requires_as_of():
    prices = _make_prices()
    events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])
    with pytest.raises(ValueError):
        build_event_feature_panel(events, prices, as_of=None)


def test_event_feature_panel_with_events():
    prices = _make_prices(20)
    ts = pd.date_range("2024-01-05", periods=5, freq="B", tz="UTC")
    events = pd.DataFrame({
        "symbol": "AAPL",
        "event_date": ts,
        "disclosure_date": ts,
        "value": [0.5] * 5,
        "event_type": ["earnings"] * 5,
    })
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel(events, prices, as_of=as_of)
    assert isinstance(result, pd.DataFrame)


def test_event_feature_panel_vectorized_method():
    prices = _make_prices()
    events = pd.DataFrame(columns=["symbol", "event_date", "disclosure_date"])
    as_of = pd.Timestamp("2024-02-01", tz="UTC")
    result = build_event_feature_panel(events, prices, as_of=as_of, method="vectorized")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# disclosure_features (Step 2.28)
# ---------------------------------------------------------------------------

def test_fog_index_returns_float():
    text = "The company reported strong results. Revenue increased significantly."
    result = compute_fog_index(text)
    assert isinstance(result, float)


def test_fog_index_empty_returns_zero():
    result = compute_fog_index("")
    assert result == 0.0


def test_fog_index_short_text_zero():
    result = compute_fog_index("Short text.")
    assert result == 0.0


def test_fog_index_complex_text_higher():
    simple = "The cat sat on the mat. The dog ran fast."
    complex_text = "The corporation demonstrated extraordinary volatility. Shareholders contemplated unprecedented opportunities."
    fog_simple = compute_fog_index(simple)
    fog_complex = compute_fog_index(complex_text)
    assert fog_complex >= fog_simple


def test_filing_length_change_returns_float():
    result = compute_filing_length_change(1200, 1000)
    assert isinstance(result, float)
    assert abs(result - 0.20) < 1e-9


def test_filing_length_change_zero_prior():
    result = compute_filing_length_change(1000, 0)
    assert result == 0.0


def test_filing_length_change_decrease():
    result = compute_filing_length_change(800, 1000)
    assert result < 0.0


# ---------------------------------------------------------------------------
# build_buyback_features (Step 2.29)
# ---------------------------------------------------------------------------

def _make_price_series(n: int = 100) -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)), index=idx)


def test_build_buyback_features_returns_df():
    prices = _make_price_series()
    result = build_buyback_features(prices)
    assert isinstance(result, pd.DataFrame)


def test_build_buyback_features_same_index():
    prices = _make_price_series(60)
    result = build_buyback_features(prices)
    assert len(result) == len(prices)


def test_build_buyback_features_has_buyback_flag():
    prices = _make_price_series()
    result = build_buyback_features(prices)
    assert "buyback_flag" in result.columns


def test_compute_buyback_yield_returns_series():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    shares = pd.Series(1e6 + rng.normal(0, 1000, 50), index=idx)
    mcap = pd.Series(1e8 + rng.normal(0, 1e6, 50), index=idx)
    result = compute_buyback_yield(shares, mcap)
    assert isinstance(result, pd.Series)


def test_detect_buyback_from_shares():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    shares = pd.Series(1e6 - np.arange(30) * 1000, index=idx)
    result = detect_buyback_from_shares(shares)
    # Returns Series of bool or a scalar bool
    assert isinstance(result, (bool, np.bool_, pd.Series))
