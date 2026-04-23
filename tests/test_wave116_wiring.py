"""Tests for wave-116 module wiring into trading_cycle.py.

Covers:
  Step 2.97 — data.sources.fred_source (fetch_fred_series)
  Step 2.98 — data.sources.newsapi_source (fetch_news_headlines)
  Step 2.99 — data.sources.polygon_source (fetch_prices_polygon)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.sources.fred_source import fetch_fred_series
from src.assembled_core.data.sources.newsapi_source import fetch_news_headlines
from src.assembled_core.data.sources.polygon_source import fetch_prices_polygon


# ---------------------------------------------------------------------------
# data.sources.fred_source (Step 2.97)
# ---------------------------------------------------------------------------

def test_fetch_fred_series_importable():
    assert fetch_fred_series is not None


def test_fetch_fred_series_empty_list():
    result = fetch_fred_series([], "2024-01-01", "2024-06-01")
    assert isinstance(result, pd.DataFrame)


def test_fetch_fred_series_returns_dataframe_no_key():
    # Without API key returns empty DataFrame gracefully
    result = fetch_fred_series(["DGS10"], "2024-01-01", "2024-01-31")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.sources.newsapi_source (Step 2.98)
# ---------------------------------------------------------------------------

def test_fetch_news_headlines_importable():
    assert fetch_news_headlines is not None


def test_fetch_news_headlines_empty_keywords():
    result = fetch_news_headlines([], "2024-01-01", "2024-01-31")
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# data.sources.polygon_source (Step 2.99)
# ---------------------------------------------------------------------------

def test_fetch_prices_polygon_importable():
    assert fetch_prices_polygon is not None


def test_fetch_prices_polygon_empty_symbols():
    result = fetch_prices_polygon([], "2024-01-01", "2024-06-01")
    assert isinstance(result, pd.DataFrame)
