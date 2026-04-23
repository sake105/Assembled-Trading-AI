"""Tests for wave-113 module wiring into trading_cycle.py.

Covers:
  Step 2.88 — data.news.entity_linking (link_news_to_symbols)
  Step 2.89 — data.news.store (load_news / store_news_parquet)
  Step 2.90 — data.news_ingest (load_news_sample / normalize_news)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.news.entity_linking import link_news_to_symbols
from src.assembled_core.data.news.store import load_news, store_news_parquet
from src.assembled_core.data.news_ingest import load_news_sample, normalize_news


# ---------------------------------------------------------------------------
# data.news.entity_linking (Step 2.88)
# ---------------------------------------------------------------------------

def test_link_news_to_symbols_importable():
    assert link_news_to_symbols is not None


def test_link_news_to_symbols_empty_df():
    result = link_news_to_symbols(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


def test_link_news_to_symbols_preserves_symbol_col():
    df = pd.DataFrame({"symbol": ["AAPL"], "headline": ["Apple news"]})
    result = link_news_to_symbols(df)
    assert "symbol" in result.columns


# ---------------------------------------------------------------------------
# data.news.store (Step 2.89)
# ---------------------------------------------------------------------------

def test_load_news_importable():
    assert load_news is not None


def test_load_news_returns_dataframe_no_path():
    result = load_news()
    assert isinstance(result, pd.DataFrame)


def test_load_news_has_columns():
    result = load_news()
    assert "timestamp" in result.columns
    assert "symbol" in result.columns


def test_store_news_parquet_importable():
    assert store_news_parquet is not None


# ---------------------------------------------------------------------------
# data.news_ingest (Step 2.90)
# ---------------------------------------------------------------------------

def test_load_news_sample_returns_dataframe():
    result = load_news_sample()
    assert isinstance(result, pd.DataFrame)


def test_load_news_sample_has_expected_columns():
    result = load_news_sample()
    assert "timestamp" in result.columns
    assert "symbol" in result.columns


def test_normalize_news_importable():
    assert normalize_news is not None


def test_normalize_news_requires_timestamp():
    with pytest.raises((KeyError, ValueError)):
        normalize_news(pd.DataFrame())
