"""Tests for wave-128 module wiring into trading_cycle.py.

Covers:
  Step 8.78 — events.news.normalize (canonicalize_url / normalize_raw_item)
  Step 8.79 — events.news.pipeline (run_news_pipeline)
  Step 8.80 — events.news.sources (NewsSource / load_sources_registry)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.news.normalize import canonicalize_url, normalize_raw_item
from src.assembled_core.events.news.pipeline import run_news_pipeline
from src.assembled_core.events.news.sources import NewsSource, load_sources_registry


# ---------------------------------------------------------------------------
# events.news.normalize (Step 8.78)
# ---------------------------------------------------------------------------

def test_canonicalize_url_importable():
    assert canonicalize_url is not None


def test_canonicalize_url_strips_tracking():
    url = "https://example.com/news?utm_source=twitter&id=123"
    result = canonicalize_url(url)
    assert "utm_source" not in result
    assert "id=123" in result


def test_canonicalize_url_empty():
    result = canonicalize_url("")
    assert result == ""


def test_normalize_raw_item_importable():
    assert normalize_raw_item is not None


# ---------------------------------------------------------------------------
# events.news.pipeline (Step 8.79)
# ---------------------------------------------------------------------------

def test_run_news_pipeline_importable():
    assert run_news_pipeline is not None


# ---------------------------------------------------------------------------
# events.news.sources (Step 8.80)
# ---------------------------------------------------------------------------

def test_news_source_importable():
    assert NewsSource is not None


def test_load_sources_registry_importable():
    assert load_sources_registry is not None


def test_load_sources_registry_missing_file():
    result = load_sources_registry("/nonexistent/sources.yaml")
    assert isinstance(result, list)
    assert len(result) == 0
