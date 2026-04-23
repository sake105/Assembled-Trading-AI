"""Tests for wave-127 module wiring into trading_cycle.py.

Covers:
  Step 8.75 — events.news.fetch_rss (fetch_rss_feed)
  Step 8.76 — events.news.health (compute_health)
  Step 8.77 — events.news.models (NewsEvent / NewsHealth)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.news.fetch_rss import fetch_rss_feed
from src.assembled_core.events.news.health import compute_health
from src.assembled_core.events.news.models import NewsEvent, NewsHealth


# ---------------------------------------------------------------------------
# events.news.fetch_rss (Step 8.75)
# ---------------------------------------------------------------------------

def test_fetch_rss_feed_importable():
    assert fetch_rss_feed is not None


# ---------------------------------------------------------------------------
# events.news.health (Step 8.76)
# ---------------------------------------------------------------------------

def test_compute_health_news_importable():
    assert compute_health is not None


def test_compute_health_news_ok():
    result = compute_health(["src1"], items_raw=5, items_after_dedupe=4, failures=[])
    assert isinstance(result, NewsHealth)
    assert result.status == "OK"


def test_compute_health_news_error_no_sources():
    result = compute_health([], items_raw=0, items_after_dedupe=0, failures=[])
    assert result.status == "ERROR"


# ---------------------------------------------------------------------------
# events.news.models (Step 8.77)
# ---------------------------------------------------------------------------

def test_news_event_importable():
    assert NewsEvent is not None


def test_news_health_importable():
    assert NewsHealth is not None


def test_news_event_creates():
    ev = NewsEvent(
        event_id="ev_001",
        source_id="reuters",
        title="Test headline",
        url="https://example.com/news/1",
        canonical_url="https://example.com/news/1",
        source_name="Reuters",
        source_domain="reuters.com",
        published_utc="2024-06-01T10:00:00Z",
        fetched_utc="2024-06-01T10:05:00Z",
    )
    assert ev.event_id == "ev_001"
    assert ev.raw == {}
