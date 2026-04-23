"""Tests for wave-76 module wiring into trading_cycle.py.

Covers:
  Step 8.81 — intel.news_contradiction (ContradictionDetector)
  Step 8.82 — intel.news_dedupe (NewsDedupeIndex)
  Step 8.83 — intel.news_enricher (NewsEventEnricher)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_contradiction import (
    ContradictionDetector,
    ContradictionEntry,
)
from src.assembled_core.intel.news_dedupe import (
    NewsDedupeIndex,
    canonical_url,
    content_fingerprint,
)
from src.assembled_core.intel.news_enricher import NewsEventEnricher


# ---------------------------------------------------------------------------
# news_contradiction (Step 8.81)
# ---------------------------------------------------------------------------

def test_contradiction_detector_creates():
    cd = ContradictionDetector()
    assert isinstance(cd, ContradictionDetector)


def test_contradiction_detector_empty_events():
    cd = ContradictionDetector()
    result = cd.analyse([])
    assert isinstance(result, dict)
    assert len(result) == 0


def test_contradiction_entry_importable():
    assert ContradictionEntry is not None


# ---------------------------------------------------------------------------
# news_dedupe (Step 8.82)
# ---------------------------------------------------------------------------

def test_news_dedupe_index_creates():
    ndi = NewsDedupeIndex()
    assert isinstance(ndi, NewsDedupeIndex)


def test_news_dedupe_index_empty():
    ndi = NewsDedupeIndex()
    assert len(ndi.seen_event_ids) == 0
    assert len(ndi.seen_fingerprints) == 0


def test_canonical_url_returns_str():
    url = canonical_url("https://www.example.com/news?utm_source=feed&id=123")
    assert isinstance(url, str)


def test_content_fingerprint_deterministic():
    fp1 = content_fingerprint("Federal Reserve raises rates", "reuters.com")
    fp2 = content_fingerprint("Federal Reserve raises rates", "reuters.com")
    assert fp1 == fp2


def test_content_fingerprint_different_titles():
    fp1 = content_fingerprint("Market rally", "reuters.com")
    fp2 = content_fingerprint("Market crash", "reuters.com")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# news_enricher (Step 8.83)
# ---------------------------------------------------------------------------

def test_news_enricher_creates():
    enricher = NewsEventEnricher()
    assert isinstance(enricher, NewsEventEnricher)


def test_news_enricher_enrich_empty():
    enricher = NewsEventEnricher()
    result = enricher.enrich([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_news_enricher_with_dedupe():
    dedupe = NewsDedupeIndex()
    enricher = NewsEventEnricher(dedupe_index=dedupe)
    result = enricher.enrich([])
    assert isinstance(result, list)
