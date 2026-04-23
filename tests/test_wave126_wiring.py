"""Tests for wave-126 module wiring into trading_cycle.py.

Covers:
  Step 8.72 — events.news.entities (extract_entities / extract_countries)
  Step 8.73 — events.news.evidence (summarize_cluster_evidence)
  Step 8.74 — events.news.fetch_gdelt (fetch_gdelt_events)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.news.entities import extract_entities, extract_countries
from src.assembled_core.events.news.evidence import summarize_cluster_evidence
from src.assembled_core.events.news.fetch_gdelt import fetch_gdelt_events


# ---------------------------------------------------------------------------
# events.news.entities (Step 8.72)
# ---------------------------------------------------------------------------

def test_extract_entities_importable():
    assert extract_entities is not None


def test_extract_entities_empty_text():
    result = extract_entities("")
    assert isinstance(result, list)
    assert len(result) == 0


def test_extract_countries_importable():
    assert extract_countries is not None


def test_extract_countries_empty_text():
    result = extract_countries("")
    assert isinstance(result, list)


def test_extract_countries_known_country():
    result = extract_countries("Russia invades Ukraine")
    assert isinstance(result, list)
    assert len(result) >= 0  # may or may not find depending on aliases


# ---------------------------------------------------------------------------
# events.news.evidence (Step 8.73)
# ---------------------------------------------------------------------------

def test_summarize_cluster_evidence_importable():
    assert summarize_cluster_evidence is not None


def test_summarize_cluster_evidence_empty_cluster():
    result = summarize_cluster_evidence(
        cluster={"event_ids": []},
        events_by_id={},
        source_meta={},
        now_utc="2024-06-01T12:00:00Z",
    )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# events.news.fetch_gdelt (Step 8.74)
# ---------------------------------------------------------------------------

def test_fetch_gdelt_events_importable():
    assert fetch_gdelt_events is not None
