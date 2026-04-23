"""Tests for wave-122 module wiring into trading_cycle.py.

Covers:
  Step 8.60 — events.disclosures.fetch_house_ptr (fetch_house_ptr_filings)
  Step 8.61 — events.disclosures.health (compute_health)
  Step 8.62 — events.disclosures.models (DisclosureEvent / DisclosuresHealth)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.disclosures.fetch_house_ptr import fetch_house_ptr_filings
from src.assembled_core.events.disclosures.health import compute_health
from src.assembled_core.events.disclosures.models import DisclosureEvent, DisclosuresHealth


# ---------------------------------------------------------------------------
# events.disclosures.fetch_house_ptr (Step 8.60)
# ---------------------------------------------------------------------------

def test_fetch_house_ptr_filings_importable():
    assert fetch_house_ptr_filings is not None


# ---------------------------------------------------------------------------
# events.disclosures.health (Step 8.61)
# ---------------------------------------------------------------------------

def test_compute_health_importable():
    assert compute_health is not None


def test_compute_health_single_ok_source():
    result = compute_health(["src1"], items_raw=5, items_after_dedupe=4, failures=[])
    assert isinstance(result, DisclosuresHealth)
    assert result.status == "OK"


def test_compute_health_no_sources():
    result = compute_health([], items_raw=0, items_after_dedupe=0, failures=[])
    assert result.status == "ERROR"


def test_compute_health_with_failure():
    result = compute_health(["src1"], items_raw=0, items_after_dedupe=0, failures=[{"source": "src1"}])
    assert result.status in ("ERROR", "DEGRADED")


# ---------------------------------------------------------------------------
# events.disclosures.models (Step 8.62)
# ---------------------------------------------------------------------------

def test_disclosure_event_importable():
    assert DisclosureEvent is not None


def test_disclosures_health_importable():
    assert DisclosuresHealth is not None


def test_disclosure_event_creates():
    ev = DisclosureEvent(
        event_id="test_001",
        source_id="edgar",
        source_name="SEC EDGAR",
        source_domain="sec.gov",
        published_utc="2024-06-01T12:00:00Z",
        fetched_utc="2024-06-01T12:05:00Z",
    )
    assert ev.event_id == "test_001"
    assert ev.raw == {}
