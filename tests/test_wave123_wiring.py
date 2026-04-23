"""Tests for wave-123 module wiring into trading_cycle.py.

Covers:
  Step 8.63 — events.disclosures.normalize (normalize_raw_item)
  Step 8.64 — events.disclosures.pipeline (run_disclosures_pipeline)
  Step 8.65 — events.disclosures.sources (DisclosureSource / load_sources_registry)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.disclosures.normalize import normalize_raw_item
from src.assembled_core.events.disclosures.pipeline import run_disclosures_pipeline
from src.assembled_core.events.disclosures.sources import DisclosureSource, load_sources_registry


# ---------------------------------------------------------------------------
# events.disclosures.normalize (Step 8.63)
# ---------------------------------------------------------------------------

def test_normalize_raw_item_importable():
    assert normalize_raw_item is not None


def test_normalize_raw_item_none_input():
    result = normalize_raw_item(None, "src", "Source", "domain.com", "2024-01-01T00:00:00Z")
    assert result is None


def test_normalize_raw_item_empty_dict():
    result = normalize_raw_item({}, "src", "Source", "domain.com", "2024-01-01T00:00:00Z")
    assert result is None


# ---------------------------------------------------------------------------
# events.disclosures.pipeline (Step 8.64)
# ---------------------------------------------------------------------------

def test_run_disclosures_pipeline_importable():
    assert run_disclosures_pipeline is not None


# ---------------------------------------------------------------------------
# events.disclosures.sources (Step 8.65)
# ---------------------------------------------------------------------------

def test_disclosure_source_importable():
    assert DisclosureSource is not None


def test_load_sources_registry_importable():
    assert load_sources_registry is not None


def test_load_sources_registry_missing_file():
    result = load_sources_registry("/nonexistent/sources.yaml")
    assert isinstance(result, list)
    assert len(result) == 0
