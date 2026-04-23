"""Tests for wave-124 module wiring into trading_cycle.py.

Covers:
  Step 8.66 — events.disclosures.triggers (score_disclosure_triggers)
  Step 8.67 — events.news.baseline (compute_version_hash)
  Step 8.68 — events.news.clustering (build_clusters)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.disclosures.triggers import score_disclosure_triggers
from src.assembled_core.events.news.baseline import compute_version_hash
from src.assembled_core.events.news.clustering import build_clusters


# ---------------------------------------------------------------------------
# events.disclosures.triggers (Step 8.66)
# ---------------------------------------------------------------------------

def test_score_disclosure_triggers_importable():
    assert score_disclosure_triggers is not None


def test_score_disclosure_triggers_empty():
    result = score_disclosure_triggers(
        events=[],
        source_meta={},
        cfg={},
        now_utc="2024-06-01T12:00:00Z",
    )
    assert isinstance(result, list)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# events.news.baseline (Step 8.67)
# ---------------------------------------------------------------------------

def test_compute_version_hash_returns_string():
    result = compute_version_hash({})
    assert isinstance(result, str)
    assert len(result) == 64


def test_compute_version_hash_deterministic():
    r1 = compute_version_hash({})
    r2 = compute_version_hash({})
    assert r1 == r2


def test_compute_version_hash_changes_with_config():
    r1 = compute_version_hash({})
    r2 = compute_version_hash({"burst": {"baseline_days": 60}})
    assert r1 != r2


# ---------------------------------------------------------------------------
# events.news.clustering (Step 8.68)
# ---------------------------------------------------------------------------

def test_build_clusters_importable():
    assert build_clusters is not None


def test_build_clusters_empty():
    result = build_clusters([], cfg={})
    assert isinstance(result, list)
    assert len(result) == 0
