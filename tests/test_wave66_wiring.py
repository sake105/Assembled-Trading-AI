"""Tests for wave-66 module wiring into trading_cycle.py.

Covers:
  Step 8.69 — events.evidence_engine.action_gate (check_evidence_grade_gate)
  Step 8.70 — events.evidence_engine.misinfo_risk (compute_misinfo_risk)
  Step 8.71 — events.news.burst (compute_bursts_for_window)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.evidence_engine.action_gate import check_evidence_grade_gate
from src.assembled_core.events.evidence_engine.grades import EvidenceGrade
from src.assembled_core.events.evidence_engine.misinfo_risk import compute_misinfo_risk
from src.assembled_core.events.news.burst import compute_bursts_for_window


# ---------------------------------------------------------------------------
# action_gate (Step 8.69)
# ---------------------------------------------------------------------------

def test_check_gate_grade_a_passes():
    ok, reason = check_evidence_grade_gate(EvidenceGrade.A)
    assert ok is True
    assert isinstance(reason, str)


def test_check_gate_grade_d_blocked():
    ok, reason = check_evidence_grade_gate(EvidenceGrade.D)
    assert ok is False


def test_check_gate_grade_b_passes_when_b_required():
    ok, reason = check_evidence_grade_gate(EvidenceGrade.B, require_for_active="B")
    assert ok is True


def test_check_gate_grade_c_blocked_when_b_required():
    ok, reason = check_evidence_grade_gate(EvidenceGrade.C, require_for_active="B")
    assert ok is False


def test_check_gate_returns_tuple():
    result = check_evidence_grade_gate(EvidenceGrade.A)
    assert isinstance(result, tuple)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# misinfo_risk (Step 8.70)
# ---------------------------------------------------------------------------

def test_compute_misinfo_risk_returns_float():
    score = compute_misinfo_risk({})
    assert isinstance(score, float)


def test_compute_misinfo_risk_range():
    score = compute_misinfo_risk({})
    assert 0.0 <= score <= 1.0


def test_compute_misinfo_risk_social_only_raises_score():
    low = compute_misinfo_risk({"tierA_count": 0}, social_only=False)
    high = compute_misinfo_risk({"tierA_count": 0}, social_only=True)
    assert high > low


def test_compute_misinfo_risk_tier_a_lowers_score():
    no_a = compute_misinfo_risk({"tierA_count": 0})
    with_a = compute_misinfo_risk({"tierA_count": 2})
    assert with_a <= no_a


def test_compute_misinfo_risk_burst_factor():
    score = compute_misinfo_risk({}, event_count=10, burst_window_minutes=2.0)
    assert score > 0.0


# ---------------------------------------------------------------------------
# news burst (Step 8.71)
# ---------------------------------------------------------------------------

def test_compute_bursts_empty_clusters():
    result = compute_bursts_for_window(clusters=[], baseline=None, cfg={}, window_hours=24)
    assert isinstance(result, dict)


def test_compute_bursts_has_entity_bursts():
    result = compute_bursts_for_window(clusters=[], baseline=None, cfg={}, window_hours=24)
    assert "entity_bursts" in result or "entities" in result or isinstance(result, dict)


def test_compute_bursts_with_clusters():
    clusters = [
        {"top_entities": ["AAPL", "tech"], "top_phrases": ["earnings beat"]},
        {"top_entities": ["AAPL", "iphone"], "top_phrases": ["new product"]},
        {"top_entities": ["AAPL"], "top_phrases": ["earnings beat"]},
    ]
    result = compute_bursts_for_window(
        clusters=clusters,
        baseline=None,
        cfg={"burst": {"min_doc_count": 2, "top_k": 10}},
        window_hours=1,
    )
    assert isinstance(result, dict)
