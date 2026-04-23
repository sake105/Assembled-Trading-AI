"""Tests for wave-65 module wiring into trading_cycle.py.

Covers:
  Step 8.66 — config.constants (TRADING_DAYS_PER_YEAR / DEFAULT_COMMISSION_BPS)
  Step 8.67 — config.policy_schema (validate_policy)
  Step 8.68 — events.evidence_engine.grader (grade_evidence / EvidenceGrade)
"""

from __future__ import annotations

import pytest

from src.assembled_core.config.constants import (
    TRADING_DAYS_PER_YEAR,
    DEFAULT_COMMISSION_BPS,
    DEFAULT_START_CAPITAL,
    DEFAULT_SEED_CAPITAL,
)
from src.assembled_core.config.policy_schema import (
    validate_policy,
    validate_policy_consistency,
)
from src.assembled_core.events.evidence_engine.grader import grade_evidence
from src.assembled_core.events.evidence_engine.grades import EvidenceGrade


# ---------------------------------------------------------------------------
# config.constants (Step 8.66)
# ---------------------------------------------------------------------------

def test_trading_days_per_year():
    assert TRADING_DAYS_PER_YEAR == 252


def test_default_commission_bps_positive():
    assert DEFAULT_COMMISSION_BPS > 0


def test_default_start_capital_positive():
    assert DEFAULT_START_CAPITAL > 0


def test_default_seed_capital_positive():
    assert DEFAULT_SEED_CAPITAL > 0


# ---------------------------------------------------------------------------
# config.policy_schema (Step 8.67)
# ---------------------------------------------------------------------------

def test_validate_policy_empty_returns_tuple():
    result = validate_policy({})
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_validate_policy_empty_has_warnings():
    _, warnings = validate_policy({})
    assert isinstance(warnings, list)
    assert len(warnings) > 0


def test_validate_policy_returns_dict():
    policy, _ = validate_policy({"policy_version": "1.0"})
    assert isinstance(policy, dict)


def test_validate_policy_consistency_empty():
    result = validate_policy_consistency({})
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# events.evidence_engine.grader (Step 8.68)
# ---------------------------------------------------------------------------

def test_grade_evidence_returns_grade():
    summary = {"tierA_count": 2, "tierB_independent_count": 1, "evidence_ok": True}
    grade = grade_evidence(summary)
    assert isinstance(grade, EvidenceGrade)


def test_grade_evidence_grade_a():
    summary = {"tierA_count": 3, "tierB_independent_count": 0, "evidence_ok": True}
    grade = grade_evidence(summary, misinfo_risk_score=0.0)
    assert grade == EvidenceGrade.A


def test_grade_evidence_grade_d_no_sources():
    summary = {"tierA_count": 0, "tierB_independent_count": 0, "evidence_ok": False}
    grade = grade_evidence(summary, misinfo_risk_score=0.0)
    assert grade == EvidenceGrade.D


def test_grade_evidence_high_misinfo_blocks_a():
    summary = {"tierA_count": 3, "tierB_independent_count": 3, "evidence_ok": True}
    grade = grade_evidence(summary, misinfo_risk_score=0.95)
    assert grade != EvidenceGrade.A


def test_evidence_grade_values():
    assert EvidenceGrade.A == "A"
    assert EvidenceGrade.D == "D"
