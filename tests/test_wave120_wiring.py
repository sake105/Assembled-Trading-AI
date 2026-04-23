"""Tests for wave-120 module wiring into trading_cycle.py.

Covers:
  Step 8.54 — events.crisis_alpha.risk_budget (apply_risk_budget)
  Step 8.55 — events.crisis_alpha.state_machine (CrisisStateRecord)
  Step 8.56 — events.disclosures.dedupe (dedupe_events)
"""

from __future__ import annotations

import pytest

from src.assembled_core.events.crisis_alpha.risk_budget import apply_risk_budget
from src.assembled_core.events.crisis_alpha.state_machine import CrisisStateRecord, compute_next_crisis_state
from src.assembled_core.events.disclosures.dedupe import dedupe_events


# ---------------------------------------------------------------------------
# events.crisis_alpha.risk_budget (Step 8.54)
# ---------------------------------------------------------------------------

def test_apply_risk_budget_importable():
    assert apply_risk_budget is not None


def test_apply_risk_budget_empty_weights():
    result, reasons = apply_risk_budget({}, baskets=[])
    assert isinstance(result, dict)
    assert isinstance(reasons, list)


# ---------------------------------------------------------------------------
# events.crisis_alpha.state_machine (Step 8.55)
# ---------------------------------------------------------------------------

def test_crisis_state_record_creates():
    record = CrisisStateRecord()
    assert isinstance(record, CrisisStateRecord)


def test_crisis_state_record_default_state():
    record = CrisisStateRecord()
    assert record.state == "WATCH"


def test_crisis_state_record_default_factory():
    record = CrisisStateRecord.default()
    assert record.state == "WATCH"
    assert len(record.entered_at_utc) > 0


def test_compute_next_crisis_state_importable():
    assert compute_next_crisis_state is not None


# ---------------------------------------------------------------------------
# events.disclosures.dedupe (Step 8.56)
# ---------------------------------------------------------------------------

def test_dedupe_events_importable():
    assert dedupe_events is not None


def test_dedupe_events_empty_list():
    result = dedupe_events([])
    assert isinstance(result, list)
    assert len(result) == 0
