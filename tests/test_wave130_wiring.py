"""Tests for wave-130 module wiring into trading_cycle.py.

Covers:
  Step 8.83 — intel.geo_trigger (aggregate_triggers / score_event)
  Step 8.84 — intel.ic_loop (ICTracker)
  Step 8.85 — intel.models (TriggerType / CrisisMode)
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.geo_trigger import aggregate_triggers
from src.assembled_core.intel.ic_loop import ICTracker
from src.assembled_core.intel.models import TriggerType, CrisisMode


# ---------------------------------------------------------------------------
# intel.geo_trigger (Step 8.83)
# ---------------------------------------------------------------------------

def test_aggregate_triggers_importable():
    assert aggregate_triggers is not None


def test_aggregate_triggers_empty():
    result = aggregate_triggers([])
    assert isinstance(result, dict)
    assert result["geo_score"] == 0
    assert result["active_triggers"] == []


# ---------------------------------------------------------------------------
# intel.ic_loop (Step 8.84)
# ---------------------------------------------------------------------------

def test_ic_tracker_creates():
    tracker = ICTracker()
    assert isinstance(tracker, ICTracker)


def test_ic_tracker_no_state_path():
    tracker = ICTracker()
    # Without state path, ic returns None for unknown trigger
    result = tracker.ic("UNKNOWN_TRIGGER")
    assert result is None


def test_ic_tracker_record():
    tracker = ICTracker()
    tracker.record("WAR_ESCALATION", signal=0.8, realized_return=0.03)
    # Still None with single observation (< 2 needed for Pearson)
    result = tracker.ic("WAR_ESCALATION")
    assert result is None


# ---------------------------------------------------------------------------
# intel.models (Step 8.85)
# ---------------------------------------------------------------------------

def test_trigger_type_importable():
    assert TriggerType is not None


def test_crisis_mode_importable():
    assert CrisisMode is not None


def test_trigger_type_has_war_escalation():
    assert "WAR_ESCALATION" in TriggerType.__members__


def test_crisis_mode_has_active():
    assert "ACTIVE" in CrisisMode.__members__


def test_trigger_type_count():
    assert len(TriggerType) > 10
