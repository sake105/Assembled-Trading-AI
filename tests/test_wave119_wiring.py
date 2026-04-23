"""Tests for wave-119 module wiring into trading_cycle.py.

Covers:
  Step 8.51 — events.crisis_alpha.entry (generate_crisis_entry)
  Step 8.52 — events.crisis_alpha.exit_rules (get_positions_to_exit)
  Step 8.53 — events.crisis_alpha.gates (check_health_gate / run_all_activation_gates)
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from src.assembled_core.events.crisis_alpha.entry import generate_crisis_entry
from src.assembled_core.events.crisis_alpha.exit_rules import get_positions_to_exit
from src.assembled_core.events.crisis_alpha.gates import (
    check_health_gate,
    run_all_activation_gates,
)


# ---------------------------------------------------------------------------
# events.crisis_alpha.entry (Step 8.51)
# ---------------------------------------------------------------------------

def test_generate_crisis_entry_importable():
    assert generate_crisis_entry is not None


# ---------------------------------------------------------------------------
# events.crisis_alpha.exit_rules (Step 8.52)
# ---------------------------------------------------------------------------

def test_get_positions_to_exit_importable():
    assert get_positions_to_exit is not None


def test_get_positions_to_exit_empty():
    now = datetime.now(tz=timezone.utc)
    result = get_positions_to_exit([], now_utc=now)
    assert isinstance(result, list)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# events.crisis_alpha.gates (Step 8.53)
# ---------------------------------------------------------------------------

def test_run_all_activation_gates_importable():
    assert run_all_activation_gates is not None


def test_check_health_gate_importable():
    assert check_health_gate is not None
