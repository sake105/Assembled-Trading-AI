"""Tests for wave-92 module wiring into trading_cycle.py.

Covers:
  Step 5.62 — execution.position_sync (SyncResult)
  Step 5.63 — execution.pre_live_gate (PreLiveGate / PreLiveGateResult)
  Step 5.64 — execution.smart_order_router (route_order / DEFAULT_VENUES)
"""

from __future__ import annotations

import pytest

from src.assembled_core.execution.position_sync import SyncResult
from src.assembled_core.execution.pre_live_gate import (
    PreLiveGate,
    PreLiveGateResult,
    GateCheckResult,
)
from src.assembled_core.execution.smart_order_router import (
    route_order,
    DEFAULT_VENUES,
    VenueConfig,
    RoutingResult,
)


# ---------------------------------------------------------------------------
# position_sync (Step 5.62)
# ---------------------------------------------------------------------------

def test_sync_result_importable():
    assert SyncResult is not None


def test_sync_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(SyncResult)


# ---------------------------------------------------------------------------
# pre_live_gate (Step 5.63)
# ---------------------------------------------------------------------------

def test_pre_live_gate_creates():
    plg = PreLiveGate()
    assert isinstance(plg, PreLiveGate)


def test_pre_live_gate_evaluate_returns_result():
    plg = PreLiveGate()
    result = plg.evaluate()
    assert isinstance(result, PreLiveGateResult)


def test_pre_live_gate_evaluate_all_failed_by_default():
    plg = PreLiveGate()
    result = plg.evaluate()
    assert isinstance(result.all_passed, bool)
    assert result.all_passed is False  # no checks pass with defaults


def test_pre_live_gate_result_has_checks():
    plg = PreLiveGate()
    result = plg.evaluate()
    assert len(result.checks) > 0


def test_gate_check_result_importable():
    assert GateCheckResult is not None


# ---------------------------------------------------------------------------
# smart_order_router (Step 5.64)
# ---------------------------------------------------------------------------

def test_default_venues_not_empty():
    assert len(DEFAULT_VENUES) > 0


def test_default_venues_are_venue_configs():
    for v in DEFAULT_VENUES:
        assert isinstance(v, VenueConfig)


def test_route_order_importable():
    assert route_order is not None


def test_routing_result_importable():
    assert RoutingResult is not None
