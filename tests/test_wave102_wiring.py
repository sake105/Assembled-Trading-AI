"""Tests for wave-102 module wiring into trading_cycle.py.

Covers:
  Step ops.6   — ops.replay_snapshot (RunSnapshot / ReplayResult)
  Step paper.1 — paper.georisk_gate (compute_georisk_multiplier / apply_georisk_to_orders)
  Step paper.2 — paper.intel_context (active_shocks_from_triggers / populate_ctx_from_artifacts)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.ops.replay_snapshot import RunSnapshot, ReplayResult, derive_seed
from src.assembled_core.paper.georisk_gate import compute_georisk_multiplier, apply_georisk_to_orders
from src.assembled_core.paper.intel_context import (
    active_shocks_from_triggers,
    populate_ctx_from_artifacts,
)


# ---------------------------------------------------------------------------
# replay_snapshot (Step ops.6)
# ---------------------------------------------------------------------------

def test_run_snapshot_importable():
    assert RunSnapshot is not None


def test_replay_result_importable():
    assert ReplayResult is not None


def test_derive_seed_returns_int():
    seed = derive_seed("run_001", "2024-06-01", None)
    assert isinstance(seed, int)


def test_derive_seed_deterministic():
    s1 = derive_seed("run_001", "2024-06-01", 42)
    s2 = derive_seed("run_001", "2024-06-01", 42)
    assert s1 == s2


# ---------------------------------------------------------------------------
# georisk_gate (Step paper.1)
# ---------------------------------------------------------------------------

def test_compute_georisk_multiplier_none_returns_one():
    result = compute_georisk_multiplier(None)
    assert result == 1.0


def test_compute_georisk_multiplier_active_state():
    result = compute_georisk_multiplier({"state_hint": "ACTIVE"})
    assert 0.0 <= result < 1.0


def test_compute_georisk_multiplier_watch_state():
    result = compute_georisk_multiplier({"state_hint": "WATCH"})
    assert result == 1.0


def test_apply_georisk_to_orders_empty():
    result = apply_georisk_to_orders(pd.DataFrame(), multiplier=0.7)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# intel_context (Step paper.2)
# ---------------------------------------------------------------------------

def test_active_shocks_from_triggers_importable():
    assert active_shocks_from_triggers is not None


def test_populate_ctx_from_artifacts_importable():
    assert populate_ctx_from_artifacts is not None
