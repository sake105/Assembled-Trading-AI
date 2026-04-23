"""Tests for wave-132 module wiring into trading_cycle.py.

Covers:
  Step ops.1 — ops.alerts (compute_alerts / write_alerts_artifact)
  Step ops.2 — ops.intel_sim (apply_intel_sim)
  Step ops.3 — ops.paper_runner (run_paper_daily_one)
"""

from __future__ import annotations

import pytest

from src.assembled_core.ops.alerts import compute_alerts, write_alerts_artifact
from src.assembled_core.ops.intel_sim import apply_intel_sim
from src.assembled_core.ops.paper_runner import run_paper_daily_one


# ---------------------------------------------------------------------------
# ops.alerts (Step ops.1)
# ---------------------------------------------------------------------------

def test_compute_alerts_importable():
    assert compute_alerts is not None


def test_compute_alerts_empty():
    result = compute_alerts(run_kpis={}, reasons={}, diff={}, cfg={})
    assert isinstance(result, list)


def test_compute_alerts_disabled():
    result = compute_alerts(
        run_kpis={}, reasons={}, diff={}, cfg={"alerts": {"enabled": False}}
    )
    assert result == []


def test_write_alerts_artifact_importable():
    assert write_alerts_artifact is not None


# ---------------------------------------------------------------------------
# ops.intel_sim (Step ops.2)
# ---------------------------------------------------------------------------

def test_apply_intel_sim_importable():
    assert apply_intel_sim is not None


# ---------------------------------------------------------------------------
# ops.paper_runner (Step ops.3)
# ---------------------------------------------------------------------------

def test_run_paper_daily_one_importable():
    assert run_paper_daily_one is not None
