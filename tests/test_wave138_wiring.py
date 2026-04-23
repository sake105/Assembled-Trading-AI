"""Tests for wave-138 module wiring into trading_cycle.py.

Covers:
  Step qa.2 — qa.drift_detection (compute_psi / detect_feature_drift)
  Step qa.3 — qa.experiment_tracking (ExperimentTracker)
  Step qa.4 — qa.health (QaCheckResult / aggregate_qa_status)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.drift_detection import compute_psi, detect_feature_drift
from src.assembled_core.qa.experiment_tracking import ExperimentRun, ExperimentTracker
from src.assembled_core.qa.health import QaCheckResult, aggregate_qa_status


# ---------------------------------------------------------------------------
# qa.drift_detection (Step qa.2)
# ---------------------------------------------------------------------------

def test_compute_psi_importable():
    assert compute_psi is not None


def test_compute_psi_identical():
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 100))
    psi = compute_psi(s, s)
    assert isinstance(psi, float)
    assert psi >= 0.0


def test_detect_feature_drift_importable():
    assert detect_feature_drift is not None


# ---------------------------------------------------------------------------
# qa.experiment_tracking (Step qa.3)
# ---------------------------------------------------------------------------

def test_experiment_tracker_importable():
    assert ExperimentTracker is not None


def test_experiment_run_importable():
    assert ExperimentRun is not None


# ---------------------------------------------------------------------------
# qa.health (Step qa.4)
# ---------------------------------------------------------------------------

def test_qa_check_result_importable():
    assert QaCheckResult is not None


def test_qa_check_result_creates():
    r = QaCheckResult(name="prices", status="ok", message="All good")
    assert r.status == "ok"
    assert r.name == "prices"


def test_qa_check_result_to_dict():
    r = QaCheckResult(name="orders", status="warning", message="Partial", details={"n": 5})
    d = r.to_dict()
    assert d["status"] == "warning"


def test_aggregate_qa_status_importable():
    assert aggregate_qa_status is not None
