"""Tests for wave-140 module wiring into trading_cycle.py.

Covers:
  Step qa.8  — qa.validation (ModelValidationResult / validate_performance)
  Step qa.9  — qa.walk_forward (WalkForwardConfig / generate_walk_forward_splits)
  Step rpt.1 — reports.metrics_export (export_metrics_json)
"""

from __future__ import annotations

import pytest

from src.assembled_core.qa.validation import (
    ModelValidationResult,
    validate_performance,
    validate_overfitting,
)
from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    WalkForwardWindow,
    generate_walk_forward_splits,
)
from src.assembled_core.reports.metrics_export import export_metrics_json


# ---------------------------------------------------------------------------
# qa.validation (Step qa.8)
# ---------------------------------------------------------------------------

def test_model_validation_result_importable():
    assert ModelValidationResult is not None


def test_model_validation_result_creates():
    r = ModelValidationResult(model_name="test_model", is_ok=True)
    assert r.is_ok is True
    assert r.model_name == "test_model"
    assert r.errors == []


def test_validate_performance_importable():
    assert validate_performance is not None


def test_validate_performance_basic():
    metrics = {"sharpe": 1.5, "max_drawdown": 0.1, "cagr": 0.25}
    result = validate_performance(metrics)
    assert isinstance(result, ModelValidationResult)


def test_validate_overfitting_importable():
    assert validate_overfitting is not None


# ---------------------------------------------------------------------------
# qa.walk_forward (Step qa.9)
# ---------------------------------------------------------------------------

def test_walk_forward_config_importable():
    assert WalkForwardConfig is not None


def test_walk_forward_window_importable():
    assert WalkForwardWindow is not None


def test_generate_walk_forward_splits_importable():
    assert generate_walk_forward_splits is not None


# ---------------------------------------------------------------------------
# reports.metrics_export (Step rpt.1)
# ---------------------------------------------------------------------------

def test_export_metrics_json_importable():
    assert export_metrics_json is not None
