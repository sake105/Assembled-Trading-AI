"""Tests for wave-104 module wiring into trading_cycle.py.

Covers:
  Step rpt.1  — reports.daily_qa_report (generate_qa_report / generate_qa_report_from_files)
  Step 3.90   — strategies.ema_trend_v0 (compute_signals / compute_target_positions)
  Step 3.91   — strategies.ic_decay_weights (compute_ic_decay_weights / ICDecayWeightResult)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.reports.daily_qa_report import generate_qa_report, generate_qa_report_from_files
from src.assembled_core.strategies.ema_trend_v0 import compute_signals, compute_target_positions
from src.assembled_core.strategies.ic_decay_weights import (
    compute_ic_decay_weights,
    ICDecayWeightResult,
)


# ---------------------------------------------------------------------------
# daily_qa_report (Step rpt.1)
# ---------------------------------------------------------------------------

def test_generate_qa_report_importable():
    assert generate_qa_report is not None


def test_generate_qa_report_from_files_importable():
    assert generate_qa_report_from_files is not None


# ---------------------------------------------------------------------------
# ema_trend_v0 (Step 3.90)
# ---------------------------------------------------------------------------

def test_compute_signals_importable():
    assert compute_signals is not None


def test_compute_target_positions_importable():
    assert compute_target_positions is not None


def test_compute_signals_empty_df():
    result = compute_signals(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# ic_decay_weights (Step 3.91)
# ---------------------------------------------------------------------------

def test_compute_ic_decay_weights_empty():
    result = compute_ic_decay_weights({})
    assert isinstance(result, ICDecayWeightResult)


def test_compute_ic_decay_weights_fallback_on_empty():
    result = compute_ic_decay_weights({})
    assert result.fallback_used is True


def test_compute_ic_decay_weights_with_positive_ic():
    result = compute_ic_decay_weights({"momentum": 0.15, "value": 0.08})
    assert isinstance(result.weights, dict)
    assert len(result.weights) > 0


def test_ic_decay_weight_result_importable():
    assert ICDecayWeightResult is not None
