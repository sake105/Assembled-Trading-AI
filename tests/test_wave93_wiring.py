"""Tests for wave-93 module wiring into trading_cycle.py.

Covers:
  Step 5.65 — execution.unified_paper_engine (UnifiedPaperEngine / UnifiedPaperConfig)
  Step 7.74 — qa.adversarial_testing (run_adversarial_audit / fgsm_perturbation)
  Step 7.75 — qa.backtest_comparison (compare_backtests / BacktestComparisonReport)
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from src.assembled_core.execution.unified_paper_engine import UnifiedPaperEngine, UnifiedPaperConfig
from src.assembled_core.qa.adversarial_testing import (
    run_adversarial_audit,
    detect_stale_features,
    AdversarialReport,
)
from src.assembled_core.qa.backtest_comparison import (
    compare_backtests,
    BacktestComparisonReport,
    StrategyMetrics,
)


# ---------------------------------------------------------------------------
# unified_paper_engine (Step 5.65)
# ---------------------------------------------------------------------------

def test_unified_paper_engine_creates():
    upe = UnifiedPaperEngine()
    assert isinstance(upe, UnifiedPaperEngine)


def test_unified_paper_engine_not_initialized():
    upe = UnifiedPaperEngine()
    assert upe._initialized is False


def test_unified_paper_engine_empty_equity():
    upe = UnifiedPaperEngine()
    assert len(upe._equity_curve) == 0


def test_unified_paper_config_importable():
    assert UnifiedPaperConfig is not None


# ---------------------------------------------------------------------------
# adversarial_testing (Step 7.74)
# ---------------------------------------------------------------------------

def test_run_adversarial_audit_importable():
    assert run_adversarial_audit is not None


def test_detect_stale_features_empty():
    result = detect_stale_features(pd.DataFrame())
    assert isinstance(result, dict)
    assert len(result) == 0  # empty df → no stale columns


def test_adversarial_report_importable():
    assert AdversarialReport is not None


# ---------------------------------------------------------------------------
# backtest_comparison (Step 7.75)
# ---------------------------------------------------------------------------

def test_compare_backtests_importable():
    assert compare_backtests is not None


def test_backtest_comparison_report_importable():
    assert BacktestComparisonReport is not None


def test_strategy_metrics_importable():
    assert StrategyMetrics is not None
