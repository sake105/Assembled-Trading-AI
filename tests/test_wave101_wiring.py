"""Tests for wave-101 module wiring into trading_cycle.py.

Covers:
  Step ops.3 — ops.inspect_data (inspect_eod_prices)
  Step ops.4 — ops.intel_activity_summary (helper functions)
  Step ops.5 — ops.intel_orchestrator (run_intel_pipelines)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.ops.inspect_data import inspect_eod_prices
from src.assembled_core.ops.intel_activity_summary import _safe_int, _safe_float
from src.assembled_core.ops.intel_orchestrator import run_intel_pipelines


# ---------------------------------------------------------------------------
# inspect_data (Step ops.3)
# ---------------------------------------------------------------------------

def test_inspect_eod_prices_returns_dict():
    result = inspect_eod_prices(pd.DataFrame())
    assert isinstance(result, dict)


def test_inspect_eod_prices_has_keys():
    result = inspect_eod_prices(pd.DataFrame())
    assert "n_rows" in result
    assert "n_symbols" in result


def test_inspect_eod_prices_empty_df():
    result = inspect_eod_prices(pd.DataFrame())
    assert result["n_rows"] == 0


# ---------------------------------------------------------------------------
# intel_activity_summary (Step ops.4)
# ---------------------------------------------------------------------------

def test_safe_int_numeric():
    assert _safe_int(5) == 5
    assert _safe_int(5.9) == 5


def test_safe_int_none():
    assert _safe_int(None) == 0


def test_safe_float_numeric():
    result = _safe_float(3.14)
    assert result == pytest.approx(3.14)


def test_safe_float_none():
    result = _safe_float(None)
    assert result is None


# ---------------------------------------------------------------------------
# intel_orchestrator (Step ops.5)
# ---------------------------------------------------------------------------

def test_run_intel_pipelines_returns_dict():
    result = run_intel_pipelines(app_cfg={})
    assert isinstance(result, dict)


def test_run_intel_pipelines_skips_when_mode_not_real():
    result = run_intel_pipelines(app_cfg={"paper_runner": {"intel": {"mode": "sim"}}})
    assert result["news"]["status"] == "SKIPPED"
    assert result["disclosures"]["status"] == "SKIPPED"


def test_run_intel_pipelines_has_expected_keys():
    result = run_intel_pipelines(app_cfg={})
    assert "news" in result
    assert "disclosures" in result
