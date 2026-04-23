"""Tests for wave-129 module wiring into trading_cycle.py.

Covers:
  Step 8.81 — events.news.state (load_fetch_state)
  Step 5.62 — execution.transaction_costs (CommissionModel)
  Step 8.82 — intel.bayesian_confidence (bayesian_update)
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from src.assembled_core.events.news.state import load_fetch_state
from src.assembled_core.execution.transaction_costs import CommissionModel
from src.assembled_core.intel.bayesian_confidence import bayesian_update


# ---------------------------------------------------------------------------
# events.news.state (Step 8.81)
# ---------------------------------------------------------------------------

def test_load_fetch_state_importable():
    assert load_fetch_state is not None


def test_load_fetch_state_missing_file():
    result = load_fetch_state(Path("/nonexistent/state.json"))
    assert isinstance(result, dict)
    assert "rss" in result
    assert "gdelt" in result


# ---------------------------------------------------------------------------
# execution.transaction_costs (Step 5.62)
# ---------------------------------------------------------------------------

def test_commission_model_creates():
    cm = CommissionModel()
    assert isinstance(cm, CommissionModel)


def test_commission_model_defaults():
    cm = CommissionModel()
    assert cm.mode == "bps"
    assert cm.commission_bps == 1.0


def test_commission_model_custom():
    cm = CommissionModel(mode="fixed", fixed_per_trade=5.0)
    assert cm.mode == "fixed"
    assert cm.fixed_per_trade == 5.0


def test_commission_model_invalid_mode():
    with pytest.raises(ValueError):
        CommissionModel(mode="invalid_mode")


# ---------------------------------------------------------------------------
# intel.bayesian_confidence (Step 8.82)
# ---------------------------------------------------------------------------

def test_bayesian_update_importable():
    assert bayesian_update is not None


def test_bayesian_update_returns_float():
    result = bayesian_update(0.5, 0.7, 0.9)
    assert isinstance(result, float)


def test_bayesian_update_range():
    result = bayesian_update(0.5, 0.7, 0.9)
    assert 0.0 <= result <= 1.0


def test_bayesian_update_higher_than_prior():
    result = bayesian_update(0.3, 0.8, 0.9)
    assert result > 0.3
