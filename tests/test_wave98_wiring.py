"""Tests for wave-98 module wiring into trading_cycle.py.

Covers:
  Step 7.88 — accounting.reconciliation_report (write_reconcile_report_json)
  Step 2.70 — data.corporate_actions (load_corporate_actions / apply_splits_for_research_prices)
  Step 2.71 — data.cost_model_policy (estimate_rebalance_cost_fraction / load_cost_tiers)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.accounting.reconciliation_report import (
    write_reconcile_report_json,
    write_reconcile_report_csv,
)
from src.assembled_core.data.corporate_actions import (
    load_corporate_actions,
    apply_splits_for_research_prices,
)
from src.assembled_core.data.cost_model_policy import (
    estimate_rebalance_cost_fraction,
    load_cost_tiers,
    get_effective_cost_params,
)


# ---------------------------------------------------------------------------
# reconciliation_report (Step 7.88)
# ---------------------------------------------------------------------------

def test_write_reconcile_report_json_importable():
    assert write_reconcile_report_json is not None


def test_write_reconcile_report_csv_importable():
    assert write_reconcile_report_csv is not None


# ---------------------------------------------------------------------------
# corporate_actions (Step 2.70)
# ---------------------------------------------------------------------------

def test_load_corporate_actions_returns_dataframe():
    df = load_corporate_actions()
    assert isinstance(df, pd.DataFrame)


def test_load_corporate_actions_empty_by_default():
    df = load_corporate_actions()
    assert len(df) == 0


def test_apply_splits_for_research_prices_importable():
    assert apply_splits_for_research_prices is not None


def test_apply_splits_for_research_prices_empty():
    result = apply_splits_for_research_prices(pd.DataFrame(), pd.DataFrame())
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# cost_model_policy (Step 2.71)
# ---------------------------------------------------------------------------

def test_estimate_rebalance_cost_fraction_importable():
    assert estimate_rebalance_cost_fraction is not None


def test_load_cost_tiers_returns_dict():
    result = load_cost_tiers()
    assert isinstance(result, dict)


def test_get_effective_cost_params_importable():
    assert get_effective_cost_params is not None
