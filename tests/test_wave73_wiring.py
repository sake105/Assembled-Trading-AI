"""Tests for wave-73 module wiring into trading_cycle.py.

Covers:
  Step 5.44 — execution.symbol_kill_switch (is_symbol_blocked / list_blocked_symbols)
  Step 5.45 — execution.cost_model_calibrator (CostModelPriors / calibrate_cost_model)
  Step 5.46 — execution.fill_model_pipeline (apply_fill_model_pipeline)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.execution.symbol_kill_switch import (
    is_symbol_blocked,
    list_blocked_symbols,
    filter_orders_by_symbol_blocks,
)
from src.assembled_core.execution.cost_model_calibrator import (
    CostModelPriors,
    CalibrationResult,
    calibrate_cost_model,
)
from src.assembled_core.execution.fill_model_pipeline import apply_fill_model_pipeline


# ---------------------------------------------------------------------------
# symbol_kill_switch (Step 5.44)
# ---------------------------------------------------------------------------

def test_list_blocked_symbols_returns_list():
    result = list_blocked_symbols()
    assert isinstance(result, (list, set, dict))


def test_is_symbol_blocked_unknown_is_false():
    assert is_symbol_blocked("__NOTREAL__") is False


def test_filter_orders_by_symbol_blocks_empty():
    orders = pd.DataFrame(columns=["symbol", "quantity"])
    filtered, blocked = filter_orders_by_symbol_blocks(orders)
    assert isinstance(filtered, pd.DataFrame)
    assert isinstance(blocked, list)


def test_filter_orders_by_symbol_blocks_no_blocked():
    orders = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "quantity": [100, 50]})
    filtered, blocked = filter_orders_by_symbol_blocks(orders)
    # No symbols blocked → all orders pass through
    assert len(filtered) == 2
    assert len(blocked) == 0


# ---------------------------------------------------------------------------
# cost_model_calibrator (Step 5.45)
# ---------------------------------------------------------------------------

def test_cost_model_priors_creates():
    priors = CostModelPriors()
    assert isinstance(priors, CostModelPriors)


def test_cost_model_priors_positive_values():
    priors = CostModelPriors()
    assert priors.half_spread_bps > 0
    assert priors.participation_cap > 0


def test_calibrate_cost_model_no_tca_dir(tmp_path):
    empty_dir = tmp_path / "empty_tca"
    empty_dir.mkdir()
    result = calibrate_cost_model(str(empty_dir))
    assert isinstance(result, CalibrationResult)


def test_calibration_result_has_fields(tmp_path):
    empty_dir = tmp_path / "empty_tca2"
    empty_dir.mkdir()
    result = calibrate_cost_model(str(empty_dir))
    assert hasattr(result, "half_spread_bps")
    assert hasattr(result, "participation_cap")


# ---------------------------------------------------------------------------
# fill_model_pipeline (Step 5.46)
# ---------------------------------------------------------------------------

def test_fill_model_pipeline_empty_orders():
    result = apply_fill_model_pipeline(
        pd.DataFrame(),
        prices=pd.DataFrame(),
        freq="1D",
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_fill_model_pipeline_importable():
    assert callable(apply_fill_model_pipeline)
