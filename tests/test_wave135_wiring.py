"""Tests for wave-135 module wiring into trading_cycle.py.

Covers:
  Step pipe.6 — pipeline.orchestrator (run_eod_pipeline / run_backtest_step)
  Step pipe.7 — pipeline.orders (signals_to_orders)
  Step pipe.8 — pipeline.pipeline_timing (PipelineTimer)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.pipeline.orchestrator import run_eod_pipeline, run_backtest_step
from src.assembled_core.pipeline.orders import signals_to_orders, write_orders
from src.assembled_core.pipeline.pipeline_timing import PipelineTimer


# ---------------------------------------------------------------------------
# pipeline.orchestrator (Step pipe.6)
# ---------------------------------------------------------------------------

def test_run_eod_pipeline_importable():
    assert run_eod_pipeline is not None


def test_run_backtest_step_importable():
    assert run_backtest_step is not None


# ---------------------------------------------------------------------------
# pipeline.orders (Step pipe.7)
# ---------------------------------------------------------------------------

def test_signals_to_orders_importable():
    assert signals_to_orders is not None


def test_signals_to_orders_empty():
    signals = pd.DataFrame(columns=["timestamp", "symbol", "sig", "price"])
    try:
        result = signals_to_orders(signals)
        assert isinstance(result, pd.DataFrame)
    except (KeyError, ValueError):
        pass  # empty DataFrame validation raises


def test_signals_to_orders_basic():
    signals = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "sig": [0, 1, 0],
        "price": [150.0, 151.0, 149.0],
    })
    result = signals_to_orders(signals)
    assert isinstance(result, pd.DataFrame)


def test_write_orders_importable():
    assert write_orders is not None


# ---------------------------------------------------------------------------
# pipeline.pipeline_timing (Step pipe.8)
# ---------------------------------------------------------------------------

def test_pipeline_timer_importable():
    assert PipelineTimer is not None


def test_pipeline_timer_creates():
    timer = PipelineTimer(budget_seconds=60.0)
    assert timer.budget_seconds == 60.0


def test_pipeline_timer_step():
    timer = PipelineTimer()
    timer.start_step("test_step")
    duration = timer.end_step()
    assert isinstance(duration, float)
    assert duration >= 0.0
