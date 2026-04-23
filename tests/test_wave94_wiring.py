"""Tests for wave-94 module wiring into trading_cycle.py.

Covers:
  Step 7.76 — qa.backtest_engine (BacktestResult / run_portfolio_backtest)
  Step 7.77 — qa.e2e_integration (E2ETestResult / E2ESuiteResult)
  Step 7.78 — qa.event_study (build_event_window_prices / compute_event_returns)
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest
from src.assembled_core.qa.e2e_integration import E2ETestResult, E2ESuiteResult, run_e2e_suite
from src.assembled_core.qa.event_study import build_event_window_prices, compute_event_returns


# ---------------------------------------------------------------------------
# backtest_engine (Step 7.76)
# ---------------------------------------------------------------------------

def test_backtest_result_importable():
    assert BacktestResult is not None


def test_run_portfolio_backtest_importable():
    assert run_portfolio_backtest is not None


def test_backtest_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(BacktestResult)


# ---------------------------------------------------------------------------
# e2e_integration (Step 7.77)
# ---------------------------------------------------------------------------

def test_e2e_test_result_importable():
    assert E2ETestResult is not None


def test_e2e_suite_result_importable():
    assert E2ESuiteResult is not None


def test_run_e2e_suite_importable():
    assert run_e2e_suite is not None


def test_e2e_test_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(E2ETestResult)


# ---------------------------------------------------------------------------
# event_study (Step 7.78)
# ---------------------------------------------------------------------------

def test_build_event_window_prices_importable():
    assert build_event_window_prices is not None


def test_compute_event_returns_importable():
    assert compute_event_returns is not None


def test_build_event_window_prices_empty():
    prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    events = pd.DataFrame(columns=["timestamp", "symbol", "event_type"])
    result = build_event_window_prices(prices, events)
    assert isinstance(result, pd.DataFrame)
