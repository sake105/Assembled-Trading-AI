"""Tests for wave-133 module wiring into trading_cycle.py.

Covers:
  Step ops.4 — ops.reconcile (build_reconcile_report)
  Step pipe.1 — pipeline.backtest (compute_metrics)
  Step pipe.2 — pipeline.backtest_legacy (_legacy_simulate_equity)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.ops.reconcile import build_reconcile_report, write_reconcile_artifact
from src.assembled_core.pipeline.backtest import compute_metrics, write_backtest_report
from src.assembled_core.pipeline.backtest_legacy import _legacy_simulate_equity


# ---------------------------------------------------------------------------
# ops.reconcile (Step ops.4)
# ---------------------------------------------------------------------------

def test_build_reconcile_report_importable():
    assert build_reconcile_report is not None


def test_build_reconcile_report_empty():
    prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    result = build_reconcile_report(
        as_of_utc="2024-06-01T16:00:00Z",
        ledger_before={"cash": 100000.0},
        ledger_after={"cash": 99000.0},
        orders=[],
        fills=[],
        prices_latest=prices,
    )
    assert isinstance(result, dict)


def test_write_reconcile_artifact_importable():
    assert write_reconcile_artifact is not None


# ---------------------------------------------------------------------------
# pipeline.backtest (Step pipe.1)
# ---------------------------------------------------------------------------

def test_compute_metrics_importable():
    assert compute_metrics is not None


def test_compute_metrics_basic():
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        "equity": [100_000, 101_000, 100_500, 102_000, 103_000],
    })
    m = compute_metrics(equity)
    assert "final_pf" in m
    assert "sharpe" in m
    assert m["rows"] == 5


def test_write_backtest_report_importable():
    assert write_backtest_report is not None


# ---------------------------------------------------------------------------
# pipeline.backtest_legacy (Step pipe.2)
# ---------------------------------------------------------------------------

def test_legacy_simulate_equity_importable():
    assert _legacy_simulate_equity is not None


def test_legacy_simulate_equity_empty():
    prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    orders = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    result = _legacy_simulate_equity(prices, orders, start_capital=100_000.0)
    assert isinstance(result, pd.DataFrame)
