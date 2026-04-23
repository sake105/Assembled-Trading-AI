"""Tests for wave-136 module wiring into trading_cycle.py.

Covers:
  Step pipe.9  — pipeline.portfolio (simulate_with_costs)
  Step pipe.10 — pipeline.run_metadata (collect_run_metadata)
  Step pipe.11 — pipeline.signals (compute_ema_signals)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.pipeline.portfolio import simulate_with_costs, write_portfolio_report
from src.assembled_core.pipeline.run_metadata import collect_run_metadata, save_run_metadata
from src.assembled_core.pipeline.signals import compute_ema_signals


# ---------------------------------------------------------------------------
# pipeline.portfolio (Step pipe.9)
# ---------------------------------------------------------------------------

def test_simulate_with_costs_importable():
    assert simulate_with_costs is not None


def test_write_portfolio_report_importable():
    assert write_portfolio_report is not None


# ---------------------------------------------------------------------------
# pipeline.run_metadata (Step pipe.10)
# ---------------------------------------------------------------------------

def test_collect_run_metadata_importable():
    assert collect_run_metadata is not None


def test_collect_run_metadata_returns_dict():
    meta = collect_run_metadata(config={})
    assert isinstance(meta, dict)
    assert "timestamp" in meta
    assert "python_version" in meta


def test_save_run_metadata_importable():
    assert save_run_metadata is not None


# ---------------------------------------------------------------------------
# pipeline.signals (Step pipe.11)
# ---------------------------------------------------------------------------

def test_compute_ema_signals_importable():
    assert compute_ema_signals is not None


def test_compute_ema_signals_empty():
    prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    result = compute_ema_signals(prices, fast=10, slow=30)
    assert isinstance(result, pd.DataFrame)


def test_compute_ema_signals_basic():
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": "AAPL",
        "close": np.linspace(100, 150, 50),
    })
    result = compute_ema_signals(prices, fast=5, slow=20)
    assert isinstance(result, pd.DataFrame)
    assert "sig" in result.columns
