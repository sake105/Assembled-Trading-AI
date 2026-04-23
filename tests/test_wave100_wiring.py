"""Tests for wave-100 module wiring into trading_cycle.py.

Covers:
  Step 2.75  — data.universe_etf (load_etf_universe / get_all_symbols)
  Step ops.1 — ops.daily_scheduler (WorkerResult)
  Step ops.2 — ops.grafana_dashboards (export_all_dashboards)
"""

from __future__ import annotations

import pytest

from src.assembled_core.data.universe_etf import (
    load_etf_universe,
    get_all_symbols,
    get_inverse_etf_map,
)
from src.assembled_core.ops.daily_scheduler import WorkerResult
from src.assembled_core.ops.grafana_dashboards import (
    export_all_dashboards,
    portfolio_performance_dashboard,
)


# ---------------------------------------------------------------------------
# universe_etf (Step 2.75)
# ---------------------------------------------------------------------------

def test_load_etf_universe_returns_dict():
    result = load_etf_universe()
    assert isinstance(result, dict)


def test_get_all_symbols_returns_list():
    universe = load_etf_universe()
    symbols = get_all_symbols(universe)
    assert isinstance(symbols, list)


def test_get_all_symbols_not_empty():
    universe = load_etf_universe()
    symbols = get_all_symbols(universe)
    assert len(symbols) > 0


def test_get_inverse_etf_map_returns_dict():
    result = get_inverse_etf_map()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# daily_scheduler (Step ops.1)
# ---------------------------------------------------------------------------

def test_worker_result_importable():
    assert WorkerResult is not None


def test_worker_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(WorkerResult)


# ---------------------------------------------------------------------------
# grafana_dashboards (Step ops.2)
# ---------------------------------------------------------------------------

def test_export_all_dashboards_returns_dict():
    result = export_all_dashboards()
    assert isinstance(result, dict)


def test_export_all_dashboards_not_empty():
    result = export_all_dashboards()
    assert len(result) > 0


def test_portfolio_performance_dashboard_returns_dict():
    result = portfolio_performance_dashboard()
    assert isinstance(result, dict)
    assert len(result) > 0
