"""Tests for wave-91 module wiring into trading_cycle.py.

Covers:
  Step 5.59 — execution.ibkr_adapter (IBKRAdapter simulation mode)
  Step 5.60 — execution.paper_monitoring (PaperMonitor)
  Step 5.61 — execution.paper_trading_engine (PaperTradingEngine)
"""

from __future__ import annotations

import pytest

from src.assembled_core.execution.ibkr_adapter import IBKRAdapter, IBOrder, IBPosition
from src.assembled_core.execution.paper_monitoring import PaperMonitor, PaperMonitorAlert
from src.assembled_core.execution.paper_trading_engine import PaperTradingEngine, PaperOrder


# ---------------------------------------------------------------------------
# ibkr_adapter (Step 5.59)
# ---------------------------------------------------------------------------

def test_ibkr_adapter_creates_simulation():
    ibkr = IBKRAdapter(simulation=True)
    assert isinstance(ibkr, IBKRAdapter)


def test_ibkr_adapter_simulation_flag():
    ibkr = IBKRAdapter(simulation=True)
    assert ibkr._simulation is True


def test_ibkr_adapter_not_connected_initially():
    ibkr = IBKRAdapter(simulation=True)
    assert ibkr._connected is False


def test_ibkr_adapter_connect_simulation():
    ibkr = IBKRAdapter(simulation=True)
    result = ibkr.connect()
    assert result is True
    assert ibkr._connected is True


def test_ib_order_importable():
    assert IBOrder is not None


# ---------------------------------------------------------------------------
# paper_monitoring (Step 5.60)
# ---------------------------------------------------------------------------

def test_paper_monitor_creates():
    pm = PaperMonitor()
    assert isinstance(pm, PaperMonitor)


def test_paper_monitor_empty_results():
    pm = PaperMonitor()
    assert len(pm._results) == 0


def test_paper_monitor_thresholds_set():
    pm = PaperMonitor()
    assert isinstance(pm._thresholds, dict)
    assert len(pm._thresholds) > 0


def test_paper_monitor_alert_importable():
    assert PaperMonitorAlert is not None


# ---------------------------------------------------------------------------
# paper_trading_engine (Step 5.61)
# ---------------------------------------------------------------------------

def test_paper_trading_engine_creates():
    pte = PaperTradingEngine()
    assert isinstance(pte, PaperTradingEngine)


def test_paper_trading_engine_empty_state():
    pte = PaperTradingEngine()
    assert len(pte._orders) == 0
    assert len(pte._positions) == 0


def test_paper_trading_engine_submit_empty():
    pte = PaperTradingEngine()
    result = pte.submit_orders([])
    assert isinstance(result, list)
    assert len(result) == 0


def test_paper_order_importable():
    assert PaperOrder is not None
