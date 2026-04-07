"""Tests for broker_execution.py — submit, poll, convert, orchestrate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.assembled_core.execution.broker_adapter import BrokerOrder
from src.assembled_core.execution.broker_execution import (
    BrokerExecutionResult,
    convert_broker_fills_to_ledger_format,
    execute_via_broker,
    poll_order_fills,
    submit_orders_to_broker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_order(
    order_id: str = "ord-1",
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float = 10.0,
    status: str = "new",
    filled_qty: float = 0.0,
    filled_avg_price: float | None = None,
) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        symbol=symbol,
        side=side,
        qty=qty,
        order_type="market",
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )


def _orders_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class FakeAdapter:
    """Minimal fake broker adapter for testing."""

    def __init__(self):
        self._orders: dict[str, BrokerOrder] = {}
        self._submit_count = 0

    def submit_market_order(self, symbol, qty, side) -> BrokerOrder:
        self._submit_count += 1
        oid = f"fake-{self._submit_count}"
        order = _make_order(order_id=oid, symbol=symbol, side=side, qty=qty, status="new")
        self._orders[oid] = order
        return order

    def get_order_status(self, order_id: str) -> BrokerOrder:
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Unknown order: {order_id}")
        # Simulate immediate fill
        return BrokerOrder(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            order_type=order.order_type,
            status="filled",
            filled_qty=order.qty,
            filled_avg_price=150.0,
        )


# ---------------------------------------------------------------------------
# Tests: submit_orders_to_broker
# ---------------------------------------------------------------------------


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=False)
def test_submit_dry_run(mock_ks):
    adapter = FakeAdapter()
    df = _orders_df([
        {"symbol": "AAPL", "side": "buy", "qty": 10},
        {"symbol": "MSFT", "side": "sell", "qty": 5},
    ])
    results, intent_keys = submit_orders_to_broker(adapter, df, dry_run=True)
    assert len(results) == 2
    assert all(o is not None for o in results)
    assert results[0].status == "dry_run"
    assert adapter._submit_count == 0  # No real calls


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=True)
def test_submit_kill_switch_blocks(mock_ks):
    adapter = FakeAdapter()
    df = _orders_df([{"symbol": "AAPL", "side": "buy", "qty": 10}])
    results, intent_keys = submit_orders_to_broker(adapter, df)
    assert all(o is None for o in results)


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=False)
def test_submit_skips_invalid_orders(mock_ks):
    adapter = FakeAdapter()
    df = _orders_df([
        {"symbol": "", "side": "buy", "qty": 10},      # empty symbol
        {"symbol": "AAPL", "side": "hold", "qty": 10},  # invalid side
        {"symbol": "AAPL", "side": "buy", "qty": 0},    # zero qty
        {"symbol": "MSFT", "side": "buy", "qty": 5},    # valid
    ])
    results, intent_keys = submit_orders_to_broker(adapter, df)
    assert len(results) == 4
    assert results[0] is None  # empty symbol
    assert results[1] is None  # invalid side
    assert results[2] is None  # zero qty
    assert results[3] is not None  # valid


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=False)
def test_submit_empty_df(mock_ks):
    adapter = FakeAdapter()
    df = pd.DataFrame(columns=["symbol", "side", "qty"])
    results, intent_keys = submit_orders_to_broker(adapter, df)
    assert results == []


# ---------------------------------------------------------------------------
# Tests: poll_order_fills
# ---------------------------------------------------------------------------


def test_poll_immediate_fill():
    adapter = FakeAdapter()
    order = adapter.submit_market_order("AAPL", 10, "buy")
    final = poll_order_fills(adapter, [order], timeout_s=5, poll_interval_s=0.1)
    assert len(final) == 1
    assert final[0].status == "filled"


def test_poll_skips_none():
    adapter = FakeAdapter()
    final = poll_order_fills(adapter, [None, None], timeout_s=1, poll_interval_s=0.1)
    assert final == []


def test_poll_dry_run_skipped():
    adapter = FakeAdapter()
    dry = _make_order(status="dry_run")
    final = poll_order_fills(adapter, [dry], timeout_s=1, poll_interval_s=0.1)
    assert len(final) == 1
    assert final[0].status == "dry_run"


# ---------------------------------------------------------------------------
# Tests: convert_broker_fills_to_ledger_format
# ---------------------------------------------------------------------------


def test_convert_filled_order():
    order = _make_order(
        status="filled",
        side="buy",
        qty=10.0,
        filled_qty=10.0,
        filled_avg_price=150.50,
    )
    fills = convert_broker_fills_to_ledger_format([order])
    assert len(fills) == 1
    assert fills[0]["side"] == "BUY"  # CRITICAL: uppercase
    assert fills[0]["qty"] == 10.0
    assert fills[0]["price"] == 150.50


def test_convert_partial_fill():
    order = _make_order(
        status="filled",
        side="sell",
        qty=100.0,
        filled_qty=75.0,  # partial
        filled_avg_price=200.0,
    )
    fills = convert_broker_fills_to_ledger_format([order])
    assert len(fills) == 1
    assert fills[0]["qty"] == 75.0  # Uses filled_qty, not requested qty


def test_convert_skips_non_filled():
    orders = [
        _make_order(status="cancelled"),
        _make_order(status="rejected"),
        _make_order(status="dry_run"),
    ]
    fills = convert_broker_fills_to_ledger_format(orders)
    assert fills == []


def test_convert_skips_zero_filled_qty():
    order = _make_order(status="filled", filled_qty=0.0, filled_avg_price=100.0)
    fills = convert_broker_fills_to_ledger_format([order])
    assert fills == []


def test_convert_skips_no_price():
    order = _make_order(status="filled", filled_qty=10.0, filled_avg_price=None)
    fills = convert_broker_fills_to_ledger_format([order])
    assert fills == []


# ---------------------------------------------------------------------------
# Tests: execute_via_broker (orchestrator)
# ---------------------------------------------------------------------------


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=False)
def test_execute_via_broker_dry_run(mock_ks):
    adapter = FakeAdapter()
    df = _orders_df([{"symbol": "AAPL", "side": "buy", "qty": 10}])
    result = execute_via_broker(adapter, df, dry_run=True)
    assert isinstance(result, BrokerExecutionResult)
    assert result.dry_run is True
    assert result.fills_for_ledger == []
    assert len(result.submitted) == 1


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=False)
def test_execute_via_broker_live(mock_ks):
    adapter = FakeAdapter()
    df = _orders_df([
        {"symbol": "AAPL", "side": "buy", "qty": 10},
        {"symbol": "MSFT", "side": "sell", "qty": 5},
    ])
    result = execute_via_broker(
        adapter, df, timeout_s=5, poll_interval_s=0.1
    )
    assert result.dry_run is False
    assert len(result.filled) == 2
    assert len(result.fills_for_ledger) == 2
    assert result.execution_time_s > 0


@patch("src.assembled_core.execution.kill_switch.is_kill_switch_engaged", return_value=False)
def test_execute_via_broker_empty(mock_ks):
    adapter = FakeAdapter()
    df = pd.DataFrame(columns=["symbol", "side", "qty"])
    result = execute_via_broker(adapter, df)
    assert result.fills_for_ledger == []
    assert result.submitted == []
