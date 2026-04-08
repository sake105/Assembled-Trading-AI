"""Tests for position_sync.py — reconciliation, rebuild, equity fetch."""

from __future__ import annotations

from unittest.mock import MagicMock


from src.assembled_core.execution.broker_adapter import BrokerPosition
from src.assembled_core.execution.position_sync import (
    get_broker_equity,
    rebuild_ledger_from_broker,
    sync_positions_from_broker,
)


def _make_adapter(positions=None, account=None):
    adapter = MagicMock()
    adapter.get_positions.return_value = positions or []
    adapter.get_account.return_value = account or {"cash": 10000.0, "equity": 10000.0}
    return adapter


def _make_position(symbol="AAPL", qty=10.0, avg=150.0):
    return BrokerPosition(
        symbol=symbol,
        qty=qty,
        avg_entry_price=avg,
        market_value=qty * avg,
        unrealized_pnl=0.0,
        unrealized_pnl_pct=0.0,
    )


# ---------------------------------------------------------------------------
# sync_positions_from_broker
# ---------------------------------------------------------------------------


def test_sync_empty_match():
    adapter = _make_adapter()
    ledger = {"cash": 10000.0, "positions": {}}
    result = sync_positions_from_broker(adapter, ledger)
    assert result.ok is True
    assert result.mismatches == []


def test_sync_positions_match():
    adapter = _make_adapter(
        positions=[_make_position("AAPL", 10.0)],
        account={"cash": 8500.0, "equity": 10000.0},
    )
    ledger = {
        "cash": 8500.0,
        "positions": {"AAPL": {"qty": 10.0, "avg_price": 150.0}},
    }
    result = sync_positions_from_broker(adapter, ledger)
    assert result.ok is True


def test_sync_detects_mismatch():
    adapter = _make_adapter(
        positions=[_make_position("AAPL", 10.0)],
        account={"cash": 8500.0, "equity": 10000.0},
    )
    ledger = {
        "cash": 8500.0,
        "positions": {"AAPL": {"qty": 5.0, "avg_price": 150.0}},  # wrong qty
    }
    result = sync_positions_from_broker(adapter, ledger)
    assert result.ok is False


def test_sync_detects_missing_in_ledger():
    adapter = _make_adapter(
        positions=[_make_position("AAPL", 10.0), _make_position("MSFT", 5.0)],
        account={"cash": 5000.0, "equity": 10000.0},
    )
    ledger = {
        "cash": 5000.0,
        "positions": {"AAPL": {"qty": 10.0, "avg_price": 150.0}},
    }
    result = sync_positions_from_broker(adapter, ledger)
    assert result.ok is False
    assert "MSFT" in result.missing_in_ledger


def test_sync_broker_error():
    adapter = MagicMock()
    adapter.get_positions.side_effect = ConnectionError("timeout")
    result = sync_positions_from_broker(adapter, {"cash": 10000.0, "positions": {}})
    assert result.ok is False
    assert "timeout" in result.message


# ---------------------------------------------------------------------------
# rebuild_ledger_from_broker
# ---------------------------------------------------------------------------


def test_rebuild_basic():
    adapter = _make_adapter(
        positions=[_make_position("AAPL", 10.0, 150.0)],
        account={"cash": 8500.0, "equity": 10000.0},
    )
    state = rebuild_ledger_from_broker(adapter)
    assert state["cash"] == 8500.0
    assert "AAPL" in state["positions"]
    assert state["positions"]["AAPL"]["qty"] == 10.0
    assert state["equity_curve"] == []  # Lost on rebuild


def test_rebuild_broker_error():
    adapter = MagicMock()
    adapter.get_positions.side_effect = ConnectionError("fail")
    state = rebuild_ledger_from_broker(adapter)
    assert state["cash"] == 10000.0  # fallback
    assert state["positions"] == {}


# ---------------------------------------------------------------------------
# get_broker_equity
# ---------------------------------------------------------------------------


def test_get_broker_equity():
    adapter = _make_adapter(account={"equity": 12345.67})
    assert get_broker_equity(adapter) == 12345.67


def test_get_broker_equity_error():
    adapter = MagicMock()
    adapter.get_account.side_effect = ConnectionError("fail")
    assert get_broker_equity(adapter) is None
