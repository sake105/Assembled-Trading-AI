"""Tests for persistent SQLite ledger store."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

import pandas as pd

from src.assembled_core.data.ledger_store import LedgerStore


@pytest.fixture
def tmp_ledger():
    """Create a temporary ledger store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_ledger.db"
        store = LedgerStore(db_path=str(db_path))
        yield store


@pytest.mark.fast
class TestLedgerStore:
    def test_init_creates_db(self, tmp_ledger):
        assert tmp_ledger is not None

    def test_cash_management(self, tmp_ledger):
        tmp_ledger.set_cash(100000.0)
        assert tmp_ledger.get_cash() == 100000.0

    def test_apply_fill_buy(self, tmp_ledger):
        tmp_ledger.set_cash(100000.0)
        fill = {
            "symbol": "AAPL",
            "side": "buy",
            "quantity": 100,
            "price": 150.0,
            "timestamp": "2024-01-15T10:00:00",
        }
        tmp_ledger.apply_fill(fill)
        pos = tmp_ledger.get_position("AAPL")
        assert pos is not None
        assert pos["quantity"] == 100

    def test_apply_fill_sell(self, tmp_ledger):
        tmp_ledger.set_cash(100000.0)
        # Buy first
        tmp_ledger.apply_fill(
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.0,
                "timestamp": "2024-01-15",
            }
        )
        # Then sell
        tmp_ledger.apply_fill(
            {
                "symbol": "AAPL",
                "side": "sell",
                "quantity": 50,
                "price": 155.0,
                "timestamp": "2024-01-16",
            }
        )
        pos = tmp_ledger.get_position("AAPL")
        assert pos["quantity"] == 50

    def test_get_positions(self, tmp_ledger):
        tmp_ledger.set_cash(100000.0)
        tmp_ledger.apply_fill(
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.0,
                "timestamp": "2024-01-15",
            }
        )
        tmp_ledger.apply_fill(
            {
                "symbol": "MSFT",
                "side": "buy",
                "quantity": 50,
                "price": 400.0,
                "timestamp": "2024-01-15",
            }
        )
        positions = tmp_ledger.get_positions()
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) == 2

    def test_query_fills(self, tmp_ledger):
        tmp_ledger.set_cash(100000.0)
        tmp_ledger.apply_fill(
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.0,
                "timestamp": "2024-01-15",
            }
        )
        fills = tmp_ledger.query_fills()
        assert isinstance(fills, pd.DataFrame)
        assert len(fills) >= 1

    def test_mark_to_market(self, tmp_ledger):
        tmp_ledger.set_cash(100000.0)
        tmp_ledger.apply_fill(
            {
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 100,
                "price": 150.0,
                "timestamp": "2024-01-15",
            }
        )
        prices = {"AAPL": 155.0}
        result = tmp_ledger.mark_to_market(prices, as_of="2024-01-16")
        assert isinstance(result, (dict, float, int))

    def test_empty_positions(self, tmp_ledger):
        positions = tmp_ledger.get_positions()
        assert isinstance(positions, pd.DataFrame)
        assert len(positions) == 0

    def test_equity_curve(self, tmp_ledger):
        curve = tmp_ledger.load_equity_curve()
        assert isinstance(curve, pd.DataFrame)
