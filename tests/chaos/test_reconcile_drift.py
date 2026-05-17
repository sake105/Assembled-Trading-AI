"""Chaos test: reconciliation drift detection (Plan C21).

When broker positions drift from ledger, the reconciliation
system must detect the difference — not silently accept it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.reconciliation import (  # noqa: E402
    reconcile_ledger_vs_broker,
)

pytestmark = pytest.mark.fast


def test_reconcile_detects_qty_drift() -> None:
    """If broker has 100 shares and ledger has 95, drift must be reported."""
    broker_pos = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [100, 50]})
    ledger_pos = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [95, 50]})

    result = reconcile_ledger_vs_broker(
        ledger_pos, 10000.0, broker_pos, 10000.0, fail_fast=False
    )
    assert not result["ok"], f"Expected mismatch, got: {result}"


def test_reconcile_detects_cash_drift() -> None:
    """Cash difference must be flagged."""
    positions = pd.DataFrame({"symbol": ["AAPL"], "qty": [100]})

    result = reconcile_ledger_vs_broker(
        positions, 10000.0, positions, 9500.0, fail_fast=False
    )
    assert not result["ok"]


def test_reconcile_detects_missing_symbol_in_broker() -> None:
    """Symbol in ledger but not in broker → flagged."""
    broker_pos = pd.DataFrame({"symbol": ["AAPL"], "qty": [100]})
    ledger_pos = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [100, 50]})

    result = reconcile_ledger_vs_broker(
        ledger_pos, 10000.0, broker_pos, 10000.0, fail_fast=False
    )
    assert not result["ok"]


def test_reconcile_detects_extra_symbol_in_broker() -> None:
    """Symbol in broker but not in ledger → flagged."""
    broker_pos = pd.DataFrame(
        {"symbol": ["AAPL", "MSFT", "NVDA"], "qty": [100, 50, 30]}
    )
    ledger_pos = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [100, 50]})

    result = reconcile_ledger_vs_broker(
        ledger_pos, 10000.0, broker_pos, 10000.0, fail_fast=False
    )
    assert not result["ok"]


def test_reconcile_perfect_match() -> None:
    """Identical state should reconcile ok."""
    positions = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "qty": [100, 50]})

    result = reconcile_ledger_vs_broker(
        positions, 10000.0, positions, 10000.0, fail_fast=False
    )
    assert result["ok"]


def test_reconcile_fail_fast_raises() -> None:
    """With fail_fast=True, mismatch should raise ValueError."""
    broker_pos = pd.DataFrame({"symbol": ["AAPL"], "qty": [100]})
    ledger_pos = pd.DataFrame({"symbol": ["AAPL"], "qty": [50]})

    with pytest.raises(ValueError):
        reconcile_ledger_vs_broker(
            ledger_pos, 10000.0, broker_pos, 10000.0, fail_fast=True
        )
