"""Tests for wave-70 module wiring into trading_cycle.py.

Covers:
  Step 7.79 — accounting.round_trips (compute_round_trips / round_trip_summary)
  Step 7.80 — accounting.tax_lots (TaxLotTracker)
  Step 7.81 — accounting.decision_audit (DecisionAuditTrail / DecisionRecord)
"""

from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from src.assembled_core.accounting.round_trips import (
    RoundTrip,
    compute_round_trips,
    round_trip_summary,
)
from src.assembled_core.accounting.tax_lots import TaxLot, TaxLotTracker
from src.assembled_core.accounting.decision_audit import DecisionRecord, DecisionAuditTrail


# ---------------------------------------------------------------------------
# round_trips (Step 7.79)
# ---------------------------------------------------------------------------

def test_compute_round_trips_empty():
    trades = pd.DataFrame(columns=["symbol", "date", "side", "price", "quantity", "commission"])
    result = compute_round_trips(trades)
    assert isinstance(result, list)
    assert len(result) == 0


def test_compute_round_trips_buy_sell():
    trades = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "date": [date(2024, 1, 2), date(2024, 1, 10)],
        "side": ["BUY", "SELL"],
        "price": [150.0, 160.0],
        "quantity": [100.0, 100.0],
        "commission": [1.0, 1.0],
    })
    trips = compute_round_trips(trades)
    assert isinstance(trips, list)


def test_round_trip_summary_empty():
    summary = round_trip_summary([])
    assert isinstance(summary, dict)


def test_round_trip_net_pnl():
    rt = RoundTrip(
        symbol="AAPL",
        entry_date=date(2024, 1, 2),
        exit_date=date(2024, 1, 10),
        entry_price=150.0,
        exit_price=160.0,
        quantity=100.0,
        gross_pnl=1000.0,
        commission=5.0,
    )
    assert rt.net_pnl == 995.0


def test_round_trip_holding_days():
    rt = RoundTrip(
        symbol="MSFT",
        entry_date=date(2024, 1, 1),
        exit_date=date(2024, 1, 11),
        entry_price=300.0,
        exit_price=310.0,
        quantity=50.0,
        gross_pnl=500.0,
    )
    assert rt.holding_days == 10


# ---------------------------------------------------------------------------
# tax_lots (Step 7.80)
# ---------------------------------------------------------------------------

def test_tax_lot_tracker_creates():
    tracker = TaxLotTracker()
    assert isinstance(tracker, TaxLotTracker)
    assert len(tracker.lots) == 0


def test_tax_lot_tracker_buy():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", quantity=100.0, price=150.0, trade_date=date(2024, 1, 2))
    assert "AAPL" in tracker.lots
    assert len(tracker.lots["AAPL"]) == 1


def test_tax_lot_tracker_buy_sell():
    tracker = TaxLotTracker()
    tracker.buy("AAPL", 100.0, 150.0, date(2024, 1, 2))
    pnl = tracker.sell("AAPL", 100.0, 160.0, date(2024, 1, 10))
    assert isinstance(pnl, float)
    assert abs(pnl - 1000.0) < 0.01


def test_tax_lot_tracker_sell_no_lots():
    tracker = TaxLotTracker()
    with pytest.raises(ValueError):
        tracker.sell("AAPL", 50.0, 160.0, date(2024, 1, 10))


# ---------------------------------------------------------------------------
# decision_audit (Step 7.81)
# ---------------------------------------------------------------------------

def test_decision_record_creates():
    rec = DecisionRecord(
        timestamp="2024-06-01T10:00:00",
        symbol="AAPL",
        direction="long",
        signal_score=0.7,
    )
    assert rec.symbol == "AAPL"


def test_decision_audit_trail_creates():
    trail = DecisionAuditTrail()
    assert isinstance(trail, DecisionAuditTrail)
    assert len(trail.records) == 0


def test_decision_audit_trail_record():
    trail = DecisionAuditTrail()
    rec = DecisionRecord(timestamp="2024-06-01", symbol="MSFT", direction="short", signal_score=-0.5)
    trail.record(rec)
    assert len(trail.records) == 1


def test_decision_audit_trail_summary():
    trail = DecisionAuditTrail()
    trail.record(DecisionRecord(timestamp="2024-06-01", symbol="AAPL", direction="long", signal_score=0.6))
    summary = trail.summary()
    assert summary["n_records"] == 1
    assert "AAPL" in summary["symbols"]
