"""Tests for wave-6 module wiring into trading_cycle.py.

Covers:
  Step 4.85 — portfolio.cost_aware_wrapper (apply_cost_aware_from_policy)
  Step 6.8  — execution.borrow_costs (compute_borrow_cost, BorrowRateTable)
  Step 6.9  — execution.order_lifecycle (OrderLifecycleTracker state machine)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.portfolio.cost_aware_wrapper import (
    apply_cost_aware_wrapper,
    apply_cost_aware_from_policy,
)
from src.assembled_core.execution.borrow_costs import (
    BorrowRateTable,
    compute_borrow_cost,
)
from src.assembled_core.execution.order_lifecycle import (
    OrderLifecycleTracker,
    OrderState,
)


# ---------------------------------------------------------------------------
# cost_aware_wrapper (Step 4.85)
# ---------------------------------------------------------------------------

def test_cost_aware_wrapper_shrinks_large_turnover():
    target = {"A": 0.5, "B": 0.3}
    current = {"A": 0.0, "B": 0.0}
    adj, reasons = apply_cost_aware_wrapper(
        target, current, penalty_factor=0.5, cost_bps_per_symbol={"A": 100.0, "B": 50.0}
    )
    # With penalty, weights should be <= original (some shrinkage expected)
    assert adj["A"] <= target["A"] + 1e-9
    assert adj["B"] <= target["B"] + 1e-9


def test_cost_aware_wrapper_zero_turnover_passthrough():
    target = {"A": 0.5, "B": 0.3}
    current = {"A": 0.5, "B": 0.3}  # no change
    adj, reasons = apply_cost_aware_wrapper(target, current, penalty_factor=0.5)
    assert adj == target
    assert reasons == []


def test_cost_aware_wrapper_empty_target():
    adj, reasons = apply_cost_aware_wrapper({}, {})
    assert adj == {}
    assert reasons == []


def test_cost_aware_from_policy_disabled():
    target = {"A": 0.5}
    policy = {"cost_aware_wrapper": {"enabled": False}}
    adj, reasons = apply_cost_aware_from_policy(target, {}, policy)
    assert adj == target
    assert reasons == []


def test_cost_aware_from_policy_no_section():
    target = {"A": 0.5}
    adj, reasons = apply_cost_aware_from_policy(target, {}, {})
    assert adj == target  # disabled by default


# ---------------------------------------------------------------------------
# borrow_costs (Step 6.8)
# ---------------------------------------------------------------------------

def test_borrow_cost_long_position_zero():
    cost = compute_borrow_cost(100, 50.0, 50.0)
    assert cost == 0.0


def test_borrow_cost_short_position_positive():
    cost = compute_borrow_cost(-100, 50.0, 50.0)
    assert cost > 0.0


def test_borrow_cost_formula():
    # -100 shares @ $50, 50bps annual, 1 day held, 365 days/year
    # = 5000 notional * (50/10000 / 365) * 1 = 5000 * 0.0001370... ≈ 0.0685
    cost = compute_borrow_cost(-100, 50.0, 50.0, days_held=1, days_in_year=365)
    expected = 5000.0 * (50.0 / 10_000.0) / 365.0
    assert abs(cost - expected) < 1e-9


def test_borrow_cost_htb_rate():
    cost_easy = compute_borrow_cost(-100, 100.0, 50.0)
    cost_htb = compute_borrow_cost(-100, 100.0, 500.0)
    assert cost_htb > cost_easy * 9  # ~10x more expensive


def test_borrow_rate_table_default():
    brt = BorrowRateTable()
    assert brt.rate_bps("UNKNOWN") == 50.0


def test_borrow_rate_table_htb_symbol():
    brt = BorrowRateTable(htb_symbols={"GME"})
    assert brt.rate_bps("GME") == brt.htb_rate_bps


def test_borrow_rate_table_override():
    brt = BorrowRateTable(overrides={"AAPL": 200.0})
    assert brt.rate_bps("AAPL") == 200.0


# ---------------------------------------------------------------------------
# order_lifecycle (Step 6.9)
# ---------------------------------------------------------------------------

def test_order_lifecycle_valid_path():
    olt = OrderLifecycleTracker()
    oid = olt.create("AAPL", "buy", 100.0, 150.0, "test")
    olt.transition(oid, OrderState.VALIDATED)
    olt.transition(oid, OrderState.SUBMITTED)
    assert olt._orders[oid].current_state == OrderState.SUBMITTED


def test_order_lifecycle_invalid_transition_raises():
    olt = OrderLifecycleTracker()
    oid = olt.create("AAPL", "buy", 100.0)
    with pytest.raises(ValueError):
        olt.transition(oid, OrderState.SUBMITTED)  # must go CREATED→VALIDATED first


def test_order_lifecycle_rejected_path():
    olt = OrderLifecycleTracker()
    oid = olt.create("MSFT", "sell", 50.0)
    olt.transition(oid, OrderState.REJECTED, reason="fat_finger")
    assert olt._orders[oid].current_state == OrderState.REJECTED


def test_order_lifecycle_multiple_orders():
    olt = OrderLifecycleTracker()
    ids = []
    for sym in ["A", "B", "C"]:
        oid = olt.create(sym, "buy", 10.0, 100.0)
        olt.transition(oid, OrderState.VALIDATED)
        olt.transition(oid, OrderState.SUBMITTED)
        ids.append(oid)
    assert all(olt._orders[oid].current_state == OrderState.SUBMITTED for oid in ids)


def test_order_lifecycle_unknown_id_raises():
    olt = OrderLifecycleTracker()
    with pytest.raises(KeyError):
        olt.transition("nonexistent", OrderState.SUBMITTED)
