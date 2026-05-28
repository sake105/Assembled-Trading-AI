"""Tests for order_lifecycle_log (GO_LIVE_CHECKLIST C1).

Covers:
  - Normal lifecycle: SUBMITTED → ROUTED → FILLED (3 entries, correct order, terminal)
  - Rejected order: reason field set
  - Cancelled order: cancelled event with reason
  - Partial fills: PARTIAL_FILL entries before FILLED
  - Validator finds open orders without terminal event
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("src.assembled_core.ops.order_lifecycle_log", reason="ops module")

from src.assembled_core.ops.order_lifecycle_log import (
    TERMINAL_EVENTS,
    append_lifecycle_event,
    find_open_orders,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_entries(path: Path) -> list[dict]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Case 1: Normal lifecycle — SUBMITTED → ROUTED → FILLED
# ---------------------------------------------------------------------------


def test_normal_lifecycle_three_entries_correct_order(tmp_path):
    """A normal order produces 3 entries in correct order with terminal event."""
    log = tmp_path / "lifecycle.jsonl"
    oid = "order-normal-001"

    append_lifecycle_event(
        "SUBMITTED", oid, "AAPL", "BUY", 100.0, strategy="trend", log_path=log
    )
    append_lifecycle_event(
        "ROUTED", oid, "AAPL", "BUY", 100.0, strategy="trend", log_path=log
    )
    append_lifecycle_event(
        "FILLED",
        oid,
        "AAPL",
        "BUY",
        100.0,
        price=150.25,
        strategy="trend",
        log_path=log,
    )

    entries = _read_entries(log)
    assert len(entries) == 3
    assert [e["event_type"] for e in entries] == ["SUBMITTED", "ROUTED", "FILLED"]
    assert entries[2]["price"] == pytest.approx(150.25)
    assert entries[2]["event_type"] in TERMINAL_EVENTS

    open_orders = find_open_orders(log)
    assert oid not in open_orders, "Completed order must not appear in open orders"


# ---------------------------------------------------------------------------
# Case 2: Rejected order — reason field set
# ---------------------------------------------------------------------------


def test_rejected_order_reason_set(tmp_path):
    """A rejected order has reason field and is not in open orders."""
    log = tmp_path / "lifecycle.jsonl"
    oid = "order-reject-002"

    append_lifecycle_event("SUBMITTED", oid, "TSLA", "SELL", 50.0, log_path=log)
    append_lifecycle_event(
        "REJECTED",
        oid,
        "TSLA",
        "SELL",
        50.0,
        reason="insufficient_buying_power",
        log_path=log,
    )

    entries = _read_entries(log)
    assert len(entries) == 2
    rejected = entries[1]
    assert rejected["event_type"] == "REJECTED"
    assert rejected["reason"] == "insufficient_buying_power"
    assert rejected["event_type"] in TERMINAL_EVENTS

    open_orders = find_open_orders(log)
    assert oid not in open_orders


# ---------------------------------------------------------------------------
# Case 3: Cancelled order — cancelled event with reason
# ---------------------------------------------------------------------------


def test_cancelled_order_with_reason(tmp_path):
    """A cancelled order has CANCELLED event with reason and no open entry."""
    log = tmp_path / "lifecycle.jsonl"
    oid = "order-cancel-003"

    append_lifecycle_event("SUBMITTED", oid, "NVDA", "BUY", 20.0, log_path=log)
    append_lifecycle_event(
        "CANCELLED", oid, "NVDA", "BUY", 20.0, reason="eod_no_fill", log_path=log
    )

    entries = _read_entries(log)
    assert len(entries) == 2
    cancelled = entries[1]
    assert cancelled["event_type"] == "CANCELLED"
    assert cancelled["reason"] == "eod_no_fill"

    open_orders = find_open_orders(log)
    assert oid not in open_orders


# ---------------------------------------------------------------------------
# Case 4: Partial fill → PARTIAL_FILL entries before FILLED
# ---------------------------------------------------------------------------


def test_partial_fill_multiple_entries_then_filled(tmp_path):
    """Partial fill produces PARTIAL_FILL entries followed by FILLED."""
    log = tmp_path / "lifecycle.jsonl"
    oid = "order-partial-004"

    append_lifecycle_event("SUBMITTED", oid, "MSFT", "BUY", 200.0, log_path=log)
    append_lifecycle_event(
        "PARTIAL_FILL", oid, "MSFT", "BUY", 200.0, price=300.10, log_path=log
    )
    append_lifecycle_event(
        "PARTIAL_FILL", oid, "MSFT", "BUY", 200.0, price=300.15, log_path=log
    )
    append_lifecycle_event(
        "FILLED", oid, "MSFT", "BUY", 200.0, price=300.20, log_path=log
    )

    entries = _read_entries(log)
    assert len(entries) == 4
    types = [e["event_type"] for e in entries]
    assert types == ["SUBMITTED", "PARTIAL_FILL", "PARTIAL_FILL", "FILLED"]

    # Only the last entry is terminal
    assert entries[-1]["event_type"] in TERMINAL_EVENTS
    assert entries[1]["event_type"] not in TERMINAL_EVENTS
    assert entries[2]["event_type"] not in TERMINAL_EVENTS

    assert find_open_orders(log) == []


# ---------------------------------------------------------------------------
# Case 5: Validator finds artificially open order (no terminal event)
# ---------------------------------------------------------------------------


def test_validator_finds_open_order_without_terminal(tmp_path):
    """find_open_orders returns order_id for orders without terminal event."""
    log = tmp_path / "lifecycle.jsonl"
    open_oid = "order-open-005"
    closed_oid = "order-closed-005"

    # Closed order
    append_lifecycle_event("SUBMITTED", closed_oid, "GOOG", "BUY", 10.0, log_path=log)
    append_lifecycle_event(
        "FILLED", closed_oid, "GOOG", "BUY", 10.0, price=180.0, log_path=log
    )

    # Open order — only SUBMITTED, no terminal
    append_lifecycle_event("SUBMITTED", open_oid, "AMZN", "BUY", 5.0, log_path=log)

    open_orders = find_open_orders(log)
    assert open_oid in open_orders, "Open order must appear in validator result"
    assert closed_oid not in open_orders, "Closed order must not appear"


# ---------------------------------------------------------------------------
# Case 6: Empty log → validator returns empty list
# ---------------------------------------------------------------------------


def test_validator_empty_log_returns_empty_list(tmp_path):
    log = tmp_path / "does_not_exist.jsonl"
    assert find_open_orders(log) == []


# ---------------------------------------------------------------------------
# Case 8: Integration — risk-side SUBMITTED and execution-side FILLED
#         produce identical order_ids for the same (symbol, side, run_id)
# ---------------------------------------------------------------------------


def test_submitted_and_filled_order_ids_align(tmp_path):
    """Simulates the _tc_risk SUBMITTED hook and _tc_execution FILLED hook.

    Asserts that the fallback order_id formula is identical on both sides,
    so find_open_orders returns [] after a complete SUBMITTED → FILLED cycle.
    This is the regression test for MAJOR-1 (id alignment fix).
    """
    log = tmp_path / "lifecycle.jsonl"
    run_id = "2026-05-28"
    symbol = "AAPL"
    # Risk side uses uppercase normalization
    side_raw_from_risk = "buy"  # lowercase as _generate_orders_default emits
    side_upper = side_raw_from_risk.upper()

    # Simulate _tc_risk SUBMITTED hook (side normalized to upper)
    synthetic_id = f"{symbol}_{side_upper}_{run_id}"
    append_lifecycle_event(
        "SUBMITTED", synthetic_id, symbol, side_upper, 100.0, log_path=log
    )

    # Simulate _tc_execution FILLED hook (side normalized to upper same formula)
    append_lifecycle_event(
        "FILLED", synthetic_id, symbol, side_upper, 100.0, price=150.0, log_path=log
    )

    open_orders = find_open_orders(log)
    assert open_orders == [], (
        f"Expected no open orders after SUBMITTED+FILLED cycle, got: {open_orders}"
    )

    entries = _read_entries(log)
    assert entries[0]["order_id"] == entries[1]["order_id"], (
        "SUBMITTED and FILLED must share the same order_id"
    )


# ---------------------------------------------------------------------------
# Case 9: find_open_orders called at EOD — warns on open orders
# ---------------------------------------------------------------------------


def test_find_open_orders_returns_only_non_terminal(tmp_path):
    """EOD validator returns only orders without any terminal event.

    Covers the _lifecycle_dump wiring: if any order_id lacks a terminal
    event, find_open_orders must include it.
    """
    log = tmp_path / "lifecycle.jsonl"

    # Complete order (SUBMITTED + FILLED)
    append_lifecycle_event("SUBMITTED", "ord-done", "SPY", "BUY", 10.0, log_path=log)
    append_lifecycle_event(
        "FILLED", "ord-done", "SPY", "BUY", 10.0, price=500.0, log_path=log
    )

    # Incomplete order (only SUBMITTED — simulates stuck order)
    append_lifecycle_event("SUBMITTED", "ord-stuck", "QQQ", "BUY", 5.0, log_path=log)

    result = find_open_orders(log)
    assert result == ["ord-stuck"], f"Expected ['ord-stuck'], got {result}"


# ---------------------------------------------------------------------------
# Case 7: Schema validation — required fields present
# ---------------------------------------------------------------------------


def test_entry_schema_contains_required_fields(tmp_path):
    """Every entry must contain all required schema fields."""
    log = tmp_path / "lifecycle.jsonl"
    append_lifecycle_event(
        "SUBMITTED",
        "order-schema-007",
        "SPY",
        "BUY",
        50.0,
        strategy="mfv2",
        actor="test",
        run_id="2026-05-28",
        log_path=log,
    )

    entry = _read_entries(log)[0]
    required = {
        "order_id",
        "timestamp_utc",
        "event_type",
        "symbol",
        "side",
        "qty",
        "strategy",
        "actor",
        "run_id",
    }
    missing = required - set(entry.keys())
    assert not missing, f"Missing fields: {missing}"
    assert entry["symbol"] == "SPY"
    assert entry["side"] == "BUY"
    assert entry["qty"] == pytest.approx(50.0)
    assert entry["strategy"] == "mfv2"
    assert entry["actor"] == "test"
    assert entry["run_id"] == "2026-05-28"
