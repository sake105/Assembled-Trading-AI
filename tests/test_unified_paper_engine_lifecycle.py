"""Phase 1 regression tests for UnifiedPaperEngine ↔ OrderLifecycleTracker.

Verifies the wiring between the engine and ``OrderLifecycleTracker``:

* every order generated for a day ends up as ``CREATED`` in the tracker
* risk-control-dropped orders land in ``REJECTED``
* normal fills walk CREATED → VALIDATED → SUBMITTED → FILLED
* partial fills end at ``PARTIAL_FILL`` with fill_qty/fill_price recorded
* fill-row-rejected orders end at ``REJECTED`` with reason preserved
* SUBMITTED orders with no fill row land in ``CANCELLED`` at EOD
* per-day JSONL dump is written with one line per terminal order
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.assembled_core.execution.order_lifecycle import OrderState
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(tmp_path: Path) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_lifecycle_tracking=True,
        run_id="life_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 100_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_lifecycle_attach_creates_tracker_entries(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 5.0, "price": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 3.0, "price": 50.0},
        ]
    )
    result = eng._lifecycle_attach(orders, "2025-01-15")

    assert "order_id" in result.columns
    assert result["order_id"].nunique() == 2
    assert eng._lifecycle is not None
    all_orders = eng._lifecycle.get_all_orders()
    assert len(all_orders) == 2
    assert {o.current_state for o in all_orders} == {OrderState.CREATED}


def test_lifecycle_rejects_orders_dropped_by_risk(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 5.0, "price": 100.0},
            {"symbol": "BBB", "side": "BUY", "qty": 3.0, "price": 50.0},
        ]
    )
    orders = eng._lifecycle_attach(orders, "2025-01-15")
    pre_ids = list(orders["order_id"])
    # Simulate risk control dropping the second row.
    post_ids = [pre_ids[0]]

    eng._lifecycle_mark_validation(pre_ids, post_ids)

    survivor = eng._lifecycle.get_order(pre_ids[0])
    dropped = eng._lifecycle.get_order(pre_ids[1])
    assert survivor.current_state == OrderState.VALIDATED
    assert dropped.current_state == OrderState.REJECTED
    assert dropped.reject_reason == "risk_control_block"


def test_lifecycle_full_happy_path_matches_by_symbol_side(tmp_path: Path) -> None:
    """Fills without order_id must match by symbol+side and walk to FILLED."""
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    orders = eng._lifecycle_attach(orders, "2025-01-15")
    pre_ids = list(orders["order_id"])
    eng._lifecycle_mark_validation(pre_ids, pre_ids)
    eng._lifecycle_mark_submitted(pre_ids)

    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "fill_qty": 10.0,
             "fill_price": 100.5, "status": "filled"},
        ]
    )
    eng._lifecycle_mark_fills(orders, fills, pre_ids)

    order = eng._lifecycle.get_order(pre_ids[0])
    assert order.current_state == OrderState.FILLED
    assert order.fill_qty == 10.0
    assert order.fill_price == 100.5


def test_lifecycle_partial_fill_is_partial_fill_state(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    orders = eng._lifecycle_attach(orders, "2025-01-15")
    pre_ids = list(orders["order_id"])
    eng._lifecycle_mark_validation(pre_ids, pre_ids)
    eng._lifecycle_mark_submitted(pre_ids)

    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "fill_qty": 4.0,
             "fill_price": 100.2, "status": "partial"},
        ]
    )
    eng._lifecycle_mark_fills(orders, fills, pre_ids)

    order = eng._lifecycle.get_order(pre_ids[0])
    assert order.current_state == OrderState.PARTIAL_FILL
    assert order.fill_qty == 4.0


def test_lifecycle_fill_row_rejected_goes_to_rejected(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    orders = eng._lifecycle_attach(orders, "2025-01-15")
    pre_ids = list(orders["order_id"])
    eng._lifecycle_mark_validation(pre_ids, pre_ids)
    eng._lifecycle_mark_submitted(pre_ids)

    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "fill_qty": 0.0,
             "fill_price": 100.0, "status": "rejected",
             "reject_reason": "INSUFFICIENT_CASH"},
        ]
    )
    eng._lifecycle_mark_fills(orders, fills, pre_ids)

    order = eng._lifecycle.get_order(pre_ids[0])
    assert order.current_state == OrderState.REJECTED
    assert order.reject_reason == "INSUFFICIENT_CASH"


def test_lifecycle_submitted_without_fill_row_is_cancelled(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    orders = eng._lifecycle_attach(orders, "2025-01-15")
    pre_ids = list(orders["order_id"])
    eng._lifecycle_mark_validation(pre_ids, pre_ids)
    eng._lifecycle_mark_submitted(pre_ids)

    # No fills DF at all — e.g. cash gate or missing price silently dropped them.
    eng._lifecycle_mark_fills(orders, pd.DataFrame(), pre_ids)

    order = eng._lifecycle.get_order(pre_ids[0])
    assert order.current_state == OrderState.CANCELLED
    assert order.events[-1].details.get("reason") == "eod_no_fill"


def test_lifecycle_dump_writes_jsonl_on_day(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10.0, "price": 100.0}]
    )
    orders = eng._lifecycle_attach(orders, "2025-01-15")
    pre_ids = list(orders["order_id"])
    eng._lifecycle_mark_validation(pre_ids, pre_ids)
    eng._lifecycle_mark_submitted(pre_ids)
    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "fill_qty": 10.0,
             "fill_price": 100.0, "status": "filled"},
        ]
    )
    eng._lifecycle_mark_fills(orders, fills, pre_ids)

    eng._lifecycle_dump("2025-01-15")

    path = eng.config.lifecycle_dir / "lifecycle_life_test_2025-01-15.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["symbol"] == "AAA"
    assert row["current_state"] == "FILLED"
    assert row["n_events"] >= 4  # CREATED + VALIDATED + SUBMITTED + FILLED


def test_lifecycle_disabled_is_noop(tmp_path: Path) -> None:
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_lifecycle_tracking=False,
        run_id="off",
    )
    eng = UnifiedPaperEngine(cfg)
    assert eng._lifecycle is None
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 5.0, "price": 100.0}]
    )
    result = eng._lifecycle_attach(orders, "2025-01-15")
    # When tracker is off, orders pass through unchanged (no order_id injected).
    assert "order_id" not in result.columns
