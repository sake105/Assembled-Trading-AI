"""Phase 6 regression tests for engine → intent-store integration.

Covers:

* ``enable_intent_store=False`` (default) → no records written
* ``enable_intent_store=True`` → every submitted order produces a paired
  ORDER_SUBMIT + ORDER_COMPLETE record with the same idempotency key
* Rejected fills still produce ORDER_COMPLETE with status=rejected
* No pending intents remain after a clean day
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.assembled_core.execution.intent_store import (
    find_pending_order_intents,
    load_intents,
)
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    enable_intent_store: bool,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_intent_store=enable_intent_store,
        intent_store_path=tmp_path / "intents.jsonl",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="intent_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_intent_store_disabled_writes_nothing(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_intent_store=False)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    keys = eng._record_submit_intents(orders)
    assert keys == []
    assert not (tmp_path / "intents.jsonl").exists()


def test_intent_store_submit_writes_record(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_intent_store=True)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0},
            {"symbol": "BBB", "side": "SELL", "qty": 50.0, "price": 50.0},
        ]
    )
    keys = eng._record_submit_intents(orders)
    assert len(keys) == 2
    assert all(keys)  # no empty strings
    records = load_intents(store_path=tmp_path / "intents.jsonl")
    assert len(records) == 2
    assert all(r["action"] == "ORDER_SUBMIT" for r in records)
    assert {r["metadata"]["symbol"] for r in records} == {"AAA", "BBB"}


def test_intent_store_submit_and_complete_pair(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_intent_store=True)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    keys = eng._record_submit_intents(orders)
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 100.0,
                "fill_qty": 100.0,
                "fill_price": 100.5,
                "status": "filled",
            }
        ]
    )
    eng._record_complete_intents(orders, fills, keys)
    records = load_intents(store_path=tmp_path / "intents.jsonl")
    assert len(records) == 2
    # Same idempotency key pairs submit + complete.
    submit = [r for r in records if r["action"] == "ORDER_SUBMIT"][0]
    complete = [r for r in records if r["action"] == "ORDER_COMPLETE"][0]
    assert submit["idempotency_key"] == complete["idempotency_key"]
    assert complete["metadata"]["status"] == "filled"
    assert complete["metadata"]["filled_qty"] == 100.0
    # No pending intents remain.
    pending = find_pending_order_intents(store_path=tmp_path / "intents.jsonl")
    assert pending == []


def test_intent_store_pairs_duplicate_symbol_side_by_order_id(tmp_path: Path) -> None:
    """H1 regression: two orders sharing (symbol, side) but different
    ``order_id`` must produce two distinct ORDER_COMPLETE records with the
    right ``filled_qty`` on each, not two copies of the first fill.
    """
    eng = _make_engine(tmp_path, enable_intent_store=True)
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0, "order_id": "o1"},
            {"symbol": "AAA", "side": "BUY", "qty": 40.0, "price": 100.0, "order_id": "o2"},
        ]
    )
    keys = eng._record_submit_intents(orders)
    assert len(keys) == 2
    # Both keys must be distinct idempotency strings.
    assert keys[0][1] != keys[1][1]
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA", "side": "BUY", "qty": 100.0,
                "fill_qty": 100.0, "fill_price": 100.5, "status": "filled",
                "order_id": "o1",
            },
            {
                "symbol": "AAA", "side": "BUY", "qty": 40.0,
                "fill_qty": 40.0, "fill_price": 100.7, "status": "filled",
                "order_id": "o2",
            },
        ]
    )
    eng._record_complete_intents(orders, fills, keys)

    records = load_intents(store_path=tmp_path / "intents.jsonl")
    completes = [r for r in records if r["action"] == "ORDER_COMPLETE"]
    assert len(completes) == 2
    qtys = sorted(float(r["metadata"]["filled_qty"]) for r in completes)
    prices = sorted(float(r["metadata"]["filled_price"]) for r in completes)
    # Each order must see its own fill — not the first fill twice.
    assert qtys == [40.0, 100.0]
    assert prices == [100.5, 100.7]
    # Each complete pairs to its own submit.
    submits = [r for r in records if r["action"] == "ORDER_SUBMIT"]
    submit_keys = {r["idempotency_key"] for r in submits}
    complete_keys = {r["idempotency_key"] for r in completes}
    assert submit_keys == complete_keys
    assert len(complete_keys) == 2


def test_intent_store_rejected_fill_marks_complete(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_intent_store=True)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    keys = eng._record_submit_intents(orders)
    # No corresponding fill → rejection.
    fills = pd.DataFrame(columns=["symbol", "side", "qty", "fill_qty", "fill_price", "status"])
    eng._record_complete_intents(orders, fills, keys)
    records = load_intents(store_path=tmp_path / "intents.jsonl")
    complete = [r for r in records if r["action"] == "ORDER_COMPLETE"][0]
    assert complete["metadata"]["status"] == "rejected"
    assert complete["metadata"]["filled_qty"] == 0.0
