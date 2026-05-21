"""Tests for src.assembled_core.execution.intent_store.auto_abandon_stale_intents.

F-RX-11 §9.12 (g): auto-abandon stale pre-submit ORDER_SUBMIT intents
(empty broker_order_id, older than N hours) so they don't accumulate and
require manual reconciliation. Already happened twice in five days during
the pilot outage 2026-05-15..21.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.execution.intent_store import (
    auto_abandon_stale_intents,
    find_pending_order_intents,
)


def _store_path(tmp_path):
    return tmp_path / "intents.jsonl"


def _make_submit(
    tmp_path,
    *,
    key: str,
    age_hours: float,
    broker_order_id: str = "",
):
    """Write a synthetic ORDER_SUBMIT intent N hours in the past.

    record_intent() always stamps with datetime.now(), so we write the JSONL
    record manually with a backdated timestamp for tests that need control
    over age.
    """
    import json

    sp = _store_path(tmp_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)).isoformat()
    record = {
        "action": "ORDER_SUBMIT",
        "idempotency_key": key,
        "timestamp_utc": ts,
        "metadata": {
            "symbol": "MSFT",
            "side": "buy",
            "qty": 5.0,
            "broker_order_id": broker_order_id,
        },
    }
    with open(sp, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def test_auto_abandon_acts_on_stale_pre_submit_intents(tmp_path):
    sp = _store_path(tmp_path)
    _make_submit(tmp_path, key="stale_pre_submit", age_hours=30.0)

    pending_before = find_pending_order_intents(sp)
    assert len(pending_before) == 1

    abandoned = auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp)
    assert len(abandoned) == 1

    pending_after = find_pending_order_intents(sp)
    assert pending_after == []


def test_auto_abandon_skips_recent_intents(tmp_path):
    sp = _store_path(tmp_path)
    _make_submit(tmp_path, key="recent_intent", age_hours=5.0)

    abandoned = auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp)
    assert abandoned == []
    assert len(find_pending_order_intents(sp)) == 1


def test_auto_abandon_skips_intents_with_broker_order_id_by_default(tmp_path):
    """Pre-submit intents have empty broker_order_id; submitted ones have one
    and must be reconciled against the broker, not auto-abandoned."""
    sp = _store_path(tmp_path)
    _make_submit(
        tmp_path,
        key="has_broker_id",
        age_hours=30.0,
        broker_order_id="abc123",
    )

    abandoned = auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp)
    assert abandoned == []
    assert len(find_pending_order_intents(sp)) == 1


def test_auto_abandon_can_force_with_broker_id_flag(tmp_path):
    sp = _store_path(tmp_path)
    _make_submit(
        tmp_path,
        key="forced",
        age_hours=30.0,
        broker_order_id="abc123",
    )

    abandoned = auto_abandon_stale_intents(
        max_age_hours=24.0,
        require_empty_broker_order_id=False,
        store_path=sp,
    )
    assert len(abandoned) == 1
    assert find_pending_order_intents(sp) == []


def test_auto_abandon_writes_audit_trail(tmp_path):
    sp = _store_path(tmp_path)
    _make_submit(tmp_path, key="audit_trail_check", age_hours=30.0)

    auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp)

    # Read raw intents to confirm the abandonment record has the expected shape
    import json

    lines = sp.read_text(encoding="utf-8").strip().splitlines()
    completions = [
        json.loads(ln)
        for ln in lines
        if json.loads(ln).get("action") == "ORDER_COMPLETE"
    ]
    assert len(completions) == 1
    meta = completions[0]["metadata"]
    assert meta["status"] == "abandoned_auto"
    assert "auto-abandoned" in meta["reason"]
    assert "broker_order_id was empty" in meta["reason"]


def test_auto_abandon_returns_empty_when_no_pending(tmp_path):
    sp = _store_path(tmp_path)
    # No intents recorded; store doesn't even exist yet
    assert auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp) == []


def test_auto_abandon_handles_tz_naive_timestamp_without_crash(tmp_path):
    """F-RX-FU-2 regression: a tz-naive timestamp_utc from older formats
    must NOT abort the whole sweep. The helper coerces to UTC and
    continues; if a single record is malformed beyond repair it logs WARN
    and skips that one only.
    """
    import json

    sp = _store_path(tmp_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    # One tz-naive (legacy) stale record + one well-formed stale record
    naive_ts = (
        datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=30)
    ).isoformat()
    aware_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    with open(sp, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "action": "ORDER_SUBMIT",
                    "idempotency_key": "naive_record",
                    "timestamp_utc": naive_ts,
                    "metadata": {
                        "symbol": "AAA",
                        "side": "buy",
                        "qty": 1.0,
                        "broker_order_id": "",
                    },
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "action": "ORDER_SUBMIT",
                    "idempotency_key": "aware_record",
                    "timestamp_utc": aware_ts,
                    "metadata": {
                        "symbol": "BBB",
                        "side": "buy",
                        "qty": 1.0,
                        "broker_order_id": "",
                    },
                }
            )
            + "\n"
        )

    abandoned = auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp)
    # Both records reach the threshold; tz-naive is coerced, both abandoned.
    keys = {a["idempotency_key"] for a in abandoned}
    assert keys == {"naive_record", "aware_record"}


def test_auto_abandon_skips_malformed_timestamp_continues_others(tmp_path):
    """One malformed record must not poison the rest (defensive sweep)."""
    import json

    sp = _store_path(tmp_path)
    sp.parent.mkdir(parents=True, exist_ok=True)
    aware_ts = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
    with open(sp, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "action": "ORDER_SUBMIT",
                    "idempotency_key": "malformed",
                    "timestamp_utc": "not-an-iso-string",
                    "metadata": {
                        "symbol": "AAA",
                        "side": "buy",
                        "qty": 1.0,
                        "broker_order_id": "",
                    },
                }
            )
            + "\n"
        )
        fh.write(
            json.dumps(
                {
                    "action": "ORDER_SUBMIT",
                    "idempotency_key": "good_record",
                    "timestamp_utc": aware_ts,
                    "metadata": {
                        "symbol": "BBB",
                        "side": "buy",
                        "qty": 1.0,
                        "broker_order_id": "",
                    },
                }
            )
            + "\n"
        )

    abandoned = auto_abandon_stale_intents(max_age_hours=24.0, store_path=sp)
    keys = {a["idempotency_key"] for a in abandoned}
    # malformed silently skipped, good_record abandoned successfully
    assert keys == {"good_record"}
