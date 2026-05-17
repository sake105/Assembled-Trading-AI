"""Tests for M4 execution.intent_store — idempotency keys and audit trail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

from src.assembled_core.execution.intent_store import (
    filter_intents_by_action,
    has_intent,
    load_intents,
    make_daily_key,
    make_run_key,
    record_intent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    """Return a temporary intent store path."""
    return tmp_path / "intent_store.jsonl"


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


class TestMakeDailyKey:
    def test_same_action_same_date_gives_same_key(self):
        k1 = make_daily_key("STOP", "2026-03-30")
        k2 = make_daily_key("STOP", "2026-03-30")
        assert k1 == k2

    def test_different_actions_give_different_keys(self):
        k1 = make_daily_key("STOP", "2026-03-30")
        k2 = make_daily_key("KILL", "2026-03-30")
        assert k1 != k2

    def test_different_dates_give_different_keys(self):
        k1 = make_daily_key("STOP", "2026-03-30")
        k2 = make_daily_key("STOP", "2026-03-31")
        assert k1 != k2

    def test_key_is_16_chars_hex(self):
        key = make_daily_key("STOP", "2026-03-30")
        assert len(key) == 16
        assert all(c in "0123456789abcdef" for c in key)

    def test_default_date_returns_string(self):
        key = make_daily_key("RECONCILE")
        assert isinstance(key, str)
        assert len(key) == 16


class TestMakeRunKey:
    def test_same_action_same_run_id_gives_same_key(self):
        k1 = make_run_key("RECONCILE", "run_001")
        k2 = make_run_key("RECONCILE", "run_001")
        assert k1 == k2

    def test_different_run_ids_give_different_keys(self):
        k1 = make_run_key("RECONCILE", "run_001")
        k2 = make_run_key("RECONCILE", "run_002")
        assert k1 != k2


# ---------------------------------------------------------------------------
# load_intents
# ---------------------------------------------------------------------------


class TestLoadIntents:
    def test_returns_empty_list_when_store_missing(self, tmp_path: Path):
        result = load_intents(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_returns_records_from_valid_store(self, store: Path):
        store.write_text(
            '{"action":"STOP","idempotency_key":"abc","timestamp_utc":"2026-03-30T00:00:00+00:00","metadata":{}}\n',
            encoding="utf-8",
        )
        records = load_intents(store)
        assert len(records) == 1
        assert records[0]["action"] == "STOP"

    def test_skips_malformed_lines(self, store: Path):
        store.write_text(
            '{"action":"STOP","idempotency_key":"abc","timestamp_utc":"t","metadata":{}}\n'
            "not-json\n"
            '{"action":"KILL","idempotency_key":"xyz","timestamp_utc":"t","metadata":{}}\n',
            encoding="utf-8",
        )
        records = load_intents(store)
        assert len(records) == 2
        assert records[0]["action"] == "STOP"
        assert records[1]["action"] == "KILL"

    def test_skips_blank_lines(self, store: Path):
        store.write_text(
            '{"action":"STOP","idempotency_key":"abc","timestamp_utc":"t","metadata":{}}\n'
            "\n"
            "\n",
            encoding="utf-8",
        )
        records = load_intents(store)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# has_intent
# ---------------------------------------------------------------------------


class TestHasIntent:
    def test_returns_false_when_store_missing(self, tmp_path: Path):
        assert has_intent("any_key", tmp_path / "nonexistent.jsonl") is False

    def test_returns_false_when_key_not_in_store(self, store: Path):
        record_intent("STOP", "key_a", store_path=store)
        assert has_intent("key_b", store) is False

    def test_returns_true_when_key_exists(self, store: Path):
        record_intent("STOP", "key_a", store_path=store)
        assert has_intent("key_a", store) is True

    def test_key_match_is_exact(self, store: Path):
        record_intent("STOP", "abc123", store_path=store)
        assert has_intent("abc12", store) is False
        assert has_intent("abc123", store) is True


# ---------------------------------------------------------------------------
# record_intent
# ---------------------------------------------------------------------------


class TestRecordIntent:
    def test_creates_store_if_missing(self, tmp_path: Path):
        store = tmp_path / "subdir" / "intent_store.jsonl"
        record_intent("STOP", "key1", store_path=store)
        assert store.exists()

    def test_returned_record_has_expected_fields(self, store: Path):
        rec = record_intent(
            "STOP", "key1", metadata={"reason": "test"}, store_path=store
        )
        assert rec["action"] == "STOP"
        assert rec["idempotency_key"] == "key1"
        assert "timestamp_utc" in rec
        assert rec["metadata"]["reason"] == "test"

    def test_record_is_persisted_to_file(self, store: Path):
        record_intent("KILL", "key2", store_path=store)
        lines = store.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["action"] == "KILL"
        assert parsed["idempotency_key"] == "key2"

    def test_multiple_records_appended_in_order(self, store: Path):
        record_intent("STOP", "k1", store_path=store)
        record_intent("KILL", "k2", store_path=store)
        record_intent("RECONCILE", "k3", store_path=store)
        records = load_intents(store)
        assert len(records) == 3
        assert records[0]["action"] == "STOP"
        assert records[1]["action"] == "KILL"
        assert records[2]["action"] == "RECONCILE"

    def test_duplicate_key_is_recorded_not_blocked(self, store: Path):
        """record_intent itself does not enforce idempotency — caller must check."""
        record_intent("STOP", "dup_key", store_path=store)
        record_intent("STOP", "dup_key", store_path=store)
        records = load_intents(store)
        assert len(records) == 2

    def test_metadata_defaults_to_empty_dict(self, store: Path):
        rec = record_intent("FLATTEN", "key_no_meta", store_path=store)
        assert rec["metadata"] == {}


# ---------------------------------------------------------------------------
# filter_intents_by_action
# ---------------------------------------------------------------------------


class TestFilterIntentsByAction:
    def test_returns_only_matching_actions(self, store: Path):
        record_intent("STOP", "k1", store_path=store)
        record_intent("KILL", "k2", store_path=store)
        record_intent("STOP", "k3", store_path=store)

        stops = filter_intents_by_action("STOP", store)
        assert len(stops) == 2
        assert all(r["action"] == "STOP" for r in stops)

    def test_returns_empty_if_no_match(self, store: Path):
        record_intent("STOP", "k1", store_path=store)
        result = filter_intents_by_action("FLATTEN", store)
        assert result == []

    def test_returns_empty_for_missing_store(self, tmp_path: Path):
        result = filter_intents_by_action("KILL", tmp_path / "missing.jsonl")
        assert result == []


# ---------------------------------------------------------------------------
# Idempotency pattern (caller-level) integration test
# ---------------------------------------------------------------------------


class TestIdempotencyPattern:
    """Tests simulating the caller-level idempotency guard used in workers."""

    def test_stop_worker_pattern_is_idempotent(self, store: Path):
        key = make_daily_key("STOP", "2026-03-30")

        # First call: key absent → record
        assert not has_intent(key, store)
        record_intent("STOP", key, metadata={"reason": "test"}, store_path=store)
        assert has_intent(key, store)

        # Second call: key present → would skip (caller's responsibility)
        records_before = load_intents(store)
        if has_intent(key, store):
            pass  # simulate skip — do NOT record again
        records_after = load_intents(store)

        assert len(records_before) == len(records_after)

    def test_force_override_adds_second_record(self, store: Path):
        key = make_daily_key("KILL", "2026-03-30")

        record_intent("KILL", key, store_path=store)
        assert has_intent(key, store)

        # Force: record regardless
        record_intent("KILL", key, metadata={"forced": True}, store_path=store)

        records = load_intents(store)
        assert len(records) == 2

    def test_daily_key_distinct_from_run_key(self, store: Path):
        daily = make_daily_key("RECONCILE", "2026-03-30")
        run = make_run_key("RECONCILE", "20260330_120000")

        record_intent("RECONCILE", daily, store_path=store)
        record_intent("RECONCILE", run, store_path=store)

        assert has_intent(daily, store)
        assert has_intent(run, store)
        assert daily != run
