"""Unit tests for news_triggers_loader and disclosures_triggers_loader (T6.7)."""

from __future__ import annotations

import pytest

pytest.importorskip("src.assembled_core.intel.news_triggers_loader")

import json
import pytest

from src.assembled_core.intel.news_triggers_loader import (
    NewsTriggerSnapshot,
    load_news_triggers,
)
from src.assembled_core.intel.disclosures_triggers_loader import (
    DisclosuresTriggerSnapshot,
    load_disclosures_triggers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# news_triggers_loader
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestNewsTriggerLoader:
    def test_happy_path_triggers_key(self, tmp_path):
        """Valid artifact with 'triggers' key (artifact schema) → parsed correctly."""
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v1",
                "generated_utc": "2026-04-19T12:00:00Z",
                "triggers": [
                    {"trigger_id": "t1", "severity": 2, "confidence": 0.8},
                    {"trigger_id": "t2", "severity": 1, "confidence": 0.6},
                    {"trigger_id": "t3", "severity": 0, "confidence": 0.3},
                ],
                "summary": {},
            },
        )
        snap = load_news_triggers(p)
        assert snap.generated_utc == "2026-04-19T12:00:00Z"
        assert len(snap.triggers) == 3
        assert snap.summary["max_severity"] == 2
        assert snap.summary["watch_count_sev1plus"] == 2
        assert snap.summary["active_count_sev2plus"] == 1

    def test_happy_path_items_key(self, tmp_path):
        """Legacy artifact with 'items' key → still parsed correctly."""
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v1",
                "generated_utc": "2026-04-19T10:00:00Z",
                "items": [{"severity": 3}],
            },
        )
        snap = load_news_triggers(p)
        assert len(snap.triggers) == 1
        assert snap.summary["max_severity"] == 3

    def test_missing_file_returns_empty(self, tmp_path):
        snap = load_news_triggers(tmp_path / "nonexistent.json")
        assert isinstance(snap, NewsTriggerSnapshot)
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0

    def test_invalid_json_returns_empty(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        snap = load_news_triggers(p)
        assert snap.triggers == []

    def test_wrong_schema_version_returns_empty(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v0",
                "triggers": [{"severity": 2}],
            },
        )
        snap = load_news_triggers(p)
        assert snap.triggers == []

    def test_items_not_list_returns_empty(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v1",
                "triggers": "not-a-list",
            },
        )
        snap = load_news_triggers(p)
        assert snap.triggers == []

    def test_empty_triggers_list(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v1",
                "generated_utc": "2026-04-19T00:00:00Z",
                "triggers": [],
            },
        )
        snap = load_news_triggers(p)
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0

    def test_trigger_missing_severity_defaults_to_zero(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v1",
                "triggers": [{"trigger_id": "x"}],  # no severity field
            },
        )
        snap = load_news_triggers(p)
        assert snap.summary["max_severity"] == 0
        assert snap.summary["watch_count_sev1plus"] == 0


# ---------------------------------------------------------------------------
# disclosures_triggers_loader
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestDisclosuresTriggerLoader:
    def test_happy_path(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "disclosures.triggers.v1",
                "generated_utc": "2026-04-19T12:00:00Z",
                "items": [
                    {"disclosure_id": "d1", "severity": 3},
                    {"disclosure_id": "d2", "severity": 1},
                ],
            },
        )
        snap = load_disclosures_triggers(p)
        assert snap.generated_utc == "2026-04-19T12:00:00Z"
        assert len(snap.triggers) == 2
        assert snap.summary["max_severity"] == 3
        assert snap.summary["count_sev1plus"] == 2
        assert snap.summary["count_sev2plus"] == 1

    def test_missing_file_returns_empty_v2(self, tmp_path):
        snap = load_disclosures_triggers(tmp_path / "nonexistent.json")
        assert isinstance(snap, DisclosuresTriggerSnapshot)
        assert snap.triggers == []

    def test_invalid_json_returns_empty_v2(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{{invalid", encoding="utf-8")
        snap = load_disclosures_triggers(p)
        assert snap.triggers == []

    def test_wrong_schema_version_returns_empty_v2(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "news.triggers.v1",  # wrong schema
                "items": [{"severity": 2}],
            },
        )
        snap = load_disclosures_triggers(p)
        assert snap.triggers == []

    def test_empty_items_list(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "disclosures.triggers.v1",
                "generated_utc": "2026-04-19T00:00:00Z",
                "items": [],
            },
        )
        snap = load_disclosures_triggers(p)
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0

    def test_non_dict_item_skipped(self, tmp_path):
        p = _write_json(
            tmp_path,
            "triggers_latest.json",
            {
                "schema_version": "disclosures.triggers.v1",
                "items": ["not-a-dict", {"severity": 2}],
            },
        )
        snap = load_disclosures_triggers(p)
        assert snap.summary["max_severity"] == 2
        assert snap.summary["count_sev2plus"] == 1


# ---------------------------------------------------------------------------
# disclosures_triggers_loader — PIT as_of snapshot gate
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestDisclosuresTriggerLoaderPITGate:
    """Snapshot-level PIT gate on the single top-level generated_utc.

    The file's generated_utc is 2026-05-21. The gate must:
    - block when the snapshot post-dates as_of (look-ahead),
    - load when the snapshot pre-dates/equals as_of,
    - stay byte-identical for as_of=None (live/paper back-compat),
    - fail safe (empty) when generated_utc cannot be dated.
    """

    GEN_UTC = "2026-05-21T12:00:00Z"

    def _snapshot_file(self, tmp_path, generated_utc=GEN_UTC):
        payload = {
            "schema_version": "disclosures.triggers.v1",
            "items": [
                {"disclosure_id": "d1", "severity": 3},
                {"disclosure_id": "d2", "severity": 1},
            ],
        }
        if generated_utc is not None:
            payload["generated_utc"] = generated_utc
        return _write_json(tmp_path, "triggers_latest.json", payload)

    def test_as_of_before_generated_returns_empty(self, tmp_path):
        """as_of earlier than the snapshot -> not yet available -> empty."""
        p = self._snapshot_file(tmp_path)
        snap = load_disclosures_triggers(p, as_of="2026-01-01")
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0
        assert snap.summary["count_sev1plus"] == 0
        assert snap.summary["count_sev2plus"] == 0

    def test_as_of_after_generated_loads_snapshot(self, tmp_path):
        """as_of later than the snapshot -> snapshot is available -> loads."""
        p = self._snapshot_file(tmp_path)
        snap = load_disclosures_triggers(p, as_of="2026-06-01")
        assert len(snap.triggers) == 2
        assert snap.summary["max_severity"] == 3
        assert snap.summary["count_sev1plus"] == 2
        assert snap.summary["count_sev2plus"] == 1

    def test_as_of_equal_generated_loads_snapshot(self, tmp_path):
        """Boundary: as_of == generated_utc -> available (gate is strict >)."""
        p = self._snapshot_file(tmp_path)
        snap = load_disclosures_triggers(p, as_of="2026-05-21T12:00:00Z")
        assert len(snap.triggers) == 2
        assert snap.summary["max_severity"] == 3

    def test_as_of_none_is_current_behaviour(self, tmp_path):
        """as_of=None -> byte-identical to the no-gate path (live/paper)."""
        p = self._snapshot_file(tmp_path)
        snap_none = load_disclosures_triggers(p, as_of=None)
        snap_default = load_disclosures_triggers(p)
        assert snap_none == snap_default
        assert len(snap_none.triggers) == 2
        assert snap_none.generated_utc == self.GEN_UTC
        assert snap_none.summary["max_severity"] == 3

    def test_as_of_set_missing_generated_fails_safe(self, tmp_path):
        """as_of set + generated_utc absent -> cannot prove PIT -> empty."""
        p = self._snapshot_file(tmp_path, generated_utc=None)
        # Sanity: without the gate this file loads its 2 items.
        assert len(load_disclosures_triggers(p).triggers) == 2
        snap = load_disclosures_triggers(p, as_of="2026-06-01")
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0

    def test_as_of_set_unparseable_generated_fails_safe(self, tmp_path):
        """as_of set + generated_utc garbage -> cannot prove PIT -> empty."""
        p = self._snapshot_file(tmp_path, generated_utc="not-a-timestamp")
        assert len(load_disclosures_triggers(p).triggers) == 2
        snap = load_disclosures_triggers(p, as_of="2026-06-01")
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0

    def test_as_of_unparseable_fails_safe(self, tmp_path):
        """Unparseable as_of -> cannot prove PIT -> empty (defensive)."""
        p = self._snapshot_file(tmp_path)
        snap = load_disclosures_triggers(p, as_of="garbage-as-of")
        assert snap.triggers == []
        assert snap.summary["max_severity"] == 0

    def test_as_of_naive_datetime_treated_as_utc(self, tmp_path):
        """Naive datetime as_of is localized to UTC (PEAD idiom), gate works."""
        from datetime import datetime

        p = self._snapshot_file(tmp_path)
        # Naive 2026-06-01 (after) -> loads.
        snap_after = load_disclosures_triggers(p, as_of=datetime(2026, 6, 1))
        assert len(snap_after.triggers) == 2
        # Naive 2026-01-01 (before) -> empty.
        snap_before = load_disclosures_triggers(p, as_of=datetime(2026, 1, 1))
        assert snap_before.triggers == []
