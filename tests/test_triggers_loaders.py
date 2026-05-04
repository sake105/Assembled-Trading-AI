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


@pytest.mark.phase12
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


@pytest.mark.phase12
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
