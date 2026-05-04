"""Tests for src/assembled_core/ops/incident_tracker.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from assembled_core.ops.incident_tracker import (
    IncidentRecord,
    IncidentTracker,
    Severity,
)

# ---------------------------------------------------------------------------
# IncidentRecord
# ---------------------------------------------------------------------------


class TestIncidentRecord:
    def test_defaults(self):
        record = IncidentRecord(title="test", severity=Severity.SEV3)
        assert record.incident_id
        assert len(record.incident_id) == 8
        assert isinstance(record.started_at, datetime)
        assert record.resolved_at is None
        assert not record.is_resolved

    def test_is_resolved_after_resolve(self):
        record = IncidentRecord(title="t", severity=Severity.SEV1)
        record.resolved_at = datetime.now(timezone.utc)
        assert record.is_resolved

    def test_duration_minutes_none_when_open(self):
        record = IncidentRecord(title="t", severity=Severity.SEV2)
        assert record.duration_minutes is None

    def test_duration_minutes_computed(self):
        from datetime import timedelta

        record = IncidentRecord(title="t", severity=Severity.SEV1)
        record.resolved_at = record.started_at + timedelta(minutes=30)
        assert abs(record.duration_minutes - 30.0) < 0.1

    def test_to_dict_roundtrip(self):
        record = IncidentRecord(
            title="API outage",
            severity=Severity.SEV1,
            impact="No orders submitted",
            root_cause="Alpaca HTTP 503",
            action_items=["Check status page"],
        )
        d = record.to_dict()
        restored = IncidentRecord.from_dict(d)
        assert restored.title == record.title
        assert restored.severity == Severity.SEV1
        assert restored.action_items == ["Check status page"]
        assert restored.incident_id == record.incident_id

    def test_severity_values(self):
        assert Severity.SEV1.value == "SEV1"
        assert Severity.SEV4.value == "SEV4"


# ---------------------------------------------------------------------------
# IncidentTracker
# ---------------------------------------------------------------------------


class TestIncidentTracker:
    def test_open_incident_creates_file(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        record = tracker.open_incident(
            "Broker API down",
            severity=Severity.SEV1,
            impact="Can't submit orders",
        )
        assert record.incident_id
        assert (tmp_path / f"{record.incident_id}.json").exists()

    def test_load_roundtrip(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        record = tracker.open_incident("News feed stale", Severity.SEV3)
        loaded = tracker.load(record.incident_id)
        assert loaded.title == "News feed stale"
        assert loaded.severity == Severity.SEV3

    def test_load_nonexistent_raises(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            tracker.load("notexist")

    def test_resolve_incident(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        record = tracker.open_incident("DB OOM", Severity.SEV1)
        resolved = tracker.resolve_incident(
            record.incident_id,
            root_cause="shared_buffers too high",
            action_items=["Tune postgres config"],
        )
        assert resolved.is_resolved
        assert resolved.root_cause == "shared_buffers too high"
        assert "Tune postgres config" in resolved.action_items

    def test_list_open_only_includes_open(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        r1 = tracker.open_incident("incident A", Severity.SEV2)
        r2 = tracker.open_incident("incident B", Severity.SEV3)
        tracker.resolve_incident(r1.incident_id, root_cause="fixed")

        open_incidents = tracker.list_open()
        ids = [r.incident_id for r in open_incidents]
        assert r1.incident_id not in ids
        assert r2.incident_id in ids

    def test_list_all_includes_resolved(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        r1 = tracker.open_incident("A", Severity.SEV1)
        r2 = tracker.open_incident("B", Severity.SEV2)
        tracker.resolve_incident(r1.incident_id, root_cause="done")

        all_incidents = tracker.list_all()
        assert len(all_incidents) == 2

    def test_generate_postmortem_contains_title(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        record = tracker.open_incident(
            "Kill-Switch tripped unexpectedly",
            Severity.SEV2,
            runbook_ref="docs/runbooks/03_kill_switch_triggered.md",
        )
        tracker.resolve_incident(
            record.incident_id,
            root_cause="consecutive_losses threshold too sensitive",
            action_items=["Adjust threshold", "Review signal"],
        )
        postmortem = tracker.generate_postmortem(record.incident_id)

        assert "Kill-Switch tripped unexpectedly" in postmortem
        assert "SEV2" in postmortem
        assert "consecutive_losses" in postmortem
        assert "Adjust threshold" in postmortem
        assert "docs/runbooks/03_kill_switch_triggered.md" in postmortem

    def test_severity_string_accepted(self, tmp_path):
        tracker = IncidentTracker(incidents_dir=tmp_path)
        record = tracker.open_incident("test", severity="SEV4")
        assert record.severity == Severity.SEV4
