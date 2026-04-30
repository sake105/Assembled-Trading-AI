"""Incident tracking and post-mortem generation.

From 51_INCIDENT_PLAYBOOK.md.

Provides a lightweight, file-backed incident registry so that SEV1/SEV2
events are recorded and auditable without requiring a running database.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class Severity(str, Enum):
    SEV1 = "SEV1"   # money-impacting, immediate action required
    SEV2 = "SEV2"   # system stopped, 1-hour window
    SEV3 = "SEV3"   # signal quality degraded, 24-hour window
    SEV4 = "SEV4"   # cosmetic / monitoring noise, next business day


@dataclass
class IncidentRecord:
    """A single incident record."""

    title: str
    severity: Severity
    root_cause: str = ""
    impact: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    runbook_ref: str = ""      # e.g. "docs/runbooks/01_broker_api_unreachable.md"
    action_items: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None

    @property
    def duration_minutes(self) -> float | None:
        if self.resolved_at is None:
            return None
        delta = self.resolved_at - self.started_at
        return delta.total_seconds() / 60

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["started_at"] = self.started_at.isoformat()
        d["resolved_at"] = self.resolved_at.isoformat() if self.resolved_at else None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "IncidentRecord":
        data = dict(data)
        data["severity"] = Severity(data["severity"])
        data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("resolved_at"):
            data["resolved_at"] = datetime.fromisoformat(data["resolved_at"])
        else:
            data["resolved_at"] = None
        return cls(**data)


class IncidentTracker:
    """File-backed incident registry.

    Stores incidents as newline-delimited JSON in *incidents_dir*.
    One file per incident: ``<incident_id>.json``.

    Args:
        incidents_dir: Directory path for incident JSON files.
            Created on first use if it does not exist.
    """

    def __init__(self, incidents_dir: str | Path = "docs/incidents") -> None:
        self.incidents_dir = Path(incidents_dir)
        self.incidents_dir.mkdir(parents=True, exist_ok=True)

    def open_incident(
        self,
        title: str,
        severity: Severity | str,
        impact: str = "",
        runbook_ref: str = "",
    ) -> IncidentRecord:
        """Create and persist a new incident."""
        if isinstance(severity, str):
            severity = Severity(severity)
        record = IncidentRecord(
            title=title,
            severity=severity,
            impact=impact,
            runbook_ref=runbook_ref,
        )
        self._save(record)
        return record

    def resolve_incident(
        self,
        incident_id: str,
        root_cause: str,
        action_items: list[str] | None = None,
        notes: str = "",
    ) -> IncidentRecord:
        """Mark an existing incident as resolved."""
        record = self.load(incident_id)
        record.resolved_at = datetime.now(timezone.utc)
        record.root_cause = root_cause
        if action_items:
            record.action_items = action_items
        record.notes = notes
        self._save(record)
        return record

    def load(self, incident_id: str) -> IncidentRecord:
        """Load a single incident by its short ID."""
        path = self._path(incident_id)
        if not path.exists():
            raise FileNotFoundError(f"Incident {incident_id!r} not found at {path}")
        return IncidentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_open(self) -> list[IncidentRecord]:
        """Return all incidents that are not yet resolved."""
        return [r for r in self._load_all() if not r.is_resolved]

    def list_all(self) -> list[IncidentRecord]:
        """Return all incidents, sorted by start time descending."""
        return sorted(self._load_all(), key=lambda r: r.started_at, reverse=True)

    def generate_postmortem(self, incident_id: str) -> str:
        """Render a Markdown post-mortem template for *incident_id*."""
        record = self.load(incident_id)
        started = record.started_at.strftime("%Y-%m-%d %H:%M UTC")
        resolved = (
            record.resolved_at.strftime("%Y-%m-%d %H:%M UTC")
            if record.resolved_at
            else "OPEN"
        )
        duration = (
            f"{record.duration_minutes:.0f} min" if record.duration_minutes else "ongoing"
        )

        action_lines = "\n".join(
            f"- [ ] {item}" for item in record.action_items
        ) or "- [ ] TBD"

        return f"""# Post-Mortem: {record.title}

**Date:** {started}
**Duration:** {started} – {resolved} ({duration})
**Impact:** {record.impact or "TBD"}
**Severity:** {record.severity.value}
**Root Cause:** {record.root_cause or "TBD"}

## Timeline

- {started} Symptom first observed
- [HH:MM] Detection (how?)
- [HH:MM] Immediate action taken
- [HH:MM] Root cause identified
- {resolved} Resolution / normal operations

## What went well

- (fill in)

## What went wrong

- (fill in)

## Action Items

{action_lines}

---
*Incident ID: {record.incident_id}*
*Runbook: {record.runbook_ref or "N/A"}*
"""

    # ── internal ────────────────────────────────────────────────────────────

    def _path(self, incident_id: str) -> Path:
        return self.incidents_dir / f"{incident_id}.json"

    def _save(self, record: IncidentRecord) -> None:
        self._path(record.incident_id).write_text(
            json.dumps(record.to_dict(), indent=2),
            encoding="utf-8",
        )

    def _load_all(self) -> list[IncidentRecord]:
        records = []
        for p in self.incidents_dir.glob("????????.json"):
            try:
                records.append(IncidentRecord.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:  # noqa: BLE001
                pass
        return records
