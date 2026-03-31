"""Load disclosures triggers snapshot for TradingContext (read-only, tolerant)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_SCHEMA = "disclosures.triggers.v1"


@dataclass
class DisclosuresTriggerSnapshot:
    """Read-only snapshot of disclosures triggers artifact."""

    generated_utc: str = ""
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_severity": 0,
            "count_sev1plus": 0,
            "count_sev2plus": 0,
        }
    )


def load_disclosures_triggers(path: str | Path) -> DisclosuresTriggerSnapshot:
    """Load triggers_latest.json. Tolerant: missing/invalid -> empty snapshot.

    - Requires schema_version == "disclosures.triggers.v1"
    - items must be a list
    - summary: max_severity, count_sev1plus (severity >= 1), count_sev2plus (severity >= 2)
    """
    p = Path(path) if not isinstance(path, Path) else path
    empty = DisclosuresTriggerSnapshot()

    if not p.exists():
        return empty
    try:
        raw = p.read_text(encoding="utf-8")
    except Exception:
        return empty
    try:
        data = json.loads(raw)
    except Exception:
        return empty
    if not isinstance(data, dict):
        return empty
    if data.get("schema_version") != REQUIRED_SCHEMA:
        return empty
    items = data.get("items")
    if not isinstance(items, list):
        return empty

    generated_utc = str(data.get("generated_utc") or "")
    max_sev = 0
    count_sev1 = 0
    count_sev2 = 0
    for t in items:
        if not isinstance(t, dict):
            continue
        sev = int(t.get("severity", 0))
        if sev > max_sev:
            max_sev = sev
        if sev >= 1:
            count_sev1 += 1
        if sev >= 2:
            count_sev2 += 1

    return DisclosuresTriggerSnapshot(
        generated_utc=generated_utc,
        triggers=list(items),
        summary={
            "max_severity": max_sev,
            "count_sev1plus": count_sev1,
            "count_sev2plus": count_sev2,
        },
    )
