"""Load news triggers snapshot for TradingContext (read-only, tolerant)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_SCHEMA = "news.triggers.v1"


@dataclass
class NewsTriggerSnapshot:
    """Read-only snapshot of news triggers artifact."""

    generated_utc: str = ""
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_severity": 0,
            "watch_count_sev1plus": 0,
            "active_count_sev2plus": 0,
        }
    )


def load_news_triggers(path: str | Path) -> NewsTriggerSnapshot:
    """Load triggers_latest.json. Tolerant: missing/invalid -> empty snapshot.

    - Requires schema_version == "news.triggers.v1"
    - items must be a list of dicts with optional 'severity' field
    - summary keys match what paper_runner expects:
        max_severity, watch_count_sev1plus, active_count_sev2plus
    """
    p = Path(path) if not isinstance(path, Path) else path
    empty = NewsTriggerSnapshot()

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
        sev = int(t.get("severity") or 0)
        if sev > max_sev:
            max_sev = sev
        if sev >= 1:
            count_sev1 += 1
        if sev >= 2:
            count_sev2 += 1

    return NewsTriggerSnapshot(
        generated_utc=generated_utc,
        triggers=list(items),
        summary={
            "max_severity": max_sev,
            "watch_count_sev1plus": count_sev1,
            "active_count_sev2plus": count_sev2,
        },
    )
