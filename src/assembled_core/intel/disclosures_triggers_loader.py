"""Load disclosures triggers snapshot for TradingContext (read-only, tolerant)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_SCHEMA = "disclosures.triggers.v1"


def _to_utc_dt(value: str | datetime | None) -> datetime | None:
    """Coerce str/datetime/naive to tz-aware UTC. Return None if unparseable.

    Naive datetimes (and naive ISO strings) are treated as genuinely-UTC and
    localized to UTC, mirroring the PEAD / pit_store PIT idiom. Any parse
    failure returns None so the caller can fail safe.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def load_disclosures_triggers(
    path: str | Path,
    as_of: str | datetime | None = None,
) -> DisclosuresTriggerSnapshot:
    """Load triggers_latest.json. Tolerant: missing/invalid -> empty snapshot.

    - Requires schema_version == "disclosures.triggers.v1"
    - items must be a list
    - summary: max_severity, count_sev1plus (severity >= 1), count_sev2plus (severity >= 2)

    PIT guard (``as_of``):
    - ``as_of is None`` -> exactly current behaviour (live/paper path),
      byte-identical, fully back-compatible.
    - ``as_of`` set -> snapshot-level PIT gate. The file carries a single
      top-level ``generated_utc`` for the whole snapshot, so per-item filtering
      is moot; snapshot-level gating is correct and sufficient. If the snapshot
      was produced AFTER the ``as_of`` instant (``generated_utc > as_of``), it
      was not yet available and the loader returns the same empty/no-triggers
      snapshot it returns for missing data. If ``generated_utc <= as_of`` the
      snapshot loads as today.
    - FAIL SAFE: when ``as_of`` is set but ``generated_utc`` is
      missing/unparseable (or ``as_of`` itself is unparseable), the snapshot
      cannot be proven point-in-time, so the loader returns empty rather than
      injecting an undatable (potential look-ahead) snapshot. This mirrors the
      market_stress PIT guard: observable degrade, never silent future-inject.
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

    # --- PIT snapshot-level gate (only when as_of is requested) ---
    if as_of is not None:
        as_of_dt = _to_utc_dt(as_of)
        gen_dt = _to_utc_dt(generated_utc or None)
        # Fail safe: cannot prove PIT -> do not inject an undatable snapshot.
        if as_of_dt is None or gen_dt is None:
            return empty
        if gen_dt > as_of_dt:
            return empty
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

    return DisclosuresTriggerSnapshot(
        generated_utc=generated_utc,
        triggers=list(items),
        summary={
            "max_severity": max_sev,
            "count_sev1plus": count_sev1,
            "count_sev2plus": count_sev2,
        },
    )
