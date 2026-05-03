"""ACLED (Armed Conflict Location & Event Data) fetcher and parser.

Parses ACLED CSV/DataFrame exports into structured conflict events
and aggregates them by country for geo-risk scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# ACLED event type → trigger_type mapping
_EVENT_TYPE_MAP: dict[str, str] = {
    "battles": "MILITARY_BUILDUP",
    "explosions/remote violence": "MILITARY_BUILDUP",
    "violence against civilians": "MILITARY_BUILDUP",
    "protests": "REGIME_CHANGE_RISK",
    "riots": "REGIME_CHANGE_RISK",
    "strategic developments": "SANCTIONS_ESCALATION",
}


@dataclass
class ACLEDEvent:
    event_date: str
    event_type: str
    country: str
    fatalities: int
    notes: str
    trigger_type: str


def parse_acled_events(df: pd.DataFrame) -> list[ACLEDEvent]:
    """Parse an ACLED DataFrame into a list of ACLEDEvent objects."""
    return [
        ACLEDEvent(
            event_date=str(getattr(row, "event_date", "") or ""),
            event_type=(etype := str(getattr(row, "event_type", "") or "").strip()),
            country=str(getattr(row, "country", "") or ""),
            fatalities=int(getattr(row, "fatalities", 0) or 0),
            notes=str(getattr(row, "notes", "") or ""),
            trigger_type=_EVENT_TYPE_MAP.get(etype.lower(), "REGIME_CHANGE_RISK"),
        )
        for row in df.itertuples(index=False)
    ]


def aggregate_acled_by_country(events: list[ACLEDEvent]) -> dict[str, dict[str, Any]]:
    """Aggregate ACLED events by country into summary dicts."""
    agg: dict[str, dict[str, Any]] = {}
    for evt in events:
        if evt.country not in agg:
            agg[evt.country] = {"total_fatalities": 0, "event_count": 0, "trigger_types": set()}
        agg[evt.country]["total_fatalities"] += evt.fatalities
        agg[evt.country]["event_count"] += 1
        agg[evt.country]["trigger_types"].add(evt.trigger_type)
    # Convert sets to lists for JSON-serializability
    for v in agg.values():
        v["trigger_types"] = sorted(v["trigger_types"])
    return agg
