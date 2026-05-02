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
    events: list[ACLEDEvent] = []
    for _, row in df.iterrows():
        etype = str(row.get("event_type", "")).strip()
        trigger = _EVENT_TYPE_MAP.get(etype.lower(), "REGIME_CHANGE_RISK")
        events.append(
            ACLEDEvent(
                event_date=str(row.get("event_date", "")),
                event_type=etype,
                country=str(row.get("country", "")),
                fatalities=int(row.get("fatalities") or 0),
                notes=str(row.get("notes", "")),
                trigger_type=trigger,
            )
        )
    return events


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
