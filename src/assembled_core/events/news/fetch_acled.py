"""ACLED Conflict Data Integration (Plan 4.6).

Maps ACLED event types to internal TriggerTypes for intel pipeline integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# ACLED event type → internal trigger type mapping
ACLED_TRIGGER_MAP: dict[str, str] = {
    "Battles": "MILITARY_BUILDUP",
    "Explosions/Remote violence": "MILITARY_BUILDUP",
    "Violence against civilians": "REGIME_CHANGE_RISK",
    "Protests": "REGIME_CHANGE_RISK",
    "Riots": "REGIME_CHANGE_RISK",
    "Strategic developments": "DIPLOMATIC_SHIFT",
}


@dataclass
class ACLEDEvent:
    """Parsed ACLED event."""
    event_date: str
    event_type: str
    country: str
    trigger_type: str
    fatalities: int = 0
    notes: str = ""


def parse_acled_events(raw_df: pd.DataFrame) -> list[ACLEDEvent]:
    """Parse raw ACLED data into structured events.

    Args:
        raw_df: DataFrame with columns: event_date, event_type, country, fatalities, notes.

    Returns:
        List of ACLEDEvent instances with mapped trigger types.
    """
    events = []
    for _, row in raw_df.iterrows():
        event_type = str(row.get("event_type", ""))
        trigger = ACLED_TRIGGER_MAP.get(event_type, "UNKNOWN_WILDCARD")
        events.append(ACLEDEvent(
            event_date=str(row.get("event_date", "")),
            event_type=event_type,
            country=str(row.get("country", "")),
            trigger_type=trigger,
            fatalities=int(row.get("fatalities", 0)),
            notes=str(row.get("notes", "")),
        ))
    return events


def aggregate_acled_by_country(events: list[ACLEDEvent]) -> dict[str, dict]:
    """Aggregate ACLED events by country.

    Returns:
        Dict of country → {n_events, total_fatalities, dominant_trigger}.
    """
    from collections import Counter, defaultdict

    country_data: dict[str, dict] = defaultdict(lambda: {"n_events": 0, "fatalities": 0, "triggers": []})
    for ev in events:
        country_data[ev.country]["n_events"] += 1
        country_data[ev.country]["fatalities"] += ev.fatalities
        country_data[ev.country]["triggers"].append(ev.trigger_type)

    result = {}
    for country, data in country_data.items():
        trigger_counts = Counter(data["triggers"])
        result[country] = {
            "n_events": data["n_events"],
            "total_fatalities": data["fatalities"],
            "dominant_trigger": trigger_counts.most_common(1)[0][0] if trigger_counts else "UNKNOWN",
        }
    return result


__all__ = ["ACLEDEvent", "ACLED_TRIGGER_MAP", "parse_acled_events", "aggregate_acled_by_country"]
