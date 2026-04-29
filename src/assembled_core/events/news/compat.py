"""B3: Compatibility bridge between intel.models.NewsEvent and events.news.models.NewsEvent.

Two NewsEvent classes exist with diverging schemas:
- events.news.models.NewsEvent: dataclass, 16 fields, published_utc/fetched_utc as str
- intel.models.NewsEvent: Pydantic BaseModel, 30+ fields, published_at/ingested_at as datetime

This module provides conversion in both directions without data loss for the common fields.
Long-term goal: unify to a single schema (see audit B3).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.assembled_core.events.news.models import NewsEvent as EvNewsEvent
    from src.assembled_core.intel.models import NewsEvent as IntelNewsEvent


def events_to_intel(ev: "EvNewsEvent") -> "IntelNewsEvent":
    """Convert events.news.models.NewsEvent → intel.models.NewsEvent.

    Fields not present in the events schema default to intel schema defaults.
    """
    from datetime import datetime, timezone

    from src.assembled_core.intel.models import NewsEvent as IntelNewsEvent

    def _parse_utc(s: str | None) -> datetime:
        if not s:
            return datetime.now(timezone.utc)
        try:
            from dateutil.parser import parse as _parse
            dt = _parse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return datetime.now(timezone.utc)

    return IntelNewsEvent(
        event_id=ev.event_id,
        source_id=ev.source_id,
        source_tier="T3",  # default — not in events schema; T3 = lowest/unclassified
        title=ev.title,
        url=ev.url,
        published_at=_parse_utc(ev.published_utc),
        ingested_at=_parse_utc(ev.fetched_utc),
        geo_tags=list(ev.countries or []),
        entities=list(ev.entities or []),
        keywords=[],
        content_hash=ev.fingerprint or ev.fingerprint64 or "",
        language=ev.language or "en",
    )


def intel_to_events(intel: "IntelNewsEvent") -> "EvNewsEvent":
    """Convert intel.models.NewsEvent → events.news.models.NewsEvent.

    Fields not present in the intel schema default to empty strings/lists.
    """
    from src.assembled_core.events.news.models import NewsEvent as EvNewsEvent

    def _fmt(dt) -> str:
        if dt is None:
            return ""
        try:
            return dt.isoformat()
        except Exception:
            return str(dt)

    return EvNewsEvent(
        event_id=intel.event_id,
        source_id=intel.source_id,
        title=intel.title,
        url=intel.url,
        canonical_url=intel.url,  # intel has no canonical_url field
        source_name=intel.source_id,  # intel has no source_name field
        source_domain="",  # intel has no source_domain field
        published_utc=_fmt(intel.published_at),
        fetched_utc=_fmt(intel.ingested_at),
        language=intel.language,
        fingerprint=intel.content_hash,
        fingerprint64=intel.content_hash,
        entities=list(intel.entities or []),
        countries=list(intel.geo_tags or []),
    )


__all__ = ["events_to_intel", "intel_to_events"]
