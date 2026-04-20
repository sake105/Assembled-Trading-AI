"""In-memory queryable NewsEvent store for the intel pipeline.

Provides fast lookups by ticker, sector, event_type, geo_tag, and time range.
Designed to be the live query layer on top of the archiver's JSONL persistence.

Usage:
    store = NewsEventStore(max_events=5000)
    store.add_many(events)

    # Query
    energy_events = store.query_by_sector("energy", hours=6)
    aapl_events = store.query_by_ticker("AAPL", hours=24)
    recent = store.query_by_time(hours=1)
    high_sev = store.query_by_severity(min_severity=6.0, hours=12)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Callable

logger = logging.getLogger(__name__)


class NewsEventStore:
    """In-memory store with indexed lookups for NewsEvent objects.

    Maintains insertion-order list + inverted indices for:
    - tickers (affected_assets and tickers fields)
    - sectors (affected_sectors)
    - event_types
    - geo_tags
    - source_id

    Eviction: when max_events is exceeded, the oldest half is dropped.
    """

    def __init__(self, max_events: int = 5_000) -> None:
        self._max_events = max_events
        self._events: list = []  # ordered by ingestion time, NewsEvent objects
        # Inverted indices: value → set of event indices
        self._idx_ticker: dict[str, set[int]] = defaultdict(set)
        self._idx_sector: dict[str, set[int]] = defaultdict(set)
        self._idx_event_type: dict[str, set[int]] = defaultdict(set)
        self._idx_geo: dict[str, set[int]] = defaultdict(set)
        self._idx_source: dict[str, set[int]] = defaultdict(set)

    def add(self, event: object) -> None:
        """Add a single NewsEvent to the store."""
        idx = len(self._events)
        self._events.append(event)

        # Index tickers
        for ticker in list(getattr(event, "tickers", []) or []) + list(getattr(event, "affected_assets", []) or []):
            self._idx_ticker[ticker.upper()].add(idx)

        # Index sectors
        for sector in getattr(event, "affected_sectors", []) or []:
            self._idx_sector[sector.lower()].add(idx)

        # Index event types
        for etype in getattr(event, "event_types", []) or []:
            self._idx_event_type[etype.lower()].add(idx)

        # Index geo tags
        for geo in getattr(event, "geo_tags", []) or []:
            self._idx_geo[geo.upper()].add(idx)

        # Index source
        source = getattr(event, "source_id", None)
        if source:
            self._idx_source[source.lower()].add(idx)

        # Evict if over capacity
        if len(self._events) > self._max_events:
            self._evict()

    def add_many(self, events: list) -> int:
        """Add multiple events. Returns count added."""
        for evt in events:
            self.add(evt)
        return len(events)

    def query_by_ticker(self, ticker: str, hours: float | None = None) -> list:
        """Return events mentioning the given ticker within the last N hours."""
        indices = self._idx_ticker.get(ticker.upper(), set())
        return self._filter_by_time([self._events[i] for i in sorted(indices)], hours)

    def query_by_sector(self, sector: str, hours: float | None = None) -> list:
        """Return events affecting the given sector within the last N hours."""
        indices = self._idx_sector.get(sector.lower(), set())
        return self._filter_by_time([self._events[i] for i in sorted(indices)], hours)

    def query_by_event_type(self, event_type: str, hours: float | None = None) -> list:
        """Return events of the given type within the last N hours."""
        indices = self._idx_event_type.get(event_type.lower(), set())
        return self._filter_by_time([self._events[i] for i in sorted(indices)], hours)

    def query_by_geo(self, iso2: str, hours: float | None = None) -> list:
        """Return events tagged with the given ISO-2 country code."""
        indices = self._idx_geo.get(iso2.upper(), set())
        return self._filter_by_time([self._events[i] for i in sorted(indices)], hours)

    def query_by_source(self, source_id: str, hours: float | None = None) -> list:
        """Return events from the given source."""
        indices = self._idx_source.get(source_id.lower(), set())
        return self._filter_by_time([self._events[i] for i in sorted(indices)], hours)

    def query_by_time(self, hours: float) -> list:
        """Return all events from the last N hours."""
        return self._filter_by_time(self._events, hours)

    def query_by_severity(self, min_severity: float, hours: float | None = None) -> list:
        """Return events at or above the given severity."""
        candidates = self._filter_by_time(self._events, hours)
        return [e for e in candidates if float(getattr(e, "severity", 0.0)) >= min_severity]

    def query_by_confidence(self, min_confidence: float, hours: float | None = None) -> list:
        """Return events at or above the given news_confidence."""
        candidates = self._filter_by_time(self._events, hours)
        return [e for e in candidates if float(getattr(e, "news_confidence", 0.0)) >= min_confidence]

    def query(self, predicate: Callable[[object], bool]) -> list:
        """Return events matching an arbitrary predicate function."""
        return [e for e in self._events if predicate(e)]

    def count(self) -> int:
        return len(self._events)

    def top_sectors(self, hours: float = 24.0, n: int = 5) -> list[tuple[str, int]]:
        """Return top N sectors by event count in the last N hours."""
        recent = self._filter_by_time(self._events, hours)
        counts: dict[str, int] = {}
        for evt in recent:
            for sector in getattr(evt, "affected_sectors", []) or []:
                counts[sector] = counts.get(sector, 0) + 1
        return sorted(counts.items(), key=lambda x: -x[1])[:n]

    def top_tickers(self, hours: float = 24.0, n: int = 10) -> list[tuple[str, int]]:
        """Return top N tickers by event count in the last N hours."""
        recent = self._filter_by_time(self._events, hours)
        counts: dict[str, int] = {}
        for evt in recent:
            for ticker in list(getattr(evt, "tickers", []) or []) + list(getattr(evt, "affected_assets", []) or []):
                counts[ticker] = counts.get(ticker, 0) + 1
        return sorted(counts.items(), key=lambda x: -x[1])[:n]

    def avg_severity(self, hours: float = 24.0) -> float:
        """Return average severity of events in the last N hours."""
        recent = self._filter_by_time(self._events, hours)
        if not recent:
            return 0.0
        return round(sum(float(getattr(e, "severity", 0.0)) for e in recent) / len(recent), 3)

    def clear(self) -> None:
        """Remove all stored events and reset indices."""
        self._events.clear()
        self._idx_ticker.clear()
        self._idx_sector.clear()
        self._idx_event_type.clear()
        self._idx_geo.clear()
        self._idx_source.clear()

    # ------------------------------------------------------------------
    # H6: archive-based warm-start
    # ------------------------------------------------------------------

    def restore_from_archive(
        self,
        path,
        *,
        hours: float | None = 24.0,
        max_events: int | None = None,
    ) -> int:
        """Rebuild the store from a NewsArchive JSONL file.

        Only events newer than `hours` (relative to now) are restored. If
        `max_events` is given, restoration stops at that count. Returns the
        number of events added.

        Designed for worker restarts: fresh process -> warm store in one call.
        """
        from datetime import datetime, timedelta, timezone
        from src.assembled_core.intel.news_archive import NewsArchiveReader

        reader = NewsArchiveReader(path)
        if not reader:
            return 0
        cutoff = None
        if hours is not None:
            cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        added = 0
        cap = max_events if max_events is not None else self._max_events
        for evt in reader.iter_events(since=cutoff):
            if added >= cap:
                break
            self.add(evt)
            added += 1
        logger.info("[OK] NewsEventStore.restore_from_archive: loaded %d events", added)
        return added

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter_by_time(self, events: list, hours: float | None) -> list:
        if hours is None:
            return list(events)
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=hours)
        result = []
        for evt in events:
            ts = getattr(evt, "published_at", None) or getattr(evt, "ingested_at", None)
            if ts is None:
                result.append(evt)
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                result.append(evt)
        return result

    def _evict(self) -> None:
        """Drop the oldest half of events and rebuild indices."""
        keep_from = len(self._events) // 2
        self._events = self._events[keep_from:]
        # Rebuild indices from scratch
        self._idx_ticker = defaultdict(set)
        self._idx_sector = defaultdict(set)
        self._idx_event_type = defaultdict(set)
        self._idx_geo = defaultdict(set)
        self._idx_source = defaultdict(set)
        for new_idx, evt in enumerate(self._events):
            for ticker in list(getattr(evt, "tickers", []) or []) + list(getattr(evt, "affected_assets", []) or []):
                self._idx_ticker[ticker.upper()].add(new_idx)
            for sector in getattr(evt, "affected_sectors", []) or []:
                self._idx_sector[sector.lower()].add(new_idx)
            for etype in getattr(evt, "event_types", []) or []:
                self._idx_event_type[etype.lower()].add(new_idx)
            for geo in getattr(evt, "geo_tags", []) or []:
                self._idx_geo[geo.upper()].add(new_idx)
            source = getattr(evt, "source_id", None)
            if source:
                self._idx_source[source.lower()].add(new_idx)
        logger.debug("[OK] NewsEventStore: evicted, kept %d events", len(self._events))


__all__ = ["NewsEventStore"]
