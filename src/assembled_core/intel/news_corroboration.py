"""Cross-source corroboration tracker.

Tracks how many independent sources report the same story. A story seen
across multiple T0/T1 outlets earns a corroboration boost; a story seen
only on one outlet is treated as un-corroborated.

Usage:
    tracker = CorroborationTracker()
    tracker.ingest(events)
    for evt in events:
        score = tracker.corroboration_score(evt)
        print(evt.title, "n_sources=", score.n_sources, "score=", score.score)

Design notes:
- Source-agnostic story fingerprint from `news_dedupe.content_fingerprint`
- Distinct source_ids are counted, not event_ids
- T0/T1 count double vs T2/T3 (more trustworthy corroboration)
- Score ∈ [0, 1], saturates at 4 distinct weighted sources
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.assembled_core.intel.news_dedupe import content_fingerprint

logger = logging.getLogger(__name__)


_TIER_WEIGHT: dict[str, float] = {
    "T0": 2.0, "T1": 1.5, "T2": 1.0, "T3": 0.5,
}


@dataclass
class CorroborationScore:
    story_key: str
    n_sources: int
    weighted_sources: float
    score: float  # [0, 1]
    sources: list[str]


class CorroborationTracker:
    """In-memory cross-source corroboration ledger.

    Args:
        retention_hours: how long a story contributes to counts (default 24h)
        saturation: weighted source count at which score saturates to 1.0
    """

    def __init__(self, retention_hours: float = 24.0, saturation: float = 4.0) -> None:
        self._retention = timedelta(hours=retention_hours)
        self._saturation = saturation
        # story_key → list of (ts, source_id, tier_str)
        self._entries: dict[str, list[tuple[datetime, str, str]]] = {}

    def ingest(self, events: list) -> None:
        """Register events with the tracker."""
        for evt in events:
            try:
                key = self._story_key(evt)
                src = (getattr(evt, "source_id", "") or "").lower().strip()
                if not src:
                    continue
                tier = getattr(evt, "source_tier", None)
                tier_str = getattr(tier, "value", str(tier)) if tier else "T2"
                ts = (
                    getattr(evt, "published_at", None)
                    or getattr(evt, "ingested_at", None)
                    or datetime.now(tz=timezone.utc)
                )
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                self._entries.setdefault(key, []).append((ts, src, tier_str))
            except Exception as exc:
                logger.debug("[SKIP] Corroboration ingest: %s", exc)

    def prune(self, now: datetime | None = None) -> int:
        """Drop entries older than retention. Returns dropped count."""
        if now is None:
            now = datetime.now(tz=timezone.utc)
        cutoff = now - self._retention
        dropped = 0
        for key in list(self._entries.keys()):
            kept = [(t, s, tier) for (t, s, tier) in self._entries[key] if t >= cutoff]
            dropped += len(self._entries[key]) - len(kept)
            if kept:
                self._entries[key] = kept
            else:
                del self._entries[key]
        return dropped

    def corroboration_score(self, event) -> CorroborationScore:
        """Return corroboration score for a single event.

        Score is based on how many distinct sources have reported the story.
        """
        key = self._story_key(event)
        entries = self._entries.get(key, [])
        seen: dict[str, str] = {}  # source_id → tier
        for _ts, src, tier in entries:
            # keep strongest tier seen for this source
            if src not in seen or _TIER_WEIGHT.get(tier, 1.0) > _TIER_WEIGHT.get(seen[src], 1.0):
                seen[src] = tier
        n_sources = len(seen)
        weighted = sum(_TIER_WEIGHT.get(t, 1.0) for t in seen.values())
        score = min(1.0, weighted / self._saturation)
        return CorroborationScore(
            story_key=key,
            n_sources=n_sources,
            weighted_sources=round(weighted, 2),
            score=round(score, 3),
            sources=sorted(seen.keys()),
        )

    def _story_key(self, event) -> str:
        title = getattr(event, "title", "") or ""
        return content_fingerprint(title, "")

    def size(self) -> int:
        return sum(len(v) for v in self._entries.values())

    def unique_stories(self) -> int:
        return len(self._entries)


__all__ = ["CorroborationTracker", "CorroborationScore"]
