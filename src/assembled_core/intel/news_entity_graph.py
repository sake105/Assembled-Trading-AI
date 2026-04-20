"""Entity co-occurrence graph (lightweight, no NetworkX dependency).

Tracks how often pairs of entities (people, organisations, tickers) appear
together in the same NewsEvent over a sliding window. Useful for spotting
emerging clusters (e.g. "OpenAI"+"Nvidia"+"Microsoft" linked to a deal),
or for de-noising single-entity events that lack confirmation.

Design notes:

* Pure stdlib — uses dicts of dicts as the graph. Avoids the NetworkX
  install cost and keeps CI lean.
* Sliding window pruning by `retention_hours`.
* Each entity gets a degree (number of distinct neighbours) and a strength
  (sum of co-occurrence counts).

Usage:
    g = EntityCoGraph(retention_hours=24)
    g.ingest(events)
    top = g.top_entities(n=10)
    nbrs = g.neighbours("openai", min_weight=2)
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class _Edge:
    weight: int
    last_seen: datetime


@dataclass
class EntityStat:
    entity: str
    degree: int       # number of distinct neighbours
    strength: int     # sum of edge weights
    n_events: int     # number of events the entity appeared in


class EntityCoGraph:
    """Per-entity neighbour graph with sliding-window decay."""

    def __init__(
        self,
        retention_hours: float = 24.0,
        max_entities_per_event: int = 12,
    ) -> None:
        self._retention = timedelta(hours=retention_hours)
        self._max_per_event = max_entities_per_event
        # adjacency: entity -> {neighbour: _Edge}
        self._adj: dict[str, dict[str, _Edge]] = defaultdict(dict)
        # event-by-event ledger for pruning
        # deque[(ts, [entities])]
        self._ledger: deque[tuple[datetime, list[str]]] = deque()
        # event count per entity
        self._counts: dict[str, int] = defaultdict(int)

    # ----- ingestion --------------------------------------------------

    def ingest(self, events: list, now: datetime | None = None) -> None:
        if now is None:
            now = datetime.now(tz=timezone.utc)
        for evt in events or []:
            try:
                ents = self._collect_entities(evt)
                if len(ents) < 2:
                    if ents:
                        for e in ents:
                            self._counts[e] += 1
                    continue
                ts = getattr(evt, "published_at", None) or getattr(evt, "ingested_at", None) or now
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                ents = ents[: self._max_per_event]
                for e in ents:
                    self._counts[e] += 1
                self._ledger.append((ts, list(ents)))
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        a, b = ents[i], ents[j]
                        self._bump(a, b, ts)
                        self._bump(b, a, ts)
            except Exception as exc:
                logger.debug("[SKIP] EntityCoGraph.ingest: %s", exc)
        self.prune(now=now)

    def _collect_entities(self, evt) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for src_attr in ("entities", "tickers", "affected_assets"):
            for raw in getattr(evt, src_attr, []) or []:
                key = str(raw).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def _bump(self, a: str, b: str, ts: datetime) -> None:
        edges = self._adj[a]
        edge = edges.get(b)
        if edge is None:
            edges[b] = _Edge(weight=1, last_seen=ts)
        else:
            edge.weight += 1
            if ts > edge.last_seen:
                edge.last_seen = ts

    # ----- pruning ----------------------------------------------------

    def prune(self, now: datetime | None = None) -> int:
        if now is None:
            now = datetime.now(tz=timezone.utc)
        cutoff = now - self._retention
        dropped = 0
        # decrement counters for stale events
        while self._ledger and self._ledger[0][0] < cutoff:
            _, ents = self._ledger.popleft()
            for e in ents:
                if self._counts.get(e, 0) > 0:
                    self._counts[e] -= 1
                    if self._counts[e] == 0:
                        self._counts.pop(e, None)
            dropped += 1
        # drop stale edges
        for a, edges in list(self._adj.items()):
            for b, edge in list(edges.items()):
                if edge.last_seen < cutoff:
                    del edges[b]
            if not edges:
                del self._adj[a]
        return dropped

    # ----- queries ----------------------------------------------------

    def neighbours(
        self,
        entity: str,
        *,
        min_weight: int = 1,
        limit: int | None = None,
    ) -> list[tuple[str, int]]:
        key = (entity or "").strip().lower()
        edges = self._adj.get(key, {})
        out = [(b, e.weight) for b, e in edges.items() if e.weight >= min_weight]
        out.sort(key=lambda x: -x[1])
        if limit is not None:
            out = out[:limit]
        return out

    def top_entities(self, n: int = 10) -> list[EntityStat]:
        stats: list[EntityStat] = []
        for ent, edges in self._adj.items():
            strength = sum(e.weight for e in edges.values())
            stats.append(EntityStat(
                entity=ent,
                degree=len(edges),
                strength=strength,
                n_events=self._counts.get(ent, 0),
            ))
        stats.sort(key=lambda s: (-s.strength, -s.degree))
        return stats[:n]

    def has_edge(self, a: str, b: str) -> bool:
        a, b = a.lower(), b.lower()
        return b in self._adj.get(a, {})

    def edge_weight(self, a: str, b: str) -> int:
        a, b = a.lower(), b.lower()
        e = self._adj.get(a, {}).get(b)
        return e.weight if e else 0

    @property
    def size(self) -> int:
        return len(self._adj)


__all__ = ["EntityCoGraph", "EntityStat"]
