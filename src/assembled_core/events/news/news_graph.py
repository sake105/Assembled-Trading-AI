"""Neo4j-backed news relationship graph (optional dependency).

Stores news events and named entities as a property graph so that
cross-article entity co-occurrence and causal chains can be queried
efficiently.

Graph schema
------------
Nodes:
  (:NewsEvent  {event_id, headline, source, published_at, sentiment, ticker})
  (:Entity     {name, entity_type})   -- "company", "person", "country", etc.

Edges:
  (:NewsEvent)-[:MENTIONS]->(: Entity)
  (:NewsEvent)-[:RELATED_TO {weight}]->(:NewsEvent)

Fallback
--------
When the ``neo4j`` Python package is not installed (or no bolt URI is
configured) the graph falls back to an in-memory adjacency dict so that
offline runs, tests, and CI work without a running database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase as _GDB

    _NEO4J_AVAILABLE = True
except ImportError:
    _GDB = None  # type: ignore[assignment]
    _NEO4J_AVAILABLE = False


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class NewsNode:
    event_id: str
    headline: str
    source: str
    published_at: datetime
    sentiment: float = 0.0  # [-1, 1] — negative to positive
    ticker: str = ""  # primary equity ticker if known
    entities: list[str] = field(default_factory=list)


@dataclass
class GraphStats:
    n_events: int
    n_entities: int
    n_mentions: int
    n_related: int
    backend: str  # "neo4j" | "memory"


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------


class _MemoryGraph:
    """Minimal in-process adjacency graph used when Neo4j is unavailable."""

    def __init__(self) -> None:
        self._events: dict[str, NewsNode] = {}
        self._entities: dict[str, dict[str, Any]] = {}
        # event_id -> {entity_name: True}
        self._mentions: dict[str, set[str]] = {}
        # event_id -> {other_event_id: weight}
        self._related: dict[str, dict[str, float]] = {}

    def add_event(self, node: NewsNode) -> None:
        self._events[node.event_id] = node
        self._mentions.setdefault(node.event_id, set())
        for ent in node.entities:
            self._entities.setdefault(ent, {"name": ent})
            self._mentions[node.event_id].add(ent)

    def add_entity(self, name: str, entity_type: str = "unknown") -> None:
        self._entities.setdefault(name, {"name": name, "entity_type": entity_type})

    def add_related(
        self, event_id_a: str, event_id_b: str, weight: float = 1.0
    ) -> None:
        self._related.setdefault(event_id_a, {})[event_id_b] = weight
        self._related.setdefault(event_id_b, {})[event_id_a] = weight

    def entity_neighbors(self, entity_name: str) -> list[str]:
        """Return event IDs that mention *entity_name*."""
        return [eid for eid, ents in self._mentions.items() if entity_name in ents]

    def related_symbols(self, ticker: str, max_hops: int = 2) -> list[str]:
        """Return tickers co-mentioned with *ticker* within *max_hops*."""
        seed_ids = [eid for eid, node in self._events.items() if node.ticker == ticker]
        visited_ids: set[str] = set(seed_ids)
        for _ in range(max_hops - 1):
            next_ids: set[str] = set()
            for eid in list(visited_ids):
                next_ids.update(self._related.get(eid, {}).keys())
            visited_ids |= next_ids

        related_tickers = {
            self._events[eid].ticker
            for eid in visited_ids
            if eid in self._events and self._events[eid].ticker not in ("", ticker)
        }
        return sorted(related_tickers)

    def stats(self) -> GraphStats:
        n_mentions = sum(len(s) for s in self._mentions.values())
        n_related = sum(len(d) for d in self._related.values()) // 2
        return GraphStats(
            n_events=len(self._events),
            n_entities=len(self._entities),
            n_mentions=n_mentions,
            n_related=n_related,
            backend="memory",
        )


# ---------------------------------------------------------------------------
# NewsGraph — unified interface
# ---------------------------------------------------------------------------


class NewsGraph:
    """Property graph for news events and entity relationships.

    Transparently routes to Neo4j when available, or falls back to the
    in-memory implementation.  Callers should never depend on the backend.

    Parameters
    ----------
    bolt_uri:
        Neo4j bolt connection string, e.g. ``"bolt://localhost:7687"``.
        When ``None`` or when neo4j is not installed, falls back to memory.
    auth:
        ``(user, password)`` tuple for Neo4j authentication.
    """

    def __init__(
        self,
        bolt_uri: str | None = None,
        auth: tuple[str, str] = ("neo4j", "password"),
    ) -> None:
        self._driver = None
        self._mem: _MemoryGraph | None = None
        self._backend = "memory"

        if bolt_uri and _NEO4J_AVAILABLE:
            try:
                self._driver = _GDB.driver(bolt_uri, auth=auth)
                self._driver.verify_connectivity()
                self._backend = "neo4j"
                self._ensure_indexes()
                log.info("[NewsGraph] Connected to Neo4j at %s", bolt_uri)
            except Exception as exc:
                log.warning(
                    "[NewsGraph] Neo4j connection failed (%s) — using memory fallback",
                    exc,
                )
                self._driver = None

        if self._driver is None:
            self._mem = _MemoryGraph()
            if bolt_uri:
                log.info("[NewsGraph] Using in-memory fallback (neo4j not reachable)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_event(self, node: NewsNode) -> None:
        """Persist a news event and create MENTIONS edges for each entity."""
        if self._driver:
            self._neo4j_add_event(node)
        else:
            if self._mem is None:
                raise RuntimeError(
                    "NewsGraph: no backend — both _driver and _mem are None"
                )
            self._mem.add_event(node)

    def add_entity(self, name: str, entity_type: str = "unknown") -> None:
        """Upsert a named entity node."""
        if self._driver:
            with self._driver.session() as session:
                session.run(
                    "MERGE (e:Entity {name: $name}) SET e.entity_type = $etype",
                    name=name,
                    etype=entity_type,
                )
        else:
            if self._mem is None:
                raise RuntimeError(
                    "NewsGraph: no backend — both _driver and _mem are None"
                )
            self._mem.add_entity(name, entity_type)

    def add_related(
        self, event_id_a: str, event_id_b: str, weight: float = 1.0
    ) -> None:
        """Create a RELATED_TO edge between two events."""
        if self._driver:
            with self._driver.session() as session:
                session.run(
                    """
                    MATCH (a:NewsEvent {event_id: $id_a})
                    MATCH (b:NewsEvent {event_id: $id_b})
                    MERGE (a)-[r:RELATED_TO]->(b)
                    SET r.weight = $w
                    """,
                    id_a=event_id_a,
                    id_b=event_id_b,
                    w=weight,
                )
        else:
            if self._mem is None:
                raise RuntimeError(
                    "NewsGraph: no backend — both _driver and _mem are None"
                )
            self._mem.add_related(event_id_a, event_id_b, weight)

    def entity_neighbors(self, entity_name: str) -> list[str]:
        """Return IDs of events that mention *entity_name*."""
        if self._driver:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (ev:NewsEvent)-[:MENTIONS]->(e:Entity {name: $name})
                    RETURN ev.event_id AS eid
                    """,
                    name=entity_name,
                )
                return [r["eid"] for r in result]
        if self._mem is None:
            raise RuntimeError("NewsGraph: no backend — both _driver and _mem are None")
        return self._mem.entity_neighbors(entity_name)

    def find_related_symbols(self, ticker: str, max_hops: int = 2) -> list[str]:
        """Return tickers co-mentioned with *ticker* within *max_hops* in graph."""
        if self._driver:
            with self._driver.session() as session:
                cypher = """
                    MATCH (start:NewsEvent {ticker: $ticker})
                    CALL apoc.path.subgraphNodes(start, {
                        relationshipFilter: 'RELATED_TO',
                        maxLevel: $hops
                    }) YIELD node
                    WHERE node:NewsEvent AND node.ticker <> $ticker AND node.ticker <> ''
                    RETURN DISTINCT node.ticker AS ticker
                    ORDER BY ticker
                """
                try:
                    result = session.run(cypher, ticker=ticker, hops=max_hops)
                    return [r["ticker"] for r in result]
                except Exception:
                    # APOC may not be installed — fall back to simple 1-hop
                    result = session.run(
                        """
                        MATCH (a:NewsEvent {ticker: $ticker})-[:RELATED_TO]-(b:NewsEvent)
                        WHERE b.ticker <> $ticker AND b.ticker <> ''
                        RETURN DISTINCT b.ticker AS ticker ORDER BY ticker
                        """,
                        ticker=ticker,
                    )
                    return [r["ticker"] for r in result]

        if self._mem is None:
            raise RuntimeError("NewsGraph: no backend — both _driver and _mem are None")
        return self._mem.related_symbols(ticker, max_hops)

    def stats(self) -> GraphStats:
        """Return graph statistics."""
        if self._driver:
            with self._driver.session() as session:
                ev = session.run("MATCH (n:NewsEvent) RETURN count(n) AS c").single()[
                    "c"
                ]
                ent = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
                men = session.run(
                    "MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c"
                ).single()["c"]
                rel = session.run(
                    "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS c"
                ).single()["c"]
            return GraphStats(
                n_events=ev,
                n_entities=ent,
                n_mentions=men,
                n_related=rel,
                backend="neo4j",
            )
        if self._mem is None:
            raise RuntimeError("NewsGraph: no backend — both _driver and _mem are None")
        return self._mem.stats()

    def close(self) -> None:
        if self._driver:
            self._driver.close()

    def __enter__(self) -> "NewsGraph":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        with self._driver.session() as session:  # type: ignore[union-attr]
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:NewsEvent) ON (n.event_id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity)    ON (n.name)")

    def _neo4j_add_event(self, node: NewsNode) -> None:
        with self._driver.session() as session:  # type: ignore[union-attr]
            session.run(
                """
                MERGE (ev:NewsEvent {event_id: $eid})
                SET ev.headline    = $headline,
                    ev.source      = $source,
                    ev.published_at= $pub,
                    ev.sentiment   = $sentiment,
                    ev.ticker      = $ticker
                """,
                eid=node.event_id,
                headline=node.headline,
                source=node.source,
                pub=node.published_at.isoformat(),
                sentiment=node.sentiment,
                ticker=node.ticker,
            )
            for ent in node.entities:
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    WITH e
                    MATCH (ev:NewsEvent {event_id: $eid})
                    MERGE (ev)-[:MENTIONS]->(e)
                    """,
                    name=ent,
                    eid=node.event_id,
                )


__all__ = [
    "NewsGraph",
    "NewsNode",
    "GraphStats",
    "NEO4J_AVAILABLE",
]

NEO4J_AVAILABLE: bool = _NEO4J_AVAILABLE
