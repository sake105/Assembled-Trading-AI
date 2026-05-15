"""Append-only SQLite event store for the Event-Replay-System.

From 42_EVENT_REPLAY_SYSTEM.md.

Events are appended once and never mutated.  Each session gets a UUID.
The store supports:
  - append / append_batch
  - load_session (returns events in sequence order)
  - session_stats

SQLite is used for simplicity and portability.  For production scale
(>1M events/day) swap for PostgreSQL or a Redis stream.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from src.assembled_core.events.schema import BaseEvent

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path("data/events/events.db")


class EventAppendError(RuntimeError):
    """Raised when an event cannot be persisted (DB error, schema mismatch, etc.).

    Distinct from duplicate-sequence INSERT OR IGNORE outcomes which are
    expected (legitimate retry) and surface via cursor.rowcount == 0 instead.
    """


class EventStore:
    """Append-only SQLite store for BaseEvent objects."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._path = Path(db_path) if db_path else _DEFAULT_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        con = sqlite3.connect(str(self._path))
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init_schema(self) -> None:
        with self._conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    sequence    INTEGER NOT NULL,
                    event_type  TEXT    NOT NULL,
                    source      TEXT    NOT NULL,
                    occurred_at TEXT    NOT NULL,
                    payload_json TEXT   NOT NULL,
                    UNIQUE (session_id, sequence)
                )
            """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)"
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(self, event: BaseEvent) -> None:
        """Append a single event.  Duplicate (session, sequence) is silently ignored."""
        with self._conn() as con:
            try:
                con.execute(
                    """INSERT OR IGNORE INTO events
                       (session_id, sequence, event_type, source, occurred_at, payload_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        event.session_id,
                        event.sequence,
                        event.event_type,
                        event.source.value,
                        event.occurred_at.isoformat(),
                        json.dumps(event.payload),
                    ),
                )
            except sqlite3.Error as exc:
                # F-B-12 MAJOR fix: re-raise instead of silent drop. The class
                # contract promises "append-only, events never mutated" — a
                # silent drop violates that. Duplicate-sequence is handled by
                # INSERT OR IGNORE (no exception); only true DB errors reach here.
                logger.error("EventStore.append failed: %s", exc)
                raise EventAppendError(str(exc)) from exc

    def append_batch(self, events: list[BaseEvent]) -> int:
        """Append a batch of events.  Returns number of rows inserted."""
        rows = [
            (
                e.session_id,
                e.sequence,
                e.event_type,
                e.source.value,
                e.occurred_at.isoformat(),
                json.dumps(e.payload, default=str),
            )
            for e in events
        ]
        with self._conn() as con:
            cur = con.executemany(
                """INSERT OR IGNORE INTO events
                   (session_id, sequence, event_type, source, occurred_at, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return cur.rowcount

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_session(self, session_id: str) -> list[dict]:
        """Load all events for a session in sequence order."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM events WHERE session_id = ? ORDER BY sequence",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_sessions(self) -> list[str]:
        """Return all unique session IDs."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT DISTINCT session_id FROM events ORDER BY MIN(occurred_at)"
            ).fetchall()
        return [r["session_id"] for r in rows]

    def session_stats(self, session_id: str) -> dict:
        """Return event count, first/last timestamp for a session."""
        with self._conn() as con:
            row = con.execute(
                """SELECT COUNT(*) as n,
                          MIN(occurred_at) as first_at,
                          MAX(occurred_at) as last_at
                   FROM events WHERE session_id = ?""",
                (session_id,),
            ).fetchone()
        if row is None:
            return {"session_id": session_id, "n_events": 0}
        return {
            "session_id": session_id,
            "n_events": row["n"],
            "first_at": row["first_at"],
            "last_at": row["last_at"],
        }


__all__ = ["EventStore", "EventAppendError"]
