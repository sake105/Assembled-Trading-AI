from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

SCHEMA_VERSION = "news.dedupe_store.v1"


def to_sqlite_i64(fp64_u: int) -> int:
    """Convert unsigned 64-bit fingerprint to signed 64-bit for SQLite INTEGER."""
    fp64_u &= (1 << 64) - 1
    return fp64_u - (1 << 64) if fp64_u >= (1 << 63) else fp64_u


def to_u64(x: int) -> int:
    """Convert stored value (signed or unsigned) back to unsigned 64-bit."""
    return x & ((1 << 64) - 1)


class DedupeStoreSQLite:
    """SQLite-backed dedupe store for NEWS events (canonical_url + fp64 bucket)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _ensure_schema(self, cur: sqlite3.Cursor) -> None:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);"
        )
        cur.execute(
            "SELECT value FROM meta WHERE key = ?;",
            ("schema_version",),
        )
        row = cur.fetchone()
        current = row[0] if row and row[0] else None
        if current != SCHEMA_VERSION:
            cur.execute("DROP INDEX IF EXISTS idx_seen_events_fp_bucket;")
            cur.execute("DROP TABLE IF EXISTS seen_events;")
            cur.execute(
                """
                CREATE TABLE seen_events(
                    canonical_url TEXT PRIMARY KEY,
                    fp64 INTEGER,
                    fp_bucket INTEGER,
                    event_id TEXT,
                    source_id TEXT,
                    published_utc TEXT,
                    ingested_utc TEXT
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX idx_seen_events_fp_bucket
                ON seen_events(fp_bucket);
                """
            )
            cur.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?);",
                ("schema_version", SCHEMA_VERSION),
            )

    def _init_db(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            self._ensure_schema(cur)
            conn.commit()

    @staticmethod
    def _bucket(fp64_u: int) -> int:
        """Bucket from unsigned 64-bit: top 8 bits."""
        return (to_u64(fp64_u) >> 56) & 0xFF

    def candidates_by_bucket(
        self, fp_bucket: int, limit: int = 200
    ) -> list[tuple[str, int]]:
        """Return list of (event_id, fp64_u) for a given bucket. fp64 as unsigned."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_id, fp64 FROM seen_events
                WHERE fp_bucket = ?
                ORDER BY ingested_utc DESC
                LIMIT ?;
                """,
                (int(fp_bucket), int(limit)),
            )
            rows = cur.fetchall()
            result: list[tuple[str, int]] = []
            for event_id, fp64 in rows:
                if event_id is None or fp64 is None:
                    continue
                result.append((str(event_id), to_u64(int(fp64))))
            return result

    def has_url(self, canonical_url: str) -> bool:
        if not canonical_url:
            return False
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT 1 FROM seen_events WHERE canonical_url = ? LIMIT 1;",
                (canonical_url,),
            )
            return cur.fetchone() is not None

    def has_fingerprint64(self, fp64: int) -> Tuple[bool, Optional[str]]:
        """Return True and event_id if an identical fp64 exists in the same bucket."""
        if fp64 == 0:
            return False, None
        fp64_u = to_u64(int(fp64))
        bucket = self._bucket(fp64_u)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT event_id, fp64 FROM seen_events
                WHERE fp_bucket = ?;
                """,
                (bucket,),
            )
            rows = cur.fetchall()
            for event_id, stored_fp in rows:
                if stored_fp is None:
                    continue
                if to_u64(int(stored_fp)) == fp64_u:
                    return True, str(event_id) if event_id is not None else None
        return False, None

    def add_event(
        self,
        event_id: str,
        canonical_url: str,
        fp64: int,
        published_utc: str,
        source_id: str,
        ingested_utc: str,
    ) -> None:
        if not canonical_url:
            return
        fp64_u = to_u64(int(fp64))
        fp64_sql = to_sqlite_i64(fp64_u)
        bucket = self._bucket(fp64_u)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO seen_events(
                    canonical_url,
                    fp64,
                    fp_bucket,
                    event_id,
                    source_id,
                    published_utc,
                    ingested_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    canonical_url,
                    fp64_sql,
                    int(bucket),
                    event_id,
                    source_id,
                    published_utc,
                    ingested_utc,
                ),
            )
            conn.commit()

    def vacuum(self) -> None:
        """Run VACUUM + PRAGMA optimize to reclaim space (call outside worker tick)."""
        with self._connect() as conn:
            conn.execute("PRAGMA optimize;")
        # VACUUM requires exclusive access — open a fresh connection outside WAL mode
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("VACUUM;")
        finally:
            conn.close()

    def prune(self, window_days: int, now_utc: str) -> int:
        """Prune entries older than window_days based on published_utc or ingested_utc.

        Returns number of deleted rows.
        """
        if window_days <= 0:
            return 0
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                DELETE FROM seen_events
                WHERE julianday(?) - julianday(COALESCE(published_utc, ingested_utc)) > ?;
                """,
                (now_utc, float(window_days)),
            )
            deleted = cur.rowcount if cur.rowcount is not None else 0
            conn.commit()
            return int(deleted)


__all__ = ["DedupeStoreSQLite", "to_sqlite_i64", "to_u64"]
