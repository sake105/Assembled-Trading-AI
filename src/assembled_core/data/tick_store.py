"""QuestDB tick store — OHLCV storage and query via PG-wire protocol.

QuestDB speaks PostgreSQL wire protocol on port 8812. This module uses
psycopg2 (or pg8000 as fallback) to:
  - Write OHLCV ticks via INSERT (batched).
  - Query OHLCV with QuestDB's SAMPLE BY time-series aggregation.

Graceful degradation: when QuestDB is unavailable (driver missing or
server down), all operations log a warning and return empty results.
The trading cycle is never blocked.

Environment variables:
    QUESTDB_HOST   (default: localhost)
    QUESTDB_PORT   (default: 8812)
    QUESTDB_USER   (default: admin)
    QUESTDB_PASS   (default: quest)
    QUESTDB_DB     (default: qdb)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Try psycopg2 first, then pg8000 as pure-Python fallback
_DRIVER: str | None = None
_connect_fn: Any = None

try:
    import psycopg2  # type: ignore[import-untyped]

    _DRIVER = "psycopg2"
    _connect_fn = psycopg2.connect
except ImportError:
    pass

if _DRIVER is None:
    try:
        import pg8000  # type: ignore[import-not-found]

        _DRIVER = "pg8000"
        _connect_fn = pg8000.connect
    except ImportError:
        pass


def _get_conn_kwargs() -> dict[str, Any]:
    return {
        "host": os.environ.get("QUESTDB_HOST", "localhost"),
        "port": int(os.environ.get("QUESTDB_PORT", "8812")),
        "user": os.environ.get("QUESTDB_USER", "admin"),
        "password": os.environ.get("QUESTDB_PASS", "quest"),
        "database": os.environ.get("QUESTDB_DB", "qdb"),
    }


def _open_conn() -> Any:
    """Open a PG-wire connection to QuestDB. Returns None on failure."""
    if _connect_fn is None:
        logger.debug(
            "[TickStore] no PG driver available (psycopg2/pg8000 not installed)"
        )
        return None
    try:
        conn = _connect_fn(**_get_conn_kwargs())
        return conn
    except Exception as exc:
        logger.warning("[TickStore] cannot connect to QuestDB: %s", exc)
        return None


@dataclass
class OHLCVTick:
    """Single OHLCV record."""

    symbol: str
    ts: datetime  # timezone-aware UTC timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    symbol   SYMBOL CAPACITY 2048 CACHE,
    ts       TIMESTAMP,
    open     DOUBLE,
    high     DOUBLE,
    low      DOUBLE,
    close    DOUBLE,
    volume   DOUBLE
) TIMESTAMP(ts) PARTITION BY DAY WAL;
"""


def ensure_table() -> bool:
    """Create the trades table if it does not exist. Returns True on success."""
    conn = _open_conn()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("[TickStore] ensure_table failed: %s", exc)
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_ticks(ticks: list[OHLCVTick]) -> int:
    """Insert OHLCV ticks into QuestDB.

    Args:
        ticks: List of OHLCVTick records.

    Returns:
        Number of rows inserted (0 on failure).
    """
    if not ticks:
        return 0

    conn = _open_conn()
    if conn is None:
        return 0

    sql = (
        "INSERT INTO trades (symbol, ts, open, high, low, close, volume) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)"
    )
    rows = [
        (t.symbol, t.ts.isoformat(), t.open, t.high, t.low, t.close, t.volume)
        for t in ticks
    ]

    try:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()
        logger.debug("[TickStore] inserted %d rows", len(rows))
        return len(rows)
    except Exception as exc:
        logger.warning("[TickStore] write_ticks failed: %s", exc)
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Read / Query
# ---------------------------------------------------------------------------


def query_ohlcv(
    symbol: str,
    start: datetime,
    end: datetime,
    sample_by: str = "1d",
) -> list[dict[str, Any]]:
    """Query OHLCV aggregated at the given time resolution.

    Uses QuestDB's SAMPLE BY syntax for native time-series aggregation.

    Args:
        symbol: Instrument symbol.
        start: Start timestamp (UTC).
        end: End timestamp (UTC).
        sample_by: QuestDB SAMPLE BY interval, e.g. "1d", "1h", "5m".

    Returns:
        List of dicts: {ts, open, high, low, close, volume}.
        Empty list on failure or when QuestDB is unavailable.
    """
    conn = _open_conn()
    if conn is None:
        return []

    sql = f"""
        SELECT
            ts,
            first(open) AS open,
            max(high)   AS high,
            min(low)    AS low,
            last(close) AS close,
            sum(volume) AS volume
        FROM trades
        WHERE symbol = %s
          AND ts >= %s
          AND ts <  %s
        SAMPLE BY {sample_by} ALIGN TO CALENDAR
        ORDER BY ts ASC
    """

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, start.isoformat(), end.isoformat()))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]
    except Exception as exc:
        logger.warning("[TickStore] query_ohlcv failed: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def query_latest(symbol: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return the most recent raw ticks for a symbol."""
    conn = _open_conn()
    if conn is None:
        return []

    sql = "SELECT * FROM trades WHERE symbol = %s ORDER BY ts DESC LIMIT %s"
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol, limit))
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]
    except Exception as exc:
        logger.warning("[TickStore] query_latest failed: %s", exc)
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

QUESTDB_DRIVER_AVAILABLE = _DRIVER is not None


def ping() -> bool:
    """Return True if QuestDB is reachable."""
    conn = _open_conn()
    if conn is None:
        return False
    try:
        conn.close()
        return True
    except Exception:
        return False
