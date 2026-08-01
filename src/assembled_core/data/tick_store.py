"""QuestDB tick store — OHLCV storage and query via PG-wire protocol.

QuestDB speaks PostgreSQL wire protocol on port 8812. This module uses
psycopg2 (or pg8000 as fallback) to:
  - Write OHLCV ticks via INSERT (batched).
  - Query OHLCV with QuestDB's SAMPLE BY time-series aggregation.

Graceful degradation: when QuestDB is unavailable (driver missing or
server down), all operations log a warning and return empty results.

TIME BOUNDS (E-063) — precise, because the earlier blanket claim "the
trading cycle is never blocked" was false:
  - CONNECT is bounded in both driver branches (driver-level timeout; the
    surrounding try/except alone bounds raises, not wall-clock time).
  - QUERY/WRITE on an established connection is NOT bounded under psycopg2:
    libpq has no read timeout, and the keepalives set here only catch a
    silently dead peer, not a live-but-hung server. Under pg8000 it is
    bounded ACCORDING TO THE DRIVER SOURCE (its socket timeout persists) —
    not measured here: no driver is installed or declared, so the contract
    tests skip everywhere including CI. Verify empirically before enabling.
    Either way this is an OPEN enablement precondition.
  - DNS resolution is bounded in neither branch (getaddrinfo runs before
    the timeout applies) — set QUESTDB_HOST to an IP literal when enabling
    against a remote host.
Authoritative precondition status: E-063 in docs/CLAUDE_CODING_ERRORS.md.
The ENABLEMENT PRECONDITIONS list at Step 7.70 in pipeline/_tc_execution.py
is out of date on item (1) — E-063 is the one to trust.

Environment variables:
    QUESTDB_HOST                (default: localhost)
    QUESTDB_PORT                (default: 8812)
    QUESTDB_USER                (default: admin)
    QUESTDB_PASS                (default: quest)
    QUESTDB_DB                  (default: qdb)
    QUESTDB_CONNECT_TIMEOUT_S   (default: 3.0; accepted range (0, 30] —
                                 anything else falls back to the default, so
                                 the unbounded connect cannot be restored)
"""

from __future__ import annotations

import inspect
import logging
import math
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


# E-063: a blocking connect is NOT limited by the try/except around it.
# try/except bounds raises, not wall-clock time — against a black-holed host
# (firewall DROP instead of REFUSE) the TCP connect hangs until the OS
# default (minutes). Since this module is reachable from a synchronous
# trading-cycle step, every connect MUST carry a driver-level timeout.
# Default deliberately small: QuestDB is an optional side-channel, never a
# reason to stall a cycle.
_DEFAULT_CONNECT_TIMEOUT_S = 3.0
# Upper bound matters as much as the lower one: "> 0" alone would let
# QUESTDB_CONNECT_TIMEOUT_S=600 re-create exactly the E-063 hang via a
# plausible operator typo. inf/nan pass a naive `<= 0` check too.
_MAX_CONNECT_TIMEOUT_S = 30.0

# _open_conn runs up to 3x per cycle (ping/ensure_table/write_ticks), so a
# static misconfiguration would otherwise emit the same WARN three times per
# cycle forever — the loudness class fixed in 7ccc59f8. warn-once instead.
_WARNED_ONCE: set[str] = set()


def _warn_once(key: str, level: int, msg: str, *args: Any) -> None:
    """Log ``msg`` only the first time for ``key`` (per process)."""
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    logger.log(level, msg, *args)


def _get_connect_timeout_s() -> float:
    """Connect timeout in seconds, clamped to (0, _MAX_CONNECT_TIMEOUT_S].

    Any unparseable / out-of-range value falls back to the default — this
    env var must never be a way back to an unbounded connect (E-063).
    """
    raw = os.environ.get("QUESTDB_CONNECT_TIMEOUT_S", "")
    if not raw:
        return _DEFAULT_CONNECT_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not math.isfinite(value) or value <= 0 or value > _MAX_CONNECT_TIMEOUT_S:
        _warn_once(
            f"bad_timeout:{raw}",
            logging.WARNING,
            "[TickStore] QUESTDB_CONNECT_TIMEOUT_S=%r outside (0, %.0f] or "
            "unparseable — using %.1fs (E-063: connect stays bounded)",
            raw,
            _MAX_CONNECT_TIMEOUT_S,
            _DEFAULT_CONNECT_TIMEOUT_S,
        )
        return _DEFAULT_CONNECT_TIMEOUT_S
    return value


def _get_conn_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "host": os.environ.get("QUESTDB_HOST", "localhost"),
        "port": int(os.environ.get("QUESTDB_PORT", "8812")),
        "user": os.environ.get("QUESTDB_USER", "admin"),
        "password": os.environ.get("QUESTDB_PASS", "quest"),
        "database": os.environ.get("QUESTDB_DB", "qdb"),
    }
    timeout_s = _get_connect_timeout_s()
    if _DRIVER == "psycopg2":
        # libpq: integer seconds; values < 2 are silently treated as 2.
        # max(2, ...) is load-bearing, NOT cosmetic: int(round(0.4)) == 0 and
        # libpq reads connect_timeout=0 as INFINITE — i.e. exactly the E-063
        # hang. Never let this floor be removed (pinned by a test).
        clamped = max(2, int(round(timeout_s)))
        if abs(clamped - timeout_s) > 0.01:
            _warn_once(
                f"clamp:{timeout_s}",
                logging.INFO,
                "[TickStore] psycopg2 needs integer seconds >= 2 — "
                "connect timeout %.2fs -> %ds",
                timeout_s,
                clamped,
            )
        kwargs["connect_timeout"] = clamped
        # Bounds ONE more hang class: a peer that disappears silently (no TCP
        # response at all). It does NOT bound a live-but-hung server — libpq
        # has no read timeout, so a stalled query still blocks (documented
        # enablement precondition in the module docstring).
        # Windows caveat: keepalives_count has no TCP_KEEPCNT equivalent and
        # is ignored there, so dead-peer detection on the pilot host is
        # ~idle + 10*interval, not idle + count*interval.
        kwargs["keepalives"] = 1
        kwargs["keepalives_idle"] = 10
        kwargs["keepalives_interval"] = 2
        kwargs["keepalives_count"] = 3
    elif _DRIVER == "pg8000":
        # pg8000 names the connect timeout `timeout` (float seconds) and
        # never resets the socket timeout afterwards, so it also bounds
        # reads/writes on the established connection.
        kwargs["timeout"] = timeout_s
        kwargs["tcp_keepalive"] = True
    return kwargs


_TIMEOUT_KEYS = ("connect_timeout", "timeout")


def _require_bounded(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Last structural gate: no timeout key -> do not connect at all.

    Deliberately checked here and not only inside the per-driver branches:
    adding a third driver (psycopg3 is the obvious candidate) to the import
    chain without extending ``_get_conn_kwargs`` would otherwise silently
    restore the unbounded connect of E-063. The invariant belongs to the
    function, not to the next author's discipline.
    """
    if not any(k in kwargs for k in _TIMEOUT_KEYS):
        logger.error(
            "[TickStore] driver %r has no bounded-connect mapping — QuestDB "
            "disabled (E-063: never connect unbounded)",
            _DRIVER,
        )
        return None
    return kwargs


def _validate_conn_kwargs(kwargs: dict[str, Any]) -> dict[str, Any] | None:
    """Verify the driver actually accepts our timeout kwargs.

    Rationale (E-063 + E-064): the kwarg names are driver-specific and were
    verified against psycopg2-binary 2.9.12 / pg8000 1.31.5 on 2026-08-01.
    If a future driver version renames or drops one, ``_connect_fn`` would
    raise, the broad ``except`` in ``_open_conn`` would report it as
    "cannot connect" — indistinguishable from "server is down" — and the
    store would be silently dead. So: validate offline, fail LOUD, and
    refuse to connect at all rather than connect without a time bound.

    Returns the (possibly reduced) kwargs, or None if no time-bounded call
    can be constructed.
    """
    if _DRIVER == "psycopg2":
        try:
            from psycopg2 import extensions as _pg_ext  # local: optional dep
        except ImportError:
            # _DRIVER says psycopg2 but the module is gone (only reachable
            # when a caller patches _DRIVER, e.g. in tests) — nothing to
            # validate against, but the timeout key is still mandatory.
            return _require_bounded(kwargs)
        try:
            _pg_ext.make_dsn(**kwargs)  # offline DSN validation, no I/O
            return _require_bounded(kwargs)
        except Exception as exc:
            # Keepalives are the optional part — connect_timeout is not.
            base = {k: v for k, v in kwargs.items() if not k.startswith("keepalives")}
            try:
                _pg_ext.make_dsn(**base)
                _warn_once(
                    "keepalives_rejected",
                    logging.WARNING,
                    "[TickStore] driver rejected the keepalive options (%s) — "
                    "continuing with connect_timeout only",
                    exc,
                )
                return _require_bounded(base)
            except Exception as exc2:
                logger.error(
                    "[TickStore] cannot build a time-bounded connection for "
                    "psycopg2 (%s) — QuestDB disabled (E-063: never connect "
                    "unbounded)",
                    exc2,
                )
                return None
    if _DRIVER == "pg8000":
        try:
            sig = inspect.signature(_connect_fn)
        except (TypeError, ValueError):
            return _require_bounded(kwargs)
        # A callable with **kwargs accepts names that are NOT in its
        # parameter list. Filtering against that list would then either
        # (a) claim `timeout` is unsupported when it is, or (b) silently
        # drop host/port/password — which would connect to the WRONG
        # database without a single log line. Name validation is simply
        # impossible in that case, so skip it.
        if any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            return _require_bounded(kwargs)
        params = set(sig.parameters)
        if not params:
            return _require_bounded(kwargs)
        if "timeout" not in params:
            logger.error(
                "[TickStore] pg8000 has no `timeout` parameter — QuestDB "
                "disabled (E-063: never connect unbounded)"
            )
            return None
        return _require_bounded({k: v for k, v in kwargs.items() if k in params})
    return _require_bounded(kwargs)


def _open_conn() -> Any:
    """Open a PG-wire connection to QuestDB. Returns None on failure."""
    if _connect_fn is None:
        logger.debug(
            "[TickStore] no PG driver available (psycopg2/pg8000 not installed)"
        )
        return None
    conn_kwargs = _validate_conn_kwargs(_get_conn_kwargs())
    if conn_kwargs is None:
        return None
    try:
        conn = _connect_fn(**conn_kwargs)
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
