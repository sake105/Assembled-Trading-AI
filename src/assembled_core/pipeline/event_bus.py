"""Pipeline event bus — Redis Streams side-channel (Phase 1).

Publishes structured events to a Redis Stream for observability and
downstream consumers (dashboards, alerting, audit logs). The trading
cycle is never blocked by this: all publishes are wrapped in try/except
and the bus degrades gracefully to a no-op when Redis is unavailable.

Design principles:
  - Zero coupling: importing this module never raises even if redis-py is absent.
  - Fire-and-forget: publish() returns True on success, False on any failure.
  - No business logic: this is a side-channel only.
  - Stream key per event type for easy consumer-group segmentation.

Usage::

    from assembled_core.pipeline.event_bus import EventBus, streamed_phase

    bus = EventBus()   # auto-connects if REDIS_URL is set

    # Manual publish
    bus.publish("trade_signal", {"symbol": "AAPL", "action": "BUY"})

    # Context manager — wraps a pipeline phase and publishes start/end events
    with streamed_phase(bus, "eod_pipeline", {"date": "2024-01-15"}):
        run_eod_pipeline()
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import Any, Generator

logger = logging.getLogger(__name__)

_REDIS_AVAILABLE = False
try:
    import redis  # type: ignore[import-not-found]

    _REDIS_AVAILABLE = True
except ImportError:
    pass


class EventBus:
    """Redis Streams event publisher with graceful degradation.

    Args:
        redis_url: Redis connection URL (default: REDIS_URL env var or
                   redis://localhost:6379/0).
        stream_prefix: Prefix for all stream keys (default "assembled").
        maxlen: Maximum stream length per key (XADD MAXLEN ~, approximate trim).
        connect_timeout: Connection timeout in seconds.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        stream_prefix: str = "assembled",
        maxlen: int = 10_000,
        connect_timeout: float = 2.0,
    ) -> None:
        self._prefix = stream_prefix
        self._maxlen = maxlen
        self._client: Any = None
        self._available = False

        if not _REDIS_AVAILABLE:
            logger.debug("[EventBus] redis-py not installed — running in no-op mode")
            return

        url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            self._client = redis.from_url(
                url,
                socket_connect_timeout=connect_timeout,
                decode_responses=True,
            )
            # Verify connection
            self._client.ping()
            self._available = True
            logger.info("[EventBus] connected to Redis at %s", url)
        except Exception as exc:
            logger.warning("[EventBus] Redis unavailable (%s) — no-op mode", exc)

    @property
    def available(self) -> bool:
        return self._available

    def publish(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        stream_key: str | None = None,
    ) -> bool:
        """Publish an event to a Redis Stream.

        Args:
            event_type: Logical event name (e.g. "trade_signal", "risk_check").
            payload: Arbitrary key-value pairs (values will be str-coerced).
            stream_key: Override stream key; defaults to "<prefix>:<event_type>".

        Returns:
            True if published successfully, False on any error.
        """
        if not self._available or self._client is None:
            return False

        key = stream_key or f"{self._prefix}:{event_type}"
        fields: dict[str, str] = {
            "event_type": event_type,
            "ts": str(time.time()),
        }
        for k, v in payload.items():
            fields[str(k)] = str(v)

        try:
            self._client.xadd(key, fields, maxlen=self._maxlen, approximate=True)
            return True
        except Exception as exc:
            logger.debug("[EventBus] publish failed for %s: %s", key, exc)
            return False

    def publish_batch(
        self,
        events: list[tuple[str, dict[str, Any]]],
    ) -> int:
        """Publish multiple events in a pipeline. Returns count of successes."""
        if not self._available or self._client is None:
            return 0

        ok = 0
        try:
            pipe = self._client.pipeline(transaction=False)
            for event_type, payload in events:
                key = f"{self._prefix}:{event_type}"
                fields: dict[str, str] = {
                    "event_type": event_type,
                    "ts": str(time.time()),
                }
                for k, v in payload.items():
                    fields[str(k)] = str(v)
                pipe.xadd(key, fields, maxlen=self._maxlen, approximate=True)
            results = pipe.execute()
            ok = sum(1 for r in results if r is not None)
        except Exception as exc:
            logger.debug("[EventBus] batch publish failed: %s", exc)
        return ok

    def read_latest(
        self,
        event_type: str,
        count: int = 10,
        stream_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read the most recent entries from a stream (for testing/diagnostics)."""
        if not self._available or self._client is None:
            return []

        key = stream_key or f"{self._prefix}:{event_type}"
        try:
            raw = self._client.xrevrange(key, count=count)
            return [{"id": entry_id, **fields} for entry_id, fields in raw]
        except Exception as exc:
            logger.debug("[EventBus] read_latest failed for %s: %s", key, exc)
            return []

    def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._available = False


@contextlib.contextmanager
def streamed_phase(
    bus: EventBus,
    phase_name: str,
    metadata: dict[str, Any] | None = None,
) -> Generator[None, None, None]:
    """Context manager that publishes phase_start and phase_end events.

    The trading cycle body is always executed regardless of bus availability.

    Args:
        bus: EventBus instance.
        phase_name: Logical pipeline phase name.
        metadata: Extra key-value pairs to include in both events.

    Example::

        with streamed_phase(bus, "signal_generation", {"date": "2024-01-15"}):
            signals = compute_signals(data)
    """
    meta = metadata or {}
    t_start = time.time()
    bus.publish("phase_start", {"phase": phase_name, **meta})

    exc_info: str | None = None
    try:
        yield
    except Exception as exc:
        exc_info = str(exc)
        raise
    finally:
        elapsed = time.time() - t_start
        bus.publish(
            "phase_end",
            {
                "phase": phase_name,
                "elapsed_s": f"{elapsed:.3f}",
                "status": "error" if exc_info else "ok",
                **({"error": exc_info[:200]} if exc_info else {}),
                **meta,
            },
        )


# Module-level no-op singleton for callers that don't want to manage bus lifecycle
_NULL_BUS: EventBus | None = None


def get_null_bus() -> EventBus:
    """Return a shared no-op EventBus (never connects to Redis)."""
    global _NULL_BUS
    if _NULL_BUS is None:
        _NULL_BUS = EventBus.__new__(EventBus)
        _NULL_BUS._prefix = "assembled"
        _NULL_BUS._maxlen = 0
        _NULL_BUS._client = None
        _NULL_BUS._available = False
    return _NULL_BUS
