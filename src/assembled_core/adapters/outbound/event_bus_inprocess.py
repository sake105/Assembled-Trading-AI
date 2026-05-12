# src/assembled_core/adapters/outbound/event_bus_inprocess.py
"""In-process EventBus adapter (audit C2-053).

A single-worker pub/sub implementation of the ``EventBus`` port. Good
enough for the current FastAPI single-worker deployment; the port
contract stays the same when we migrate to Redis Streams / NATS / Kafka.

Design notes:
    - Subscribers are invoked synchronously on the publisher's thread.
      If a subscriber raises, we log + continue — one bad handler does
      not blast-radius the others.
    - Topic strings are exact-match (no wildcards). A separate adapter
      can layer wildcards on top when needed.
    - No on-disk persistence — event durability is not the bus's job;
      the audit-logger port handles that.
    - Thread-safe via a single ``threading.Lock`` around the handler
      registry. The publish path holds the lock only to snapshot the
      current handler list, then dispatches outside the lock so a
      handler that itself publishes does not self-deadlock.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Mapping

from src.assembled_core.ports.event_bus import EventBus, Handler

logger = logging.getLogger(__name__)


class InProcessEventBus(EventBus):
    """Thread-safe single-worker pub/sub."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._publish_count: int = 0
        self._dispatch_errors: int = 0

    # ------------------------------------------------------------------
    # EventBus port
    # ------------------------------------------------------------------

    def publish(self, topic: str, event: Mapping[str, Any]) -> None:
        with self._lock:
            handlers = list(self._handlers.get(topic, ()))
            self._publish_count += 1
        if not handlers:
            logger.debug("[event_bus] no subscribers for topic=%s", topic)
            return
        # Take a defensive copy of the payload so a handler can't
        # mutate the dict and surprise downstream subscribers.
        payload: Mapping[str, Any] = dict(event)
        for h in handlers:
            try:
                h(payload)
            except Exception as exc:  # noqa: BLE001 — isolation is the contract
                self._dispatch_errors += 1
                logger.warning(
                    "[event_bus] handler %s raised on topic=%s: %s",
                    getattr(h, "__name__", repr(h)),
                    topic,
                    exc,
                )

    def subscribe(self, topic: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[topic].append(handler)

    # ------------------------------------------------------------------
    # Diagnostics (not part of the port)
    # ------------------------------------------------------------------

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def dispatch_errors(self) -> int:
        return self._dispatch_errors

    def topics(self) -> list[str]:
        with self._lock:
            return sorted(self._handlers.keys())


__all__ = ["InProcessEventBus"]
