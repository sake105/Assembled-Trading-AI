# src/assembled_core/ports/event_bus.py
"""EventBus port — pub/sub for domain events.

The audit's signal-bus (C2-053) is a SignalEvent stream the master
allocator subscribes to; an in-process implementation is enough for a
single-worker FastAPI deployment, but the port stays the same if we
ever switch to Redis Streams / NATS / Kafka.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, runtime_checkable


Handler = Callable[[Mapping[str, Any]], None]


@runtime_checkable
class EventBus(Protocol):
    def publish(self, topic: str, event: Mapping[str, Any]) -> None: ...

    def subscribe(self, topic: str, handler: Handler) -> None: ...
