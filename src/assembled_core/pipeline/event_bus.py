"""Lightweight Event Bus for Event-Driven Architecture (M23 Task 23.2).

asyncio.Queue-based event bus for decoupled module communication.
Replaces synchronous time.sleep() polling with event-driven handlers.

Event types: PRICE_UPDATE, FILL, SIGNAL_UPDATE, RISK_BREACH, NEWS_EVENT
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(Enum):
    """System event types."""
    PRICE_UPDATE = "price_update"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_CANCELLED = "order_cancelled"
    SIGNAL_UPDATE = "signal_update"
    RISK_BREACH = "risk_breach"
    RISK_WARNING = "risk_warning"
    KILL_SWITCH = "kill_switch"
    NEWS_EVENT = "news_event"
    REBALANCE_TRIGGER = "rebalance_trigger"
    MODEL_RETRAIN = "model_retrain"
    DATA_STALE = "data_stale"
    HEARTBEAT = "heartbeat"
    SYSTEM_ERROR = "system_error"


@dataclass
class Event:
    """A typed event with payload."""
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0
    source: str = ""
    priority: int = 0  # Higher = more urgent

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


# Handler type: sync or async callable
HandlerFunc = Callable[[Event], Any] | Callable[[Event], Coroutine]


class EventBus:
    """Lightweight event bus with async queue and handler registration.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.FILL, handle_fill)
        bus.subscribe(EventType.RISK_BREACH, handle_risk)
        await bus.start()

        # From another coroutine:
        await bus.publish(Event(EventType.FILL, {"symbol": "AAPL", "qty": 100}))
    """

    def __init__(self, max_queue_size: int = 10000):
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=max_queue_size)
        self._handlers: dict[EventType, list[HandlerFunc]] = {}
        self._wildcard_handlers: list[HandlerFunc] = []
        self._running = False
        self._processed_count = 0
        self._error_count = 0

    def subscribe(
        self,
        event_type: EventType | None,
        handler: HandlerFunc,
    ) -> None:
        """Register an event handler.

        Args:
            event_type: Event type to listen for. None = wildcard (all events).
            handler: Sync or async callable receiving Event.
        """
        if event_type is None:
            self._wildcard_handlers.append(handler)
        else:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(
        self,
        event_type: EventType | None,
        handler: HandlerFunc,
    ) -> None:
        """Remove an event handler."""
        if event_type is None:
            if handler in self._wildcard_handlers:
                self._wildcard_handlers.remove(handler)
        else:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to the bus.

        Args:
            event: Event to publish.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error("[EventBus] Queue full — dropping %s event", event.event_type.value)
            self._error_count += 1

    def publish_sync(self, event: Event) -> None:
        """Synchronous publish (for non-async code).

        Args:
            event: Event to publish.
        """
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error("[EventBus] Queue full (sync) — dropping %s", event.event_type.value)

    async def start(self) -> None:
        """Start the event processing loop."""
        self._running = True
        logger.info("[EventBus] Started with %d handler registrations",
                     sum(len(h) for h in self._handlers.values()) + len(self._wildcard_handlers))

        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                await self._dispatch(event)
                self._processed_count += 1
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.error("[EventBus] Processing error: %s", exc)
                self._error_count += 1

    async def _dispatch(self, event: Event) -> None:
        """Dispatch event to all matching handlers."""
        handlers = list(self._handlers.get(event.event_type, []))
        handlers.extend(self._wildcard_handlers)

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.error(
                    "[EventBus] Handler error for %s: %s",
                    event.event_type.value, exc,
                )
                self._error_count += 1

    async def stop(self) -> None:
        """Stop the event processing loop."""
        self._running = False
        # Drain remaining events
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch(event)
            except asyncio.QueueEmpty:
                break
        logger.info("[EventBus] Stopped (processed=%d, errors=%d)",
                     self._processed_count, self._error_count)

    def get_stats(self) -> dict:
        """Get bus statistics."""
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "processed": self._processed_count,
            "errors": self._error_count,
            "handlers": {
                et.value: len(handlers)
                for et, handlers in self._handlers.items()
            },
            "wildcard_handlers": len(self._wildcard_handlers),
        }


__all__ = [
    "EventType",
    "Event",
    "EventBus",
]
