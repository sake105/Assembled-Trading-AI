"""Order lifecycle tracking with full audit trail.

Provides an enum-based lifecycle and a lightweight tracker that records
every state transition with timestamps.  Designed to be used alongside
the existing DataFrame-based order pipeline without replacing it.

Usage::

    from src.assembled_core.execution.order_lifecycle import (
        OrderState, OrderLifecycleTracker,
    )

    tracker = OrderLifecycleTracker()
    oid = tracker.create("AAPL", "BUY", 100, price=150.0, source="SIGNAL")
    tracker.transition(oid, OrderState.VALIDATED)
    tracker.transition(oid, OrderState.SUBMITTED)
    tracker.transition(oid, OrderState.FILLED, fill_price=150.05)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class OrderState(str, Enum):
    """Possible states in the order lifecycle."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# Valid transitions: from_state -> set of allowed to_states
_VALID_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CREATED: {OrderState.VALIDATED, OrderState.REJECTED},
    OrderState.VALIDATED: {OrderState.SUBMITTED, OrderState.REJECTED},
    OrderState.SUBMITTED: {
        OrderState.PARTIAL_FILL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
    },
    OrderState.PARTIAL_FILL: {
        OrderState.PARTIAL_FILL,
        OrderState.FILLED,
        OrderState.CANCELLED,
    },
    # Terminal states — no outgoing transitions
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
}


@dataclass(slots=True)
class OrderEvent:
    """A single lifecycle event for an order."""

    state: OrderState
    timestamp: datetime
    details: dict = field(default_factory=dict)


@dataclass(slots=True)
class TrackedOrder:
    """An order with its full lifecycle history."""

    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float | None
    source: str
    current_state: OrderState
    events: list[OrderEvent] = field(default_factory=list)
    fill_price: float | None = None
    fill_qty: float | None = None
    reject_reason: str | None = None

    @property
    def created_at(self) -> datetime | None:
        for e in self.events:
            if e.state == OrderState.CREATED:
                return e.timestamp
        return None

    @property
    def terminal_at(self) -> datetime | None:
        """Timestamp when the order reached a terminal state."""
        terminal = {OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED}
        for e in reversed(self.events):
            if e.state in terminal:
                return e.timestamp
        return None

    @property
    def submitted_at(self) -> datetime | None:
        """Timestamp of the first SUBMITTED transition (if any)."""
        for e in self.events:
            if e.state == OrderState.SUBMITTED:
                return e.timestamp
        return None

    @property
    def is_terminal(self) -> bool:
        return self.current_state in {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
        }

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "source": self.source,
            "current_state": self.current_state.value,
            "fill_price": self.fill_price,
            "fill_qty": self.fill_qty,
            "reject_reason": self.reject_reason,
            "created_at": str(self.created_at) if self.created_at else None,
            "terminal_at": str(self.terminal_at) if self.terminal_at else None,
            "n_events": len(self.events),
        }


class OrderLifecycleTracker:
    """Tracks order lifecycle with state machine validation."""

    def __init__(self) -> None:
        self._orders: dict[str, TrackedOrder] = {}

    def create(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        source: str = "UNKNOWN",
        order_id: str | None = None,
    ) -> str:
        """Create a new tracked order.

        Returns:
            The order ID.
        """
        oid = order_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        order = TrackedOrder(
            order_id=oid,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            source=source,
            current_state=OrderState.CREATED,
            events=[OrderEvent(state=OrderState.CREATED, timestamp=now)],
        )
        self._orders[oid] = order
        return oid

    def transition(
        self,
        order_id: str,
        new_state: OrderState,
        *,
        fill_price: float | None = None,
        fill_qty: float | None = None,
        reason: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Transition an order to a new state.

        Validates that the transition is legal according to the state machine.

        Raises:
            KeyError: If order_id not found.
            ValueError: If the transition is not valid.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id} not found in tracker")

        allowed = _VALID_TRANSITIONS.get(order.current_state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition for order {order_id}: "
                f"{order.current_state.value} -> {new_state.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )

        now = datetime.now(timezone.utc)
        event_details = details or {}
        if reason:
            event_details["reason"] = reason

        order.events.append(
            OrderEvent(state=new_state, timestamp=now, details=event_details)
        )
        order.current_state = new_state

        if fill_price is not None:
            order.fill_price = fill_price
        if fill_qty is not None:
            order.fill_qty = fill_qty
        if reason and new_state == OrderState.REJECTED:
            order.reject_reason = reason

        logger.debug(
            "[OrderLifecycle] %s %s %s %.0f %s -> %s",
            order_id[:8],
            order.symbol,
            order.side,
            order.quantity,
            order.events[-2].state.value,
            new_state.value,
        )

    def get_order(self, order_id: str) -> TrackedOrder | None:
        return self._orders.get(order_id)

    def get_all_orders(self) -> list[TrackedOrder]:
        return list(self._orders.values())

    def get_active_orders(self) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if not o.is_terminal]

    def summary(self) -> dict[str, int]:
        """Count orders by current state."""
        counts: dict[str, int] = {}
        for order in self._orders.values():
            key = order.current_state.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def find_stuck_orders(
        self,
        max_age_seconds: float = 30.0,
        *,
        now: datetime | None = None,
    ) -> list[TrackedOrder]:
        """Return orders in SUBMITTED for longer than ``max_age_seconds``.

        Audit C4-020: an order whose broker-ack got lost stays in SUBMITTED
        forever. Detect this so the reconciler can mark it UNKNOWN and the
        operator gets paged before duplicate orders get sent.

        Args:
            max_age_seconds: orders older than this (since SUBMITTED) are
                considered stuck. Default 30 s.
            now: clock override (test seam). Defaults to current UTC time.

        Returns:
            List of TrackedOrders currently SUBMITTED and older than the cap.
            Sorted by submitted_at ascending (oldest first).
        """
        ref = now if now is not None else datetime.now(timezone.utc)
        stuck: list[TrackedOrder] = []
        for o in self._orders.values():
            if o.current_state != OrderState.SUBMITTED:
                continue
            ts = o.submitted_at
            if ts is None:
                continue
            age = (ref - ts).total_seconds()
            if age > max_age_seconds:
                stuck.append(o)
        stuck.sort(key=lambda o: o.submitted_at or ref)
        return stuck
