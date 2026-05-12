# src/assembled_core/ports/order_router.py
"""OrderRouter port — domain-facing API for submitting orders to a broker.

The domain emits an ``OrderRequest``-shaped object (left intentionally
loose during the skeleton phase) and gets back an ``OrderResult``. The
broker-specific complexity (Alpaca vs IBKR vs paper vs FIX) lives in
adapters/outbound/broker/*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(slots=True)
class OrderResult:
    """Minimal outcome envelope returned by an OrderRouter."""

    order_id: str
    status: str  # "ACCEPTED" | "REJECTED" | "FILLED"
    reason: str | None = None
    fill_price: float | None = None
    fill_qty: float | None = None
    raw: Mapping[str, Any] | None = None


@runtime_checkable
class OrderRouter(Protocol):
    """Submit a single order. Implementations MUST be idempotent on the
    ``client_order_id`` field of the request (audit C4-035).
    """

    def submit(self, order_request: Mapping[str, Any]) -> OrderResult: ...

    def cancel(self, order_id: str) -> OrderResult: ...
