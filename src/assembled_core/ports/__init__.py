"""Ports — Protocol interfaces the domain needs from the outside world.

Audit C-001: "What the domain needs from outside, expressed as
Protocols (PEP 544 / typing.Protocol) so test fakes and production
adapters are both first-class implementations."

Every Port lives in its own module so a future hexagonal refactor can
move it without rippling. Each adapter under ``assembled_core.adapters``
implements one or more ports.

The rule: domain modules import from ``assembled_core.ports`` but never
from ``assembled_core.adapters``. Adapters import port Protocols and
provide concrete implementations.
"""

from src.assembled_core.ports.alert_channel import AlertChannel, Severity
from src.assembled_core.ports.audit_logger import AuditLogger
from src.assembled_core.ports.clock import Clock
from src.assembled_core.ports.event_bus import EventBus
from src.assembled_core.ports.order_router import OrderRouter, OrderResult
from src.assembled_core.ports.prices_repository import PricesRepository

__all__ = [
    "AlertChannel",
    "AuditLogger",
    "Clock",
    "EventBus",
    "OrderResult",
    "OrderRouter",
    "PricesRepository",
    "Severity",
]
