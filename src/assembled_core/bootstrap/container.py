# src/assembled_core/bootstrap/container.py
"""Composition root (DI container) — audit C-003.

A *tiny* hand-written container — no Flask / no injector / no
dependency-injector. The audit explicitly recommends constructor
injection without a framework because the dependency graph in a
solo-quant backend is small enough to keep wiring explicit and
typecheckable.

Two factory functions:

- ``build_production_container()`` — real adapters everywhere.
- ``build_test_container(...)``    — fakes for everything by default;
  the caller can swap any single adapter for a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assembled_core.ports.alert_channel import AlertChannel
from src.assembled_core.ports.audit_logger import AuditLogger
from src.assembled_core.ports.clock import Clock


@dataclass(slots=True)
class Container:
    """Resolved dependency graph passed to use-cases via constructor.

    Every field is a port (Protocol). Use-cases receive a Container
    instance and pull the ports they need from it.
    """

    clock: Clock
    alerts: AlertChannel
    audit: AuditLogger


def build_production_container() -> Container:
    """Wire the real-world adapter implementations."""
    from src.assembled_core.adapters.outbound.alerting_adapter import (
        AlertManagerChannel,
    )
    from src.assembled_core.adapters.outbound.audit_logger_adapter import (
        JsonlAuditLogger,
    )
    from src.assembled_core.adapters.outbound.clock_adapter import SystemClock

    return Container(
        clock=SystemClock(),
        alerts=AlertManagerChannel(),
        audit=JsonlAuditLogger(),
    )


def build_test_container(
    *,
    clock: Clock | None = None,
    alerts: AlertChannel | None = None,
    audit: AuditLogger | None = None,
    audit_path: Path | str | None = None,
) -> Container:
    """Default to in-memory / frozen fakes; caller overrides any port.

    Each parameter defaults to a deterministic test fake. To use a
    real adapter for a single port in a test, pass it explicitly.
    """
    from src.assembled_core.adapters.outbound.alerting_adapter import NullAlertChannel
    from src.assembled_core.adapters.outbound.audit_logger_adapter import (
        InMemoryAuditLogger,
        JsonlAuditLogger,
    )
    from src.assembled_core.adapters.outbound.clock_adapter import FrozenClock

    if audit is None:
        audit = (
            JsonlAuditLogger(audit_path=audit_path)
            if audit_path is not None
            else InMemoryAuditLogger()
        )
    return Container(
        clock=clock or FrozenClock(),
        alerts=alerts or NullAlertChannel(),
        audit=audit,
    )


__all__ = ["Container", "build_production_container", "build_test_container"]
