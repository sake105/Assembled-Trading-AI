# src/assembled_core/ports/clock.py
"""Clock port — the domain's view of "what time is it now?".

Why a port? Because every piece of domain logic that uses
``datetime.now(timezone.utc)`` directly becomes untestable — you cannot
freeze the wall clock in a property test. With a Clock port the domain
asks the container for the time; production passes SystemClock,
tests pass FrozenClock.

Audit C-001 (hexagonal ports), C-002 (Clock as one of the canonical
ports). Adapter implementations live in
``assembled_core.adapters.outbound.clock``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Returns the current UTC time."""

    def now(self) -> datetime: ...
