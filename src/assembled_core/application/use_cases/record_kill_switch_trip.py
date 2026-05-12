# src/assembled_core/application/use_cases/record_kill_switch_trip.py
"""RecordKillSwitchTrip use-case — first hexagonal use-case (audit C-001).

Orchestrates three ports — Clock, AlertChannel, AuditLogger — to
deliver a single domain behaviour: "the kill-switch just tripped;
write it down, alert the operator, and stamp the time".

The point of this exercise is to demonstrate the pattern end-to-end,
not to replace the existing kill_switch.py call path. Both coexist:

- ``execution.kill_switch.activate_kill_switch`` keeps its current
  in-line wiring (alerting, audit, time-stamping).
- New domain code that needs to record a trip can call this use-case
  with a Container and get the same behaviour through ports, ready to
  be tested with fakes.

The migration path the audit specifies (C-007) is to move
``activate_kill_switch`` to call this use-case once enough call sites
exist to make the swap meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.assembled_core.bootstrap.container import Container
from src.assembled_core.ports.alert_channel import Severity


@dataclass(slots=True)
class TripRecord:
    """Outcome of recording a kill-switch trip."""

    ts: str  # iso-8601 from Clock.now()
    alerted: bool
    audit_record_count: int


class RecordKillSwitchTrip:
    """Single-call use-case — Clock + AlertChannel + AuditLogger."""

    def __init__(self, container: Container) -> None:
        self._container = container

    def execute(
        self,
        *,
        reason: str,
        actor: str,
        throttle_pct: float,
        extra_context: Mapping[str, Any] | None = None,
    ) -> TripRecord:
        ts = self._container.clock.now().isoformat()
        context: dict[str, Any] = {
            "ts": ts,
            "reason": reason,
            "actor": actor,
            "throttle_pct": throttle_pct,
        }
        if extra_context:
            context.update(extra_context)

        # 1. Append to the audit log (the source-of-truth for forensics).
        self._container.audit.append(
            {
                "kind": "kill_switch_trip",
                **context,
            }
        )

        # 2. Notify operators via the configured alert channel(s).
        alerted = self._container.alerts.fire(
            "kill_switch_activated",
            context=context,
            severity=Severity.CRITICAL,
        )

        # 3. Verify chain integrity AFTER the write so a tamper is
        #    surfaced immediately to the use-case caller, not at the
        #    next reconciliation tick.
        _ok, n_records = self._container.audit.verify_chain()

        return TripRecord(ts=ts, alerted=alerted, audit_record_count=n_records)


__all__ = ["RecordKillSwitchTrip", "TripRecord"]
