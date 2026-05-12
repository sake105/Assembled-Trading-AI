"""End-to-end smoke tests for the hexagonal skeleton (Wave 17).

Verifies that the new path actually works — Container + AlertChannel +
Clock + AuditLogger + RecordKillSwitchTrip use-case all the way through.
"""

from __future__ import annotations

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Container smoke
# ---------------------------------------------------------------------------


def test_build_production_container_resolves_all_ports() -> None:
    from src.assembled_core.bootstrap import build_production_container

    container = build_production_container()
    assert container.clock is not None
    assert container.alerts is not None
    assert container.audit is not None
    # Clock is callable.
    assert isinstance(container.clock.now(), datetime)


def test_build_test_container_defaults_to_fakes() -> None:
    from src.assembled_core.adapters.outbound.alerting_adapter import NullAlertChannel
    from src.assembled_core.adapters.outbound.audit_logger_adapter import (
        InMemoryAuditLogger,
    )
    from src.assembled_core.adapters.outbound.clock_adapter import FrozenClock
    from src.assembled_core.bootstrap import build_test_container

    container = build_test_container()
    assert isinstance(container.clock, FrozenClock)
    assert isinstance(container.alerts, NullAlertChannel)
    assert isinstance(container.audit, InMemoryAuditLogger)


def test_frozen_clock_ticks_forward() -> None:
    from src.assembled_core.adapters.outbound.clock_adapter import FrozenClock

    fc = FrozenClock(initial=datetime(2026, 1, 1, tzinfo=timezone.utc))
    t0 = fc.now()
    fc.tick(minutes=15)
    t1 = fc.now()
    assert (t1 - t0).total_seconds() == 15 * 60


# ---------------------------------------------------------------------------
# Use-case smoke
# ---------------------------------------------------------------------------


def test_record_kill_switch_trip_writes_audit_and_alerts() -> None:
    from src.assembled_core.adapters.outbound.alerting_adapter import NullAlertChannel
    from src.assembled_core.adapters.outbound.audit_logger_adapter import (
        InMemoryAuditLogger,
    )
    from src.assembled_core.application.use_cases.record_kill_switch_trip import (
        RecordKillSwitchTrip,
    )
    from src.assembled_core.bootstrap import build_test_container

    alerts = NullAlertChannel()
    audit = InMemoryAuditLogger()
    container = build_test_container(alerts=alerts, audit=audit)

    uc = RecordKillSwitchTrip(container)
    result = uc.execute(
        reason="vol spike — short / long ratio 2.4",
        actor="auto_trip_check",
        throttle_pct=0.0,
        extra_context={"ratio": 2.4},
    )

    assert result.audit_record_count == 1
    assert result.alerted is True
    assert audit.events[0]["kind"] == "kill_switch_trip"
    assert audit.events[0]["reason"].startswith("vol spike")
    assert audit.events[0]["throttle_pct"] == 0.0
    assert audit.events[0]["ratio"] == 2.4
    assert alerts.calls[0][0] == "kill_switch_activated"


def test_record_kill_switch_trip_uses_clock_for_timestamp() -> None:
    from datetime import datetime, timezone

    from src.assembled_core.adapters.outbound.clock_adapter import FrozenClock
    from src.assembled_core.application.use_cases.record_kill_switch_trip import (
        RecordKillSwitchTrip,
    )
    from src.assembled_core.bootstrap import build_test_container

    fixed = datetime(2026, 5, 12, 9, 30, tzinfo=timezone.utc)
    container = build_test_container(clock=FrozenClock(initial=fixed))
    uc = RecordKillSwitchTrip(container)
    result = uc.execute(reason="test", actor="t", throttle_pct=1.0)
    assert result.ts == fixed.isoformat()


# ---------------------------------------------------------------------------
# Ports are Protocols (runtime-checkable)
# ---------------------------------------------------------------------------


def test_ports_are_runtime_checkable_protocols() -> None:
    from src.assembled_core.ports import AlertChannel, AuditLogger, Clock

    from src.assembled_core.adapters.outbound.alerting_adapter import NullAlertChannel
    from src.assembled_core.adapters.outbound.audit_logger_adapter import (
        InMemoryAuditLogger,
    )
    from src.assembled_core.adapters.outbound.clock_adapter import SystemClock

    assert isinstance(SystemClock(), Clock)
    assert isinstance(NullAlertChannel(), AlertChannel)
    assert isinstance(InMemoryAuditLogger(), AuditLogger)


# ---------------------------------------------------------------------------
# JsonlAuditLogger writes a valid hash chain
# ---------------------------------------------------------------------------


def test_jsonl_audit_logger_writes_verifiable_chain(tmp_path) -> None:
    from src.assembled_core.adapters.outbound.audit_logger_adapter import (
        JsonlAuditLogger,
    )

    p = tmp_path / "audit.jsonl"
    logger = JsonlAuditLogger(audit_path=p)
    logger.append({"kind": "test", "x": 1})
    logger.append({"kind": "test", "x": 2})
    ok, n = logger.verify_chain()
    assert ok is True
    assert n == 2
