"""OPS-02 regression: the scheduler's heartbeat must be watchdog-visible.

Before OPS-02 ``paper_trading_scheduler.py`` wrote a divergent file
``output/ops/scheduler_heartbeat.json`` with field ``timestamp_utc`` that the
Dead-Man's Switch (which reads ``output/state/heartbeat.json`` field
``timestamp`` via ``ops.heartbeat.check_liveness``) NEVER read. The scheduler's
frequent liveness beat was therefore invisible to the watchdog. OPS-02 routes
the scheduler through ``ops.heartbeat.write_heartbeat`` so it lands on the SAME
canonical file + schema. These tests lock that contract so a future refactor
cannot silently re-diverge the two systems.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.paper_trading_scheduler as sched  # noqa: E402
from src.assembled_core.ops.heartbeat import (  # noqa: E402
    check_liveness,
    read_heartbeat,
)


def test_scheduler_writes_canonical_heartbeat_schema(
    monkeypatch, tmp_path: Path
) -> None:
    """_write_heartbeat emits the canonical ``timestamp`` schema, not ``timestamp_utc``."""
    hb = tmp_path / "state" / "heartbeat.json"
    monkeypatch.setattr(sched, "HEARTBEAT_PATH", hb)

    sched._write_heartbeat("alive")

    assert hb.exists(), "scheduler heartbeat was not written"
    data = read_heartbeat(hb)
    assert data is not None
    # Canonical schema: ``timestamp`` present, divergent ``timestamp_utc`` absent.
    assert "timestamp" in data
    assert "timestamp_utc" not in data
    assert data.get("status") == "alive"
    details = data.get("details") or {}
    assert details.get("source") == "paper_trading_scheduler"
    assert isinstance(details.get("pid"), int)


def test_scheduler_heartbeat_is_live_for_watchdog(monkeypatch, tmp_path: Path) -> None:
    """A fresh scheduler beat is read as ALIVE by the same check the DMS uses."""
    hb = tmp_path / "state" / "heartbeat.json"
    monkeypatch.setattr(sched, "HEARTBEAT_PATH", hb)

    sched._write_heartbeat("alive")

    res = check_liveness(hb, max_age_seconds=900)
    assert res["alive"] is True
    assert res["reason"] == "ok"


def test_scheduler_heartbeat_default_path_is_canonical() -> None:
    """The module default points at the canonical watchdog file, not the old one."""
    parts = sched.HEARTBEAT_PATH.parts
    assert parts[-2:] == ("state", "heartbeat.json"), (
        f"HEARTBEAT_PATH must be output/state/heartbeat.json, got {sched.HEARTBEAT_PATH}"
    )
    assert sched.HEARTBEAT_PATH.name != "scheduler_heartbeat.json"


def test_scheduler_heartbeat_write_never_raises(monkeypatch, tmp_path: Path) -> None:
    """A heartbeat write failure must not propagate out of the monitor loop."""
    hb = tmp_path / "state" / "heartbeat.json"
    monkeypatch.setattr(sched, "HEARTBEAT_PATH", hb)

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sched, "write_heartbeat", _boom)

    # Must swallow the error (logged), not raise.
    sched._write_heartbeat("alive")
