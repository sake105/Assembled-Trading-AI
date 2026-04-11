"""Tests for ops/heartbeat.py (Sprint 4 / Plan C16)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.heartbeat import (  # noqa: E402
    check_liveness,
    heartbeat_age_seconds,
    read_heartbeat,
    write_heartbeat,
)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    now = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
    write_heartbeat(p, status="ok", details={"run_id": "abc"}, now=now)
    data = read_heartbeat(p)
    assert data is not None
    assert data["status"] == "ok"
    assert data["details"] == {"run_id": "abc"}
    assert data["timestamp"].startswith("2026-04-11T12:00:00")


def test_read_missing_returns_none(tmp_path: Path) -> None:
    assert read_heartbeat(tmp_path / "nope.json") is None


def test_read_corrupt_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    p.write_text("not json", encoding="utf-8")
    assert read_heartbeat(p) is None


def test_age_seconds_basic(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    t0 = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
    write_heartbeat(p, now=t0)
    age = heartbeat_age_seconds(p, now=t0 + timedelta(seconds=120))
    assert age is not None
    assert 119 <= age <= 121


def test_age_seconds_missing_file(tmp_path: Path) -> None:
    assert heartbeat_age_seconds(tmp_path / "nope.json") is None


def test_liveness_ok(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    t0 = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
    write_heartbeat(p, status="ok", now=t0)
    res = check_liveness(p, max_age_seconds=900, now=t0 + timedelta(seconds=60))
    assert res["alive"] is True
    assert res["reason"] == "ok"
    assert res["status"] == "ok"


def test_liveness_stale(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    t0 = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
    write_heartbeat(p, status="ok", now=t0)
    res = check_liveness(p, max_age_seconds=60, now=t0 + timedelta(seconds=600))
    assert res["alive"] is False
    assert res["reason"].startswith("stale:")


def test_liveness_halt_status(tmp_path: Path) -> None:
    p = tmp_path / "hb.json"
    t0 = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
    write_heartbeat(p, status="halt", now=t0)
    res = check_liveness(p, max_age_seconds=900, now=t0 + timedelta(seconds=1))
    assert res["alive"] is False
    assert res["reason"] == "status_halt"


def test_liveness_missing_file(tmp_path: Path) -> None:
    res = check_liveness(tmp_path / "nope.json", max_age_seconds=900)
    assert res["alive"] is False
    assert res["reason"] == "missing_or_unreadable"
