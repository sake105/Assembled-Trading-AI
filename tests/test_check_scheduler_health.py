"""OPS-03 — scheduler-health detector reads the REAL deployed heartbeat.

These tests pin the behaviour the 2026-04-10 silent-stall postmortem demanded:
the detector must (a) read the production heartbeat path/schema, (b) flag a
stale or missing heartbeat, and (c) be drivable against an arbitrary path so
the synthetic drill stays hermetic.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from scripts.check_scheduler_health import (
    DEFAULT_HEARTBEAT_PATH,
    _resolve_heartbeat_path,
    main,
)


def _write_hb(path, *, ts: datetime, field: str = "timestamp", status: str = "ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, field: ts.isoformat()}
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.fast
def test_default_path_is_deployed_state_heartbeat():
    # Regression: the detector must default to the file the DEPLOYED pilot
    # writes (_tc_execution -> output/state/heartbeat.json), not the daemon's
    # output/ops/scheduler_heartbeat.json that production never writes.
    assert DEFAULT_HEARTBEAT_PATH.as_posix() == "output/state/heartbeat.json"


@pytest.mark.fast
def test_fresh_production_heartbeat_is_ok(tmp_path):
    hb = tmp_path / "state" / "heartbeat.json"
    _write_hb(hb, ts=datetime.now(timezone.utc), field="timestamp")
    rc = main(["--heartbeat-path", str(hb), "--ignore-market-hours"])
    assert rc == 0


@pytest.mark.fast
def test_timestamp_utc_field_also_parsed(tmp_path):
    # The daemon/drill schema uses ``timestamp_utc``; the detector must accept
    # both field names (full writer unification is OPS-02).
    hb = tmp_path / "ops" / "scheduler_heartbeat.json"
    _write_hb(hb, ts=datetime.now(timezone.utc), field="timestamp_utc")
    rc = main(["--heartbeat-path", str(hb), "--ignore-market-hours"])
    assert rc == 0


@pytest.mark.fast
def test_stale_heartbeat_returns_1(tmp_path):
    hb = tmp_path / "state" / "heartbeat.json"
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    _write_hb(hb, ts=stale, field="timestamp")
    rc = main(
        ["--heartbeat-path", str(hb), "--ignore-market-hours", "--stale-minutes", "10"]
    )
    assert rc == 1


@pytest.mark.fast
def test_missing_heartbeat_returns_2(tmp_path):
    hb = tmp_path / "state" / "heartbeat.json"  # never created
    rc = main(["--heartbeat-path", str(hb), "--ignore-market-hours"])
    assert rc == 2


@pytest.mark.fast
def test_no_timestamp_field_returns_2(tmp_path):
    hb = tmp_path / "state" / "heartbeat.json"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    rc = main(["--heartbeat-path", str(hb), "--ignore-market-hours"])
    assert rc == 2


@pytest.mark.fast
def test_resolve_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("SCHEDULER_HEARTBEAT_PATH", raising=False)
    # default
    assert _resolve_heartbeat_path(None) == DEFAULT_HEARTBEAT_PATH
    # env overrides default
    monkeypatch.setenv("SCHEDULER_HEARTBEAT_PATH", str(tmp_path / "env.json"))
    assert _resolve_heartbeat_path(None) == tmp_path / "env.json"
    # cli overrides env
    assert _resolve_heartbeat_path(str(tmp_path / "cli.json")) == tmp_path / "cli.json"


@pytest.mark.fast
def test_notify_posts_on_stale(tmp_path, monkeypatch):
    import src.assembled_core.ops.alert_sinks as sinks

    calls: list[str] = []
    monkeypatch.setenv("DISCORD_WEBHOOK", "https://example.invalid/webhook")
    monkeypatch.setattr(
        sinks, "post_discord", lambda webhook, content: calls.append(content) or True
    )

    hb = tmp_path / "state" / "heartbeat.json"
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    _write_hb(hb, ts=stale, field="timestamp")

    rc = main(
        [
            "--heartbeat-path",
            str(hb),
            "--ignore-market-hours",
            "--stale-minutes",
            "10",
            "--notify",
        ]
    )
    assert rc == 1
    assert calls and "stale" in calls[0].lower()


@pytest.mark.fast
def test_notify_absent_does_not_post(tmp_path, monkeypatch):
    import src.assembled_core.ops.alert_sinks as sinks

    calls: list[str] = []
    monkeypatch.setenv("DISCORD_WEBHOOK", "https://example.invalid/webhook")
    monkeypatch.setattr(
        sinks, "post_discord", lambda webhook, content: calls.append(content) or True
    )

    hb = tmp_path / "state" / "heartbeat.json"
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    _write_hb(hb, ts=stale, field="timestamp")

    rc = main(
        ["--heartbeat-path", str(hb), "--ignore-market-hours", "--stale-minutes", "10"]
    )
    assert rc == 1
    assert calls == []
