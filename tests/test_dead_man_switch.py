"""Tests for src/assembled_core/ops/dead_man_switch.py."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from src.assembled_core.ops.dead_man_switch import (
    _cfg,
    auto_flatten_on_stale,
    dms_monitor_loop,
    record_dms_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LIVE_LIVENESS = {"alive": True, "reason": "ok", "age_seconds": 10.0, "status": "ok"}
_STALE_LIVENESS = {
    "alive": False,
    "reason": "stale:950s>900s",
    "age_seconds": 950.0,
    "status": "ok",
}

_BASE_POLICY = {
    "dead_man_switch": {
        "enabled": True,
        "timeout_seconds": 900,
        "check_interval_seconds": 60,
        "flatten_mode": "market",
        "log_path": "output/ops/dms_audit.jsonl",
    }
}

_SHADOW_POLICY = {
    "dead_man_switch": {
        "enabled": True,
        "timeout_seconds": 900,
        "check_interval_seconds": 60,
        "flatten_mode": "shadow",
        "log_path": "output/ops/dms_audit.jsonl",
    }
}


# ---------------------------------------------------------------------------
# record_dms_event tests
# ---------------------------------------------------------------------------


def test_record_dms_event_writes_jsonl(tmp_path: Path) -> None:
    log_file = tmp_path / "dms_audit.jsonl"
    record_dms_event("heartbeat_timeout", log_file)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["event"] == "DMS_TRIGGER"
    assert rec["reason"] == "heartbeat_timeout"
    assert "ts" in rec
    assert "action_taken" in rec


def test_record_dms_event_appends(tmp_path: Path) -> None:
    log_file = tmp_path / "dms_audit.jsonl"
    record_dms_event("first_event", log_file)
    record_dms_event("second_event", log_file)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["reason"] == "first_event"
    assert second["reason"] == "second_event"


def test_record_dms_event_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "dms_audit.jsonl"
    assert not nested.parent.exists()
    record_dms_event("test_reason", nested)
    assert nested.exists()
    rec = json.loads(nested.read_text(encoding="utf-8").strip())
    assert rec["reason"] == "test_reason"


def test_record_dms_event_custom_action_taken(tmp_path: Path) -> None:
    log_file = tmp_path / "dms.jsonl"
    record_dms_event("timeout", log_file, action_taken="kill_switch_activated")
    rec = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert rec["action_taken"] == "kill_switch_activated"


def test_record_dms_event_extra_fields(tmp_path: Path) -> None:
    log_file = tmp_path / "dms.jsonl"
    record_dms_event("timeout", log_file, extra={"pid": 42, "node": "worker1"})
    rec = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert rec["pid"] == 42
    assert rec["node"] == "worker1"


# ---------------------------------------------------------------------------
# auto_flatten_on_stale tests
# ---------------------------------------------------------------------------


def test_auto_flatten_logs_reason(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 60,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }
    with (
        patch("src.assembled_core.ops.dead_man_switch.activate_kill_switch"),
        caplog.at_level("CRITICAL", logger="src.assembled_core.ops.dead_man_switch"),
    ):
        auto_flatten_on_stale(policy, reason="test_heartbeat_timeout")
    assert any("DMS-TRIGGER" in r.message for r in caplog.records)


def test_flatten_mode_shadow_skips_broker(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """shadow mode must NOT call activate_kill_switch."""
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 60,
            "flatten_mode": "shadow",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }
    with patch(
        "src.assembled_core.ops.dead_man_switch.activate_kill_switch"
    ) as mock_ks:
        auto_flatten_on_stale(policy, reason="test_shadow")
    mock_ks.assert_not_called()

    log_file = tmp_path / "dms.jsonl"
    assert log_file.exists()
    rec = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert rec["action_taken"] == "shadow_log_only"


def test_flatten_mode_market_calls_kill_switch(tmp_path: Path) -> None:
    """market mode MUST call activate_kill_switch with throttle_pct=0.0."""
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 60,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }
    with patch(
        "src.assembled_core.ops.dead_man_switch.activate_kill_switch"
    ) as mock_ks:
        auto_flatten_on_stale(policy, reason="market_test")
    mock_ks.assert_called_once_with(
        throttle_pct=0.0,
        reason="DMS: market_test",
        actor="dead_man_switch",
    )


# ---------------------------------------------------------------------------
# dms_monitor_loop tests
# ---------------------------------------------------------------------------


def test_no_action_when_heartbeat_fresh(tmp_path: Path) -> None:
    """When heartbeat is alive, no flatten must be called."""
    stop_event = threading.Event()
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 0.01,  # fast for tests
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }

    call_count = 0

    def fake_check_liveness(path, *, max_age_seconds):  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        stop_event.set()  # stop after first check
        return _LIVE_LIVENESS

    with (
        patch(
            "src.assembled_core.ops.dead_man_switch.check_liveness",
            side_effect=fake_check_liveness,
        ) as _mock_cl,
        patch(
            "src.assembled_core.ops.dead_man_switch.auto_flatten_on_stale"
        ) as mock_flatten,
    ):
        dms_monitor_loop(policy, stop_event=stop_event)

    mock_flatten.assert_not_called()
    assert call_count >= 1


def test_auto_flatten_triggered_on_stale(tmp_path: Path) -> None:
    """When liveness returns alive=False, auto_flatten_on_stale must be called."""
    stop_event = threading.Event()
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 0.01,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }

    def fake_check_liveness(path, *, max_age_seconds):  # noqa: ARG001
        stop_event.set()
        return _STALE_LIVENESS

    with (
        patch(
            "src.assembled_core.ops.dead_man_switch.check_liveness",
            side_effect=fake_check_liveness,
        ) as _mock_cl,
        patch(
            "src.assembled_core.ops.dead_man_switch.auto_flatten_on_stale"
        ) as mock_flatten,
    ):
        dms_monitor_loop(policy, stop_event=stop_event)

    mock_flatten.assert_called_once()
    call_kwargs = mock_flatten.call_args
    assert "heartbeat_timeout" in call_kwargs[1][
        "reason"
    ] or "heartbeat_timeout" in str(call_kwargs)


def test_dms_disabled_via_policy(tmp_path: Path) -> None:
    """enabled=false must exit the loop without ever calling check_liveness."""
    policy = {
        "dead_man_switch": {
            "enabled": False,
            "timeout_seconds": 900,
            "check_interval_seconds": 0.01,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }

    with patch("src.assembled_core.ops.dead_man_switch.check_liveness") as mock_cl:
        dms_monitor_loop(policy)

    mock_cl.assert_not_called()


def test_monitor_loop_exits_on_stop_event(tmp_path: Path) -> None:
    """A pre-set stop_event must cause immediate exit without polling."""
    stop_event = threading.Event()
    stop_event.set()  # already set before loop starts

    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 0.01,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }

    with (
        patch("src.assembled_core.ops.dead_man_switch.check_liveness") as mock_cl,
        patch(
            "src.assembled_core.ops.dead_man_switch.auto_flatten_on_stale"
        ) as mock_flatten,
    ):
        dms_monitor_loop(policy, stop_event=stop_event)

    # stop_event is pre-set, so the loop exits at the top-of-loop guard
    # before calling check_liveness or auto_flatten_on_stale.
    assert mock_cl.call_count == 0, (
        f"Expected check_liveness not called (stop pre-set), got {mock_cl.call_count} calls"
    )
    mock_flatten.assert_not_called()


def test_policy_defaults_used_when_missing(tmp_path: Path) -> None:
    """Empty policy dict must not crash — safe defaults are applied."""
    stop_event = threading.Event()

    def fake_check_liveness(path, *, max_age_seconds):  # noqa: ARG001
        # max_age_seconds must come from the default (900)
        assert max_age_seconds == 900.0
        stop_event.set()
        return _LIVE_LIVENESS

    with patch(
        "src.assembled_core.ops.dead_man_switch.check_liveness",
        side_effect=fake_check_liveness,
    ):
        dms_monitor_loop({}, stop_event=stop_event)


# ---------------------------------------------------------------------------
# _cfg helper tests
# ---------------------------------------------------------------------------


def test_cfg_defaults_when_block_absent() -> None:
    cfg = _cfg({})
    assert cfg["enabled"] is True
    assert cfg["timeout_seconds"] == 900.0
    assert cfg["check_interval_seconds"] == 60.0
    assert cfg["flatten_mode"] == "market"


def test_cfg_policy_wins_over_defaults() -> None:
    policy = {
        "dead_man_switch": {
            "timeout_seconds": 300,
            "flatten_mode": "shadow",
        }
    }
    cfg = _cfg(policy)
    assert cfg["timeout_seconds"] == 300
    assert cfg["flatten_mode"] == "shadow"
    # non-overridden defaults preserved
    assert cfg["enabled"] is True


# ---------------------------------------------------------------------------
# MAJOR 1 — fail-safe default: missing "alive" key must trigger flatten
# ---------------------------------------------------------------------------


def test_missing_alive_key_triggers_flatten(tmp_path: Path) -> None:
    """When check_liveness returns {} (no 'alive' key), auto_flatten_on_stale MUST be called.

    This verifies the fail-safe default: unknown liveness = assume dead (not alive).
    """
    stop_event = threading.Event()
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 0.01,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }

    def fake_check_liveness(path, *, max_age_seconds):  # noqa: ARG001
        stop_event.set()
        return {}  # no "alive" key — must be treated as dead

    with (
        patch(
            "src.assembled_core.ops.dead_man_switch.check_liveness",
            side_effect=fake_check_liveness,
        ),
        patch(
            "src.assembled_core.ops.dead_man_switch.auto_flatten_on_stale"
        ) as mock_flatten,
    ):
        dms_monitor_loop(policy, stop_event=stop_event)

    mock_flatten.assert_called_once()


# ---------------------------------------------------------------------------
# MAJOR 2 — consecutive check_liveness failures escalate to CRITICAL + flatten
# ---------------------------------------------------------------------------


def test_consecutive_liveness_failures_trigger_critical_flatten(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """After N=4 consecutive check_liveness exceptions, CRITICAL is logged
    and auto_flatten_on_stale is called as a conservative fail-safe.
    """
    stop_event = threading.Event()
    policy = {
        "dead_man_switch": {
            "enabled": True,
            "timeout_seconds": 900,
            "check_interval_seconds": 0.01,
            "flatten_mode": "market",
            "log_path": str(tmp_path / "dms.jsonl"),
        }
    }

    call_count = 0

    def raising_check_liveness(path, *, max_age_seconds):  # noqa: ARG001
        nonlocal call_count
        call_count += 1
        if call_count >= 5:
            # Stop after enough failures to trigger escalation.
            stop_event.set()
        raise RuntimeError("simulated liveness check failure")

    with (
        patch(
            "src.assembled_core.ops.dead_man_switch.check_liveness",
            side_effect=raising_check_liveness,
        ),
        patch(
            "src.assembled_core.ops.dead_man_switch.auto_flatten_on_stale"
        ) as mock_flatten,
        caplog.at_level("CRITICAL", logger="src.assembled_core.ops.dead_man_switch"),
    ):
        dms_monitor_loop(policy, stop_event=stop_event)

    assert any("DMS-CRITICAL" in r.message for r in caplog.records), (
        "Expected [DMS-CRITICAL] log after consecutive failures"
    )
    mock_flatten.assert_called()
    assert mock_flatten.call_count == 1, (
        f"Expected exactly one escalation call, got {mock_flatten.call_count} "
        f"— possible kill-switch activation storm."
    )
