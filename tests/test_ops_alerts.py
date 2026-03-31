"""Tests for OPS-3 alerts (alerts_latest.json, compute_alerts, write_alerts_artifact)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from src.assembled_core.ops.alerts import compute_alerts, write_alerts_artifact


pytestmark = [pytest.mark.unit, pytest.mark.phase6]

# Default config with alerts enabled
DEFAULT_CFG: dict[str, Any] = {
    "alerts": {
        "enabled": True,
        "thresholds": {
            "final_multiplier_drop": 0.20,
            "abs_delta_weight_sum": 0.20,
            "turnover_scale_factor_below": 0.70,
            "max_severity_ge": 2,
        },
        "severity_map": {"info": 0, "warn": 1, "critical": 2},
    }
}


def test_alerts_no_prev_generates_info() -> None:
    """When diff.notes contains no_prev_run_found, an info-level NO_PREV alert is generated."""
    run_kpis: dict[str, Any] = {
        "generated_utc": "2025-01-15T12:00:00+00:00",
        "multipliers": {},
    }
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {
        "notes": ["no_prev_run_found"],
        "generated_utc": run_kpis["generated_utc"],
    }

    alerts = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    assert len(alerts) >= 1
    no_prev = [a for a in alerts if a["kind"] == "NO_PREV"]
    assert len(no_prev) == 1
    assert no_prev[0]["level"] == "info"
    assert no_prev[0]["kind"] == "NO_PREV"
    assert "alert_id" in no_prev[0]
    assert "No previous run found" in no_prev[0]["message"]


def test_alerts_state_change_generates_warn() -> None:
    """When risk state changes (e.g. ACTIVE -> COOLDOWN), a STATE_CHANGE warn alert is generated."""
    run_kpis: dict[str, Any] = {"generated_utc": "2025-01-15T12:00:00+00:00"}
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {
        "generated_utc": run_kpis["generated_utc"],
        "notes": [],
        "delta_risk_state": {
            "prev": {"state": "ACTIVE"},
            "curr": {"state": "COOLDOWN"},
        },
    }

    alerts = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    state_change = [a for a in alerts if a["kind"] == "STATE_CHANGE"]
    assert len(state_change) == 1
    assert state_change[0]["level"] == "warn"
    assert (
        "ACTIVE" in state_change[0]["message"]
        and "COOLDOWN" in state_change[0]["message"]
    )


def test_alerts_state_change_pause_generates_critical() -> None:
    """When risk state changes to PAUSE, STATE_CHANGE is critical."""
    run_kpis: dict[str, Any] = {"generated_utc": "2025-01-15T12:00:00+00:00"}
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {
        "generated_utc": run_kpis["generated_utc"],
        "notes": [],
        "delta_risk_state": {
            "prev": {"state": "ACTIVE"},
            "curr": {"state": "PAUSE"},
        },
    }

    alerts = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    state_change = [a for a in alerts if a["kind"] == "STATE_CHANGE"]
    assert len(state_change) == 1
    assert state_change[0]["level"] == "critical"


def test_alerts_underinvested_drop() -> None:
    """When final_exposure_multiplier delta <= -threshold, UNDERINVESTED warn is generated."""
    run_kpis: dict[str, Any] = {"generated_utc": "2025-01-15T12:00:00+00:00"}
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {
        "generated_utc": run_kpis["generated_utc"],
        "notes": [],
        "delta_multipliers": {
            "final_exposure_multiplier": {"prev": 0.9, "curr": 0.6, "delta": -0.3},
        },
        "summary": {},
    }

    alerts = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    under = [a for a in alerts if a["kind"] == "UNDERINVESTED"]
    assert len(under) == 1
    assert under[0]["level"] == "warn"
    assert "0.30" in under[0]["message"] or "0.3" in under[0]["message"]


def test_alerts_turnover_gate() -> None:
    """When turnover_budget.scale_factor < threshold or behavior==block, TURNOVER_GATE alert."""
    generated = "2025-01-15T12:00:00+00:00"
    run_kpis: dict[str, Any] = {
        "generated_utc": generated,
        "turnover_budget": {"scale_factor": 0.5, "behavior": "scale"},
    }
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {"generated_utc": generated, "notes": [], "summary": {}}

    alerts = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    gate = [a for a in alerts if a["kind"] == "TURNOVER_GATE"]
    assert len(gate) == 1
    assert gate[0]["level"] == "warn"
    assert "0.5" in gate[0]["message"] or "0.70" in gate[0]["message"]


def test_alerts_turnover_gate_block_critical() -> None:
    """When behavior is block, TURNOVER_GATE is critical."""
    generated = "2025-01-15T12:00:00+00:00"
    run_kpis: dict[str, Any] = {
        "generated_utc": generated,
        "turnover_budget": {"scale_factor": 0.5, "behavior": "block"},
    }
    reasons: dict[str, Any] = {}
    diff: dict[str, Any] = {"generated_utc": generated, "notes": [], "summary": {}}

    alerts = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    gate = [a for a in alerts if a["kind"] == "TURNOVER_GATE"]
    assert len(gate) == 1
    assert gate[0]["level"] == "critical"


def test_write_alerts_artifact(tmp_path) -> None:
    """write_alerts_artifact produces alerts_latest.json with schema run.alerts.v1."""
    alerts = [
        {
            "alert_id": "abc123def456",
            "level": "info",
            "kind": "NO_PREV",
            "message": "No previous run found.",
            "details": {},
        },
    ]
    generated_utc = datetime.now(timezone.utc).isoformat()
    path = write_alerts_artifact(tmp_path, alerts, generated_utc, DEFAULT_CFG)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "run.alerts.v1"
    assert data["generated_utc"] == generated_utc
    assert data["count"] == 1
    assert len(data["items"]) == 1
    assert data["items"][0]["kind"] == "NO_PREV"


def test_alerts_deterministic_ordering() -> None:
    """Alerts are sorted by severity desc, then kind, then alert_id."""
    run_kpis: dict[str, Any] = {
        "generated_utc": "2025-01-15T12:00:00+00:00",
        "turnover_budget": {"scale_factor": 0.5},
    }
    reasons: dict[str, Any] = {"qc_flags": ["flag1"]}
    diff: dict[str, Any] = {
        "notes": ["no_prev_run_found"],
        "generated_utc": run_kpis["generated_utc"],
        "summary": {},
    }

    alerts1 = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    alerts2 = compute_alerts(run_kpis, reasons, diff, DEFAULT_CFG)
    assert [a["alert_id"] for a in alerts1] == [a["alert_id"] for a in alerts2]
    # critical/warn before info
    levels = [a["level"] for a in alerts1]
    if "warn" in levels and "info" in levels:
        assert levels.index("warn") < levels.index("info")
