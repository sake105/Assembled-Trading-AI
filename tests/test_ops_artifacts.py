"""Tests for OPS-2 artifacts: targets, orders, diff_vs_prev, reasons."""

from __future__ import annotations

import json
from types import SimpleNamespace
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.assembled_core.ops.kpi_artifacts import (
    write_diff_vs_prev,
    write_reasons_artifact,
    write_targets_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def test_targets_artifact_written(tmp_path) -> None:
    """targets_latest.json should contain symbols and weights."""
    df = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "target_weight": [0.6, 0.4],
            "target_qty": [60.0, 40.0],
        }
    )
    out_dir = tmp_path / "run_curr"
    path = write_targets_artifact(out_dir, df)
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "run.targets.v1"
    items = data["items"]
    assert len(items) == 2
    symbols = {it["symbol"] for it in items}
    assert symbols == {"A", "B"}
    weights = {it["symbol"]: it["target_weight"] for it in items}
    assert pytest.approx(weights["A"]) == 0.6
    assert pytest.approx(weights["B"]) == 0.4


def test_diff_vs_prev_no_prev(tmp_path) -> None:
    """When prev_dir is missing, diff_vs_prev should record a note and empty deltas."""
    curr_dir = tmp_path / "run_curr"
    curr_dir.mkdir(parents=True, exist_ok=True)

    # Minimal current targets and kpis
    current_targets = pd.DataFrame(
        {
            "symbol": ["A"],
            "target_weight": [1.0],
        }
    )
    current_kpis = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "multipliers": {
            "georisk": 1.0,
            "profit_lock": 1.0,
            "final_exposure_multiplier": 1.0,
            "turnover_scale_factor": 1.0,
        },
        "risk_state": {"state": "ACTIVE"},
    }

    prev_dir = tmp_path / "run_prev_missing"
    path = write_diff_vs_prev(curr_dir, prev_dir, current_targets, current_kpis)
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "run.diff.v1"
    assert "no_prev_run_found" in data["notes"]
    # No previous targets -> deltas based only on current; summary still valid
    assert data["summary"]["n_symbols_changed"] == 1


def test_diff_vs_prev_computes_deltas(tmp_path) -> None:
    """Diff artifact should compute per-symbol and multiplier deltas."""
    # Previous run artifacts
    prev_dir = tmp_path / "run_prev"
    prev_dir.mkdir(parents=True, exist_ok=True)

    prev_kpis = {
        "schema_version": "run.kpis.v1",
        "generated_utc": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        "multipliers": {
            "georisk": 0.8,
            "profit_lock": 0.9,
            "final_exposure_multiplier": 0.72,
            "turnover_scale_factor": 0.7,
        },
        "risk_state": {"state": "ACTIVE"},
    }
    (prev_dir / "run_kpis.json").write_text(json.dumps(prev_kpis), encoding="utf-8")

    prev_targets = {
        "schema_version": "run.targets.v1",
        "generated_utc": datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat(),
        "items": [
            {"symbol": "A", "target_weight": 0.5},
            {"symbol": "B", "target_weight": 0.5},
        ],
    }
    (prev_dir / "targets_latest.json").write_text(
        json.dumps(prev_targets), encoding="utf-8"
    )

    # Current run
    curr_dir = tmp_path / "run_curr"
    curr_dir.mkdir(parents=True, exist_ok=True)

    current_targets = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "target_weight": [0.6, 0.4],
        }
    )
    current_kpis = {
        "schema_version": "run.kpis.v1",
        "generated_utc": datetime(2025, 1, 2, tzinfo=timezone.utc).isoformat(),
        "multipliers": {
            "georisk": 0.9,
            "profit_lock": 1.0,
            "final_exposure_multiplier": 0.9,
            "turnover_scale_factor": 0.8,
        },
        "risk_state": {"state": "ACTIVE"},
    }

    path = write_diff_vs_prev(curr_dir, prev_dir, current_targets, current_kpis)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == "run.diff.v1"
    assert data["prev_date"] == "2025-01-01"
    assert data["current_date"] == "2025-01-02"

    # Check multipliers delta
    dm = data["delta_multipliers"]
    assert pytest.approx(dm["georisk"]["delta"]) == 0.1
    assert pytest.approx(dm["turnover_scale_factor"]["delta"]) == 0.1

    # Check target deltas
    deltas = {d["symbol"]: d for d in data["delta_targets"]}
    assert pytest.approx(deltas["A"]["delta_weight"]) == 0.1
    assert pytest.approx(deltas["B"]["delta_weight"]) == -0.1
    # Summary
    assert pytest.approx(data["summary"]["abs_delta_weight_sum"]) == 0.2
    assert data["summary"]["n_symbols_changed"] == 2


def test_write_reasons_artifact_basic(tmp_path) -> None:
    """reasons_latest.json should capture risk_state, geo and turnover info."""
    ctx = SimpleNamespace(
        risk_state={"state": "ACTIVE", "reason": "test_reason"},
        news_geo={"geo_score": 2, "geo_confidence": 0.9, "state_hint": "ACTIVE"},
        market_stress={"stress_ok": True},
        intel_health_flags={"intel_geo_score": "OK"},
    )
    meta = {
        "turnover_budget": {"scale_factor": 0.8, "estimated_turnover": 0.2},
        "profit_lock": {"multiplier": 0.9},
    }
    result = SimpleNamespace(meta=meta)
    policy = {
        "georisk_overlay": {
            "enabled": True,
            "mapping": {"ACTIVE": {"multiplier": 0.7, "hedge": {"enabled": False}}},
            "by_geo_score": {},
            "confidence_floor": 0.6,
        },
        "turnover_budget": {
            "enabled": True,
            "cap": 0.15,
            "behavior": "scale",
        },
    }

    out_dir = tmp_path / "run1"
    path = write_reasons_artifact(
        out_dir, ctx=ctx, result=result, policy=policy, mode="shadow"
    )
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "run.reasons.v1"
    assert data["risk_state"]["state"] == "ACTIVE"
    assert data["geo"]["geo_score"] == 2
    assert data["turnover_gate"]["behavior"] == "scale"
