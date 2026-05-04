"""Tests for OPS-6 paper summary builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.assembled_core.ops.paper_summary import (
    build_paper_summary,
    write_paper_summary,
)

pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def test_build_paper_summary_from_synthetic_artifacts(tmp_path: Path) -> None:
    """Create tmp dirs for 3 days with minimal run_kpis + ledger_state equity_curve + alerts; assert summary keys and counts."""
    output_root = tmp_path / "runs"
    dates = ["2026-02-01", "2026-02-02", "2026-02-03"]
    for i, d in enumerate(dates):
        day_dir = output_root / d
        day_dir.mkdir(parents=True)
        # run_kpis.json
        kpis: dict[str, Any] = {
            "schema_version": "run.kpis.v1",
            "generated_utc": f"2026-02-0{i+1}T12:00:00+00:00",
            "multipliers": {
                "final_exposure_multiplier": 0.9 + i * 0.05,
                "turnover_scale_factor": 0.85,
            },
            "risk_state": {
                "state": "ACTIVE" if i != 1 else "COOLDOWN",
                "reason": (
                    "activate_score"
                    if i == 0
                    else ("cooldown_timer" if i == 1 else "cooldown_to_active")
                ),
            },
        }
        (day_dir / "run_kpis.json").write_text(json.dumps(kpis), encoding="utf-8")
        # ledger_state.json with equity_curve (one point per day)
        equity = 10000.0 + i * 100.0
        ledger = {
            "schema_version": "paper.ledger_state.v1",
            "cash": 10000 - i * 50,
            "positions": {},
            "equity_curve": [
                {"utc": f"2026-02-0{i+1}T12:00:00+00:00", "equity": equity}
            ],
        }
        (day_dir / "ledger_state.json").write_text(json.dumps(ledger), encoding="utf-8")
        # alerts_latest.json
        alerts = {
            "schema_version": "run.alerts.v1",
            "count": 1,
            "items": [
                {
                    "alert_id": "abc",
                    "level": "info" if i == 0 else "warn",
                    "kind": "NO_PREV" if i == 0 else "QC_DEGRADED",
                },
            ],
        }
        (day_dir / "alerts_latest.json").write_text(
            json.dumps(alerts), encoding="utf-8"
        )

    summary = build_paper_summary(output_root, dates)
    assert summary["schema_version"] == "paper.summary.v1"
    assert summary["start_date"] == "2026-02-01"
    assert summary["end_date"] == "2026-02-03"
    assert summary["n_dates"] == 3
    assert summary["total_return"] is not None
    assert summary["max_drawdown"] is not None
    assert summary["risk_state_transitions"] == 2  # ACTIVE->COOLDOWN, COOLDOWN->ACTIVE
    assert "alerts_count_by_level" in summary
    assert "alerts_count_by_kind" in summary
    assert summary["alerts_count_by_level"].get("info", 0) >= 1
    assert summary["alerts_count_by_kind"].get("NO_PREV", 0) >= 1
    assert summary["avg_final_multiplier"] is not None
    assert summary["avg_turnover_scale_factor"] is not None
    assert len(summary["equity_curve_dates"]) == 3
    # OPS-9: state distribution and reason counts
    assert "risk_state_distribution" in summary
    assert summary["risk_state_distribution"].get("ACTIVE") == 2
    assert summary["risk_state_distribution"].get("COOLDOWN") == 1
    assert "risk_state_reason_counts" in summary
    assert summary["risk_state_reason_counts"].get("activate_score") == 1
    assert summary["risk_state_reason_counts"].get("cooldown_timer") == 1
    assert summary["risk_state_reason_counts"].get("cooldown_to_active") == 1
    assert "risk_state_pct" in summary
    assert summary["risk_state_pct"].get("ACTIVE") == pytest.approx(0.6667, rel=1e-3)
    assert summary["risk_state_pct"].get("COOLDOWN") == pytest.approx(0.3333, rel=1e-3)

    path = write_paper_summary(output_root, "2026-02-01", "2026-02-03", summary)
    assert path.exists()
    assert "_summaries" in str(path)
    assert "paper_summary_2026-02-01_2026-02-03.json" == path.name
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_dates"] == 3
    assert data["risk_state_transitions"] == 2
