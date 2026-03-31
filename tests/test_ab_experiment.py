"""Tests for A/B experiment runner (summary + compare)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_ab_experiment import (
    build_summary_from_run,
    compare_summaries,
    run_ab_experiment,
)

pytestmark = [pytest.mark.unit]


def _write_run_summary(run_dir, per_day, intel_orchestration=None, intel_summary=None):
    """Write a minimal paper_track_run_summary.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": "test_run",
        "strategy_name": "test_strategy",
        "days_attempted": len(per_day),
        "days_successful": sum(1 for d in per_day if d["status"] == "success"),
        "days_failed": sum(1 for d in per_day if d["status"] == "error"),
        "days_skipped": 0,
        "per_day_statuses": per_day,
        "date_range": {
            "start": per_day[0]["date"] if per_day else None,
            "end": per_day[-1]["date"] if per_day else None,
        },
        "intel_orchestration": intel_orchestration or {"mode": "none"},
    }
    with open(run_dir / "paper_track_run_summary.json", "w") as f:
        json.dump(summary, f)

    if intel_summary:
        with open(run_dir / "intel_summary.json", "w") as f:
            json.dump(intel_summary, f)


class TestBuildSummary:
    def test_summary_from_two_days(self, tmp_path):
        run_dir = tmp_path / "run1"
        per_day = [
            {"date": "2025-10-16", "status": "success", "equity": 100000.0,
             "cash": 90000.0, "daily_return_pct": 0.0, "daily_pnl": 0.0,
             "trades_count": 2, "buy_count": 2, "sell_count": 0, "error_message": None},
            {"date": "2025-10-17", "status": "success", "equity": 100500.0,
             "cash": 89500.0, "daily_return_pct": 0.5, "daily_pnl": 500.0,
             "trades_count": 1, "buy_count": 0, "sell_count": 1, "error_message": None},
        ]
        _write_run_summary(run_dir, per_day)

        summary = build_summary_from_run(run_dir)
        assert summary["schema_version"] == "paper.track.summary.v1"
        assert summary["n_days"] == 2
        assert summary["equity"]["start"] == 100000.0
        assert summary["equity"]["end"] == 100500.0
        assert summary["equity"]["total_return"] == pytest.approx(0.005, abs=1e-6)
        assert summary["trading"]["total_trades"] == 3

    def test_summary_no_success_days(self, tmp_path):
        run_dir = tmp_path / "run_fail"
        per_day = [
            {"date": "2025-10-16", "status": "error", "equity": None,
             "cash": None, "daily_return_pct": None, "daily_pnl": None,
             "trades_count": None, "buy_count": None, "sell_count": None,
             "error_message": "data missing"},
        ]
        _write_run_summary(run_dir, per_day)

        summary = build_summary_from_run(run_dir)
        assert summary["n_days"] == 0
        assert summary["equity"]["total_return"] == 0

    def test_summary_with_intel(self, tmp_path):
        run_dir = tmp_path / "run_intel"
        per_day = [
            {"date": "2025-10-16", "status": "success", "equity": 100000.0,
             "cash": 95000.0, "daily_return_pct": -0.3, "daily_pnl": -300.0,
             "trades_count": 3, "buy_count": 1, "sell_count": 2, "error_message": None},
        ]
        intel_orch = {"mode": "real", "georisk_gate": {"enabled": True, "multiplier_applied": 0.70, "state_hint": "ACTIVE"}}
        intel_summary = {
            "schema_version": "paper.intel_summary.v1",
            "intel_orchestration": intel_orch,
            "georisk_gate": {"enabled": True, "multiplier_applied": 0.70, "state_hint": "ACTIVE"},
        }
        _write_run_summary(run_dir, per_day, intel_orchestration=intel_orch, intel_summary=intel_summary)

        summary = build_summary_from_run(run_dir)
        assert summary["intel"]["mode"] == "real"
        assert summary["intel"]["avg_multiplier_applied"] == 0.70
        assert summary["intel"]["active_pct"] == 1.0

    def test_summary_max_drawdown(self, tmp_path):
        run_dir = tmp_path / "run_dd"
        per_day = [
            {"date": "2025-10-16", "status": "success", "equity": 100000.0,
             "cash": 90000.0, "daily_return_pct": 2.0, "daily_pnl": 2000.0,
             "trades_count": 1, "buy_count": 1, "sell_count": 0, "error_message": None},
            {"date": "2025-10-17", "status": "success", "equity": 98000.0,
             "cash": 88000.0, "daily_return_pct": -4.0, "daily_pnl": -4000.0,
             "trades_count": 0, "buy_count": 0, "sell_count": 0, "error_message": None},
        ]
        _write_run_summary(run_dir, per_day)

        summary = build_summary_from_run(run_dir)
        assert summary["equity"]["max_drawdown"] < 0


class TestCompareSummaries:
    def test_delta_computation(self):
        sa = {
            "run_name": "gate_off",
            "n_days": 5,
            "equity": {"total_return": 0.02, "max_drawdown": -0.05},
            "trading": {"total_trades": 10},
            "intel": {"avg_multiplier_applied": 1.0, "active_pct": 0.0, "watch_pct": 1.0},
        }
        sb = {
            "run_name": "gate_on",
            "n_days": 5,
            "equity": {"total_return": 0.015, "max_drawdown": -0.03},
            "trading": {"total_trades": 7},
            "intel": {"avg_multiplier_applied": 0.7, "active_pct": 0.6, "watch_pct": 0.4},
        }

        comp = compare_summaries(sa, sb)
        assert comp["schema_version"] == "paper.track.compare.v1"
        assert comp["delta"]["total_return"] == pytest.approx(-0.005, abs=1e-6)
        assert comp["delta"]["max_drawdown"] == pytest.approx(0.02, abs=1e-6)
        assert comp["delta"]["avg_multiplier_applied"] == pytest.approx(-0.3, abs=1e-4)
        assert comp["delta"]["active_pct"] == pytest.approx(0.6, abs=1e-4)
        assert comp["delta"]["total_trades"] == -3

    def test_compare_symmetric_zero(self):
        sa = {
            "run_name": "same",
            "n_days": 3,
            "equity": {"total_return": 0.01, "max_drawdown": -0.02},
            "trading": {"total_trades": 5},
            "intel": {"avg_multiplier_applied": 1.0, "active_pct": 0.0, "watch_pct": 1.0},
        }
        comp = compare_summaries(sa, sa)
        assert comp["delta"]["total_return"] == 0.0
        assert comp["delta"]["max_drawdown"] == 0.0
        assert comp["delta"]["total_trades"] == 0


def test_run_ab_experiment_returns_nonzero_when_any_arm_fails(tmp_path, monkeypatch):
    import scripts.run_paper_track as rpt

    config_file = Path("configs/paper_track/trend_baseline.yaml")
    assert config_file.exists()

    monkeypatch.setattr(rpt, "run_paper_track_from_cli", lambda **kwargs: 1)
    code = run_ab_experiment(
        config_file=config_file,
        start_date="2025-01-01",
        end_date="2025-01-01",
        output_root=tmp_path / "ab_out",
        rerun=True,
    )
    assert code == 1
