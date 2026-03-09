"""Tests for OPS-13 intel activity summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.assembled_core.ops.intel_activity_summary import (
    SCHEMA_VERSION,
    build_intel_activity_summary,
)


pytestmark = [pytest.mark.unit, pytest.mark.phase6]


def _write_kpis(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_kpis.json").write_text(json.dumps(payload), encoding="utf-8")


def test_build_intel_activity_summary_synthetic(tmp_path: Path) -> None:
    """Synthetic run dirs with minimal run_kpis; assert schema and counts."""
    # Day 1: news OK, discl OK, WATCH, geo_score 0
    d1 = tmp_path / "2025-06-26"
    _write_kpis(d1, {
        "intel_orchestration": {"news": {"ran": True, "status": "OK"}, "disclosures": {"ran": True, "status": "OK"}},
        "risk_state": {"state": "WATCH", "reason": "watch_hold"},
        "news_geo": {"geo_score": 0, "geo_confidence": 0.0},
        "triggers_summary": {},
        "top_triggers": [],
    })
    # Day 2: news OK, geo_score 2, has triggers
    d2 = tmp_path / "2025-06-27"
    _write_kpis(d2, {
        "intel_orchestration": {"news": {"ran": True, "status": "OK"}, "disclosures": {"ran": True, "status": "DEGRADED"}},
        "risk_state": {"state": "WATCH", "reason": "watch_hold"},
        "news_geo": {"geo_score": 2, "geo_confidence": 0.8},
        "triggers_summary": {"max_severity": 1},
        "top_triggers": [{"severity": 1}, {"severity": 2}],
    })
    # Day 3: news ERROR, ACTIVE
    d3 = tmp_path / "2025-06-30"
    _write_kpis(d3, {
        "intel_orchestration": {"news": {"ran": True, "status": "ERROR"}, "disclosures": {"ran": True, "status": "OK"}},
        "risk_state": {"state": "ACTIVE", "reason": "activate"},
        "news_geo": {"geo_score": 3, "geo_confidence": 0.9},
        "triggers_summary": {"max_severity": 2},
        "top_triggers": [{"severity": 2}],
    })

    summary = build_intel_activity_summary(tmp_path)

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["n_days"] == 3

    news = summary["news"]
    assert news["days_ok"] == 2
    assert news["days_degraded"] == 0
    assert news["days_error"] == 1
    assert news["days_with_triggers"] == 2  # day 2 and day 3 have top_triggers
    assert news["max_trigger_severity_seen"] == 2
    assert news["days_geo_score_ge_1"] == 2
    assert news["days_geo_score_ge_2"] == 2
    assert news["days_geo_score_ge_3"] == 1

    discl = summary["disclosures"]
    assert discl["days_ok"] == 2
    assert discl["days_degraded"] == 1
    assert discl["days_error"] == 0

    risk = summary["risk_state"]
    assert risk["WATCH"] == 2
    assert risk["ACTIVE"] == 1
    assert risk["COOLDOWN"] == 0
    assert risk["PAUSE"] == 0


def test_build_intel_activity_summary_prefers_per_run_summaries(tmp_path: Path) -> None:
    """OPS-14: Prefer news_triggers_summary and disclosures_triggers_summary over triggers_summary/top_triggers."""
    # Day with only per-run summaries (no top_triggers / triggers_summary)
    d1 = tmp_path / "2025-07-01"
    _write_kpis(d1, {
        "intel_orchestration": {"news": {"status": "OK"}, "disclosures": {"status": "OK"}},
        "risk_state": {"state": "WATCH"},
        "news_geo": None,
        "triggers_summary": {},
        "top_triggers": [],
        "news_triggers_summary": {"count": 3, "max_severity": 2, "count_sev1plus": 2, "count_sev2plus": 1},
        "disclosures_triggers_summary": {"count": 5, "max_severity": 1, "count_sev1plus": 5, "count_sev2plus": 0},
    })
    # Day with zero disclosures per-run summary
    d2 = tmp_path / "2025-07-02"
    _write_kpis(d2, {
        "intel_orchestration": {"news": {"status": "OK"}, "disclosures": {"status": "OK"}},
        "risk_state": {"state": "WATCH"},
        "news_geo": None,
        "triggers_summary": {},
        "top_triggers": [],
        "news_triggers_summary": {"count": 0, "max_severity": 0, "count_sev1plus": 0, "count_sev2plus": 0},
        "disclosures_triggers_summary": {"count": 0, "max_severity": 0, "count_sev1plus": 0, "count_sev2plus": 0},
    })

    summary = build_intel_activity_summary(tmp_path)
    assert summary["n_days"] == 2
    assert summary["news"]["days_with_triggers"] == 1  # only day 1 has count > 0
    assert summary["news"]["max_trigger_severity_seen"] == 2
    assert summary["disclosures"]["days_with_triggers"] == 1
    assert summary["disclosures"]["max_trigger_severity_seen"] == 1


def test_build_intel_activity_summary_geo_score_thresholds(tmp_path: Path) -> None:
    """Assert days_geo_score_ge_1/2/3 counts."""
    for i, (date, geo_score) in enumerate([("2025-07-01", 0), ("2025-07-02", 1), ("2025-07-03", 2), ("2025-07-04", 3)]):
        d = tmp_path / date
        _write_kpis(d, {
            "intel_orchestration": {"news": {"status": "OK"}, "disclosures": {"status": "OK"}},
            "risk_state": {"state": "WATCH"},
            "news_geo": {"geo_score": geo_score},
            "triggers_summary": {},
            "top_triggers": [],
        })

    summary = build_intel_activity_summary(tmp_path)
    assert summary["n_days"] == 4
    assert summary["news"]["days_geo_score_ge_1"] == 3
    assert summary["news"]["days_geo_score_ge_2"] == 2
    assert summary["news"]["days_geo_score_ge_3"] == 1


def test_build_intel_activity_summary_empty_runs_root(tmp_path: Path) -> None:
    """Empty runs root yields zero counts."""
    summary = build_intel_activity_summary(tmp_path)
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["n_days"] == 0
    assert summary["news"]["days_ok"] == 0
    assert summary["news"]["days_geo_score_ge_1"] == 0
    assert summary["risk_state"]["WATCH"] == 0


def test_build_intel_activity_summary_aggregates_news_funnel(tmp_path: Path) -> None:
    """NEWS-DEBUG-1: When run_kpis contain news_debug_funnel, summary aggregates into news.news_funnel."""
    d1 = tmp_path / "2025-07-01"
    _write_kpis(d1, {
        "intel_orchestration": {"news": {"status": "OK"}, "disclosures": {"status": "OK"}},
        "risk_state": {"state": "WATCH"},
        "news_geo": None,
        "news_debug_funnel": {
            "raw_items_count": 10,
            "normalized_events_count": 9,
            "deduped_events_count": 8,
            "clusters_count": 3,
            "candidate_triggers_count": 2,
            "triggers_count": 1,
            "triggers_severity_ge_1_count": 1,
            "triggers_severity_ge_2_count": 0,
            "triggers_evidence_blocked_count": 0,
            "triggers_qc_capped_count": 0,
        },
    })
    d2 = tmp_path / "2025-07-02"
    _write_kpis(d2, {
        "intel_orchestration": {"news": {"status": "OK"}, "disclosures": {"status": "OK"}},
        "risk_state": {"state": "WATCH"},
        "news_geo": None,
        "news_debug_funnel": {
            "raw_items_count": 5,
            "normalized_events_count": 5,
            "deduped_events_count": 4,
            "clusters_count": 5,
            "candidate_triggers_count": 3,
            "triggers_count": 2,
            "triggers_severity_ge_1_count": 2,
            "triggers_severity_ge_2_count": 1,
            "triggers_evidence_blocked_count": 1,
            "triggers_qc_capped_count": 1,
        },
    })

    summary = build_intel_activity_summary(tmp_path)
    funnel = summary["news"]["news_funnel"]
    assert funnel["total_candidate_triggers"] == 5
    assert funnel["total_triggers"] == 3
    assert funnel["total_triggers_sev1plus"] == 3
    assert funnel["total_triggers_evidence_blocked"] == 1
    assert funnel["total_triggers_qc_capped"] == 1
    assert funnel["max_clusters_count_seen"] == 5


def test_summarize_intel_activity_cli_writes_file(tmp_path: Path) -> None:
    """CLI summarize_intel_activity writes intel_activity_summary.json."""
    from unittest.mock import patch

    from scripts.cli import summarize_intel_activity_subcommand

    exp_name = "test_exp"
    output_runs = tmp_path / "output" / "runs"
    output_runs.mkdir(parents=True)
    exp_root_real = output_runs / "_experiments" / exp_name
    exp_root_real.mkdir(parents=True)
    (exp_root_real / "runs" / "2025-10-31").mkdir(parents=True)
    (exp_root_real / "runs" / "2025-10-31" / "run_kpis.json").write_text(
        json.dumps({
            "intel_orchestration": {"news": {"status": "OK"}, "disclosures": {"status": "OK"}},
            "risk_state": {"state": "WATCH"},
            "news_geo": None,
        }),
        encoding="utf-8",
    )
    args = type("Args", (), {"experiment": exp_name, "output_root": output_runs})()
    with patch("scripts.cli.ROOT", tmp_path):
        code = summarize_intel_activity_subcommand(args)
    assert code == 0
    out_path = output_runs / "_experiments" / exp_name / "intel_activity_summary.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "run.intel_activity.v1"
    assert data["n_days"] == 1
    assert data["news"]["days_ok"] == 1
    assert data["risk_state"]["WATCH"] == 1
