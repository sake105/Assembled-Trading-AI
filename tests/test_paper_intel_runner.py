"""Tests for paper track intel runner integration."""

from __future__ import annotations

import json

import pytest

from src.assembled_core.paper.intel_runner import (
    build_intel_summary,
    compute_news_geo,
    load_intel_summaries,
    run_real_intel_once,
)

pytestmark = [pytest.mark.unit]


def test_run_real_intel_once_calls_news_pipeline(tmp_path, monkeypatch):
    """Real intel mode runs NEWS pipeline and returns status."""
    captured = {}

    def fake_run_news_pipeline(*, sources_path, news_path, cadence, output_dir):
        captured["called"] = True
        captured["output_dir"] = str(output_dir)

        from src.assembled_core.events.news.models import NewsHealth
        health = NewsHealth(
            status="OK", fetched_utc="2025-01-01T00:00:00Z",
            sources_total=2, sources_ok=2, sources_failed=0,
            items_raw=10, items_after_dedupe=10,
        )
        return {"events": [], "health": health}

    monkeypatch.setattr(
        "src.assembled_core.events.news.run_news_pipeline",
        fake_run_news_pipeline,
    )

    result = run_real_intel_once(output_dir=tmp_path, run_news=True)

    assert captured["called"] is True
    assert result["news"]["ran"] is True
    assert result["news"]["status"] == "OK"
    assert result["disclosures"]["ran"] is False
    assert result["disclosures"]["status"] == "SKIPPED"


def test_run_real_intel_once_handles_failure(tmp_path, monkeypatch):
    """NEWS pipeline failure is caught and returns ERROR."""
    def failing_pipeline(**kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(
        "src.assembled_core.events.news.run_news_pipeline",
        failing_pipeline,
    )

    result = run_real_intel_once(output_dir=tmp_path, run_news=True)
    assert result["news"]["ran"] is True
    assert result["news"]["status"] == "ERROR"


def test_load_intel_summaries_reads_triggers(tmp_path):
    """Load intel summaries from triggers_latest.json."""
    news_dir = tmp_path / "intel" / "news"
    news_dir.mkdir(parents=True)

    triggers = {
        "schema_version": "news.triggers.v1",
        "count": 3,
        "items": [
            {"severity": 3, "confidence": 0.8},
            {"severity": 2, "confidence": 0.6},
            {"severity": 1, "confidence": 0.4},
        ],
    }
    (news_dir / "triggers_latest.json").write_text(json.dumps(triggers))

    summaries = load_intel_summaries(tmp_path)
    news_sum = summaries["news_triggers_summary"]
    assert news_sum["count"] == 3
    assert news_sum["max_severity"] == 3
    assert news_sum["sev1plus"] == 3
    assert news_sum["sev2plus"] == 2


def test_load_intel_summaries_missing_file(tmp_path):
    """Missing triggers file returns empty summary."""
    summaries = load_intel_summaries(tmp_path)
    assert summaries["news_triggers_summary"]["count"] == 0


def test_compute_news_geo_from_triggers(tmp_path):
    """news_geo is derived from NEWS triggers."""
    news_dir = tmp_path / "intel" / "news"
    news_dir.mkdir(parents=True)

    triggers = {
        "schema_version": "news.triggers.v1",
        "count": 2,
        "items": [
            {"trigger_id": "trg_1", "topic_id": "geo_risk", "severity": 3,
             "confidence": 0.85, "sample_title": "War escalates"},
            {"trigger_id": "trg_2", "topic_id": "shipping", "severity": 2,
             "confidence": 0.6, "sample_title": "Shipping disruption"},
        ],
    }
    (news_dir / "triggers_latest.json").write_text(json.dumps(triggers))

    geo = compute_news_geo(tmp_path)
    assert geo["geo_score"] == 3
    assert geo["geo_confidence"] == 0.85
    assert geo["state_hint"] == "ACTIVE"
    assert len(geo["top_triggers"]) == 2


def test_compute_news_geo_no_triggers(tmp_path):
    """No triggers → geo_score=0, WATCH."""
    geo = compute_news_geo(tmp_path)
    assert geo["geo_score"] == 0
    assert geo["state_hint"] == "WATCH"
    assert geo["top_triggers"] == []


def test_compute_news_geo_low_severity(tmp_path):
    """Severity 1 → WATCH."""
    news_dir = tmp_path / "intel" / "news"
    news_dir.mkdir(parents=True)

    triggers = {
        "schema_version": "news.triggers.v1",
        "count": 1,
        "items": [{"severity": 1, "confidence": 0.4, "trigger_id": "t1",
                    "topic_id": "macro", "sample_title": "Rate cut"}],
    }
    (news_dir / "triggers_latest.json").write_text(json.dumps(triggers))

    geo = compute_news_geo(tmp_path)
    assert geo["geo_score"] == 1
    assert geo["state_hint"] == "WATCH"


def test_build_intel_summary_schema():
    """Intel summary has required schema fields."""
    summary = build_intel_summary(
        intel_orchestration={"mode": "real", "news": {"ran": True, "status": "OK"}},
        news_triggers_summary={"count": 5, "max_severity": 2, "sev1plus": 5, "sev2plus": 3},
        disclosures_triggers_summary={"count": 0, "max_severity": 0, "sev1plus": 0, "sev2plus": 0},
        news_geo={"geo_score": 2, "geo_confidence": 0.6, "state_hint": "ACTIVE", "top_triggers": []},
    )
    assert summary["schema_version"] == "paper.intel_summary.v1"
    assert "generated_utc" in summary
    assert summary["intel_orchestration"]["mode"] == "real"
    assert summary["news_triggers_summary"]["count"] == 5
    assert summary["news_geo"]["state_hint"] == "ACTIVE"


def test_runner_writes_intel_summary_with_real_mode(tmp_path, monkeypatch):
    """Runner in intel_mode=real writes intel_summary.json."""
    call_log = []

    def fake_intel(*, output_dir, **kwargs):
        call_log.append("intel_called")
        news_dir = output_dir / "intel" / "news"
        news_dir.mkdir(parents=True, exist_ok=True)
        triggers = {"schema_version": "news.triggers.v1", "count": 1,
                     "items": [{"severity": 2, "confidence": 0.5, "trigger_id": "t1",
                                "topic_id": "shipping_disruption", "sample_title": "Test"}]}
        (news_dir / "triggers_latest.json").write_text(json.dumps(triggers))
        return {"news": {"ran": True, "status": "OK"}, "disclosures": {"ran": False, "status": "SKIPPED"}}

    monkeypatch.setattr(
        "src.assembled_core.paper.intel_runner.run_real_intel_once",
        fake_intel,
    )

    result = fake_intel(output_dir=tmp_path)
    summaries = load_intel_summaries(tmp_path)
    geo = compute_news_geo(tmp_path)

    assert result["news"]["status"] == "OK"
    assert summaries["news_triggers_summary"]["count"] == 1
    assert geo["geo_score"] == 2
    assert geo["state_hint"] == "ACTIVE"
    assert "intel_called" in call_log
