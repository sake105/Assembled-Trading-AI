"""Tests for NEWS v1 trigger scoring (Phase 4)."""

from __future__ import annotations

import json

import pytest

from src.assembled_core.events.news.models import NewsEvent
from src.assembled_core.events.news.trigger_scoring import score_triggers
from src.assembled_core.events.news.sources import load_news_params
from src.assembled_core.events.news import run_news_pipeline

pytestmark = [pytest.mark.fast, pytest.mark.unit]


def _make_cluster(
    cluster_id: str,
    event_ids: list[str],
    top_entities: list[str],
    top_phrases: list[str],
    sample_titles: list[str],
    countries: list[str] | None = None,
    evidence: dict | None = None,
) -> dict:
    return {
        "cluster_id": cluster_id,
        "event_ids": event_ids,
        "top_entities": top_entities,
        "top_phrases": top_phrases,
        "sample_titles": sample_titles,
        "countries": countries or [],
        "evidence": evidence or {"evidence_ok": True},
    }


def _make_events_by_id(event_ids: list[str]) -> dict[str, NewsEvent]:
    result = {}
    for eid in event_ids:
        result[eid] = NewsEvent(
            event_id=eid,
            source_id="test",
            title="placeholder",
            url=f"https://example.com/{eid}",
            canonical_url=f"https://example.com/{eid}",
            source_name="Test",
            source_domain="example.com",
            published_utc="2025-01-01T00:00:00Z",
            fetched_utc="2025-01-01T00:00:00Z",
        )
    return result


def test_score_triggers_matches_geopolitical_keywords():
    """Cluster with war/sanctions keywords produces geo_risk trigger."""
    cluster = _make_cluster(
        "clu_1",
        ["e1", "e2"],
        top_entities=["US", "RU"],
        top_phrases=["sanctions pressure"],
        sample_titles=["US imposes new sanctions on Russia over conflict"],
        countries=["US", "RU"],
    )
    events_by_id = _make_events_by_id(["e1", "e2"])

    triggers = score_triggers([cluster], events_by_id)
    assert len(triggers) > 0
    topic_ids = {t["topic_id"] for t in triggers}
    assert "sanctions_trade" in topic_ids or "geopolitical_conflict" in topic_ids


def test_score_triggers_shipping_disruption():
    """Cluster with Red Sea/shipping keywords produces supply_chain trigger."""
    cluster = _make_cluster(
        "clu_ship",
        ["e1", "e2"],
        top_entities=["Red Sea"],
        top_phrases=["shipping disruption"],
        sample_titles=["Red Sea shipping route disruptions push costs higher"],
        countries=["EG"],
    )
    events_by_id = _make_events_by_id(["e1", "e2"])

    triggers = score_triggers([cluster], events_by_id)
    topic_ids = {t["topic_id"] for t in triggers}
    assert "shipping_disruption" in topic_ids
    ship_trigger = next(t for t in triggers if t["topic_id"] == "shipping_disruption")
    assert ship_trigger["trigger_type"] == "supply_chain"
    assert ship_trigger["severity"] >= 2


def test_score_triggers_empty_clusters():
    """No clusters → no triggers."""
    triggers = score_triggers([], {})
    assert triggers == []


def test_score_triggers_no_keyword_match():
    """Cluster without any matching keywords → no triggers."""
    cluster = _make_cluster(
        "clu_boring",
        ["e1", "e2"],
        top_entities=[],
        top_phrases=["cooking recipe"],
        sample_titles=["Best cake recipe of the year"],
    )
    triggers = score_triggers([cluster], _make_events_by_id(["e1", "e2"]))
    assert triggers == []


def test_severity_capped_on_degraded():
    """DEGRADED health caps severity to severity_cap_degraded."""
    cluster = _make_cluster(
        "clu_war",
        ["e1", "e2"],
        top_entities=[],
        top_phrases=[],
        sample_titles=["Nuclear threat escalates as war intensifies"],
    )
    triggers = score_triggers(
        [cluster],
        _make_events_by_id(["e1", "e2"]),
        health_status="DEGRADED",
        severity_cap_degraded=1,
    )
    for t in triggers:
        assert t["severity"] <= 1


def test_severity_capped_on_error():
    """ERROR health caps severity to 0."""
    cluster = _make_cluster(
        "clu_war",
        ["e1", "e2"],
        top_entities=[],
        top_phrases=[],
        sample_titles=["Nuclear threat escalates as war intensifies"],
    )
    triggers = score_triggers(
        [cluster],
        _make_events_by_id(["e1", "e2"]),
        health_status="ERROR",
        severity_cap_error=0,
    )
    for t in triggers:
        assert t["severity"] == 0


def test_trigger_fields_complete():
    """Each trigger has all required schema fields."""
    cluster = _make_cluster(
        "clu_oil",
        ["e1", "e2"],
        top_entities=["OPEC"],
        top_phrases=["oil price surge"],
        sample_titles=["Oil prices surge as OPEC cuts supply"],
        countries=["SA"],
    )
    triggers = score_triggers(
        [cluster],
        _make_events_by_id(["e1", "e2"]),
        generated_utc="2025-01-01T00:00:00Z",
    )
    assert len(triggers) >= 1
    required_fields = {
        "trigger_id",
        "cluster_id",
        "trigger_type",
        "topic_id",
        "severity",
        "confidence",
        "keyword_hits",
        "event_count",
        "countries",
        "evidence_ok",
        "sample_title",
        "generated_utc",
    }
    for t in triggers:
        assert required_fields.issubset(
            t.keys()
        ), f"Missing fields: {required_fields - t.keys()}"


def test_triggers_sorted_by_severity_desc():
    """Triggers are sorted by severity descending."""
    clusters = [
        _make_cluster(
            "clu_a",
            ["e1", "e2"],
            top_entities=[],
            top_phrases=[],
            sample_titles=["Central bank cuts interest rate"],
        ),
        _make_cluster(
            "clu_b",
            ["e3", "e4"],
            top_entities=[],
            top_phrases=[],
            sample_titles=["Nuclear threat from conflict zone"],
        ),
    ]
    events_by_id = _make_events_by_id(["e1", "e2", "e3", "e4"])
    triggers = score_triggers(clusters, events_by_id)
    if len(triggers) >= 2:
        for i in range(len(triggers) - 1):
            assert triggers[i]["severity"] >= triggers[i + 1]["severity"]


def test_news_yaml_trigger_scoring_enabled():
    """Production news.yaml has trigger_scoring.enabled = true."""
    params = load_news_params("configs/news/news.yaml")
    assert params["trigger_scoring"]["enabled"] is True


def test_pipeline_produces_triggers_with_mock(tmp_path, monkeypatch):
    """Full pipeline with mocked fetchers produces triggers when scoring is enabled."""
    import src.assembled_core.events.news.pipeline as pm

    def fake_rss(source_id, url, **kwargs):
        items = [
            {
                "title": "Oil prices surge as Red Sea shipping disrupted",
                "link": "https://bbc.co.uk/1",
                "published": "2025-01-15T10:00:00Z",
                "summary": "Freight costs rise after Red Sea attacks.",
                "raw": {},
            },
            {
                "title": "Red Sea shipping crisis deepens freight costs",
                "link": "https://bbc.co.uk/2",
                "published": "2025-01-15T11:00:00Z",
                "summary": "Shipping rerouted around Red Sea.",
                "raw": {},
            },
        ]
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": 2,
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    def fake_gdelt(source_id, query, **kwargs):
        items = [
            {
                "title": "Red Sea shipping route disruptions hit global trade",
                "link": "https://gdelt.org/1",
                "published": "2025-01-15T09:00:00Z",
                "summary": "Shipping disruptions in Red Sea.",
                "raw": {},
            },
        ]
        stats = {
            "source_id": source_id,
            "type": "gdelt",
            "ok": True,
            "http_status": 200,
            "duration_ms": 100,
            "items": 1,
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    monkeypatch.setattr(pm, "fetch_rss_feed", fake_rss)
    monkeypatch.setattr(pm, "fetch_gdelt_events", fake_gdelt)

    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: rss_test
    name: Test RSS
    domain: test.com
    type: rss
    tier: A
    weight: 1.0
    active: true
    url: https://test.com/rss
  - source_id: gdelt_test
    name: GDELT Test
    domain: gdeltproject.org
    type: gdelt
    tier: B
    weight: 0.6
    active: true
    query: "shipping OR Red Sea"
""",
        encoding="utf-8",
    )

    news_cfg = tmp_path / "news.yaml"
    news_cfg.write_text(
        """
fetch:
  timeout_s: 10
  retries: 0
  backoff_base_s: 0.1
  max_concurrency: 2
  user_agent: UA
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: false
clustering:
  enabled: true
  similarity_threshold: 0.15
  min_cluster_size: 2
  top_phrases_k: 8
  top_entities_k: 8
  max_pair_checks: 5000
  require_overlap: false
  same_day_only: false
trigger_scoring:
  enabled: true
  severity_cap_degraded: 1
  severity_cap_error: 0
gdelt:
  enabled: true
health:
  min_sources_ok: 1
""",
        encoding="utf-8",
    )

    out = tmp_path / "out"
    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out,
    )

    triggers_path = out / "triggers_latest.json"
    assert triggers_path.exists()
    data = json.loads(triggers_path.read_text())
    assert data["schema_version"] == "news.triggers.v1"
    assert data["count"] >= 1
    assert any(t["topic_id"] == "shipping_disruption" for t in data["items"])

    health = result["health"]
    assert health.metrics["triggers"]["trigger_count"] >= 1
    assert health.metrics["triggers"]["max_severity"] >= 2
