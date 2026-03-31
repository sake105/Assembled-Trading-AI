"""Targeted tests for BBC + GDELT activation in NEWS v1 pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.assembled_core.events.news.sources import (
    load_sources_registry,
    load_news_params,
)
from src.assembled_core.events.news import run_news_pipeline

pytestmark = [pytest.mark.phase6, pytest.mark.unit]


# ---------------------------------------------------------------------------
# A) Config-level tests
# ---------------------------------------------------------------------------


def test_production_sources_bbc_and_gdelt_active():
    """BBC RSS and GDELT are active in the shipped sources.yaml."""
    sources = load_sources_registry("configs/news/sources.yaml")
    active = {s.source_id: s for s in sources if s.active}
    assert "rss_bbc_world" in active, "BBC RSS must be active"
    assert "gdelt_default" in active, "GDELT must be active"
    assert active["rss_bbc_world"].type == "rss"
    assert active["gdelt_default"].type == "gdelt"


def test_production_sources_no_placeholder_active():
    """No example.com placeholder source should be active."""
    sources = load_sources_registry("configs/news/sources.yaml")
    for s in sources:
        if s.active:
            assert (
                "example.com" not in s.domain
            ), f"Placeholder source {s.source_id} with domain {s.domain} must not be active"


def test_production_gdelt_query_is_broad():
    """GDELT query must contain multiple geopolitical keywords."""
    sources = load_sources_registry("configs/news/sources.yaml")
    gdelt = next((s for s in sources if s.source_id == "gdelt_default"), None)
    assert gdelt is not None
    query = str(gdelt.config.get("query", ""))
    for keyword in ["Red Sea", "Suez", "Taiwan"]:
        assert keyword in query, f"GDELT query should contain '{keyword}'"


def test_news_yaml_gdelt_enabled():
    """gdelt.enabled must be true in shipped news.yaml."""
    params = load_news_params("configs/news/news.yaml")
    assert params["gdelt"]["enabled"] is True


def test_bbc_and_gdelt_types_compatible_with_pipeline():
    """RSS type='rss' and GDELT type='gdelt' are both handled by _collect_raw_items."""
    sources = load_sources_registry("configs/news/sources.yaml")
    active = [s for s in sources if s.active]
    types = {s.type for s in active}
    assert "rss" in types
    assert "gdelt" in types


# ---------------------------------------------------------------------------
# B) Pipeline-level tests (mocked, no network)
# ---------------------------------------------------------------------------


def _fake_rss_fetcher(
    source_id,
    url,
    *,
    timeout,
    user_agent,
    sanitize_cfg,
    fetch_state,
    retries,
    backoff_base_s,
):
    items = [
        {
            "title": "BBC: Shipping disruptions in Red Sea escalate costs",
            "link": f"https://bbc.co.uk/news/{source_id}/article1",
            "published": "2025-01-15T10:00:00Z",
            "summary": "Red Sea shipping route disruptions push freight costs higher.",
            "raw": {},
        },
    ]
    stats = {
        "source_id": source_id,
        "type": "rss",
        "ok": True,
        "http_status": 200,
        "duration_ms": 42,
        "items": len(items),
        "not_modified": False,
        "cached": False,
        "error": None,
    }
    return items, None, stats


def _fake_gdelt_fetcher(source_id, query, *, gdelt_cfg, cadence, fetch_state):
    items = [
        {
            "title": "GDELT: US sanctions pressure on Russia intensifies",
            "link": f"https://gdelt.example.org/{source_id}/ev1",
            "published": "2025-01-15T11:00:00Z",
            "summary": None,
            "raw": {"sourceCountryCode": "US"},
        },
        {
            "title": "GDELT: Strait of Hormuz tensions raise oil prices",
            "link": f"https://gdelt.example.org/{source_id}/ev2",
            "published": "2025-01-15T12:00:00Z",
            "summary": "Geopolitical tensions near the Strait of Hormuz affect markets.",
            "raw": {},
        },
    ]
    stats = {
        "source_id": source_id,
        "type": "gdelt",
        "ok": True,
        "http_status": 200,
        "duration_ms": 150,
        "items": len(items),
        "not_modified": False,
        "cached": False,
        "error": None,
    }
    return items, None, stats


def _write_test_configs(tmp_path: Path) -> tuple[Path, Path]:
    """Write sources + news config with BBC + GDELT active."""
    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_bbc_world"
    name: "BBC World"
    domain: "bbc.co.uk"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://feeds.bbci.co.uk/news/world/rss.xml"

  - source_id: "gdelt_default"
    name: "GDELT"
    domain: "gdeltproject.org"
    type: "gdelt"
    tier: "B"
    weight: 0.6
    active: true
    query: "war OR sanctions OR shipping OR Red Sea"
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
  user_agent: "Assembled-Trading-AI/NEWS-v1"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: false
gdelt:
  enabled: true
  cache_minutes: 10
  stale_on_error_minutes: 60
  window_hours:
    hourly: 1
    daily: 6
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )
    return sources_cfg, news_cfg


def test_gdelt_source_processed_in_pipeline(tmp_path, monkeypatch):
    """Active GDELT source is fetched and its items appear in the result."""
    import src.assembled_core.events.news.pipeline as pm

    monkeypatch.setattr(pm, "fetch_rss_feed", _fake_rss_fetcher)
    monkeypatch.setattr(pm, "fetch_gdelt_events", _fake_gdelt_fetcher)

    sources_cfg, news_cfg = _write_test_configs(tmp_path)
    out_dir = tmp_path / "out"

    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    events = result["events"]
    sources_seen = {e.source_id for e in events}
    assert (
        "gdelt_default" in sources_seen
    ), "GDELT events must appear in pipeline output"
    assert "rss_bbc_world" in sources_seen, "BBC events must appear in pipeline output"
    assert len(events) == 3


def test_fetch_report_contains_gdelt_entry(tmp_path, monkeypatch):
    """fetch_report per_source must list a GDELT entry."""
    import src.assembled_core.events.news.pipeline as pm

    monkeypatch.setattr(pm, "fetch_rss_feed", _fake_rss_fetcher)
    monkeypatch.setattr(pm, "fetch_gdelt_events", _fake_gdelt_fetcher)

    sources_cfg, news_cfg = _write_test_configs(tmp_path)
    out_dir = tmp_path / "out"

    run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    report = json.loads(
        (out_dir / "fetch_report_latest.json").read_text(encoding="utf-8")
    )
    per_source = report["per_source"]
    gdelt_entries = [s for s in per_source if s.get("type") == "gdelt"]
    rss_entries = [s for s in per_source if s.get("type") == "rss"]
    assert len(gdelt_entries) == 1, "Exactly one GDELT entry expected in per_source"
    assert gdelt_entries[0]["source_id"] == "gdelt_default"
    assert gdelt_entries[0]["ok"] is True
    assert len(rss_entries) == 1
    assert rss_entries[0]["source_id"] == "rss_bbc_world"


def test_bbc_and_gdelt_no_type_mismatch(tmp_path, monkeypatch):
    """Both RSS and GDELT sources produce valid events without type/config errors."""
    import src.assembled_core.events.news.pipeline as pm

    monkeypatch.setattr(pm, "fetch_rss_feed", _fake_rss_fetcher)
    monkeypatch.setattr(pm, "fetch_gdelt_events", _fake_gdelt_fetcher)

    sources_cfg, news_cfg = _write_test_configs(tmp_path)
    out_dir = tmp_path / "out"

    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    health = result["health"]
    assert health.status in ("OK", "DEGRADED"), f"Unexpected health: {health.status}"
    for ev in result["events"]:
        assert ev.title, "Every event must have a title"
        assert ev.event_id.startswith("news_"), "event_id format check"


# ---------------------------------------------------------------------------
# C) Dedupe store isolation
# ---------------------------------------------------------------------------


def test_output_dir_isolation_dedupe_store(tmp_path, monkeypatch):
    """When output_dir is specified, dedupe_store.sqlite lives inside it, not globally."""
    import src.assembled_core.events.news.pipeline as pm

    monkeypatch.setattr(pm, "fetch_rss_feed", _fake_rss_fetcher)
    monkeypatch.setattr(pm, "fetch_gdelt_events", _fake_gdelt_fetcher)

    sources_cfg, news_cfg = _write_test_configs(tmp_path)

    # Enable dedupe with a global-looking config path
    news_cfg.write_text(
        news_cfg.read_text(encoding="utf-8").replace(
            "dedupe:\n  enabled: false",
            "dedupe:\n  enabled: true\n  window_days: 14\n  store:\n    backend: sqlite\n    path: output/intel/news/cache/dedupe_store.sqlite",
        ),
        encoding="utf-8",
    )

    experiment_dir = tmp_path / "experiments" / "smoke_test" / "intel" / "news"

    # Use a sentinel directory that definitely doesn't exist yet
    sentinel_global = (
        tmp_path / "global_output" / "intel" / "news" / "cache" / "dedupe_store.sqlite"
    )
    assert not sentinel_global.exists()

    run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=experiment_dir,
    )

    expected_db = experiment_dir / "cache" / "dedupe_store.sqlite"
    assert (
        expected_db.exists()
    ), "dedupe_store.sqlite must be in experiment-specific dir"
    assert (
        not sentinel_global.exists()
    ), "No global db should be created for unrelated path"


def test_fresh_experiment_gets_fresh_output(tmp_path, monkeypatch):
    """Two different experiment names produce independent output directories."""
    import src.assembled_core.events.news.pipeline as pm

    monkeypatch.setattr(pm, "fetch_rss_feed", _fake_rss_fetcher)
    monkeypatch.setattr(pm, "fetch_gdelt_events", _fake_gdelt_fetcher)

    sources_cfg, news_cfg = _write_test_configs(tmp_path)

    for name in ["exp_a", "exp_b"]:
        out = tmp_path / name
        run_news_pipeline(
            sources_path=str(sources_cfg),
            news_path=str(news_cfg),
            cadence="hourly",
            output_dir=out,
        )
        assert (out / "events_latest.json").exists()
        assert (out / "fetch_report_latest.json").exists()

    data_a = json.loads((tmp_path / "exp_a" / "events_latest.json").read_text())
    data_b = json.loads((tmp_path / "exp_b" / "events_latest.json").read_text())
    assert data_a["count"] == data_b["count"] == 3
