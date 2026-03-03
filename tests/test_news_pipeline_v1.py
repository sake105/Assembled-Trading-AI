"""Tests for NEWS v1 pipeline core logic (no external calls)."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.assembled_core.events.news.dedupe import dedupe_events
from src.assembled_core.events.news.fingerprint import hamming_distance, simhash64
from src.assembled_core.events.news.health import compute_health
from src.assembled_core.events.news.models import NewsEvent
from src.assembled_core.events.news.clustering import build_clusters
from src.assembled_core.events.news.baseline import compute_version_hash, update_baseline
from src.assembled_core.events.news.burst import compute_bursts_for_window
from src.assembled_core.events.news.evidence import summarize_cluster_evidence
from src.assembled_core.events.news.entities import (
    extract_countries,
    extract_entities,
)
from src.assembled_core.events.news.normalize import (
    canonicalize_url,
    normalize_raw_item,
)
from src.assembled_core.events.news.sources import load_sources_registry
from src.assembled_core.events.news.emit import emit_json_artifact
from src.assembled_core.events.news.fetch_rss import fetch_rss_feed
from src.assembled_core.events.news.fetch_gdelt import fetch_gdelt_events
from src.assembled_core.events.news import run_news_pipeline


pytestmark = [pytest.mark.phase6, pytest.mark.unit]


def test_canonicalize_url_strips_tracking_params():
    url = "https://example.com/path?a=1&utm_source=foo&gclid=bar#section"
    canon = canonicalize_url(url)
    assert "utm_source" not in canon
    assert "gclid" not in canon
    assert "#" not in canon
    assert canon.startswith("https://example.com/path")


def test_canonicalize_url_strips_mc_cid_mc_eid_and_fragments():
    url = "https://example.com/a?mc_cid=abc&mc_eid=def&x=1#section"
    canon = canonicalize_url(url)
    assert "mc_cid" not in canon
    assert "mc_eid" not in canon
    assert "#" not in canon
    assert canon == "https://example.com/a?x=1"


def test_canonicalize_url_removes_default_ports():
    url_http = "http://example.com:80/a?utm_source=x"
    url_https = "https://example.com:443/a"
    canon_http = canonicalize_url(url_http)
    canon_https = canonicalize_url(url_https)
    assert canon_http == "http://example.com/a"
    assert canon_https == "https://example.com/a"


def test_canonicalize_url_trailing_slash_normalized():
    url1 = "https://example.com/path/"
    url2 = "https://example.com/path"
    canon1 = canonicalize_url(url1)
    canon2 = canonicalize_url(url2)
    assert canon1 == canon2


def _make_event(
    title: str,
    url: str,
    published: str,
    fetched_utc: str,
) -> NewsEvent:
    raw: Dict[str, Any] = {
        "title": title,
        "link": url,
        "published": published,
        "summary": "",
        "raw": {},
    }
    ev = normalize_raw_item(
        raw,
        source_id="test_source",
        source_name="Test Source",
        source_domain="example.com",
        fetched_utc=fetched_utc,
    )
    assert ev is not None
    return ev


def test_fingerprint_is_deterministic():
    fetched = "2025-01-15T00:00:00+00:00"
    ev1 = _make_event(
        "Sample Title One",
        "https://example.com/article?utm_source=x",
        "2025-01-10T00:00:00Z",
        fetched,
    )
    ev2 = _make_event(
        "Sample Title One",
        "https://example.com/article?utm_medium=y",
        "2025-01-10T00:00:00Z",
        fetched,
    )
    # canonical URL is identical -> fingerprint must be identical
    assert ev1.canonical_url == ev2.canonical_url
    assert ev1.fingerprint == ev2.fingerprint
    assert ev1.event_id == ev2.event_id


def test_dedupe_reduces_duplicates_and_prefers_older_or_longer_summary():
    fetched = "2025-01-15T00:00:00+00:00"
    ev_old = _make_event(
        "Same news title",
        "https://example.com/article",
        "2025-01-01T00:00:00Z",
        fetched,
    )
    ev_new = _make_event(
        "Same news title",
        "https://example.com/article",
        "2025-01-02T00:00:00Z",
        fetched,
    )
    # same canonical_url, ev_old earlier -> keep ev_old
    unique = dedupe_events([ev_old, ev_new])
    assert len(unique) == 1
    assert unique[0].published_utc == ev_old.published_utc


def test_text_normalization_drops_or_fallbacks_short_title():
    fetched = "2025-01-15T00:00:00+00:00"
    # Case 1: short title, good summary -> title derived from summary
    raw = {
        "title": "Hi",
        "link": "https://example.com/a",
        "published": "2025-01-01T00:00:00Z",
        "summary": "This is a longer summary that should become the title.",
        "raw": {},
    }
    ev = normalize_raw_item(
        raw,
        source_id="src",
        source_name="Source",
        source_domain="example.com",
        fetched_utc=fetched,
    )
    assert ev is not None
    assert len(ev.title.split()) >= 2

    # Case 2: short title and short summary -> drop event (None)
    raw2 = {
        "title": "Hi",
        "link": "https://example.com/b",
        "published": "2025-01-01T00:00:00Z",
        "summary": "short",
        "raw": {},
    }
    ev2 = normalize_raw_item(
        raw2,
        source_id="src",
        source_name="Source",
        source_domain="example.com",
        fetched_utc=fetched,
    )
    assert ev2 is None


def test_extract_countries_basic_aliases():
    text = "U.S. imposes new sanctions on Russia over conflict."
    codes = extract_countries(text)
    # Order is deterministic (alphabetical)
    assert sorted(codes) == ["RU", "US"]


def test_extract_entities_high_impact():
    text = "Shipping disruption in Red Sea near Suez Canal blocks global trade."
    ents = extract_entities(text)
    assert "Red Sea" in ents
    assert "Suez Canal" in ents


def test_normalize_uses_gdelt_provided_country_when_present():
    fetched = "2025-01-15T00:00:00+00:00"
    raw = {
        "title": "Market update",
        "link": "https://example.com/market",
        "published": "2025-01-01T00:00:00Z",
        "summary": "Neutral market headline without explicit country mention.",
        "raw": {
            "sourceCountryCode": "DE",
        },
    }
    ev = normalize_raw_item(
        raw,
        source_id="gdelt_source",
        source_name="GDELT",
        source_domain="gdelt.org",
        fetched_utc=fetched,
    )
    assert ev is not None
    assert ev.countries == ["DE"]


def test_simhash64_deterministic_same_input():
    text = "US and Germany discuss new sanctions."
    fp1 = simhash64(text)
    fp2 = simhash64(text)
    assert fp1 == fp2


def test_hamming_distance_basic():
    a = 0b0
    b = 0b1111
    assert hamming_distance(a, b) == 4
    assert hamming_distance(a, a) == 0


def test_simhash64_changes_on_text_change():
    base = "Market rally led by tech stocks in the US."
    changed = "Market selloff led by energy stocks in the US."
    fp_base = simhash64(base)
    fp_changed = simhash64(changed)
    assert fp_base != fp_changed


def test_health_status_transitions():
    failures = [{"source": "s1", "reason": "fail"}]

    # ERROR: no sources ok
    h_error = compute_health([], items_raw=0, items_after_dedupe=0, failures=failures)
    assert h_error.status == "ERROR"

    # DEGRADED: some items but at least one failure
    h_deg_2 = compute_health(["s1", "s2"], items_raw=10, items_after_dedupe=5, failures=failures)
    assert h_deg_2.status == "DEGRADED"

    # OK: at least one source ok, items_after_dedupe > 0, no failures
    h_ok = compute_health(["s1"], items_raw=5, items_after_dedupe=3, failures=[])
    assert h_ok.status == "OK"


def test_health_ok_when_no_new_items_but_sources_ok():
    # No failures, one source ok, but 0 items_after_dedupe -> OK with note "no_new_items"
    h = compute_health(
        ["s1"],
        items_raw=5,
        items_after_dedupe=0,
        failures=[],
        min_sources_ok=1,
    )
    assert h.status == "OK"
    assert "no_new_items" in h.notes


def test_sources_registry_loads_and_filters_active(tmp_path):
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(
        """
sources:
  - source_id: "rss_active"
    name: "Active Source"
    domain: "active.example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://active.example.com/rss"

  - source_id: "rss_inactive"
    name: "Inactive Source"
    domain: "inactive.example.com"
    type: "rss"
    tier: "B"
    weight: 0.6
    active: false
    url: "https://inactive.example.com/rss"
        """,
        encoding="utf-8",
    )
    sources = load_sources_registry(cfg)
    assert len(sources) == 2
    active = [s for s in sources if s.active]
    assert len(active) == 1
    assert active[0].source_id == "rss_active"


def test_emit_wrapper_contains_schema_version_and_count(tmp_path):
    # Simuliere Wrapper-Objekt und schreibe ihn mit emit_json_artifact
    out = tmp_path / "events_latest.json"
    wrapper = {
        "schema_version": "news.v1",
        "generated_utc": "2025-01-15T00:00:00Z",
        "count": 2,
        "items": [{"event_id": "news_a"}, {"event_id": "news_b"}],
    }
    emit_json_artifact(wrapper, out)
    import json

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "news.v1"
    assert data["count"] == 2
    assert len(data["items"]) == 2


def test_dedupe_store_drops_seen_url(tmp_path):
    from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite

    db_path = tmp_path / "dedupe.sqlite"
    store = DedupeStoreSQLite(db_path)

    ev_id = "news_1"
    url = "https://example.com/a"
    fp64 = 123456789
    store.add_event(
        event_id=ev_id,
        canonical_url=url,
        fp64=fp64,
        published_utc="2025-01-01T00:00:00Z",
        source_id="src",
        ingested_utc="2025-01-02T00:00:00Z",
    )

    assert store.has_url(url) is True
    assert store.has_url("https://example.com/other") is False


def test_dedupe_store_drops_fp64_distance0(tmp_path):
    from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite

    db_path = tmp_path / "dedupe.sqlite"
    store = DedupeStoreSQLite(db_path)

    fp = 0xDEADBEEF12345678
    store.add_event(
        event_id="news_1",
        canonical_url="https://example.com/a",
        fp64=fp,
        published_utc="2025-01-01T00:00:00Z",
        source_id="src",
        ingested_utc="2025-01-02T00:00:00Z",
    )

    has_same, event_id = store.has_fingerprint64(fp)
    assert has_same is True
    assert event_id == "news_1"

    has_other, _ = store.has_fingerprint64(0xAAAAAAAAAAAAAAAA)
    assert has_other is False


def test_prune_removes_old_entries(tmp_path):
    from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite

    db_path = tmp_path / "dedupe.sqlite"
    store = DedupeStoreSQLite(db_path)

    store.add_event(
        event_id="old",
        canonical_url="https://example.com/old",
        fp64=1,
        published_utc="2024-01-01T00:00:00Z",
        source_id="src",
        ingested_utc="2024-01-02T00:00:00Z",
    )
    store.add_event(
        event_id="new",
        canonical_url="https://example.com/new",
        fp64=2,
        published_utc="2025-01-10T00:00:00Z",
        source_id="src",
        ingested_utc="2025-01-11T00:00:00Z",
    )

    # Now is 2025-02-01, window 365d -> old gets pruned, new kept
    deleted = store.prune(window_days=365, now_utc="2025-02-01T00:00:00Z")
    assert deleted >= 1
    assert store.has_url("https://example.com/old") is False
    assert store.has_url("https://example.com/new") is True


def test_store_bucket_range_uses_full_0_255(tmp_path):
    """Bucket 0..255: fp64 with MSB=1 (e.g. 0xFF00...) is stored and found by bucket."""
    from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite

    db_path = tmp_path / "dedupe.sqlite"
    store = DedupeStoreSQLite(db_path)

    # Unsigned 64-bit with top byte 0xFF -> bucket 255
    fp64_u = 0xFF00000000000000
    bucket = DedupeStoreSQLite._bucket(fp64_u)
    assert bucket == 255

    store.add_event(
        event_id="msb_event",
        canonical_url="https://example.com/msb",
        fp64=fp64_u,
        published_utc="2025-01-01T00:00:00Z",
        source_id="src",
        ingested_utc="2025-01-02T00:00:00Z",
    )

    candidates = store.candidates_by_bucket(bucket)
    assert len(candidates) == 1
    assert candidates[0][0] == "msb_event"
    assert candidates[0][1] == fp64_u


def test_has_fingerprint64_exact_match_with_msb(tmp_path):
    """add_event(fp64_u with MSB=1), then has_fingerprint64(fp64_u) -> True."""
    from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite

    db_path = tmp_path / "dedupe.sqlite"
    store = DedupeStoreSQLite(db_path)

    fp64_u = 0x8000000000000001  # MSB set
    store.add_event(
        event_id="news_msb",
        canonical_url="https://example.com/msb",
        fp64=fp64_u,
        published_utc="2025-01-01T00:00:00Z",
        source_id="src",
        ingested_utc="2025-01-02T00:00:00Z",
    )

    has_fp, event_id = store.has_fingerprint64(fp64_u)
    assert has_fp is True
    assert event_id == "news_msb"


def test_near_dupe_tagged_not_dropped(tmp_path, monkeypatch):
    """Near-duplicates should be tagged via raw.near_duplicate_* but not dropped."""
    from src.assembled_core.events.news.dedupe_store import DedupeStoreSQLite
    from src.assembled_core.events.news.fingerprint import simhash64 as real_simhash64
    from src.assembled_core.events.news import run_news_pipeline

    # Prepare configs
    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
        """,
        encoding="utf-8",
    )
    news_cfg = tmp_path / "news.yaml"
    dedupe_db = tmp_path / "dedupe.sqlite"
    news_cfg.write_text(
        f"""
fetch:
  timeout_s: 10
  retries: 0
  backoff_base_s: 0.1
  max_concurrency: 2
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: true
  window_days: 2000
  store:
    backend: "sqlite"
    path: "{dedupe_db.as_posix()}"
  fingerprint:
    treat_distance0_as_duplicate: true
  near_duplicate:
    enabled: true
    hamming_threshold: 3
gdelt:
  enabled: false
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    # Seed store with a base fingerprint
    base_text = "Base news headline about shipping in the Red Sea"
    base_fp = real_simhash64(base_text)
    store = DedupeStoreSQLite(dedupe_db)
    store.add_event(
        event_id="seed_event",
        canonical_url="https://example.com/seed",
        fp64=base_fp,
        published_utc="2025-01-01T00:00:00Z",
        source_id="src",
        ingested_utc="2025-01-02T00:00:00Z",
    )

    # Monkeypatch simhash64 so that new event is a near-duplicate (distance 1)
    from src.assembled_core.events.news import normalize as normalize_module

    def fake_simhash(text: str) -> int:
        # Flip exactly one bit of base_fp to guarantee Hamming distance 1
        return base_fp ^ 0x1

    monkeypatch.setattr(normalize_module, "simhash64", fake_simhash)

    # Stub RSS fetcher to return a single item (patch at pipeline import site)
    from src.assembled_core.events.news import pipeline as pipeline_module

    def fake_fetch_rss_feed(
        source_id: str,
        url: str,
        *,
        timeout: float,
        user_agent: str,
        sanitize_cfg: dict,
        fetch_state: dict,
        retries: int,
        backoff_base_s: float,
    ):
        item = {
            "title": "Near-duplicate headline about shipping in the Red Sea",
            "link": "https://example.com/new-article",
            "published": "2025-01-15T00:00:00Z",
            "summary": "Short summary.",
            "raw": {},
        }
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": 1,
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return [item], None, stats

    monkeypatch.setattr(pipeline_module, "fetch_rss_feed", fake_fetch_rss_feed)

    # Run pipeline
    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=tmp_path / "out",
    )

    events = result["events"]
    health = result["health"]
    assert len(events) == 1
    ev = events[0]
    assert ev.raw.get("near_duplicate_of") == "seed_event"
    assert isinstance(ev.raw.get("near_duplicate_distance"), int)
    # Ensure event was not dropped by dedupe
    assert ev.canonical_url == "https://example.com/new-article"
    assert any("near_dupes_tagged:" in note for note in health.notes)


def test_clusters_union_by_near_duplicate_of():
    fetched = "2025-01-15T00:00:00+00:00"
    ev1 = NewsEvent(
        event_id="news_a",
        source_id="s1",
        title="Title A",
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-01T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000001",
        entities=[],
        countries=[],
    )
    ev2 = NewsEvent(
        event_id="news_b",
        source_id="s1",
        title="Title B",
        url="https://example.com/b",
        canonical_url="https://example.com/b",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-02T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={"near_duplicate_of": "news_a"},
        fingerprint="",
        fingerprint64="0000000000000002",
        entities=[],
        countries=[],
    )
    cfg = {
        "enabled": False,
        "algorithm": "tfidf_cosine",
        "similarity_threshold": 0.45,
        "min_cluster_size": 2,
        "top_phrases_k": 8,
        "top_entities_k": 8,
        "max_pair_checks": 2000,
        "require_overlap": True,
        "same_day_only": True,
        "generated_utc": fetched,
    }
    clusters = build_clusters([ev1, ev2], cfg)
    assert len(clusters) == 1
    clu = clusters[0]
    assert clu["representative_event_id"] == "news_a"
    assert clu["cluster_id"] == "clu_news_a"
    assert clu["event_ids"] == ["news_a", "news_b"]


def test_clusters_pairwise_simhash_overlap():
    fetched = "2025-01-15T00:00:00+00:00"
    # Same day, same country, similar texts (TF-IDF cosine >= threshold)
    ev1 = NewsEvent(
        event_id="news_c1",
        source_id="s1",
        title="Title C1",
        url="https://example.com/c1",
        canonical_url="https://example.com/c1",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-03T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000001",
        entities=[],
        countries=["US"],
    )
    ev2 = NewsEvent(
        event_id="news_c2",
        source_id="s1",
        title="Title C2",
        url="https://example.com/c2",
        canonical_url="https://example.com/c2",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-03T05:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000003",
        entities=[],
        countries=["US"],
    )
    cfg = {
        "enabled": True,
        "algorithm": "tfidf_cosine",
        "similarity_threshold": 0.2,
        "min_cluster_size": 2,
        "top_phrases_k": 8,
        "top_entities_k": 8,
        "max_pair_checks": 2000,
        "require_overlap": True,
        "same_day_only": True,
        "generated_utc": fetched,
    }
    clusters = build_clusters([ev1, ev2], cfg)
    assert len(clusters) == 1
    clu = clusters[0]
    assert set(clu["event_ids"]) == {"news_c1", "news_c2"}
    assert clu["countries"] == ["US"]


def test_clusters_deterministic_ordering():
    fetched = "2025-01-15T00:00:00+00:00"
    ev1 = NewsEvent(
        event_id="news_x",
        source_id="s1",
        title="Title X",
        url="https://example.com/x",
        canonical_url="https://example.com/x",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-01T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000001",
        entities=[],
        countries=[],
    )
    ev2 = NewsEvent(
        event_id="news_y",
        source_id="s1",
        title="Title Y",
        url="https://example.com/y",
        canonical_url="https://example.com/y",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-02T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={"near_duplicate_of": "news_x"},
        fingerprint="",
        fingerprint64="0000000000000002",
        entities=[],
        countries=[],
    )
    ev3 = NewsEvent(
        event_id="news_z",
        source_id="s1",
        title="Title Z",
        url="https://example.com/z",
        canonical_url="https://example.com/z",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-03T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={"near_duplicate_of": "news_x"},
        fingerprint="",
        fingerprint64="0000000000000003",
        entities=[],
        countries=[],
    )
    cfg = {
        "enabled": False,
        "algorithm": "tfidf_cosine",
        "similarity_threshold": 0.45,
        "min_cluster_size": 2,
        "top_phrases_k": 8,
        "top_entities_k": 8,
        "max_pair_checks": 2000,
        "require_overlap": True,
        "same_day_only": True,
        "generated_utc": fetched,
    }
    clusters = build_clusters([ev2, ev3, ev1], cfg)
    assert len(clusters) == 1
    clu = clusters[0]
    # Representative based on oldest published_utc
    assert clu["representative_event_id"] == "news_x"
    assert clu["cluster_id"] == "clu_news_x"
    # event_ids are sorted deterministically
    assert clu["event_ids"] == ["news_x", "news_y", "news_z"]


def test_clusters_tfidf_cosine_merges_similar_text():
    fetched = "2025-01-20T00:00:00+00:00"
    ev1 = NewsEvent(
        event_id="news_t1",
        source_id="s1",
        title="Stocks rally on Fed news",
        url="https://example.com/t1",
        canonical_url="https://example.com/t1",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-10T10:00:00+00:00",
        fetched_utc=fetched,
        summary="Equities jump after central bank announcement.",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000001",
        entities=["FED"],
        countries=["US"],
    )
    ev2 = NewsEvent(
        event_id="news_t2",
        source_id="s1",
        title="Fed announcement lifts stock market",
        url="https://example.com/t2",
        canonical_url="https://example.com/t2",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-10T11:00:00+00:00",
        fetched_utc=fetched,
        summary="Central bank decision sends US equities higher.",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000002",
        entities=["FED"],
        countries=["US"],
    )
    ev3 = NewsEvent(
        event_id="news_t3",
        source_id="s1",
        title="Tech stocks surge after Fed move",
        url="https://example.com/t3",
        canonical_url="https://example.com/t3",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-10T12:00:00+00:00",
        fetched_utc=fetched,
        summary="US technology shares rise following central bank comments.",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000003",
        entities=["FED"],
        countries=["US"],
    )
    cfg = {
        "enabled": True,
        "algorithm": "tfidf_cosine",
        "similarity_threshold": 1e-6,
        "min_cluster_size": 3,
        "top_phrases_k": 8,
        "top_entities_k": 8,
        "max_pair_checks": 2000,
        "require_overlap": True,
        "same_day_only": True,
        "generated_utc": fetched,
    }
    clusters = build_clusters([ev1, ev2, ev3], cfg)
    assert len(clusters) == 1
    clu = clusters[0]
    assert set(clu["event_ids"]) == {"news_t1", "news_t2", "news_t3"}


def test_clusters_min_size_filters_small_groups():
    fetched = "2025-01-20T00:00:00+00:00"
    ev1 = NewsEvent(
        event_id="news_m1",
        source_id="s1",
        title="Small cluster A",
        url="https://example.com/m1",
        canonical_url="https://example.com/m1",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-05T00:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000001",
        entities=[],
        countries=[],
    )
    ev2 = NewsEvent(
        event_id="news_m2",
        source_id="s1",
        title="Small cluster B",
        url="https://example.com/m2",
        canonical_url="https://example.com/m2",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-05T01:00:00+00:00",
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={"near_duplicate_of": "news_m1"},
        fingerprint="",
        fingerprint64="0000000000000002",
        entities=[],
        countries=[],
    )
    cfg = {
        "enabled": False,
        "algorithm": "tfidf_cosine",
        "similarity_threshold": 0.45,
        "min_cluster_size": 3,
        "top_phrases_k": 8,
        "top_entities_k": 8,
        "max_pair_checks": 2000,
        "require_overlap": True,
        "same_day_only": True,
        "generated_utc": fetched,
    }
    clusters = build_clusters([ev1, ev2], cfg)
    assert clusters == []


def test_clusters_outputs_top_entities_top_phrases_present():
    fetched = "2025-01-20T00:00:00+00:00"
    ev1 = NewsEvent(
        event_id="news_e1",
        source_id="s1",
        title="Red Sea shipping disruption hits global trade",
        url="https://example.com/e1",
        canonical_url="https://example.com/e1",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-08T00:00:00+00:00",
        fetched_utc=fetched,
        summary="Ships rerouted around Red Sea increase costs.",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000001",
        entities=["Red Sea"],
        countries=["EG"],
    )
    ev2 = NewsEvent(
        event_id="news_e2",
        source_id="s1",
        title="Global trade impact from Red Sea shipping delays",
        url="https://example.com/e2",
        canonical_url="https://example.com/e2",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-08T02:00:00+00:00",
        fetched_utc=fetched,
        summary="Red Sea disruptions force ships to reroute, hitting trade.",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000002",
        entities=["Red Sea"],
        countries=["EG"],
    )
    ev3 = NewsEvent(
        event_id="news_e3",
        source_id="s1",
        title="Red Sea shipping crisis pushes up freight rates",
        url="https://example.com/e3",
        canonical_url="https://example.com/e3",
        source_name="S1",
        source_domain="example.com",
        published_utc="2025-01-08T04:00:00+00:00",
        fetched_utc=fetched,
        summary="Freight costs surge as Red Sea shipping routes face disruption.",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="0000000000000003",
        entities=["Red Sea"],
        countries=["EG"],
    )
    cfg = {
        "enabled": True,
        "algorithm": "tfidf_cosine",
        "similarity_threshold": 0.3,
        "min_cluster_size": 3,
        "top_phrases_k": 5,
        "top_entities_k": 5,
        "max_pair_checks": 2000,
        "require_overlap": True,
        "same_day_only": True,
        "generated_utc": fetched,
    }
    clusters = build_clusters([ev1, ev2, ev3], cfg)
    assert len(clusters) == 1
    clu = clusters[0]
    assert "top_entities" in clu and clu["top_entities"]
    assert "top_phrases" in clu and clu["top_phrases"]


def test_version_hash_changes_when_config_changes():
    cfg1 = {
        "burst": {
            "baseline_days": 30,
            "min_doc_count": 3,
            "top_k": 50,
            "version_salt": "v1",
        },
        "clustering": {
            "top_phrases_k": 8,
            "top_entities_k": 8,
        },
    }
    cfg2 = {
        "burst": {
            "baseline_days": 60,  # change
            "min_doc_count": 3,
            "top_k": 50,
            "version_salt": "v1",
        },
        "clustering": {
            "top_phrases_k": 8,
            "top_entities_k": 8,
        },
    }
    h1 = compute_version_hash(cfg1)
    h2 = compute_version_hash(cfg2)
    assert h1 != h2


def test_daily_updates_baseline_and_prunes_old_days(tmp_path):
    """update_baseline prunes days older than baseline_days and rebuilds aggregates."""
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    # Seed state with three days
    state_path = baseline_dir / "baseline_state.json"
    seed_state = {
        "schema_version": "news.baseline_state.v1",
        "version_hash": "old",
        "days": {
            "2025-01-01": {
                "entity_counts": {"US": 1},
                "phrase_counts": {"red sea": 1},
            },
            "2025-01-02": {
                "entity_counts": {"US": 2},
                "phrase_counts": {"red sea": 2},
            },
            "2025-01-03": {
                "entity_counts": {"US": 3},
                "phrase_counts": {"red sea": 3},
            },
        },
    }
    import json

    state_path.write_text(json.dumps(seed_state), encoding="utf-8")

    cfg = {
        "burst": {
            "enabled": True,
            "baseline_days": 2,
            "min_doc_count": 1,
            "top_k": 10,
            "version_salt": "v1",
        },
        "clustering": {
            "top_phrases_k": 8,
            "top_entities_k": 8,
        },
    }
    # One simple cluster for 2025-01-04
    clusters = [
        {
            "event_ids": ["e1"],
            "top_entities": ["US"],
            "entities": [],
            "top_phrases": ["red sea"],
        }
    ]
    now_utc = "2025-01-04T00:00:00+00:00"

    meta = update_baseline(clusters, cfg, now_utc, baseline_dir)
    assert meta["days_covered"] >= 1

    # Load state and baseline_latest
    state = json.loads(state_path.read_text(encoding="utf-8"))
    days = state["days"]
    # Oldest day (2025-01-01) should have been pruned for baseline_days=2
    assert "2025-01-01" not in days

    latest = json.loads((baseline_dir / "baseline_latest.json").read_text(encoding="utf-8"))
    assert latest["schema_version"] == "news.baseline.v1"
    assert latest["baseline_days"] == 2
    # Aggregated counts should be deterministic and limited
    assert "entity_counts" in latest and "phrase_counts" in latest


def test_baseline_top_k_deterministic(tmp_path):
    baseline_dir = tmp_path / "baseline2"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "burst": {
            "enabled": True,
            "baseline_days": 30,
            "min_doc_count": 1,
            "top_k": 2,
            "version_salt": "v1",
        },
        "clustering": {
            "top_phrases_k": 8,
            "top_entities_k": 8,
        },
    }
    clusters = [
        {
            "event_ids": ["e1"],
            "top_entities": ["B", "A"],
            "entities": [],
            "top_phrases": ["x y", "y z"],
        },
        {
            "event_ids": ["e2"],
            "top_entities": ["A", "B"],
            "entities": [],
            "top_phrases": ["x y", "y z"],
        },
    ]
    now_utc = "2025-01-10T00:00:00+00:00"

    update_baseline(clusters, cfg, now_utc, baseline_dir)

    import json

    latest = json.loads((baseline_dir / "baseline_latest.json").read_text(encoding="utf-8"))
    ents = list(latest["entity_counts"].keys())
    # A and B have same counts -> sorted alphabetically
    assert ents == sorted(ents)


def test_burst_ratio_higher_when_current_spikes():
    baseline = {
        "baseline_days": 30,
        "entity_counts": {"US": 30},
        "phrase_counts": {},
    }
    cfg = {
        "burst": {
            "min_doc_count": 1,
            "top_k": 10,
        },
        "clustering": {},
    }
    # Low current
    clusters_low = [{"top_entities": ["US"], "entities": [], "top_phrases": []}]
    bursts_low_dict = compute_bursts_for_window(clusters_low, baseline, cfg, window_hours=1)
    bursts_low = bursts_low_dict["top_entities_burst"]
    # High current
    clusters_high = [{"top_entities": ["US"], "entities": [], "top_phrases": []}] * 10
    bursts_high_dict = compute_bursts_for_window(clusters_high, baseline, cfg, window_hours=1)
    bursts_high = bursts_high_dict["top_entities_burst"]
    ratio_low = bursts_low[0]["ratio"]
    ratio_high = bursts_high[0]["ratio"]
    assert ratio_high > ratio_low


def test_min_doc_count_filters():
    baseline = None
    cfg = {
        "burst": {
            "min_doc_count": 3,
            "top_k": 10,
        },
        "clustering": {},
    }
    clusters = [{"top_entities": ["US"], "entities": [], "top_phrases": []}]
    bursts_dict = compute_bursts_for_window(clusters, baseline, cfg, window_hours=1)
    assert bursts_dict["top_entities_burst"] == [] and bursts_dict["top_phrases_burst"] == []


def test_top_clusters_burst_present_when_cluster_has_bursty_keys():
    baseline = None
    cfg = {
        "burst": {
            "min_doc_count": 1,
            "top_k": 10,
        },
        "clustering": {},
    }
    clusters = [
        {
            "cluster_id": "clu_a",
            "top_entities": ["US"],
            "entities": [],
            "top_phrases": ["red sea"],
        },
        {
            "cluster_id": "clu_b",
            "top_entities": ["EU"],
            "entities": [],
            "top_phrases": [],
        },
    ]
    bursts = compute_bursts_for_window(clusters, baseline, cfg, window_hours=1)
    top_clusters = bursts["top_clusters_burst"]
    assert any(c["cluster_id"] == "clu_a" for c in top_clusters)


def test_evidence_tierA_alone_ok():
    fetched = "2025-01-10T00:00:00+00:00"
    ev = NewsEvent(
        event_id="e1",
        source_id="srcA",
        title="Headline",
        url="https://example.com/a",
        canonical_url="https://example.com/a",
        source_name="S1",
        source_domain="domA.com",
        published_utc=fetched,
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="",
        entities=[],
        countries=[],
    )
    events_by_id = {"e1": ev}
    source_meta = {"srcA": {"tier": "A", "domain": "doma.com"}}
    cluster = {"event_ids": ["e1"]}
    evidence = summarize_cluster_evidence(cluster, events_by_id, source_meta, fetched)
    assert evidence["tierA_count"] == 1
    assert evidence["tierB_count"] == 0
    assert evidence["evidence_ok"] is True


def test_evidence_tierB_requires_two_domains():
    fetched = "2025-01-10T00:00:00+00:00"
    ev1 = NewsEvent(
        event_id="b1",
        source_id="srcB1",
        title="B1",
        url="https://example.com/b1",
        canonical_url="https://example.com/b1",
        source_name="S1",
        source_domain="domB1.com",
        published_utc=fetched,
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="",
        entities=[],
        countries=[],
    )
    ev2 = NewsEvent(
        event_id="b2",
        source_id="srcB2",
        title="B2",
        url="https://example.com/b2",
        canonical_url="https://example.com/b2",
        source_name="S2",
        source_domain="domB2.com",
        published_utc=fetched,
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="",
        entities=[],
        countries=[],
    )
    events_by_id = {"b1": ev1, "b2": ev2}
    # Case 1: same domain -> not independent enough
    source_meta_same = {
        "srcB1": {"tier": "B", "domain": "domx.com"},
        "srcB2": {"tier": "B", "domain": "domx.com"},
    }
    cluster = {"event_ids": ["b1", "b2"]}
    ev_same = summarize_cluster_evidence(cluster, events_by_id, source_meta_same, fetched)
    assert ev_same["tierB_count"] == 2
    assert ev_same["tierB_independent_count"] == 1
    assert ev_same["evidence_ok"] is False

    # Case 2: different domains -> evidence_ok=True
    source_meta_diff = {
        "srcB1": {"tier": "B", "domain": "dom1.com"},
        "srcB2": {"tier": "B", "domain": "dom2.com"},
    }
    ev_diff = summarize_cluster_evidence(cluster, events_by_id, source_meta_diff, fetched)
    assert ev_diff["tierB_independent_count"] == 2
    assert ev_diff["evidence_ok"] is True


def test_cluster_contains_evidence_block():
    fetched = "2025-01-10T00:00:00+00:00"
    ev = NewsEvent(
        event_id="c1",
        source_id="srcB",
        title="C1",
        url="https://example.com/c1",
        canonical_url="https://example.com/c1",
        source_name="S1",
        source_domain="domC.com",
        published_utc=fetched,
        fetched_utc=fetched,
        summary="",
        language=None,
        raw={},
        fingerprint="",
        fingerprint64="",
        entities=[],
        countries=[],
    )
    events_by_id = {"c1": ev}
    source_meta = {"srcB": {"tier": "B", "domain": "domc.com"}}
    cluster = {"event_ids": ["c1"]}
    cluster["evidence"] = summarize_cluster_evidence(cluster, events_by_id, source_meta, fetched)
    assert "evidence" in cluster
    assert "evidence_ok" in cluster["evidence"]


def test_bursts_artifact_written_wrapper_fields(tmp_path, monkeypatch):
    from src.assembled_core.events.news import run_news_pipeline

    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
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
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: false
clustering:
  enabled: false
burst:
  enabled: true
gdelt:
  enabled: false
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    from src.assembled_core.events.news import pipeline as pipeline_module

    def fake_fetch_rss_feed(
        source_id: str,
        url: str,
        *,
        timeout: float,
        user_agent: str,
        sanitize_cfg: dict,
        fetch_state: dict,
        retries: int,
        backoff_base_s: float,
    ):
        items = [
            {
                "title": "Red Sea shipping disruption hits global trade",
                "link": "https://example.com/e1",
                "published": "2025-01-08T00:00:00Z",
                "summary": "Ships rerouted around Red Sea increase costs.",
                "raw": {},
            }
        ]
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": len(items),
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    monkeypatch.setattr(pipeline_module, "fetch_rss_feed", fake_fetch_rss_feed)

    out_dir = tmp_path / "out_bursts"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    assert len(result["events"]) == 1

    import json

    bursts_path = out_dir / "bursts_latest.json"
    data = json.loads(bursts_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "news.bursts.v1"
    assert "window_hours" in data
    assert "baseline_version_hash" in data
    assert "count" in data and isinstance(data["count"], int)
    assert isinstance(data["items"], list)
    # New multi-window structure
    assert "windows" in data
    assert isinstance(data["windows"], list)


def test_bursts_windows_contains_1_6_24(tmp_path, monkeypatch):
    from src.assembled_core.events.news import run_news_pipeline

    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
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
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: false
clustering:
  enabled: false
burst:
  enabled: true
  windows_hours: [1, 6, 24]
gdelt:
  enabled: false
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    from src.assembled_core.events.news import pipeline as pipeline_module

    def fake_fetch_rss_feed(
        source_id: str,
        url: str,
        *,
        timeout: float,
        user_agent: str,
        sanitize_cfg: dict,
        fetch_state: dict,
        retries: int,
        backoff_base_s: float,
    ):
        items = [
            {
                "title": "Simple headline",
                "link": "https://example.com/e1",
                "published": "2025-01-08T00:00:00Z",
                "summary": "Some summary.",
                "raw": {},
            }
        ]
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": len(items),
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    monkeypatch.setattr(pipeline_module, "fetch_rss_feed", fake_fetch_rss_feed)

    out_dir = tmp_path / "out_bursts_windows"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    import json

    bursts_path = out_dir / "bursts_latest.json"
    data = json.loads(bursts_path.read_text(encoding="utf-8"))
    whs = sorted(int(w["window_hours"]) for w in data["windows"])
    assert whs == [1, 6, 24]


def test_health_contains_cluster_quality_metrics(tmp_path, monkeypatch):
    """Pipeline health_latest.json enthält Cluster-Qualitätsmetriken."""
    from src.assembled_core.events.news import run_news_pipeline

    # Prepare configs
    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
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
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: false
clustering:
  enabled: false
gdelt:
  enabled: false
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    # Stub RSS fetcher at pipeline import site
    from src.assembled_core.events.news import pipeline as pipeline_module

    def fake_fetch_rss_feed(
        source_id: str,
        url: str,
        *,
        timeout: float,
        user_agent: str,
        sanitize_cfg: dict,
        fetch_state: dict,
        retries: int,
        backoff_base_s: float,
    ):
        items = [
            {
                "title": "Stocks rally on Fed news",
                "link": "https://example.com/t1",
                "published": "2025-01-10T10:00:00Z",
                "summary": "Equities jump after central bank announcement.",
                "raw": {},
            },
            {
                "title": "Fed announcement lifts stock market",
                "link": "https://example.com/t2",
                "published": "2025-01-10T11:00:00Z",
                "summary": "Central bank decision sends US equities higher.",
                "raw": {},
            },
            {
                "title": "Tech stocks surge after Fed move",
                "link": "https://example.com/t3",
                "published": "2025-01-10T12:00:00Z",
                "summary": "US technology shares rise following central bank comments.",
                "raw": {},
            },
        ]
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": len(items),
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    monkeypatch.setattr(pipeline_module, "fetch_rss_feed", fake_fetch_rss_feed)

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    assert len(result["events"]) == 3

    import json

    health_path = out_dir / "health_latest.json"
    data = json.loads(health_path.read_text(encoding="utf-8"))
    h = data["health"]
    assert "metrics" in h
    cq = h["metrics"]["cluster_quality"]
    # Clustering disabled -> no clusters, all events unclustered
    assert cq["cluster_count"] == 0
    assert cq["total_events"] == 3
    assert cq["clustered_events"] == 0
    assert cq["unclustered_events"] == 3
    assert cq["pct_unclustered"] == 1.0


def test_pct_unclustered_when_no_clusters(tmp_path, monkeypatch):
    """pct_unclustered == 1.0, wenn keine Cluster gebildet werden."""
    from src.assembled_core.events.news import run_news_pipeline

    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
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
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
dedupe:
  enabled: false
clustering:
  enabled: false
gdelt:
  enabled: false
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    from src.assembled_core.events.news import pipeline as pipeline_module

    def fake_fetch_rss_feed(
        source_id: str,
        url: str,
        *,
        timeout: float,
        user_agent: str,
        sanitize_cfg: dict,
        fetch_state: dict,
        retries: int,
        backoff_base_s: float,
    ):
        items = [
            {
                "title": "Single headline no clustering",
                "link": "https://example.com/s1",
                "published": "2025-01-10T10:00:00Z",
                "summary": "Some neutral text.",
                "raw": {},
            }
        ]
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": len(items),
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    monkeypatch.setattr(pipeline_module, "fetch_rss_feed", fake_fetch_rss_feed)

    out_dir = tmp_path / "out2"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_news_pipeline(
        sources_path=str(sources_cfg),
        news_path=str(news_cfg),
        cadence="hourly",
        output_dir=out_dir,
    )

    assert len(result["events"]) == 1

    import json

    health_path = out_dir / "health_latest.json"
    data = json.loads(health_path.read_text(encoding="utf-8"))
    h = data["health"]
    assert "metrics" in h
    cq = h["metrics"]["cluster_quality"]
    assert cq["cluster_count"] == 0
    assert cq["total_events"] == 1
    assert cq["clustered_events"] == 0
    assert cq["unclustered_events"] == 1
    assert cq["pct_unclustered"] == 1.0


def test_rss_headers_if_modified_since_etag_applied(monkeypatch):
    # Prepare fake requests + feedparser
    import types
    import sys

    calls = {}

    def fake_get(url, headers=None, timeout=None):
        class Resp:
            status_code = 200

            def raise_for_status(self):
                return None

            @property
            def content(self):
                return b"feed"

            @property
            def headers(self):
                return {}

        calls["headers"] = headers or {}
        return Resp()

    fake_requests = types.SimpleNamespace(get=fake_get)
    sys.modules["requests"] = fake_requests

    # Stub feedparser module so that import inside fetch_rss_feed succeeds
    fake_feedparser = types.SimpleNamespace(
        parse=lambda content: types.SimpleNamespace(
            entries=[
                {
                    "title": "Title",
                    "link": "https://example.com/article",
                    "published": "2025-01-01T00:00:00Z",
                    "summary": "Summary",
                }
            ]
        )
    )
    sys.modules["feedparser"] = fake_feedparser

    fetch_state = {
        "rss": {
            "src1": {
                "etag": "etag-1",
                "last_modified": "Wed, 01 Jan 2025 00:00:00 GMT",
            }
        }
    }
    sanitize_cfg = {"strip_html": True, "title_max_chars": 100, "summary_max_chars": 200}

    items, failure, stats = fetch_rss_feed(
        "src1",
        "https://example.com/rss",
        timeout=5.0,
        user_agent="UA",
        sanitize_cfg=sanitize_cfg,
        fetch_state=fetch_state,
        retries=0,
        backoff_base_s=0.1,
    )

    assert failure is None
    assert stats["ok"] is True
    # Headers must include If-None-Match / If-Modified-Since
    assert "If-None-Match" in calls["headers"]
    assert "If-Modified-Since" in calls["headers"]


def test_gdelt_cache_hit_no_request(monkeypatch):
    # Prepare fake requests that would fail if called
    import types
    import sys

    def fake_get(*args, **kwargs):
        raise AssertionError("requests.get should not be called on cache hit")

    fake_requests = types.SimpleNamespace(get=fake_get)
    sys.modules["requests"] = fake_requests

    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(minutes=5)).isoformat()
    from hashlib import sha256

    query = "war OR sanctions OR shipping"
    window_hours = 1
    cache_key = sha256(f"{query}:{window_hours}".encode("utf-8")).hexdigest()
    fetch_state = {
        "gdelt": {
            cache_key: {
                "cached_utc": recent,
                "items": [{"title": "t", "link": "u", "published": "p", "summary": None}],
            }
        }
    }
    gdelt_cfg = {
        "rate_limit_rps": 1.0,
        "cache_minutes": 10,
        "stale_on_error_minutes": 60,
        "window_hours": {"hourly": 1, "daily": 6},
    }

    # Use same key computation (sha256(query:window_hours)) by calling with same query/window
    items, failure, stats = fetch_gdelt_events(
        "gdelt_source",
        query,
        gdelt_cfg=gdelt_cfg,
        cadence="hourly",
        fetch_state=fetch_state,
    )

    # On pure cache hit, no HTTP request should be made and failure should be None.
    assert failure is None
    assert stats["cached"] is True
    assert stats["ok"] is True


def test_fetch_report_written_wrapper_fields(tmp_path, monkeypatch):
    """Pipeline writes fetch_report_latest.json with consistent totals/per_source."""
    # Prepare minimal sources/news config under tmp_path
    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
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
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
gdelt:
  enabled: false
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    # Monkeypatch fetch_rss_feed/fetch_gdelt_events to avoid net calls

    def fake_fetch_rss(source_id, url, *, timeout, user_agent, sanitize_cfg, fetch_state, retries, backoff_base_s):
        items = [
            {"title": "A", "link": "https://example.com/a", "published": "2025-01-01T00:00:00Z", "summary": "sa", "raw": {}},
            {"title": "B", "link": "https://example.com/b", "published": "2025-01-02T00:00:00Z", "summary": "sb", "raw": {}},
        ]
        stats = {
            "source_id": source_id,
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 10,
            "items": len(items),
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return items, None, stats

    def fake_fetch_gdelt(source_id, query, *, gdelt_cfg, cadence, fetch_state):
        stats = {
            "source_id": source_id,
            "type": "gdelt",
            "ok": True,
            "http_status": 200,
            "duration_ms": 5,
            "items": 0,
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return [], None, stats

    import src.assembled_core.events.news.pipeline as news_pipeline

    monkeypatch.setattr(news_pipeline, "fetch_rss_feed", fake_fetch_rss)
    monkeypatch.setattr(news_pipeline, "fetch_gdelt_events", fake_fetch_gdelt)

    out_dir = tmp_path / "out"
    result = run_news_pipeline(
        sources_path=sources_cfg,
        news_path=news_cfg,
        cadence="hourly",
        output_dir=out_dir,
    )
    # Ensure pipeline ran and returned events/health
    assert "events" in result and "health" in result

    import json

    fetch_report_path = out_dir / "fetch_report_latest.json"
    assert fetch_report_path.exists()
    data = json.loads(fetch_report_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "news.fetch_report.v1"
    assert data["cadence"] == "hourly"
    assert "totals" in data and "per_source" in data
    per_source = data["per_source"]
    totals = data["totals"]
    assert totals["sources_total"] == len(per_source)
    assert totals["sources_ok"] + totals["sources_failed"] == len(per_source)
    assert totals["items_raw"] == sum(s.get("items", 0) for s in per_source)


def test_daily_housekeeping_written_when_daily(tmp_path, monkeypatch):
    # Seed a fetch_state with an old gdelt cache entry
    from datetime import datetime, timezone, timedelta
    import json

    now = datetime.now(timezone.utc)
    old = (now - timedelta(minutes=500)).isoformat()
    fetch_state_dir = tmp_path / "out" / "cache"
    fetch_state_dir.mkdir(parents=True, exist_ok=True)
    fetch_state_path = fetch_state_dir / "fetch_state.json"
    fetch_state_path.write_text(
        json.dumps(
            {
                "schema_version": "news.fetch_state.v1",
                "updated_utc": now.isoformat(),
                "rss": {},
                "gdelt": {
                    "old_key": {
                        "cached_utc": old,
                        "items": [{"title": "t", "link": "u", "published": "p", "summary": None}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    # Minimal sources/news config
    sources_cfg = tmp_path / "sources.yaml"
    sources_cfg.write_text(
        """
sources:
  - source_id: "rss_example"
    name: "Example"
    domain: "example.com"
    type: "rss"
    tier: "A"
    weight: 1.0
    active: true
    url: "https://example.com/rss"
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
  user_agent: "UA"
  sanitize:
    strip_html: true
    title_max_chars: 300
    summary_max_chars: 800
gdelt:
  enabled: false
  stale_on_error_minutes: 60
health:
  min_sources_ok: 1
        """,
        encoding="utf-8",
    )

    # Monkeypatch fetchers to no-op
    import src.assembled_core.events.news.pipeline as news_pipeline

    def fake_fetch_rss(*args, **kwargs):
        stats = {
            "source_id": "rss_example",
            "type": "rss",
            "ok": True,
            "http_status": 200,
            "duration_ms": 1,
            "items": 0,
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return [], None, stats

    def fake_fetch_gdelt(*args, **kwargs):
        stats = {
            "source_id": "gdelt_source",
            "type": "gdelt",
            "ok": True,
            "http_status": 200,
            "duration_ms": 1,
            "items": 0,
            "not_modified": False,
            "cached": False,
            "error": None,
        }
        return [], None, stats

    monkeypatch.setattr(news_pipeline, "fetch_rss_feed", fake_fetch_rss)
    monkeypatch.setattr(news_pipeline, "fetch_gdelt_events", fake_fetch_gdelt)

    out_dir = tmp_path / "out"
    result = run_news_pipeline(
        sources_path=sources_cfg,
        news_path=news_cfg,
        cadence="daily",
        output_dir=out_dir,
    )
    assert "events" in result and "health" in result

    housekeeping_path = out_dir / "daily_housekeeping_latest.json"
    assert housekeeping_path.exists()
    data = json.loads(housekeeping_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "news.housekeeping.v1"
    assert data["cadence"] == "daily"
    assert data["pruned_gdelt_cache_entries"] >= 1

