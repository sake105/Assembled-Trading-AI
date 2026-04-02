"""Tests for GDELT intel ingest, dedupe, and clustering modules.

No real network calls — requests.get is mocked throughout.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_dedupe import NewsDedupeIndex, canonical_url, content_fingerprint
from src.assembled_core.intel.news_ingest import (
    GdeltBatchRecord,
    GdeltFetcher,
    parse_lastupdate,
    records_to_news_events,
)
from src.assembled_core.intel.news_cluster import ClusterManager


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_LASTUPDATE = (
    "1234567 abc123 http://data.gdeltproject.org/gdeltv2/20260402123000.export.CSV.zip\n"
    "2345678 def456 http://data.gdeltproject.org/gdeltv2/20260402123000.mentions.CSV.zip\n"
    "3456789 ghi789 http://data.gdeltproject.org/gdeltv2/20260402123000.gkg.csv.zip\n"
)

# Minimal GKG TSV row: 13 tab-separated fields (0-indexed 0..12)
# Indices: 0=GKGRECORDID, 1=DATE, 2=SrcCollection, 3=SrcName, 4=DocId,
#          5,6=unused, 7=Themes, 8=unused, 9=Locations, 10=unused, 11=Orgs, 12=Tone
SAMPLE_GKG_ROW = (
    "20260402123000-1\t20260402123000\t1\treuters.com\t"
    "https://reuters.com/article/oil-disruption\t\t\t"
    "ENV_OIL;TAX_MILITARY;CONFLICT\t\t"
    "3#Iran#IR#32.0#53.0\t\t"
    "Reuters;AP\t"
    "-4.5,1.2,5.8,4.3,0.2,0.1\t"
)

SAMPLE_GKG_ROW_IRRELEVANT = (
    "20260402123001-1\t20260402123001\t1\texample.com\t"
    "https://example.com/sports\t\t\t"
    "SPORTS;ENTERTAINMENT\t\t"
    "1#USA#US#37.0#-95.0\t\t"
    "SportsCorp\t"
    "1.0,2.0,0.5,1.5,0.1,0.0\t"
)


def _make_gkg_zip(rows: list[str]) -> bytes:
    """Build a minimal GKG zip file from TSV row strings."""
    content = "\n".join(rows).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20260402123000.gkg.csv", content)
    return buf.getvalue()


def _make_news_event(
    event_id: str = "ne_test001",
    source_id: str = "gdelt",
    source_tier: SourceTier = SourceTier.T2,
    title: str = "Test oil disruption event",
    url: str = "https://reuters.com/article/oil",
    keywords: list[str] | None = None,
    geo_tags: list[str] | None = None,
) -> NewsEvent:
    return NewsEvent(
        event_id=event_id,
        source_id=source_id,
        source_tier=source_tier,
        title=title,
        url=url,
        published_at=datetime(2026, 4, 2, 12, 30, tzinfo=timezone.utc),
        ingested_at=datetime.now(tz=timezone.utc),
        geo_tags=geo_tags or ["IR", "SA"],
        entities=["Reuters"],
        keywords=keywords or ["oil", "conflict", "tax_military"],
        content_hash=hashlib.sha256(url.encode()).hexdigest()[:16],
    )


# ===========================================================================
# news_ingest: parse_lastupdate
# ===========================================================================


def test_parse_lastupdate_extracts_gkg_url():
    url = parse_lastupdate(SAMPLE_LASTUPDATE)
    assert url == "http://data.gdeltproject.org/gdeltv2/20260402123000.gkg.csv.zip"


def test_parse_lastupdate_empty_returns_none():
    result = parse_lastupdate("")
    assert result is None


def test_parse_lastupdate_too_few_lines_returns_none():
    result = parse_lastupdate("line1\nline2\n")
    assert result is None


# ===========================================================================
# news_ingest: records_to_news_events
# ===========================================================================


def test_records_to_news_events_basic():
    rec = GdeltBatchRecord(
        record_id="20260402123000-1",
        date_str="20260402123000",
        source_name="reuters.com",
        url="https://reuters.com/article/oil-disruption",
        themes=["ENV_OIL", "CONFLICT"],
        country_codes=["IR"],
        organizations=["Reuters", "OPEC"],
        tone=-4.5,
        batch_ts=datetime.now(tz=timezone.utc),
    )
    events = records_to_news_events([rec])
    assert len(events) == 1
    evt = events[0]
    assert evt.event_id.startswith("ne_")
    assert evt.source_id == "gdelt"
    assert evt.source_tier == SourceTier.T2
    assert evt.url == "https://reuters.com/article/oil-disruption"
    assert "env_oil" in evt.keywords
    assert "IR" in evt.geo_tags
    assert "Reuters" in evt.entities
    # published_at should be UTC
    assert evt.published_at.tzinfo is not None
    assert evt.published_at.year == 2026


def test_records_to_news_events_event_id_is_deterministic():
    rec = GdeltBatchRecord(
        record_id="20260402123000-1",
        date_str="20260402123000",
        source_name="test.com",
        url="https://test.com/article",
        batch_ts=datetime.now(tz=timezone.utc),
    )
    events1 = records_to_news_events([rec])
    events2 = records_to_news_events([rec])
    assert events1[0].event_id == events2[0].event_id


# ===========================================================================
# news_dedupe: canonical_url
# ===========================================================================


def test_canonical_url_strips_utm():
    url = "https://Reuters.com/article/oil?utm_source=twitter&utm_medium=social"
    result = canonical_url(url)
    assert "utm_source" not in result
    assert "utm_medium" not in result
    assert "reuters.com" in result


def test_canonical_url_lowercases_host():
    url = "https://Reuters.COM/article/oil"
    result = canonical_url(url)
    assert "reuters.com" in result
    assert "Reuters.COM" not in result


def test_canonical_url_keeps_path():
    url = "https://example.com/news/article/123"
    result = canonical_url(url)
    assert "/news/article/123" in result


def test_canonical_url_strips_fbclid():
    url = "https://example.com/article?fbclid=abc123&q=oil"
    result = canonical_url(url)
    assert "fbclid" not in result
    assert "q=oil" in result


# ===========================================================================
# news_dedupe: content_fingerprint
# ===========================================================================


def test_content_fingerprint_stable():
    fp1 = content_fingerprint("Oil disruption in Iran!", "reuters")
    fp2 = content_fingerprint("Oil disruption in Iran!", "reuters")
    assert fp1 == fp2
    assert len(fp1) == 12


def test_content_fingerprint_normalizes_punctuation():
    fp1 = content_fingerprint("Oil, disruption. In Iran!", "reuters")
    fp2 = content_fingerprint("Oil disruption In Iran", "reuters")
    # After stripping punctuation and lowercasing they should match
    assert fp1 == fp2


def test_content_fingerprint_differs_by_source():
    fp1 = content_fingerprint("Oil news", "reuters")
    fp2 = content_fingerprint("Oil news", "gdelt")
    assert fp1 != fp2


# ===========================================================================
# news_dedupe: NewsDedupeIndex
# ===========================================================================


def test_dedupe_exact_id_blocks_duplicate():
    idx = NewsDedupeIndex()
    evt = _make_news_event(event_id="ne_unique001")
    idx.add(evt)
    assert idx.is_duplicate(evt) is True


def test_dedupe_fingerprint_blocks_similar():
    idx = NewsDedupeIndex()
    evt1 = _make_news_event(event_id="ne_a001", title="Oil pipeline disruption in Iran")
    evt2 = _make_news_event(event_id="ne_b002", title="Oil pipeline disruption in Iran")
    idx.add(evt1)
    # Different event_id but same title+source → fingerprint match
    assert idx.is_duplicate(evt2) is True


def test_dedupe_filter_new_returns_only_novel():
    idx = NewsDedupeIndex()
    evt1 = _make_news_event(event_id="ne_x001", title="Event Alpha", url="https://a.com/1")
    evt2 = _make_news_event(event_id="ne_x002", title="Event Beta", url="https://b.com/2")
    evt3 = _make_news_event(event_id="ne_x001", title="Event Alpha", url="https://a.com/1")  # duplicate of evt1

    result = idx.filter_new([evt1, evt2, evt3])
    assert len(result) == 2
    ids = {e.event_id for e in result}
    assert "ne_x001" in ids
    assert "ne_x002" in ids


def test_dedupe_persist_and_reload(tmp_path):
    persist_path = tmp_path / "dedupe.json"
    idx = NewsDedupeIndex(persist_path=persist_path)
    evt = _make_news_event(event_id="ne_persist001")
    idx.add(evt)
    idx.save()

    idx2 = NewsDedupeIndex(persist_path=persist_path)
    assert idx2.is_duplicate(evt) is True


# ===========================================================================
# news_cluster: ClusterManager
# ===========================================================================


def test_cluster_manager_groups_by_trigger_type():
    mgr = ClusterManager(cluster_ttl_minutes=360)
    now = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)

    events = [
        _make_news_event(
            event_id=f"ne_{i:04d}",
            url=f"https://example.com/{i}",
            keywords=["oil", "conflict", "war"],
            title="War escalation in Middle East",
        )
        for i in range(3)
    ]

    clusters = mgr.update_clusters(events, now=now)
    assert len(clusters) >= 1
    trigger_types = {cl.trigger_type for cl in clusters}
    # War/conflict keywords should produce WAR_ESCALATION or ENERGY_SUPPLY_RISK
    assert len(trigger_types) >= 1


def test_cluster_manager_expires_old_clusters():
    mgr = ClusterManager(cluster_ttl_minutes=1)
    past = datetime(2026, 4, 2, 10, 0, tzinfo=timezone.utc)
    now = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)  # 2 hours later

    events = [
        _make_news_event(
            event_id="ne_old001",
            url="https://example.com/old",
            keywords=["oil", "conflict"],
            title="Old oil conflict event",
        )
    ]

    # Seed clusters at 'past'
    mgr.update_clusters(events, now=past)
    assert len(mgr.active_clusters) >= 1

    # Run at 'now' with no new events — old clusters should be expired
    mgr.update_clusters([], now=now)
    assert len(mgr.active_clusters) == 0


def test_cluster_manager_confidence_increases_with_events():
    mgr = ClusterManager(cluster_ttl_minutes=360)
    now = datetime(2026, 4, 2, 12, 0, tzinfo=timezone.utc)

    # Add one event
    ev1 = _make_news_event(
        event_id="ne_c001",
        url="https://a.com/1",
        keywords=["oil", "energy", "lng"],
        title="LNG pipeline explosion",
    )
    clusters_1 = mgr.update_clusters([ev1], now=now)
    conf_1 = clusters_1[0].confidence if clusters_1 else 0.0

    # Add more events to the same hour bucket
    more_events = [
        _make_news_event(
            event_id=f"ne_c{i:03d}",
            url=f"https://a.com/{i}",
            keywords=["oil", "energy", "lng"],
            title="LNG pipeline explosion",
        )
        for i in range(2, 7)
    ]
    clusters_2 = mgr.update_clusters(more_events, now=now)
    conf_2 = max((cl.confidence for cl in clusters_2), default=0.0)

    # Confidence should be >= first cycle
    assert conf_2 >= conf_1


# ===========================================================================
# news_ingest: GdeltFetcher
# ===========================================================================


def test_gdelt_fetcher_skips_already_seen_batch(tmp_path):
    state_path = tmp_path / "gdelt_state.json"
    gkg_url = "http://data.gdeltproject.org/gdeltv2/20260402123000.gkg.csv.zip"

    # Pre-seed state with the URL
    state_path.write_text(
        json.dumps({
            "last_batch_url": gkg_url,
            "last_fetch_ts": "2026-04-02T12:30:00+00:00",
            "total_events_ingested": 10,
            "consecutive_failures": 0,
        })
    )

    fetcher = GdeltFetcher(state_path)

    lastupdate_text = (
        f"100 abc {gkg_url.replace('.gkg.csv.zip', '.export.CSV.zip')}\n"
        f"200 def {gkg_url.replace('.gkg.csv.zip', '.mentions.CSV.zip')}\n"
        f"300 ghi {gkg_url}\n"
    )

    with patch("src.assembled_core.intel.news_ingest.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.text = lastupdate_text
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        events, is_new = fetcher.fetch_new_events()

    assert is_new is False
    assert events == []


def test_gdelt_fetcher_handles_network_failure_gracefully(tmp_path):
    state_path = tmp_path / "gdelt_state.json"
    fetcher = GdeltFetcher(state_path)

    with patch("src.assembled_core.intel.news_ingest.requests.get") as mock_get:
        import requests as req_lib
        mock_get.side_effect = req_lib.ConnectionError("Network unreachable")

        events, is_new = fetcher.fetch_new_events()

    assert events == []
    assert is_new is False
    assert fetcher._state.consecutive_failures == 1


def test_gdelt_fetcher_processes_new_batch(tmp_path):
    state_path = tmp_path / "gdelt_state.json"
    fetcher = GdeltFetcher(state_path)

    gkg_url = "http://data.gdeltproject.org/gdeltv2/20260402123000.gkg.csv.zip"
    lastupdate_text = (
        f"100 abc http://data.gdeltproject.org/gdeltv2/20260402123000.export.CSV.zip\n"
        f"200 def http://data.gdeltproject.org/gdeltv2/20260402123000.mentions.CSV.zip\n"
        f"300 ghi {gkg_url}\n"
    )
    zip_bytes = _make_gkg_zip([SAMPLE_GKG_ROW])

    with patch("src.assembled_core.intel.news_ingest.requests.get") as mock_get:
        # First call: lastupdate.txt
        lastupdate_resp = MagicMock()
        lastupdate_resp.text = lastupdate_text
        lastupdate_resp.raise_for_status.return_value = None

        # Second call: gkg zip
        gkg_resp = MagicMock()
        gkg_resp.content = zip_bytes
        gkg_resp.raise_for_status.return_value = None

        mock_get.side_effect = [lastupdate_resp, gkg_resp]

        events, is_new = fetcher.fetch_new_events()

    assert is_new is True
    assert len(events) >= 1
    assert fetcher._state.last_batch_url == gkg_url
    assert fetcher._state.consecutive_failures == 0
