"""Tests for disclosures pipeline: schema wrappers and health transitions (DISCL-0, DISCL-1.1)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.assembled_core.events.disclosures import run_disclosures_pipeline
from src.assembled_core.events.disclosures.fetch_edgar import fetch_edgar_form4
from src.assembled_core.events.disclosures.fetch_house_ptr import (
    fetch_house_ptr_filings,
    _compute_pdf_meta,
)
from src.assembled_core.events.disclosures.health import compute_health
from src.assembled_core.events.disclosures.models import DisclosureEvent
from src.assembled_core.events.disclosures.normalize import normalize_raw_item
from src.assembled_core.events.disclosures.triggers import (
    score_disclosure_triggers,
    apply_qc_caps,
)

pytestmark = [pytest.mark.unit, pytest.mark.phase6]

# Minimal Atom XML with two entries (no real network)
ATOM_TWO_ENTRIES = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>SEC Form 4</title>
  <entry>
    <title>Form 4 - ACME Corp</title>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0001234567"/>
    <id>https://www.sec.gov/Archives/edgar/data/1234567/0001234567-22-000001</id>
    <updated>2022-01-15T18:30:00Z</updated>
    <published>2022-01-15T18:30:00Z</published>
  </entry>
  <entry>
    <title>Form 4 - Beta Inc</title>
    <link href="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&amp;CIK=0009876543"/>
    <id>https://www.sec.gov/Archives/edgar/data/9876543/00009876543-22-000002</id>
    <updated>2022-01-16T12:00:00Z</updated>
    <published>2022-01-16T12:00:00Z</published>
  </entry>
</feed>
"""


def test_emit_wrappers_schema_version(tmp_path: Path) -> None:
    """Emitted JSON wrappers have correct schema_version (disclosures.v1, health.v1, triggers.v1, fetch_report.v1)."""

    # No net calls: mock fetch_edgar_form4 and fetch_house_ptr_filings so pipeline does not hit network
    def _mock_fetch_edgar_form4(
        source_id: str,
        cfg: Dict[str, Any],
        fetch_state: Any = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        return (
            [],
            None,
            {"source_id": source_id, "type": "edgar_form4", "ok": True, "items": 0},
        )

    def _mock_fetch_house_ptr(
        source_id: str,
        cfg: Dict[str, Any],
        fetch_state: Any = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        return (
            [],
            None,
            {"source_id": source_id, "type": "house_ptr", "ok": True, "items": 0},
        )

    with (
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_edgar_form4",
            side_effect=_mock_fetch_edgar_form4,
        ),
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_house_ptr_filings",
            side_effect=_mock_fetch_house_ptr,
        ),
    ):
        result = run_disclosures_pipeline(
            sources_path="configs/disclosures/sources.yaml",
            disclosures_path="configs/disclosures/disclosures.yaml",
            cadence="hourly",
            output_dir=tmp_path,
        )
    assert "events" in result
    assert "health" in result

    events_path = tmp_path / "events_latest.json"
    health_path = tmp_path / "health_latest.json"
    triggers_path = tmp_path / "triggers_latest.json"
    fetch_report_path = tmp_path / "fetch_report_latest.json"

    assert events_path.exists()
    assert health_path.exists()
    assert triggers_path.exists()
    assert fetch_report_path.exists()

    events_data = json.loads(events_path.read_text(encoding="utf-8"))
    health_data = json.loads(health_path.read_text(encoding="utf-8"))
    triggers_data = json.loads(triggers_path.read_text(encoding="utf-8"))
    fetch_data = json.loads(fetch_report_path.read_text(encoding="utf-8"))

    assert events_data.get("schema_version") == "disclosures.v1"
    assert "count" in events_data
    assert "items" in events_data

    assert health_data.get("schema_version") == "disclosures.health.v1"
    assert "health" in health_data
    assert health_data["health"].get("status") in ("OK", "DEGRADED", "ERROR")

    assert triggers_data.get("schema_version") == "disclosures.triggers.v1"
    assert triggers_data.get("items") == []

    assert fetch_data.get("schema_version") == "disclosures.fetch_report.v1"
    assert "totals" in fetch_data
    assert "per_source" in fetch_data


def test_health_status_transitions() -> None:
    """Health status: OK when sources_ok >= min; DEGRADED when some fail; ERROR when sources_ok == 0."""
    # OK: one source, no failures
    h = compute_health(
        sources=["src1"],
        items_raw=0,
        items_after_dedupe=0,
        failures=[],
        min_sources_ok=1,
    )
    assert h.status == "OK"

    # DEGRADED: one source failed
    h2 = compute_health(
        sources=["src1", "src2"],
        items_raw=0,
        items_after_dedupe=0,
        failures=[{"source": "src2", "reason": "timeout"}],
        min_sources_ok=1,
    )
    assert h2.status == "DEGRADED"
    assert "One or more sources failed" in h2.notes or any(
        "failed" in n for n in h2.notes
    )

    # ERROR: no sources succeeded
    h3 = compute_health(
        sources=["src1"],
        items_raw=0,
        items_after_dedupe=0,
        failures=[{"source": "src1", "reason": "error"}],
        min_sources_ok=1,
    )
    assert h3.status == "ERROR"
    assert "No sources succeeded" in h3.notes or "sources_ok" in str(h3.notes)


def test_edgar_atom_parsing_extracts_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Atom feed parsing extracts entries with link, published, title (no real net)."""

    class FakeResponse:
        status_code = 200
        content = ATOM_TWO_ENTRIES.encode("utf-8")

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()

    # Patch so fetch_edgar_form4 gets fake requests when it does "import requests"
    mock_requests = MagicMock()
    mock_requests.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mock_requests)

    cfg = {
        "feed_url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&count=100&output=atom",
        "user_agent": "Test/1.0",
        "timeout_s": 5.0,
    }
    items, failure, stats = fetch_edgar_form4("edgar_form4", cfg, fetch_state=None)
    assert failure is None
    assert len(items) == 2
    assert stats.get("ok") is True
    assert stats.get("items") == 2
    for it in items:
        assert "link" in it
        assert "published" in it
        assert "title" in it
    assert "ACME" in items[0].get("title", "") or items[0].get("company", "")
    assert "Beta" in items[1].get("title", "") or items[1].get("company", "")


def test_pipeline_writes_fetch_report_with_edgar_stats(tmp_path: Path) -> None:
    """Pipeline with edgar_form4 source writes fetch_report_latest.json with per_source stats (mocked, no net)."""

    def _mock_fetch_edgar_form4(
        source_id: str,
        cfg: Dict[str, Any],
        fetch_state: Any = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        raw = {
            "title": "Form 4 - Test Co",
            "link": "https://www.sec.gov/Archives/edgar/data/111/0000111-22-000001",
            "published": "2022-01-10T12:00:00Z",
            "company": "Test Co",
            "cik": "0000000111",
            "accession": "0000111-22-000001",
        }
        return (
            [raw],
            None,
            {
                "source_id": source_id,
                "type": "edgar_form4",
                "ok": True,
                "items": 1,
                "http_status": 200,
            },
        )

    def _mock_fetch_house_ptr(
        source_id: str,
        cfg: Dict[str, Any],
        fetch_state: Any = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        return (
            [],
            None,
            {"source_id": source_id, "type": "house_ptr", "ok": True, "items": 0},
        )

    with (
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_edgar_form4",
            side_effect=_mock_fetch_edgar_form4,
        ),
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_house_ptr_filings",
            side_effect=_mock_fetch_house_ptr,
        ),
    ):
        run_disclosures_pipeline(
            sources_path="configs/disclosures/sources.yaml",
            disclosures_path="configs/disclosures/disclosures.yaml",
            cadence="hourly",
            output_dir=tmp_path,
        )

    fetch_report_path = tmp_path / "fetch_report_latest.json"
    assert fetch_report_path.exists()
    data = json.loads(fetch_report_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "disclosures.fetch_report.v1"
    per_source = data.get("per_source") or []
    edgar_sources = [
        s
        for s in per_source
        if s.get("source_id") == "edgar_form4" or s.get("type") == "edgar_form4"
    ]
    assert len(edgar_sources) >= 1
    edgar_stat = edgar_sources[0]
    assert edgar_stat.get("ok") is True
    assert edgar_stat.get("items") == 1
    assert "status" in edgar_stat or "http_status" in edgar_stat or "ok" in edgar_stat


def test_normalize_form4_raw_to_disclosure_event() -> None:
    """Raw item with company/link/cik normalizes to DisclosureEvent with action_type FORM4_FILED and edgar fingerprint."""
    from src.assembled_core.events.disclosures.normalize import now_utc_iso

    raw = {
        "title": "Form 4 - ACME Corp",
        "link": "https://www.sec.gov/Archives/edgar/data/1234567/0001234567-22-000001",
        "published": "2022-01-15T18:30:00Z",
        "company": "ACME Corp",
        "cik": "0001234567",
        "accession": "0001234567-22-000001",
        "source_id": "edgar_form4",
        "source_name": "SEC EDGAR Form 4",
        "source_domain": "sec.gov",
    }
    ev = normalize_raw_item(
        raw,
        source_id="edgar_form4",
        source_name="SEC EDGAR Form 4",
        source_domain="sec.gov",
        fetched_utc=now_utc_iso(),
    )
    assert ev is not None
    assert ev.action_type == "FORM4_FILED"
    assert ev.person_or_entity == "ACME Corp"
    assert ev.ticker is None
    assert ev.notional is None
    assert (
        ev.fingerprint
        == hashlib.sha256(b"edgar_form4|0001234567-22-000001").hexdigest()
    )


# --- House PTR (DISCL-2.1) ---

HOUSE_PTR_RSS_TWO_ITEMS = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>House PTR</title>
  <entry>
    <title>Rep. Smith - Periodic Transaction Report</title>
    <link href="https://disclosures.house.gov/ptr/12345.pdf"/>
    <id>https://disclosures.house.gov/ptr/12345</id>
    <published>2023-06-01T14:00:00Z</published>
  </entry>
  <entry>
    <title>Rep. Jones - PTR</title>
    <link href="https://disclosures.house.gov/ptr/67890.pdf"/>
    <id>https://disclosures.house.gov/ptr/67890</id>
    <updated>2023-06-02T10:00:00Z</updated>
  </entry>
</feed>
"""


def test_house_ptr_rss_parsing_extracts_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RSS/Atom parsing extracts entries with link, published, person, doc_id (no net, download_pdfs false)."""

    class FakeResponse:
        status_code = 200
        content = HOUSE_PTR_RSS_TWO_ITEMS.encode("utf-8")

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse()

    mock_requests = MagicMock()
    mock_requests.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", mock_requests)

    cfg = {
        "index_url": "https://example.com/house-ptr.xml",
        "user_agent": "Test/1.0",
        "timeout_s": 5.0,
        "download_pdfs": False,
        "max_items": 50,
    }
    items, failure, stats = fetch_house_ptr_filings("house_ptr", cfg, fetch_state=None)
    assert failure is None
    assert len(items) == 2
    assert stats.get("ok") is True
    assert stats.get("items") == 2
    for it in items:
        assert "link" in it
        assert "published" in it or it.get("link")
        assert "person" in it or "title" in it
        assert "doc_id" in it or it.get("link")
    assert "Smith" in (items[0].get("person") or items[0].get("title") or "")
    assert "Jones" in (items[1].get("person") or items[1].get("title") or "")


def test_house_ptr_cache_hit_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """With fresh cached_items in fetch_state, no requests.get is called."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    cached = [
        {
            "title": "Cached PTR",
            "link": "https://example.com/1.pdf",
            "published": "2023-01-01T00:00:00Z",
            "person": "Cached",
            "doc_id": "1",
            "raw": {},
        },
    ]
    fetch_state = {
        "cached_utc": now.isoformat(),
        "last_ids": ["1"],
        "cached_items": cached,
    }
    cfg = {
        "index_url": "https://example.com/feed.xml",
        "user_agent": "Test/1.0",
        "cache_minutes": 60,
        "stale_on_error_minutes": 240,
    }
    get_called: List[bool] = []

    def track_get(*args: Any, **kwargs: Any) -> None:
        get_called.append(True)

    mock_requests = MagicMock()
    mock_requests.get = track_get
    monkeypatch.setitem(sys.modules, "requests", mock_requests)

    items, failure, stats = fetch_house_ptr_filings(
        "house_ptr", cfg, fetch_state=fetch_state
    )
    assert len(get_called) == 0
    assert stats.get("cached") is True
    assert failure is None
    assert len(items) == 1
    assert items[0].get("title") == "Cached PTR"


def test_pipeline_fetch_report_includes_house_ptr_stats(tmp_path: Path) -> None:
    """Pipeline with house_ptr source writes fetch_report with house_ptr per_source entry and cached flag (mocked).

    Uses a test-local sources config that activates house_ptr so the mock is actually reached.
    The production sources.yaml disables house_ptr (placeholder index_url), but the plumbing
    must work when the source is active — verified here in isolation.
    """
    # Write a test-local sources config with house_ptr active
    sources_cfg = tmp_path / "sources_test.yaml"
    sources_cfg.write_text(
        "sources:\n"
        "  - source_id: house_ptr\n"
        "    name: House PTR\n"
        "    domain: disclosures.house.gov\n"
        "    type: house_ptr\n"
        "    tier: A\n"
        "    weight: 1.0\n"
        "    active: true\n"
        "  - source_id: edgar_form4\n"
        "    name: SEC EDGAR Form 4\n"
        "    domain: sec.gov\n"
        "    type: edgar_form4\n"
        "    tier: A\n"
        "    weight: 1.0\n"
        "    active: true\n",
        encoding="utf-8",
    )

    def _mock_fetch_house_ptr(
        source_id: str,
        cfg: Dict[str, Any],
        fetch_state: Any = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        raw = {
            "title": "Rep. Test - PTR",
            "link": "https://example.com/ptr.pdf",
            "published": "2023-06-01T12:00:00Z",
            "person": "Rep. Test",
            "doc_id": "ptr",
            "raw": {},
        }
        return (
            [raw],
            None,
            {
                "source_id": source_id,
                "type": "house_ptr",
                "ok": True,
                "items": 1,
                "cached": True,
            },
        )

    def _mock_fetch_edgar(
        source_id: str,
        cfg: Dict[str, Any],
        fetch_state: Any = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        return (
            [],
            None,
            {"source_id": source_id, "type": "edgar_form4", "ok": True, "items": 0},
        )

    with (
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_house_ptr_filings",
            side_effect=_mock_fetch_house_ptr,
        ),
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_edgar_form4",
            side_effect=_mock_fetch_edgar,
        ),
    ):
        run_disclosures_pipeline(
            sources_path=str(sources_cfg),
            disclosures_path="configs/disclosures/disclosures.yaml",
            cadence="hourly",
            output_dir=tmp_path,
        )

    fetch_report_path = tmp_path / "fetch_report_latest.json"
    assert fetch_report_path.exists()
    data = json.loads(fetch_report_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "disclosures.fetch_report.v1"
    per_source = data.get("per_source") or []
    house_ptrs = [
        s
        for s in per_source
        if s.get("type") == "house_ptr" or s.get("source_id") == "house_ptr"
    ]
    assert len(house_ptrs) >= 1
    hp = house_ptrs[0]
    assert hp.get("ok") is True
    assert hp.get("items") == 1
    assert hp.get("cached") is True

    events_path = tmp_path / "events_latest.json"
    assert events_path.exists()
    events_data = json.loads(events_path.read_text(encoding="utf-8"))
    items = events_data.get("items") or []
    house_events = [e for e in items if e.get("action_type") == "HOUSE_PTR_FILED"]
    assert len(house_events) == 1
    assert house_events[0].get("person_or_entity") == "Rep. Test"


def test_normalize_house_ptr_raw_to_disclosure_event() -> None:
    """Raw house_ptr item normalizes to DisclosureEvent with action_type HOUSE_PTR_FILED and house_ptr fingerprint."""
    from src.assembled_core.events.disclosures.normalize import now_utc_iso

    raw = {
        "title": "Rep. Smith - PTR",
        "link": "https://disclosures.house.gov/ptr/12345.pdf",
        "published": "2023-06-01T12:00:00Z",
        "person": "Rep. Smith",
        "doc_id": "12345.pdf",
        "source_id": "house_ptr",
        "source_name": "House PTR",
        "source_domain": "disclosures.house.gov",
        "raw": {},
    }
    ev = normalize_raw_item(
        raw,
        source_id="house_ptr",
        source_name="House PTR",
        source_domain="disclosures.house.gov",
        fetched_utc=now_utc_iso(),
    )
    assert ev is not None
    assert ev.action_type == "HOUSE_PTR_FILED"
    assert ev.person_or_entity == "Rep. Smith"
    assert ev.ticker is None
    assert ev.notional is None
    assert ev.fingerprint == hashlib.sha256(b"house_ptr|12345.pdf").hexdigest()


# --- House PTR PDF metadata (DISCL-2.2) ---


def test_pdf_meta_sha256_computed_when_downloaded(tmp_path: Path) -> None:
    """Helper _compute_pdf_meta computes sha256 for a local file; result has 64-char hex and hashed=True."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content for hashing\n")
    pdf_meta_cfg = {"enabled": True, "compute_sha256": True, "max_mb": 25}
    fetched_utc = "2024-01-15T12:00:00Z"
    meta = _compute_pdf_meta(pdf_path, pdf_meta_cfg, fetched_utc)
    assert meta.get("hashed") is True
    assert "sha256" in meta
    assert len(meta["sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in meta["sha256"])
    assert meta.get("size_bytes") == len(pdf_path.read_bytes())
    assert meta.get("local_path") == str(pdf_path)
    assert meta.get("fetched_utc") == fetched_utc


def test_normalize_includes_pdf_sha_in_fingerprint_key() -> None:
    """Two house_ptr items with same doc_id but different pdf_meta.sha256 get different fingerprints."""
    from src.assembled_core.events.disclosures.normalize import now_utc_iso

    base = {
        "title": "Rep. X - PTR",
        "link": "https://example.com/ptr.pdf",
        "published": "2023-06-01T12:00:00Z",
        "person": "Rep. X",
        "doc_id": "ptr.pdf",
        "source_id": "house_ptr",
        "source_name": "House PTR",
        "source_domain": "disclosures.house.gov",
    }
    raw1 = {**base, "pdf_meta": {"sha256": "a" * 64, "size_bytes": 100, "hashed": True}}
    raw2 = {**base, "pdf_meta": {"sha256": "b" * 64, "size_bytes": 200, "hashed": True}}
    ev1 = normalize_raw_item(
        raw1,
        source_id="house_ptr",
        source_name="House PTR",
        source_domain="disclosures.house.gov",
        fetched_utc=now_utc_iso(),
    )
    ev2 = normalize_raw_item(
        raw2,
        source_id="house_ptr",
        source_name="House PTR",
        source_domain="disclosures.house.gov",
        fetched_utc=now_utc_iso(),
    )
    assert ev1 is not None and ev2 is not None
    assert ev1.fingerprint != ev2.fingerprint
    assert (
        ev1.fingerprint
        == hashlib.sha256(("house_ptr|ptr.pdf|" + "a" * 64).encode("utf-8")).hexdigest()
    )
    assert (
        ev2.fingerprint
        == hashlib.sha256(("house_ptr|ptr.pdf|" + "b" * 64).encode("utf-8")).hexdigest()
    )


# --- Trigger scoring v1 (DISCL-3.1) ---


def _make_event(
    event_id: str = "ev1",
    source_id: str = "edgar_form4",
    source_domain: str = "sec.gov",
    action_type: str = "FORM4_FILED",
    published_utc: str = "2024-01-10T12:00:00Z",
) -> DisclosureEvent:
    return DisclosureEvent(
        event_id=event_id,
        source_id=source_id,
        source_name="Test",
        source_domain=source_domain,
        published_utc=published_utc,
        fetched_utc="2024-01-15T12:00:00Z",
        person_or_entity="ACME",
        ticker=None,
        action_type=action_type,
        notional=None,
        raw={},
        fingerprint="fp1",
    )


def test_disclosures_trigger_scoring_base_severity() -> None:
    """Base severity from config (FORM4_FILED=1, HOUSE_PTR_FILED=1); tier A => evidence_ok, confidence tierA_alone."""
    cfg = {
        "severity": {
            "base_by_action": {"FORM4_FILED": 1, "HOUSE_PTR_FILED": 1},
            "max": 3,
        },
        "confidence": {
            "tierA_alone": 0.85,
            "tierB_two_domains": 0.70,
            "otherwise": 0.40,
        },
        "gating": {"require_evidence_ok": True},
        "ttl": {
            "default_hours": 168,
            "by_action": {"FORM4_FILED": 96, "HOUSE_PTR_FILED": 168},
        },
        "decay": {
            "half_life_hours": 72,
            "min_confidence_floor": 0.25,
            "severity_floor": 0,
        },
    }
    source_meta = {"edgar_form4": {"tier": "A", "domain": "sec.gov"}}
    ev = _make_event(action_type="FORM4_FILED", source_id="edgar_form4")
    now_utc = "2024-01-10T12:00:01Z"  # 1 second after published -> age ~0, no decay
    triggers = score_disclosure_triggers([ev], source_meta, cfg, now_utc)
    assert len(triggers) == 1
    t = triggers[0]
    assert t["severity"] == 1
    assert t["evidence_ok"] is True
    assert t["confidence"] == 0.85
    assert t["action_type"] == "FORM4_FILED"
    assert t["trigger_id"].startswith("dtr_")
    assert len(t["trigger_id"]) == 4 + 12


def test_disclosures_trigger_gating_sets_severity0() -> None:
    """When require_evidence_ok and not evidence_ok (e.g. tier B single), severity=0, confidence=otherwise."""
    cfg = {
        "severity": {"base_by_action": {"HOUSE_PTR_FILED": 1}, "max": 3},
        "confidence": {
            "tierA_alone": 0.85,
            "tierB_two_domains": 0.70,
            "otherwise": 0.40,
        },
        "gating": {"require_evidence_ok": True},
        "ttl": {"default_hours": 168, "by_action": {"HOUSE_PTR_FILED": 168}},
        "decay": {
            "half_life_hours": 72,
            "min_confidence_floor": 0.25,
            "severity_floor": 0,
        },
    }
    source_meta = {"house_ptr": {"tier": "B", "domain": "disclosures.house.gov"}}
    ev = _make_event(
        event_id="ev_ptr",
        source_id="house_ptr",
        action_type="HOUSE_PTR_FILED",
        source_domain="disclosures.house.gov",
    )
    now_utc = "2024-01-10T12:00:01Z"  # age ~0 so no decay
    triggers = score_disclosure_triggers([ev], source_meta, cfg, now_utc)
    assert len(triggers) == 1
    t = triggers[0]
    assert t["evidence_ok"] is False
    assert t["severity"] == 0
    assert abs(t["confidence"] - 0.40) < 0.01


def test_disclosures_trigger_ttl_expires_sets_severity0() -> None:
    """When age_hours >= ttl_hours, severity=0 and confidence decayed."""
    cfg = {
        "severity": {"base_by_action": {"FORM4_FILED": 1}, "max": 3},
        "confidence": {
            "tierA_alone": 0.85,
            "tierB_two_domains": 0.70,
            "otherwise": 0.40,
        },
        "gating": {"require_evidence_ok": True},
        "ttl": {"default_hours": 168, "by_action": {"FORM4_FILED": 96}},
        "decay": {
            "half_life_hours": 72,
            "min_confidence_floor": 0.25,
            "severity_floor": 0,
        },
    }
    source_meta = {"edgar_form4": {"tier": "A", "domain": "sec.gov"}}
    ev = _make_event(published_utc="2024-01-01T00:00:00Z")  # 10 days ago
    now_utc = "2024-01-11T00:00:00Z"  # 240h > 96h TTL
    triggers = score_disclosure_triggers([ev], source_meta, cfg, now_utc)
    assert len(triggers) == 1
    t = triggers[0]
    assert t["severity"] == 0
    assert t["decay"]["age_hours"] >= 96
    assert t["confidence"] <= 0.5


def test_disclosures_qc_caps_severity_when_degraded() -> None:
    """QC cap: DEGRADED => severity capped at degraded_max_severity (1)."""
    triggers_in = [
        {"trigger_id": "dtr_abc", "severity": 2, "confidence": 0.8},
        {"trigger_id": "dtr_def", "severity": 1, "confidence": 0.5},
    ]
    qc_gates = {"degraded_max_severity": 1, "error_max_severity": 0}
    capped = apply_qc_caps(triggers_in, "DEGRADED", qc_gates)
    assert len(capped) == 2
    assert capped[0]["severity"] == 1
    assert capped[1]["severity"] == 1
    capped_err = apply_qc_caps(triggers_in, "ERROR", qc_gates)
    assert capped_err[0]["severity"] == 0
    assert capped_err[1]["severity"] == 0


def test_disclosures_triggers_wrapper_written(tmp_path: Path) -> None:
    """Pipeline writes triggers_latest.json with schema disclosures.triggers.v1 and items when trigger_scoring enabled."""

    def _mock_edgar(
        *args: Any, **kwargs: Any
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        return (
            [],
            None,
            {"source_id": "edgar_form4", "type": "edgar_form4", "ok": True, "items": 0},
        )

    def _mock_house(
        *args: Any, **kwargs: Any
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
        raw = {
            "title": "Rep. X",
            "link": "https://x.com/1.pdf",
            "published": "2024-01-10T12:00:00Z",
            "person": "Rep. X",
            "doc_id": "1",
            "raw": {},
        }
        return (
            [raw],
            None,
            {"source_id": "house_ptr", "type": "house_ptr", "ok": True, "items": 1},
        )

    with (
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_edgar_form4",
            side_effect=_mock_edgar,
        ),
        patch(
            "src.assembled_core.events.disclosures.pipeline.fetch_house_ptr_filings",
            side_effect=_mock_house,
        ),
    ):
        run_disclosures_pipeline(
            sources_path="configs/disclosures/sources.yaml",
            disclosures_path="configs/disclosures/disclosures.yaml",
            output_dir=tmp_path,
        )
    triggers_path = tmp_path / "triggers_latest.json"
    assert triggers_path.exists()
    data = json.loads(triggers_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "disclosures.triggers.v1"
    assert "generated_utc" in data
    assert "cadence" in data
    assert "count" in data
    assert "items" in data
    assert isinstance(data["items"], list)
    assert data["count"] == len(data["items"])
    if data["items"]:
        t = data["items"][0]
        assert "trigger_id" in t
        assert "severity" in t
        assert "confidence" in t
        assert t.get("schema_version") == "disclosures.trigger.v1"
    health_path = tmp_path / "health_latest.json"
    health_data = json.loads(health_path.read_text(encoding="utf-8"))
    assert "metrics" in health_data.get("health", {})
    assert "triggers" in health_data["health"]["metrics"]
    assert "trigger_count" in health_data["health"]["metrics"]["triggers"]
    assert "max_severity" in health_data["health"]["metrics"]["triggers"]


# --- Disclosures triggers into TradingContext (DISCL-4.1) ---

from src.assembled_core.intel.disclosures_triggers_loader import (
    DisclosuresTriggerSnapshot,
    load_disclosures_triggers,
)


def test_load_disclosures_triggers_valid(tmp_path: Path) -> None:
    """Valid triggers_latest.json loads to snapshot with summary (max_severity, count_sev1plus, count_sev2plus)."""
    payload = {
        "schema_version": "disclosures.triggers.v1",
        "generated_utc": "2024-01-15T12:00:00Z",
        "cadence": "hourly",
        "count": 2,
        "items": [
            {"trigger_id": "dtr_abc", "severity": 2, "event_id": "e1"},
            {"trigger_id": "dtr_def", "severity": 1, "event_id": "e2"},
        ],
    }
    (tmp_path / "triggers_latest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    snap = load_disclosures_triggers(tmp_path / "triggers_latest.json")
    assert snap.generated_utc == "2024-01-15T12:00:00Z"
    assert len(snap.triggers) == 2
    assert snap.summary["max_severity"] == 2
    assert snap.summary["count_sev1plus"] == 2
    assert snap.summary["count_sev2plus"] == 1


def test_load_disclosures_triggers_missing_invalid(tmp_path: Path) -> None:
    """Missing file or invalid schema -> empty snapshot (generated_utc empty)."""
    missing = load_disclosures_triggers(tmp_path / "nonexistent.json")
    assert missing.generated_utc == ""
    assert missing.triggers == []
    assert missing.summary["max_severity"] == 0

    (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
    invalid = load_disclosures_triggers(tmp_path / "bad.json")
    assert invalid.generated_utc == ""

    (tmp_path / "wrong_schema.json").write_text(
        json.dumps({"schema_version": "other.v1", "items": []}), encoding="utf-8"
    )
    wrong = load_disclosures_triggers(tmp_path / "wrong_schema.json")
    assert wrong.generated_utc == ""

    (tmp_path / "no_list.json").write_text(
        json.dumps(
            {"schema_version": "disclosures.triggers.v1", "items": "not a list"}
        ),
        encoding="utf-8",
    )
    no_list = load_disclosures_triggers(tmp_path / "no_list.json")
    assert no_list.generated_utc == ""


def test_context_sets_degraded_flag(tmp_path: Path) -> None:
    """When intel.disclosures_triggers.enabled and path missing/invalid, context gets DEGRADED flag and no snapshot."""
    import pandas as pd
    from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
    from src.assembled_core.pipeline.trading_cycle_v2 import run_trading_cycle

    policy_with_intel = {
        "intel": {
            "disclosures_triggers": {
                "enabled": True,
                "path": "output/intel/disclosures/triggers_latest.json",
            },
        },
        "risk_state_machine": {
            "enabled": True,
            "state_path": str(tmp_path / "risk_state.json"),
            "persistence": {"mode": "live"},
        },
    }
    with (
        patch(
            "src.assembled_core.pipeline.trading_cycle.load_policy",
            return_value=policy_with_intel,
        ),
        patch(
            "src.assembled_core.pipeline.trading_cycle.get_base_dir",
            return_value=tmp_path,
        ),
    ):
        df = pd.DataFrame(
            [{"timestamp": pd.Timestamp("2024-01-01"), "symbol": "X", "close": 100.0}]
        )
        ctx = TradingContext(
            prices=df,
            as_of=pd.Timestamp("2024-01-01"),
            signal_fn=lambda _: pd.DataFrame(
                columns=["timestamp", "symbol", "direction", "score"]
            ),
            position_sizing_fn=lambda s, c: pd.DataFrame(
                columns=["symbol", "target_weight", "target_qty"]
            ),
        )
        run_trading_cycle(ctx)
        assert ctx.disclosures_triggers is None
        assert ctx.intel_health_flags.get("intel_disclosures_triggers") == "DEGRADED"


# --- Disclosures confirm boost (DISCL-4.2) archived — section removed ---
pytest.importorskip("src.assembled_core.risk.disclosures_confirm")
from src.assembled_core.risk.disclosures_confirm import (
    apply_disclosures_confirm,
)  # noqa: E402


def test_disclosures_boost_applies_when_sev_ge_1() -> None:
    """When disclosures triggers max_severity >= min_severity (1), geo_confidence is boosted and boost block set."""

    class Ctx:
        news_geo = {"geo_score": 1, "geo_confidence": 0.60, "state_hint": "WATCH"}
        disclosures_triggers = DisclosuresTriggerSnapshot(
            generated_utc="2024-01-15T12:00:00Z",
            triggers=[{"trigger_id": "dtr_1", "severity": 1}],
            summary={"max_severity": 1, "count_sev1plus": 1, "count_sev2plus": 0},
        )
        intel_health_flags = {}

    policy = {
        "disclosures_confirm": {
            "enabled": True,
            "boost": {
                "min_severity": 1,
                "add_confidence": 0.10,
                "max_confidence": 0.95,
            },
        },
    }
    ctx = Ctx()
    apply_disclosures_confirm(ctx, policy)
    assert ctx.news_geo["geo_confidence"] == 0.70
    assert ctx.news_geo.get("boost") == {
        "source": "disclosures",
        "added": 0.10,
        "max_discl_sev": 1,
    }


def test_disclosures_no_boost_when_degraded_or_missing() -> None:
    """When disclosures triggers missing or intel_disclosures_triggers DEGRADED, no boost applied."""

    class Ctx:
        news_geo = {"geo_score": 1, "geo_confidence": 0.60, "state_hint": "WATCH"}
        disclosures_triggers = None
        intel_health_flags = {}

    policy = {
        "disclosures_confirm": {
            "enabled": True,
            "boost": {
                "min_severity": 1,
                "add_confidence": 0.10,
                "max_confidence": 0.95,
            },
        },
    }
    ctx = Ctx()
    apply_disclosures_confirm(ctx, policy)
    assert ctx.news_geo["geo_confidence"] == 0.60
    assert "boost" not in ctx.news_geo

    # DEGRADED flag -> no boost
    ctx2 = Ctx()
    ctx2.disclosures_triggers = DisclosuresTriggerSnapshot(
        generated_utc="2024-01-15T12:00:00Z",
        triggers=[{"severity": 2}],
        summary={"max_severity": 2, "count_sev1plus": 1, "count_sev2plus": 1},
    )
    ctx2.intel_health_flags = {"intel_disclosures_triggers": "DEGRADED"}
    apply_disclosures_confirm(ctx2, policy)
    assert ctx2.news_geo["geo_confidence"] == 0.60
    assert "boost" not in ctx2.news_geo
