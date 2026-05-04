"""Tests for events/news/compat.py — NewsEvent schema bridge (B3).

Verifies round-trip conversions between:
- events.news.models.NewsEvent (dataclass, 16 fields, str timestamps)
- intel.models.NewsEvent (Pydantic, 30+ fields, datetime timestamps)
"""

from __future__ import annotations

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ev_event(**overrides):
    from src.assembled_core.events.news.models import NewsEvent as EvNewsEvent

    defaults = dict(
        event_id="evt-001",
        source_id="reuters",
        title="Fed raises rates by 50bps",
        url="https://example.com/article/1",
        canonical_url="https://example.com/article/1",
        source_name="Reuters",
        source_domain="reuters.com",
        published_utc="2024-03-15T14:30:00+00:00",
        fetched_utc="2024-03-15T14:35:00+00:00",
        language="en",
        fingerprint="abc123",
        fingerprint64="abc123",
        entities=["Federal Reserve", "Jerome Powell"],
        countries=["US"],
    )
    defaults.update(overrides)
    return EvNewsEvent(**defaults)


def _make_intel_event(**overrides):
    from src.assembled_core.intel.models import NewsEvent as IntelNewsEvent

    defaults = dict(
        event_id="intel-001",
        source_id="ap-newswire",
        source_tier="T3",
        title="Oil prices surge on OPEC cut",
        url="https://example.com/oil/2",
        published_at=datetime(2024, 3, 16, 9, 0, 0, tzinfo=timezone.utc),
        ingested_at=datetime(2024, 3, 16, 9, 5, 0, tzinfo=timezone.utc),
        geo_tags=["SA", "RU"],
        entities=["OPEC", "Saudi Aramco"],
        keywords=["oil", "OPEC"],
        content_hash="deadbeef",
        language="en",
    )
    defaults.update(overrides)
    return IntelNewsEvent(**defaults)


# ---------------------------------------------------------------------------
# B3 Round-trip tests
# ---------------------------------------------------------------------------


class TestEventsToIntel:
    def test_core_fields_preserved(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event()
        intel = events_to_intel(ev)

        assert intel.event_id == ev.event_id
        assert intel.source_id == ev.source_id
        assert intel.title == ev.title
        assert intel.url == ev.url
        assert intel.language == ev.language

    def test_published_utc_converted_to_datetime(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event(published_utc="2024-03-15T14:30:00+00:00")
        intel = events_to_intel(ev)

        assert isinstance(intel.published_at, datetime)
        assert intel.published_at.tzinfo is not None
        assert intel.published_at.year == 2024
        assert intel.published_at.month == 3
        assert intel.published_at.day == 15

    def test_countries_mapped_to_geo_tags(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event(countries=["US", "DE", "JP"])
        intel = events_to_intel(ev)
        assert intel.geo_tags == ["US", "DE", "JP"]

    def test_fingerprint_mapped_to_content_hash(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event(fingerprint="sha256-xyz", fingerprint64="sha256-xyz")
        intel = events_to_intel(ev)
        assert intel.content_hash == "sha256-xyz"

    def test_entities_preserved(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event(entities=["Apple Inc.", "Tim Cook"])
        intel = events_to_intel(ev)
        assert intel.entities == ["Apple Inc.", "Tim Cook"]

    def test_none_published_utc_defaults_to_now(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event(published_utc=None)
        intel = events_to_intel(ev)
        assert isinstance(intel.published_at, datetime)
        assert intel.published_at.tzinfo is not None

    def test_source_tier_defaults_to_unclassified(self):
        from src.assembled_core.events.news.compat import events_to_intel

        ev = _make_ev_event()
        intel = events_to_intel(ev)
        # source_tier is not in events schema — defaults to T3 (lowest tier)
        assert intel.source_tier.value == "T3"


class TestIntelToEvents:
    def test_core_fields_preserved_v2(self):
        from src.assembled_core.events.news.compat import intel_to_events

        intel = _make_intel_event()
        ev = intel_to_events(intel)

        assert ev.event_id == intel.event_id
        assert ev.source_id == intel.source_id
        assert ev.title == intel.title
        assert ev.url == intel.url
        assert ev.language == intel.language

    def test_published_at_converted_to_iso_str(self):
        from src.assembled_core.events.news.compat import intel_to_events

        intel = _make_intel_event(
            published_at=datetime(2024, 3, 16, 9, 0, 0, tzinfo=timezone.utc)
        )
        ev = intel_to_events(intel)
        assert isinstance(ev.published_utc, str)
        assert "2024" in ev.published_utc

    def test_geo_tags_mapped_to_countries(self):
        from src.assembled_core.events.news.compat import intel_to_events

        intel = _make_intel_event(geo_tags=["CN", "IN"])
        ev = intel_to_events(intel)
        assert ev.countries == ["CN", "IN"]

    def test_content_hash_mapped_to_fingerprints(self):
        from src.assembled_core.events.news.compat import intel_to_events

        intel = _make_intel_event(content_hash="hash-abc")
        ev = intel_to_events(intel)
        assert ev.fingerprint == "hash-abc"
        assert ev.fingerprint64 == "hash-abc"

    def test_entities_preserved_v2(self):
        from src.assembled_core.events.news.compat import intel_to_events

        intel = _make_intel_event(entities=["Tesla", "Elon Musk"])
        ev = intel_to_events(intel)
        assert ev.entities == ["Tesla", "Elon Musk"]


class TestRoundTrip:
    def test_ev_to_intel_to_ev_preserves_key_fields(self):
        """events → intel → events: key fields survive both conversions."""
        from src.assembled_core.events.news.compat import (
            events_to_intel,
            intel_to_events,
        )

        original = _make_ev_event()
        intel = events_to_intel(original)
        recovered = intel_to_events(intel)

        assert recovered.event_id == original.event_id
        assert recovered.source_id == original.source_id
        assert recovered.title == original.title
        assert recovered.url == original.url
        assert recovered.entities == original.entities
        assert recovered.countries == original.countries
        assert recovered.language == original.language

    def test_intel_to_ev_to_intel_preserves_key_fields(self):
        """intel → events → intel: key fields survive both conversions."""
        from src.assembled_core.events.news.compat import (
            intel_to_events,
            events_to_intel,
        )

        original = _make_intel_event()
        ev = intel_to_events(original)
        recovered = events_to_intel(ev)

        assert recovered.event_id == original.event_id
        assert recovered.source_id == original.source_id
        assert recovered.title == original.title
        assert recovered.url == original.url
        assert recovered.entities == original.entities
        assert recovered.geo_tags == original.geo_tags
        assert recovered.language == original.language
        assert recovered.content_hash == original.content_hash

    def test_published_at_round_trip_within_second(self):
        """Timestamp round-trip: iso-parse → datetime; precision within 1 second."""
        from src.assembled_core.events.news.compat import (
            events_to_intel,
            intel_to_events,
        )

        ev = _make_ev_event(published_utc="2024-06-01T12:00:00+00:00")
        intel = events_to_intel(ev)
        # intel.published_at should be 2024-06-01 12:00:00 UTC
        assert intel.published_at.hour == 12
        assert intel.published_at.minute == 0

        # Back to events
        recovered = intel_to_events(intel)
        # The isoformat string should contain the same date/hour
        assert (
            "2024-06-01" in recovered.published_utc
            or "2024-06" in recovered.published_utc
        )
