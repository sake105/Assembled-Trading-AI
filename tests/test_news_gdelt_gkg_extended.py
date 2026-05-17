"""Tests for extended GDELT GKG parsing (F6): Tone/V2Persons enrichment."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.news_ingest import (
    GdeltBatchRecord,
    _parse_persons,
    records_to_news_events,
)


@pytest.mark.fast
class TestParsePersons:
    def test_empty(self):
        assert _parse_persons("") == []
        assert _parse_persons(None) == []  # type: ignore[arg-type]

    def test_multiple(self):
        raw = "Vladimir Putin;Joe Biden;Olaf Scholz"
        out = _parse_persons(raw)
        assert out == ["Vladimir Putin", "Joe Biden", "Olaf Scholz"]

    def test_trims_whitespace_and_blanks(self):
        raw = "  Putin ; ;Biden;"
        out = _parse_persons(raw)
        assert out == ["Putin", "Biden"]


@pytest.mark.fast
class TestRecordsToNewsEvents:
    def _rec(
        self,
        tone: float = 0.0,
        persons: list[str] | None = None,
        orgs: list[str] | None = None,
    ) -> GdeltBatchRecord:
        return GdeltBatchRecord(
            record_id="r1",
            date_str="20260420120000",
            source_name="example.com",
            url="https://example.com/article",
            themes=["WAR", "CONFLICT"],
            country_codes=["UA", "RU"],
            organizations=orgs or ["UN", "NATO"],
            persons=persons or ["Putin"],
            tone=tone,
            batch_ts=datetime.now(tz=timezone.utc),
        )

    def test_sentiment_from_tone_negative(self):
        events = records_to_news_events([self._rec(tone=-60.0)])
        assert len(events) == 1
        assert events[0].sentiment_score == -1.0  # clamped

    def test_sentiment_from_tone_mild(self):
        events = records_to_news_events([self._rec(tone=-5.0)])
        assert events[0].sentiment_score == pytest.approx(-0.5)

    def test_sentiment_clamped_to_range(self):
        events = records_to_news_events([self._rec(tone=150.0)])
        assert events[0].sentiment_score == 1.0

    def test_entities_merges_orgs_and_persons(self):
        rec = self._rec(orgs=["UN"], persons=["Biden", "Putin"])
        events = records_to_news_events([rec])
        ents = events[0].entities
        assert "UN" in ents and "Biden" in ents and "Putin" in ents

    def test_entities_deduped_case_insensitive(self):
        rec = self._rec(orgs=["NATO", "nato"], persons=["NATO"])
        events = records_to_news_events([rec])
        ents = [e.lower() for e in events[0].entities]
        assert ents.count("nato") == 1
