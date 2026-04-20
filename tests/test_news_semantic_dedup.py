"""Tests for SemanticDedup (lexical fallback path only — gated backend skipped)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_semantic_dedup import SemanticDedup


def _evt(event_id: str, title: str, ts: datetime | None = None) -> NewsEvent:
    ts = ts or datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=event_id,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=title,
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256((title + event_id).encode()).hexdigest()[:16],
    )


@pytest.mark.phase12
class TestSemanticDedupLexical:
    def test_backend_defaults_to_lexical(self):
        d = SemanticDedup(enabled=False)
        assert d.backend == "lexical"

    def test_first_event_not_duplicate(self):
        d = SemanticDedup(enabled=False)
        assert d.is_duplicate(_evt("e1", "Russia attacks Ukrainian grid")) is False

    def test_near_identical_is_duplicate(self):
        d = SemanticDedup(enabled=False)
        d.is_duplicate(_evt("e1", "Russia attacks Ukrainian grid"))
        assert d.is_duplicate(
            _evt("e2", "Russia attacks Ukrainian grid"), threshold=0.6,
        ) is True

    def test_different_topic_not_duplicate(self):
        d = SemanticDedup(enabled=False)
        d.is_duplicate(_evt("e1", "Apple unveils new iPhone"))
        assert d.is_duplicate(
            _evt("e2", "Russia attacks Ukrainian grid"), threshold=0.6,
        ) is False

    def test_prune_drops_stale_entries(self):
        d = SemanticDedup(enabled=False, retention_hours=1.0)
        old = datetime.now(tz=timezone.utc) - timedelta(hours=5)
        d.is_duplicate(_evt("old", "old headline", ts=old), now=old)
        assert d.size() == 1
        dropped = d.prune(now=datetime.now(tz=timezone.utc))
        assert dropped == 1
        assert d.size() == 0

    def test_empty_title_not_duplicate(self):
        d = SemanticDedup(enabled=False)
        assert d.is_duplicate(_evt("e1", "")) is False
        assert d.size() == 0
