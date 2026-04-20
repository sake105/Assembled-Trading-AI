"""Tests for require_corroboration gate (F10)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.news_position_bridge import (
    PositionSignal,
    require_corroboration,
)


def _evt(source_id: str, tier: SourceTier) -> NewsEvent:
    ts = datetime.now(tz=timezone.utc)
    return NewsEvent(
        event_id=f"e_{source_id}",
        source_id=source_id,
        source_tier=tier,
        title=f"hl {source_id}",
        url=f"https://example.com/{source_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=hashlib.sha256(source_id.encode()).hexdigest()[:16],
    )


def _signal() -> PositionSignal:
    return PositionSignal(
        signal_id="ps_test",
        source_cluster_id="cls_1",
        direction="short",
        confidence=0.8,
        affected_assets=["XOP"],
    )


@pytest.mark.phase12
class TestRequireCorroboration:
    def test_none_signal_returns_none(self):
        assert require_corroboration(None, []) is None  # type: ignore[arg-type]

    def test_passes_with_two_distinct_T1(self):
        sig = _signal()
        evs = [_evt("reuters", SourceTier.T1), _evt("ap", SourceTier.T1)]
        assert require_corroboration(sig, evs) is sig

    def test_blocks_with_one_source(self):
        sig = _signal()
        evs = [_evt("reuters", SourceTier.T1)]
        assert require_corroboration(sig, evs) is None

    def test_blocks_when_duplicate_source(self):
        sig = _signal()
        evs = [_evt("reuters", SourceTier.T1), _evt("reuters", SourceTier.T1)]
        assert require_corroboration(sig, evs) is None

    def test_blocks_when_all_T2(self):
        sig = _signal()
        evs = [_evt("blog1", SourceTier.T2), _evt("blog2", SourceTier.T2)]
        assert require_corroboration(sig, evs) is None

    def test_T0_counts(self):
        sig = _signal()
        evs = [_evt("ofac", SourceTier.T0), _evt("reuters", SourceTier.T1)]
        assert require_corroboration(sig, evs) is sig

    def test_custom_threshold(self):
        sig = _signal()
        evs = [_evt("reuters", SourceTier.T1), _evt("ap", SourceTier.T1), _evt("bbc", SourceTier.T1)]
        assert require_corroboration(sig, evs, min_independent_high_tier=3) is sig
        assert require_corroboration(sig, evs, min_independent_high_tier=4) is None
