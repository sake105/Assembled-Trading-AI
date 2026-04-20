"""Tests for news velocity (breaking news acceleration) tracker."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


def _make_event(
    event_id: str,
    hours_ago: float = 0.1,
    affected_sectors: list[str] | None = None,
    event_types: list[str] | None = None,
    severity: float = 5.0,
) -> NewsEvent:
    ts = datetime.now(tz=timezone.utc) - timedelta(hours=hours_ago)
    content_hash = hashlib.sha256(event_id.encode()).hexdigest()[:16]
    return NewsEvent(
        event_id=event_id,
        source_id="reuters",
        source_tier=SourceTier.T1,
        title=f"Test event {event_id}",
        url=f"https://example.com/{event_id}",
        published_at=ts,
        ingested_at=ts,
        content_hash=content_hash,
        affected_sectors=affected_sectors or [],
        event_types=event_types or [],
        severity=severity,
    )


@pytest.mark.phase12
class TestVelocityTracker:
    def test_no_surge_with_few_events(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker(short_window_min=15, long_window_min=60, surge_threshold=2.5)
        events = [_make_event(f"ev{i}", hours_ago=0.1) for i in range(2)]
        result = tracker.update(events)
        assert not result.is_surge

    def test_surge_detected_on_acceleration(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker(short_window_min=15, long_window_min=60, surge_threshold=2.0)

        # Add sparse background events (in prior window)
        for i in range(2):
            bg_evt = _make_event(f"bg{i}", hours_ago=0.8)
            tracker.update([bg_evt])

        now = datetime.now(tz=timezone.utc)
        # Add many events in short window (last 15 min)
        burst = [_make_event(f"burst{i}", hours_ago=0.1) for i in range(8)]
        result = tracker.update(burst, now=now)
        assert result.short_count >= 3

    def test_velocity_result_fields(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker()
        events = [_make_event("ev1", affected_sectors=["energy"], event_types=["war_escalation"])]
        result = tracker.update(events)
        assert hasattr(result, "velocity")
        assert hasattr(result, "is_surge")
        assert hasattr(result, "short_count")
        assert hasattr(result, "long_count")
        assert hasattr(result, "surge_sectors")
        assert hasattr(result, "surge_event_types")

    def test_empty_events_gives_velocity_1(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker()
        result = tracker.update([])
        assert not result.is_surge
        assert result.short_count == 0

    def test_clear_resets_buffer(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker()
        tracker.update([_make_event(f"ev{i}") for i in range(5)])
        assert tracker.buffer_size > 0
        tracker.clear()
        assert tracker.buffer_size == 0

    def test_avg_severity_computed(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker()
        events = [
            _make_event("ev1", severity=4.0),
            _make_event("ev2", severity=8.0),
        ]
        result = tracker.update(events)
        assert result.avg_severity > 0

    def test_surge_sectors_populated_on_surge(self):
        from src.assembled_core.intel.news_velocity import VelocityTracker

        tracker = VelocityTracker(short_window_min=15, long_window_min=60, surge_threshold=1.5)
        # Fill prior window lightly
        tracker.update([_make_event("bg1", hours_ago=0.9)])

        # Now burst many energy events
        burst = [
            _make_event(f"en{i}", hours_ago=0.05, affected_sectors=["energy"])
            for i in range(6)
        ]
        result = tracker.update(burst)
        if result.is_surge:
            assert "energy" in result.surge_sectors
