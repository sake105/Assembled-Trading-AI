"""Integration smoke tests for RSS wiring in run_intel_cycle.py."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from src.assembled_core.intel.models import NewsEvent, SourceTier


def _make_news_event(
    title: str = "Sanctions escalation against Russia", n: int = 0
) -> NewsEvent:
    return NewsEvent(
        event_id=f"rss_test_{n}",
        source_id="reuters_world",
        source_tier=SourceTier.T1,
        title=title,
        url=f"http://test.com/{n}",
        published_at=datetime.now(tz=timezone.utc),
        ingested_at=datetime.now(tz=timezone.utc),
        content_hash=f"hash{n:04d}",
    )


@pytest.mark.phase12
class TestIntelCycleRSSWiring:
    """Smoke tests — verify run_single_cycle() merges RSS events with GDELT."""

    def _build_config(self, rss_events: list[NewsEvent], dry_run: bool = True) -> dict:
        """Minimal config dict for run_single_cycle()."""
        from src.assembled_core.intel.news_cluster import ClusterManager
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex
        from src.assembled_core.intel.health_monitor import HealthMonitor
        from src.assembled_core.intel.models import CrisisMode, CrisisState
        from pathlib import Path
        import tempfile

        tmp = Path(tempfile.mkdtemp())

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_new_events.return_value = ([], False)

        mock_rss = MagicMock()
        mock_rss.fetch_all.return_value = rss_events

        health = HealthMonitor()
        health.register("gdelt", stale_threshold_minutes=30)
        health.register("rss", stale_threshold_minutes=30)

        return {
            "dry_run": dry_run,
            "output_dir": tmp,
            "state_dir": tmp,
            "fetcher": mock_fetcher,
            "dedupe": NewsDedupeIndex(),
            "cluster_mgr": ClusterManager(),
            "health": health,
            "dep_graph": None,
            "crisis_state": CrisisState(
                mode=CrisisMode.NORMAL,
                geo_score=0,
                entered_at=datetime.now(tz=timezone.utc),
            ),
            "rss_fetcher": mock_rss,
            "rss_enabled": True,
        }

    def test_rss_events_merged_with_gdelt(self):
        from scripts.run_intel_cycle import run_single_cycle

        titles = [
            "Sanctions escalation against Russia over Ukraine",
            "War escalation: missile strike reported in Ukraine",
            "Energy supply crisis deepens amid conflict tensions",
        ]
        rss_events = [_make_news_event(title, n=i) for i, title in enumerate(titles)]
        config = self._build_config(rss_events)
        result = run_single_cycle(config)

        assert result["raw_events"] == 3  # 0 GDELT + 3 RSS
        assert result["new_events"] == 3

    def test_no_rss_when_disabled(self):
        from scripts.run_intel_cycle import run_single_cycle

        rss_events = [_make_news_event("War escalation in Ukraine", n=0)]
        config = self._build_config(rss_events)
        config["rss_enabled"] = False  # disable

        result = run_single_cycle(config)
        # With RSS disabled, only GDELT events (0)
        assert result["raw_events"] == 0

    def test_rss_fetch_failure_does_not_crash_cycle(self):
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        config["rss_fetcher"].fetch_all.side_effect = RuntimeError("network timeout")

        result = run_single_cycle(config)
        assert "raw_events" in result  # cycle completes despite RSS failure

    def test_deduplication_across_gdelt_and_rss(self):
        from scripts.run_intel_cycle import run_single_cycle

        event = _make_news_event("Conflict escalation", n=0)
        config = self._build_config([event])

        # Pre-seed dedupe index with the same event
        config["dedupe"].add(event)
        result = run_single_cycle(config)
        assert result["new_events"] == 0  # duplicate filtered

    def test_sector_overlay_lazy_init(self):
        """SectorNewsOverlay is lazy-init'd into config on first cycle run."""
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        run_single_cycle(config)
        assert "_sector_overlay" in config
        assert config["_sector_overlay"] is not None

    def test_alert_engine_lazy_init(self):
        """AlertEngine is lazy-init'd into config on first cycle run."""
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        run_single_cycle(config)
        assert "_alert_engine" in config
        assert config["_alert_engine"] is not None

    def test_macro_calendar_lazy_init(self):
        """MacroCalendar is lazy-init'd into config on first cycle run."""
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        run_single_cycle(config)
        assert "_macro_cal" in config
        assert config["_macro_cal"] is not None

    def test_sentiment_tracker_lazy_init(self):
        """SentimentDriftTracker is lazy-init'd into config on first cycle run."""
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        run_single_cycle(config)
        assert "_sentiment_tracker" in config
        assert config["_sentiment_tracker"] is not None

    def test_ticker_velocity_lazy_init(self):
        """TickerVelocityTracker is lazy-init'd into config on first cycle run."""
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        run_single_cycle(config)
        assert "_ticker_velocity" in config
        assert config["_ticker_velocity"] is not None

    def test_semantic_dedup_lazy_init(self):
        """SemanticDedup is lazy-init'd into config on first cycle run."""
        from scripts.run_intel_cycle import run_single_cycle

        config = self._build_config([])
        run_single_cycle(config)
        assert "_semantic_dedup" in config
        assert config["_semantic_dedup"] is not None

    def test_intel_signal_contains_sector_overlay_key(self):
        """intel_signal.json artifact written with sector_overlay key."""
        from scripts.run_intel_cycle import run_single_cycle
        import json
        from pathlib import Path
        import tempfile

        tmp = Path(tempfile.mkdtemp())
        config = self._build_config(
            [_make_news_event("War escalation in Ukraine", n=0)], dry_run=False
        )
        config["output_dir"] = tmp
        config["state_dir"] = tmp
        run_single_cycle(config)
        artifact_path = tmp / "intel_signal.json"
        assert artifact_path.exists(), "intel_signal.json not written"
        data = json.loads(artifact_path.read_text())
        assert "sector_overlay" in data
        assert "macro" in data
        assert "ticker_surges" in data

    def test_semantic_dedup_catches_near_duplicates(self):
        """SemanticDedup removes near-identical events that hash-dedup misses."""
        from scripts.run_intel_cycle import run_single_cycle
        from src.assembled_core.intel.news_semantic_dedup import SemanticDedup

        # Two events with slightly different titles but same core story
        e1 = _make_news_event("Russia escalates sanctions pressure on Ukraine", n=10)
        e2 = _make_news_event(
            "Russia escalates sanctions pressure on Ukraine today", n=11
        )
        config = self._build_config([e1, e2])
        # Pre-seed semantic dedup with e1
        sem = SemanticDedup(enabled=False, retention_hours=1.0)
        sem.is_duplicate(e1)  # records e1
        config["_semantic_dedup"] = sem
        result = run_single_cycle(config)
        # e2 is a near-duplicate of e1 and should be filtered
        assert result["new_events"] <= 1
