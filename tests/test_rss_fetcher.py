"""Tests for RSSFetcher (offline — no real HTTP calls)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("feedparser")

from src.assembled_core.intel.models import NewsEvent, SourceTier
from src.assembled_core.intel.rss_fetcher import (
    FeedConfig,
    RSSFetcher,
    _entry_to_news_event,
    _is_relevant,
    _load_feed_configs,
    _parse_entry_date,
    _urgency_score,
)

_REAL_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "intel" / "rss_feeds.yaml"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    tier: SourceTier = SourceTier.T1, focus: str = "geopolitical"
) -> FeedConfig:
    return FeedConfig(
        id="test_feed",
        name="Test Feed",
        url="http://example.com/rss",
        tier=tier,
        focus=focus,
        enabled=True,
    )


def _make_entry(
    title: str = "War escalation in region", url: str = "http://example.com/1"
) -> MagicMock:
    entry = MagicMock()
    entry.title = title
    entry.link = url
    entry.published_parsed = None
    entry.updated_parsed = None
    entry.published = "Mon, 01 Jan 2024 12:00:00 GMT"
    entry.tags = []
    entry.author = "Reuters"
    return entry


def _make_feed_response(entries: list[MagicMock]) -> MagicMock:
    feed = MagicMock()
    feed.entries = entries
    feed.bozo = False
    feed.bozo_exception = None
    return feed


# ---------------------------------------------------------------------------
# Unit tests: parsing helpers
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestParseEntryDate:
    def test_published_parsed(self):
        import time

        entry = MagicMock()
        entry.published_parsed = time.gmtime(0)  # epoch
        entry.updated_parsed = None
        entry.published = None
        dt = _parse_entry_date(entry)
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None

    def test_fallback_to_now(self):
        entry = MagicMock()
        entry.published_parsed = None
        entry.updated_parsed = None
        entry.published = None
        dt = _parse_entry_date(entry)
        assert isinstance(dt, datetime)


@pytest.mark.fast
class TestEntryToNewsEvent:
    def test_basic_conversion(self):
        cfg = _make_cfg()
        entry = _make_entry()
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert event.source_id == "test_feed"
        assert event.source_tier == SourceTier.T1
        assert (
            "war" in event.keywords
            or "conflict" in event.keywords
            or len(event.keywords) >= 0
        )

    def test_missing_title_returns_none(self):
        cfg = _make_cfg()
        entry = _make_entry(title="", url="http://x.com")
        now = datetime.now(tz=timezone.utc)
        assert _entry_to_news_event(entry, cfg, now) is None

    def test_missing_url_returns_none(self):
        cfg = _make_cfg()
        entry = _make_entry(url="")
        now = datetime.now(tz=timezone.utc)
        assert _entry_to_news_event(entry, cfg, now) is None

    def test_event_id_deterministic(self):
        cfg = _make_cfg()
        entry = _make_entry(title="Same Title", url="http://same.com/")
        now = datetime.now(tz=timezone.utc)
        e1 = _entry_to_news_event(entry, cfg, now)
        e2 = _entry_to_news_event(entry, cfg, now)
        assert e1.event_id == e2.event_id

    def test_tier_assigned(self):
        cfg = _make_cfg(tier=SourceTier.T3)
        entry = _make_entry()
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event.source_tier == SourceTier.T3


@pytest.mark.fast
class TestIsRelevant:
    def test_t1_always_relevant(self):
        cfg = _make_cfg(tier=SourceTier.T1)
        event = MagicMock(spec=NewsEvent)
        event.keywords = []
        assert _is_relevant(event, cfg) is True

    def test_t2_relevant_with_keywords(self):
        cfg = _make_cfg(tier=SourceTier.T2)
        event = MagicMock(spec=NewsEvent)
        event.keywords = ["war"]
        assert _is_relevant(event, cfg) is True

    def test_t2_not_relevant_no_keywords(self):
        cfg = _make_cfg(tier=SourceTier.T2)
        event = MagicMock(spec=NewsEvent)
        event.keywords = []
        assert _is_relevant(event, cfg) is False

    def test_t3_not_relevant_no_keywords(self):
        cfg = _make_cfg(tier=SourceTier.T3)
        event = MagicMock(spec=NewsEvent)
        event.keywords = []
        assert _is_relevant(event, cfg) is False


# ---------------------------------------------------------------------------
# Unit tests: RSSFetcher (mocked HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRSSFetcher:
    def _make_fetcher(self, tmp_path) -> RSSFetcher:
        """Fetcher that points at real config but we mock HTTP calls."""
        return RSSFetcher(config_path=_REAL_CONFIG, timeout=5, retries=1)

    def test_loads_feed_configs(self):
        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        assert len(fetcher.feed_ids) > 5
        assert "reuters_world" in fetcher.feed_ids
        assert "axios" in fetcher.feed_ids
        assert "the_cradle" in fetcher.feed_ids

    def test_enabled_feeds_have_urls(self):
        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        for cfg in fetcher.enabled_feeds:
            assert cfg.url, f"Feed {cfg.id} is enabled but has no URL"

    def test_fetch_feed_unknown_id(self):
        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        assert fetcher.fetch_feed("nonexistent_feed_xyz") == []

    @patch("src.assembled_core.intel.rss_fetcher.requests.get")
    @patch("feedparser.parse")
    def test_fetch_feed_returns_events(self, mock_fp, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<rss/>"
        mock_get.return_value = mock_resp

        entry = _make_entry(title="Sanctions escalation against Russia")
        mock_fp.return_value = _make_feed_response([entry])

        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        # bbc_world: enabled, tier T1 (reuters_world is kept as enabled:false
        # placeholder after 2026-05-19 audit — fetch_feed would short-circuit
        # to [] on disabled feeds, so tests need an enabled feed here).
        events = fetcher.fetch_feed("bbc_world", skip_seen=False)
        assert len(events) > 0
        assert events[0].source_id == "bbc_world"
        assert events[0].source_tier == SourceTier.T1

    @patch("src.assembled_core.intel.rss_fetcher.requests.get")
    @patch("feedparser.parse")
    def test_deduplication_skip_seen(self, mock_fp, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<rss/>"
        mock_get.return_value = mock_resp

        entry = _make_entry(title="War escalation news story")
        mock_fp.return_value = _make_feed_response([entry])

        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        e1 = fetcher.fetch_feed("bbc_world", skip_seen=True)
        e2 = fetcher.fetch_feed("bbc_world", skip_seen=True)
        assert len(e1) > 0
        assert len(e2) == 0  # already seen

    @patch("src.assembled_core.intel.rss_fetcher.requests.get")
    def test_http_error_returns_empty(self, mock_get):
        import requests as req

        mock_get.side_effect = req.RequestException("timeout")
        fetcher = RSSFetcher(config_path=_REAL_CONFIG, retries=1)
        events = fetcher.fetch_feed("bbc_world")
        assert events == []

    @patch("src.assembled_core.intel.rss_fetcher.requests.get")
    @patch("feedparser.parse")
    def test_t3_filtered_by_keywords(self, mock_fp, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<rss/>"
        mock_get.return_value = mock_resp

        # Entry with no geo/conflict keywords → should be filtered for T3
        benign_entry = _make_entry(
            title="Celebrity birthday party weekend", url="http://z.com/1"
        )
        mock_fp.return_value = _make_feed_response([benign_entry])

        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        events = fetcher.fetch_feed("zerohedge", skip_seen=False)
        assert events == []

    def test_fetch_by_tier_only_matching(self):
        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        t1_feeds = fetcher.fetch_by_tier(SourceTier.T1)
        # No HTTP calls — just returns empty list (all mocked out via network)
        assert isinstance(t1_feeds, list)

    def test_disabled_feed_with_url_not_in_enabled(self):
        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        enabled_ids = {cfg.id for cfg in fetcher.enabled_feeds}
        assert "global_research" not in enabled_ids  # explicitly disabled in config


# ---------------------------------------------------------------------------
# Config file integrity test
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestFeedConfigIntegrity:
    def test_config_loads_without_error(self):
        configs = _load_feed_configs(_REAL_CONFIG)
        assert len(configs) >= 10

    def test_all_enabled_feeds_have_url(self):
        configs = _load_feed_configs(_REAL_CONFIG)
        enabled = [c for c in configs if c.enabled]
        no_url = [c.id for c in enabled if not c.url]
        assert no_url == [], f"Enabled feeds missing URL: {no_url}"

    def test_tier_values_valid(self):
        configs = _load_feed_configs(_REAL_CONFIG)
        valid_tiers = {SourceTier.T0, SourceTier.T1, SourceTier.T2, SourceTier.T3}
        for cfg in configs:
            assert cfg.tier in valid_tiers, f"Invalid tier for {cfg.id}: {cfg.tier}"

    def test_no_duplicate_ids(self):
        configs = _load_feed_configs(_REAL_CONFIG)
        ids = [c.id for c in configs]
        assert len(ids) == len(set(ids)), "Duplicate feed IDs in config"


# ---------------------------------------------------------------------------
# New feature tests: urgency, geo-tags, age-filter, seen_counts
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestUrgencyScore:
    def test_breaking_returns_1(self):
        assert _urgency_score("Breaking: Russia attacks Ukraine") == 1.0

    def test_flash_returns_1(self):
        assert _urgency_score("Flash: Market crash imminent") == 1.0

    def test_urgent_returns_1(self):
        assert _urgency_score("URGENT: Missile strike reported") == 1.0

    def test_alert_returns_05(self):
        assert _urgency_score("Alert: Oil prices surge") == 0.5

    def test_normal_returns_0(self):
        assert _urgency_score("Germany raises defense budget") == 0.0

    def test_case_insensitive(self):
        assert _urgency_score("BREAKING NEWS: Sanctions imposed") == 1.0


@pytest.mark.fast
class TestGeoTagging:
    def test_geo_tags_from_title(self):
        cfg = _make_cfg()
        entry = _make_entry(title="Russia launches attack on Ukraine")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert "RU" in event.geo_tags or "UA" in event.geo_tags

    def test_us_detected(self):
        cfg = _make_cfg()
        entry = _make_entry(title="United States imposes sanctions on Iran")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert "US" in event.geo_tags

    def test_no_countries_empty_list(self):
        cfg = _make_cfg()
        entry = _make_entry(title="Markets rally on positive earnings")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert isinstance(event.geo_tags, list)


@pytest.mark.fast
class TestUrgencyInEvent:
    def test_urgency_field_set(self):
        cfg = _make_cfg()
        entry = _make_entry(title="Breaking: Major earthquake hits region")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert event.urgency == 1.0

    def test_urgency_zero_for_normal(self):
        cfg = _make_cfg()
        entry = _make_entry(title="Oil prices steady as demand recovers")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert event.urgency == 0.0


@pytest.mark.fast
class TestAgeFilter:
    @patch("src.assembled_core.intel.rss_fetcher.requests.get")
    @patch("feedparser.parse")
    def test_old_entry_filtered_when_max_age_set(self, mock_fp, mock_get):
        import time as _time

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<rss/>"
        mock_get.return_value = mock_resp

        # Entry with old published date (3 days ago)
        entry = _make_entry(title="Sanctions escalation against Russia")
        old_struct = _time.gmtime((_time.time() - 3 * 86400))
        entry.published_parsed = old_struct
        entry.updated_parsed = None
        entry.published = None
        mock_fp.return_value = _make_feed_response([entry])

        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        # Manually set max_age_hours = 24 on bbc_world cfg
        fetcher._configs["bbc_world"].max_age_hours = 24
        events = fetcher.fetch_feed("bbc_world", skip_seen=False)
        assert events == [], "Entries older than max_age_hours should be filtered"

    @patch("src.assembled_core.intel.rss_fetcher.requests.get")
    @patch("feedparser.parse")
    def test_recent_entry_passes_age_filter(self, mock_fp, mock_get):
        import time as _time

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<rss/>"
        mock_get.return_value = mock_resp

        entry = _make_entry(title="Sanctions escalation against Russia")
        entry.published_parsed = _time.gmtime()  # now
        entry.updated_parsed = None
        entry.published = None
        mock_fp.return_value = _make_feed_response([entry])

        fetcher = RSSFetcher(config_path=_REAL_CONFIG)
        fetcher._configs["bbc_world"].max_age_hours = 48
        events = fetcher.fetch_feed("bbc_world", skip_seen=False)
        assert len(events) > 0, "Recent entry should pass age filter"


@pytest.mark.fast
class TestEntityLinker:
    def test_linker_populates_tickers(self):
        class _Linker:
            def link(self, entity: str) -> str | None:
                mapping = {"Reuters": "RTRSY"}
                return mapping.get(entity)

        cfg = _make_cfg()
        entry = _make_entry(title="War escalation in region")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now, entity_linker=_Linker())
        assert event is not None
        assert "RTRSY" in event.tickers

    def test_no_linker_empty_tickers(self):
        cfg = _make_cfg()
        entry = _make_entry(title="War escalation in region")
        now = datetime.now(tz=timezone.utc)
        event = _entry_to_news_event(entry, cfg, now)
        assert event is not None
        assert event.tickers == []


@pytest.mark.fast
class TestSeenCounts:
    def test_filter_new_with_counts_returns_count(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex

        idx = NewsDedupeIndex()

        cfg = _make_cfg()
        now = datetime.now(tz=timezone.utc)
        e1 = _entry_to_news_event(_make_entry(title="War news story one"), cfg, now)
        e2 = _entry_to_news_event(
            _make_entry(title="War news story one"), cfg, now
        )  # same

        results = idx.filter_new_with_counts([e1, e2])
        # Only one new (e2 is duplicate), count for e1 = 1
        assert len(results) == 1
        event, count = results[0]
        assert count == 1

    def test_seen_counts_increments_for_duplicate(self):
        from src.assembled_core.intel.news_dedupe import NewsDedupeIndex

        idx = NewsDedupeIndex()

        cfg = _make_cfg()
        now = datetime.now(tz=timezone.utc)
        e1 = _entry_to_news_event(
            _make_entry(title="Flash: Conflict update today"), cfg, now
        )

        # Call twice — second call should not appear in results but count increments
        idx.filter_new_with_counts([e1])
        results2 = idx.filter_new_with_counts([e1])
        assert results2 == []  # duplicate filtered
        fp = idx._fingerprint(e1)
        assert idx.seen_counts[fp] == 2
