"""Generic RSS/Atom feed fetcher for the Intel pipeline.

Reads the feed registry from configs/intel/rss_feeds.yaml and fetches
enabled feeds via feedparser. Converts entries to NewsEvent objects with
appropriate SourceTier assignments.

Usage:
    fetcher = RSSFetcher()
    events = fetcher.fetch_all()          # all enabled feeds
    events = fetcher.fetch_feed("axios")  # single feed by id
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import feedparser
import requests

from src.assembled_core.intel.models import NewsEvent, SourceTier

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "intel" / "rss_feeds.yaml"

# Geopolitical keywords for relevance filtering (T2/T3 sources)
_GEO_KEYWORDS = {
    "war", "conflict", "sanctions", "military", "strike", "attack",
    "troops", "missile", "nato", "coup", "crisis", "escalation",
    "invasion", "blockade", "embargo", "shutdown", "explosion",
    "protest", "uprising", "seized", "captured", "offensive",
    "energy", "oil", "gas", "supply", "pipeline", "shortage",
    "tariff", "trade", "deficit", "central bank", "rate", "inflation",
    "debt", "default", "downgrade", "recession",
}


@dataclass
class FeedConfig:
    id: str
    name: str
    url: str
    tier: SourceTier
    focus: str
    enabled: bool = True
    note: str = ""


@dataclass
class RSSFetchState:
    last_seen_ids: dict[str, set[str]] = field(default_factory=dict)  # feed_id → seen entry hashes


def _load_feed_configs(config_path: Path = _CONFIG_PATH) -> list[FeedConfig]:
    """Load feed registry from YAML."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        configs = []
        for entry in raw.get("feeds", []):
            tier_str = entry.get("tier", "T3")
            try:
                tier = SourceTier(tier_str)
            except ValueError:
                tier = SourceTier.T3
            configs.append(FeedConfig(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                url=entry.get("url", ""),
                tier=tier,
                focus=entry.get("focus", "general"),
                enabled=entry.get("enabled", True),
                note=entry.get("note", ""),
            ))
        return configs
    except Exception as exc:
        logger.warning("[WARN] RSSFetcher: failed to load feed config %s: %s", config_path, exc)
        return []


def _parse_entry_date(entry: Any) -> datetime:
    """Extract published datetime from a feedparser entry."""
    # Try published_parsed (struct_time)
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            ts = time.mktime(entry.published_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass
    # Try updated_parsed
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        try:
            ts = time.mktime(entry.updated_parsed)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass
    # Try published string (RFC 2822)
    if hasattr(entry, "published") and entry.published:
        try:
            return parsedate_to_datetime(entry.published).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(tz=timezone.utc)


def _entry_to_news_event(entry: Any, feed_cfg: FeedConfig, now: datetime) -> NewsEvent | None:
    """Convert a feedparser entry to a NewsEvent."""
    title = getattr(entry, "title", "").strip()
    url = getattr(entry, "link", "").strip()
    if not title or not url:
        return None

    published_at = _parse_entry_date(entry)

    # Unique content hash: title + url
    raw = (title + url).encode("utf-8")
    content_hash = hashlib.sha256(raw).hexdigest()[:16]
    event_id = f"rss_{feed_cfg.id}_{content_hash}"

    # Extract tags/keywords from title (simple keyword match)
    title_lower = title.lower()
    keywords = [kw for kw in _GEO_KEYWORDS if kw in title_lower]

    # Extract geo tags from tags field if present
    geo_tags: list[str] = []
    if hasattr(entry, "tags") and entry.tags:
        for tag in entry.tags:
            term = getattr(tag, "term", "") or ""
            if len(term) == 2 and term.isalpha():
                geo_tags.append(term.upper())

    # Entities from author or source
    entities: list[str] = []
    if hasattr(entry, "author") and entry.author:
        entities = [entry.author[:80]]

    return NewsEvent(
        event_id=event_id,
        source_id=feed_cfg.id,
        source_tier=feed_cfg.tier,
        title=title,
        url=url,
        published_at=published_at,
        ingested_at=now,
        geo_tags=geo_tags,
        entities=entities,
        keywords=keywords[:10],
        content_hash=content_hash,
    )


def _is_relevant(event: NewsEvent, feed_cfg: FeedConfig) -> bool:
    """T1 feeds: always relevant. T2/T3: only if keywords match."""
    if feed_cfg.tier == SourceTier.T1:
        return True
    return len(event.keywords) > 0


class RSSFetcher:
    """Multi-feed RSS fetcher that converts entries to NewsEvent objects.

    Maintains a deduplication set per feed so repeated fetches don't
    re-emit already-seen entries.
    """

    def __init__(
        self,
        config_path: Path = _CONFIG_PATH,
        *,
        timeout: int = 20,
        retries: int = 2,
        backoff_base: float = 2.0,
        max_entries_per_feed: int = 50,
    ) -> None:
        self._configs = {cfg.id: cfg for cfg in _load_feed_configs(config_path)}
        self._timeout = timeout
        self._retries = retries
        self._backoff_base = backoff_base
        self._max_entries = max_entries_per_feed
        self._seen: dict[str, set[str]] = {}  # feed_id → set of content_hash

    @property
    def feed_ids(self) -> list[str]:
        return list(self._configs.keys())

    @property
    def enabled_feeds(self) -> list[FeedConfig]:
        return [cfg for cfg in self._configs.values() if cfg.enabled and cfg.url]

    def fetch_feed(
        self,
        feed_id: str,
        *,
        skip_seen: bool = True,
    ) -> list[NewsEvent]:
        """Fetch a single feed by id. Returns [] if feed unknown or disabled."""
        cfg = self._configs.get(feed_id)
        if not cfg:
            logger.warning("[WARN] RSSFetcher: unknown feed id=%s", feed_id)
            return []
        if not cfg.enabled:
            logger.debug("[SKIP] RSSFetcher: feed disabled: %s", feed_id)
            return []
        if not cfg.url:
            logger.debug("[SKIP] RSSFetcher: no URL for feed: %s", feed_id)
            return []
        return self._fetch_one(cfg, skip_seen=skip_seen)

    def fetch_all(self, *, skip_seen: bool = True) -> list[NewsEvent]:
        """Fetch all enabled feeds and return merged deduplicated NewsEvent list."""
        all_events: list[NewsEvent] = []
        for cfg in self.enabled_feeds:
            try:
                events = self._fetch_one(cfg, skip_seen=skip_seen)
                all_events.extend(events)
            except Exception as exc:
                logger.warning("[WARN] RSSFetcher.fetch_all: feed=%s error=%s", cfg.id, exc)
        logger.info("[OK] RSSFetcher.fetch_all: %d events from %d feeds", len(all_events), len(self.enabled_feeds))
        return all_events

    def fetch_by_tier(self, tier: SourceTier, *, skip_seen: bool = True) -> list[NewsEvent]:
        """Fetch only feeds of a specific tier."""
        all_events: list[NewsEvent] = []
        for cfg in self.enabled_feeds:
            if cfg.tier == tier:
                all_events.extend(self._fetch_one(cfg, skip_seen=skip_seen))
        return all_events

    def fetch_by_focus(self, focus: str, *, skip_seen: bool = True) -> list[NewsEvent]:
        """Fetch only feeds matching a focus keyword (partial match)."""
        all_events: list[NewsEvent] = []
        for cfg in self.enabled_feeds:
            if focus.lower() in cfg.focus.lower():
                all_events.extend(self._fetch_one(cfg, skip_seen=skip_seen))
        return all_events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_one(self, cfg: FeedConfig, *, skip_seen: bool) -> list[NewsEvent]:
        """Fetch a single feed with retry + backoff."""
        import time as _time

        feed_data = None
        last_exc = None

        for attempt in range(max(1, self._retries)):
            try:
                # feedparser can handle HTTP itself, but we use requests for
                # consistent retry/timeout behavior
                resp = requests.get(cfg.url, timeout=self._timeout, headers={
                    "User-Agent": "AssembledTradingAI/1.0 (+https://github.com/sake105/Assembled-Trading-AI)"
                })
                if resp.status_code == 429:
                    wait = self._backoff_base ** attempt
                    logger.warning("[WARN] RSSFetcher: 429 rate-limit on %s, waiting %.1fs", cfg.id, wait)
                    _time.sleep(wait)
                    continue
                resp.raise_for_status()
                feed_data = feedparser.parse(resp.content)
                last_exc = None
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._retries - 1:
                    wait = self._backoff_base ** attempt
                    logger.warning("[WARN] RSSFetcher: %s attempt %d failed (%s), retry in %.1fs",
                                   cfg.id, attempt + 1, exc, wait)
                    _time.sleep(wait)

        if feed_data is None:
            logger.warning("[WARN] RSSFetcher: all retries failed for %s: %s", cfg.id, last_exc)
            return []

        if feed_data.bozo and not feed_data.entries:
            logger.warning("[WARN] RSSFetcher: malformed feed %s: %s", cfg.id, feed_data.bozo_exception)
            return []

        now = datetime.now(tz=timezone.utc)
        seen = self._seen.setdefault(cfg.id, set())
        events: list[NewsEvent] = []

        for entry in feed_data.entries[: self._max_entries]:
            try:
                event = _entry_to_news_event(entry, cfg, now)
                if event is None:
                    continue
                if skip_seen and event.content_hash in seen:
                    continue
                if not _is_relevant(event, cfg):
                    continue
                seen.add(event.content_hash)
                events.append(event)
            except Exception as exc:
                logger.debug("[SKIP] RSSFetcher: entry parse error in %s: %s", cfg.id, exc)
                continue

        logger.info("[OK] RSSFetcher: feed=%s tier=%s events=%d", cfg.id, cfg.tier.value, len(events))
        return events
