"""News event deduplication — prevents same event from triggering multiple times."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
import time
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs

from src.assembled_core.intel.models import NewsEvent

logger = logging.getLogger(__name__)

# Query parameters to strip (tracking / referral params)
_STRIP_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_reader", "utm_name",
    "ref", "source", "fbclid", "gclid", "msclkid",
    "mc_cid", "mc_eid",
})


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------


def canonical_url(url: str) -> str:
    """
    Canonicalize a URL for deduplication:
    - Lowercase scheme and host
    - Strip known tracking query params (utm_*, ref, source, fbclid, gclid, …)
    - Keep path as-is
    """
    try:
        parsed = urlparse(url)
        # Lowercase scheme and netloc
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Filter query params
        qs = parse_qs(parsed.query, keep_blank_values=False)
        filtered_qs = {k: v for k, v in qs.items() if k.lower() not in _STRIP_PARAMS}
        new_query = urlencode(filtered_qs, doseq=True) if filtered_qs else ""

        canonical = urlunparse((scheme, netloc, parsed.path, parsed.params, new_query, ""))
        return canonical
    except Exception:
        return url.lower()


# ---------------------------------------------------------------------------
# Content fingerprint
# ---------------------------------------------------------------------------


def content_fingerprint(title: str, source: str) -> str:
    """
    Produce a short fingerprint for near-duplicate detection.

    Normalizes title (lowercase, strip punctuation, collapse whitespace),
    takes first 60 chars, appends source, returns sha256[:12].
    """
    translator = str.maketrans("", "", string.punctuation)
    normalized = title.lower().translate(translator)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    key = normalized[:60] + "|" + source.lower()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Deduplication index
# ---------------------------------------------------------------------------


class NewsDedupeIndex:
    """
    In-memory + optionally file-persistent deduplication index.

    Two-layer deduplication:
    1. Exact event_id match
    2. Content fingerprint match (near-duplicate detection)

    Entries are stored in insertion order (OrderedDict) with a monotonic timestamp.
    Eviction is time-based via evict_older_than(); size-based fallback removes the
    oldest entries when max_size is exceeded.
    """

    DEFAULT_TTL = timedelta(days=7)

    def __init__(
        self,
        persist_path: str | Path | None = None,
        max_size: int = 10_000,
        ttl: timedelta | None = None,
    ) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_size = max_size
        self._ttl = ttl if ttl is not None else self.DEFAULT_TTL
        # key -> monotonic timestamp of first insertion
        self.seen_event_ids: OrderedDict[str, float] = OrderedDict()
        self.seen_fingerprints: OrderedDict[str, float] = OrderedDict()
        # fingerprint → how many distinct sources reported the same story
        self.seen_counts: dict[str, int] = {}
        # URL deduplication (canonical URLs)
        self.seen_urls: set[str] = set()

        if self._persist_path and self._persist_path.exists():
            self.load()

    def _fingerprint(self, event: NewsEvent) -> str:
        return content_fingerprint(event.title, event.source_id)

    def evict_older_than(self, ttl: timedelta | None = None) -> int:
        """Remove entries older than ttl. Returns total number of evicted entries."""
        cutoff = time.monotonic() - (ttl or self._ttl).total_seconds()
        count = 0
        while self.seen_event_ids:
            oldest_key, oldest_ts = next(iter(self.seen_event_ids.items()))
            if oldest_ts > cutoff:
                break
            del self.seen_event_ids[oldest_key]
            count += 1
        while self.seen_fingerprints:
            oldest_key, oldest_ts = next(iter(self.seen_fingerprints.items()))
            if oldest_ts > cutoff:
                break
            del self.seen_fingerprints[oldest_key]
            count += 1
        if count:
            logger.debug("[OK] NewsDedupeIndex: evicted %d stale entries", count)
        return count

    def is_duplicate(self, event: NewsEvent) -> bool:
        """Return True if event_id, URL, OR content fingerprint was already seen."""
        if event.event_id in self.seen_event_ids:
            return True
        if event.url:
            canon = canonical_url(event.url)
            if canon in self.seen_urls:
                return True
        fp = self._fingerprint(event)
        return fp in self.seen_fingerprints

    def add(self, event: NewsEvent) -> None:
        """Add an event to the index. Evicts oldest entries if max_size is exceeded."""
        now = time.monotonic()
        if len(self.seen_event_ids) >= self._max_size:
            keep_count = self._max_size // 2
            while len(self.seen_event_ids) > keep_count:
                self.seen_event_ids.popitem(last=False)
            while len(self.seen_fingerprints) > keep_count:
                self.seen_fingerprints.popitem(last=False)
            logger.debug("[SKIP] NewsDedupeIndex: size-evicted to %d entries", keep_count)

        if event.event_id in self.seen_event_ids:
            self.seen_event_ids.move_to_end(event.event_id)
        else:
            self.seen_event_ids[event.event_id] = now

        fp = self._fingerprint(event)
        if fp in self.seen_fingerprints:
            self.seen_fingerprints.move_to_end(fp)
        else:
            self.seen_fingerprints[fp] = now

        if event.url:
            self.seen_urls.add(canonical_url(event.url))

    def filter_new(self, events: list[NewsEvent]) -> list[NewsEvent]:
        """Return only non-duplicate events and add them to the index."""
        new_events: list[NewsEvent] = []
        for event in events:
            if not self.is_duplicate(event):
                new_events.append(event)
                self.add(event)
        return new_events

    def filter_new_with_counts(
        self, events: list[NewsEvent]
    ) -> list[tuple[NewsEvent, int]]:
        """Return (event, n_sources) for new events; track cross-source counts.

        Even duplicate events increment seen_counts so the caller can detect
        how many independent sources reported the same story.
        """
        result: list[tuple[NewsEvent, int]] = []
        for event in events:
            fp = self._fingerprint(event)
            self.seen_counts[fp] = self.seen_counts.get(fp, 0) + 1
            if not self.is_duplicate(event):
                self.add(event)
                result.append((event, self.seen_counts[fp]))
        return result

    def save(self) -> None:
        """Persist index to JSON if a persist_path was configured."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        now = time.monotonic()
        try:
            with open(self._persist_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "event_ids": list(self.seen_event_ids.keys()),
                        "fingerprints": list(self.seen_fingerprints.keys()),
                        # store relative age so load() can re-anchor timestamps
                        "event_id_ages": [
                            now - ts for ts in self.seen_event_ids.values()
                        ],
                        "fingerprint_ages": [
                            now - ts for ts in self.seen_fingerprints.values()
                        ],
                    },
                    fh,
                )
        except Exception as exc:
            logger.warning("[WARN] NewsDedupeIndex.save: %s", exc)

    def load(self) -> None:
        """Load persisted index from JSON."""
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            now = time.monotonic()
            event_ids: list[str] = data.get("event_ids", [])
            fingerprints: list[str] = data.get("fingerprints", [])
            event_id_ages: list[float] = data.get("event_id_ages", [])
            fingerprint_ages: list[float] = data.get("fingerprint_ages", [])
            self.seen_event_ids = OrderedDict(
                (eid, now - age)
                for eid, age in zip(
                    event_ids,
                    event_id_ages if event_id_ages else [0.0] * len(event_ids),
                )
            )
            self.seen_fingerprints = OrderedDict(
                (fp, now - age)
                for fp, age in zip(
                    fingerprints,
                    fingerprint_ages if fingerprint_ages else [0.0] * len(fingerprints),
                )
            )
            logger.debug(
                "[OK] NewsDedupeIndex.load: %d ids, %d fingerprints",
                len(self.seen_event_ids),
                len(self.seen_fingerprints),
            )
        except Exception as exc:
            logger.warning("[WARN] NewsDedupeIndex.load: %s", exc)
