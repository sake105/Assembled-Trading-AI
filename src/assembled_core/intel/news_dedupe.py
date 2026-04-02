"""News event deduplication — prevents same event from triggering multiple times."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import string
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
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        max_size: int = 10_000,
    ) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_size = max_size
        self.seen_event_ids: set[str] = set()
        self.seen_fingerprints: set[str] = set()

        if self._persist_path and self._persist_path.exists():
            self.load()

    def _fingerprint(self, event: NewsEvent) -> str:
        return content_fingerprint(event.title, event.source_id)

    def is_duplicate(self, event: NewsEvent) -> bool:
        """Return True if event_id OR content fingerprint was already seen."""
        if event.event_id in self.seen_event_ids:
            return True
        fp = self._fingerprint(event)
        return fp in self.seen_fingerprints

    def add(self, event: NewsEvent) -> None:
        """Add an event to the index. Evicts half the index if max_size is exceeded."""
        # Evict if full — simple strategy: clear half
        if len(self.seen_event_ids) >= self._max_size:
            keep_count = self._max_size // 2
            self.seen_event_ids = set(list(self.seen_event_ids)[-keep_count:])
            self.seen_fingerprints = set(list(self.seen_fingerprints)[-keep_count:])
            logger.debug("[SKIP] NewsDedupeIndex: evicted to %d entries", keep_count)

        self.seen_event_ids.add(event.event_id)
        self.seen_fingerprints.add(self._fingerprint(event))

    def filter_new(self, events: list[NewsEvent]) -> list[NewsEvent]:
        """
        Return only non-duplicate events and add them to the index.

        Mutates the index in-place.
        """
        new_events: list[NewsEvent] = []
        for event in events:
            if not self.is_duplicate(event):
                new_events.append(event)
                self.add(event)
        return new_events

    def save(self) -> None:
        """Persist index to JSON if a persist_path was configured."""
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._persist_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "event_ids": list(self.seen_event_ids),
                        "fingerprints": list(self.seen_fingerprints),
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
            self.seen_event_ids = set(data.get("event_ids", []))
            self.seen_fingerprints = set(data.get("fingerprints", []))
            logger.debug(
                "[OK] NewsDedupeIndex.load: %d ids, %d fingerprints",
                len(self.seen_event_ids),
                len(self.seen_fingerprints),
            )
        except Exception as exc:
            logger.warning("[WARN] NewsDedupeIndex.load: %s", exc)
