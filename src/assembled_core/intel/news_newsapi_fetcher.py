"""NewsAPI.org fetcher (gated; requires NEWSAPI_KEY env var).

Defensive: returns [] on any error, logs at WARN level, never raises into
the intel cycle. Keyed via env var — never read from disk, never in code.

Usage:
    fetcher = NewsAPIFetcher(api_key=os.getenv("NEWSAPI_KEY", ""))
    if fetcher.enabled:
        events = fetcher.fetch(query="geopolitics", page_size=50)
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone

from src.assembled_core.intel.models import NewsEvent, SourceTier

logger = logging.getLogger(__name__)

_NEWSAPI_EVERYTHING = "https://newsapi.org/v2/everything"
_NEWSAPI_HEADLINES = "https://newsapi.org/v2/top-headlines"


class NewsAPIFetcher:
    """Thin wrapper around newsapi.org; gated by API key presence."""

    def __init__(
        self,
        api_key: str = "",
        source_tier: SourceTier = SourceTier.T2,
        timeout: int = 20,
    ) -> None:
        self._api_key = (api_key or os.getenv("NEWSAPI_KEY", "") or "").strip()
        self._tier = source_tier
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def fetch(
        self,
        query: str = "",
        page_size: int = 50,
        language: str = "en",
        *,
        top_headlines: bool = False,
    ) -> list[NewsEvent]:
        """Return a list of NewsEvents, or [] on any error/disabled."""
        if not self.enabled:
            logger.debug("[SKIP] NewsAPIFetcher: disabled (no API key)")
            return []
        try:
            import requests  # local import so missing lib doesn't break the module
        except Exception as exc:
            logger.warning("[WARN] NewsAPIFetcher: requests not available: %s", exc)
            return []

        url = _NEWSAPI_HEADLINES if top_headlines else _NEWSAPI_EVERYTHING
        params: dict[str, str | int] = {
            "apiKey": self._api_key,
            "pageSize": max(1, min(100, page_size)),
            "language": language,
        }
        if query:
            params["q"] = query
        try:
            resp = requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                logger.warning(
                    "[WARN] NewsAPIFetcher: HTTP %s — %s",
                    resp.status_code, resp.text[:200],
                )
                return []
            data = resp.json() or {}
        except Exception as exc:
            logger.warning("[WARN] NewsAPIFetcher: request failed: %s", exc)
            return []

        articles = data.get("articles") or []
        if not isinstance(articles, list):
            return []
        return self._articles_to_events(articles)

    def _articles_to_events(self, articles: list[dict]) -> list[NewsEvent]:
        now = datetime.now(tz=timezone.utc)
        events: list[NewsEvent] = []
        for art in articles:
            try:
                title = (art.get("title") or "").strip()
                url_v = (art.get("url") or "").strip()
                if not title or not url_v:
                    continue
                source_block = art.get("source") or {}
                source_id = (source_block.get("id") or source_block.get("name") or "newsapi").lower()
                published_raw = art.get("publishedAt") or ""
                try:
                    pub = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
                    if pub.tzinfo is None:
                        pub = pub.replace(tzinfo=timezone.utc)
                except Exception:
                    pub = now
                event_id = "na_" + hashlib.sha256(url_v.encode("utf-8")).hexdigest()[:16]
                ch = hashlib.sha256((title + url_v).encode("utf-8")).hexdigest()[:16]
                events.append(NewsEvent(
                    event_id=event_id,
                    source_id=source_id,
                    source_tier=self._tier,
                    title=title,
                    url=url_v,
                    published_at=pub,
                    ingested_at=now,
                    content_hash=ch,
                ))
            except Exception as exc:
                logger.debug("[SKIP] NewsAPI article parse: %s", exc)
                continue
        return events


__all__ = ["NewsAPIFetcher"]
