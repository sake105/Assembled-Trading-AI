"""News headline source via newsapi.ai (EventRegistry).

Fetches news articles via newsapi.ai (EventRegistry-based API).

Requires environment variable::

    NEWSAPI_KEY=<your UUID token from newsapi.ai/dashboard>

Free tier: limited requests/day.
Dashboard: https://newsapi.ai/dashboard
API docs:  https://newsapi.ai/documentation

Usage::

    from assembled_core.data.sources.newsapi_source import fetch_news_headlines

    df = fetch_news_headlines(
        keywords=["Federal Reserve", "inflation"],
        from_date="2024-01-01",
        to_date="2024-01-07",
    )
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_EMPTY = pd.DataFrame(
    columns=["timestamp", "title", "description", "source", "url", "query"]
)
_BASE_URL = "https://newsapi.ai/api/v1/article/getArticles"

# Item 152: Daily call-count guard (free tier = 100 calls/day).
# Counter persists to .newsapi_call_counter.json so restarts don't reset the count.
_DAILY_CALL_LIMIT: int = int(os.environ.get("NEWSAPI_DAILY_LIMIT", "100"))
_COUNTER_PATH: Path = Path(
    os.environ.get("NEWSAPI_COUNTER_PATH", ".newsapi_call_counter.json")
)
_COUNTER_LOCK: threading.Lock = threading.Lock()


def _load_counter() -> tuple[str, int]:
    """Return (date_str, count) from the counter file."""
    try:
        if _COUNTER_PATH.exists():
            data = json.loads(_COUNTER_PATH.read_text())
            return data.get("date", ""), int(data.get("count", 0))
    except Exception:
        pass
    return "", 0


def _save_counter(date_str: str, count: int) -> None:
    try:
        _COUNTER_PATH.write_text(json.dumps({"date": date_str, "count": count}))
    except Exception as exc:
        logger.debug("[newsapi] counter save failed: %s", exc)


def _increment_counter() -> tuple[int, int]:
    """Increment daily counter; return (current_count, limit)."""
    today = date.today().isoformat()
    with _COUNTER_LOCK:
        stored_date, stored_count = _load_counter()
        count = (stored_count + 1) if stored_date == today else 1
        _save_counter(today, count)
    return count, _DAILY_CALL_LIMIT


def _get_api_key() -> str | None:
    # Multi-key rotation (2026-05-22): try rotator first; backward-compat
    # fallback to single NEWSAPI_KEY env var if rotator pool is empty or
    # the import fails (defensive against circular-import in some tests).
    try:
        from src.assembled_core.utils.api_key_rotator import get_rotator

        rotated = get_rotator().get_key("newsapi")
        if rotated:
            return rotated
    except Exception:  # noqa: BLE001
        pass
    key = os.environ.get("NEWSAPI_KEY", "").strip()
    return key if key else None


def fetch_news_headlines(
    keywords: list[str],
    from_date: str,
    to_date: str,
    *,
    language: str = "eng",
    max_per_query: int = 10,
) -> pd.DataFrame:
    """Fetch news articles matching keywords from newsapi.ai.

    Args:
        keywords:      List of search terms, e.g. ["Federal Reserve", "S&P 500"].
                       Each keyword is fetched as a separate query; results are combined.
        from_date:     Inclusive start date, "YYYY-MM-DD".
        to_date:       Inclusive end date, "YYYY-MM-DD".
        language:      Article language code (default: "eng" for English).
        max_per_query: Max articles per keyword query (default: 10).

    Returns:
        DataFrame with columns: timestamp (UTC), title, description, source, url, query.
        Empty DataFrame if key missing or all fetches fail.
    """
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        logger.error("[ERROR] requests not installed.")
        return _EMPTY.copy()

    if not keywords:
        return _EMPTY.copy()

    api_key = _get_api_key()
    if api_key is None:
        logger.warning(
            "[WARN] newsapi: NEWSAPI_KEY not set — returning empty DataFrame."
        )
        return _EMPTY.copy()

    frames: list[pd.DataFrame] = []

    for query in keywords:
        count, limit = _increment_counter()
        if count > limit:
            logger.warning(
                "[WARN] newsapi: daily call limit reached (%d/%d) — skipping query '%s'",
                count,
                limit,
                query,
            )
            continue
        try:
            payload = {
                "apiKey": api_key,
                "keyword": query,
                "lang": language,
                "dateStart": from_date,
                "dateEnd": to_date,
                "articlesCount": max_per_query,
                "articlesSortBy": "date",
                "resultType": "articles",
                "articleBodyLen": 0,
            }
            resp = requests.post(
                _BASE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()

            articles_block = data.get("articles") or {}
            articles = articles_block.get("results") or []

            if not articles:
                logger.debug("[SKIP] newsapi: no articles for query '%s'", query)
                continue

            rows = []
            for art in articles:
                published = art.get("dateTime") or art.get("date") or ""
                source = (art.get("source") or {}).get("title") or ""
                rows.append(
                    {
                        "timestamp": pd.to_datetime(
                            published, utc=True, errors="coerce"
                        ),
                        "title": art.get("title") or "",
                        "description": art.get("body") or art.get("description") or "",
                        "source": source,
                        "url": art.get("url") or "",
                        "query": query,
                    }
                )
            frames.append(pd.DataFrame(rows))
            logger.debug("[OK] newsapi: %d articles for query '%s'", len(rows), query)

        except Exception as exc:
            logger.error("[ERROR] newsapi: failed for query '%s' — %s", query, exc)

    if not frames:
        logger.warning(
            "[WARN] newsapi: no articles returned for any of %d queries.", len(keywords)
        )
        return _EMPTY.copy()

    result = pd.concat(frames, ignore_index=True)
    result = (
        result.drop_duplicates(subset=["url"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info(
        "[OK] newsapi: fetched %d articles across %d queries.",
        len(result),
        len(keywords),
    )
    return result
