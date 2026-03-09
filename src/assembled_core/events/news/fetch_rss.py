from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .normalize import now_utc_iso, sanitize_text


def fetch_rss_feed(
    source_id: str,
    url: str,
    *,
    timeout: float,
    user_agent: str,
    sanitize_cfg: Dict[str, Any],
    fetch_state: Dict[str, Any],
    retries: int,
    backoff_base_s: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    """Fetch a single RSS feed with ETag/Last-Modified, retries and sanitization.

    Returns (items, failure, stats) where:
      - items: list of raw RSS entries (dict-like)
      - failure: None on success, or {source, reason, http_status}
      - stats: dict for fetch_report (http_status, duration_ms, items, not_modified, cached, error)
    """
    import time
    import requests
    import feedparser  # type: ignore[import]

    rss_state = fetch_state.setdefault("rss", {}).setdefault(source_id, {})
    headers: Dict[str, str] = {
        "User-Agent": user_agent,
    }
    etag = rss_state.get("etag")
    last_modified = rss_state.get("last_modified")
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    stats: Dict[str, Any] = {
        "source_id": source_id,
        "type": "rss",
        "ok": False,
        "http_status": None,
        "duration_ms": 0,
        "items": 0,
        "not_modified": False,
        "cached": False,
        "error": None,
    }

    items: List[Dict[str, Any]] = []
    failure: Dict[str, Any] | None = None

    attempt = 0
    start = time.time()
    while True:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            status = resp.status_code
            stats["http_status"] = status
            if status == 304:
                stats["not_modified"] = True
                rss_state["last_status"] = 304
                break
            resp.raise_for_status()
        except Exception as exc:
            transient_codes = {429, 500, 502, 503, 504}
            status = getattr(locals().get("resp", None), "status_code", None)
            should_retry = (
                isinstance(exc, requests.RequestException)
                and (status in transient_codes or status is None)
            )
            # 403 höchstens ein Retry
            if status == 403 and attempt >= 1:
                should_retry = False
            if attempt < retries and should_retry:
                sleep_s = backoff_base_s * (2 ** attempt)
                time.sleep(max(sleep_s, 0.0))
                attempt += 1
                continue
            failure = {
                "source": source_id,
                "reason": f"rss_fetch_error: {exc}",
                "http_status": status,
            }
            stats["error"] = str(exc)
            break

        # Erfolgreich (2xx)
        try:
            parsed = feedparser.parse(resp.content)
            entries = list(parsed.entries or [])
        except Exception as exc:
            failure = {
                "source": source_id,
                "reason": f"rss_parse_error: {exc}",
                "http_status": stats["http_status"],
            }
            stats["error"] = str(exc)
            break

        strip_html = bool(sanitize_cfg.get("strip_html", True))
        title_max = int(sanitize_cfg.get("title_max_chars", 300))
        summary_max = int(sanitize_cfg.get("summary_max_chars", 800))

        for e in entries:
            # feedparser returns attribute-style objects; convert to dict
            title_raw = getattr(e, "title", "") or ""
            summary_raw = getattr(e, "summary", None)
            item: Dict[str, Any] = {
                "title": sanitize_text(title_raw, strip_html, title_max),
                "link": getattr(e, "link", "") or "",
                "published": getattr(e, "published", None)
                or getattr(e, "updated", None),
                "summary": sanitize_text(summary_raw, strip_html, summary_max)
                if summary_raw is not None
                else None,
                "raw": dict(e),  # type: ignore[arg-type]
            }
            items.append(item)

        # Update State
        rss_state["etag"] = resp.headers.get("ETag") or rss_state.get("etag")
        rss_state["last_modified"] = resp.headers.get("Last-Modified") or rss_state.get(
            "last_modified"
        )
        rss_state["last_status"] = stats["http_status"]
        rss_state["last_success_utc"] = now_utc_iso()
        stats["ok"] = True
        stats["items"] = len(items)
        break

    stats["duration_ms"] = int((time.time() - start) * 1000)
    return items, failure, stats

