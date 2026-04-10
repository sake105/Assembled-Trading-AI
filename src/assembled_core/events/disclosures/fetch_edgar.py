"""SEC EDGAR fetch — Form 4 Atom feed (DISCL-1.1)."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"


def _get_text(el: ET.Element | None, default: str = "") -> str:
    if el is None:
        return default
    return (el.text or "").strip()


def _get_href(link_el: ET.Element | None) -> str:
    if link_el is None:
        return ""
    return (link_el.get("href") or "").strip()


def _best_effort_cik_and_company(title: str, entry_id: str) -> Tuple[str, str]:
    """Extract CIK and company name from title or id (best-effort)."""
    company = title or ""
    cik = ""
    if title:
        m = re.search(r"Form\s*4\s*[-–]\s*(.+)", title, re.IGNORECASE)
        if m:
            company = m.group(1).strip()
    m = re.search(r"(?:CIK|/)(\d{10})", entry_id or "")
    if m:
        cik = m.group(1)
    return cik, company


def fetch_edgar_form4(
    source_id: str,
    cfg: Dict[str, Any],
    fetch_state: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    """Fetch SEC EDGAR Form 4 Atom feed. Returns (items, failure, stats). No heavy parsing; metadata only."""
    import requests  # optional at import time

    feed_url = str(cfg.get("feed_url") or "").strip()
    user_agent = str(
        cfg.get("user_agent") or "Assembled-Trading-AI/Disclosures-v1"
    ).strip()
    timeout_s = float(cfg.get("timeout_s", 15.0))
    cache_minutes = float(cfg.get("cache_minutes", 10))

    items: List[Dict[str, Any]] = []
    failure: Dict[str, Any] | None = None
    stats: Dict[str, Any] = {
        "source_id": source_id,
        "type": "edgar_form4",
        "ok": False,
        "items": 0,
        "http_status": None,
        "duration_ms": None,
        "cached": False,
    }

    # Optional cache hit
    if fetch_state and isinstance(fetch_state, dict):
        cached_entries = fetch_state.get("cached_entries")
        cached_utc = fetch_state.get("cached_utc")
        if cached_entries is not None and cached_utc:
            try:
                from datetime import datetime, timezone, timedelta

                then = datetime.fromisoformat(cached_utc.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - then) <= timedelta(minutes=cache_minutes):
                    stats["ok"] = True
                    stats["items"] = len(cached_entries)
                    stats["cached"] = True
                    stats["http_status"] = 200
                    stats["duration_ms"] = 0
                    return list(cached_entries), None, stats
            except Exception as exc:
                logger.warning("[FetchEdgar] failed to parse cached state for %s: %s", source_id, exc)

    if not feed_url:
        failure = {"source": source_id, "reason": "missing_feed_url"}
        return items, failure, stats

    start = time.perf_counter()
    try:
        resp = requests.get(
            feed_url,
            headers={"User-Agent": user_agent},
            timeout=timeout_s,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["http_status"] = resp.status_code
        stats["duration_ms"] = duration_ms

        if resp.status_code != 200:
            failure = {
                "source": source_id,
                "reason": "http_error",
                "status": resp.status_code,
            }
            return items, failure, stats

        root = ET.fromstring(resp.content)
        ns = {"atom": ATOM_NS}
        entries = root.findall(".//atom:entry", ns)
        if not entries:
            entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for entry in entries:
            title_el = entry.find("atom:title", ns) or entry.find(
                "{http://www.w3.org/2005/Atom}title"
            )
            title = _get_text(title_el)

            link_el = None
            for link in entry.findall("atom:link", ns) or entry.findall(
                "{http://www.w3.org/2005/Atom}link"
            ):
                if link.get("rel") in (None, "alternate"):
                    link_el = link
                    break
            if link_el is None:
                link_el = entry.find("atom:link", ns) or entry.find(
                    "{http://www.w3.org/2005/Atom}link"
                )
            link = _get_href(link_el)

            updated_el = entry.find("atom:updated", ns) or entry.find(
                "{http://www.w3.org/2005/Atom}updated"
            )
            published_el = entry.find("atom:published", ns) or entry.find(
                "{http://www.w3.org/2005/Atom}published"
            )
            published = _get_text(published_el) or _get_text(updated_el)

            id_el = entry.find("atom:id", ns) or entry.find(
                "{http://www.w3.org/2005/Atom}id"
            )
            entry_id = _get_text(id_el)
            accession = entry_id

            cik, company = _best_effort_cik_and_company(title, entry_id)

            raw_item: Dict[str, Any] = {
                "title": title,
                "link": link,
                "published": published,
                "cik": cik or None,
                "company": company or None,
                "accession": accession or None,
                "raw": {
                    "title": title,
                    "link": link,
                    "published": published,
                    "id": entry_id,
                },
            }
            items.append(raw_item)

        stats["ok"] = True
        stats["items"] = len(items)
        return items, None, stats

    except requests.RequestException as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["duration_ms"] = duration_ms
        stats["http_status"] = getattr(
            getattr(e, "response", None), "status_code", None
        )
        failure = {"source": source_id, "reason": "request_error", "error": str(e)}
        return items, failure, stats
    except ET.ParseError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["duration_ms"] = duration_ms
        failure = {"source": source_id, "reason": "parse_error", "error": str(e)}
        return items, failure, stats
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["duration_ms"] = duration_ms
        failure = {"source": source_id, "reason": "error", "error": str(e)}
        return items, failure, stats


def fetch_edgar(
    source_id: str,
    config: Dict[str, Any],
    *,
    timeout_s: float = 15.0,
    user_agent: str = "Assembled-Trading-AI/Disclosures-v1",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    """Fetch SEC EDGAR (generic). Stub for non-Form4 types; Form 4 use source type edgar_form4 and fetch_edgar_form4."""
    items: List[Dict[str, Any]] = []
    failure: Dict[str, Any] | None = None
    stats: Dict[str, Any] = {
        "source_id": source_id,
        "type": "edgar",
        "ok": True,
        "items": 0,
        "stub": True,
    }
    return items, failure, stats
