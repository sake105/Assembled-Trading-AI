"""House PTR fetch — index/RSS + optional PDF download (DISCL-2.1)."""

from __future__ import annotations

import hashlib
import logging
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"
RSS_NS = "http://purl.org/rss/1.0/"
RSS2_NS = ""


def _get_text(el: ET.Element | None, default: str = "") -> str:
    if el is None:
        return default
    return (el.text or "").strip() if el.text else default


def _safe_filename(doc_id: str) -> str:
    """Derive filesystem-safe filename from doc_id."""
    if not doc_id:
        return "unknown"
    safe = re.sub(r"[^\w\-.]", "_", doc_id)
    return safe[:200] or "unknown"


def _best_effort_person_from_title(title: str) -> str:
    """Extract person/name from title (best-effort)."""
    if not title:
        return ""
    return title.strip()


def _best_effort_doc_id(link: str) -> str:
    """Extract doc_id from link (last path segment or full link)."""
    if not link:
        return ""
    link = link.strip()
    if "/" in link:
        return link.rsplit("/", 1)[-1].split("?")[0] or link
    return link


def _parse_rss_xml(content: bytes, max_items: int) -> List[Dict[str, Any]]:
    """Parse RSS/Atom XML into list of raw item dicts."""
    items: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return items

    # Atom
    ns = {"atom": ATOM_NS}
    entries = root.findall(".//atom:entry", ns)
    if not entries:
        entries = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for entry in entries[:max_items]:
        title_el = entry.find("atom:title", ns) or entry.find(
            "{http://www.w3.org/2005/Atom}title"
        )
        title = _get_text(title_el)

        link = ""
        for link_el in entry.findall("atom:link", ns) or entry.findall(
            "{http://www.w3.org/2005/Atom}link"
        ):
            href = link_el.get("href") or ""
            if href and (link_el.get("type") or "").find("pdf") >= 0:
                link = href.strip()
                break
        if not link:
            link_el = entry.find("atom:link", ns) or entry.find(
                "{http://www.w3.org/2005/Atom}link"
            )
            link = (link_el.get("href") or "").strip() if link_el is not None else ""

        published_el = entry.find("atom:published", ns) or entry.find(
            "atom:updated", ns
        )
        if published_el is None:
            published_el = entry.find(
                "{http://www.w3.org/2005/Atom}published"
            ) or entry.find("{http://www.w3.org/2005/Atom}updated")
        published = _get_text(published_el)

        doc_id = _best_effort_doc_id(link) or _get_text(
            entry.find("atom:id", ns) or entry.find("{http://www.w3.org/2005/Atom}id")
        )
        person = _best_effort_person_from_title(title)

        raw_item: Dict[str, Any] = {
            "title": title,
            "link": link,
            "published": published,
            "person": person or None,
            "doc_id": doc_id or None,
            "raw": {"title": title, "link": link, "published": published},
        }
        items.append(raw_item)

    if items:
        return items

    # RSS 2.0 <channel><item>
    for item_el in root.findall(".//item")[:max_items]:
        title_el = item_el.find("title")
        title = _get_text(title_el)
        link_el = item_el.find("link")
        link = _get_text(link_el) if link_el is not None else ""
        pub_el = item_el.find("pubDate") or item_el.find("published")
        published = _get_text(pub_el)
        doc_id = _best_effort_doc_id(link)
        person = _best_effort_person_from_title(title)
        raw_item = {
            "title": title,
            "link": link,
            "published": published,
            "person": person or None,
            "doc_id": doc_id or None,
            "raw": {"title": title, "link": link, "published": published},
        }
        items.append(raw_item)

    return items


def _parse_json_list(content: bytes, max_items: int) -> List[Dict[str, Any]]:
    """Best-effort parse JSON list or object with items array."""
    import json

    items: List[Dict[str, Any]] = []
    try:
        text = content.decode("utf-8", errors="replace").strip()
        if not text or (not text.startswith("{") and not text.startswith("[")):
            return items
        data = json.loads(text)
        if isinstance(data, list):
            for entry in data[:max_items]:
                if isinstance(entry, dict):
                    title = str(entry.get("title") or entry.get("name") or "").strip()
                    link = str(
                        entry.get("link") or entry.get("url") or entry.get("href") or ""
                    ).strip()
                    published = str(
                        entry.get("published")
                        or entry.get("updated")
                        or entry.get("date")
                        or ""
                    ).strip()
                    doc_id = (
                        _best_effort_doc_id(link) or str(entry.get("id") or "").strip()
                    )
                    person = (
                        _best_effort_person_from_title(title)
                        or str(entry.get("person") or entry.get("author") or "").strip()
                    )
                    items.append(
                        {
                            "title": title,
                            "link": link,
                            "published": published,
                            "person": person or None,
                            "doc_id": doc_id or None,
                            "raw": dict(entry),
                        }
                    )
        elif isinstance(data, dict):
            arr = data.get("items") or data.get("entries") or data.get("data") or []
            for entry in (arr if isinstance(arr, list) else [])[:max_items]:
                if isinstance(entry, dict):
                    title = str(entry.get("title") or entry.get("name") or "").strip()
                    link = str(
                        entry.get("link") or entry.get("url") or entry.get("href") or ""
                    ).strip()
                    published = str(
                        entry.get("published")
                        or entry.get("updated")
                        or entry.get("date")
                        or ""
                    ).strip()
                    doc_id = (
                        _best_effort_doc_id(link) or str(entry.get("id") or "").strip()
                    )
                    person = (
                        _best_effort_person_from_title(title)
                        or str(entry.get("person") or entry.get("author") or "").strip()
                    )
                    items.append(
                        {
                            "title": title,
                            "link": link,
                            "published": published,
                            "person": person or None,
                            "doc_id": doc_id or None,
                            "raw": dict(entry),
                        }
                    )
    except Exception as exc:
        logger.warning("[FetchHousePtr] failed to parse RSS/Atom feed: %s", exc)
    return items


def _compute_pdf_meta(
    local_path: Path | str,
    pdf_meta_cfg: Dict[str, Any],
    fetched_utc: str,
) -> Dict[str, Any]:
    """Compute pdf_meta for a local file: size_bytes, optional sha256 (if size <= max_mb and compute_sha256).
    Returns dict with sha256 (if hashed), size_bytes, local_path, hashed (bool), fetched_utc.
    """
    path = Path(local_path) if not isinstance(local_path, Path) else local_path
    if not path.exists() or not path.is_file():
        return {
            "local_path": str(path),
            "size_bytes": 0,
            "hashed": False,
            "fetched_utc": fetched_utc,
        }
    size_bytes = path.stat().st_size
    max_mb = float(pdf_meta_cfg.get("max_mb", 25))
    compute_sha256 = bool(pdf_meta_cfg.get("compute_sha256", True))
    size_mb = size_bytes / (1024 * 1024)
    hashed = False
    sha256_hex: str | None = None
    if size_mb <= max_mb and compute_sha256:
        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            sha256_hex = h.hexdigest()
            hashed = True
        except Exception as exc:
            logger.warning("[FetchHousePtr] failed to compute sha256 for %s: %s", path, exc)
    out: Dict[str, Any] = {
        "local_path": str(path),
        "size_bytes": size_bytes,
        "hashed": hashed,
        "fetched_utc": fetched_utc,
    }
    if sha256_hex is not None:
        out["sha256"] = sha256_hex
    return out


def _download_pdf(url: str, dest_path: Path, user_agent: str, timeout_s: float) -> bool:
    """Download PDF to dest_path. Returns True on success."""
    try:
        import requests

        resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=timeout_s)
        if resp.status_code == 200 and resp.content:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(resp.content)
            return True
    except Exception as exc:
        logger.warning("[FetchHousePtr] PDF download failed for %s: %s", url, exc)
    return False


def fetch_house_ptr_filings(
    source_id: str,
    cfg: Dict[str, Any],
    fetch_state: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    """Fetch House PTR filings from configurable index (RSS or JSON). Returns (items, failure, stats)."""
    import requests

    index_url = str(cfg.get("index_url") or "").strip()
    user_agent = str(
        cfg.get("user_agent") or "Assembled-Trading-AI/Disclosures-v1"
    ).strip()
    timeout_s = float(cfg.get("timeout_s", 15.0))
    cache_minutes = float(cfg.get("cache_minutes", 60))
    stale_on_error_minutes = float(cfg.get("stale_on_error_minutes", 240))
    download_pdfs = bool(cfg.get("download_pdfs", False))
    download_dir = str(
        cfg.get("download_dir") or "output/intel/disclosures/raw/house_ptr"
    ).strip()
    max_items = int(cfg.get("max_items", 50))

    items: List[Dict[str, Any]] = []
    failure: Dict[str, Any] | None = None
    stats: Dict[str, Any] = {
        "source_id": source_id,
        "type": "house_ptr",
        "ok": False,
        "items": 0,
        "http_status": None,
        "duration_ms": None,
        "cached": False,
        "downloaded_count": 0,
        "pdf_hashed_count": 0,
        "pdf_skipped_count": 0,
        "error": None,
    }

    # Cache hit (fresh)
    if fetch_state and isinstance(fetch_state, dict):
        cached_utc = fetch_state.get("cached_utc")
        cached_items = fetch_state.get("cached_items")
        if cached_items is not None and cached_utc:
            try:
                from datetime import datetime, timezone, timedelta

                then = datetime.fromisoformat(cached_utc.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if (now - then) <= timedelta(minutes=cache_minutes):
                    stats["ok"] = True
                    stats["items"] = len(cached_items)
                    stats["cached"] = True
                    stats["http_status"] = 200
                    stats["duration_ms"] = 0
                    return list(cached_items), None, stats
            except Exception as exc:
                logger.warning("[FetchHousePtr] failed to parse cached state for %s: %s", source_id, exc)

    if not index_url or index_url.startswith("https://<"):
        failure = {"source": source_id, "reason": "missing_index_url"}
        stats["error"] = "missing_index_url"
        # Stale-on-error: serve cached if within window
        if fetch_state and isinstance(fetch_state, dict):
            cached_items = fetch_state.get("cached_items")
            cached_utc = fetch_state.get("cached_utc")
            if cached_items is not None and cached_utc:
                try:
                    from datetime import datetime, timezone, timedelta

                    then = datetime.fromisoformat(cached_utc.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - then) <= timedelta(minutes=stale_on_error_minutes):
                        failure["reason"] = "stale-on-error"
                        stats["ok"] = True
                        stats["items"] = len(cached_items)
                        stats["cached"] = True
                        return list(cached_items), failure, stats
                except Exception as exc:
                    logger.warning("[FetchHousePtr] stale-on-error cache parse failed for %s: %s", source_id, exc)
        return items, failure, stats

    start = time.perf_counter()
    try:
        resp = requests.get(
            index_url,
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
            stats["error"] = f"http_{resp.status_code}"
            # Stale-on-error
            if fetch_state and isinstance(fetch_state, dict):
                cached_items = fetch_state.get("cached_items")
                cached_utc = fetch_state.get("cached_utc")
                if cached_items is not None and cached_utc:
                    try:
                        from datetime import datetime, timezone, timedelta

                        then = datetime.fromisoformat(cached_utc.replace("Z", "+00:00"))
                        now = datetime.now(timezone.utc)
                        if (now - then) <= timedelta(minutes=stale_on_error_minutes):
                            failure["reason"] = "stale-on-error"
                            stats["ok"] = True
                            stats["items"] = len(cached_items)
                            stats["cached"] = True
                            return list(cached_items), failure, stats
                    except Exception as exc:
                        logger.warning("[FetchHousePtr] stale-on-error cache parse failed (http) for %s: %s", source_id, exc)
            return items, failure, stats

        content = resp.content
        text_preview = (
            content[:100].decode("utf-8", errors="replace") if content else ""
        ).strip()
        if text_preview.lstrip().startswith(
            "<?xml"
        ) or text_preview.lstrip().startswith("<"):
            items = _parse_rss_xml(content, max_items)
        else:
            items = _parse_json_list(content, max_items)

        # Optional PDF download
        download_count = 0
        if download_pdfs and items:
            base = Path(download_dir)
            for it in items:
                link = it.get("link") or ""
                if not link or not link.lower().endswith(".pdf"):
                    continue
                doc_id = it.get("doc_id") or _best_effort_doc_id(link)
                fname = _safe_filename(doc_id)
                if not fname.endswith(".pdf"):
                    fname += ".pdf"
                dest = base / fname
                if _download_pdf(link, dest, user_agent, timeout_s):
                    download_count += 1
                    it["local_path"] = str(dest)
                if "raw" in it and isinstance(it["raw"], dict):
                    it["raw"]["local_path"] = it.get("local_path")

        # PDF metadata: when download_pdfs=true or local_path present, compute file_sha256/size
        pdf_meta_cfg = cfg.get("pdf_meta") or {}
        if isinstance(pdf_meta_cfg, dict) and pdf_meta_cfg.get("enabled", True):
            from datetime import datetime, timezone

            fetched_utc = datetime.now(timezone.utc).isoformat()
            for it in items:
                lp = it.get("local_path")
                if not lp:
                    continue
                meta = _compute_pdf_meta(lp, pdf_meta_cfg, fetched_utc)
                it["pdf_meta"] = meta
                if "raw" in it and isinstance(it["raw"], dict):
                    it["raw"]["pdf_meta"] = meta
                if meta.get("hashed"):
                    stats["pdf_hashed_count"] = stats.get("pdf_hashed_count", 0) + 1
                elif meta.get("size_bytes", 0) > 0:
                    stats["pdf_skipped_count"] = stats.get("pdf_skipped_count", 0) + 1

        stats["downloaded_count"] = download_count
        stats["ok"] = True
        stats["items"] = len(items)
        return items, None, stats

    except requests.RequestException as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["duration_ms"] = duration_ms
        stats["http_status"] = getattr(
            getattr(e, "response", None), "status_code", None
        )
        stats["error"] = str(e)
        failure = {"source": source_id, "reason": "request_error", "error": str(e)}
        # Stale-on-error
        if fetch_state and isinstance(fetch_state, dict):
            cached_items = fetch_state.get("cached_items")
            cached_utc = fetch_state.get("cached_utc")
            if cached_items is not None and cached_utc:
                try:
                    from datetime import datetime, timezone, timedelta

                    then = datetime.fromisoformat(cached_utc.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - then) <= timedelta(minutes=stale_on_error_minutes):
                        failure["reason"] = "stale-on-error"
                        stats["ok"] = True
                        stats["items"] = len(cached_items)
                        stats["cached"] = True
                        return list(cached_items), failure, stats
                except Exception as exc:
                    logger.warning("[FetchHousePtr] stale-on-error cache parse failed (request) for %s: %s", source_id, exc)
        return items, failure, stats
    except ET.ParseError as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["duration_ms"] = duration_ms
        stats["error"] = str(e)
        failure = {"source": source_id, "reason": "parse_error", "error": str(e)}
        if fetch_state and isinstance(fetch_state, dict):
            cached_items = fetch_state.get("cached_items")
            cached_utc = fetch_state.get("cached_utc")
            if cached_items is not None and cached_utc:
                try:
                    from datetime import datetime, timezone, timedelta

                    then = datetime.fromisoformat(cached_utc.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if (now - then) <= timedelta(minutes=stale_on_error_minutes):
                        failure["reason"] = "stale-on-error"
                        stats["ok"] = True
                        stats["items"] = len(cached_items)
                        stats["cached"] = True
                        return list(cached_items), failure, stats
                except Exception as exc:
                    logger.warning("[FetchHousePtr] stale-on-error cache parse failed (parse_error) for %s: %s", source_id, exc)
        return items, failure, stats
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stats["duration_ms"] = duration_ms
        stats["error"] = str(e)
        failure = {"source": source_id, "reason": "error", "error": str(e)}
        return items, failure, stats


def fetch_house_ptr(
    source_id: str,
    config: Dict[str, Any],
    *,
    timeout_s: float = 15.0,
    user_agent: str = "Assembled-Trading-AI/Disclosures-v1",
    fetch_state: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    """Convenience wrapper: build cfg from config + kwargs and call fetch_house_ptr_filings."""
    cfg = dict(config)
    cfg.setdefault("timeout_s", timeout_s)
    cfg.setdefault("user_agent", user_agent)
    return fetch_house_ptr_filings(source_id, cfg, fetch_state=fetch_state)
