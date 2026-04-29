from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Tuple

from .normalize import now_utc_iso

# ---------------------------------------------------------------------------
# E2: Specialised multi-domain GDELT query bank
# ---------------------------------------------------------------------------

GDELT_QUERIES: Dict[str, str] = {
    "geopolitical": (
        "war OR military OR invasion OR missile OR troops OR conflict OR airstrike"
    ),
    "sanctions": (
        "sanctions OR embargo OR export control OR OFAC OR blacklist OR entity list "
        "OR secondary sanctions OR SWIFT exclusion"
    ),
    "energy": (
        "oil OR gas OR OPEC OR pipeline OR refinery OR LNG OR energy crisis OR "
        "natural gas OR crude oil OR oil price"
    ),
    "shipping": (
        "shipping OR port OR freight OR blockade OR strait OR maritime OR "
        "chokepoint OR Hormuz OR Suez OR Malacca OR Red Sea OR Houthi"
    ),
    "tech_war": (
        "chip ban OR semiconductor OR TSMC OR rare earth OR tech war OR decoupling "
        "OR export controls OR CHIPS Act OR entity list"
    ),
    "currency": (
        "currency crisis OR devaluation OR central bank OR rate hike OR inflation OR "
        "IMF bailout OR reserve depletion OR capital flight"
    ),
    "cyber": (
        "cyberattack OR ransomware OR hack OR data breach OR infrastructure attack "
        "OR zero-day OR state-sponsored hack"
    ),
    "climate": (
        "drought OR flood OR hurricane OR crop failure OR supply chain disruption OR "
        "heatwave OR wildfire OR food crisis"
    ),
    "military": (
        "nuclear OR escalation OR mobilization OR NATO OR defense spending OR "
        "military buildup OR troops deployment OR nuclear threat"
    ),
    "finance": (
        "bank crisis OR credit crunch OR sovereign default OR debt ceiling OR "
        "banking crisis OR yield curve OR recession OR financial crisis"
    ),
}


def fetch_gdelt_events(
    source_id: str,
    query: str,
    *,
    gdelt_cfg: Dict[str, Any],
    cadence: str,
    fetch_state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, Dict[str, Any]]:
    """Fetch events from GDELT Doc API with simple caching and rate limiting."""
    import time

    import requests
    from dateutil import parser as date_parser

    rate_limit_rps = float(gdelt_cfg.get("rate_limit_rps", 1.0) or 1.0)
    cache_minutes = int(gdelt_cfg.get("cache_minutes", 10))
    stale_on_error_minutes = int(gdelt_cfg.get("stale_on_error_minutes", 60))
    window_hours_cfg = gdelt_cfg.get("window_hours") or {}
    window_hours = int(window_hours_cfg.get(cadence, window_hours_cfg.get("hourly", 1)))

    cache_key_str = f"{query}:{window_hours}"
    cache_key = sha256(cache_key_str.encode("utf-8")).hexdigest()
    gdelt_state = fetch_state.setdefault("gdelt", {})

    def _age_minutes(ts: str) -> float:
        try:
            dt = date_parser.parse(ts)
            return (date_parser.parse(now_utc_iso()) - dt).total_seconds() / 60.0
        except Exception:
            return 1e9

    stats: Dict[str, Any] = {
        "source_id": source_id,
        "type": "gdelt",
        "ok": False,
        "http_status": None,
        "duration_ms": 0,
        "items": 0,
        "not_modified": False,
        "cached": False,
        "error": None,
    }

    cached_entry = gdelt_state.get(cache_key)
    if isinstance(cached_entry, dict):
        cached_utc = cached_entry.get("cached_utc")
        if isinstance(cached_utc, str) and _age_minutes(cached_utc) <= cache_minutes:
            stats["cached"] = True
            items_cached = cached_entry.get("items") or []
            stats["items"] = len(items_cached)
            stats["ok"] = True
            return items_cached, None, stats

    api_query = (
        query if query.startswith("(") else f"({query})" if " OR " in query else query
    )
    params = {
        "query": api_query,
        "format": "json",
        "maxrecords": 50,
        "sort": "datedesc",
    }

    start = time.time()
    failure: Dict[str, Any] | None = None
    items: List[Dict[str, Any]] = []

    # Simple rate limit
    if rate_limit_rps > 0:
        time.sleep(1.0 / rate_limit_rps)

    try:
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params=params,
            timeout=10.0,
        )
        stats["http_status"] = resp.status_code
        resp.raise_for_status()
    except Exception as exc:
        cached_ok = False
        if isinstance(cached_entry, dict):
            cached_utc = cached_entry.get("cached_utc")
            if (
                isinstance(cached_utc, str)
                and _age_minutes(cached_utc) <= stale_on_error_minutes
            ):
                items = cached_entry.get("items") or []
                stats["cached"] = True
                stats["items"] = len(items)
                stats["ok"] = True
                failure = {
                    "source": source_id,
                    "reason": f"stale-on-error: {exc}",
                    "http_status": stats["http_status"],
                }
                cached_ok = True
        if not cached_ok:
            failure = {
                "source": source_id,
                "reason": f"gdelt_fetch_error: {exc}",
                "http_status": stats["http_status"],
            }
            stats["error"] = str(exc)
        stats["duration_ms"] = int((time.time() - start) * 1000)
        return items, failure, stats

    try:
        data = resp.json()
    except Exception as exc:
        failure = {
            "source": source_id,
            "reason": f"gdelt_json_error: {exc}",
            "http_status": stats["http_status"],
        }
        stats["error"] = str(exc)
        stats["duration_ms"] = int((time.time() - start) * 1000)
        return [], failure, stats

    articles = data.get("articles") or []
    for a in articles:
        if not isinstance(a, dict):
            continue
        item: Dict[str, Any] = {
            "title": a.get("title") or "",
            "link": a.get("url") or "",
            "published": a.get("seendate") or a.get("date"),
            "summary": a.get("title", ""),  # GDELT has no summary field; reuse title
            "raw": a,
        }
        items.append(item)

    gdelt_state[cache_key] = {
        "cached_utc": now_utc_iso(),
        "items": items,
    }

    stats["ok"] = True
    stats["items"] = len(items)
    stats["duration_ms"] = int((time.time() - start) * 1000)
    return items, None, stats


def fetch_gdelt_multi_domain(
    *,
    gdelt_cfg: Dict[str, Any],
    cadence: str,
    fetch_state: Dict[str, Any],
    domains: List[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """E2: Fetch GDELT events across multiple domain-specialised queries.

    Runs one GDELT query per domain (e.g. geopolitical, sanctions, energy, etc.)
    and merges the results, deduplicating by article URL.

    Args:
        gdelt_cfg: GDELT configuration dict (from sources config).
        cadence: Scheduling cadence ("hourly", "15min", etc.).
        fetch_state: Mutable state dict for caching between calls.
        domains: List of domain keys to query (default: all GDELT_QUERIES).

    Returns:
        (all_items, failures, all_stats) tuple with merged results.
    """
    active_domains = domains or list(GDELT_QUERIES.keys())
    all_items: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    seen_urls: set = set()

    for domain in active_domains:
        query = GDELT_QUERIES.get(domain)
        if not query:
            continue
        source_id = f"gdelt_{domain}"
        items, failure, stats = fetch_gdelt_events(
            source_id=source_id,
            query=query,
            gdelt_cfg=gdelt_cfg,
            cadence=cadence,
            fetch_state=fetch_state,
        )
        # Deduplicate by URL
        for item in items:
            url = item.get("link", "")
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            item["domain"] = domain  # Tag with domain for downstream routing
            all_items.append(item)

        if failure:
            failures.append(failure)
        all_stats.append(stats)

    return all_items, failures, all_stats
