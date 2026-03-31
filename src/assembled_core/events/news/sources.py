from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml  # type: ignore[import]


@dataclass
class NewsSource:
    """Configuration for a single news source."""

    source_id: str
    name: str
    domain: str
    type: str  # "rss" | "gdelt" | other
    tier: str  # "A" | "B"
    weight: float
    active: bool
    config: Dict[str, Any]


def load_sources_registry(config_path: str | Path) -> List[NewsSource]:
    """Load news sources registry from YAML config (configs/news/sources.yaml).

    The function is intentionally forgiving: any parse error results in an empty list.
    Callers treat empty lists as ERROR/DEGRADED via NewsHealth.
    """
    path = Path(config_path)
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []

    sources_cfg = data.get("sources") or []
    sources: List[NewsSource] = []
    for entry in sources_cfg:
        if not isinstance(entry, dict):
            continue
        source_id = str(entry.get("source_id") or "").strip()
        name = str(entry.get("name") or "").strip()
        domain = str(entry.get("domain") or "").strip().lower()
        src_type = str(entry.get("type") or "").strip().lower()
        if not source_id or not name or not domain or not src_type:
            continue
        tier = str(entry.get("tier") or "B").strip().upper()
        weight = float(entry.get("weight", 1.0 if tier == "A" else 0.6))
        active = bool(entry.get("active", True))
        cfg = {
            k: v
            for k, v in entry.items()
            if k
            not in {
                "source_id",
                "name",
                "domain",
                "type",
                "tier",
                "weight",
                "active",
                "notes",
            }
        }
        sources.append(
            NewsSource(
                source_id=source_id,
                name=name,
                domain=domain,
                type=src_type,
                tier=tier,
                weight=weight,
                active=active,
                config=cfg,
            )
        )
    return sources


def load_news_params(config_path: str | Path) -> Dict[str, Any]:
    """Load high-level NEWS parameters from YAML config (configs/news/news.yaml).

    Always returns a dict with subdicts: fetch, gdelt, health.
    Missing fields are filled with Phase-2 defaults.
    """
    path = Path(config_path)
    if not path.exists():
        data: Dict[str, Any] = {}
    else:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            data = {}

    if not isinstance(data, dict):
        data = {}

    fetch_cfg = data.get("fetch") or {}
    if not isinstance(fetch_cfg, dict):
        fetch_cfg = {}
    fetch_cfg.setdefault("timeout_s", 10.0)
    fetch_cfg.setdefault("retries", 2)
    fetch_cfg.setdefault("backoff_base_s", 0.5)
    fetch_cfg.setdefault("max_concurrency", 5)
    fetch_cfg.setdefault("user_agent", "Assembled-Trading-AI/NEWS-v1")
    sanitize_cfg = fetch_cfg.get("sanitize") or {}
    if not isinstance(sanitize_cfg, dict):
        sanitize_cfg = {}
    sanitize_cfg.setdefault("strip_html", True)
    sanitize_cfg.setdefault("title_max_chars", 300)
    sanitize_cfg.setdefault("summary_max_chars", 800)
    fetch_cfg["sanitize"] = sanitize_cfg

    gdelt_cfg = data.get("gdelt") or {}
    if not isinstance(gdelt_cfg, dict):
        gdelt_cfg = {}
    gdelt_cfg.setdefault("enabled", False)
    gdelt_cfg.setdefault("rate_limit_rps", 1.0)
    gdelt_cfg.setdefault("cache_minutes", 10)
    gdelt_cfg.setdefault("stale_on_error_minutes", 60)
    window_hours = gdelt_cfg.get("window_hours") or {}
    if not isinstance(window_hours, dict):
        window_hours = {}
    window_hours.setdefault("hourly", 1)
    window_hours.setdefault("daily", 6)
    gdelt_cfg["window_hours"] = window_hours
    if "queries" not in gdelt_cfg or gdelt_cfg["queries"] is None:
        gdelt_cfg["queries"] = ["war OR sanctions OR shipping"]

    health_cfg = data.get("health") or {}
    if not isinstance(health_cfg, dict):
        health_cfg = {}
    health_cfg.setdefault("min_sources_ok", 1)

    dedupe_cfg = data.get("dedupe") or {}
    if not isinstance(dedupe_cfg, dict):
        dedupe_cfg = {}
    dedupe_cfg.setdefault("enabled", True)
    dedupe_cfg.setdefault("window_days", 14)
    store_cfg = dedupe_cfg.get("store") or {}
    if not isinstance(store_cfg, dict):
        store_cfg = {}
    store_cfg.setdefault("backend", "sqlite")
    store_cfg.setdefault("path", "output/intel/news/cache/dedupe_store.sqlite")
    dedupe_cfg["store"] = store_cfg
    fp_cfg = dedupe_cfg.get("fingerprint") or {}
    if not isinstance(fp_cfg, dict):
        fp_cfg = {}
    fp_cfg.setdefault("treat_distance0_as_duplicate", True)
    dedupe_cfg["fingerprint"] = fp_cfg

    near_cfg = dedupe_cfg.get("near_duplicate") or {}
    if not isinstance(near_cfg, dict):
        near_cfg = {}
    near_cfg.setdefault("enabled", True)
    near_cfg.setdefault("hamming_threshold", 3)
    dedupe_cfg["near_duplicate"] = near_cfg

    clustering_cfg = data.get("clustering") or {}
    if not isinstance(clustering_cfg, dict):
        clustering_cfg = {}
    clustering_cfg.setdefault("enabled", False)
    clustering_cfg.setdefault("algorithm", "tfidf_cosine")
    clustering_cfg.setdefault("similarity_threshold", 0.45)
    clustering_cfg.setdefault("min_cluster_size", 3)
    clustering_cfg.setdefault("top_phrases_k", 8)
    clustering_cfg.setdefault("top_entities_k", 8)
    clustering_cfg.setdefault("max_pair_checks", 2000)
    clustering_cfg.setdefault("require_overlap", True)
    clustering_cfg.setdefault("same_day_only", True)

    burst_cfg = data.get("burst") or {}
    if not isinstance(burst_cfg, dict):
        burst_cfg = {}
    burst_cfg.setdefault("enabled", False)
    burst_cfg.setdefault("baseline_days", 30)
    burst_cfg.setdefault("min_doc_count", 3)
    burst_cfg.setdefault("top_k", 50)
    burst_cfg.setdefault("version_salt", "v1")
    # windows for burst detection; default 1h/6h/24h
    windows = burst_cfg.get("windows_hours")
    if not isinstance(windows, list) or not windows:
        windows = [1, 6, 24]
    burst_cfg["windows_hours"] = [int(w) for w in windows]

    trigger_cfg = data.get("trigger_scoring") or {}
    if not isinstance(trigger_cfg, dict):
        trigger_cfg = {}
    trigger_cfg.setdefault("enabled", False)
    trigger_cfg.setdefault("severity_cap_degraded", 1)
    trigger_cfg.setdefault("severity_cap_error", 0)

    return {
        "fetch": fetch_cfg,
        "gdelt": gdelt_cfg,
        "health": health_cfg,
        "cadence": data.get("cadence") or {},
        "dedupe": dedupe_cfg,
        "clustering": clustering_cfg,
        "burst": burst_cfg,
        "trigger_scoring": trigger_cfg,
    }
