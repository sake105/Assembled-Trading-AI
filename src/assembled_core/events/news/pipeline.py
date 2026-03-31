from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.assembled_core.logging_config import setup_logging

from .baseline import update_baseline
from .burst import compute_bursts_for_window
from .clustering import build_clusters
from .dedupe import dedupe_events
from .dedupe_store import DedupeStoreSQLite
from .emit import emit_json_artifact
from .evidence import summarize_cluster_evidence
from .fetch_gdelt import fetch_gdelt_events
from .fetch_rss import fetch_rss_feed
from .health import compute_health
from .fingerprint import hamming_distance
from .models import NewsEvent, NewsHealth
from .normalize import normalize_raw_item, now_utc_iso
from .sources import NewsSource, load_news_params, load_sources_registry
from .state import load_fetch_state, save_fetch_state
from .trigger_scoring import score_triggers


def _collect_raw_items(
    sources: List[NewsSource],
    fetch_cfg: Dict[str, Any],
    gdelt_cfg: Dict[str, Any],
    fetch_state: Dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch RSS (concurrently) and GDELT (sequentially) and collect stats."""
    items: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    per_source_stats: list[dict[str, Any]] = []

    # RSS in parallel
    rss_sources = [s for s in sources if s.active and s.type == "rss"]
    max_workers = int(fetch_cfg.get("max_concurrency", 5) or 1)
    timeout = float(fetch_cfg.get("timeout_s", 10.0))
    retries = int(fetch_cfg.get("retries", 2))
    backoff_base_s = float(fetch_cfg.get("backoff_base_s", 0.5))
    user_agent = str(fetch_cfg.get("user_agent", "Assembled-Trading-AI/NEWS-v1"))
    sanitize_cfg = fetch_cfg.get("sanitize") or {}

    if rss_sources and max_workers > 0:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_src = {
                executor.submit(
                    fetch_rss_feed,
                    src.source_id,
                    str(src.config.get("url") or ""),
                    timeout=timeout,
                    user_agent=user_agent,
                    sanitize_cfg=sanitize_cfg,
                    fetch_state=fetch_state,
                    retries=retries,
                    backoff_base_s=backoff_base_s,
                ): src
                for src in rss_sources
                if str(src.config.get("url") or "")
            }
            for future in as_completed(future_to_src):
                src = future_to_src[future]
                try:
                    src_items, failure, stats = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    failure = {
                        "source": src.source_id,
                        "reason": f"rss_fetch_exception: {exc}",
                    }
                    stats = {
                        "source_id": src.source_id,
                        "type": "rss",
                        "ok": False,
                        "http_status": None,
                        "duration_ms": 0,
                        "items": 0,
                        "not_modified": False,
                        "cached": False,
                        "error": str(exc),
                    }
                    src_items = []
                per_source_stats.append(stats)
                if failure is not None:
                    failures.append(failure)
                for it in src_items:
                    it["source_id"] = src.source_id
                    it["source_name"] = src.name
                    it["source_domain"] = src.domain
                    items.append(it)

    # GDELT sequential
    for src in sources:
        if not src.active or src.type != "gdelt":
            continue
        query = str(src.config.get("query") or "")
        if not query:
            failures.append({"source": src.source_id, "reason": "gdelt_missing_query"})
            continue
        src_items, failure, stats = fetch_gdelt_events(
            src.source_id, query, gdelt_cfg=gdelt_cfg, cadence="hourly", fetch_state=fetch_state
        )
        per_source_stats.append(stats)
        if failure is not None:
            failures.append(failure)
        for it in src_items:
            it["source_id"] = src.source_id
            it["source_name"] = src.name
            it["source_domain"] = src.domain
            items.append(it)

    return items, failures, per_source_stats


def run_news_pipeline(
    sources_path: str | Path = "configs/news/sources.yaml",
    news_path: str | Path = "configs/news/news.yaml",
    cadence: str = "hourly",
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run NEWS v1 pipeline: fetch -> normalize -> dedupe -> health -> emit.

    Returns a dict with:
      - events: List[NewsEvent]
      - health: NewsHealth
    """
    # Minimal logging setup (no custom run_id necessary for this MVP)
    setup_logging(run_id="news_v1", level="INFO")

    params = load_news_params(news_path)
    fetch_cfg = params.get("fetch", {})
    gdelt_cfg = params.get("gdelt", {})
    health_cfg = params.get("health", {})
    dedupe_cfg = params.get("dedupe", {})
    clustering_cfg = params.get("clustering", {})
    burst_cfg = params.get("burst", {})
    trigger_scoring_cfg = params.get("trigger_scoring", {})
    min_sources_ok = int(health_cfg.get("min_sources_ok", 1))

    # Determine base output directory
    base_dir = Path(output_dir) if output_dir is not None else Path("output") / "intel" / "news"

    # Load persistent fetch state (rss etag/last-modified + gdelt cache)
    cache_dir = base_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fetch_state_path = cache_dir / "fetch_state.json"
    fetch_state = load_fetch_state(fetch_state_path)

    all_sources = load_sources_registry(sources_path)
    sources = [s for s in all_sources if s.active]
    source_meta: Dict[str, Dict[str, str]] = {
        s.source_id: {
            "tier": str(getattr(s, "tier", "")).upper(),
            "domain": str(getattr(s, "domain", "")).lower(),
        }
        for s in all_sources
    }
    fetched_utc = now_utc_iso()

    # Initialize dedupe store
    dedupe_enabled = bool(dedupe_cfg.get("enabled", True))
    dedupe_store = None
    dropped_url = 0
    dropped_fp0 = 0
    near_dupes_tagged = 0
    if dedupe_enabled:
        store_cfg = dedupe_cfg.get("store") or {}
        cfg_store_path = store_cfg.get("path")
        if output_dir is not None and (not cfg_store_path or not Path(cfg_store_path).is_absolute()):
            store_path = base_dir / "cache" / "dedupe_store.sqlite"
        else:
            store_path = Path(str(cfg_store_path or (base_dir / "cache" / "dedupe_store.sqlite")))
        dedupe_store = DedupeStoreSQLite(store_path)

        # Prune old entries
        window_days = int(dedupe_cfg.get("window_days", 14) or 14)
        try:
            dedupe_store.prune(window_days=window_days, now_utc=fetched_utc)
        except Exception:
            # Defensive: dedupe failure must not break pipeline
            dedupe_store = None

    raw_items, failures, per_source_stats = _collect_raw_items(
        sources, fetch_cfg=fetch_cfg, gdelt_cfg=gdelt_cfg, fetch_state=fetch_state
    )

    # NEWS-DEBUG-1: Funnel counts (observability only)
    funnel_counts: Dict[str, int] = {
        "raw_items_count": len(raw_items),
        "normalized_events_count": 0,
        "normalized_ok_count": 0,
        "dedupe_store_dropped_url_count": 0,
        "dedupe_store_dropped_fp0_count": 0,
        "post_store_kept_count": 0,
        "normalize_exception_count": 0,
        "normalize_none_count": 0,
        "dropped_short_title_count": 0,
        "deduped_events_count": 0,
        "clusters_count": 0,
        "clusters_with_topics_count": 0,
        "candidate_triggers_count": 0,
        "triggers_count": 0,
        "triggers_severity_ge_1_count": 0,
        "triggers_severity_ge_2_count": 0,
        "triggers_evidence_blocked_count": 0,
        "triggers_qc_capped_count": 0,
    }
    funnel_notes: List[str] = []
    # NEWS-DEBUG-2: normalize failure reasons + samples (max 3 total)
    normalize_exception_reasons: Dict[str, int] = {}
    normalize_none_reasons: Dict[str, int] = {}
    normalize_failure_samples: List[Dict[str, Any]] = []
    MAX_NORMALIZE_SAMPLES = 3

    events: List[NewsEvent] = []
    dropped_short_title = 0
    for raw in raw_items:
        source_id = str(raw.get("source_id") or "unknown")
        source_name = str(raw.get("source_name") or "unknown")
        source_domain = str(raw.get("source_domain") or "")
        raw_preview = {
            "title": raw.get("title"),
            "link": raw.get("link"),
            "published": raw.get("published"),
            "keys": sorted(list(raw.keys()))[:20],
        }
        try:
            ev = normalize_raw_item(
                {
                    "title": raw.get("title"),
                    "link": raw.get("link"),
                    "published": raw.get("published"),
                    "summary": raw.get("summary"),
                    "raw": raw.get("raw", raw),
                },
                source_id=source_id,
                source_name=source_name,
                source_domain=source_domain,
                fetched_utc=fetched_utc,
            )
        except Exception as exc:
            funnel_counts["normalize_exception_count"] += 1
            reason = f"{type(exc).__name__}: {str(exc)[:200]}"
            normalize_exception_reasons[reason] = normalize_exception_reasons.get(reason, 0) + 1
            if len(normalize_failure_samples) < MAX_NORMALIZE_SAMPLES:
                normalize_failure_samples.append({
                    "kind": "exception",
                    "reason": reason,
                    "raw_preview": raw_preview,
                })
            continue
        if ev is None:
            funnel_counts["normalize_none_count"] += 1
            none_reason = "returned_none"
            normalize_none_reasons[none_reason] = normalize_none_reasons.get(none_reason, 0) + 1
            if len(normalize_failure_samples) < MAX_NORMALIZE_SAMPLES:
                normalize_failure_samples.append({
                    "kind": "returned_none",
                    "reason": none_reason,
                    "raw_preview": raw_preview,
                })
            dropped_short_title += 1
            continue

        funnel_counts["normalized_ok_count"] += 1
        # Persistent dedupe: URL + fingerprint bucket (distance 0)
        if dedupe_store is not None and dedupe_enabled:
            try:
                if ev.canonical_url and dedupe_store.has_url(ev.canonical_url):
                    dropped_url += 1
                    continue
                fp64_hex = getattr(ev, "fingerprint64", "") or ""
                treat_fp0 = bool(
                    (dedupe_cfg.get("fingerprint") or {}).get(
                        "treat_distance0_as_duplicate", True
                    )
                )
                if treat_fp0 and fp64_hex:
                    fp64_int = int(fp64_hex, 16)
                    has_fp, _ = dedupe_store.has_fingerprint64(fp64_int)
                    if has_fp:
                        dropped_fp0 += 1
                        continue

                # Near-duplicate tagging (do not drop)
                near_cfg = dedupe_cfg.get("near_duplicate") or {}
                if fp64_hex and bool(near_cfg.get("enabled", True)):
                    threshold = int(near_cfg.get("hamming_threshold", 3) or 3)
                    if threshold > 0:
                        fp64_int_nd = int(fp64_hex, 16)
                        bucket = DedupeStoreSQLite._bucket(fp64_int_nd)
                        candidates = dedupe_store.candidates_by_bucket(bucket)
                        best_event_id: str | None = None
                        best_dist: int | None = None
                        for cand_event_id, cand_fp in candidates:
                            dist = hamming_distance(fp64_int_nd, cand_fp)
                            if dist == 0:
                                # Exact duplicate already handled above
                                continue
                            if best_dist is None or dist < best_dist:
                                best_dist = dist
                                best_event_id = cand_event_id
                        if best_event_id is not None and best_dist is not None and best_dist <= threshold:
                            if not isinstance(ev.raw, dict):
                                ev.raw = {}
                            ev.raw["near_duplicate_of"] = best_event_id
                            ev.raw["near_duplicate_distance"] = int(best_dist)
                            near_dupes_tagged += 1
            except Exception:
                # Do not break pipeline on store errors
                pass

        events.append(ev)

    funnel_counts["normalized_events_count"] = len(events)
    funnel_counts["post_store_kept_count"] = len(events)
    funnel_counts["dedupe_store_dropped_url_count"] = dropped_url
    funnel_counts["dedupe_store_dropped_fp0_count"] = dropped_fp0
    funnel_counts["dropped_short_title_count"] = dropped_short_title

    deduped = dedupe_events(events)
    funnel_counts["deduped_events_count"] = len(deduped)

    # Clustering
    clusters: List[Dict[str, Any]] = []
    if bool(clustering_cfg.get("enabled", True)):
        cfg_for_clusters = dict(clustering_cfg)
        cfg_for_clusters["generated_utc"] = fetched_utc
        clusters = build_clusters(deduped, cfg_for_clusters)

    # Evidence summary per cluster
    events_by_id: Dict[str, NewsEvent] = {e.event_id: e for e in deduped}
    for clu in clusters:
        clu["evidence"] = summarize_cluster_evidence(
            clu, events_by_id, source_meta, fetched_utc
        )

    funnel_counts["clusters_count"] = len(clusters)
    funnel_counts["clusters_with_topics_count"] = sum(
        1 for c in clusters if (c.get("topics") or c.get("candidate_triggers"))
    )
    funnel_counts["candidate_triggers_count"] = sum(
        len(c.get("candidate_triggers") or []) for c in clusters
    )

    # Baseline update (daily cadence only)
    baseline_meta = {"version_hash": "", "days_covered": 0}
    if cadence == "daily" and bool(burst_cfg.get("enabled", False)):
        try:
            cfg_for_baseline = {
                "burst": burst_cfg,
                "clustering": clustering_cfg,
            }
            baseline_dir = base_dir / "baseline"
            baseline_meta = update_baseline(
                clusters=clusters,
                cfg=cfg_for_baseline,
                now_utc=fetched_utc,
                baseline_dir=baseline_dir,
            )
        except Exception:
            baseline_meta = {"version_hash": "", "days_covered": 0}

    # Burst detection (all cadences; baseline optional)
    bursts_primary: List[Dict[str, Any]] = []
    window_hours_primary = 24 if cadence == "daily" else 1
    baseline_loaded = False
    baseline_latest: Dict[str, Any] | None = None
    baseline_dir = base_dir / "baseline"
    baseline_latest_path = baseline_dir / "baseline_latest.json"
    if baseline_latest_path.exists():
        try:
            import json

            baseline_latest = json.loads(baseline_latest_path.read_text(encoding="utf-8"))
            baseline_loaded = isinstance(baseline_latest, dict)
        except Exception:
            baseline_latest = None
            baseline_loaded = False

    bursts_windows: List[Dict[str, Any]] = []
    windows_cfg = burst_cfg.get("windows_hours") or [1, 6, 24]
    windows: List[int] = sorted({int(w) for w in windows_cfg})

    if bool(burst_cfg.get("enabled", False)):
        cfg_for_burst = {
            "burst": burst_cfg,
            "clustering": clustering_cfg,
        }
        for wh in windows:
            bw = compute_bursts_for_window(
                clusters=clusters,
                baseline=baseline_latest,
                cfg=cfg_for_burst,
                window_hours=wh,
            )
            bursts_windows.append(bw)

        # Determine primary window for backward-compatible count/items
        primary_map = (burst_cfg.get("primary_window_by_cadence") or {}) if isinstance(
            burst_cfg.get("primary_window_by_cadence"), dict
        ) else {}
        if cadence in primary_map:
            window_hours_primary = int(primary_map[cadence])
        else:
            window_hours_primary = 1 if cadence == "hourly" else 24
        if window_hours_primary not in windows:
            window_hours_primary = windows[0]

        primary = next(
            (bw for bw in bursts_windows if int(bw.get("window_hours")) == window_hours_primary),
            {"top_entities_burst": [], "top_phrases_burst": []},
        )

        # Flatten primary window bursts into legacy items list
        items_flat: List[Dict[str, Any]] = []
        for it in primary.get("top_entities_burst", []):
            items_flat.append(it)
        for it in primary.get("top_phrases_burst", []):
            items_flat.append(it)
        # Keep same ranking as window-level (already sorted per type, but we ensure global order)
        items_flat.sort(key=lambda x: (-x["score"], x["kind"], x["key"]))
        bursts_primary = items_flat

    # After in-memory dedupe, persist kept events to store
    if dedupe_store is not None and dedupe_enabled:
        for ev in deduped:
            try:
                fp64_hex = getattr(ev, "fingerprint64", "") or ""
                fp64_int = int(fp64_hex, 16) if fp64_hex else 0
                dedupe_store.add_event(
                    event_id=ev.event_id,
                    canonical_url=ev.canonical_url,
                    fp64=fp64_int,
                    published_utc=ev.published_utc,
                    source_id=ev.source_id,
                    ingested_utc=fetched_utc,
                )
            except Exception:
                continue

    health: NewsHealth = compute_health(
        [s.source_id for s in sources],
        items_raw=len(raw_items),
        items_after_dedupe=len(deduped),
        failures=failures,
        fetched_utc=fetched_utc,
        min_sources_ok=min_sources_ok,
    )

    # Cluster quality metrics
    total_events = len(deduped)
    cluster_count = len(clusters)
    clustered_events = sum(len(c.get("event_ids", [])) for c in clusters)
    unclustered_events = max(total_events - clustered_events, 0)
    avg_cluster_size = (
        float(clustered_events) / float(cluster_count) if cluster_count > 0 else 0.0
    )
    pct_unclustered = (
        float(unclustered_events) / float(total_events) if total_events > 0 else 0.0
    )
    health.metrics["cluster_quality"] = {
        "cluster_count": cluster_count,
        "avg_cluster_size": avg_cluster_size,
        "pct_unclustered": pct_unclustered,
        "total_events": total_events,
        "clustered_events": clustered_events,
        "unclustered_events": unclustered_events,
    }

    # Baseline metrics
    health.metrics["baseline"] = {
        "version_hash": baseline_meta.get("version_hash", ""),
        "has_baseline": bool(baseline_meta.get("days_covered", 0) > 0),
        "days_covered": int(baseline_meta.get("days_covered", 0)),
    }
    # Bursts metrics
    burst_counts_by_window: Dict[str, int] = {}
    for w in bursts_windows:
        wh = int(w.get("window_hours", 0))
        n = len(w.get("top_entities_burst", [])) + len(w.get("top_phrases_burst", []))
        burst_counts_by_window[str(wh)] = n

    health.metrics["bursts"] = {
        "window_hours": int(window_hours_primary),
        "burst_count": len(bursts_primary),
        "baseline_loaded": bool(baseline_loaded),
        "windows_hours": windows,
        "burst_counts_by_window": burst_counts_by_window,
    }

    # Trigger scoring (M1-T09): score clusters against topic rules
    trigger_items: List[Dict[str, Any]] = []
    if bool(trigger_scoring_cfg.get("enabled", True)):
        _sev_cap_degraded = int(trigger_scoring_cfg.get("severity_cap_degraded", 1))
        _sev_cap_error = int(trigger_scoring_cfg.get("severity_cap_error", 0))
        trigger_items = score_triggers(
            clusters=clusters,
            events_by_id=events_by_id,
            health_status=health.status,
            severity_cap_degraded=_sev_cap_degraded,
            severity_cap_error=_sev_cap_error,
            generated_utc=fetched_utc,
        )
    health.metrics["triggers"] = {
        "trigger_count": len(trigger_items),
        "max_severity": max((t["severity"] for t in trigger_items), default=0),
    }

    if dropped_short_title > 0:
        health.notes.append(f"dropped_short_title:{dropped_short_title}")
    if dropped_url > 0:
        health.notes.append(f"dedupe_dropped_url:{dropped_url}")
    if dropped_fp0 > 0:
        health.notes.append(f"dedupe_dropped_fp0:{dropped_fp0}")
    if near_dupes_tagged > 0:
        health.notes.append(f"near_dupes_tagged:{near_dupes_tagged}")
    if clusters:
        health.notes.append(f"clusters_count:{len(clusters)}")
    health.notes.append(f"dedupe_kept:{len(deduped)}")

    # Emit artifacts
    events_path = base_dir / "events_latest.json"
    health_path = base_dir / "health_latest.json"
    clusters_path = base_dir / "clusters_latest.json"
    triggers_path = base_dir / "triggers_latest.json"
    fetch_report_path = base_dir / "fetch_report_latest.json"
    bursts_path = base_dir / "bursts_latest.json"

    events_wrapper: Dict[str, Any] = {
        "schema_version": "news.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "count": len(deduped),
        "items": [e.to_dict() for e in deduped],
    }
    health_wrapper: Dict[str, Any] = {
        "schema_version": "news.health.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "health": health.to_dict(),
    }
    clusters_wrapper: Dict[str, Any] = {
        "schema_version": "news.clusters.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "count": len(clusters),
        "items": clusters,
    }
    triggers_wrapper: Dict[str, Any] = {
        "schema_version": "news.triggers.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "count": len(trigger_items),
        "items": trigger_items,
    }
    # Funnel: trigger counts
    funnel_counts["triggers_count"] = len(trigger_items)
    # candidate_triggers_count already set from clusters above
    for t in trigger_items:
        if not isinstance(t, dict):
            continue
        sev = int(t.get("severity", 0))
        if sev >= 1:
            funnel_counts["triggers_severity_ge_1_count"] += 1
        if sev >= 2:
            funnel_counts["triggers_severity_ge_2_count"] += 1
        if t.get("evidence_blocked"):
            funnel_counts["triggers_evidence_blocked_count"] += 1
        if t.get("qc_capped"):
            funnel_counts["triggers_qc_capped_count"] += 1

    bursts_wrapper: Dict[str, Any] = {
        "schema_version": "news.bursts.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "window_hours": int(window_hours_primary),
        "baseline_version_hash": baseline_latest.get("version_hash") if baseline_latest else None,
        # Backward-compatible primary window view
        "count": len(bursts_primary),
        "items": bursts_primary,
        # New multi-window structure
        "windows": bursts_windows,
    }

    emit_json_artifact(events_wrapper, events_path)
    emit_json_artifact(health_wrapper, health_path)
    emit_json_artifact(clusters_wrapper, clusters_path)
    emit_json_artifact(triggers_wrapper, triggers_path)
    emit_json_artifact(
        {
            "schema_version": "news.debug_funnel.v1",
            "generated_utc": fetched_utc,
            "cadence": cadence,
            "counts": funnel_counts,
            "normalize_exception_reasons": normalize_exception_reasons,
            "normalize_none_reasons": normalize_none_reasons,
            "samples": normalize_failure_samples,
            "notes": funnel_notes,
        },
        base_dir / "debug_funnel_latest.json",
    )
    emit_json_artifact(bursts_wrapper, bursts_path)

    # Daily housekeeping: prune old GDELT cache entries and emit housekeeping artifact
    if cadence == "daily":
        from dateutil import parser as date_parser

        stale_minutes = int(gdelt_cfg.get("stale_on_error_minutes", 60))
        prune_threshold = stale_minutes + 120
        pruned = 0
        gdelt_state = fetch_state.get("gdelt") or {}
        keys = list(gdelt_state.keys())

        def _age_minutes(ts: str) -> float:
            try:
                dt = date_parser.parse(ts)
                return (
                    date_parser.parse(fetched_utc) - dt
                ).total_seconds() / 60.0
            except Exception:
                return 1e9

        for key in keys:
            entry = gdelt_state.get(key)
            if not isinstance(entry, dict):
                continue
            cached_utc = entry.get("cached_utc")
            if not isinstance(cached_utc, str):
                continue
            if _age_minutes(cached_utc) > prune_threshold:
                gdelt_state.pop(key, None)
                pruned += 1

        fetch_state["gdelt"] = gdelt_state
        housekeeping = {
            "schema_version": "news.housekeeping.v1",
            "generated_utc": fetched_utc,
            "cadence": "daily",
            "pruned_gdelt_cache_entries": pruned,
            "notes": [],
        }
        emit_json_artifact(housekeeping, base_dir / "daily_housekeeping_latest.json")

    # Fetch report
    totals = {
        "sources_total": len(per_source_stats),
        "sources_ok": sum(1 for s in per_source_stats if s.get("ok")),
        "sources_failed": sum(1 for s in per_source_stats if not s.get("ok")),
        "items_raw": len(raw_items),
    }
    fetch_report: Dict[str, Any] = {
        "schema_version": "news.fetch_report.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "totals": totals,
        "per_source": per_source_stats,
    }
    emit_json_artifact(fetch_report, fetch_report_path)

    # Best-effort: persist fetch_state (even on failures)
    try:
        save_fetch_state(fetch_state, fetch_state_path)
    except Exception:
        # Logging already configured; do not crash pipeline due to state persistence
        import logging

        logging.getLogger(__name__).warning(
            "Failed to save fetch_state.json (non-fatal).", exc_info=True
        )

    return {"events": deduped, "health": health}

