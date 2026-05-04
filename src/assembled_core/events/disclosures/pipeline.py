"""Disclosures pipeline: fetch (stub) -> normalize -> dedupe -> health -> emit."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from .dedupe import dedupe_events
from .emit import emit_json_artifact
from .fetch_edgar import fetch_edgar, fetch_edgar_form4
from .fetch_house_ptr import fetch_house_ptr_filings
from .health import compute_health
from .models import DisclosureEvent, DisclosuresHealth
from .normalize import normalize_raw_item, now_utc_iso
from .sources import DisclosureSource, load_disclosures_params, load_sources_registry
from .triggers import apply_qc_caps, score_disclosure_triggers


def _collect_raw_items(
    sources: List[DisclosureSource],
    fetch_cfg: Dict[str, Any],
    params: Dict[str, Any],
    base_dir: Path,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Call fetch per source; edgar_form4 uses real Atom fetch (or mock in tests)."""
    items: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    per_source_stats: List[Dict[str, Any]] = []

    timeout_s = float(fetch_cfg.get("timeout_s", 15.0))
    user_agent = str(fetch_cfg.get("user_agent", "Assembled-Trading-AI/Disclosures-v1"))
    edgar_cfg = params.get("edgar") or {}
    form4_cfg = dict(edgar_cfg.get("form4") or {})
    form4_cfg.setdefault("timeout_s", timeout_s)
    form4_cfg.setdefault("user_agent", user_agent)
    house_ptr_cfg = dict(params.get("house_ptr") or {})
    house_ptr_cfg.setdefault("timeout_s", timeout_s)
    house_ptr_cfg.setdefault("user_agent", user_agent)

    cache_dir = base_dir / "cache"
    fetch_state_path = cache_dir / "fetch_state.json"

    for src in sources:
        if not src.active:
            continue
        if src.type == "house_ptr":
            fetch_state = None
            if fetch_state_path.exists():
                try:
                    full_state = json.loads(
                        fetch_state_path.read_text(encoding="utf-8")
                    )
                    fetch_state = (
                        full_state.get(src.source_id)
                        if isinstance(full_state, dict)
                        else None
                    )
                except Exception as exc:
                    logger.warning(
                        "[DisclosuresPipeline] failed to load fetch state for %s: %s",
                        src.source_id,
                        exc,
                    )
                    fetch_state = None
            cfg = {**house_ptr_cfg, **src.config}
            src_items, failure, stats = fetch_house_ptr_filings(
                src.source_id, cfg, fetch_state=fetch_state
            )
            if failure is None:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    state = {}
                    if fetch_state_path.exists():
                        try:
                            state = json.loads(
                                fetch_state_path.read_text(encoding="utf-8")
                            )
                        except Exception as exc:
                            logger.warning(
                                "[DisclosuresPipeline] failed to read cache state (house_ptr): %s",
                                exc,
                            )
                        if not isinstance(state, dict):
                            state = {}
                    state[src.source_id] = {
                        "cached_utc": now_utc_iso(),
                        "last_ids": [
                            it.get("doc_id") or it.get("link") for it in src_items[:50]
                        ],
                        "cached_items": src_items,
                    }
                    fetch_state_path.parent.mkdir(parents=True, exist_ok=True)
                    fetch_state_path.write_text(
                        json.dumps(state, indent=2), encoding="utf-8"
                    )
                except Exception as exc:
                    logger.warning(
                        "[DisclosuresPipeline] failed to write cache state (house_ptr) for %s: %s",
                        src.source_id,
                        exc,
                    )
        elif src.type == "edgar_form4":
            fetch_state: Dict[str, Any] | None = None
            if fetch_state_path.exists():
                try:
                    fetch_state = json.loads(
                        fetch_state_path.read_text(encoding="utf-8")
                    )
                    fetch_state = fetch_state.get(src.source_id)
                except Exception as exc:
                    logger.warning(
                        "[DisclosuresPipeline] failed to load fetch state for %s: %s",
                        src.source_id,
                        exc,
                    )
                    fetch_state = None
            cfg = {**form4_cfg, **src.config}
            src_items, failure, stats = fetch_edgar_form4(
                src.source_id, cfg, fetch_state=fetch_state
            )
            if failure is None and src_items:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    state: Dict[str, Any] = {}
                    if fetch_state_path.exists():
                        try:
                            state = json.loads(
                                fetch_state_path.read_text(encoding="utf-8")
                            )
                        except Exception as exc:
                            logger.warning(
                                "[DisclosuresPipeline] failed to read cache state (edgar_form4): %s",
                                exc,
                            )
                        if not isinstance(state, dict):
                            state = {}
                    state[src.source_id] = {
                        "cached_utc": now_utc_iso(),
                        "cached_entries": src_items,
                    }
                    fetch_state_path.parent.mkdir(parents=True, exist_ok=True)
                    fetch_state_path.write_text(
                        json.dumps(state, indent=2), encoding="utf-8"
                    )
                except Exception as exc:
                    logger.warning(
                        "[DisclosuresPipeline] failed to write cache state (edgar_form4) for %s: %s",
                        src.source_id,
                        exc,
                    )
        elif src.type == "edgar":
            src_items, failure, stats = fetch_edgar(
                src.source_id, src.config, timeout_s=timeout_s, user_agent=user_agent
            )
        else:
            failure = {"source": src.source_id, "reason": "unknown_type"}
            stats = {
                "source_id": src.source_id,
                "type": src.type,
                "ok": False,
                "items": 0,
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

    return items, failures, per_source_stats


def run_disclosures_pipeline(
    sources_path: str | Path = "configs/disclosures/sources.yaml",
    disclosures_path: str | Path = "configs/disclosures/disclosures.yaml",
    cadence: str = "hourly",
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run disclosures pipeline: fetch (stub) -> normalize -> dedupe -> health -> emit.

    Returns dict with events, health. No real network calls in v0.
    """
    params = load_disclosures_params(disclosures_path)
    fetch_cfg = params.get("fetch", {})
    health_cfg = params.get("health", {})
    min_sources_ok = int(health_cfg.get("min_sources_ok", 1))

    base_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path("output") / "intel" / "disclosures"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    all_sources = load_sources_registry(sources_path)
    sources = [s for s in all_sources if s.active]
    fetched_utc = now_utc_iso()

    raw_items, failures, per_source_stats = _collect_raw_items(
        sources, fetch_cfg=fetch_cfg, params=params, base_dir=base_dir
    )

    events: List[DisclosureEvent] = []
    for raw in raw_items:
        source_id = str(raw.get("source_id") or "unknown")
        source_name = str(raw.get("source_name") or "unknown")
        source_domain = str(raw.get("source_domain") or "")
        ev = normalize_raw_item(
            raw,
            source_id=source_id,
            source_name=source_name,
            source_domain=source_domain,
            fetched_utc=fetched_utc,
        )
        if ev is not None:
            events.append(ev)

    deduped = dedupe_events(events)

    health: DisclosuresHealth = compute_health(
        [s.source_id for s in sources],
        items_raw=len(raw_items),
        items_after_dedupe=len(deduped),
        failures=failures,
        fetched_utc=fetched_utc,
        min_sources_ok=min_sources_ok,
    )

    # Trigger scoring (v1)
    trigger_scoring_cfg = params.get("trigger_scoring") or {}
    triggers: List[Dict[str, Any]] = []
    if trigger_scoring_cfg.get("enabled", False):
        source_meta = {
            s.source_id: {"tier": s.tier, "domain": s.domain} for s in sources
        }
        triggers = score_disclosure_triggers(
            deduped,
            source_meta,
            trigger_scoring_cfg,
            fetched_utc,
        )
        qc_gates = trigger_scoring_cfg.get("qc_gates") or {}
        triggers = apply_qc_caps(triggers, health.status, qc_gates)
    health.metrics["triggers"] = {
        "trigger_count": len(triggers),
        "max_severity": max((t.get("severity", 0) for t in triggers), default=0),
    }

    # Emit wrappers with schema versions
    events_wrapper: Dict[str, Any] = {
        "schema_version": "disclosures.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "count": len(deduped),
        "items": [e.to_dict() for e in deduped],
    }
    health_wrapper: Dict[str, Any] = {
        "schema_version": "disclosures.health.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "health": health.to_dict(),
    }
    triggers_wrapper: Dict[str, Any] = {
        "schema_version": "disclosures.triggers.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "count": len(triggers),
        "items": triggers,
    }
    fetch_report: Dict[str, Any] = {
        "schema_version": "disclosures.fetch_report.v1",
        "generated_utc": fetched_utc,
        "cadence": cadence,
        "totals": {
            "sources_total": len(per_source_stats),
            "sources_ok": sum(1 for s in per_source_stats if s.get("ok")),
            "sources_failed": sum(1 for s in per_source_stats if not s.get("ok")),
            "items_raw": len(raw_items),
        },
        "per_source": per_source_stats,
    }

    emit_json_artifact(events_wrapper, base_dir / "events_latest.json")
    emit_json_artifact(health_wrapper, base_dir / "health_latest.json")
    emit_json_artifact(triggers_wrapper, base_dir / "triggers_latest.json")
    emit_json_artifact(fetch_report, base_dir / "fetch_report_latest.json")

    return {"events": deduped, "health": health}
