from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .emit import emit_json_artifact


def compute_version_hash(cfg: Dict[str, Any]) -> str:
    """Compute a stable version hash for baseline config."""
    burst = cfg.get("burst") or {}
    clustering = cfg.get("clustering") or {}
    subset = {
        "burst_baseline_days": int(burst.get("baseline_days", 30) or 30),
        "burst_min_doc_count": int(burst.get("min_doc_count", 3) or 3),
        "burst_top_k": int(burst.get("top_k", 50) or 50),
        "burst_version_salt": str(burst.get("version_salt", "v1")),
        "clustering_top_phrases_k": int(clustering.get("top_phrases_k", 8) or 8),
        "clustering_top_entities_k": int(clustering.get("top_entities_k", 8) or 8),
    }
    payload = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str[:10]).replace(tzinfo=timezone.utc)


def update_baseline(
    clusters: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    now_utc: str,
    baseline_dir: Path,
) -> Dict[str, Any]:
    """Update rolling baseline (daily cadence) and return simple status."""
    baseline_dir = Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    state_path = baseline_dir / "baseline_state.json"
    latest_path = baseline_dir / "baseline_latest.json"

    version_hash = compute_version_hash(cfg)

    state: Dict[str, Any] = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    if (
        not isinstance(state, dict)
        or state.get("schema_version") != "news.baseline_state.v1"
        or state.get("version_hash") != version_hash
    ):
        state = {
            "schema_version": "news.baseline_state.v1",
            "version_hash": version_hash,
            "days": {},
        }

    days: Dict[str, Any] = state.get("days") or {}
    if not isinstance(days, dict):
        days = {}

    # Today's bucket
    today = (now_utc or "")[:10]
    if not today:
        today = datetime.now(timezone.utc).date().isoformat()

    day_bucket = days.get(today) or {"entity_counts": {}, "phrase_counts": {}}
    if not isinstance(day_bucket, dict):
        day_bucket = {"entity_counts": {}, "phrase_counts": {}}
    ent_counts = day_bucket.get("entity_counts") or {}
    phr_counts = day_bucket.get("phrase_counts") or {}
    if not isinstance(ent_counts, dict):
        ent_counts = {}
    if not isinstance(phr_counts, dict):
        phr_counts = {}

    # Aggregate counts for today from clusters
    for clu in clusters:
        ents = list(clu.get("top_entities") or []) + list(clu.get("entities") or [])
        for e in ents:
            ent_counts[e] = int(ent_counts.get(e, 0)) + 1
        for p in clu.get("top_phrases") or []:
            phr_counts[p] = int(phr_counts.get(p, 0)) + 1

    day_bucket["entity_counts"] = ent_counts
    day_bucket["phrase_counts"] = phr_counts
    days[today] = day_bucket

    # Prune old days based on baseline_days
    burst = cfg.get("burst") or {}
    baseline_days = int(burst.get("baseline_days", 30) or 30)
    try:
        today_dt = _parse_date(today)
    except Exception:
        today_dt = datetime.now(timezone.utc)

    pruned_days: Dict[str, Any] = {}
    for key, val in days.items():
        try:
            d = _parse_date(key)
            age_days = (today_dt.date() - d.date()).days
        except Exception:
            continue
        if age_days <= max(baseline_days - 1, 0):
            pruned_days[key] = val

    days = pruned_days
    state["days"] = days

    # Aggregate across window
    agg_entities: Dict[str, int] = defaultdict(int)
    agg_phrases: Dict[str, int] = defaultdict(int)
    for bucket in days.values():
        ec = bucket.get("entity_counts") or {}
        pc = bucket.get("phrase_counts") or {}
        if isinstance(ec, dict):
            for k, v in ec.items():
                agg_entities[k] += int(v)
        if isinstance(pc, dict):
            for k, v in pc.items():
                agg_phrases[k] += int(v)

    top_k = int(burst.get("top_k", 50) or 50)

    def _top_sorted(counter: Dict[str, int]) -> Dict[str, int]:
        items: List[Tuple[str, int]] = [(k, int(v)) for k, v in counter.items()]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        if top_k > 0:
            items = items[:top_k]
        return {k: v for k, v in items}

    entity_counts = _top_sorted(agg_entities)
    phrase_counts = _top_sorted(agg_phrases)

    if days:
        start_day = min(days.keys())
        end_day = max(days.keys())
    else:
        start_day = today
        end_day = today

    baseline_latest = {
        "schema_version": "news.baseline.v1",
        "generated_utc": now_utc,
        "baseline_days": baseline_days,
        "version_hash": version_hash,
        "entity_counts": entity_counts,
        "phrase_counts": phrase_counts,
        "window": {
            "start_utc": f"{start_day}T00:00:00+00:00",
            "end_utc": f"{end_day}T23:59:59+00:00",
        },
    }

    state["schema_version"] = "news.baseline_state.v1"
    state["version_hash"] = version_hash

    emit_json_artifact(state, state_path)
    emit_json_artifact(baseline_latest, latest_path)

    return {
        "version_hash": version_hash,
        "days_covered": len(days),
    }


__all__ = ["compute_version_hash", "update_baseline"]

