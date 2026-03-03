from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def _accumulate_counts(clusters: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    ent_counts: Dict[str, int] = {}
    phr_counts: Dict[str, int] = {}
    for clu in clusters:
        for e in clu.get("top_entities") or clu.get("entities") or []:
            ent_counts[e] = ent_counts.get(e, 0) + 1
        for p in clu.get("top_phrases") or []:
            phr_counts[p] = phr_counts.get(p, 0) + 1
    return ent_counts, phr_counts


def compute_bursts_for_window(
    clusters: List[Dict[str, Any]],
    baseline: Dict[str, Any] | None,
    cfg: Dict[str, Any],
    window_hours: int,
) -> Dict[str, Any]:
    """Compute bursts for a specific window (hours)."""
    burst_cfg = cfg.get("burst") or {}
    min_doc_count = int(burst_cfg.get("min_doc_count", 3) or 3)
    top_k = int(burst_cfg.get("top_k", 50) or 50)
    window_h = float(window_hours)

    cur_ent, cur_phr = _accumulate_counts(clusters)

    base_days = 0
    base_entities: Dict[str, int] = {}
    base_phrases: Dict[str, int] = {}
    if isinstance(baseline, dict):
        try:
            base_days = int(baseline.get("baseline_days", 0) or 0)
        except Exception:
            base_days = 0
        base_entities = baseline.get("entity_counts") or {}
        base_phrases = baseline.get("phrase_counts") or {}
        if not isinstance(base_entities, dict):
            base_entities = {}
        if not isinstance(base_phrases, dict):
            base_phrases = {}

    def _compute_items(kind: str, cur: Dict[str, int], base: Dict[str, int]) -> List[Dict[str, Any]]:
        res: List[Dict[str, Any]] = []
        for key, current in cur.items():
            if current < min_doc_count:
                continue
            if base_days > 0:
                baseline_total = float(base.get(key, 0))
                baseline_avg_per_day = baseline_total / float(base_days)
                expected = baseline_avg_per_day * (window_h / 24.0)
            else:
                expected = 0.0
            ratio = (float(current) + 1.0) / (expected + 1.0)
            score = math.log(ratio) * float(current)
            res.append(
                {
                    "kind": kind,
                    "key": key,
                    "current": int(current),
                    "expected": float(expected),
                    "ratio": float(ratio),
                    "score": float(score),
                }
            )
        # sort and top_k
        res.sort(key=lambda x: (-x["score"], x["key"]))
        if top_k > 0:
            res = res[:top_k]
        return res

    ent_items = _compute_items("entity", cur_ent, base_entities)
    phr_items = _compute_items("phrase", cur_phr, base_phrases)

    # Cluster bursts: max score across bursty keys in each cluster
    key_to_score: Dict[str, float] = {}
    for it in ent_items + phr_items:
        prev = key_to_score.get(it["key"], float("-inf"))
        if it["score"] > prev:
            key_to_score[it["key"]] = float(it["score"])

    cluster_items: List[Dict[str, Any]] = []
    for clu in clusters:
        keys = list(clu.get("top_entities") or []) + list(clu.get("entities") or []) + list(
            clu.get("top_phrases") or []
        )
        best = float("-inf")
        for k in keys:
            sc = key_to_score.get(k)
            if sc is not None and sc > best:
                best = sc
        if best > 0.0:
            cid = str(clu.get("cluster_id") or "")
            cluster_items.append({"cluster_id": cid, "score": float(best)})

    cluster_items.sort(key=lambda x: (-x["score"], x["cluster_id"]))
    if top_k > 0:
        cluster_items = cluster_items[:top_k]

    return {
        "window_hours": int(window_h),
        "top_entities_burst": ent_items,
        "top_phrases_burst": phr_items,
        "top_clusters_burst": cluster_items,
    }


__all__ = ["compute_bursts_for_window"]

