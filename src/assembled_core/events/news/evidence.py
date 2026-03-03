from __future__ import annotations

from typing import Dict

from .models import NewsEvent


def summarize_cluster_evidence(
    cluster: Dict,
    events_by_id: Dict[str, NewsEvent],
    source_meta: Dict[str, Dict[str, str]],
    now_utc: str,
) -> Dict[str, object]:
    event_ids = list(cluster.get("event_ids") or [])
    tier_a_count = 0
    tier_b_count = 0
    tier_b_domains: set[str] = set()
    last_evidence_utc = ""

    for eid in event_ids:
        ev = events_by_id.get(eid)
        if ev is None:
            continue
        meta = source_meta.get(ev.source_id, {})
        tier = str(meta.get("tier", "")).upper()
        domain = str(meta.get("domain", "")).lower()

        if tier == "A":
            tier_a_count += 1
        elif tier == "B":
            tier_b_count += 1
            if domain:
                tier_b_domains.add(domain)

        ts = ev.published_utc or ""
        if ts > last_evidence_utc:
            last_evidence_utc = ts

    tier_b_independent_count = len(tier_b_domains)
    independent_domains = sorted(tier_b_domains)

    evidence_ok = (tier_a_count >= 1) or (tier_b_independent_count >= 2)

    return {
        "tierA_count": tier_a_count,
        "tierB_count": tier_b_count,
        "tierB_independent_count": tier_b_independent_count,
        "independent_domains": independent_domains,
        "last_evidence_utc": last_evidence_utc or now_utc,
        "evidence_ok": bool(evidence_ok),
    }


__all__ = ["summarize_cluster_evidence"]

