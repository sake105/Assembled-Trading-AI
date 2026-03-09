"""Evidence summarization for disclosures (single-event or multi-event)."""

from __future__ import annotations

from typing import Any, Dict, List

from .models import DisclosureEvent


def summarize_evidence(
    events: List[DisclosureEvent],
    source_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize evidence from a list of disclosure events using source tier/domain.

    source_meta: map source_id -> {tier, domain} (from sources registry).
    Returns: tierA_count, tierB_count, tierB_independent_domains, evidence_ok, last_evidence_utc.
    """
    tier_a_count = 0
    tier_b_count = 0
    tier_b_domains: set[str] = set()
    last_evidence_utc = ""

    for ev in events:
        meta = source_meta.get(ev.source_id, {})
        tier = str(meta.get("tier", "")).strip().upper()
        domain = str(meta.get("domain", "")).strip().lower()

        if tier == "A":
            tier_a_count += 1
        elif tier == "B":
            tier_b_count += 1
            if domain:
                tier_b_domains.add(domain)

        ts = ev.published_utc or ""
        if ts > last_evidence_utc:
            last_evidence_utc = ts

    tier_b_independent_domains = len(tier_b_domains)
    evidence_ok = (tier_a_count >= 1) or (tier_b_independent_domains >= 2)

    return {
        "tierA_count": tier_a_count,
        "tierB_count": tier_b_count,
        "tierB_independent_domains": tier_b_independent_domains,
        "evidence_ok": bool(evidence_ok),
        "last_evidence_utc": last_evidence_utc,
    }
