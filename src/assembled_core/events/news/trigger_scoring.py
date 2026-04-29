"""Minimal keyword-based trigger scoring for NEWS v1 (Phase 4).

Produces triggers from clusters by matching top_entities, top_phrases
and countries against geopolitical/market-relevant keyword rules.
Severity scale 0–3: 0=noise, 1=watch, 2=elevated, 3=critical.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List

TOPIC_RULES: List[Dict[str, Any]] = [
    {
        "topic_id": "geopolitical_conflict",
        "trigger_type": "geo_risk",
        "keywords": [
            "war",
            "conflict",
            "military",
            "attack",
            "missile",
            "invasion",
            "troops",
        ],
        "base_severity": 2,
    },
    {
        "topic_id": "sanctions_trade",
        "trigger_type": "geo_risk",
        "keywords": [
            "sanctions",
            "export controls",
            "trade war",
            "tariff",
            "embargo",
            "ban",
        ],
        "base_severity": 2,
    },
    {
        "topic_id": "shipping_disruption",
        "trigger_type": "supply_chain",
        "keywords": [
            "red sea",
            "suez",
            "strait of hormuz",
            "shipping",
            "freight",
            "supply chain",
            "port",
            "blockade",
            "reroute",
        ],
        "base_severity": 2,
    },
    {
        "topic_id": "taiwan_strait",
        "trigger_type": "geo_risk",
        "keywords": ["taiwan", "strait", "china sea", "pla navy", "pla military"],
        "base_severity": 3,
    },
    {
        "topic_id": "energy_crisis",
        "trigger_type": "commodity",
        "keywords": [
            "oil price",
            "energy crisis",
            "opec",
            "gas price",
            "crude",
            "oil surge",
        ],
        "base_severity": 2,
    },
    {
        "topic_id": "market_crash",
        "trigger_type": "market_stress",
        "keywords": [
            "crash",
            "slump",
            "plunge",
            "sell-off",
            "selloff",
            "panic",
            "circuit breaker",
        ],
        "base_severity": 2,
    },
    {
        "topic_id": "central_bank",
        "trigger_type": "macro",
        "keywords": [
            "fed",
            "ecb",
            "rate hike",
            "rate cut",
            "interest rate",
            "monetary policy",
            "central bank",
        ],
        "base_severity": 1,
    },
    {
        "topic_id": "nuclear_risk",
        "trigger_type": "geo_risk",
        "keywords": ["nuclear", "uranium", "atomic"],
        "base_severity": 3,
    },
]


def _match_score(text_lower: str, keywords: List[str]) -> int:
    """Count how many distinct keywords appear in the text."""
    return sum(1 for kw in keywords if kw in text_lower)


def score_triggers(
    clusters: List[Dict[str, Any]],
    events_by_id: Dict[str, Any],
    *,
    health_status: str = "OK",
    severity_cap_degraded: int = 1,
    severity_cap_error: int = 0,
    generated_utc: str = "",
) -> List[Dict[str, Any]]:
    """Score clusters against topic rules and produce trigger dicts.

    Returns a list of trigger dicts ready for triggers_latest.json.
    """
    severity_cap = None
    if health_status == "ERROR":
        severity_cap = severity_cap_error
    elif health_status == "DEGRADED":
        severity_cap = severity_cap_degraded

    triggers: List[Dict[str, Any]] = []

    for clu in clusters:
        cluster_id = clu.get("cluster_id", "")
        event_ids = clu.get("event_ids", [])
        top_entities = clu.get("top_entities", [])
        top_phrases = clu.get("top_phrases", [])
        sample_titles = clu.get("sample_titles", [])
        countries = clu.get("countries", [])
        evidence = clu.get("evidence", {})

        cluster_text_parts: List[str] = []
        cluster_text_parts.extend(top_entities)
        cluster_text_parts.extend(top_phrases)
        cluster_text_parts.extend(sample_titles)
        for eid in event_ids[:5]:
            ev = events_by_id.get(eid)
            if ev is None:
                continue
            title = (
                getattr(ev, "title", "")
                if hasattr(ev, "title")
                else ev.get("title", "")
            )
            summary = (
                getattr(ev, "summary", "")
                if hasattr(ev, "summary")
                else ev.get("summary", "")
            )
            cluster_text_parts.append(str(title or ""))
            cluster_text_parts.append(str(summary or ""))

        combined = " ".join(cluster_text_parts).lower()

        for rule in TOPIC_RULES:
            hits = _match_score(combined, rule["keywords"])
            if hits == 0:
                continue

            severity = rule["base_severity"]
            confidence = min(1.0, 0.3 + 0.15 * hits)

            if len(event_ids) >= 5:
                severity = min(severity + 1, 3)
                confidence = min(confidence + 0.1, 1.0)

            if severity_cap is not None:
                severity = min(severity, severity_cap)

            evidence_ok = bool(evidence.get("evidence_ok", True))

            trigger_id_raw = f"{cluster_id}:{rule['topic_id']}"
            trigger_id = f"trg_{sha256(trigger_id_raw.encode()).hexdigest()[:12]}"

            triggers.append(
                {
                    "trigger_id": trigger_id,
                    "cluster_id": cluster_id,
                    "trigger_type": rule["trigger_type"],
                    "topic_id": rule["topic_id"],
                    "severity": severity,
                    "confidence": round(confidence, 3),
                    "keyword_hits": hits,
                    "event_count": len(event_ids),
                    "countries": countries,
                    "evidence_ok": evidence_ok,
                    "sample_title": sample_titles[0] if sample_titles else "",
                    "generated_utc": generated_utc,
                }
            )

    triggers.sort(key=lambda t: (-t["severity"], -t["confidence"], t["trigger_id"]))
    return triggers
