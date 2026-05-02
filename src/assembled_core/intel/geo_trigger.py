"""Rules engine for scoring news events/clusters into geo triggers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .models import (
    EvidenceCluster,
    GeoTrigger,
    NewsEvent,
    SourceTier,
    TriggerType,
)
from .source_registry import get_source_tier, get_trust_weight

# ---------------------------------------------------------------------------
# Keyword classification rules
# ---------------------------------------------------------------------------

KEYWORD_RULES: dict[TriggerType, list[str]] = {
    # ── Original 8 ──
    TriggerType.CHOKEPOINT_STRESS: [
        "hormuz", "suez", "strait", "chokepoint", "shipping", "blockade", "tanker"
    ],
    TriggerType.WAR_ESCALATION: [
        "war", "military", "attack", "invasion", "airstrike", "troops", "conflict", "missile"
    ],
    TriggerType.SANCTIONS_ESCALATION: [
        "sanctions", "export control", "embargo", "blacklist", "ofac"
    ],
    TriggerType.ENERGY_SUPPLY_RISK: [
        "oil", "gas", "energy", "pipeline", "refinery", "opec", "lng"
    ],
    TriggerType.SHIPPING_DISRUPTION: [
        "port", "freight", "container", "vessel", "maritime", "shipping route"
    ],
    TriggerType.CYBER_ESCALATION: [
        "cyberattack", "hack", "malware", "ransomware", "cyber"
    ],
    TriggerType.COUP_RISK: [
        "coup", "overthrow", "junta", "political instability"
    ],
    TriggerType.POLICY_SHIFT: [
        "tariff", "trade war", "decoupling", "sanctions policy", "doctrine"
    ],
    # ── Batch 1: Financial (Plan 4.3) ──
    TriggerType.BANKING_CRISIS: [
        "bank run", "bail-in", "bailout", "deposit freeze", "bank failure",
        "bank collapse", "systemic risk", "liquidity crisis", "svb",
        "contagion", "bank insolvency", "credit crunch", "bank panic",
        "fdic", "deposit flight", "interbank freeze",
    ],
    TriggerType.CREDIT_DOWNGRADE: [
        "downgrade", "junk status", "rating cut", "credit watch negative",
        "outlook negative", "moody's downgrade", "fitch downgrade",
        "s&p downgrade", "fallen angel", "investment grade",
        "creditworthiness", "debt rating", "credit deterioration",
    ],
    TriggerType.RATE_SURPRISE: [
        "hawkish surprise", "dovish pivot", "emergency rate cut",
        "rate hike surprise", "fed pivot", "rate shock",
        "unexpected tightening", "surprise easing", "yield spike",
        "central bank shock", "policy rate", "overnight rate",
        "fomc surprise", "ecb surprise", "boj surprise",
    ],
    TriggerType.PEG_STRESS: [
        "peg break", "currency floor", "currency intervention",
        "forex reserve depletion", "devaluation", "capital controls",
        "currency crisis", "peg defense", "speculative attack",
        "currency board", "dollarization", "managed float",
    ],
    TriggerType.FISCAL_CLIFF: [
        "debt ceiling", "government shutdown", "sequester",
        "fiscal cliff", "budget crisis", "spending bill",
        "continuing resolution", "default risk", "treasury",
        "appropriations", "fiscal impasse", "debt limit",
    ],
    # ── Batch 2: Military/Nuclear ──
    TriggerType.MILITARY_BUILDUP: [
        "troop deployment", "naval buildup", "mobilization",
        "military exercise", "force posture", "carrier group",
        "brigade deployment", "military staging", "armor movement",
        "airbase activation", "fleet movement", "rapid deployment",
        "military escalation", "force projection", "readiness level",
    ],
    TriggerType.NUCLEAR_THREAT: [
        "nuclear posture", "defcon", "warhead", "nuclear test",
        "nuclear alert", "nuclear rhetoric", "strategic deterrent",
        "nuclear doctrine", "icbm", "nuclear escalation",
        "nuclear submarine", "nuclear capable", "atomic",
        "nuclear arsenal", "first strike", "mutual assured",
    ],
    TriggerType.CAPABILITY_SHIFT: [
        "hypersonic", "missile test", "defense budget",
        "arms race", "weapons program", "military spending",
        "defense procurement", "next-gen weapon", "stealth",
        "ballistic missile", "cruise missile", "anti-satellite",
    ],
    # ── Batch 3: Tech/Cyber ──
    TriggerType.NEW_EXPORT_CONTROL: [
        "entity list", "chip ban", "technology denial",
        "export restriction", "semiconductor ban", "huawei",
        "technology embargo", "dual-use", "end-user restriction",
        "commerce department", "bis", "deemed export",
        "foreign direct product rule", "chip restriction",
    ],
    TriggerType.ENTITY_LISTING: [
        "sdn list", "ofac designation", "sanctions list",
        "entity list addition", "blocked person", "specially designated",
        "sanctioned entity", "restricted party", "denied person",
        "proliferation concern", "money laundering",
    ],
    TriggerType.ZERO_DAY_DISCLOSURE: [
        "zero-day", "critical vulnerability", "cve", "ransomware attack",
        "exploit", "buffer overflow", "remote code execution",
        "supply chain attack", "backdoor", "patch tuesday",
        "critical patch", "security advisory", "apt group",
    ],
    # ── Batch 4: Economic ──
    TriggerType.TRADE_WAR_ESCALATION: [
        "trade war", "retaliatory tariff", "import duty",
        "trade retaliation", "counter-tariff", "trade dispute",
        "wto complaint", "dumping", "anti-dumping", "trade barrier",
        "protectionism", "near-shoring", "friend-shoring",
    ],
    TriggerType.RESOURCE_NATIONALIZATION: [
        "nationalization", "expropriation", "resource sovereignty",
        "mining nationalization", "state takeover", "confiscation",
        "sovereign wealth", "resource tax", "windfall tax",
        "strategic resource", "critical mineral",
    ],
    TriggerType.SUPPLY_CHAIN_BREAK: [
        "supply chain disruption", "semiconductor shortage",
        "chip shortage", "supply crunch", "component shortage",
        "production halt", "factory shutdown", "supply bottleneck",
        "logistics breakdown", "inventory crisis",
    ],
    # ── Batch 5: Geopolitical ──
    TriggerType.DIPLOMATIC_CRISIS: [
        "embassy closure", "ambassador recall", "diplomatic expulsion",
        "diplomatic incident", "severed relations", "persona non grata",
        "diplomatic freeze", "consulate closure",
    ],
    TriggerType.ALLIANCE_SHIFT: [
        "nato expansion", "alliance realignment", "security pact",
        "defense agreement", "mutual defense", "aukus", "quad",
        "brics expansion", "sco", "alliance withdrawal",
    ],
    TriggerType.TERRITORIAL_ESCALATION: [
        "annexation", "territorial claim", "border clash",
        "sovereignty dispute", "territorial waters", "airspace violation",
        "occupation", "buffer zone", "demilitarized zone",
    ],
    TriggerType.STRAIT_BLOCKADE: [
        "strait blockade", "naval blockade", "maritime blockade",
        "freedom of navigation", "sea lane", "exclusion zone",
        "mine laying", "naval mine", "anti-ship missile",
    ],
}


# ---------------------------------------------------------------------------
# Event scoring
# ---------------------------------------------------------------------------


def _text_tokens(event: NewsEvent) -> str:
    """Combine all searchable text from an event into one lowercase string."""
    parts = [event.title.lower()]
    parts.extend(k.lower() for k in event.keywords)
    parts.extend(g.lower() for g in event.geo_tags)
    parts.extend(e.lower() for e in event.entities)
    return " ".join(parts)


def _kw_in_text(kw: str, text: str) -> bool:
    """Check if keyword appears in text using word-boundary matching.

    Multi-word keywords use simple substring matching (they are specific enough).
    Single-word keywords use regex word boundaries to avoid partial matches
    (e.g. 'sco' should not match 'score').
    """
    if " " in kw:
        return kw in text
    return bool(re.search(r"\b" + re.escape(kw) + r"\b", text))


def score_event(event: NewsEvent, keyword_rules: dict[TriggerType, list[str]] | None = None) -> float:
    """
    Score a single event based on keyword matches.

    Returns a float in [0, 1] representing the fraction of trigger types matched
    weighted by the trust weight of the source.
    """
    rules = keyword_rules if keyword_rules is not None else KEYWORD_RULES
    text = _text_tokens(event)
    matched_types = 0
    for _trigger_type, keywords in rules.items():
        for kw in keywords:
            if _kw_in_text(kw, text):
                matched_types += 1
                break  # count each type at most once

    if not rules:
        return 0.0

    raw_score = matched_types / len(rules)
    trust = get_trust_weight(event.source_id)
    return min(1.0, raw_score * trust)


def classify_trigger_type(
    event: NewsEvent,
    keyword_rules: dict[TriggerType, list[str]] | None = None,
) -> TriggerType | None:
    """
    Classify the most prominent trigger type for an event based on keyword matches.
    Returns None if no keywords match.
    """
    rules = keyword_rules if keyword_rules is not None else KEYWORD_RULES
    text = _text_tokens(event)

    best_type: TriggerType | None = None
    best_count = 0

    for trigger_type, keywords in rules.items():
        count = sum(1 for kw in keywords if _kw_in_text(kw, text))
        if count > best_count:
            best_count = count
            best_type = trigger_type

    return best_type if best_count > 0 else None


# ---------------------------------------------------------------------------
# Cluster scoring
# ---------------------------------------------------------------------------


def score_cluster(
    cluster: EvidenceCluster,
    events: list[NewsEvent],
    source_registry: Any = None,  # for dependency injection / testing
) -> GeoTrigger:
    """
    Score an evidence cluster and produce a GeoTrigger.

    Scoring rules:
    - Score 3: T0/T1 source present OR >= 2 independent sources with >= 1 T1/T2
    - Score 2: >= 2 independent T2+ sources
    - Score 1: single T2+ source
    - Score 0: otherwise (T3-only or no sources)
    """
    # Build source breakdown from the events in this cluster
    event_map = {e.event_id: e for e in events}
    cluster_events = [event_map[eid] for eid in cluster.supporting_events if eid in event_map]

    source_breakdown: dict[str, int] = {}
    for evt in cluster_events:
        tier = get_source_tier(evt.source_id)
        source_breakdown[tier.value] = source_breakdown.get(tier.value, 0) + 1

    # Determine tier counts
    t0_count = source_breakdown.get(SourceTier.T0.value, 0)
    t1_count = source_breakdown.get(SourceTier.T1.value, 0)
    t2_count = source_breakdown.get(SourceTier.T2.value, 0)
    _t3_count = source_breakdown.get(SourceTier.T3.value, 0)

    # Count independent sources (unique source_ids)
    unique_sources = {e.source_id for e in cluster_events}
    independent_count = len(unique_sources)

    # Scoring rules:
    # Score 3: T0/T1 source present  OR  ≥2 independent sources where ≥1 is T1 (not just T2)
    # Score 2: ≥2 independent T2+ sources (but no T0/T1)
    # Score 1: single T2+ source
    # Score 0: T3-only or no sources
    has_t0_or_t1 = (t0_count + t1_count) > 0
    has_t1 = t1_count > 0  # at least one T1 in the multi-source case

    if has_t0_or_t1 or (independent_count >= 2 and has_t1):
        trigger_score = 3
    elif independent_count >= 2 and (t2_count >= 2):
        trigger_score = 2
    elif (t2_count + t1_count + t0_count) >= 1:
        trigger_score = 1
    else:
        trigger_score = 0

    # Confidence: weighted average of source trust weights
    if cluster_events:
        total_trust = sum(get_trust_weight(e.source_id) for e in cluster_events)
        confidence = min(1.0, (total_trust / len(cluster_events)) * cluster.confidence)
    else:
        confidence = 0.0

    # TTL and decay defaults
    ttl_minutes = 360
    decay_half_life = 180

    # Generate trigger_id
    raw = f"{cluster.cluster_id}:{cluster.trigger_type.value}:{cluster.created_at.isoformat()}"
    trigger_id = "trig_" + hashlib.sha256(raw.encode()).hexdigest()[:16]

    return GeoTrigger(
        trigger_id=trigger_id,
        trigger_type=cluster.trigger_type,
        trigger_score=trigger_score,
        confidence=confidence,
        evidence_cluster_id=cluster.cluster_id,
        ttl_minutes=ttl_minutes,
        decay_half_life_minutes=decay_half_life,
        created_at=cluster.created_at,
        expires_at=cluster.expires_at,
        source_breakdown=source_breakdown,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_triggers(triggers: list[GeoTrigger]) -> dict[str, Any]:
    """
    Aggregate a list of triggers into a geo_score (max of trigger scores)
    and list of active trigger_ids.
    """
    if not triggers:
        return {"geo_score": 0, "active_triggers": []}

    geo_score = max(0.0, max(t.trigger_score for t in triggers))
    active_trigger_ids = [t.trigger_id for t in triggers]

    return {
        "geo_score": geo_score,
        "active_triggers": active_trigger_ids,
    }
