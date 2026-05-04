"""Bayesian confidence updating for the intel pipeline.

Replaces simple multiplicative confidence calculation with proper
Bayesian sequential updates. Each new piece of evidence updates
the posterior probability of a geopolitical trigger being real.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Source tier reliability priors (P(E|H) — likelihood of seeing this evidence given trigger is real)
SOURCE_RELIABILITY: dict[str, float] = {
    "T0": 0.95,  # Government sanctions lists — very reliable
    "T1": 0.85,  # Licensed newswires (AP, Reuters)
    "T1.5": 0.75,  # CFTC-regulated prediction markets (Polymarket, Kalshi) — crowd-sourced, liquid
    "T2": 0.65,  # Open/aggregator (GDELT, ACLED, World Bank)
    "T3": 0.35,  # Scrapes/social — low reliability
}

# False positive rate per source tier (P(E|~H) — probability of seeing evidence even if trigger is false)
FALSE_POSITIVE_RATE: dict[str, float] = {
    "T0": 0.02,
    "T1": 0.10,
    "T1.5": 0.15,  # Prediction markets occasionally price in noise; slightly higher FPR than newswires
    "T2": 0.20,
    "T3": 0.40,
}

# Initial priors per TriggerType (baseline probability before any evidence)
TRIGGER_BASE_PRIORS: dict[str, float] = {
    "CHOKEPOINT_STRESS": 0.05,
    "WAR_ESCALATION": 0.03,
    "SANCTIONS_ESCALATION": 0.08,
    "ENERGY_SUPPLY_RISK": 0.10,
    "SHIPPING_DISRUPTION": 0.07,
    "CYBER_ESCALATION": 0.06,
    "COUP_RISK": 0.02,
    "POLICY_SHIFT": 0.15,
    # New trigger types — less frequent
    "TRADE_WAR_ESCALATION": 0.05,
    "ALLIANCE_SHIFT": 0.02,
    "RESOURCE_NATIONALIZATION": 0.03,
    "STRAIT_BLOCKADE": 0.02,
    "HEGEMONIC_CHALLENGE": 0.04,
    "DIPLOMATIC_CRISIS": 0.08,
    "MILITARY_BUILDUP": 0.05,
    "NUCLEAR_THREAT": 0.005,
    "NEW_EXPORT_CONTROL": 0.06,
    "ENTITY_LISTING": 0.07,
    "ZERO_DAY_DISCLOSURE": 0.04,
    "MAJOR_BREACH_DETECTED": 0.03,
    "SEVERE_WEATHER_ALERT": 0.10,
    "BANKING_CRISIS": 0.02,
    "RATE_SURPRISE": 0.08,
    "CREDIT_DOWNGRADE": 0.04,
}

DEFAULT_PRIOR = 0.05


def bayesian_update(
    prior: float,
    evidence_strength: float,
    source_reliability: float,
    false_positive_rate: float = 0.10,
) -> float:
    """Compute posterior probability via Bayes' theorem.

    P(H|E) = P(E|H) * P(H) / [P(E|H)*P(H) + P(E|~H)*(1-P(H))]

    Args:
        prior: Prior probability of the trigger being real (0-1)
        evidence_strength: How strongly this evidence supports the trigger (0-1)
        source_reliability: P(E|H) — how reliable is this source (0-1)
        false_positive_rate: P(E|~H) — base rate of false positives

    Returns:
        Posterior probability (0-1), capped at 0.99 (never 100% certain)
    """
    # Effective likelihood = source_reliability * evidence_strength
    likelihood = source_reliability * evidence_strength

    # Clamp to prevent degenerate values
    likelihood = max(0.001, min(0.999, likelihood))
    false_positive_rate = max(0.001, min(0.999, false_positive_rate))
    prior = max(0.001, min(0.999, prior))

    # Bayes update
    numerator = likelihood * prior
    denominator = numerator + false_positive_rate * (1.0 - prior)

    if denominator < 1e-12:
        return prior

    posterior = numerator / denominator
    return min(posterior, 0.99)  # Never 100% certain


def sequential_bayesian_update(
    trigger_type: str,
    evidence_list: list[dict],
) -> float:
    """Apply sequential Bayesian updates from a list of evidence items.

    Each evidence item: {"source_tier": "T1", "strength": 0.8, "corroborates": True}

    Corroborating evidence from independent sources → strong update.
    Contradicting evidence → reverse update (reduces confidence).

    Returns final posterior probability.
    """
    prior = TRIGGER_BASE_PRIORS.get(trigger_type, DEFAULT_PRIOR)
    current = prior

    seen_sources: set[str] = set()

    for ev in evidence_list:
        tier = ev.get("source_tier", "T2")
        strength = float(ev.get("strength", 0.5))
        corroborates = bool(ev.get("corroborates", True))
        source_id = ev.get("source_id", "unknown")

        # Independence discount: same source twice = diminishing returns
        independence_factor = 0.5 if source_id in seen_sources else 1.0
        seen_sources.add(source_id)

        reliability = SOURCE_RELIABILITY.get(tier, 0.50) * independence_factor
        fp_rate = FALSE_POSITIVE_RATE.get(tier, 0.15)

        if corroborates:
            current = bayesian_update(current, strength, reliability, fp_rate)
        else:
            # Contradicting evidence: reduces probability
            # Use complement: P(H|not_E) via similar formula
            not_likelihood = 1.0 - reliability * strength
            not_fp = 1.0 - fp_rate * 0.5
            current = bayesian_update(current, not_likelihood, not_fp, fp_rate)
            # But cap the reduction at 0.5x to prevent single contradiction destroying all evidence
            current = max(current, prior * 0.3)

    logger.debug(
        "[BayesianConfidence] %s: prior=%.3f → posterior=%.3f (n_evidence=%d)",
        trigger_type,
        prior,
        current,
        len(evidence_list),
    )
    return current


def compute_cluster_confidence(
    trigger_type: str,
    source_tiers: list[str],
    n_independent_sources: int,
    keyword_match_strength: float = 0.7,
) -> float:
    """Compute cluster confidence using Bayesian updates from source tiers.

    This replaces the legacy count-based scoring (0/1/2/3) with a
    continuous probability estimate.

    Args:
        trigger_type: Type of trigger being scored
        source_tiers: List of source tier strings for each supporting event
        n_independent_sources: Number of independent sources (deduped)
        keyword_match_strength: How well keywords matched (0-1)

    Returns:
        Confidence score (0-1)
    """
    if not source_tiers:
        return 0.0

    _prior = TRIGGER_BASE_PRIORS.get(trigger_type, DEFAULT_PRIOR)

    # Build evidence list
    evidence_list = []
    for i, tier in enumerate(source_tiers):
        source_id = f"src_{i}"  # Use index as proxy for independence check
        evidence_list.append(
            {
                "source_tier": tier,
                "strength": keyword_match_strength,
                "corroborates": True,
                "source_id": source_id,
            }
        )

    # Apply independence bonus: more distinct sources = stronger signal
    independence_boost = min(math.log1p(n_independent_sources) / math.log(10), 0.3)
    posterior = sequential_bayesian_update(trigger_type, evidence_list)
    return min(posterior + independence_boost, 0.99)


def detect_conflicting_evidence(
    evidence_list: list[dict],
    threshold: float = 0.3,
) -> bool:
    """Detect if evidence list contains significant contradictions.

    Returns True if more than `threshold` fraction of evidence is contradicting.
    """
    if not evidence_list:
        return False
    contradicting = sum(1 for e in evidence_list if not e.get("corroborates", True))
    return contradicting / len(evidence_list) > threshold
