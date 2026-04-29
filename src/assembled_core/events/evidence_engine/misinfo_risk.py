"""Misinfo risk scorer for news cluster evidence.

Detects patterns associated with misinformation:
- Social-only sourcing (no Tier-A, no established Tier-B domains)
- Single-source spike (all events from one domain)
- Anomalous burst (unnaturally rapid clustering)
- No cross-domain corroboration
"""

from __future__ import annotations

from typing import Any


def compute_misinfo_risk(
    evidence_summary: dict[str, Any],
    *,
    social_only: bool = False,
    event_count: int = 0,
    burst_window_minutes: float | None = None,
) -> float:
    """Compute a misinfo risk score (0.0 = no risk, 1.0 = maximum risk).

    Args:
        evidence_summary: Output from summarize_cluster_evidence().
        social_only: Whether all evidence comes from social media.
        event_count: Total number of events in cluster.
        burst_window_minutes: If set, time in minutes over which events arrived.
            Very short windows (< 5 min) with many events suggest synthetic amplification.

    Returns:
        float in [0.0, 1.0]. Higher = more likely misinformation.
    """
    score = 0.0
    tier_a = int(evidence_summary.get("tierA_count", 0))
    tier_b_ind = int(evidence_summary.get("tierB_independent_count", 0))
    tier_b_total = int(evidence_summary.get("tierB_count", 0))

    # Social-only: highest risk factor (+0.60)
    if social_only:
        score += 0.60

    # No Tier-A sources: elevated risk (+0.20)
    if tier_a == 0:
        score += 0.20

    # Single-source dominance: all Tier-B from one domain (+0.15)
    if tier_b_total >= 2 and tier_b_ind <= 1 and tier_a == 0:
        score += 0.15

    # No cross-domain corroboration (no Tier-A, only 1 Tier-B domain) (+0.10)
    if tier_a == 0 and tier_b_ind <= 1:
        score += 0.10

    # Anomalous burst: many events in very short window (+0.20)
    if (
        burst_window_minutes is not None
        and event_count >= 5
        and burst_window_minutes < 5.0
    ):
        score += 0.20

    return min(1.0, score)
