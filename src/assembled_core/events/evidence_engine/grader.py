"""Evidence grader: derives EvidenceGrade from cluster evidence summary."""

from __future__ import annotations
from typing import Any
from .grades import EvidenceGrade


def grade_evidence(
    evidence_summary: dict[str, Any],
    misinfo_risk_score: float = 0.0,
    misinfo_risk_threshold: float = 0.70,
) -> EvidenceGrade:
    """Assign an EvidenceGrade to a cluster based on source counts and misinfo risk.

    Args:
        evidence_summary: Output from summarize_cluster_evidence() -- must have
            tierA_count, tierB_independent_count, evidence_ok keys.
        misinfo_risk_score: 0.0-1.0 misinfo risk (from compute_misinfo_risk).
        misinfo_risk_threshold: Misinfo score above this blocks grade A.

    Returns:
        EvidenceGrade (A/B/C/D).
    """
    tier_a = int(evidence_summary.get("tierA_count", 0))
    tier_b_ind = int(evidence_summary.get("tierB_independent_count", 0))
    evidence_ok = bool(evidence_summary.get("evidence_ok", False))

    high_misinfo = misinfo_risk_score >= misinfo_risk_threshold

    # Grade A: strong multi-source, low misinfo
    if not high_misinfo and (tier_a >= 2 or tier_b_ind >= 3):
        return EvidenceGrade.A

    # Grade B: adequate evidence, acceptable misinfo risk
    if evidence_ok and not (misinfo_risk_score >= 0.90):
        return EvidenceGrade.B

    # Grade C: weak but some evidence
    if tier_a >= 1 or tier_b_ind >= 1:
        return EvidenceGrade.C

    # Grade D: insufficient
    return EvidenceGrade.D
