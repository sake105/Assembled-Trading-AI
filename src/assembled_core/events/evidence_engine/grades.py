"""Evidence grade definitions for the Evidence Engine."""

from __future__ import annotations
from enum import Enum


class EvidenceGrade(str, Enum):
    """Evidence quality grade for a signal cluster.

    A = Strong: >=2 Tier-A sources OR >=3 independent Tier-B domains, low misinfo risk
    B = Moderate: >=1 Tier-A source OR >=2 independent Tier-B domains, moderate misinfo risk ok
    C = Weak: >=1 source total but below B threshold, higher misinfo risk tolerated
    D = Insufficient: no qualifying evidence or misinfo risk too high
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"

    def allows_active(self) -> bool:
        """Grade A or B allows WATCH->ACTIVE transition."""
        return self in (EvidenceGrade.A, EvidenceGrade.B)

    def allows_watch(self) -> bool:
        """Grade B or above allows WATCH state."""
        return self in (EvidenceGrade.A, EvidenceGrade.B, EvidenceGrade.C)
