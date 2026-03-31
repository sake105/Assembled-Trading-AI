"""Action gate: controls which Crisis Alpha actions are permitted based on evidence grade."""

from __future__ import annotations
from .grades import EvidenceGrade


def check_evidence_grade_gate(
    grade: EvidenceGrade,
    require_for_active: str = "B",
) -> tuple[bool, str]:
    """Gate check: is evidence grade sufficient for WATCH->ACTIVE transition?

    Args:
        grade: EvidenceGrade for the current cluster.
        require_for_active: Minimum grade required for ACTIVE (default "B").
            Must be one of: "A", "B", "C", "D".

    Returns:
        (ok, reason) -- same pattern as other gate functions in gates.py.
    """
    grade_order = {
        EvidenceGrade.A: 0,
        EvidenceGrade.B: 1,
        EvidenceGrade.C: 2,
        EvidenceGrade.D: 3,
    }
    required_grade = EvidenceGrade(require_for_active)

    if grade_order[grade] <= grade_order[required_grade]:
        return (
            True,
            f"evidence grade gate: OK (grade={grade.value}, required<={required_grade.value})",
        )

    return (
        False,
        f"evidence grade gate: BLOCKED (grade={grade.value} is below required {required_grade.value})",
    )
