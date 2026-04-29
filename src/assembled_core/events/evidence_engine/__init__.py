"""Evidence Engine — M8: Fake-News Defense and Evidence Grading."""

from .action_gate import check_evidence_grade_gate
from .grader import grade_evidence
from .grades import EvidenceGrade
from .misinfo_risk import compute_misinfo_risk

__all__ = [
    "EvidenceGrade",
    "grade_evidence",
    "compute_misinfo_risk",
    "check_evidence_grade_gate",
]
