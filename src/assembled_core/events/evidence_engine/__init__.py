"""Evidence Engine — M8: Fake-News Defense and Evidence Grading."""

from .grades import EvidenceGrade
from .grader import grade_evidence
from .misinfo_risk import compute_misinfo_risk
from .action_gate import check_evidence_grade_gate

__all__ = [
    "EvidenceGrade",
    "grade_evidence",
    "compute_misinfo_risk",
    "check_evidence_grade_gate",
]
