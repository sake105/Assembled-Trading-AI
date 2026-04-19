"""Evidence grade artifact writer (T7.9).

Writes A/B/C/D evidence grades into a per-run JSON artifact alongside
other run artifacts. Complements the accounting evidence_index without
modifying the accounting layer.

Usage:
    writer = EvidenceGradeWriter("output/intel/evidence")
    writer.write(run_id, grade="B", sources=["GDELT", "EDGAR"], geo_score=2)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_GRADE_DESCRIPTIONS = {
    "A": "Strong: >=2 Tier-A OR >=3 independent Tier-B, low misinfo risk",
    "B": "Moderate: >=1 Tier-A OR >=2 independent Tier-B, acceptable misinfo",
    "C": "Weak: >=1 source but below B threshold",
    "D": "Insufficient: no qualifying evidence or misinfo too high",
}


class EvidenceGradeWriter:
    """Writes per-run evidence grade artifacts for audit trail (T7.9)."""

    def __init__(self, output_dir: str | Path) -> None:
        self._dir = Path(output_dir)

    def write(
        self,
        run_id: str,
        grade: str,
        *,
        sources: list[str] | None = None,
        geo_score: int | None = None,
        crisis_mode: str | None = None,
        misinfo_score: float | None = None,
        extra: dict | None = None,
    ) -> Path:
        """Write evidence grade artifact for a run.

        Returns path of written artifact.
        """
        if grade not in _GRADE_DESCRIPTIONS:
            logger.warning(
                "[WARN] EvidenceGradeWriter: unknown grade %r for run_id=%s", grade, run_id
            )

        self._dir.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": "evidence_grade.v1",
            "run_id": run_id,
            "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
            "evidence_grade": grade,
            "grade_description": _GRADE_DESCRIPTIONS.get(grade, "unknown"),
            "sources": sources or [],
            "geo_score": geo_score,
            "crisis_mode": crisis_mode,
            "misinfo_score": misinfo_score,
        }
        if extra:
            artifact["extra"] = extra

        path = self._dir / f"evidence_grade_{run_id}.json"
        path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        logger.info(
            "[OK] EvidenceGradeWriter: grade=%s written for run_id=%s → %s",
            grade, run_id, path,
        )
        return path

    def load(self, run_id: str) -> dict | None:
        """Load evidence grade artifact for a run_id."""
        path = self._dir / f"evidence_grade_{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[WARN] EvidenceGradeWriter: load failed for %s: %s", path, exc)
            return None

    def list_run_ids(self) -> list[str]:
        """List all run_ids with evidence grade artifacts."""
        if not self._dir.exists():
            return []
        return sorted(
            p.stem.removeprefix("evidence_grade_")
            for p in self._dir.glob("evidence_grade_*.json")
        )
