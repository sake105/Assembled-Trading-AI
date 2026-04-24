"""Versioned Model Registry mit Metadaten-Tracking.

Problem: Mit 35+ ML-Modulen und Shadow-Deploys gibt es schnell viele Modelle
in `models/`. Ohne Registry weiß niemand, welche Version wann trainiert wurde,
auf welchen Daten, mit welchen Metriken, und ob sie deployed ist.

Diese Registry:
- Speichert Modelle mit Metadata (trained_at, n_samples, IC, features, train_window)
- Versioniert automatisch (v1, v2, ... + latest-Symlink)
- Rollback-Fähigkeit
- Human-Review-Workflow: candidate → approved → deployed (auto_deploy=False!)

Einhaltung von CLAUDE.md/Rule 30:
- auto_deploy bleibt False. promote_to_deployed() erfordert explizite Anwendung.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    """Metadata für ein registriertes Modell."""

    model_id: str
    """Eindeutige ID (z.B. 'meta_labeler_v3')."""
    version: int
    model_type: str
    """Class-Name des Modells (GradientBoostingClassifier, etc.)."""
    trained_at: str
    """ISO-UTC-Timestamp."""
    file_path: str
    """Relativer Pfad zum joblib-File."""
    metrics: dict = field(default_factory=dict)
    """Training-Metriken: {'ic': 0.12, 'auc': 0.68, 'n_samples': 5000, ...}"""
    features: list[str] = field(default_factory=list)
    train_start: str | None = None
    train_end: str | None = None
    status: str = "candidate"
    """'candidate' | 'approved' | 'deployed' | 'archived'"""
    deployed_by: str | None = None
    notes: str = ""


class ModelRegistry:
    """Registry für versionierte ML-Modelle mit JSON-Index.

    Verzeichnisstruktur:
        models/
          {model_id}/
            v1.joblib
            v2.joblib
            ...
            deployed.joblib → Symlink oder Kopie der aktuell deployten Version
        models/_registry.json   ← Metadaten-Index
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("models")
        self.registry_path = self.base_dir / "_registry.json"
        self._records: dict[str, list[ModelRecord]] = self._load()

    def _load(self) -> dict[str, list[ModelRecord]]:
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            out: dict[str, list[ModelRecord]] = {}
            for mid, recs in data.get("models", {}).items():
                out[mid] = [ModelRecord(**r) for r in recs]
            return out
        except Exception as exc:
            logger.warning("[Registry] Konnte registry nicht laden: %s", exc)
            return {}

    def _save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "models": {
                mid: [
                    {
                        "model_id": r.model_id,
                        "version": r.version,
                        "model_type": r.model_type,
                        "trained_at": r.trained_at,
                        "file_path": r.file_path,
                        "metrics": r.metrics,
                        "features": r.features,
                        "train_start": r.train_start,
                        "train_end": r.train_end,
                        "status": r.status,
                        "deployed_by": r.deployed_by,
                        "notes": r.notes,
                    }
                    for r in recs
                ]
                for mid, recs in self._records.items()
            },
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        self.registry_path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def register(
        self,
        model: object,
        model_id: str,
        metrics: dict | None = None,
        features: list[str] | None = None,
        train_start: str | None = None,
        train_end: str | None = None,
        notes: str = "",
    ) -> ModelRecord:
        """Registriert ein neues Modell als Candidate.

        Args:
            model: Serialisierbares Modell-Objekt (joblib-kompatibel)
            model_id: Logische ID (z.B. 'meta_labeler')
            metrics: Training-Metriken
            features: Feature-Liste für Inferenz-Kompatibilität
            train_start, train_end: PIT-Fenster
            notes: Beschreibung

        Returns:
            ModelRecord (Status='candidate', nie auto-deployed)
        """
        existing = self._records.get(model_id, [])
        version = max((r.version for r in existing), default=0) + 1

        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        file_path = model_dir / f"v{version}.joblib"
        joblib.dump(model, file_path)

        record = ModelRecord(
            model_id=model_id,
            version=version,
            model_type=type(model).__name__,
            trained_at=datetime.now(timezone.utc).isoformat(),
            file_path=str(file_path.relative_to(self.base_dir)),
            metrics=metrics or {},
            features=features or [],
            train_start=train_start,
            train_end=train_end,
            status="candidate",
            notes=notes,
        )
        existing.append(record)
        self._records[model_id] = existing
        self._save()

        logger.info(
            "[Registry] %s v%d registriert (status=candidate, %d metrics)",
            model_id, version, len(record.metrics),
        )
        return record

    def approve(self, model_id: str, version: int, approver: str = "human") -> ModelRecord:
        """Setzt Status auf 'approved' (nach menschlicher Review)."""
        record = self._get_record(model_id, version)
        record.status = "approved"
        record.deployed_by = approver
        self._save()
        logger.info("[Registry] %s v%d APPROVED by %s", model_id, version, approver)
        return record

    def promote_to_deployed(self, model_id: str, version: int) -> ModelRecord:
        """Macht aus einem 'approved' Modell das 'deployed' Modell.

        Muss manuell aufgerufen werden — auto_deploy=False in CLAUDE.md Rule 30.
        Kopiert v{N}.joblib → deployed.joblib (keine Symlink-Probleme auf Windows).
        """
        record = self._get_record(model_id, version)
        if record.status != "approved":
            raise ValueError(
                f"[Registry] v{version} ist nicht approved (status={record.status}). "
                "Bitte zuerst .approve() aufrufen."
            )

        # Andere Versionen auf 'archived' setzen
        for r in self._records[model_id]:
            if r.status == "deployed":
                r.status = "archived"

        record.status = "deployed"

        src = self.base_dir / record.file_path
        dst = self.base_dir / model_id / "deployed.joblib"
        shutil.copy2(src, dst)

        self._save()
        logger.info("[Registry] %s v%d → DEPLOYED", model_id, version)
        return record

    def rollback(self, model_id: str, target_version: int) -> ModelRecord:
        """Rollt zum angegebenen archivierten Version zurück."""
        record = self._get_record(model_id, target_version)
        for r in self._records[model_id]:
            if r.status == "deployed":
                r.status = "archived"
        record.status = "deployed"

        src = self.base_dir / record.file_path
        dst = self.base_dir / model_id / "deployed.joblib"
        shutil.copy2(src, dst)

        self._save()
        logger.warning("[Registry] %s ROLLBACK to v%d", model_id, target_version)
        return record

    def load_deployed(self, model_id: str) -> object:
        """Lädt das aktuell deployte Modell."""
        dst = self.base_dir / model_id / "deployed.joblib"
        if not dst.exists():
            raise FileNotFoundError(
                f"Kein deployed-Modell für {model_id} — nutze promote_to_deployed()"
            )
        return joblib.load(dst)

    def list_versions(self, model_id: str) -> list[ModelRecord]:
        return list(self._records.get(model_id, []))

    def compare_metrics(self, model_id: str, metric_key: str = "ic") -> list[tuple[int, float, str]]:
        """Gibt (version, metric, status) Liste zurück, sortiert nach Version."""
        recs = sorted(self._records.get(model_id, []), key=lambda r: r.version)
        return [(r.version, float(r.metrics.get(metric_key, 0.0)), r.status) for r in recs]

    def _get_record(self, model_id: str, version: int) -> ModelRecord:
        for r in self._records.get(model_id, []):
            if r.version == version:
                return r
        raise KeyError(f"{model_id} v{version} not found")


__all__ = [
    "ModelRecord",
    "ModelRegistry",
]
