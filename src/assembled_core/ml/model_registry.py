"""Model registry — hash verification and safe model loading (Backlog Item 74).

Registry file: models/registry.json (tracked in git, model binaries are not).
Use verify_model_hash() to check integrity before loading, or safe_load_model()
as a drop-in replacement for joblib.load() that includes the hash check.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "models" / "registry.json"


def _hash_file(path: Path) -> str:
    """Compute SHA256 of *path* using 64 KB streaming to avoid OOM on large models."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_model_file_hash(path: Path, expected: str) -> bool:
    """Return True if the SHA256 of *path* matches *expected*."""
    return _hash_file(path) == expected


_registry_cache: dict | None = None


def _load_registry() -> dict:
    global _registry_cache  # noqa: PLW0603
    if _registry_cache is not None:
        return _registry_cache
    if not _REGISTRY_PATH.exists():
        logger.debug(
            "[MODEL-REGISTRY] registry.json not found at %s — hash checks disabled",
            _REGISTRY_PATH,
        )
        _registry_cache = {}
        return _registry_cache
    try:
        _registry_cache = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8")).get(
            "models", {}
        )
    except Exception as exc:
        logger.warning("[MODEL-REGISTRY] Failed to load registry.json: %s", exc)
        _registry_cache = {}
    return _registry_cache


def verify_model_hash(model_path: str | Path, *, strict: bool = False) -> bool:
    """Verify a model file's SHA256 hash against the registry.

    Args:
        model_path: Path to the model file.
        strict: If True, raise RuntimeError on mismatch. If False (default),
            log a warning and return False.

    Returns:
        True if hash matches (or if file not in registry). False on mismatch.
    """
    path = Path(model_path)
    registry = _load_registry()

    if not registry:
        logger.warning(
            "[MODEL-REGISTRY] Registry is empty or missing — hash checks disabled for %s",
            path.name,
        )
        return True

    entry = registry.get(path.name)
    if entry is None:
        logger.debug(
            "[MODEL-REGISTRY] %s not in registry — skipping hash check", path.name
        )
        return True

    if not path.exists():
        msg = f"[MODEL-REGISTRY] Model file not found: {path}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg)
        return False

    expected = entry.get("sha256", "")
    if not expected:
        return True

    matches = _verify_model_file_hash(path, expected)
    if not matches:
        msg = (
            f"[MODEL-REGISTRY] Hash mismatch for {path.name}: "
            f"expected {expected[:16]}... — model may have been replaced or corrupted"
        )
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)
        return False

    logger.debug("[MODEL-REGISTRY] Hash OK: %s (%s...)", path.name, expected[:16])
    return True


def safe_load_model(model_path: str | Path, *, strict: bool = False) -> object:
    """Load a joblib model after verifying its hash against the registry.

    Args:
        model_path: Path to the .joblib model file.
        strict: If True, raise on hash mismatch. If False, warn and load anyway.

    Returns:
        Loaded model object.
    """
    import joblib

    path = Path(model_path)
    verify_model_hash(path, strict=strict)
    model = joblib.load(path)
    logger.info("[MODEL-REGISTRY] Loaded %s", path.name)
    return model


def register_model(model_path: str | Path) -> dict:
    """Compute and add a model's hash to the registry file.

    Call this after training a new model to keep the registry current.

    Args:
        model_path: Path to the new model file.

    Returns:
        The registry entry dict {sha256, size_bytes, path}.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    sha256 = _hash_file(path)
    entry = {"sha256": sha256, "size_bytes": path.stat().st_size, "path": str(path)}

    registry_data: dict = {"version": 1, "models": {}}
    if _REGISTRY_PATH.exists():
        try:
            registry_data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass

    registry_data.setdefault("models", {})[path.name] = entry
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(registry_data, indent=2), encoding="utf-8")

    global _registry_cache  # noqa: PLW0603
    _registry_cache = None

    logger.info("[MODEL-REGISTRY] Registered %s (sha256=%s...)", path.name, sha256[:16])
    return entry


class ModelVersion:
    """Metadata for a single registered model version."""

    def __init__(
        self,
        model_id: str,
        version: int,
        status: str,
        metrics: dict,
        path: Path,
        sha256: str,
        approver: str | None = None,
        registered_at: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.version = version
        self.status = status
        self.metrics = metrics
        self.path = path
        self.sha256 = sha256
        self.approver = approver
        # ISO-format timestamp; set to now if not provided (new registrations)
        self.registered_at: str = (
            registered_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "status": self.status,
            "metrics": self.metrics,
            "path": str(self.path),
            "sha256": self.sha256,
            "approver": self.approver,
            "registered_at": self.registered_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        return cls(
            model_id=d["model_id"],
            version=d["version"],
            status=d["status"],
            metrics=d.get("metrics", {}),
            path=Path(d["path"]),
            sha256=d.get("sha256", ""),
            approver=d.get("approver"),
            registered_at=d.get("registered_at"),
        )


class ModelRegistry:
    """File-backed model registry with versioning, approval, and deployment workflow.

    Each model_id has its own subdirectory under base_dir.
    Metadata is stored in <base_dir>/<model_id>/registry.json.
    Model binaries are stored as <base_dir>/<model_id>/v<N>.joblib.
    The active deployed model is symlinked/copied to deployed.joblib.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_id: str) -> Path:
        d = self.base_dir / model_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _meta_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "registry.json"

    def _load_meta(self, model_id: str) -> list[ModelVersion]:
        p = self._meta_path(model_id)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return [ModelVersion.from_dict(v) for v in data.get("versions", [])]
        except Exception as exc:
            logger.warning(
                "[ModelRegistry] Failed to load meta for %s: %s", model_id, exc
            )
            return []

    def _save_meta(self, model_id: str, versions: list[ModelVersion]) -> None:
        p = self._meta_path(model_id)
        p.write_text(
            json.dumps({"versions": [v.to_dict() for v in versions]}, indent=2),
            encoding="utf-8",
        )

    def register(
        self, model: object, *, model_id: str, metrics: dict | None = None
    ) -> ModelVersion:
        """Save a model as a new candidate version and return its metadata."""
        import joblib

        versions = self._load_meta(model_id)
        next_ver = max((v.version for v in versions), default=0) + 1
        model_path = self._model_dir(model_id) / f"v{next_ver}.joblib"
        joblib.dump(model, model_path)
        sha256 = _hash_file(model_path)
        mv = ModelVersion(
            model_id=model_id,
            version=next_ver,
            status="candidate",
            metrics=metrics or {},
            path=model_path,
            sha256=sha256,
        )
        versions.append(mv)
        self._save_meta(model_id, versions)
        logger.info(
            "[ModelRegistry] Registered %s v%d (sha256=%s...)",
            model_id,
            next_ver,
            sha256[:16],
        )
        return mv

    def list_versions(self, model_id: str) -> list[ModelVersion]:
        return self._load_meta(model_id)

    def _get_version(self, model_id: str, version: int) -> ModelVersion:
        versions = self._load_meta(model_id)
        for v in versions:
            if v.version == version:
                return v
        raise ValueError(f"Version {version} not found for model_id={model_id}")

    def approve(
        self, model_id: str, version: int, *, approver: str = "system"
    ) -> ModelVersion:
        versions = self._load_meta(model_id)
        for v in versions:
            if v.version == version:
                v.status = "approved"
                v.approver = approver
                self._save_meta(model_id, versions)
                logger.info(
                    "[ModelRegistry] Approved %s v%d by %s", model_id, version, approver
                )
                return v
        raise ValueError(f"Version {version} not found for model_id={model_id}")

    def promote_to_deployed(self, model_id: str, version: int) -> ModelVersion:
        versions = self._load_meta(model_id)
        target: ModelVersion | None = None
        for v in versions:
            if v.version == version:
                target = v
                break
        if target is None:
            raise ValueError(f"Version {version} not found for model_id={model_id}")
        if target.status not in ("approved", "deployed"):
            raise ValueError(
                f"Version {version} ist nicht approved (status={target.status!r}) — "
                "approve zuerst via registry.approve()"
            )
        import shutil

        src = self._model_dir(model_id) / f"v{version}.joblib"
        dst = self._model_dir(model_id) / "deployed.joblib"
        shutil.copy2(src, dst)

        for v in versions:
            if v.status == "deployed":
                v.status = "archived"
            if v.version == version:
                v.status = "deployed"
        self._save_meta(model_id, versions)
        logger.info("[ModelRegistry] Deployed %s v%d", model_id, version)
        return target

    def load_deployed(self, model_id: str) -> object:
        import joblib

        deployed = self._model_dir(model_id) / "deployed.joblib"
        if not deployed.exists():
            raise FileNotFoundError(f"No deployed model for model_id={model_id}")

        # Hash verification: find the deployed version entry and check sha256 if set
        versions = self._load_meta(model_id)
        deployed_version: ModelVersion | None = next(
            (v for v in versions if v.status == "deployed"), None
        )
        if deployed_version is not None and deployed_version.sha256:
            if not _verify_model_file_hash(deployed, deployed_version.sha256):
                raise ValueError("Model hash mismatch — possible tampering")

        return joblib.load(deployed)

    def check_model_age(self, model_id: str) -> dict:
        """Check if the deployed model is older than the policy limit.

        Returns dict with keys: model_id, deployed_version, age_days, stale, registered_at.
        stale=True if age_days > ml.model_max_age_days from configs/policy.yaml (default 14).
        If no deployed version found, deployed_version and age_days are None, stale=False.
        """
        versions = self._load_meta(model_id)
        deployed: ModelVersion | None = next(
            (v for v in versions if v.status == "deployed"), None
        )
        if deployed is None:
            return {
                "model_id": model_id,
                "deployed_version": None,
                "age_days": None,
                "stale": False,
                "registered_at": None,
            }

        # Parse registered_at timestamp
        try:
            reg_dt = datetime.datetime.fromisoformat(deployed.registered_at)
            if reg_dt.tzinfo is None:
                reg_dt = reg_dt.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            age_days = (now - reg_dt).days
        except Exception as exc:
            logger.warning(
                "[ModelRegistry] Could not parse registered_at for %s: %s",
                model_id,
                exc,
            )
            age_days = None

        # Load max age from policy.yaml
        max_age = 14  # default
        try:
            policy_path = (
                Path(__file__).parent.parent.parent.parent / "configs" / "policy.yaml"
            )
            if policy_path.exists():
                import yaml  # type: ignore[import]

                policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
                max_age = int(policy.get("ml", {}).get("model_max_age_days", 14))
        except Exception as exc:
            logger.debug(
                "[ModelRegistry] Could not load policy.yaml for age check: %s", exc
            )

        stale = (age_days is not None) and (age_days > max_age)
        return {
            "model_id": model_id,
            "deployed_version": deployed.version,
            "age_days": age_days,
            "stale": stale,
            "registered_at": deployed.registered_at,
        }

    def rollback(self, model_id: str, version: int) -> ModelVersion:
        """Approve and deploy a specific version, archiving the current deployed version."""
        versions = self._load_meta(model_id)
        target: ModelVersion | None = None
        for v in versions:
            if v.version == version:
                target = v
                break
        if target is None:
            raise ValueError(f"Version {version} not found for model_id={model_id}")
        if target.status not in ("approved", "deployed"):
            target.status = "approved"
        self._save_meta(model_id, versions)
        return self.promote_to_deployed(model_id, version)


__all__ = [
    "verify_model_hash",
    "safe_load_model",
    "register_model",
    "ModelRegistry",
    "ModelVersion",
]
