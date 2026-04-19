"""Run manifest for a single paper-engine day.

Each run produces a small JSON manifest that enumerates every artifact the
run emitted and carries enough run-level metadata to reproduce or compare
two runs. The manifest is written twice:

- ``output/manifests/<run_id>/manifest_<date>.json`` — canonical per-day file
- ``output/manifests/<run_id>/manifest.latest.json`` — pointer to the last run

The schema is intentionally minimal and stable so cross-run tooling can rely
on it. Fields are forward-compatible: unknown keys should be ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunManifest:
    """Per-day run manifest.

    Attributes:
        run_id: Logical run identifier (caller-chosen).
        date: Trading date (ISO ``YYYY-MM-DD``).
        started_at_utc: ISO-8601 UTC timestamp when the run started.
        finished_at_utc: ISO-8601 UTC timestamp when the run finished.
        status: ``"success"`` / ``"error"`` / ``"kill_switch"``.
        git_sha: 40-char git SHA if derivable, else empty.
        config_hash: 16-char sha256 prefix of the run config.
        phase_versions: Engine phase-version tags for forward/back compat.
        artifacts: ``{artifact_name: path_string}`` map. Only existing paths.
        metrics: Lightweight key metrics surface for the cross-run index.
    """

    run_id: str
    date: str
    started_at_utc: str
    finished_at_utc: str
    status: str
    git_sha: str = ""
    config_hash: str = ""
    phase_versions: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    # Version tag so downstream readers can detect schema drift instead of
    # silently parsing an unknown-version manifest. Bump on incompatible
    # field changes (add-only changes stay at v1 by design — unknown keys
    # are meant to be ignored).
    schema_version: str = "run.manifest.v1"


def _compute_git_sha() -> str:
    """Return the short git SHA of HEAD, empty string if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover
        pass
    return ""


def compute_config_hash(config: Any) -> str:
    """SHA-256 prefix of a deterministic repr of the config.

    Paths and non-JSON-serialisable values are coerced to strings.
    """
    try:
        if hasattr(config, "__dict__"):
            raw: dict[str, Any] = {}
            for k, v in vars(config).items():
                try:
                    json.dumps(v, default=str)
                    raw[k] = v
                except Exception:
                    raw[k] = str(v)
        else:
            raw = {"repr": repr(config)}
        blob = json.dumps(raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]
    except Exception:
        return ""


def _resolve_artifacts(candidate_paths: dict[str, Path]) -> dict[str, str]:
    """Keep only paths that exist; normalise to POSIX strings for stability."""
    resolved: dict[str, str] = {}
    for name, path in candidate_paths.items():
        if path is None:
            continue
        try:
            p = Path(path)
            if p.exists():
                resolved[name] = p.as_posix()
        except Exception:  # pragma: no cover
            pass
    return resolved


def write_run_manifest(
    *,
    run_id: str,
    date: str,
    started_at_utc: str,
    finished_at_utc: str | None = None,
    status: str = "success",
    config: Any = None,
    artifacts: dict[str, Path] | None = None,
    metrics: dict[str, Any] | None = None,
    phase_versions: dict[str, str] | None = None,
    manifests_dir: Path = Path("output/manifests"),
) -> Path:
    """Write the per-day manifest file and the ``manifest.latest.json`` pointer.

    Returns the path to the per-day manifest.
    """
    finished = finished_at_utc or datetime.now(timezone.utc).isoformat()
    manifest = RunManifest(
        run_id=run_id,
        date=date,
        started_at_utc=started_at_utc,
        finished_at_utc=finished,
        status=status,
        git_sha=_compute_git_sha(),
        config_hash=compute_config_hash(config) if config is not None else "",
        phase_versions=phase_versions or {},
        artifacts=_resolve_artifacts(artifacts or {}),
        metrics=metrics or {},
    )

    out_dir = Path(manifests_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    per_day = out_dir / f"manifest_{date}.json"
    latest = out_dir / "manifest.latest.json"

    payload = asdict(manifest)
    blob = json.dumps(payload, indent=2, default=str)
    # Atomic write via tmp + os.replace — a crash mid-write previously left a
    # truncated JSON that downstream tooling parsed as "the latest run's
    # manifest," masking the failed run as in-progress. `manifest.latest.json`
    # is the pointer file that cross-run tooling reads, so partial writes here
    # poison every subsequent comparison until the next successful run.
    per_day_tmp = per_day.with_suffix(per_day.suffix + ".tmp")
    per_day_tmp.write_text(blob, encoding="utf-8")
    os.replace(per_day_tmp, per_day)
    # Symlinks are unreliable on Windows — always write a regular file copy.
    latest_tmp = latest.with_suffix(latest.suffix + ".tmp")
    latest_tmp.write_text(blob, encoding="utf-8")
    os.replace(latest_tmp, latest)
    logger.info("[MANIFEST] Wrote %s", per_day)
    return per_day


__all__ = [
    "RunManifest",
    "compute_config_hash",
    "write_run_manifest",
]
