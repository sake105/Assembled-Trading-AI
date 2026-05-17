"""Reproducibility certificate generator.

From 43_BACKTEST_REPRODUCIBILITY_CERTIFICATE.md.

Usage:
    cert = generate_certificate(
        config_path="configs/strategy_v4.yaml",
        data_paths=["data/prices/sp500_2023.parquet"],
        output_dir="output/backtest_20260424",
    )
    save_certificate(cert, "output/backtest_20260424/certificate.json")
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.assembled_core.certify.schema import (
    EnvironmentFingerprint,
    InputFingerprint,
    OutputFingerprint,
    ReproducibilityCertificate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------


def file_sha256(path: Path | str) -> str:
    """Return SHA-256 hex digest of a file.  Returns 'NOT_FOUND' if missing."""
    p = Path(path)
    if not p.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def object_sha256(obj) -> str:
    """Return SHA-256 of the JSON-serialisable representation of *obj*."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Environment fingerprint
# ---------------------------------------------------------------------------


def get_git_info() -> tuple[str, bool]:
    """Return (commit_sha, is_dirty).  Returns ('unknown', False) if not a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        return sha, bool(dirty_out.strip())
    except Exception:
        return "unknown", False


def get_environment_fingerprint(
    extra_seeds: dict[str, int] | None = None,
) -> EnvironmentFingerprint:
    """Capture the current Python environment."""
    git_sha, git_dirty = get_git_info()

    pkg_versions: dict[str, str] = {}
    try:
        import importlib.metadata as im

        for dist in im.distributions():
            pkg_versions[dist.metadata["Name"]] = dist.metadata["Version"]
    except Exception:
        pass

    seeds = {
        "python_random": random.randint(0, 2**31),
    }
    try:
        import numpy as np

        seeds["numpy_global"] = int(np.random.get_state()[1][0])
    except Exception:
        pass
    if extra_seeds:
        seeds.update(extra_seeds)

    return EnvironmentFingerprint(
        python_version=sys.version,
        platform=platform.platform(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        package_hashes=pkg_versions,
        random_seeds=seeds,
    )


# ---------------------------------------------------------------------------
# Input fingerprint
# ---------------------------------------------------------------------------


def build_input_fingerprint(
    data_paths: list[Path | str],
    config_path: Path | str | None = None,
    model_paths: dict[str, Path | str] | None = None,
) -> InputFingerprint:
    data_hashes = {str(p): file_sha256(p) for p in data_paths}
    config_hash = file_sha256(config_path) if config_path else ""
    model_hashes = {name: file_sha256(p) for name, p in (model_paths or {}).items()}
    return InputFingerprint(
        data_file_hashes=data_hashes,
        config_hash=config_hash,
        config_path=str(config_path) if config_path else "",
        model_hashes=model_hashes,
    )


# ---------------------------------------------------------------------------
# Output fingerprint
# ---------------------------------------------------------------------------


def build_output_fingerprint(output_dir: Path | str) -> OutputFingerprint:
    """Hash output artefacts in *output_dir*."""
    d = Path(output_dir)
    summary: dict[str, float] = {}

    # Try to load summary metrics from a JSON if present
    summary_file = d / "summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            pass

    return OutputFingerprint(
        equity_curve_hash=file_sha256(d / "equity_curve.parquet"),
        trades_hash=file_sha256(d / "trades.parquet"),
        signals_hash=file_sha256(d / "signals.parquet"),
        summary_metrics={
            k: float(v) for k, v in summary.items() if isinstance(v, (int, float))
        },
    )


# ---------------------------------------------------------------------------
# Certificate generation
# ---------------------------------------------------------------------------


def generate_certificate(
    data_paths: list[Path | str] | None = None,
    config_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    model_paths: dict[str, Path | str] | None = None,
    notes: str = "",
) -> ReproducibilityCertificate:
    """Generate a complete ReproducibilityCertificate for the current run.

    Args:
        data_paths: Input data files to hash.
        config_path: Strategy config YAML to hash.
        output_dir: Directory containing backtest output artefacts.
        model_paths: Named model files to hash.
        notes: Free-text description of the run.

    Returns:
        Populated ReproducibilityCertificate.
    """
    env = get_environment_fingerprint()
    inp = build_input_fingerprint(data_paths or [], config_path, model_paths)
    out = build_output_fingerprint(output_dir) if output_dir else OutputFingerprint()

    return ReproducibilityCertificate(
        certificate_id=str(uuid.uuid4()),
        created_at=datetime.now(tz=timezone.utc),
        environment=env,
        inputs=inp,
        outputs=out,
        notes=notes,
    )


def save_certificate(cert: ReproducibilityCertificate, path: Path | str) -> None:
    """Write certificate JSON to *path*."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(cert.to_json(indent=2))
    logger.info("Certificate saved → %s", p)


def verify_certificate(
    cert_path: Path | str,
    output_dir: Path | str,
) -> dict[str, bool]:
    """Compare a saved certificate against current output files.

    Args:
        cert_path: Path to previously saved certificate.json.
        output_dir: Directory with the new run's output artefacts.

    Returns:
        Dict mapping artefact name → hash_match (bool).
    """
    with open(cert_path, encoding="utf-8") as f:
        cert = ReproducibilityCertificate.from_dict(json.load(f))

    new_out = build_output_fingerprint(output_dir)
    results = {
        "equity_curve": cert.outputs.equity_curve_hash == new_out.equity_curve_hash,
        "trades": cert.outputs.trades_hash == new_out.trades_hash,
        "signals": cert.outputs.signals_hash == new_out.signals_hash,
    }
    all_match = all(results.values())
    logger.info(
        "Certificate verification: %s — %s", "PASS" if all_match else "FAIL", results
    )
    return results


__all__ = [
    "file_sha256",
    "object_sha256",
    "get_git_info",
    "get_environment_fingerprint",
    "build_input_fingerprint",
    "build_output_fingerprint",
    "generate_certificate",
    "save_certificate",
    "verify_certificate",
]
