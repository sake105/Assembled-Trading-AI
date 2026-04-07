"""Run Metadata and Reproducibility (Plan 11.4).

Captures config hash, git commit, Python version, and dependency versions per run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def collect_run_metadata(
    config: dict | None = None,
    output_dir: str = "output",
) -> dict:
    """Collect metadata for reproducibility.

    Args:
        config: Pipeline configuration dict (will be hashed).
        output_dir: Output directory for metadata file.

    Returns:
        Metadata dict.
    """
    meta: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        meta["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        meta["git_commit"] = "unknown"

    # Git dirty flag
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        )
        meta["git_dirty"] = bool(result.stdout.strip()) if result.returncode == 0 else True
    except Exception:
        meta["git_dirty"] = True

    # Config hash
    if config:
        config_str = json.dumps(config, sort_keys=True, default=str)
        meta["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    else:
        meta["config_hash"] = "none"

    # Key dependency versions
    dep_versions: dict[str, str] = {}
    for pkg in ["pandas", "numpy", "scipy", "sklearn", "lightgbm", "xgboost"]:
        try:
            mod = __import__(pkg)
            dep_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            dep_versions[pkg] = "not_installed"
    meta["dependency_versions"] = dep_versions

    return meta


def save_run_metadata(
    metadata: dict,
    output_dir: str = "output",
) -> str:
    """Save run metadata to JSON file.

    Args:
        metadata: Metadata dict from collect_run_metadata().
        output_dir: Output directory.

    Returns:
        Path to saved file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = out_path / f"run_metadata_{ts}.json"
    filepath.write_text(json.dumps(metadata, indent=2, default=str))

    return str(filepath)


__all__ = ["collect_run_metadata", "save_run_metadata"]
