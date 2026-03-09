"""Atomic JSON emit for disclosures artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _atomic_write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    tmp_dir = path.parent / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{path.name}.{os.getpid()}.tmp"
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp_path, path)


def emit_json_artifact(obj: Any, path: str | Path) -> None:
    """Write JSON atomically (tmp + replace)."""
    _atomic_write_json(obj, Path(path))
