"""Canonical atomic JSON write helper.

Replaces ad-hoc json.dump/write_text calls that can leave truncated files on
crash. The tmp+replace idiom guarantees readers always see a complete file.

Usage:
    from src.assembled_core.utils.atomic_io import atomic_write_json

    atomic_write_json(path, {"key": "value"})
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def atomic_write_json(
    path: Path | str,
    data: Any,
    *,
    retries: int = 5,
    backoff_ms: int = 50,
    indent: int = 2,
    sort_keys: bool = False,
    default: Any = str,
) -> None:
    """Write *data* as JSON to *path* atomically (tmp + os.replace).

    Retries on PermissionError/OSError with exponential backoff (Windows
    file-lock friendly). If all retries fail, re-raises the last exception.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=indent, sort_keys=sort_keys, default=default)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass  # network/tmpfs may not support fsync
            os.replace(str(tmp_path), str(path))
            return
        except (PermissionError, OSError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(backoff_ms * (2**attempt) / 1000.0)
    # All retries exhausted — clean up orphaned temp file before raising
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    if last_err is not None:
        raise last_err


# Alias for callers importing the old risk.state_machine name
atomic_write_json_with_retry = atomic_write_json

__all__ = ["atomic_write_json", "atomic_write_json_with_retry"]
