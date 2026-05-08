"""Canonical atomic write helpers for JSON, Parquet, and CSV.

Replaces ad-hoc write calls that can leave truncated files on crash.
The tmp+replace idiom guarantees readers always see a complete file.

Usage:
    from src.assembled_core.utils.atomic_io import (
        atomic_write_json,
        atomic_write_parquet,
        atomic_write_csv,
    )

    atomic_write_json(path, {"key": "value"})
    atomic_write_parquet(df, path)
    atomic_write_csv(df, path, index=False)
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


def atomic_write_parquet(
    df: "Any",
    path: Path | str,
    *,
    retries: int = 5,
    backoff_ms: int = 50,
    **kwargs: Any,
) -> None:
    """Write a pandas DataFrame to *path* as Parquet atomically (tmp + os.replace).

    Retries on PermissionError/OSError with exponential backoff (Windows
    file-lock friendly). If all retries fail, re-raises the last exception.

    Args:
        df: pandas DataFrame to write.
        path: Destination file path (.parquet).
        retries: Number of write attempts before giving up.
        backoff_ms: Base back-off between retries in milliseconds.
        **kwargs: Additional keyword arguments forwarded to ``DataFrame.to_parquet()``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            df.to_parquet(str(tmp_path), **kwargs)
            os.replace(str(tmp_path), str(path))
            return
        except (PermissionError, OSError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(backoff_ms * (2**attempt) / 1000.0)
        except Exception:
            # Non-IO errors (e.g. serialisation) — clean up tmp and re-raise immediately
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    if last_err is not None:
        raise last_err


def atomic_write_csv(
    df: "Any",
    path: Path | str,
    *,
    retries: int = 5,
    backoff_ms: int = 50,
    encoding: str = "utf-8",
    **kwargs: Any,
) -> None:
    """Write a pandas DataFrame to *path* as CSV atomically (tmp + os.replace).

    Retries on PermissionError/OSError with exponential backoff (Windows
    file-lock friendly). If all retries fail, re-raises the last exception.

    Args:
        df: pandas DataFrame to write.
        path: Destination file path (.csv).
        retries: Number of write attempts before giving up.
        backoff_ms: Base back-off between retries in milliseconds.
        encoding: File encoding (default utf-8).
        **kwargs: Additional keyword arguments forwarded to ``DataFrame.to_csv()``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / (path.name + ".tmp")
    last_err: BaseException | None = None
    for attempt in range(retries):
        try:
            df.to_csv(str(tmp_path), encoding=encoding, **kwargs)
            os.replace(str(tmp_path), str(path))
            return
        except (PermissionError, OSError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(backoff_ms * (2**attempt) / 1000.0)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except OSError:
        pass
    if last_err is not None:
        raise last_err


# Alias for callers importing the old risk.state_machine name
atomic_write_json_with_retry = atomic_write_json

__all__ = [
    "atomic_write_json",
    "atomic_write_json_with_retry",
    "atomic_write_parquet",
    "atomic_write_csv",
]
