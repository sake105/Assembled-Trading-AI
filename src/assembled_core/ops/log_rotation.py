"""Rotating file handler setup for long-running pilots.

Prevents disk-full issues during 30-day pilots with detailed logging.

Usage:
    from src.assembled_core.ops.log_rotation import setup_rotating_log

    setup_rotating_log("output/logs/pilot.log", max_bytes=100*1024*1024, backup_count=10)
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_rotating_log(
    log_path: str | Path,
    max_bytes: int = 100 * 1024 * 1024,  # 100 MB
    backup_count: int = 10,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> RotatingFileHandler:
    """Add a rotating file handler to the root logger.

    Creates parent directories if needed. Returns the handler so the caller
    can remove it later if required.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(handler)

    logging.getLogger(__name__).info(
        "[log_rotation] Rotating log: %s (max %.0f MB × %d backups)",
        log_path,
        max_bytes / 1024 / 1024,
        backup_count,
    )
    return handler
