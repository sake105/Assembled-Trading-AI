"""Retention utility for date-stamped report files.

Background:
    Several pipeline hooks (signal_decay_YYYYMMDD.json,
    signal_correlation_YYYYMMDD.json, tca_report_YYYYMMDD.json,
    stress_test_YYYYMMDD.json) write one file per run with no cleanup.
    Over months these grow unbounded.

This module provides a single helper used at write sites.
Non-blocking: failures are swallowed; return 0 on any issue.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def purge_old_dated_reports(
    directory: Path,
    prefix: str,
    suffix: str = ".json",
    keep_last_n: int = 60,
) -> int:
    """Keep only the most recent ``keep_last_n`` files matching ``<prefix>*<suffix>``.

    Files are sorted by modification time (most recent first). Older files are
    deleted. Used at dated-report write sites to bound disk usage.

    Args:
        directory: Directory to scan.
        prefix: Required filename prefix (e.g. ``"signal_decay_"``).
        suffix: Required suffix, default ``".json"``.
        keep_last_n: Number of newest files to keep.

    Returns:
        Number of files purged. 0 on error or if nothing to purge.
    """
    try:
        if keep_last_n < 0:
            return 0
        d = Path(directory)
        if not d.exists() or not d.is_dir():
            return 0
        candidates = [
            p
            for p in d.iterdir()
            if p.is_file() and p.name.startswith(prefix) and p.name.endswith(suffix)
        ]
        if len(candidates) <= keep_last_n:
            return 0
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        to_delete = candidates[keep_last_n:]
        n_deleted = 0
        for p in to_delete:
            try:
                p.unlink()
                n_deleted += 1
            except Exception as exc:
                logger.debug("[retention] unlink failed for %s: %s", p, exc)
        if n_deleted:
            logger.info(
                "[retention] %s%s: purged %d old files (kept %d)",
                prefix,
                suffix,
                n_deleted,
                keep_last_n,
            )
        return n_deleted
    except Exception as exc:
        logger.debug("[retention] purge failed for %s/%s*: %s", directory, prefix, exc)
        return 0


__all__ = ["purge_old_dated_reports"]
