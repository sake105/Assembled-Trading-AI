"""Cross-run aggregate index for paper-engine runs.

Each row in ``output/manifests/index.csv`` summarises one finished run. The
index is append-only and deterministic-ordered. It enables simple Before/After
comparisons, status dashboards, and anomaly queries without walking individual
manifest files.

Schema (v1):
    run_id, date, status, final_equity, total_return, n_fills,
    avg_cost_bps, git_sha, config_hash, manifest_path, written_at_utc
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INDEX_COLUMNS = (
    "run_id",
    "date",
    "status",
    "final_equity",
    "total_return",
    "n_fills",
    "avg_cost_bps",
    "git_sha",
    "config_hash",
    "manifest_path",
    "written_at_utc",
)


def _read_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows


def append_run_index(
    *,
    run_id: str,
    date: str,
    status: str,
    metrics: dict[str, Any],
    git_sha: str,
    config_hash: str,
    manifest_path: Path,
    index_path: Path = Path("output/manifests/index.csv"),
) -> Path:
    """Append one run to the CSV index, rewriting for deterministic sort.

    If a row for ``(run_id, date)`` already exists it is replaced.

    Args:
        run_id: Logical run id.
        date: Trading date (ISO ``YYYY-MM-DD``).
        status: ``success`` / ``error`` / ``kill_switch``.
        metrics: Dict with ``final_equity``, ``total_return``, ``n_fills``,
            ``avg_cost_bps`` (missing keys become empty strings).
        git_sha: git SHA string (may be empty).
        config_hash: 16-char config hash (may be empty).
        manifest_path: Path to the per-day manifest.
        index_path: Where to write the aggregate index CSV.

    Returns:
        ``index_path``.
    """
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing(index_path)

    key = (run_id, date)
    updated: list[dict] = [r for r in existing if (r.get("run_id"), r.get("date")) != key]

    new_row = {
        "run_id": run_id,
        "date": date,
        "status": status,
        "final_equity": str(metrics.get("final_equity", "")),
        "total_return": str(metrics.get("total_return", "")),
        "n_fills": str(metrics.get("n_fills", "")),
        "avg_cost_bps": str(metrics.get("avg_cost_bps", "")),
        "git_sha": git_sha,
        "config_hash": config_hash,
        "manifest_path": Path(manifest_path).as_posix(),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    updated.append(new_row)

    # Deterministic sort: by date then run_id.
    updated.sort(key=lambda r: (str(r.get("date", "")), str(r.get("run_id", ""))))

    with open(index_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for r in updated:
            writer.writerow({c: r.get(c, "") for c in INDEX_COLUMNS})

    logger.info("[RUN_INDEX] Appended %s/%s to %s", run_id, date, index_path)
    return index_path


__all__ = ["INDEX_COLUMNS", "append_run_index"]
