"""Evidence index writer for accounting artifacts.

This module provides a small helper to write an "evidence index" JSON file
that links together all relevant artifacts for a run (snapshot, ledger pack,
reconciliation report, accounting report, manifest).

The goal is to make Ops/Support workflows easier by having a single, stable
entry point per run/day that references all downstream files.

Paths schema: The "paths" object in the JSON always contains all keys (missing -> null).
Stable key order via sort_keys=True on dump. Keys: broker_snapshot_path, ledger_pack_path,
reconcile_report_path, accounting_report_path, manifest_path.
"""

from __future__ import annotations

# Fixed paths schema: all keys always present (value None if missing). Order stable via sort_keys.
PATHS_KEYS = (
    "broker_snapshot_path",
    "ledger_pack_path",
    "reconcile_report_path",
    "accounting_report_path",
    "manifest_path",
)

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from src.assembled_core import __version__ as CORE_VERSION


def _rel_posix(path: str | Path | None, base_dir: Path) -> str | None:
    """Convert a path to a POSIX-style path relative to base_dir, if possible."""
    if path is None:
        return None

    p = Path(path)
    try:
        rel = p.relative_to(base_dir)
        return rel.as_posix()
    except Exception:
        # Fall back to POSIX representation (may still be absolute if outside base_dir)
        return p.as_posix()


def write_evidence_index_json(
    output_dir: Path | str,
    run_id: str,
    as_of_date: pd.Timestamp | str,
    *,
    paths: dict[str, Any],
    broker_meta: dict[str, Any] | None = None,
    reconciliation_ok: bool | None = None,
) -> Path:
    """Write evidence index JSON for a given run and date.

    Args:
        output_dir: Base output directory
        run_id: Run identifier (e.g. ledger run id)
        as_of_date: Report date (UTC, tz-aware or string)
        paths: Dictionary with known path keys:
            - broker_snapshot_path
            - ledger_pack_path
            - reconcile_report_path
            - accounting_report_path
            - manifest_path (optional, may be None)
        broker_meta: Optional broker metadata dict (same shape as in reports)
        reconciliation_ok: Optional reconciliation status

    Returns:
        Path to written evidence index JSON file.

    Determinism: Output JSON uses sort_keys=True, indent=2, trailing newline; as_of_date is date-only (YYYY-MM-DD).
    """
    base = Path(output_dir)

    # Normalize as_of_date (accept str, pd.Timestamp, or datetime; always UTC, date-only for determinism)
    as_of_ts = pd.to_datetime(as_of_date, utc=True)
    if as_of_ts.tz is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    date_str = as_of_ts.strftime("%Y-%m-%d")

    evidence_dir = base / f"evidence_{run_id}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    json_path = evidence_dir / f"evidence_{date_str}.json"

    # Normalize known paths to POSIX + relative where possible; missing -> None (paths always has all keys)
    paths_block: dict[str, Any] = {}
    for key in PATHS_KEYS:
        paths_block[key] = _rel_posix(paths.get(key), base)

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "as_of_date": date_str,
        "paths": paths_block,
        # Optional metadata
        "reconciliation_ok": reconciliation_ok,
        "tool_version": CORE_VERSION,
    }

    if broker_meta is not None:
        evidence["broker_meta"] = broker_meta

    # Write JSON deterministically (sort_keys=True, indent=2, trailing newline); atomic write (temp + replace)
    # evidence is schema-fixed: only dict/list/str/int/bool/None (PATHS_KEYS + date-only as_of_date)
    content = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=str(json_path.parent),
            prefix=json_path.name + ".tmp.",
            suffix=".json",
        ) as f:
            tmp_path = Path(f.name)
            f.write(content)
            f.flush()
        tmp_path.replace(json_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass  # best-effort cleanup — original exception already propagated

    return json_path
