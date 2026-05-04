"""Part D — Shadow-mode infrastructure for wiring activation (D1–D5).

Plan's D-standard: each module that is wired but policy-gated runs in
*shadow mode* for 5 paper-days before the user flips it live. Shadow mode
must:

* execute the module's logic and compute its output
* write the output as a JSON snapshot to ``output/shadow/<module>_<date>.json``
* **not** apply the output to trading decisions

This file centralises the snapshot writer so every D-module shares the same
format, path layout, and safety guarantees (atomic write, UTF-8, no
secrets).

Governance intent
-----------------

* Shadow artifacts are the only evidence backing a go/no-go decision.
* They must be ID-able by module name + date + run_id for diff reports.
* They must survive a crashed paper cycle (atomic write).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SHADOW_ROOT_ENV = "ATI_SHADOW_ROOT"


def default_shadow_root() -> Path:
    """Resolve the shadow-artifact root.

    Env override via ``ATI_SHADOW_ROOT`` beats the default for tests and for
    cleaner CI artifact paths.
    """
    override = os.environ.get(_SHADOW_ROOT_ENV)
    if override:
        return Path(override)
    return Path("output/shadow")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.json")
    try:
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_shadow_snapshot(
    module: str,
    payload: dict[str, Any],
    *,
    snapshot_date: date | None = None,
    run_id: str | None = None,
    shadow_root: Path | None = None,
) -> Path:
    """Persist one shadow-mode snapshot.

    Args:
        module: Module identifier (e.g. ``"correlation_guard"``,
            ``"zombie_killer"``, ``"signal_decay"``).
        payload: JSON-serialisable dict describing the module's output for
            this cycle. Caller owns the schema; this writer only wraps it
            with metadata.
        snapshot_date: Trading date the snapshot applies to. Defaults to
            today in UTC.
        run_id: Optional run identifier. Useful when multiple paper cycles
            share a date (e.g. intraday re-runs).
        shadow_root: Override root dir. Tests pass a tmp_path.

    Returns:
        Absolute path to the written file.
    """
    if not module or "/" in module or "\\" in module:
        raise ValueError(f"invalid module id: {module!r}")

    snap_date = snapshot_date or datetime.now(tz=timezone.utc).date()
    root = shadow_root or default_shadow_root()

    envelope = {
        "module": module,
        "snapshot_date": snap_date.isoformat(),
        "written_at": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "payload": payload,
    }

    suffix = f"_{run_id}" if run_id else ""
    file_path = root / f"{module}_{snap_date.isoformat()}{suffix}.json"
    _atomic_write_json(file_path, envelope)
    logger.info(
        "[SHADOW] module=%s snapshot_date=%s path=%s", module, snap_date, file_path
    )
    return file_path


def read_shadow_snapshot(path: Path) -> dict[str, Any]:
    """Parse a snapshot file. Raises ``FileNotFoundError`` if missing and
    ``ValueError`` if the envelope shape is wrong — both are loud failures,
    because a silent read is worse than no read for audit purposes."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"module", "snapshot_date", "written_at", "payload"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"shadow envelope missing fields: {sorted(missing)}")
    return data
