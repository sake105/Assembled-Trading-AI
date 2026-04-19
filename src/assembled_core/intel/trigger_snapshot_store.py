"""Trigger snapshot store — archives triggers_latest.json per run_id (T6.1 / X1-lite).

Prevents triggers_latest.json from being overwritten between runs, enabling
PIT-reproducibility of intel signals in evidence packs and backtests.

Usage:
    store = TriggerSnapshotStore("output/intel/snapshots")
    store.archive("news", run_id, Path("data/intel/triggers_latest.json"))
    snap = store.load("news", run_id)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class TriggerSnapshotStore:
    """Archives and retrieves per-run trigger snapshots.

    Directory layout:
        <root>/<source>/<run_id>/triggers.json
        <root>/<source>/<run_id>/meta.json
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def archive(
        self,
        source: str,
        run_id: str,
        artifact_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path | None:
        """Copy `artifact_path` into the snapshot store for `(source, run_id)`.

        Returns the destination path on success, None if artifact_path missing.
        """
        if not artifact_path.exists():
            logger.warning(
                "[SKIP] T6.1 snapshot archive: %s not found for run_id=%s",
                artifact_path, run_id,
            )
            return None

        dest_dir = self._root / source / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "triggers.json"

        if dest.exists() and not overwrite:
            logger.debug(
                "[SKIP] T6.1 snapshot already exists: %s", dest
            )
            return dest

        shutil.copy2(artifact_path, dest)

        meta = {
            "source": source,
            "run_id": run_id,
            "archived_utc": datetime.now(tz=timezone.utc).isoformat(),
            "original_path": str(artifact_path),
        }
        (dest_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info(
            "[OK] T6.1 trigger snapshot archived: source=%s run_id=%s → %s",
            source, run_id, dest,
        )
        return dest

    def load(self, source: str, run_id: str) -> dict | None:
        """Load archived trigger snapshot for `(source, run_id)`.

        Returns parsed JSON dict or None if not found.
        """
        snap_path = self._root / source / run_id / "triggers.json"
        if not snap_path.exists():
            return None
        try:
            return json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[WARN] T6.1 failed to load snapshot %s: %s", snap_path, exc)
            return None

    def list_run_ids(self, source: str) -> list[str]:
        """List all archived run_ids for a source."""
        source_dir = self._root / source
        if not source_dir.exists():
            return []
        return sorted(d.name for d in source_dir.iterdir() if d.is_dir())

    def latest(self, source: str) -> dict | None:
        """Load the most recently archived snapshot for a source (by run_id sort order)."""
        run_ids = self.list_run_ids(source)
        if not run_ids:
            return None
        return self.load(source, run_ids[-1])
