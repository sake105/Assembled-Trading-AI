"""Full Point-in-Time (PIT) artifact store (X1).

Generalises TriggerSnapshotStore to archive any JSON-serialisable artifact by
(source, run_id, artifact_type). Supports:
  - Multi-artifact-type storage per run_id
  - "as_of" lookup: latest run_id whose archived_utc <= as_of
  - Run manifest listing all artifact types in a run
  - Cross-source iteration for backtesting replay
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_MANIFEST_FILE = "_manifest.json"


class PITStore:
    """General-purpose PIT artifact store (X1 full).

    Directory layout::

        <root>/<source>/<run_id>/<artifact_type>.json
        <root>/<source>/<run_id>/_manifest.json

    Artifact types can be any string: ``"triggers"``, ``"clusters"``,
    ``"features"``, ``"evidence_grade"``, etc.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def archive(
        self,
        source: str,
        run_id: str,
        artifact_type: str,
        data: dict | list | Any,
        *,
        overwrite: bool = False,
        archived_utc: datetime | None = None,
    ) -> Path:
        """Persist `data` as a JSON artifact under (source, run_id, artifact_type).

        Returns path of the written file.
        """
        dest_dir = self._root / source / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{artifact_type}.json"

        if dest.exists() and not overwrite:
            logger.debug("[SKIP] PITStore: already exists %s", dest)
            return dest

        dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

        now_utc = (archived_utc or datetime.now(tz=timezone.utc)).isoformat()
        self._update_manifest(dest_dir, artifact_type, now_utc)

        logger.info(
            "[OK] PITStore.archive: source=%s run_id=%s type=%s → %s",
            source, run_id, artifact_type, dest,
        )
        return dest

    def archive_file(
        self,
        source: str,
        run_id: str,
        artifact_type: str,
        file_path: Path,
        *,
        overwrite: bool = False,
    ) -> Path | None:
        """Copy a file into the store as a given artifact_type.

        Returns dest path or None if source file is missing.
        """
        if not file_path.exists():
            logger.warning("[SKIP] PITStore.archive_file: %s not found", file_path)
            return None
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[WARN] PITStore.archive_file: JSON parse error %s: %s", file_path, exc)
            return None
        return self.archive(source, run_id, artifact_type, data, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load(
        self,
        source: str,
        run_id: str,
        artifact_type: str,
    ) -> dict | list | None:
        """Load a specific artifact. Returns None if not found."""
        path = self._root / source / run_id / f"{artifact_type}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[WARN] PITStore.load: failed %s: %s", path, exc)
            return None

    def load_as_of(
        self,
        source: str,
        artifact_type: str,
        as_of: str | datetime,
    ) -> dict | list | None:
        """Return the latest artifact whose archived_utc <= as_of.

        Useful for PIT-correct backtesting — returns what was available at `as_of`.
        """
        if isinstance(as_of, str):
            as_of_dt = datetime.fromisoformat(as_of)
        else:
            as_of_dt = as_of
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)

        candidates: list[tuple[datetime, str]] = []
        source_dir = self._root / source
        if not source_dir.exists():
            return None

        for run_id_dir in sorted(source_dir.iterdir()):
            if not run_id_dir.is_dir():
                continue
            manifest = self._load_manifest(run_id_dir)
            entry = manifest.get(artifact_type)
            if not entry:
                continue
            try:
                archived_dt = datetime.fromisoformat(entry["archived_utc"])
                if archived_dt.tzinfo is None:
                    archived_dt = archived_dt.replace(tzinfo=timezone.utc)
                if archived_dt <= as_of_dt:
                    candidates.append((archived_dt, run_id_dir.name))
            except Exception:
                continue

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0])
        _, best_run_id = candidates[-1]
        return self.load(source, best_run_id, artifact_type)

    def list_run_ids(self, source: str) -> list[str]:
        """Return all run_ids for a source, sorted."""
        source_dir = self._root / source
        if not source_dir.exists():
            return []
        return sorted(d.name for d in source_dir.iterdir() if d.is_dir())

    def manifest(self, source: str, run_id: str) -> dict[str, dict]:
        """Return the manifest for a (source, run_id) pair."""
        run_dir = self._root / source / run_id
        return self._load_manifest(run_dir)

    def latest(self, source: str, artifact_type: str) -> dict | list | None:
        """Load the most recently archived artifact of a given type for a source."""
        run_ids = self.list_run_ids(source)
        for run_id in reversed(run_ids):
            data = self.load(source, run_id, artifact_type)
            if data is not None:
                return data
        return None

    # ------------------------------------------------------------------
    # Replay / iteration
    # ------------------------------------------------------------------

    def iter_chronological(
        self,
        source: str,
        artifact_type: str,
    ) -> Iterator[tuple[str, dict | list]]:
        """Yield (run_id, data) in chronological order of archived_utc.

        Used for X5 News-Replay: iterate all archived snapshots in time order.
        """
        candidates: list[tuple[datetime, str]] = []
        source_dir = self._root / source
        if not source_dir.exists():
            return

        for run_id_dir in source_dir.iterdir():
            if not run_id_dir.is_dir():
                continue
            manifest = self._load_manifest(run_id_dir)
            entry = manifest.get(artifact_type)
            if not entry:
                continue
            try:
                archived_dt = datetime.fromisoformat(entry["archived_utc"])
                if archived_dt.tzinfo is None:
                    archived_dt = archived_dt.replace(tzinfo=timezone.utc)
                candidates.append((archived_dt, run_id_dir.name))
            except Exception:
                continue

        candidates.sort(key=lambda t: t[0])
        for _, run_id in candidates:
            data = self.load(source, run_id, artifact_type)
            if data is not None:
                yield run_id, data

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_manifest(self, run_dir: Path, artifact_type: str, archived_utc: str) -> None:
        manifest = self._load_manifest(run_dir)
        manifest[artifact_type] = {"archived_utc": archived_utc}
        manifest_path = run_dir / _MANIFEST_FILE
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _load_manifest(self, run_dir: Path) -> dict[str, dict]:
        manifest_path = run_dir / _MANIFEST_FILE
        if not manifest_path.exists():
            return {}
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
