"""scripts/backup_databases.py — Backup DuckDB and SQLite databases.

Creates compressed tarballs in ~/backups/assembled-trading-ai/<date>/
or a custom --backup-dir.

Usage:
    python scripts/backup_databases.py
    python scripts/backup_databases.py --backup-dir /mnt/backup --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = Path.home() / "backups" / "assembled-trading-ai"
DB_PATTERNS = ["**/*.duckdb", "**/*.db", "**/*.sqlite", "**/*.sqlite3"]
EXCLUDE_DIRS = {".venv", "__pycache__", "node_modules", ".git"}


def _find_dbs(root: Path) -> list[Path]:
    found = []
    for pattern in DB_PATTERNS:
        for p in root.glob(pattern):
            if not any(ex in p.parts for ex in EXCLUDE_DIRS):
                found.append(p)
    return sorted(set(found))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _main() -> int:
    ap = argparse.ArgumentParser(description="Backup DuckDB/SQLite databases")
    ap.add_argument("--backup-dir", type=Path, default=DEFAULT_BACKUP_DIR)
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--source-dir", type=Path, default=ROOT)
    args = ap.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = args.backup_dir / today

    dbs = _find_dbs(args.source_dir)
    if not dbs:
        log.info("[backup] No database files found under %s", args.source_dir)
        return 0

    log.info("[backup] Found %d database file(s)", len(dbs))

    if args.dry_run:
        for db in dbs:
            size_mb = db.stat().st_size / 1024 / 1024
            log.info(
                "[DRY-RUN] Would backup: %s (%.1f MB)",
                db.relative_to(args.source_dir),
                size_mb,
            )
        log.info("[DRY-RUN] Target dir: %s", dest)
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    total_mb = 0.0
    for db in dbs:
        rel = db.relative_to(args.source_dir)
        out_name = str(rel).replace("/", "_").replace("\\", "_") + ".gz"
        out_path = dest / out_name
        size_mb = db.stat().st_size / 1024 / 1024
        sha = _sha256(db)
        with open(db, "rb") as fin, gzip.open(out_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        total_mb += size_mb
        log.info("[backup] %s → %s (%.1f MB, sha=%s)", rel, out_name, size_mb, sha)

    log.info("[backup] Done — %d files, %.1f MB total → %s", len(dbs), total_mb, dest)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
