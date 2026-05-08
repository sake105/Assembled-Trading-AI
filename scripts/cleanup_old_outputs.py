"""scripts/cleanup_old_outputs.py — Backlog Item 71: Storage cleanup.

Prunes old output files to prevent disk-full during extended pilots.

Always kept (never deleted):
  - Files starting with: equity_curve*, manifest*, pilot_v2_manifest*
  - Pilot verdicts, baselines, markdown reports, system maps

Usage:
    python scripts/cleanup_old_outputs.py --dry-run               # safe preview (default)
    python scripts/cleanup_old_outputs.py --max-age-days 60       # preview files >60 days old
    python scripts/cleanup_old_outputs.py --execute               # actually delete
    python scripts/cleanup_old_outputs.py --output-dir output/ --max-age-days 14 --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Prefixes/patterns that are ALWAYS kept regardless of age
ALWAYS_KEEP_PREFIXES = (
    "equity_curve",
    "manifest",
    "pilot_v2_manifest",
    "pilot_manifest",
    "pilot_verdict",
    "baseline_post",
    "KNOWN_ISSUES",
    "system_map",
)

ALWAYS_KEEP_SUFFIXES = (".md",)


def _is_always_keep(path: Path) -> bool:
    name = path.name
    if name.startswith(ALWAYS_KEEP_PREFIXES):
        return True
    if name.endswith(ALWAYS_KEEP_SUFFIXES):
        return True
    return False


def _file_age_days(path: Path) -> float:
    mtime = path.stat().st_mtime
    now = datetime.now(tz=timezone.utc).timestamp()
    return (now - mtime) / 86400


def _format_mb(n_bytes: int) -> str:
    return f"{n_bytes / 1_048_576:.2f} MB"


def run_cleanup(
    output_dir: Path,
    max_age_days: int = 30,
    execute: bool = False,
) -> dict:
    """Collect and optionally delete files older than max_age_days.

    Returns summary dict with counts and bytes.
    """
    candidates: list[Path] = []
    for item in sorted(output_dir.rglob("*")):
        if not item.is_file():
            continue
        if _is_always_keep(item):
            continue
        if _file_age_days(item) > max_age_days:
            candidates.append(item)

    total_bytes = sum(f.stat().st_size for f in candidates)
    deleted = 0
    errors = 0
    prefix = "[DELETE]" if execute else "[DRY-RUN]"

    for path in candidates:
        size = path.stat().st_size
        print(f"{prefix} {path}  ({_format_mb(size)})")
        if execute:
            try:
                path.unlink()
                deleted += 1
            except OSError as exc:
                print(f"[ERROR] Could not delete {path}: {exc}")
                errors += 1

    # Remove empty directories (only when executing)
    if execute:
        for d in sorted(output_dir.rglob("*"), reverse=True):
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except OSError:
                    pass

    return {
        "candidates": len(candidates),
        "total_bytes": total_bytes,
        "deleted": deleted,
        "errors": errors,
        "execute": execute,
        "max_age_days": max_age_days,
    }


def _main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete output files older than N days.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output",
        help="Directory to clean (default: output/).",
    )
    ap.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Delete files older than this many days (default: 30).",
    )
    # Dry-run is the default; --execute is the opt-in for real deletions.
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview deletions without removing anything (this is the default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually delete files. Without this flag the script is always a dry-run.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = args.output_dir.resolve()
    if not output_dir.exists():
        logger.error("[cleanup] Output directory does not exist: %s", output_dir)
        return 1

    execute = args.execute  # False → dry-run (safe default)

    logger.info(
        "[cleanup] Scanning %s  max_age_days=%d  mode=%s",
        output_dir,
        args.max_age_days,
        "EXECUTE" if execute else "DRY-RUN",
    )

    result = run_cleanup(output_dir, args.max_age_days, execute)

    print()
    if execute:
        print(
            f"Deleted {result['deleted']} file(s)"
            + (f", {result['errors']} error(s)" if result["errors"] else "")
            + f" — {_format_mb(result['total_bytes'])} freed."
        )
    else:
        print(
            f"{result['candidates']} file(s) would be deleted"
            f" — {_format_mb(result['total_bytes'])} would be freed."
            f"  (run with --execute to delete)"
        )

    return 0 if result["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
