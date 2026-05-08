"""Memory profiling script for the Assembled-Trading-AI pilot (Item 27).

Tracks per-module memory allocations using tracemalloc and logs daily growth.
Run weekly or daily during the pilot to catch memory leaks early.

Usage::

    python scripts/memory_profile.py [--top N] [--output output/memory_profile.json]

    # Run a quick allocation snapshot (no live process needed):
    python scripts/memory_profile.py --snapshot

    # Diff two snapshots to see what grew:
    python scripts/memory_profile.py --diff snap1.json snap2.json

Exit codes:
    0 — snapshot taken / diff OK
    1 — growth alert: RSS increase exceeds threshold
    2 — error (bad arguments, file not found)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_DEFAULT_OUTPUT = Path("output") / "memory_profile.json"
_DEFAULT_TOP = 20
_GROWTH_ALERT_MB = 100  # alert if RSS grew by more than this in a diff


def _take_snapshot(top_n: int) -> dict:
    """Take a tracemalloc snapshot and return a JSON-serialisable summary."""
    tracemalloc.start()
    # Import the heaviest modules to give tracemalloc data to work with
    try:
        import src.assembled_core  # noqa: F401 — populate module cache
    except Exception:
        pass

    snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    top_stats = snapshot.statistics("lineno")[:top_n]
    stats = [
        {
            "traceback": str(s.traceback),
            "size_kb": round(s.size / 1024, 2),
            "count": s.count,
        }
        for s in top_stats
    ]
    total_kb = sum(s.size for s in top_stats) / 1024
    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "total_kb": round(total_kb, 2),
        "top_allocations": stats,
    }


def _process_rss_mb() -> float:
    """Return current process RSS in MB (best-effort, 0 if unavailable)."""
    try:
        import resource  # POSIX only

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        pass
    # Windows fallback via tasklist
    try:
        import subprocess

        pid = os.getpid()
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            parts = line.strip('"').split('","')
            if len(parts) >= 5:
                mem_str = parts[4].replace(",", "").replace("K", "").strip()
                try:
                    return float(mem_str) / 1024
                except ValueError:
                    pass
    except Exception:
        pass
    return 0.0


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Take a fresh memory snapshot and write it to --output."""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log.info("[memory_profile] Taking snapshot (top %d allocations)...", args.top)
    data = _take_snapshot(args.top)
    data["rss_mb"] = round(_process_rss_mb(), 2)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log.info(
        "[memory_profile] Snapshot written → %s (%.1f KB total)",
        output,
        data["total_kb"],
    )
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Diff two snapshot files and alert if growth exceeds threshold."""
    p1, p2 = Path(args.diff[0]), Path(args.diff[1])
    if not p1.exists():
        log.error("Snapshot file not found: %s", p1)
        return 2
    if not p2.exists():
        log.error("Snapshot file not found: %s", p2)
        return 2
    s1 = json.loads(p1.read_text(encoding="utf-8"))
    s2 = json.loads(p2.read_text(encoding="utf-8"))
    rss_delta = s2.get("rss_mb", 0) - s1.get("rss_mb", 0)
    kb_delta = s2.get("total_kb", 0) - s1.get("total_kb", 0)
    log.info(
        "[memory_profile] Diff: rss_delta=%.1f MB, traced_kb_delta=%.1f KB",
        rss_delta,
        kb_delta,
    )
    if rss_delta > _GROWTH_ALERT_MB:
        log.warning(
            "[memory_profile] ALERT: RSS grew by %.1f MB (threshold: %d MB) — "
            "possible memory leak!",
            rss_delta,
            _GROWTH_ALERT_MB,
        )
        return 1
    log.info("[memory_profile] Growth within bounds.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Memory profiling tool for the Assembled-Trading-AI pilot."
    )
    parser.add_argument(
        "--snapshot", action="store_true", help="Take a memory snapshot"
    )
    parser.add_argument(
        "--diff", nargs=2, metavar="FILE", help="Diff two snapshot JSON files"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=_DEFAULT_TOP,
        help=f"Number of top allocations to track (default: {_DEFAULT_TOP})",
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {_DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    if args.diff:
        return cmd_diff(args)
    # Default: always snapshot (--snapshot is optional flag)
    return cmd_snapshot(args)


if __name__ == "__main__":
    sys.exit(main())
