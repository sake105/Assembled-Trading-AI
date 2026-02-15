#!/usr/bin/env python3
"""Local Sprint-13 release helper: run blocking checks and optional ops_evidence (ASCII-only summary).

One command to run: release_sprint13 -> evidence_pack -> (optional) ops_evidence.
Exit 0 if blocking steps OK, else 1. Output is ASCII-only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_CHECKS = REPO_ROOT / "scripts" / "dev" / "run_checks.py"


def _ascii(s: str) -> str:
    return s.encode("ascii", errors="ignore").decode("ascii")


def run_cmd(cmd: list[str], dry_run: bool) -> int:
    """Run command; return exit code. If dry_run, print command and return 0."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(_ascii(f"[dry-run] {cmd_str}"))
        return 0
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Sprint-13 release checks locally (release_sprint13, evidence_pack, optional ops_evidence)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not execute",
    )
    parser.add_argument(
        "--ops-evidence",
        action="store_true",
        help="Also run ops_evidence preset (optional, skip-compile and skip-ruff)",
    )
    args = parser.parse_args()
    dry_run = args.dry_run
    python = sys.executable

    steps: list[tuple[str, list[str]]] = [
        ("release_sprint13", [python, str(RUN_CHECKS), "--preset", "release_sprint13"]),
        ("evidence_pack", [python, str(RUN_CHECKS), "--preset", "evidence_pack"]),
    ]
    if args.ops_evidence:
        steps.append(
            (
                "ops_evidence",
                [python, str(RUN_CHECKS), "--preset", "ops_evidence", "--skip-compile", "--skip-ruff"],
            )
        )

    results: list[tuple[str, int]] = []
    for name, cmd in steps:
        code = run_cmd(cmd, dry_run)
        results.append((name, code))
        if code != 0 and not dry_run:
            # Blocking steps (first two) failing -> we will exit 1
            pass

    # Summary line: OK: release_sprint13=PASS evidence_pack=PASS ops_evidence=PASS
    status_parts = [f"{name}=PASS" if c == 0 else f"{name}=FAIL" for name, c in results]
    summary = "OK: " + " ".join(status_parts)
    print(_ascii(summary))

    # Exit 0 only if blocking steps (release_sprint13, evidence_pack) passed
    blocking_ok = all(c == 0 for name, c in results if name in ("release_sprint13", "evidence_pack"))
    return 0 if blocking_ok else 1


if __name__ == "__main__":
    sys.exit(main())
