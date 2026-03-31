#!/usr/bin/env python3
"""Single system-run command (dev): smoke backtest + analysis; optional evidence verify.

Runs:
  1. scripts/dev/smoke_backtest_local.py (synthetic data backtest + analyze_backtest_results)
  2. Optionally: evidence pack verify if --verify-evidence and pack exists

Usage:
    py -3 scripts/dev/run_system_run.py
    py -3 scripts/dev/run_system_run.py --verify-evidence
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dev system run: smoke backtest + analysis."
    )
    parser.add_argument(
        "--verify-evidence",
        action="store_true",
        help="Run evidence pack verify after backtest (if pack was written)",
    )
    args = parser.parse_args()

    smoke = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "dev" / "smoke_backtest_local.py")],
        cwd=str(ROOT),
        timeout=120,
    )
    if smoke.returncode != 0:
        return smoke.returncode

    if args.verify_evidence:
        pack_dir = ROOT / "output" / "analysis_run" / "smoke"
        # Smoke run does not write evidence pack by default; skip if no pack
        evidence_zip = (
            next((pack_dir / "evidence").glob("*.zip"), None)
            if (pack_dir / "evidence").exists()
            else None
        )
        if evidence_zip:
            r = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "verify_evidence_pack.py"),
                    str(evidence_zip),
                ],
                cwd=str(ROOT),
                timeout=30,
            )
            if r.returncode != 0:
                return r.returncode
        # else: no pack, skip verify

    print("System run OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
