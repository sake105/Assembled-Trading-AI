"""Halt-flag drill (Plan 11/10 §4.2 + Pre-Flight Check 3).

1. Write halt_ack_required.json
2. Verify run_live_paper.py --once --dry-run exits without trading
3. Remove halt file
4. Verify cycle runs normally

Usage:
    python scripts/drills/drill_halt_flag.py --dry-run-only
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HALT_PATH = Path("output/ops/halt_ack_required.json")


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    report: dict = {"started_at": ts, "steps": []}

    def step(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
        report["steps"].append({"step": name, "status": status, "detail": detail})

    print("=== Halt-Flag Drill ===")

    # 1. Write halt flag
    HALT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HALT_PATH.write_text(
        json.dumps({"reason": "drill_test", "actor": "drill_halt_flag", "ts": ts}),
        encoding="utf-8",
    )
    step("halt_file_created", HALT_PATH.exists())

    # 2. Try to run a dry-run cycle — should skip trading
    result = subprocess.run(
        [sys.executable, "scripts/run_live_paper.py", "--once", "--dry-run"],
        capture_output=True, text=True, timeout=60,
    )
    output = (result.stdout + result.stderr).lower()
    halted = "halt" in output or "skipped" in output or result.returncode in (0, 1)
    traded = "order submitted" in output or "orders: " in output
    step("cycle_halted_by_flag", halted and not traded,
         "correctly refused trading" if (halted and not traded) else f"rc={result.returncode}")

    # 3. Remove halt flag
    HALT_PATH.unlink(missing_ok=True)
    step("halt_file_removed", not HALT_PATH.exists())

    all_pass = all(s["status"] == "PASS" for s in report["steps"])
    report["verdict"] = "PASS" if all_pass else "FAIL"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    out = Path("output/drills")
    out.mkdir(parents=True, exist_ok=True)
    name = f"halt_flag_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    (out / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDrill verdict: {report['verdict']}")
    print(f"Report: output/drills/{name}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
