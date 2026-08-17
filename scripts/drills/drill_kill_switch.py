"""Weekly kill-switch drill (Plan 11/10 §4.2).

1. Activate kill-switch
2. Verify is_kill_switch_engaged() returns True
3. Deactivate kill-switch
4. Verify is_kill_switch_engaged() returns False
5. Write drill report to output/drills/

Usage:
    python scripts/drills/drill_kill_switch.py

Run via cron: 09:00 ET Monday weekly
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.assembled_core.execution.kill_switch import (
    activate_kill_switch,
    deactivate_kill_switch,
    is_kill_switch_engaged,
)


def main() -> int:
    ts = datetime.now(timezone.utc).isoformat()
    report: dict = {"started_at": ts, "steps": []}

    def step(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
        report["steps"].append({"step": name, "status": status, "detail": detail})

    print("=== Kill-Switch Drill ===")

    # 1. Should start disengaged
    initial = is_kill_switch_engaged()
    step(
        "initial_state_disengaged",
        not initial,
        (
            # Keine Nicht-cp1252-Glyphen in print-Pfaden: der GitHub-
            # Windows-Runner printet mit cp1252 - ein U+2713 hier crashte
            # den Drill 5 Wochen lang mit UnicodeEncodeError, BEVOR
            # irgendein Drill-Schritt lief (E-177).
            "already engaged at drill start - deactivate first"
            if initial
            else "disengaged"
        ),
    )

    if initial:
        try:
            deactivate_kill_switch(
                reason="drill_cleanup",
                actor="drill_kill_switch",
                operator_token=os.environ.get("OPERATOR_KILL_TOKEN"),
            )
        except PermissionError as _pe:
            step(
                "drill_precondition_deactivate",
                False,
                f"OPERATOR_KILL_TOKEN not configured: {_pe}",
            )
            report["verdict"] = "FAIL"
            _write(report)
            return 1

    # 2. Activate
    try:
        activate_kill_switch(
            throttle_pct=0.0, reason="drill_test", actor="drill_kill_switch"
        )
        engaged = is_kill_switch_engaged()
        step("activation_works", engaged, "engaged" if engaged else "FAILED to engage")
    except Exception as exc:
        step("activation_works", False, str(exc))
        report["verdict"] = "FAIL"
        _write(report)
        return 1

    # 3. Deactivate
    try:
        deactivate_kill_switch(
            reason="drill_done",
            actor="drill_kill_switch",
            operator_token=os.environ.get("OPERATOR_KILL_TOKEN"),
        )
        disengaged = not is_kill_switch_engaged()
        step(
            "deactivation_works",
            disengaged,
            "disengaged" if disengaged else "FAILED to disengage",
        )
    except Exception as exc:
        step("deactivation_works", False, str(exc))
        report["verdict"] = "FAIL"
        _write(report)
        return 1

    all_pass = all(s["status"] == "PASS" for s in report["steps"])
    report["verdict"] = "PASS" if all_pass else "FAIL"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()

    _write(report)
    print(f"\nDrill verdict: {report['verdict']}")
    return 0 if all_pass else 1


def _write(report: dict) -> None:
    out = Path("output/drills")
    out.mkdir(parents=True, exist_ok=True)
    name = f"kill_switch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    (out / name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: output/drills/{name}")


if __name__ == "__main__":
    sys.exit(main())
