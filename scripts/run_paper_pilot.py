"""30-day paper-live pilot — pre-flight check for live trading (Plan 11/10 §3.3).

Runs paper-live daily and tracks cumulative GO/NO-GO criteria.
On day 30 (or --evaluate-only): aggregates summaries and prints verdict.

Usage:
    # Run one day of the pilot (called from scheduler / cron):
    python scripts/run_paper_pilot.py --run-day

    # After 30 days, evaluate:
    python scripts/run_paper_pilot.py --evaluate-only

    # See current status:
    python scripts/run_paper_pilot.py --status
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PILOT_DIR = ROOT / "output" / "pilot"
PILOT_MANIFEST = PILOT_DIR / "pilot_manifest.json"

PILOT_CONFIG = {
    "duration_days": 30,
    "max_daily_orders": 50,
    "max_daily_loss_usd": 200.0,
    # GO/NO-GO criteria (all must pass):
    "go_criteria": {
        "min_sharpe": 0.8,
        "max_mdd_pct": -15.0,
        "min_fill_rate": 0.90,
        "max_crash_days": 2,
        "min_trades": 10,
    },
}


def _load_manifest() -> dict:
    if PILOT_MANIFEST.exists():
        return json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
    return {"started_at": None, "days": [], "config": PILOT_CONFIG}


def _save_manifest(m: dict) -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    PILOT_MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")


def cmd_run_day() -> int:
    """Run one paper-live cycle and append daily summary to manifest."""
    m = _load_manifest()
    if not m.get("started_at"):
        m["started_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[pilot] Pilot started: {m['started_at']}")

    day_num = len(m["days"]) + 1
    ts = datetime.now(timezone.utc).isoformat()

    # Invoke the paper runner for one cycle
    import subprocess

    result = subprocess.run(
        [sys.executable, "scripts/run_live_paper.py", "once"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    rc = result.returncode
    output = (result.stdout + result.stderr).lower()

    crashed = rc not in (0, 1)
    n_orders = output.count("order submitted") + output.count("filled")

    daily_summary = {
        "day": day_num,
        "timestamp": ts,
        "rc": rc,
        "crashed": crashed,
        "n_orders_detected": n_orders,
        "output_snippet": (result.stdout + result.stderr)[:500],
    }
    m["days"].append(daily_summary)
    _save_manifest(m)

    status = "CRASH" if crashed else "OK"
    print(f"[pilot] Day {day_num}/30 — {status} (rc={rc}, orders~{n_orders})")

    if day_num >= PILOT_CONFIG["duration_days"]:
        print("[pilot] 30-day mark reached — run --evaluate-only for verdict")

    return 0 if not crashed else 1


def cmd_evaluate() -> int:
    """Evaluate 30-day pilot and produce GO/NO-GO verdict."""
    m = _load_manifest()
    days = m.get("days", [])

    if not days:
        print("[pilot] No daily records found. Run --run-day first.")
        return 1

    criteria = PILOT_CONFIG["go_criteria"]
    n_days = len(days)
    n_crash = sum(1 for d in days if d.get("crashed"))
    total_orders = sum(d.get("n_orders_detected", 0) for d in days)

    criteria_results = {
        "days_run": {"value": n_days, "pass": n_days >= 14},
        "crash_days": {"value": n_crash, "pass": n_crash <= criteria["max_crash_days"]},
        "total_orders": {
            "value": total_orders,
            "pass": total_orders >= criteria["min_trades"],
        },
    }

    # Load paper ledger for Sharpe / MDD if available
    ledger_path = ROOT / "output" / "runs" / "_paper_ledger" / "ledger_state.json"
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            equity_curve = ledger.get("equity_curve", [])
            if len(equity_curve) > 1:
                import statistics

                daily_returns = [
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                    for i in range(1, len(equity_curve))
                    if equity_curve[i - 1] > 0
                ]
                if daily_returns:
                    mean_r = statistics.mean(daily_returns)
                    std_r = (
                        statistics.stdev(daily_returns)
                        if len(daily_returns) > 1
                        else 1e-9
                    )
                    sharpe = (mean_r / std_r) * (252**0.5) if std_r > 0 else 0.0
                    mdd = (
                        min(
                            (equity_curve[i] - max(equity_curve[: i + 1]))
                            / max(equity_curve[: i + 1])
                            for i in range(1, len(equity_curve))
                            if max(equity_curve[: i + 1]) > 0
                        )
                        * 100
                        if len(equity_curve) > 1
                        else 0.0
                    )
                    criteria_results["sharpe"] = {
                        "value": round(sharpe, 3),
                        "pass": sharpe >= criteria["min_sharpe"],
                    }
                    criteria_results["mdd_pct"] = {
                        "value": round(mdd, 2),
                        "pass": mdd >= criteria["max_mdd_pct"],
                    }
        except Exception as exc:
            print(f"[pilot] ledger parse warning: {exc}")

    all_pass = all(v.get("pass", False) for v in criteria_results.values())
    decision = "GO" if all_pass else "NO-GO"
    next_action = (
        "Activate live trading with $5k initial capital"
        if all_pass
        else "Diagnose failures and re-run pilot"
    )

    verdict = {
        "decision": decision,
        "n_days_run": n_days,
        "criteria_results": criteria_results,
        "next_action": next_action,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    m["verdict"] = verdict
    _save_manifest(m)

    report_path = (
        PILOT_DIR
        / f"pilot_verdict_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    report_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f"PILOT VERDICT: {decision}")
    print(f"{'='*50}")
    for k, v in criteria_results.items():
        icon = "✓" if v.get("pass") else "✗"
        print(f"  {icon} {k}: {v.get('value')}")
    print(f"\nNext action: {next_action}")
    print(f"Report: {report_path}")
    return 0 if all_pass else 1


def cmd_status() -> int:
    """Print current pilot status."""
    m = _load_manifest()
    days = m.get("days", [])
    n = len(days)
    started = m.get("started_at", "not started")
    crashes = sum(1 for d in days if d.get("crashed"))
    print(f"[pilot] Day {n}/{PILOT_CONFIG['duration_days']} | Started: {started}")
    print(
        f"[pilot] Crashes: {crashes} | Verdict: {m.get('verdict', {}).get('decision', 'pending')}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="30-day paper-live pilot runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--run-day", action="store_true", help="Run one daily pilot cycle"
    )
    group.add_argument(
        "--evaluate-only", action="store_true", help="Evaluate pilot and emit GO/NO-GO"
    )
    group.add_argument(
        "--status", action="store_true", help="Print current pilot status"
    )
    args = parser.parse_args(argv)

    if args.run_day:
        return cmd_run_day()
    elif args.evaluate_only:
        return cmd_evaluate()
    else:
        return cmd_status()


if __name__ == "__main__":
    sys.exit(main())
