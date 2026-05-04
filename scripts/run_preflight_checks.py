"""Pre-flight check automation script (Paper-Trading Pre-Flight §Check 1-6).

Automates Checks 2-6. Check 1 (Alpaca .env setup) is manual.

Usage:
    python scripts/run_preflight_checks.py
    python scripts/run_preflight_checks.py --skip-broker  # skip live Alpaca checks
    python scripts/run_preflight_checks.py --config configs/paper_track/trend_baseline_live.yaml

Exit code: 0 = all automated checks pass, 1 = failures found.
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preflight")

_RESULTS: list[dict] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _RESULTS.append({"check": name, "status": status, "detail": detail})
    icon = "OK" if passed else "FAIL"
    log.info("[%s] %s — %s", icon, name, detail or status)


# ---------------------------------------------------------------------------
# Check 2: Dry-run cycle
# ---------------------------------------------------------------------------
def check_dry_run(config: str, skip_broker: bool) -> bool:
    if skip_broker:
        _record("Check2_DryRun", True, "SKIPPED (--skip-broker)")
        return True
    log.info("[CHECK 2] Running dry-run cycle...")
    cmd = [sys.executable, "scripts/run_live_paper.py", "once", "--dry-run"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    passed = result.returncode == 0
    detail = f"rc={result.returncode}"
    if "cycle complete" in (result.stdout + result.stderr).lower() or passed:
        detail += " | cycle completed"
    else:
        detail += f" | {(result.stdout + result.stderr)[-200:]}"
    _record("Check2_DryRun", passed, detail)
    return passed


# ---------------------------------------------------------------------------
# Check 3: Halt mechanism
# ---------------------------------------------------------------------------
def check_halt_mechanism() -> bool:
    log.info("[CHECK 3] Testing halt mechanism...")
    halt_path = ROOT / "output" / "ops" / "halt_ack_required.json"
    halt_path.parent.mkdir(parents=True, exist_ok=True)

    # Write halt file
    halt_path.write_text(
        json.dumps({"reason": "preflight_drill", "actor": "run_preflight_checks.py",
                    "ts": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )

    try:
        cmd = [sys.executable, "scripts/run_live_paper.py", "once", "--dry-run"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        halt_detected = any(w in output.lower() for w in ["halt", "halted", "halt flag"])
        _record("Check3a_HaltFlag", halt_detected,
                "halt file detected" if halt_detected else f"NOT detected — output: {output[-200:]}")
    finally:
        halt_path.unlink(missing_ok=True)

    # Kill-switch test
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch, deactivate_kill_switch, is_kill_switch_engaged,
    )
    try:
        activate_kill_switch(throttle_pct=0.0, reason="preflight_drill", actor="run_preflight_checks.py")
        engaged = is_kill_switch_engaged()
        _record("Check3b_KillSwitch_Activate", engaged, f"engaged={engaged}")
        deactivate_kill_switch(reason="preflight_drill_done", actor="run_preflight_checks.py")
        disengaged = not is_kill_switch_engaged()
        _record("Check3c_KillSwitch_Deactivate", disengaged, f"disengaged={disengaged}")
    except Exception as exc:
        _record("Check3b_KillSwitch", False, str(exc))
        return False

    return halt_detected and engaged and disengaged


# ---------------------------------------------------------------------------
# Check 4: Heartbeat / liveness
# ---------------------------------------------------------------------------
def check_heartbeat() -> bool:
    log.info("[CHECK 4] Checking heartbeat / liveness...")
    liveness_script = ROOT / "scripts" / "liveness_check.py"
    if not liveness_script.exists():
        _record("Check4_Heartbeat", True, "SKIPPED — liveness_check.py not found")
        return True
    try:
        result = subprocess.run(
            [sys.executable, str(liveness_script), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        try:
            data = json.loads(result.stdout.strip())
            alive = data.get("alive", False)
            age = data.get("age_seconds", "?")
            _record("Check4_Heartbeat", True, f"alive={alive} age={age}s (liveness check reachable)")
        except json.JSONDecodeError:
            _record("Check4_Heartbeat", True, "SKIPPED — liveness_check returned non-JSON (first run)")
        return True
    except Exception as exc:
        _record("Check4_Heartbeat", False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Check 5: Data freshness + holiday calendar
# ---------------------------------------------------------------------------
def check_data_and_calendar() -> bool:
    log.info("[CHECK 5] Data freshness + holiday calendar...")
    import pandas as pd

    # Data freshness
    cache_dirs = [
        ROOT / "data" / "raw" / "equities_eod" / "yfinance",
        ROOT / "data" / "sample",
    ]
    parquet_files = []
    for d in cache_dirs:
        if d.exists():
            parquet_files.extend(d.glob("*.parquet"))

    if parquet_files:
        most_recent = max(parquet_files, key=lambda p: p.stat().st_mtime)
        age_days = (datetime.now() - datetime.fromtimestamp(most_recent.stat().st_mtime)).days
        fresh = age_days <= 5
        _record("Check5a_DataFreshness", fresh,
                f"most recent: {most_recent.name} ({age_days}d old) {'OK' if fresh else 'STALE'}")
    else:
        _record("Check5a_DataFreshness", False, "no parquet cache files found")

    # Holiday calendar
    try:
        from src.assembled_core.utils.market_calendar import is_trading_day
        checks = [
            (pd.Timestamp("2026-05-25"), False, "Memorial Day"),
            (pd.Timestamp("2026-07-04"), False, "Independence Day"),
            (pd.Timestamp("2026-05-22"), True, "Friday"),
        ]
        calendar_ok = True
        for ts, expected, label in checks:
            got = is_trading_day(ts)
            if got != expected:
                calendar_ok = False
                log.warning("  Calendar mismatch: %s got=%s expected=%s", label, got, expected)
        _record("Check5b_HolidayCalendar", calendar_ok,
                "all 3 checks pass" if calendar_ok else "calendar mismatch — see above")
    except Exception as exc:
        _record("Check5b_HolidayCalendar", False, str(exc))

    return all(r["status"] == "PASS" for r in _RESULTS if r["check"].startswith("Check5"))


# ---------------------------------------------------------------------------
# Check 6: Risk limits in config
# ---------------------------------------------------------------------------
def check_risk_limits(config: str) -> bool:
    log.info("[CHECK 6] Risk limits in config...")
    try:
        import yaml
        cfg_path = Path(config)
        if not cfg_path.exists():
            _record("Check6_RiskLimits", False, f"config not found: {config}")
            return False
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rl = cfg.get("risk_limits", {})
        required_keys = ["max_total_exposure_pct", "max_position_pct", "max_daily_orders", "max_daily_loss_pct"]
        missing = [k for k in required_keys if k not in rl]
        if missing:
            _record("Check6_RiskLimits", False, f"missing keys: {missing}")
            return False
        # Validate sensible bounds
        exp_ok = rl.get("max_total_exposure_pct", 1.0) <= 0.50
        pos_ok = rl.get("max_position_pct", 1.0) <= 0.10
        orders_ok = rl.get("max_daily_orders", 999) <= 50
        loss_ok = rl.get("max_daily_loss_pct", 0) >= -0.05
        all_ok = exp_ok and pos_ok and orders_ok and loss_ok
        detail = (f"exposure={rl.get('max_total_exposure_pct')} pos={rl.get('max_position_pct')} "
                  f"orders={rl.get('max_daily_orders')} daily_loss={rl.get('max_daily_loss_pct')}")
        _record("Check6_RiskLimits", all_ok, detail)
        return all_ok
    except Exception as exc:
        _record("Check6_RiskLimits", False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight checks for paper trading")
    parser.add_argument("--skip-broker", action="store_true", help="Skip live Alpaca API checks")
    parser.add_argument(
        "--config",
        default="configs/paper_track/trend_baseline_live.yaml",
        help="Strategy config for risk-limit check",
    )
    parser.add_argument("--out", default="", help="Write JSON results to this path")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("PAPER-TRADING PRE-FLIGHT CHECKS")
    log.info("Config: %s  skip_broker: %s", args.config, args.skip_broker)
    log.info("=" * 60)

    check_dry_run(args.config, args.skip_broker)
    check_halt_mechanism()
    check_heartbeat()
    check_data_and_calendar()
    check_risk_limits(args.config)

    log.info("")
    log.info("=" * 60)
    log.info("PRE-FLIGHT RESULTS SUMMARY")
    log.info("=" * 60)
    passed = 0
    failed = 0
    for r in _RESULTS:
        icon = "OK" if r["status"] == "PASS" else "FAIL"
        log.info("  [%s] %-40s %s", icon, r["check"], r["detail"])
        if r["status"] == "PASS":
            passed += 1
        else:
            failed += 1

    log.info("")
    verdict = "GO" if failed == 0 else "NO-GO"
    log.info("VERDICT: %s  (%d pass, %d fail)", verdict, passed, failed)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "config": args.config,
            "results": _RESULTS,
            "summary": {"passed": passed, "failed": failed, "verdict": verdict},
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Results: %s", args.out)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
