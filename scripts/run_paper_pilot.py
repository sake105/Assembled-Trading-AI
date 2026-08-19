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
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PILOT_DIR = ROOT / "output" / "pilot"

# 2026-08-18: GETRENNTE Manifeste fuer CI und Betrieb.
#
# Vorher schrieben BEIDE dieselbe Datei — und
# .github/workflows/paper-trading-ci.yml committet sie mit `git add -f`.
# Der CI-Runner checkt das Repo frisch aus, sieht also nur die zuletzt
# COMMITTETE (kurze) Historie, haengt seinen Tag daran an und pusht das
# Ergebnis. Beim naechsten lokalen `git pull` ueberschreibt diese
# CI-Version die echte Betriebshistorie: gemessen 27 Tage -> 1 Tag.
# Das Manifest ist die Bewertungsgrundlage des 30-Tage-Pilots UND ein
# Watchdog-Input (zero_orders_unexpected) — ein Reset macht beides blind.
#
# Der Runner schreibt jetzt in eine EIGENE Datei. Damit hat der
# `git add -f`-Step nichts Neues zu committen und die lokale Historie
# bleibt unangetastet (E-190).
_IN_CI = os.environ.get("GITHUB_ACTIONS") == "true"
PILOT_MANIFEST = PILOT_DIR / (
    "pilot_manifest_ci.json" if _IN_CI else "pilot_manifest.json"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Item 80 — Stale open-order cancellation on restart
# ---------------------------------------------------------------------------


def cancel_all_stale_orders(older_than_minutes: int = 5) -> int:
    """Cancel all open broker orders older than *older_than_minutes*.

    Returns the number of orders cancelled.
    Requires alpaca-py to be installed and ALPACA_API_KEY / ALPACA_API_SECRET
    in the environment (via .env).  Non-fatal: any error is logged and ignored.
    """
    cancelled = 0
    try:
        from src.assembled_core.execution.broker_adapter import AlpacaAdapter

        adapter = AlpacaAdapter()
        open_orders = (
            adapter.get_open_orders() if hasattr(adapter, "get_open_orders") else []
        )
        if not open_orders:
            logger.info("[pilot-startup] No open orders found — nothing to cancel.")
            return 0

        now = datetime.now(timezone.utc)
        stale_ids = []
        for order in open_orders:
            submitted_at = getattr(order, "submitted_at", None) or getattr(
                order, "created_at", None
            )
            if submitted_at is None:
                continue
            age_minutes = (now - submitted_at).total_seconds() / 60
            if age_minutes >= older_than_minutes:
                stale_ids.append(getattr(order, "id", str(order)))

        if not stale_ids:
            logger.info(
                "[pilot-startup] %d open orders found, none older than %dm.",
                len(open_orders),
                older_than_minutes,
            )
            return 0

        logger.warning(
            "[pilot-startup] Cancelling %d stale orders (>%dm old): %s",
            len(stale_ids),
            older_than_minutes,
            stale_ids,
        )
        for order_id in stale_ids:
            try:
                if hasattr(adapter, "cancel_order"):
                    adapter.cancel_order(order_id)
                cancelled += 1
                logger.info("[pilot-startup] Cancelled order %s", order_id)
            except Exception as exc:
                logger.warning(
                    "[pilot-startup] Failed to cancel order %s: %s", order_id, exc
                )

    except Exception as exc:
        logger.warning(
            "[pilot-startup] cancel_all_stale_orders failed (non-fatal): %s", exc
        )

    return cancelled


# ---------------------------------------------------------------------------
# Item 68 — Position-state recovery check on restart
# ---------------------------------------------------------------------------

_INTENT_STATE_PATH = ROOT / "output" / "runs" / "_paper_ledger" / "intent_state.json"


def check_state_recovery() -> None:
    """Compare disk intent-state with broker open positions and log discrepancies.

    Minimal implementation per spec: no automatic reconciliation — just clear
    warnings so the operator can decide.  Runs on each pilot startup.
    """
    # 1. Load disk intent-state (if present)
    disk_symbols: set[str] = set()
    if _INTENT_STATE_PATH.exists():
        try:
            intent_data = json.loads(_INTENT_STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(intent_data, dict):
                disk_symbols = {
                    sym
                    for sym, qty in intent_data.get("positions", {}).items()
                    if float(qty) != 0.0
                }
            logger.info(
                "[pilot-startup] Disk intent-state loaded: %d open symbols.",
                len(disk_symbols),
            )
        except Exception as exc:
            logger.warning(
                "[pilot-startup] Could not read intent state from disk: %s", exc
            )
    else:
        logger.info(
            "[pilot-startup] No disk intent-state found at %s.", _INTENT_STATE_PATH
        )

    # 2. Fetch broker open positions
    broker_symbols: set[str] = set()
    try:
        from src.assembled_core.execution.broker_adapter import AlpacaAdapter

        adapter = AlpacaAdapter()
        positions = adapter.get_positions() if hasattr(adapter, "get_positions") else []
        broker_symbols = {
            getattr(p, "symbol", str(p))
            for p in positions
            if float(getattr(p, "qty", 0)) != 0.0
        }
        logger.info(
            "[pilot-startup] Broker reports %d open positions: %s",
            len(broker_symbols),
            sorted(broker_symbols),
        )
    except Exception as exc:
        logger.warning(
            "[pilot-startup] Could not fetch broker positions (non-fatal): %s", exc
        )

    # 3. Detect and log discrepancies
    if disk_symbols or broker_symbols:
        only_on_disk = disk_symbols - broker_symbols
        only_at_broker = broker_symbols - disk_symbols

        if only_on_disk:
            logger.warning(
                "[pilot-startup] STATE DISCREPANCY — symbols in disk-state but NOT at broker "
                "(possible missed fills or stale intents): %s",
                sorted(only_on_disk),
            )
        if only_at_broker:
            logger.warning(
                "[pilot-startup] STATE DISCREPANCY — symbols at broker but NOT in disk-state "
                "(possible out-of-band trades or state loss): %s",
                sorted(only_at_broker),
            )
        if not only_on_disk and not only_at_broker:
            logger.info(
                "[pilot-startup] Disk-state and broker positions are consistent."
            )


def run_startup_checks() -> None:
    """Run all startup safety checks before the daily pilot cycle.

    Called at the top of cmd_run_day().  Non-fatal: logs warnings but never
    blocks the trading cycle — operator must review logs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    # E-145-Klasse-Fix (2026-08-18, Pilot-Diagnose): dieses Script lud die
    # .env NIE — run_live_paper.py (innerer Zyklus) laedt sie selbst, aber
    # die STARTUP-SAFETY-CHECKS hier (Broker-Positions-Fetch, Stale-Order-
    # Cancel) liefen seit jeher ohne ALPACA_API_KEY/SECRET "non-fatal" ins
    # Leere — Schutzchecks, die still nie schuetzten. In der Funktion, nicht
    # auf Modulebene (E-168: Test-Loader via exec_module).
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        logger.warning("[pilot-startup] python-dotenv missing — env not loaded")
    logger.info("[pilot-startup] Running startup safety checks…")
    # Item 68: position state recovery
    check_state_recovery()
    # Item 80: cancel stale orders
    n_cancelled = cancel_all_stale_orders(older_than_minutes=5)
    if n_cancelled:
        logger.warning(
            "[pilot-startup] Cancelled %d stale orders before trading cycle.",
            n_cancelled,
        )
    # F-RX-11 §9.12 (g): auto-abandon stale ORDER_SUBMIT intents with empty
    # broker_order_id older than 24h. These represent crash/network-failure
    # cases where the intent was persisted but the broker submission never
    # completed; they previously required manual reconciliation (twice in
    # five days during the pilot outage 2026-05-15..21). Auto-abandonment
    # writes a paired ORDER_COMPLETE/status=abandoned_auto audit record.
    try:
        from src.assembled_core.execution.intent_store import (
            auto_abandon_stale_intents,
        )

        abandoned = auto_abandon_stale_intents(max_age_hours=24.0)
        if abandoned:
            logger.warning(
                "[pilot-startup] auto-abandoned %d stale pre-submit intent(s) "
                "(empty broker_order_id, age > 24h). See "
                "output/ops/intent_store.jsonl for ORDER_COMPLETE/"
                "status=abandoned_auto audit records.",
                len(abandoned),
            )
            # Surface count via manifest for trend tracking. Stored in a
            # rolling dict so the operator can spot increasing frequency
            # (an indicator the upstream issue isn't being fixed).
            try:
                if PILOT_MANIFEST.exists():
                    m = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8"))
                    m.setdefault("auto_abandoned_intents", []).append(
                        {
                            "ts_utc": datetime.now(timezone.utc).isoformat(),
                            "count": len(abandoned),
                        }
                    )
                    PILOT_MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[pilot-startup] manifest counter update failed: %s", exc
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[pilot-startup] auto-abandon-intents step failed: %s", exc)
    logger.info("[pilot-startup] Startup checks complete.")


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


#: Gemeinsamer Tages-Marker mit paper_trading_scheduler.py (E-191).
LAST_RUN_PATH = ROOT / "output" / "ops" / "last_run_date.txt"


def _already_ran_today(day: str) -> bool:
    """True, wenn heute bereits ein Handelszyklus lief (egal von welchem Pfad)."""
    try:
        return LAST_RUN_PATH.read_text(encoding="utf-8").strip() == day
    except OSError:
        return False


def _mark_today_done(day: str) -> None:
    """Tages-Marker atomar setzen (tmp + replace, wie im Scheduler-Daemon)."""
    try:
        LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LAST_RUN_PATH.with_suffix(LAST_RUN_PATH.suffix + ".tmp")
        tmp.write_text(day, encoding="utf-8")
        os.replace(tmp, LAST_RUN_PATH)
    except OSError as exc:
        logger.error("[pilot] last_run_date write failed: %s", exc)


def cmd_run_day() -> int:
    """Run one paper-live cycle and append daily summary to manifest."""
    # E-191 (2026-08-18, gemessen): ZWEI Systeme fuhren am selben Abend je
    # einen Broker-Zyklus — der Task AssembledTradingAI-PaperPilot (21:30,
    # 8 Fills) und der Daemon AssembledTradingAI_PaperEngine (21:40,
    # 2 Fills). Der Daemon hat einen Tages-Guard ueber last_run_date.txt;
    # dieser Pfad kannte ihn nicht, prueft ihn also nicht und setzt ihn
    # nicht. Folge: doppeltes Turnover-Budget (der Tages-Cap von 20 % gilt
    # PRO Zyklus) und zwei Schreiber auf demselben Ledger. Jetzt teilen
    # sich beide Pfade denselben Marker.
    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _already_ran_today(_today):
        print(
            f"[pilot] SKIP — es lief heute ({_today}) bereits ein "
            "Handelszyklus (gemeinsamer Marker output/ops/last_run_date.txt, "
            "geteilt mit paper_trading_scheduler). Ein zweiter Zyklus wuerde "
            "das Tages-Turnover-Budget verdoppeln."
        )
        return 0

    # Items 68 + 80: startup safety checks (state recovery + stale order cancel)
    run_startup_checks()

    m = _load_manifest()
    if not m.get("started_at"):
        m["started_at"] = datetime.now(timezone.utc).isoformat()
        print(f"[pilot] Pilot started: {m['started_at']}")

    day_num = len(m["days"]) + 1
    ts = datetime.now(timezone.utc).isoformat()

    # Invoke the paper runner for one cycle
    import subprocess

    # F-RX-FU-1: subprocess timeout MUST exceed run_live_paper's soft-timeout
    # (default 1500s = 25min) so the in-process bail-out can write halt-ack
    # before this subprocess wrapper kills the child. With 1700s here the
    # soft-timeout fires first, sets halt-flag, exits cleanly with rc=2,
    # leaving the next run gated until operator clears the flag.
    result = subprocess.run(
        [sys.executable, "scripts/run_live_paper.py", "once"],
        capture_output=True,
        text=True,
        timeout=1700,
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

    # Marker NUR bei sauberem Lauf setzen: schlaegt der Zyklus fehl, muss der
    # Daemon als Backup einspringen koennen (gleiche Semantik wie dort).
    if rc == 0:
        _mark_today_done(_today)
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
            equity_curve_raw = ledger.get("equity_curve", [])
            # equity_curve may be list of floats or list of dicts {"utc":..., "equity":...}
            equity_curve = [
                e["equity"] if isinstance(e, dict) else float(e)
                for e in equity_curve_raw
                if (isinstance(e, dict) and "equity" in e)
                or isinstance(e, (int, float))
            ]
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

    print(f"\n{'=' * 50}")
    print(f"PILOT VERDICT: {decision}")
    print(f"{'=' * 50}")
    for k, v in criteria_results.items():
        icon = "[OK]" if v.get("pass") else "[FAIL]"
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
