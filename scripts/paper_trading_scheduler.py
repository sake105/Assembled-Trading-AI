"""Autonomous Paper Trading Scheduler.

Runs as a long-lived process, automatically executing the daily paper trading
cycle at the configured time (default: 15:30 ET, 30 min before NYSE close).

Usage:
  python scripts/paper_trading_scheduler.py              # run with defaults
  python scripts/paper_trading_scheduler.py --hour 10    # run at 10:00 ET
  python scripts/paper_trading_scheduler.py --test       # immediate test run

Press Ctrl+C to stop gracefully.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

HEARTBEAT_PATH = ROOT / "output" / "ops" / "scheduler_heartbeat.json"
LAST_RUN_PATH = ROOT / "output" / "ops" / "last_run_date.txt"
LOCK_PATH = ROOT / "output" / "ops" / ".paper_trading_lock"
CHECK_INTERVAL = 300  # 5 minutes

_running = True


def _signal_handler(signum, frame):
    global _running
    logger.info("[Scheduler] Shutdown requested (signal %d)", signum)
    _running = False


def _get_eastern_now() -> datetime:
    """Get current time in US/Eastern (handles EST/EDT)."""
    try:
        from zoneinfo import ZoneInfo
        from datetime import timezone as tz

        utc_now = datetime.now(tz.utc)
        return utc_now.astimezone(ZoneInfo("America/New_York"))
    except ImportError:
        from datetime import timedelta, timezone as tz

        utc_now = datetime.now(tz.utc)
        return utc_now.astimezone(tz(timedelta(hours=-5)))


def _is_trading_day(dt: datetime) -> bool:
    """Check if date is a weekday (simple check; ignores holidays)."""
    return dt.weekday() < 5


def _already_ran_today(dt: datetime) -> bool:
    """Check if we already ran today."""
    today = dt.strftime("%Y-%m-%d")
    if LAST_RUN_PATH.exists():
        try:
            last = LAST_RUN_PATH.read_text(encoding="utf-8").strip()
            return last == today
        except Exception:
            pass
    return False


def _mark_today_done(dt: datetime) -> None:
    """Mark today as completed (atomic tmp+replace).

    Non-atomic write could produce an empty/truncated file on crash, which
    would cause ``_already_ran_today`` to return False and re-trigger a
    duplicate trading cycle — exactly the failure mode the marker exists to
    prevent.
    """
    import os as _os

    LAST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LAST_RUN_PATH.with_suffix(LAST_RUN_PATH.suffix + ".tmp")
    try:
        tmp.write_text(dt.strftime("%Y-%m-%d"), encoding="utf-8")
        _os.replace(tmp, LAST_RUN_PATH)
    except Exception as exc:
        logger.error("[Scheduler] last_run_date write failed: %s", exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


def _write_heartbeat(status: str = "alive") -> None:
    """Write heartbeat file for monitoring (atomic + timezone-aware).

    Uses tmp+replace to avoid producing a truncated JSON file if the process
    dies mid-write; a truncated file is read back as ``None`` by the health
    monitor, which would mask a dead scheduler as a merely missing file.
    """
    import os as _os

    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": _os.getpid(),
    })
    tmp = HEARTBEAT_PATH.with_suffix(HEARTBEAT_PATH.suffix + ".tmp")
    try:
        tmp.write_text(payload, encoding="utf-8")
        _os.replace(tmp, HEARTBEAT_PATH)
    except Exception as exc:
        logger.error("[Scheduler] heartbeat write failed: %s", exc)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _run_price_update() -> bool:
    """Run price cache update. Returns True on success."""
    logger.info("[Scheduler] Updating price cache...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "update_prices.py"), "--days", "10"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(ROOT),
        )
        if result.returncode == 0:
            logger.info("[Scheduler] Price update OK")
            return True
        else:
            logger.warning(
                "[Scheduler] Price update failed (exit %d): %s",
                result.returncode,
                result.stderr[-500:] if result.stderr else "",
            )
            return False
    except Exception as exc:
        logger.warning("[Scheduler] Price update error: %s", exc)
        return False


def _run_trading_cycle() -> int:
    """Run the paper trading cycle. Returns exit code."""
    logger.info("[Scheduler] Running paper trading cycle...")
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_live_paper.py"), "once"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(ROOT),
        )
        logger.info("[Scheduler] Trading cycle exit code: %d", result.returncode)
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-10:]:
                logger.info("[Cycle] %s", line)
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                logger.warning("[Cycle] %s", line)
        return result.returncode
    except Exception as exc:
        logger.error("[Scheduler] Trading cycle error: %s", exc)
        return 1


def _execute_daily_cycle(now_et: datetime) -> None:
    """Execute the full daily cycle: prices -> trade -> mark done."""
    logger.info(
        "[Scheduler] === Starting daily cycle at %s ET ===",
        now_et.strftime("%H:%M:%S"),
    )
    _write_heartbeat("running_cycle")

    # Step 1: Update prices (non-blocking if fails)
    _run_price_update()

    # Step 2: Run trading cycle
    exit_code = _run_trading_cycle()

    # Step 3: Mark done — ONLY on success. Without this guard, a crashed
    # trading cycle is indistinguishable from a successful one to
    # _already_ran_today, so the scheduler would skip the retry window on
    # the next polling pass. This is exactly the E3 (7-day stillstand)
    # failure mode the ops audit flagged.
    if exit_code == 0:
        _mark_today_done(now_et)
        _write_heartbeat("cycle_complete")
        logger.info("[Scheduler] === Daily cycle COMPLETE (success) ===")
    else:
        # Heartbeat still written so monitoring sees liveness, but with
        # explicit failure state. last_run_date is NOT advanced so the
        # scheduler will retry on its next poll.
        _write_heartbeat("cycle_failed")
        logger.warning(
            "[Scheduler] === Daily cycle FAILED (exit=%d) — will retry ===",
            exit_code,
        )


def main():
    parser = argparse.ArgumentParser(description="Autonomous Paper Trading Scheduler")
    parser.add_argument(
        "--hour", type=int, default=15, help="ET hour to execute (default: 15 = 3 PM)"
    )
    parser.add_argument(
        "--minute", type=int, default=30, help="ET minute to execute (default: 30)"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run one cycle immediately and exit"
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(
        "[Scheduler] Starting — execution time: %02d:%02d ET, check every %ds",
        args.hour,
        args.minute,
        CHECK_INTERVAL,
    )

    if args.test:
        logger.info("[Scheduler] Test mode — running cycle now")
        _execute_daily_cycle(_get_eastern_now())
        return

    _write_heartbeat("started")

    while _running:
        try:
            now_et = _get_eastern_now()
            _write_heartbeat("alive")

            if (
                _is_trading_day(now_et)
                and now_et.hour == args.hour
                and now_et.minute >= args.minute
                and not _already_ran_today(now_et)
            ):
                _execute_daily_cycle(now_et)
            else:
                if now_et.minute == 0:  # Log status once per hour
                    status = "waiting"
                    if not _is_trading_day(now_et):
                        status = "weekend"
                    elif _already_ran_today(now_et):
                        status = "already_ran"
                    logger.info(
                        "[Scheduler] %s — %s ET, next check in %ds",
                        status,
                        now_et.strftime("%Y-%m-%d %H:%M"),
                        CHECK_INTERVAL,
                    )

            # Sleep until next check
            for _ in range(CHECK_INTERVAL):
                if not _running:
                    break
                time.sleep(1)

        except Exception as exc:
            logger.error("[Scheduler] Unexpected error: %s", exc)
            time.sleep(60)

    _write_heartbeat("stopped")
    logger.info("[Scheduler] Stopped gracefully")


if __name__ == "__main__":
    main()
