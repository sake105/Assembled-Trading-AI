"""Paper-trading live runner — real-time Alpaca bars (Plan 11/10 §3.1.1).

Runs continuously during market hours. Pulls the latest bar snapshot from
Alpaca every N minutes and triggers the full trading cycle. Designed for
the 30-day pilot (run_paper_pilot.py --run-day) before going live.

Prerequisites:
    ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env or environment.

Usage:
    python scripts/run_paper_live.py --symbols AAPL,MSFT,NVDA --cycle-min 15
    python scripts/run_paper_live.py --once  # one cycle then exit (for pilot runner)
    python scripts/run_paper_live.py --dry-run  # no order submission
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("run_paper_live")

_DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "AVGO",
    "TSLA",
    "TSM",
    "AMD",
]


# NYSE market holidays 2026-2027 (hardcoded; refresh annually)
_NYSE_HOLIDAYS = frozenset(
    [
        # 2026
        "2026-01-01",  # New Year's Day
        "2026-01-19",  # MLK Day
        "2026-02-16",  # Presidents Day
        "2026-04-03",  # Good Friday
        "2026-05-25",  # Memorial Day
        "2026-06-19",  # Juneteenth
        "2026-07-03",  # Independence Day (observed, Jul 4 = Sat)
        "2026-09-07",  # Labor Day
        "2026-11-26",  # Thanksgiving
        "2026-12-25",  # Christmas
        # 2027
        "2027-01-01",  # New Year's Day
        "2027-01-18",  # MLK Day
        "2027-02-15",  # Presidents Day
        "2027-03-26",  # Good Friday
        "2027-05-31",  # Memorial Day
        "2027-06-18",  # Juneteenth (observed, Jun 19 = Sat)
        "2027-07-05",  # Independence Day (observed, Jul 4 = Sun)
        "2027-09-06",  # Labor Day
        "2027-11-25",  # Thanksgiving
        "2027-12-24",  # Christmas (observed, Dec 25 = Sat)
    ]
)


def _is_market_hours(now: datetime) -> bool:
    """Return True if *now* falls within NYSE regular trading hours (09:30–16:00 ET).

    Uses ``pandas_market_calendars`` when available; falls back to the hardcoded
    ``_NYSE_HOLIDAYS`` frozenset if the package is not importable.
    """
    try:
        import pandas as pd
        import pandas_market_calendars as mcal  # type: ignore[import]

        nyse = mcal.get_calendar("NYSE")
        date_str = now.strftime("%Y-%m-%d")
        schedule = nyse.schedule(start_date=date_str, end_date=date_str)
        if schedule.empty:
            return False
        market_open = schedule.iloc[0]["market_open"]
        market_close = schedule.iloc[0]["market_close"]
        import datetime as _dt

        now_utc = (
            now if now.tzinfo is not None else now.replace(tzinfo=_dt.timezone.utc)
        )
        now_ts = pd.Timestamp(now_utc)
        return bool(market_open <= now_ts < market_close)
    except Exception:
        # Fallback: hardcoded holiday list + simple ET offset logic
        try:
            from zoneinfo import ZoneInfo

            et = now.astimezone(ZoneInfo("America/New_York"))
        except ImportError:
            import datetime as _dt

            et_offset = _dt.timezone(_dt.timedelta(hours=-4))
            et = now.astimezone(et_offset)
        if et.weekday() >= 5:
            return False
        if et.strftime("%Y-%m-%d") in _NYSE_HOLIDAYS:
            return False
        return (et.hour, et.minute) >= (9, 30) and (et.hour, et.minute) < (16, 0)


def _write_cycle_summary(cycle_id: str, result: dict) -> None:
    out_dir = Path("output/paper_live") / datetime.now(timezone.utc).strftime(
        "%Y-%m-%d"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cycle_{cycle_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("[live] cycle summary -> %s", out_path)


def _run_one_cycle(symbols: list[str], dry_run: bool) -> dict:
    """Run one trading cycle using latest Alpaca bar snapshot."""
    from src.assembled_core.execution.kill_switch import is_kill_switch_engaged

    if is_kill_switch_engaged():
        logger.critical("[live] kill-switch engaged — skipping cycle")
        return {"status": "halted", "reason": "kill_switch"}

    # Check halt file
    halt_path = Path("output/ops/halt_ack_required.json")
    if halt_path.exists():
        logger.warning("[live] halt flag present — skipping cycle")
        return {"status": "halted", "reason": "halt_flag"}

    try:
        from src.assembled_core.execution.broker_adapter import AlpacaAdapter

        adapter = AlpacaAdapter.from_env()
    except Exception as exc:
        logger.error("[live] broker adapter init failed: %s", exc)
        try:
            from src.assembled_core.ops.alerting import AlertManager

            AlertManager().fire("broker_connection_failure", {"error": str(exc)[:200]})
        except Exception:
            pass
        return {"status": "error", "error": str(exc)}

    # Pull latest bars from Alpaca
    try:
        prices_latest = adapter.get_latest_bars(symbols)
        if prices_latest is None or prices_latest.empty:
            logger.warning("[live] no bars returned from broker — skipping cycle")
            return {"status": "skip", "reason": "no_bars"}
    except AttributeError:
        # Adapter doesn't have get_latest_bars — use existing paper runner path
        logger.info(
            "[live] get_latest_bars not available — falling back to paper_runner path"
        )
        return _run_via_paper_runner(symbols, dry_run)
    except Exception as exc:
        logger.error("[live] bar fetch failed: %s", exc)
        return {"status": "error", "error": str(exc)}

    return _run_via_paper_runner(symbols, dry_run)


def _run_via_paper_runner(symbols: list[str], dry_run: bool) -> dict:
    """Delegate to paper_runner for a single cycle (uses existing ledger state)."""
    import subprocess

    cmd = [sys.executable, "scripts/run_live_paper.py", "--once"]
    if dry_run:
        cmd.append("--dry-run")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    rc = result.returncode
    output = result.stdout + result.stderr
    return {
        "status": "ok" if rc == 0 else "error",
        "rc": rc,
        "n_orders": output.count("order submitted") + output.count("filled"),
        "output_snippet": output[-500:],
    }


async def paper_live_loop(
    symbols: list[str], cycle_minutes: int, dry_run: bool
) -> None:
    """Async loop: run one cycle every N minutes during market hours."""
    logger.info(
        "[live] starting paper-live loop: %d symbols, cycle=%dmin, dry_run=%s",
        len(symbols),
        cycle_minutes,
        dry_run,
    )

    while True:
        now = datetime.now(timezone.utc)
        if not _is_market_hours(now):
            logger.info("[live] outside market hours — sleeping 5min")
            await asyncio.sleep(300)
            continue

        cycle_id = now.strftime("%H%M%S")
        logger.info("[live] cycle start %s", cycle_id)

        summary = _run_one_cycle(symbols, dry_run)
        summary["cycle_id"] = cycle_id
        summary["timestamp"] = now.isoformat()
        _write_cycle_summary(cycle_id, summary)

        logger.info("[live] cycle %s done: status=%s", cycle_id, summary.get("status"))
        await asyncio.sleep(cycle_minutes * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paper-live runner")
    parser.add_argument("--symbols", default=",".join(_DEFAULT_SYMBOLS))
    parser.add_argument("--cycle-min", type=int, default=15)
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    if args.once:
        result = _run_one_cycle(symbols, args.dry_run)
        logger.info("[live] single cycle result: %s", result)
        return 0 if result.get("status") in ("ok", "halted", "skip") else 1

    asyncio.run(paper_live_loop(symbols, args.cycle_min, args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
