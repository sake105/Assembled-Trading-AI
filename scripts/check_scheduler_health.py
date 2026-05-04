"""A2 — Scheduler heartbeat health monitor.

Reads ``output/ops/scheduler_heartbeat.json`` and exits non-zero when the
timestamp is stale beyond the configured threshold during market hours.

Market-hours definition (America/New_York):
  Mon–Fri, 09:30–16:00 ET.

Default staleness threshold: 10 minutes.

Usage (as cron / Task Scheduler / systemd-timer job):
    python scripts/check_scheduler_health.py
    # exit code 0 = healthy (or out-of-hours, silent no-op)
    # exit code 1 = stale during market hours
    # exit code 2 = heartbeat file missing during market hours

Intended to be composed with a Discord-alert wrapper in CI or cron.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
except ImportError:  # pragma: no cover — Py < 3.9 fallback
    import pytz  # type: ignore

    _ET = pytz.timezone("America/New_York")

logger = logging.getLogger("check_scheduler_health")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

HEARTBEAT_PATH = Path("output/ops/scheduler_heartbeat.json")
DEFAULT_STALE_MINUTES = 10


def _in_market_hours(now_et: datetime) -> bool:
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    start = (9, 30)
    end = (16, 0)
    if (t.hour, t.minute) < start:
        return False
    if (t.hour, t.minute) >= end:
        return False
    return True


def _parse_heartbeat(payload: dict) -> datetime | None:
    ts = payload.get("timestamp_utc")
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as exc:
        logger.warning("[HEALTH] could not parse heartbeat ts=%r: %s", ts, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=DEFAULT_STALE_MINUTES,
        help="Max heartbeat age before flagging stale during market hours.",
    )
    parser.add_argument(
        "--ignore-market-hours",
        action="store_true",
        help="Flag stale even outside market hours (useful for local debug).",
    )
    args = parser.parse_args(argv)

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_ET)
    in_hours = _in_market_hours(now_et)

    if not in_hours and not args.ignore_market_hours:
        logger.info(
            "[HEALTH] %s ET — out of market hours, silent ok", now_et.isoformat()
        )
        return 0

    if not HEARTBEAT_PATH.exists():
        logger.error(
            "[HEALTH] heartbeat missing at %s during market hours", HEARTBEAT_PATH
        )
        return 2

    try:
        payload = json.loads(HEARTBEAT_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("[HEALTH] heartbeat unparseable: %s", exc)
        return 2

    hb_dt = _parse_heartbeat(payload)
    if hb_dt is None:
        return 2

    age_seconds = (now_utc - hb_dt).total_seconds()
    age_minutes = age_seconds / 60.0
    limit_seconds = args.stale_minutes * 60

    if age_seconds > limit_seconds:
        logger.error(
            "[HEALTH] heartbeat stale: age=%.1f min > limit=%d min (status=%s)",
            age_minutes,
            args.stale_minutes,
            payload.get("status"),
        )
        return 1

    logger.info(
        "[HEALTH] heartbeat ok: age=%.1f min, status=%s",
        age_minutes,
        payload.get("status"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
