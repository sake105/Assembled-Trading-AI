"""A2 / OPS-03 — Scheduler heartbeat health monitor.

Reads the production heartbeat (default ``output/state/heartbeat.json`` — the
file the *deployed* daily pilot writes via
``src/assembled_core/ops/heartbeat.py::write_heartbeat`` in ``_tc_execution``)
and exits non-zero when its timestamp is stale beyond the configured threshold
during market hours.

The path is overridable via ``--heartbeat-path`` or the ``SCHEDULER_HEARTBEAT_PATH``
environment variable so the same detector can also watch the long-lived
daemon's heartbeat (``output/ops/scheduler_heartbeat.json``) or a synthetic
drill heartbeat. Two heartbeat schemas coexist in this repo (the production
file uses ``timestamp``; the daemon/drill file uses ``timestamp_utc``), so the
parser accepts **both** field names — full writer-path/schema unification is a
separate task (OPS-02).

Market-hours definition (America/New_York):
  Mon–Fri, 09:30–16:00 ET.

Default staleness threshold: 10 minutes. The deployed pilot writes once per
day, so the production wrapper (``scripts/check_scheduler_health.bat``) runs
this with ``--ignore-market-hours`` and a day-scale ``--stale-minutes`` after
the pilot window: a healthy day shows an age of minutes, any stall shows ≥ a
full day.

Usage (as cron / Task Scheduler / systemd-timer job):
    python scripts/check_scheduler_health.py
    python scripts/check_scheduler_health.py --heartbeat-path output/ops/scheduler_heartbeat.json
    # exit code 0 = healthy (or out-of-hours, silent no-op)
    # exit code 1 = stale during market hours
    # exit code 2 = heartbeat file missing / unparseable during market hours

With ``--notify`` a non-zero result is posted to ``DISCORD_WEBHOOK`` (with SMTP
email fallback) via ``assembled_core.ops.alert_sinks`` — the same human-visible
path the weekly drill keeps warm.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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

# Make ``src`` importable when run directly as ``python scripts/...`` (sys.path[0]
# is the scripts/ dir, not the repo root). Only needed for the optional --notify
# path, but harmless otherwise.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Default = the heartbeat the DEPLOYED daily pilot actually writes
# (``_tc_execution`` -> ``output/state/heartbeat.json``). The legacy daemon path
# ``output/ops/scheduler_heartbeat.json`` is still reachable via --heartbeat-path.
DEFAULT_HEARTBEAT_PATH = Path("output/state/heartbeat.json")
DEFAULT_STALE_MINUTES = 10


def _resolve_heartbeat_path(cli_value: str | None) -> Path:
    """Heartbeat path precedence: --heartbeat-path > env > deployed default."""
    if cli_value:
        return Path(cli_value)
    env_value = os.environ.get("SCHEDULER_HEARTBEAT_PATH", "").strip()
    if env_value:
        return Path(env_value)
    return DEFAULT_HEARTBEAT_PATH


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
    # Accept both schemas: daemon/drill write ``timestamp_utc``; the deployed
    # pilot (ops/heartbeat.py) writes ``timestamp``. Prefer the explicit-UTC
    # field when present.
    ts = payload.get("timestamp_utc") or payload.get("timestamp")
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


def _emit_alert(message: str) -> None:
    """Best-effort human-visible alert via the shared sinks (Discord + email).

    Never raises: a delivery failure must not change the monitor's exit code.
    """
    try:
        from src.assembled_core.ops.alert_sinks import (
            post_discord,
            post_email_fallback,
        )

        webhook = os.environ.get("DISCORD_WEBHOOK", "").strip()
        delivered = post_discord(webhook, message) if webhook else False
        if not delivered:
            post_email_fallback("[Scheduler Health] stale heartbeat", message)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[HEALTH] alert dispatch failed: %s", exc)


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
    parser.add_argument(
        "--heartbeat-path",
        default=None,
        help=(
            "Heartbeat file to check. Overrides $SCHEDULER_HEARTBEAT_PATH and the "
            "deployed default (output/state/heartbeat.json)."
        ),
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="On a non-zero result, post to DISCORD_WEBHOOK (with email fallback).",
    )
    args = parser.parse_args(argv)

    heartbeat_path = _resolve_heartbeat_path(args.heartbeat_path)

    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_ET)
    in_hours = _in_market_hours(now_et)

    if not in_hours and not args.ignore_market_hours:
        logger.info(
            "[HEALTH] %s ET — out of market hours, silent ok", now_et.isoformat()
        )
        return 0

    def _fail(rc: int, message: str) -> int:
        logger.error("%s", message)
        if args.notify:
            _emit_alert(f"[ALERT] scheduler health: {message}")
        return rc

    if not heartbeat_path.exists():
        return _fail(2, f"heartbeat missing at {heartbeat_path} during market hours")

    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _fail(2, f"heartbeat unparseable at {heartbeat_path}: {exc}")

    hb_dt = _parse_heartbeat(payload)
    if hb_dt is None:
        return _fail(2, f"heartbeat has no usable timestamp at {heartbeat_path}")

    age_seconds = (now_utc - hb_dt).total_seconds()
    age_minutes = age_seconds / 60.0
    limit_seconds = args.stale_minutes * 60

    if age_seconds > limit_seconds:
        return _fail(
            1,
            f"heartbeat stale: age={age_minutes:.1f} min > limit={args.stale_minutes} "
            f"min (status={payload.get('status')}, path={heartbeat_path})",
        )

    logger.info(
        "[HEALTH] heartbeat ok: age=%.1f min, status=%s",
        age_minutes,
        payload.get("status"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
