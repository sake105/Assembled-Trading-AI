"""Heartbeat + liveness helpers (Sprint 4 / Plan C16).

File-based heartbeat that external monitoring systems (cron, systemd
watchdog, uptime-kuma, a GitHub Action probe, etc.) can poll without
running any HTTP server. The trading cycle writes a heartbeat file on
every successful run; a separate liveness check reads that file and
decides whether the system is alive based on file age.

The contract is intentionally small so it works on Windows and Linux
identically and so it can be called from tests without touching real
timekeeping:

    write_heartbeat(path, status="ok", details={"invested_pct": 0.82})
    state = read_heartbeat(path)
    age_s = heartbeat_age_seconds(path, now=...)
    res   = check_liveness(path, max_age_seconds=900, now=...)

``check_liveness`` returns a dict with ``alive: bool`` plus diagnostic
fields so it can be consumed by alert sinks (C14) directly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_PATH = Path("output") / "state" / "heartbeat.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def write_heartbeat(
    path: str | Path | None = None,
    *,
    status: str = "ok",
    details: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> Path:
    """Write the current heartbeat snapshot to ``path``. Overwrites.

    Args:
        path: Target file. Defaults to ``output/state/heartbeat.json``.
        status: Free-form status string. Convention: ``ok`` | ``degraded`` | ``halt``.
        details: Optional payload (e.g. KPIs, run id). Must be JSON-serialisable.
        now: Injectable clock for tests. Defaults to ``datetime.now(UTC)``.

    Returns:
        The resolved ``Path`` that was written.
    """
    p = Path(path or _DEFAULT_HEARTBEAT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)

    ts = (now or _now_utc())
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    payload = {
        "status": status,
        "timestamp": ts.isoformat(),
        "details": details or {},
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return p


def read_heartbeat(path: str | Path | None = None) -> dict[str, Any] | None:
    """Read a heartbeat file. Returns ``None`` if missing or unparseable."""
    p = Path(path or _DEFAULT_HEARTBEAT_PATH)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[heartbeat] could not parse %s: %s", p, exc)
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def heartbeat_age_seconds(
    path: str | Path | None = None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return age of the heartbeat file in seconds, or ``None`` if missing.

    Age is computed from the ``timestamp`` field inside the file rather
    than mtime, so it survives file copies and is timezone-safe.
    """
    data = read_heartbeat(path)
    if not data:
        return None
    ts_str = data.get("timestamp")
    if not isinstance(ts_str, str):
        return None
    try:
        ts = datetime.fromisoformat(ts_str)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    current = now or _now_utc()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (current - ts).total_seconds())


def check_liveness(
    path: str | Path | None = None,
    *,
    max_age_seconds: float = 900.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate whether the system is live.

    A system is considered alive if:

    * the heartbeat file exists
    * the ``status`` field is not ``halt``
    * the age in seconds is below ``max_age_seconds``

    Returns a diagnostic dict with ``alive``, ``reason``, ``age_seconds``,
    ``status``, and ``path``.
    """
    p = Path(path or _DEFAULT_HEARTBEAT_PATH)
    data = read_heartbeat(p)
    if data is None:
        return {
            "alive": False,
            "reason": "missing_or_unreadable",
            "age_seconds": None,
            "status": None,
            "path": str(p),
        }

    status = str(data.get("status", "")) or "unknown"
    age = heartbeat_age_seconds(p, now=now)

    if status == "halt":
        return {
            "alive": False,
            "reason": "status_halt",
            "age_seconds": age,
            "status": status,
            "path": str(p),
        }
    if age is None:
        return {
            "alive": False,
            "reason": "unparseable_timestamp",
            "age_seconds": None,
            "status": status,
            "path": str(p),
        }
    if age > float(max_age_seconds):
        return {
            "alive": False,
            "reason": f"stale:{age:.0f}s>{max_age_seconds:.0f}s",
            "age_seconds": age,
            "status": status,
            "path": str(p),
        }

    return {
        "alive": True,
        "reason": "ok",
        "age_seconds": age,
        "status": status,
        "path": str(p),
    }


__all__ = [
    "write_heartbeat",
    "read_heartbeat",
    "heartbeat_age_seconds",
    "check_liveness",
]
