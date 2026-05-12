#!/usr/bin/env python
"""Data-source health check (audit C3-061).

Pings the public health/status endpoints of the data sources we depend
on and emits a single JSON record summarising up/down/degraded per
provider. Designed to run as a daily cron job; the output JSON is the
input for any operator dashboard.

Run::

    python scripts/check_data_sources_health.py [--output PATH]

Exit code:
    0 — every provider returned a usable response.
    1 — at least one critical provider was unreachable.

The check is **best-effort and read-only** — it sends one HEAD/GET
request per provider with a 5 s timeout, never authenticates, never
mutates state. Network failure on a single provider does not abort the
others.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("check_data_sources_health")

# (name, url, critical?) — "critical" providers contribute to exit code 1.
_PROVIDERS: list[tuple[str, str, bool]] = [
    ("yahoo_finance", "https://finance.yahoo.com/", True),
    ("alpaca", "https://api.alpaca.markets/", True),
    ("polygon", "https://api.polygon.io/", False),
    ("fred", "https://fred.stlouisfed.org/", False),
    ("sec_edgar", "https://www.sec.gov/edgar/searchedgar/companysearch", False),
    ("ntp_pool", "https://www.pool.ntp.org/en/", False),
]

_TIMEOUT_SEC = 5.0


def _probe(url: str, timeout: float = _TIMEOUT_SEC) -> dict[str, Any]:
    """Single HTTP GET, return up/down/latency. Never raises."""
    started = time.monotonic()
    try:
        req = urllib.request.Request(
            url, method="GET", headers={"User-Agent": "assembled-trading-ai/health"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            elapsed_ms = (time.monotonic() - started) * 1000.0
            return {
                "up": 200 <= status < 400,
                "http_status": status,
                "latency_ms": round(elapsed_ms, 1),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.monotonic() - started) * 1000.0
        # HTTPError is also a Response — treat as degraded, not down.
        return {
            "up": False,
            "http_status": exc.code,
            "latency_ms": round(elapsed_ms, 1),
            "error": f"http-error: {exc.code}",
        }
    except Exception as exc:  # noqa: BLE001 — any failure is "down"
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return {
            "up": False,
            "http_status": None,
            "latency_ms": round(elapsed_ms, 1),
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def run_health_check() -> dict[str, Any]:
    """Probe every provider, return a summary dict."""
    started = datetime.now(timezone.utc).isoformat()
    results: dict[str, Any] = {}
    n_down_critical = 0
    for name, url, critical in _PROVIDERS:
        rec = _probe(url)
        rec["critical"] = critical
        rec["url"] = url
        results[name] = rec
        if critical and not rec["up"]:
            n_down_critical += 1
            logger.warning("[health] CRITICAL down: %s — %s", name, rec["error"])
        elif not rec["up"]:
            logger.info("[health] non-critical degraded: %s — %s", name, rec["error"])
        else:
            logger.info(
                "[health] %s OK (HTTP %s, %.1fms)",
                name,
                rec["http_status"],
                rec["latency_ms"],
            )
    return {
        "checked_at": started,
        "results": results,
        "n_providers": len(_PROVIDERS),
        "n_down_critical": n_down_critical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="output/ops/data_sources_health.json",
        help="Where to write the JSON summary (default: output/ops/data_sources_health.json)",
    )
    args = parser.parse_args()

    summary = run_health_check()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("[health] wrote summary to %s", out_path)

    return 1 if summary["n_down_critical"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
