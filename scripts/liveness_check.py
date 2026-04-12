"""Liveness check CLI (Plan C16).

Reads the heartbeat file written by the trading cycle and reports whether
the system is alive. Designed for external monitoring (cron, systemd timer,
Nagios/Zabbix custom check, etc.).

Exit codes:
    0 — system alive
    1 — system NOT alive (stale, halted, or missing heartbeat)
    2 — usage / argument error

Usage:
    python scripts/liveness_check.py
    python scripts/liveness_check.py --max-age 1800
    python scripts/liveness_check.py --heartbeat-path output/state/heartbeat.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.ops.heartbeat import check_liveness  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check trading system liveness via heartbeat file.",
    )
    parser.add_argument(
        "--heartbeat-path",
        default=None,
        help="Path to heartbeat.json (default: output/state/heartbeat.json)",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=900.0,
        help="Maximum acceptable age in seconds (default: 900)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output result as JSON instead of human-readable text",
    )
    args = parser.parse_args(argv)

    result = check_liveness(path=args.heartbeat_path, max_age_seconds=args.max_age)

    if args.json_output:
        print(json.dumps(result, indent=2))
    else:
        status = "ALIVE" if result["alive"] else "NOT ALIVE"
        age = result.get("age_seconds")
        age_str = f"{age:.0f}s" if age is not None else "n/a"
        reason = result.get("reason", "")
        print(f"[{status}] age={age_str} status={result.get('status', 'n/a')} reason={reason}")

    return 0 if result["alive"] else 1


if __name__ == "__main__":
    sys.exit(main())
