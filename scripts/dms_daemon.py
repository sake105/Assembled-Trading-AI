#!/usr/bin/env python
"""Dead-Man's Switch daemon — passive heartbeat monitor.

Loads configs/policy.yaml, checks policy.dead_man_switch.enabled, then
enters the monitor loop that auto-flattens all positions when the trading
cycle heartbeat goes stale.

Usage:
    python scripts/dms_daemon.py
    python scripts/dms_daemon.py --policy configs/policy.yaml

Exit codes:
    0 — clean shutdown (SIGTERM, KeyboardInterrupt, or disabled via policy)
    1 — startup failure (bad policy, missing file, etc.)
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap so the script works both via `python scripts/dms_daemon.py`
# and via the installed package.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import yaml  # type: ignore[import]
except ImportError:
    print(
        "[DMS-START] ERROR: PyYAML not installed. Run: pip install pyyaml",
        file=sys.stderr,
    )
    sys.exit(1)

from src.assembled_core.ops.dead_man_switch import dms_monitor_loop  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("dms_daemon")

_DEFAULT_POLICY = "configs/policy.yaml"


def _load_policy(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        logger.error("[DMS-START] Policy file not found: %s", p)
        sys.exit(1)
    try:
        with open(p, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("[DMS-START] Failed to parse policy file %s: %s", p, exc)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dead-Man's Switch daemon")
    parser.add_argument(
        "--policy",
        default=os.environ.get("ASSEMBLED_POLICY_PATH", _DEFAULT_POLICY),
        help="Path to policy.yaml (default: configs/policy.yaml)",
    )
    args = parser.parse_args()

    pid = os.getpid()
    logger.info("[DMS-START] PID=%d policy=%s", pid, args.policy)

    policy = _load_policy(args.policy)

    dms_cfg = policy.get("dead_man_switch", {}) or {}
    if not dms_cfg.get("enabled", True):
        logger.info(
            "[DMS-START] dead_man_switch.enabled=false in policy — exiting cleanly."
        )
        sys.exit(0)

    # Threading event to signal graceful shutdown.
    stop_event = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(
            "[DMS-STOP] Signal %s received (PID=%d) — shutting down.", sig_name, pid
        )
        stop_event.set()

    # Register SIGTERM / SIGINT.
    # Windows note: SIGTERM handler is registered but may not fire on forced kill (taskkill /F). Stateless design means no data loss.
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        dms_monitor_loop(policy, stop_event=stop_event)
    except KeyboardInterrupt:
        logger.info("[DMS-STOP] KeyboardInterrupt (PID=%d) — shutting down.", pid)
    except Exception as exc:
        logger.critical("[DMS-STOP] Unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        logger.info("[DMS-STOP] PID=%d exiting.", pid)


if __name__ == "__main__":
    main()
