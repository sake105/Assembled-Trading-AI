"""STOP worker — M4 Execution Workers (Ops v1).

Registers a STOP intent (idempotent) and writes a sentinel file
``output/ops/.stop_active``. The sentinel signals to operators and
downstream tooling that new order generation should be suppressed.

This worker is independent of the main trading cycle. It can be invoked
manually or via a scheduler in emergency or end-of-day stop scenarios.

Idempotency:
    A second run on the same UTC day is a no-op (returns exit code 0 with
    a [SKIP] log). Use ``--force`` to override and re-register.

Clearing the stop:
    Delete ``output/ops/.stop_active`` to clear the stop sentinel.
    The intent store record is preserved for audit purposes.

Usage:
    python scripts/run_stop_worker.py
    python scripts/run_stop_worker.py --reason "Market circuit breaker"
    python scripts/run_stop_worker.py --output-dir output/ops --force
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.intent_store import (
    has_intent,
    make_daily_key,
    record_intent,
)

logger = logging.getLogger("stop_worker")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STOP worker — register a stop intent and write sentinel file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reason",
        default="manual stop",
        help="Human-readable reason for the stop.",
    )
    p.add_argument(
        "--output-dir",
        default="output/ops",
        help="Directory for sentinel file and intent store.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-register even if today's STOP intent is already recorded.",
    )
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store_path = output_dir / "intent_store.jsonl"
    sentinel_path = output_dir / ".stop_active"
    idempotency_key = make_daily_key("STOP")

    # --- Idempotency checks ---
    if sentinel_path.exists() and not args.force:
        logger.warning(
            "[SKIP] stop_worker: sentinel already exists at %s — stop is already active. "
            "Use --force to re-register.",
            sentinel_path,
        )
        return 0

    if has_intent(idempotency_key, store_path) and not args.force:
        logger.warning(
            "[SKIP] stop_worker: STOP intent already recorded today (key=%s). "
            "Use --force to override.",
            idempotency_key,
        )
        return 0

    # --- Register stop ---
    t0 = time.monotonic()
    logger.info("[START] stop_worker reason=%r", args.reason)

    try:
        now_utc = datetime.now(timezone.utc).isoformat()

        record_intent(
            "STOP",
            idempotency_key,
            metadata={
                "reason": args.reason,
                "sentinel_path": str(sentinel_path),
                "timestamp_utc": now_utc,
            },
            store_path=store_path,
        )

        sentinel_path.write_text(
            f"stop_active\ntimestamp_utc={now_utc}\nreason={args.reason}\n",
            encoding="utf-8",
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "[OK] stop_worker engaged in %.2fs | sentinel=%s",
            elapsed,
            sentinel_path,
        )
        logger.warning(
            "[WARN] STOP is now active. New order generation should be suppressed. "
            "Remove %s to clear.",
            sentinel_path,
        )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "[ERROR] stop_worker failed after %.2fs: %s",
            elapsed,
            exc,
            exc_info=True,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
