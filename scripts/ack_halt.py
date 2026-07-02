"""E0.4 / A2 — Manual halt acknowledgement CLI.

When the reconciliation-halt policy engages (E0.4) the engine writes
``output/ops/halt_ack_required.json``. The next paper-trading-ci run's
halt-ack gate aborts until a reviewer clears this file deliberately.

This CLI is the only sanctioned way to clear it. A reason is mandatory
so the activation ledger has an explicit audit trail.

Usage:
    python scripts/ack_halt.py --reason="reviewed_2026-04-17_cash_drift_404usd"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the repo root is importable when run as a script: sys.path[0] is
# otherwise scripts/, so ``from src.assembled_core...`` fails and the
# halt_cleared alert silently never fires. Mirrors scripts/run_live_paper.py.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("ack_halt")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

HALT_FLAG_PATH = Path("output/ops/halt_ack_required.json")
ACK_LEDGER_PATH = Path("output/ops/halt_ack_ledger.jsonl")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Acknowledge and clear a paper-cycle halt flag."
    )
    parser.add_argument(
        "--reason",
        required=True,
        help="Required audit string. Include date + what was reviewed, e.g. reviewed_YYYY-MM-DD_<topic>.",
    )
    parser.add_argument(
        "--actor",
        default="manual",
        help="Who acknowledged the halt (default: manual).",
    )
    args = parser.parse_args(argv)

    reason = args.reason.strip()
    if len(reason) < 10:
        logger.error("[ACK_HALT] --reason too short; include date + topic")
        return 2

    if not HALT_FLAG_PATH.exists():
        logger.info(
            "[ACK_HALT] no halt flag present at %s — nothing to clear", HALT_FLAG_PATH
        )
        return 0

    try:
        flag_payload = json.loads(HALT_FLAG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "[ACK_HALT] could not parse existing flag (%s); will still clear", exc
        )
        flag_payload = {"parse_error": str(exc)}

    ACK_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ack_at_utc": datetime.now(timezone.utc).isoformat(),
        "actor": args.actor,
        "reason": reason,
        "cleared_flag": flag_payload,
    }
    with open(ACK_LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")

    try:
        HALT_FLAG_PATH.unlink()
    except Exception as exc:
        logger.error("[ACK_HALT] could not remove %s: %s", HALT_FLAG_PATH, exc)
        return 1

    logger.info(
        "[ACK_HALT] cleared halt flag — ledger entry written to %s", ACK_LEDGER_PATH
    )
    try:
        from src.assembled_core.ops.alerting import AlertManager

        AlertManager().fire("halt_cleared", {"actor": args.actor, "reason": reason})
    except Exception as exc:
        logger.error("[ACK_HALT] all-clear alert failed: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
