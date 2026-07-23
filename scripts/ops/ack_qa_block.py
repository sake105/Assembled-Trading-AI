"""Acknowledge and clear the QA-block flag (W4, GESAMTBEWERTUNG Schritt 8).

Stage-1 H2 (2026-07-24): a bare file delete would be an unaudited,
accident-indistinguishable unblock of a safety flag (output/ is touched by
cleanup routines). This script is the ONLY sanctioned clear path — mirrors
scripts/ack_halt.py:

  - --reason is mandatory (>= 10 chars),
  - the ack is appended to output/ops/qa_block_ack_ledger.jsonl
    (who/when/why + the full cleared flag content),
  - the flag is ARCHIVED (renamed to qa_block.acked_<UTCts>.json), not
    deleted — the evidence trail survives.

Usage:
    python scripts/ops/ack_qa_block.py --reason "reviewed sharpe gate; ..."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

ACK_LEDGER_PATH = ROOT / "output" / "ops" / "qa_block_ack_ledger.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reason",
        required=True,
        help="Operator reason for clearing the QA block (>= 10 chars).",
    )
    parser.add_argument(
        "--actor",
        default="manual",
        help="Who is acking (default: manual).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    reason = args.reason.strip()
    if len(reason) < 10:
        logger.error("[ack_qa_block] --reason must be >= 10 characters")
        return 1

    from src.assembled_core.qa.qa_gates import QA_BLOCK_FLAG_PATH

    flag_path = QA_BLOCK_FLAG_PATH
    if not flag_path.exists():
        logger.info(
            "[ack_qa_block] no QA-block flag at %s — nothing to clear", flag_path
        )
        return 0

    try:
        flag_content = json.loads(flag_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[ack_qa_block] flag unreadable (%s) — archiving raw file anyway", exc
        )
        flag_content = {"schema": "unreadable"}

    now = datetime.now(timezone.utc)
    ts_tag = now.strftime("%Y%m%dT%H%M%SZ")
    archive_path = flag_path.with_name(f"qa_block.acked_{ts_tag}.json")

    # Ledger BEFORE the state change (an ack that fails to persist must not
    # silently unblock — same discipline as the drawdown halt-write, E-049).
    entry = {
        "ack_at_utc": now.isoformat(),
        "actor": args.actor,
        "reason": reason,
        "cleared_flag": flag_content,
        "archived_to": str(archive_path),
    }
    ACK_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ACK_LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    flag_path.replace(archive_path)
    logger.info(
        "[ack_qa_block] cleared — flag archived to %s, ledger entry written to %s. "
        "Pilot preflight will trade again on the next run.",
        archive_path.name,
        ACK_LEDGER_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
