"""Rebuild the root-level reconcile artifact for the ARMED reconcile-block gate.

Stage-1 review M3 (2026-07-21, GESAMTBEWERTUNG K5): the armed gate
(``paper_runner.reconcile_block.enabled: true``) is fail-closed and runs
BEFORE the trading cycle — after a FAIL artifact or >stale-hours gap, no
cycle ever reaches the code that would refresh ``output/reconcile_latest.json``,
so the pilot deadlocks by design. This script is the documented operator
recovery path: after the underlying problem is fixed (and, where relevant,
positions adopted via scripts/ops_adopt_external_positions.py + halt acked
via scripts/ack_halt.py), it re-runs the ledger invariants on the CURRENT
ledger state and writes a fresh root artifact.

It does NOT talk to the broker and does NOT bypass the invariants: a ledger
that still violates cash/equity/positions invariants produces a FAIL
artifact and the gate keeps blocking.

Usage:
    python scripts/ops/rebuild_reconcile_artifact.py --reason "why"
    python scripts/ops/rebuild_reconcile_artifact.py --reason "why" --dry-run
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

LEDGER_PATH = ROOT / "output" / "runs" / "_paper_ledger" / "ledger_state.json"
OUTPUT_DIR = ROOT / "output"
AUDIT_PATH = ROOT / "output" / "ops" / "reconcile_artifact_rebuild_log.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reason",
        required=True,
        help="Operator reason (>=10 chars) — audit-logged, mandatory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate invariants and print the would-be status without writing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    if len(args.reason.strip()) < 10:
        logger.error("[rebuild-reconcile] --reason must be >= 10 characters")
        return 1

    import pandas as pd

    from src.assembled_core.ops.paper_ledger import load_ledger_state
    from src.assembled_core.ops.reconcile import (
        build_reconcile_report,
        write_reconcile_artifact,
    )

    if not LEDGER_PATH.exists():
        logger.error("[rebuild-reconcile] ledger not found: %s", LEDGER_PATH)
        return 1

    state = load_ledger_state(LEDGER_PATH)
    now_iso = datetime.now(timezone.utc).isoformat()

    # No-trade snapshot: before == after, no orders/fills. The invariants
    # (cash_non_negative, equity_finite, positions_finite, fills_match_orders)
    # still run for real — a broken ledger yields status=FAIL.
    report = build_reconcile_report(
        as_of_utc=now_iso,
        ledger_before=state,
        ledger_after=state,
        orders=[],
        fills=[],
        prices_latest=pd.DataFrame(),
        cost_model_cfg={},
    )
    report["rebuild"] = {
        "actor": "operator",
        "reason": args.reason.strip(),
        "rebuilt_at_utc": now_iso,
        "source": "scripts/ops/rebuild_reconcile_artifact.py",
    }
    status = report.get("status", "?")
    logger.info("[rebuild-reconcile] invariants evaluated — status=%s", status)

    if args.dry_run:
        logger.info("[rebuild-reconcile] --dry-run set, not writing")
        return 0

    path = write_reconcile_artifact(OUTPUT_DIR, report)
    logger.info("[rebuild-reconcile] wrote %s (status=%s)", path, status)

    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "ts_utc": now_iso,
                        "reason": args.reason.strip(),
                        "status": status,
                        "artifact": str(path),
                    }
                )
                + "\n"
            )
    except Exception as exc:  # audit is best-effort, the artifact write is not
        logger.warning("[rebuild-reconcile] audit append failed: %s", exc)

    return 0 if status == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
