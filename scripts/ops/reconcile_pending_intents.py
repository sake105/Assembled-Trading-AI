"""One-shot reconciliation tool for stale pending ORDER_SUBMIT intents.

R6-Pilot followup: the 9-day pilot dormancy + chronic MSFT-buy targeting
(never fills) left 101 ORDER_SUBMIT records without matching ORDER_COMPLETE.
The pilot startup logs "N pending intents from prior crash — reconcile
manually" but provides no drain mechanism.

This script:
1. Loads intent_store, finds unmatched SUBMITs.
2. Cross-checks against broker open orders (best-effort).
3. For orders NOT in broker's open-orders list: writes a
   `cancelled_stale_reconciliation` COMPLETE record.
4. Lists any remaining open-at-broker entries for manual cancellation.

Usage:
    python scripts/ops/reconcile_pending_intents.py --dry-run
    python scripts/ops/reconcile_pending_intents.py --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def find_pending(store_path: Path) -> list[dict]:
    """Find ORDER_SUBMIT entries without matching ORDER_COMPLETE."""
    if not store_path.exists():
        return []
    records = [
        json.loads(line)
        for line in store_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    submits = {
        r["idempotency_key"]: r for r in records if r.get("action") == "ORDER_SUBMIT"
    }
    completes = {
        r["idempotency_key"] for r in records if r.get("action") == "ORDER_COMPLETE"
    }
    pending = [r for k, r in submits.items() if k not in completes]
    return pending


def fetch_broker_open_orders() -> set[str]:
    """Return set of (symbol, side, qty) tuples of broker-side open orders.

    Returns empty set on import / connection failure (non-fatal — pilot may
    not have alpaca-py configured in dev mode).
    """
    try:
        from src.assembled_core.execution.broker_adapter import AlpacaAdapter

        adapter = AlpacaAdapter()
        orders = (
            adapter.get_open_orders() if hasattr(adapter, "get_open_orders") else []
        )
        return {
            (
                getattr(o, "symbol", "?"),
                getattr(o, "side", "?"),
                float(getattr(o, "qty", 0) or 0),
            )
            for o in orders
        }
    except Exception as exc:
        logger.warning("Broker open-orders fetch failed (non-fatal): %s", exc)
        return set()


def reconcile(store_path: Path, dry_run: bool = True) -> int:
    """Reconcile pending intents. Returns count of records drained."""
    pending = find_pending(store_path)
    if not pending:
        print("[reconcile] no pending intents — nothing to do")
        return 0

    print(f"[reconcile] found {len(pending)} pending ORDER_SUBMIT records")
    # Aggregate for visibility
    agg = Counter(
        (r["metadata"].get("symbol", "?"), r["metadata"].get("side", "?"))
        for r in pending
    )
    print("[reconcile] by symbol/side:")
    for (sym, side), n in agg.most_common():
        print(f"  {sym:6} {side:4} x{n}")

    # Cross-check with broker
    broker_open = fetch_broker_open_orders()
    if broker_open:
        print(f"[reconcile] broker reports {len(broker_open)} open orders")
    else:
        print("[reconcile] no broker open orders (or fetch unavailable)")

    drainable = []
    still_open = []
    for rec in pending:
        m = rec.get("metadata", {})
        key = (m.get("symbol", "?"), m.get("side", "?"), float(m.get("qty", 0) or 0))
        if key in broker_open:
            still_open.append(rec)
        else:
            drainable.append(rec)

    if still_open:
        print(
            f"[reconcile] WARNING: {len(still_open)} pending intents still match "
            f"broker open orders — cancel them manually first."
        )
        for r in still_open[:5]:
            m = r["metadata"]
            print(
                f"  ts={r['timestamp_utc'][:19]} sym={m.get('symbol')} "
                f"side={m.get('side')} qty={m.get('qty')}"
            )
        if len(still_open) > 5:
            print(f"  ... ({len(still_open) - 5} more)")

    print(f"[reconcile] drainable (safe to mark cancelled_stale): {len(drainable)}")

    if dry_run:
        print("[reconcile] DRY RUN — no changes written. Re-run with --apply.")
        return 0

    if not drainable:
        return 0

    # Apply: append COMPLETE records with status=cancelled_stale_reconciliation
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()
    appended = 0
    with store_path.open("a", encoding="utf-8") as f:
        for rec in drainable:
            m = rec["metadata"]
            complete = {
                "action": "ORDER_COMPLETE",
                "idempotency_key": rec["idempotency_key"],
                "timestamp_utc": now,
                "metadata": {
                    "symbol": m.get("symbol", ""),
                    "side": m.get("side", ""),
                    "qty": m.get("qty", 0),
                    "filled_qty": 0.0,
                    "filled_price": None,
                    "status": "cancelled_stale_reconciliation",
                },
            }
            f.write(json.dumps(complete) + "\n")
            appended += 1

    print(f"[reconcile] APPLIED — drained {appended} pending intents")
    return appended


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the reconciliation (default: dry-run)",
    )
    parser.add_argument(
        "--store",
        default=str(ROOT / "output" / "ops" / "intent_store.jsonl"),
        help="Path to intent_store.jsonl",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    reconcile(Path(args.store), dry_run=not args.apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
