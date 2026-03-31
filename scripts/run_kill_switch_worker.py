"""KILL SWITCH worker — M4 Execution Workers (Ops v1).

Activates the kill switch for emergency order blocking.  This is more
severe than a regular STOP: it signals that all positions should be
flattened and no new orders generated until the kill switch is explicitly
cleared.

Actions taken:
1. Registers a KILL intent (idempotent — once per day unless --force).
2. Writes a persistent sentinel file ``output/ops/.kill_switch_active``.
3. If ``--positions-path`` is provided: generates flatten orders as a
   SAFE-Bridge CSV (``output/ops/flatten_orders_<date>.csv``) for manual
   review before execution.  No orders are sent automatically.

Safety:
    This worker never sends orders to a broker directly.  Flatten orders
    are written to a CSV for human review (SAFE-Bridge pattern).  Actual
    execution requires a separate manual step.

Clearing the kill switch:
    Delete ``output/ops/.kill_switch_active`` to clear the sentinel.
    The intent store record is preserved for audit purposes.

Usage:
    python scripts/run_kill_switch_worker.py
    python scripts/run_kill_switch_worker.py --reason "Geo event escalation"
    python scripts/run_kill_switch_worker.py \\
        --positions-path output/ledger_paper/positions.csv \\
        --reason "Emergency flatten"
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

import pandas as pd

from src.assembled_core.execution.intent_store import (
    has_intent,
    make_daily_key,
    record_intent,
)
from src.assembled_core.execution.safe_bridge import write_safe_orders_csv

logger = logging.getLogger("kill_switch_worker")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="KILL SWITCH worker — emergency order block + optional paper flatten.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reason",
        default="emergency kill switch",
        help="Human-readable reason for the kill switch activation.",
    )
    p.add_argument(
        "--output-dir",
        default="output/ops",
        help="Directory for sentinel file, intent store, and flatten orders.",
    )
    p.add_argument(
        "--positions-path",
        default=None,
        help="Optional CSV with current positions (columns: symbol, qty). "
        "If provided, generates SAFE-Bridge flatten orders for manual review.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-register even if today's KILL intent is already recorded.",
    )
    return p.parse_args()


def _generate_flatten_orders(positions_path: str, date_str: str) -> pd.DataFrame | None:
    """Load positions and generate flatten (close-to-zero) orders.

    Returns a SAFE-Bridge compatible DataFrame or None on failure.
    """
    path = Path(positions_path)
    if not path.exists():
        logger.warning("[WARN] positions path not found: %s — skipping flatten", path)
        return None

    df = pd.read_csv(path)
    required = ["symbol", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.warning(
            "[WARN] positions CSV missing columns %s — skipping flatten", missing
        )
        return None

    # Only include non-zero positions
    open_positions = df[df["qty"].abs() > 1e-8].copy()
    if open_positions.empty:
        logger.info("[INFO] no open positions to flatten")
        return pd.DataFrame(columns=["symbol", "side", "qty"])

    # Flatten = sell longs, buy back shorts
    orders_rows = []
    for _, row in open_positions.iterrows():
        qty = float(row["qty"])
        side = "SELL" if qty > 0 else "BUY"
        orders_rows.append(
            {"symbol": str(row["symbol"]), "side": side, "qty": abs(qty)}
        )

    orders_df = pd.DataFrame(orders_rows)
    logger.info("[INFO] generated %d flatten order(s)", len(orders_df))
    return orders_df


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
    sentinel_path = output_dir / ".kill_switch_active"
    idempotency_key = make_daily_key("KILL")

    # --- Idempotency checks ---
    if sentinel_path.exists() and not args.force:
        logger.warning(
            "[SKIP] kill_switch_worker: sentinel already exists at %s. "
            "Kill switch is already active. Use --force to re-register.",
            sentinel_path,
        )
        return 0

    if has_intent(idempotency_key, store_path) and not args.force:
        logger.warning(
            "[SKIP] kill_switch_worker: KILL intent already recorded today (key=%s). "
            "Use --force to override.",
            idempotency_key,
        )
        return 0

    # --- Activate kill switch ---
    t0 = time.monotonic()
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.isoformat()
    date_str = now_utc.strftime("%Y%m%d")

    logger.info("[START] kill_switch_worker reason=%r", args.reason)
    logger.warning(
        "[WARN] KILL SWITCH ACTIVATION IN PROGRESS — all order generation will be blocked"
    )

    exit_code = 0
    flatten_path: Path | None = None

    try:
        # Write sentinel
        sentinel_path.write_text(
            f"kill_switch_active\ntimestamp_utc={now_str}\nreason={args.reason}\n",
            encoding="utf-8",
        )
        logger.info("[OK] kill switch sentinel written to %s", sentinel_path)

        # Optional: generate paper flatten orders
        flatten_order_count = 0
        if args.positions_path is not None:
            orders_df = _generate_flatten_orders(args.positions_path, date_str)
            if orders_df is not None and not orders_df.empty:
                flatten_path = output_dir / f"flatten_orders_{date_str}.csv"
                write_safe_orders_csv(
                    orders_df,
                    output_path=flatten_path,
                    date=now_utc,
                    comment="KILL_SWITCH_FLATTEN",
                )
                flatten_order_count = len(orders_df)
                logger.info(
                    "[OK] flatten orders written to %s (%d orders — REVIEW BEFORE EXECUTING)",
                    flatten_path,
                    flatten_order_count,
                )
                logger.warning(
                    "[WARN] FLATTEN ORDERS require manual review at %s. "
                    "They are NOT submitted automatically.",
                    flatten_path,
                )

        # Record intent
        record_intent(
            "KILL",
            idempotency_key,
            metadata={
                "reason": args.reason,
                "sentinel_path": str(sentinel_path),
                "flatten_path": str(flatten_path) if flatten_path else None,
                "flatten_order_count": flatten_order_count,
                "timestamp_utc": now_str,
            },
            store_path=store_path,
        )

        elapsed = time.monotonic() - t0
        logger.info(
            "[OK] kill_switch_worker done in %.2fs | sentinel=%s | flatten_orders=%d",
            elapsed,
            sentinel_path,
            flatten_order_count,
        )
        logger.warning(
            "[WARN] KILL SWITCH IS ACTIVE. "
            "Set ASSEMBLED_KILL_SWITCH=1 in the environment of any trading process. "
            "Remove %s to clear.",
            sentinel_path,
        )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "[ERROR] kill_switch_worker failed after %.2fs: %s",
            elapsed,
            exc,
            exc_info=True,
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
