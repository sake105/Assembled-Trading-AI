"""RECONCILE worker — M4 Execution Workers (Ops v1).

Compares the ledger's computed positions against a broker snapshot and
writes a reconcile manifest.  The worker is read-only with respect to
trading state — it never generates orders, never modifies the ledger, and
never alters positions.  It is always safe to run.

Inputs:
    --ledger-path       Parquet file produced by ledger_store (ledger_events.parquet)
    --broker-path       CSV with columns: symbol, qty  (and optionally: cash)
    --ledger-cash       Ledger starting cash (float, default 0.0)
    --broker-cash       Broker cash balance (float, default 0.0)

Outputs:
    A JSON manifest written to ``output/ops/reconcile_manifest_<ts>.json``
    containing ok/mismatch/diff details.  A RECONCILE intent is appended to
    the intent store for audit purposes.

Idempotency:
    Each run generates a unique timestamped manifest.  There is no daily
    idempotency guard because reconcile is inherently read-only and safe to
    run multiple times.

Usage:
    python scripts/run_reconcile_worker.py \\
        --ledger-path output/ledger_paper/ledger_events.parquet \\
        --broker-path output/broker/snapshot_20260330.csv \\
        --broker-cash 100000.0 --ledger-cash 100000.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.assembled_core.accounting.position_engine import build_positions_from_ledger
from src.assembled_core.accounting.reconciliation import reconcile_ledger_vs_broker
from src.assembled_core.execution.intent_store import make_run_key, record_intent

logger = logging.getLogger("reconcile_worker")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RECONCILE worker — compare ledger vs broker snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ledger-path",
        default=None,
        help="Path to ledger_events.parquet. If omitted, positions are treated as empty.",
    )
    p.add_argument(
        "--broker-path",
        default=None,
        help="Path to broker snapshot CSV (columns: symbol, qty). "
        "If omitted, broker positions are treated as empty.",
    )
    p.add_argument(
        "--ledger-cash",
        type=float,
        default=0.0,
        help="Ledger starting/current cash balance.",
    )
    p.add_argument(
        "--broker-cash",
        type=float,
        default=0.0,
        help="Broker cash balance.",
    )
    p.add_argument(
        "--output-dir",
        default="output/ops",
        help="Directory for reconcile manifest and intent store.",
    )
    p.add_argument(
        "--cash-tol",
        type=float,
        default=1e-6,
        help="Cash tolerance for reconciliation.",
    )
    p.add_argument(
        "--qty-tol",
        type=float,
        default=1e-8,
        help="Quantity tolerance for reconciliation.",
    )
    return p.parse_args()


def _load_ledger_positions(ledger_path: str | None) -> pd.DataFrame:
    """Load ledger events parquet and compute current positions."""
    if ledger_path is None:
        logger.info("[INFO] no --ledger-path supplied — treating ledger as empty")
        return pd.DataFrame(columns=["symbol", "qty"])

    path = Path(ledger_path)
    if not path.exists():
        logger.warning("[WARN] ledger path not found: %s — treating as empty", path)
        return pd.DataFrame(columns=["symbol", "qty"])

    events_df = pd.read_parquet(path)
    result = build_positions_from_ledger(events_df)
    positions_df = result.get("positions_df", pd.DataFrame(columns=["symbol", "qty"]))
    logger.info("[INFO] ledger loaded: %d positions from %s", len(positions_df), path)
    return positions_df[["symbol", "qty"]].copy()


def _load_broker_snapshot(broker_path: str | None) -> pd.DataFrame:
    """Load broker snapshot CSV."""
    if broker_path is None:
        logger.info("[INFO] no --broker-path supplied — treating broker as empty")
        return pd.DataFrame(columns=["symbol", "qty"])

    path = Path(broker_path)
    if not path.exists():
        logger.warning("[WARN] broker path not found: %s — treating as empty", path)
        return pd.DataFrame(columns=["symbol", "qty"])

    df = pd.read_csv(path)
    required = ["symbol", "qty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Broker snapshot missing columns {missing}: {path}")

    logger.info("[INFO] broker snapshot loaded: %d positions from %s", len(df), path)
    return df[["symbol", "qty"]].copy()


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
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y%m%d_%H%M%S")
    manifest_path = output_dir / f"reconcile_manifest_{ts_str}.json"

    t0 = time.monotonic()
    logger.info(
        "[START] reconcile_worker ledger=%s broker=%s",
        args.ledger_path,
        args.broker_path,
    )

    exit_code = 0

    try:
        ledger_positions = _load_ledger_positions(args.ledger_path)
        broker_positions = _load_broker_snapshot(args.broker_path)

        result = reconcile_ledger_vs_broker(
            ledger_positions_df=ledger_positions,
            ledger_cash=args.ledger_cash,
            broker_positions_df=broker_positions,
            broker_cash=args.broker_cash,
            cash_tol=args.cash_tol,
            qty_tol=args.qty_tol,
        )

        ok: bool = result.get("ok", False)
        cash_match: bool = result.get("cash_match", False)
        cash_diff: float = result.get("cash_diff", 0.0)
        position_diffs_df: pd.DataFrame = result.get(
            "position_diffs_df", pd.DataFrame()
        )
        missing_in_ledger: list[str] = result.get("missing_in_ledger", [])
        missing_in_broker: list[str] = result.get("missing_in_broker", [])
        message: str = result.get("message", "")

        elapsed = time.monotonic() - t0

        # Serialize position diffs
        diffs_records = (
            position_diffs_df.to_dict(orient="records")
            if not position_diffs_df.empty
            else []
        )

        manifest: dict = {
            "timestamp_utc": now_utc.isoformat(),
            "ledger_path": str(args.ledger_path),
            "broker_path": str(args.broker_path),
            "ledger_cash": args.ledger_cash,
            "broker_cash": args.broker_cash,
            "ok": ok,
            "cash_match": cash_match,
            "cash_diff": cash_diff,
            "position_diffs": diffs_records,
            "missing_in_ledger": missing_in_ledger,
            "missing_in_broker": missing_in_broker,
            "message": message,
            "elapsed_s": round(elapsed, 3),
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

        # Record intent for audit
        run_id = ts_str
        record_intent(
            "RECONCILE",
            make_run_key("RECONCILE", run_id),
            metadata={
                "ok": ok,
                "manifest_path": str(manifest_path),
                "position_diff_count": len(diffs_records),
                "missing_in_ledger_count": len(missing_in_ledger),
                "missing_in_broker_count": len(missing_in_broker),
            },
            store_path=store_path,
        )

        if ok:
            logger.info(
                "[OK] reconcile_worker done in %.2fs | match=True | manifest=%s",
                elapsed,
                manifest_path,
            )
        else:
            logger.warning(
                "[WARN] reconcile_worker done in %.2fs | match=False | "
                "cash_diff=%.6f | position_diffs=%d | manifest=%s",
                elapsed,
                cash_diff,
                len(diffs_records),
                manifest_path,
            )
            if missing_in_ledger:
                logger.warning(
                    "[WARN] symbols in broker but not ledger: %s", missing_in_ledger
                )
            if missing_in_broker:
                logger.warning(
                    "[WARN] symbols in ledger but not broker: %s", missing_in_broker
                )

    except Exception as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "[ERROR] reconcile_worker failed after %.2fs: %s",
            elapsed,
            exc,
            exc_info=True,
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
