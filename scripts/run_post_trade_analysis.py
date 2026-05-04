#!/usr/bin/env python3
"""Post-Trade Analysis Runner — M11.

Loads ledger fills, computes forward returns from price data,
writes a learning record to the JSONL store.

Usage:
    python scripts/run_post_trade_analysis.py --run-id <id> --price-file <path> \
        [--date <YYYY-MM-DD>] [--horizon-days <N>] [--ledger-path <path>] \
        [--store-path <path>]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.assembled_core.qa.learning_store import (
    append_learning_record,
    summarize_learning_store,
)
from src.assembled_core.qa.post_trade_analyzer import (
    build_learning_record,
    compute_forward_returns,
    compute_signal_hit_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_post_trade_analysis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-trade analysis runner")
    p.add_argument("--run-id", required=True, help="Run identifier")
    p.add_argument("--price-file", required=True, help="Path to prices parquet file")
    p.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Analysis date (YYYY-MM-DD)",
    )
    p.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Forward return horizon in calendar days",
    )
    p.add_argument(
        "--ledger-path",
        default=None,
        help="Path to ledger parquet file (optional)",
    )
    p.add_argument(
        "--store-path",
        default="output/learning/post_trade_learning.jsonl",
        help="Learning store path",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write to store",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("[START] post_trade_analysis run_id=%s date=%s", args.run_id, args.date)

    # Load prices
    price_path = Path(args.price_file)
    if not price_path.exists():
        logger.error("[ERROR] price file not found: %s", price_path)
        sys.exit(1)
    prices_df = pd.read_parquet(price_path)
    symbol_count = prices_df["symbol"].nunique() if "symbol" in prices_df.columns else 0
    logger.info("[OK] loaded prices: %d rows, %d symbols", len(prices_df), symbol_count)

    # Load ledger (optional)
    trades_df = pd.DataFrame(columns=["symbol", "side", "event_ts", "qty", "price"])
    if args.ledger_path:
        ledger_path = Path(args.ledger_path)
        if ledger_path.exists():
            ledger_df = pd.read_parquet(ledger_path)
            # Filter to FILL events only
            if "event_type" in ledger_df.columns:
                fills = ledger_df[ledger_df["event_type"] == "FILL"]
            else:
                fills = ledger_df
            if not fills.empty:
                if "event_ts" in fills.columns:
                    trades_df = fills[["symbol", "event_ts", "qty", "price"]].copy()
                else:
                    trades_df = fills.copy()
                trades_df["side"] = trades_df["qty"].apply(
                    lambda q: "BUY" if float(q) > 0 else "SELL"
                )
                logger.info("[OK] loaded %d fills from ledger", len(trades_df))
        else:
            logger.warning(
                "[WARN] ledger not found at %s — running without trade data",
                ledger_path,
            )

    # Compute forward returns
    fwd_df = compute_forward_returns(prices_df, horizon_days=args.horizon_days)
    logger.info("[OK] computed forward returns: %d rows", len(fwd_df))

    # Compute hit rate
    hit_df = compute_signal_hit_rate(trades_df, fwd_df)
    if hit_df.empty:
        logger.info(
            "[SKIP] no trade/return matches found — recording empty learning record"
        )
    else:
        logger.info(
            "[OK] hit rate summary: %d symbols, overall hit rate=%.2f%%",
            len(hit_df),
            hit_df["hit_rate"].mean() * 100,
        )

    # Build learning record
    record = build_learning_record(
        run_id=args.run_id,
        analysis_date=args.date,
        hit_rate_df=hit_df,
        horizon_days=args.horizon_days,
    )

    if args.dry_run:
        logger.info("[DRY-RUN] record: %s", json.dumps(record, indent=2))
    else:
        store_path = append_learning_record(record, store_path=args.store_path)
        summary = summarize_learning_store(store_path)
        logger.info(
            "[OK] record written to %s (total records: %d)",
            store_path,
            summary["total_records"],
        )

    logger.info("[DONE] post_trade_analysis complete")


if __name__ == "__main__":
    main()
