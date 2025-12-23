# scripts/sprint9_backtest.py
# LEGACY: Dieses Script gehört zur alten Sprint-Phase.
# Für neue Entwicklungen bitte src/assembled_core/* und scripts/run_*.py verwenden.
# Dieses Script wird noch von run_all_sprint10.ps1 verwendet, ist aber deprecated.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.pipeline.io import (
    load_prices,
    load_prices_with_fallback,
    load_orders,
)
from src.assembled_core.pipeline.backtest import (
    simulate_equity,
    compute_metrics,
    write_backtest_report,
)


def main():
    """CLI wrapper for backtest simulation."""
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, choices=["1d", "5min"], default="5min")
    p.add_argument("--start-capital", type=float, default=10000.0)
    p.add_argument(
        "--price-file",
        type=str,
        default=None,
        help="Optional: Pfad zu einem Parquet mit ['timestamp','symbol','close']",
    )
    args = p.parse_args()

    print(f"[BT9] START Backtest | freq={args.freq}")

    # Load data
    if args.price_file:
        prices = load_prices(
            args.freq, price_file=args.price_file, output_dir=OUTPUT_DIR
        )
        print(
            f"[BT9] Prices: {args.price_file}  rows={len(prices)}  symbols={prices['symbol'].nunique()}"
        )
    else:
        prices = load_prices_with_fallback(args.freq, output_dir=OUTPUT_DIR)
        print(
            f"[BT9] Prices: loaded  rows={len(prices)}  symbols={prices['symbol'].nunique()}"
        )

    orders = load_orders(args.freq, output_dir=OUTPUT_DIR, strict=False)
    if not orders.empty:
        print(f"[BT9] Reading orders: rows={len(orders)}")
    else:
        print("[BT9] Orders nicht gefunden → flache Equity (nur Startkapital).")

    # Simulate
    eq = simulate_equity(prices, orders, start_capital=float(args.start_capital))
    metrics = compute_metrics(eq)

    # Write results
    curve_path, rep_path = write_backtest_report(
        eq, metrics, args.freq, output_dir=OUTPUT_DIR
    )
    print(f"[BT9] [OK] written: {curve_path}")
    print(f"[BT9] [OK] written: {rep_path}")
    print("[BT9] DONE Backtest")


if __name__ == "__main__":
    main()
