# scripts/sprint9_execute.py
# LEGACY: Dieses Script gehört zur alten Sprint-Phase.
# Für neue Entwicklungen bitte src/assembled_core/* und scripts/run_*.py verwenden.
# Dieses Script wird noch von run_all_sprint10.ps1 verwendet, ist aber deprecated.
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.assembled_core.config import OUTPUT_DIR
from src.assembled_core.ema_config import get_default_ema_config
from src.assembled_core.pipeline.io import load_prices
from src.assembled_core.pipeline.signals import compute_ema_signals
from src.assembled_core.pipeline.orders import signals_to_orders, write_orders


@dataclass
class ExecArgs:
    freq: str
    ema_fast: int
    ema_slow: int
    price_file: str | None
    out_dir: Path


def make_orders(freq: str, fast: int, slow: int, price_file: str | None = None, output_dir: str | Path = "output") -> pd.DataFrame:
    """Generate orders from EMA crossover strategy.
    
    Args:
        freq: Frequency string ("1d" or "5min")
        fast: Fast EMA period
        slow: Slow EMA period
        price_file: Optional explicit path to price file
        output_dir: Base output directory
    
    Returns:
        DataFrame with columns: timestamp, symbol, side, qty, price
    """
    prices = load_prices(freq, price_file=price_file, output_dir=output_dir)
    signals = compute_ema_signals(prices, fast, slow)
    orders = signals_to_orders(signals)
    return orders


def _print_symbol_counts(orders: pd.DataFrame) -> None:
    """Print order counts per symbol."""
    print("[EXEC] Orders by symbol:")
    if orders.empty:
        print("  - (keine Orders)")
        return
    for sym, grp in orders.groupby("symbol"):
        print(f"  - {sym}: {len(grp)}")


def run_execution(a: ExecArgs) -> Tuple[Path, pd.DataFrame]:
    """Run execution pipeline.
    
    Args:
        a: Execution arguments
    
    Returns:
        Tuple of (output_path, orders DataFrame)
    
    Side effects:
        Writes orders CSV file
        Prints execution status to stdout
    """
    print(f"[EXEC] START Execution | freq={a.freq}")
    orders = make_orders(a.freq, a.ema_fast, a.ema_slow, price_file=a.price_file, output_dir=a.out_dir)
    out_path = write_orders(orders, a.freq, a.out_dir)
    print(f"[EXEC] [OK] written: {out_path} | rows={len(orders)}")
    _print_symbol_counts(orders)
    print("[EXEC] DONE Execution")
    return out_path, orders


def parse_args() -> ExecArgs:
    p = argparse.ArgumentParser(description="EMA-Crossover Execution (orders csv)")
    p.add_argument("--freq", choices=["1d", "5min"], required=True, help="Zeitebene")
    
    # Parse freq first to get frequency-based defaults
    args_partial, remaining = p.parse_known_args()
    default_ema = get_default_ema_config(args_partial.freq)
    
    # Add remaining arguments with frequency-based defaults
    p.add_argument("--ema-fast", type=int, default=default_ema.fast,
                   help=f"Fast EMA period (default: {default_ema.fast} for {args_partial.freq})")
    p.add_argument("--ema-slow", type=int, default=default_ema.slow,
                   help=f"Slow EMA period (default: {default_ema.slow} for {args_partial.freq})")
    p.add_argument("--price-file", type=str, default=None,
                   help="Optional eigener Pfad zu Preisen (Parquet mit timestamp,symbol,close)")
    p.add_argument("--out", type=str, default=str(OUTPUT_DIR), help="Output-Ordner (default: from config)")
    
    # Parse all arguments
    args = p.parse_args()
    
    return ExecArgs(
        freq=args.freq,
        ema_fast=int(args.ema_fast),
        ema_slow=int(args.ema_slow),
        price_file=args.price_file,
        out_dir=Path(args.out),
    )


def main() -> None:
    a = parse_args()
    if a.ema_fast <= 0 or a.ema_slow <= 0:
        raise ValueError("EMA-Längen müssen > 0 sein.")
    if a.ema_fast >= a.ema_slow:
        print(f"[WARN] ema_fast ({a.ema_fast}) >= ema_slow ({a.ema_slow}) – üblich ist fast < slow.")
    run_execution(a)


if __name__ == "__main__":
    main()
