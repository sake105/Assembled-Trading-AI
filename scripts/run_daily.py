# scripts/run_daily.py
"""EOD-MVP Runner: Daily order generation using core data/features/signals/portfolio/execution layers.

This script is the EOD-MVP (Minimum Viable Product) for daily order generation.
It uses the new modular layers from Phase 3:
- data.prices_ingest: Load EOD prices with OHLCV
- features.ta_features: Compute technical indicators
- signals.rules_trend: Generate trend-following signals
- portfolio.position_sizing: Determine target positions
- execution.order_generation: Generate orders from targets
- execution.safe_bridge: Write SAFE-Bridge compatible CSV files

Difference from run_eod_pipeline.py:
- run_eod_pipeline.py: Full pipeline with backtest, portfolio simulation, QA
- run_daily.py: Focused EOD-MVP that only generates SAFE orders (no backtest/portfolio equity)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, date
from pathlib import Path

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import OUTPUT_DIR, get_base_dir
from src.assembled_core.data.prices_ingest import load_eod_prices_for_universe
from src.assembled_core.execution.order_generation import generate_orders_from_signals
from src.assembled_core.execution.safe_bridge import write_safe_orders_csv
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.portfolio.position_sizing import compute_target_positions_from_trend_signals
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices


def run_daily_eod(
    date_str: str | None = None,
    universe_file: Path | str | None = None,
    price_file: Path | str | None = None,
    output_dir: Path | str | None = None,
    total_capital: float = 1.0,
    top_n: int | None = None,
    ma_fast: int = 20,
    ma_slow: int = 50,
    min_score: float = 0.0
) -> Path:
    """Run daily EOD order generation.
    
    Args:
        date_str: Date string (YYYY-MM-DD) or None for today
        universe_file: Path to universe file (default: watchlist.txt)
        price_file: Optional explicit path to price file
        output_dir: Output directory (default: config.OUTPUT_DIR)
        total_capital: Total capital for position sizing (default: 1.0)
        top_n: Optional maximum number of positions (default: None = all)
        ma_fast: Fast moving average window (default: 20)
        ma_slow: Slow moving average window (default: 50)
        min_score: Minimum signal score threshold (default: 0.0)
    
    Returns:
        Path to generated SAFE orders CSV file
    
    Raises:
        FileNotFoundError: If price data or universe file not found
        ValueError: If date string is invalid
    """
    # Parse date
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
    else:
        target_date = datetime.utcnow()
    
    # Determine output directory
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[DAILY] Starting EOD-MVP for {target_date.strftime('%Y-%m-%d')}")
    print(f"[DAILY] Output directory: {out_dir}")
    
    # Step 1: Load EOD prices
    print(f"[DAILY] Step 1: Loading EOD prices...")
    try:
        if price_file:
            from src.assembled_core.data.prices_ingest import load_eod_prices
            prices = load_eod_prices(price_file=price_file, freq="1d")
        else:
            prices = load_eod_prices_for_universe(
                universe_file=universe_file,
                freq="1d"
            )
        print(f"[DAILY] [OK] Loaded prices: {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except Exception as e:
        print(f"[DAILY] ERROR loading prices: {e}")
        raise
    
    # Step 2: Compute TA features
    print(f"[DAILY] Step 2: Computing TA features...")
    try:
        prices_with_features = add_all_features(
            prices,
            ma_windows=(ma_fast, ma_slow),
            atr_window=14,
            rsi_window=14,
            include_rsi=True
        )
        print(f"[DAILY] [OK] Features computed: {len(prices_with_features)} rows")
    except Exception as e:
        print(f"[DAILY] ERROR computing features: {e}")
        raise
    
    # Step 3: Generate trend signals
    print(f"[DAILY] Step 3: Generating trend signals...")
    try:
        signals = generate_trend_signals_from_prices(
            prices_with_features,
            ma_fast=ma_fast,
            ma_slow=ma_slow
        )
        long_signals = signals[signals["direction"] == "LONG"]
        print(f"[DAILY] [OK] Signals generated: {len(long_signals)} LONG signals from {len(signals)} total")
    except Exception as e:
        print(f"[DAILY] ERROR generating signals: {e}")
        raise
    
    # Step 4: Compute target positions
    print(f"[DAILY] Step 4: Computing target positions...")
    try:
        targets = compute_target_positions_from_trend_signals(
            signals,
            total_capital=total_capital,
            top_n=top_n,
            min_score=min_score
        )
        print(f"[DAILY] [OK] Target positions: {len(targets)} symbols")
        if not targets.empty:
            print(f"[DAILY] Symbols: {', '.join(targets['symbol'].tolist())}")
    except Exception as e:
        print(f"[DAILY] ERROR computing targets: {e}")
        raise
    
    # Step 5: Generate orders
    print(f"[DAILY] Step 5: Generating orders...")
    try:
        orders = generate_orders_from_signals(
            signals,
            total_capital=total_capital,
            top_n=top_n,
            timestamp=target_date,
            prices=prices_with_features
        )
        print(f"[DAILY] [OK] Orders generated: {len(orders)} orders")
        if not orders.empty:
            buy_count = len(orders[orders["side"] == "BUY"])
            sell_count = len(orders[orders["side"] == "SELL"])
            print(f"[DAILY] Order breakdown: {buy_count} BUY, {sell_count} SELL")
    except Exception as e:
        print(f"[DAILY] ERROR generating orders: {e}")
        raise
    
    # Step 6: Write SAFE-Bridge CSV
    print(f"[DAILY] Step 6: Writing SAFE-Bridge CSV...")
    try:
        safe_path = write_safe_orders_csv(
            orders,
            date=target_date,
            output_path=None,  # Use default: output/orders_YYYYMMDD.csv
            price_type="MARKET",
            comment="EOD Strategy - Daily MVP"
        )
        print(f"[DAILY] [OK] SAFE orders written: {safe_path}")
        print(f"[DAILY] [OK] Total orders: {len(orders)}")
    except Exception as e:
        print(f"[DAILY] ERROR writing SAFE orders: {e}")
        raise
    
    print(f"[DAILY] SUCCESS: EOD-MVP completed for {target_date.strftime('%Y-%m-%d')}")
    return safe_path


def main() -> None:
    """CLI entry point for daily EOD runner."""
    p = argparse.ArgumentParser(
        description="EOD-MVP Runner: Generate daily SAFE orders using core layers"
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date (YYYY-MM-DD), default: today"
    )
    p.add_argument(
        "--universe",
        type=str,
        default=None,
        help="Path to universe file (default: watchlist.txt)"
    )
    p.add_argument(
        "--price-file",
        type=str,
        default=None,
        help="Optional explicit path to price file"
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory (default: from config)"
    )
    p.add_argument(
        "--total-capital",
        type=float,
        default=1.0,
        help="Total capital for position sizing (default: 1.0)"
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Maximum number of positions (default: None = all)"
    )
    p.add_argument(
        "--ma-fast",
        type=int,
        default=20,
        help="Fast moving average window (default: 20)"
    )
    p.add_argument(
        "--ma-slow",
        type=int,
        default=50,
        help="Slow moving average window (default: 50)"
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum signal score threshold (default: 0.0)"
    )
    
    args = p.parse_args()
    
    try:
        safe_path = run_daily_eod(
            date_str=args.date,
            universe_file=Path(args.universe) if args.universe else None,
            price_file=Path(args.price_file) if args.price_file else None,
            output_dir=Path(args.out),
            total_capital=args.total_capital,
            top_n=args.top_n,
            ma_fast=args.ma_fast,
            ma_slow=args.ma_slow,
            min_score=args.min_score
        )
        
        print(f"[DAILY] Output file: {safe_path}")
        sys.exit(0)
    
    except Exception as e:
        print(f"[DAILY] FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

