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
import logging
import sys
from datetime import datetime, date
from pathlib import Path

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import OUTPUT_DIR, get_base_dir
from src.assembled_core.data.prices_ingest import load_eod_prices, load_eod_prices_for_universe
from src.assembled_core.execution.order_generation import generate_orders_from_signals
from src.assembled_core.execution.safe_bridge import write_safe_orders_csv
from src.assembled_core.features.ta_features import add_all_features
from src.assembled_core.logging_utils import setup_logging
from src.assembled_core.portfolio.position_sizing import compute_target_positions_from_trend_signals
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices


def parse_target_date(date_str: str | None) -> datetime:
    """Parse target date string.
    
    Args:
        date_str: Date string (YYYY-MM-DD) or None for today
    
    Returns:
        datetime object (UTC, time set to 00:00:00)
    
    Raises:
        ValueError: If date string format is invalid
    """
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            # Ensure UTC timezone
            if target_date.tzinfo is None:
                target_date = target_date.replace(tzinfo=pd.Timestamp.utcnow().tz)
            return target_date
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
    else:
        return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def filter_prices_for_date(
    prices: pd.DataFrame,
    target_date: datetime,
    mode: str = "last_available"
) -> pd.DataFrame:
    """Filter prices for a specific date.
    
    This function filters price data based on the target date.
    
    **Date Filtering Modes:**
    - `"last_available"` (default): Returns data up to and including target_date.
      For each symbol, uses the last available timestamp <= target_date.
      This is useful for EOD runs where we want the most recent data available.
    - `"exact"`: Returns only data exactly matching target_date (may be empty if no data for that day).
    
    Args:
        prices: DataFrame with columns: timestamp, symbol, ...
        target_date: Target date (datetime, UTC)
        mode: Filtering mode ("last_available" or "exact"), default: "last_available"
    
    Returns:
        Filtered DataFrame with data for the target date (or last available before/on target_date)
    """
    if prices.empty:
        return prices
    
    # Ensure timestamp is timezone-aware UTC
    if prices["timestamp"].dt.tz is None:
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    
    if mode == "exact":
        # Filter to exact date (may be empty)
        target_date_only = target_date.date()
        prices["date_only"] = pd.to_datetime(prices["timestamp"]).dt.date
        filtered = prices[prices["date_only"] == target_date_only].copy()
        filtered = filtered.drop(columns=["date_only"])
    else:  # mode == "last_available"
        # For each symbol, get the last available timestamp <= target_date
        filtered_list = []
        for symbol in prices["symbol"].unique():
            sym_data = prices[prices["symbol"] == symbol].copy()
            sym_data = sym_data[sym_data["timestamp"] <= target_date]
            if not sym_data.empty:
                # Get last available timestamp for this symbol
                last_ts = sym_data["timestamp"].max()
                last_row = sym_data[sym_data["timestamp"] == last_ts]
                filtered_list.append(last_row)
        
        if filtered_list:
            filtered = pd.concat(filtered_list, ignore_index=True)
        else:
            filtered = pd.DataFrame(columns=prices.columns)
    
    return filtered


def validate_universe_vs_data(
    universe_symbols: list[str],
    prices: pd.DataFrame,
    target_date: datetime
) -> tuple[pd.DataFrame, list[str]]:
    """Validate that universe symbols have price data and filter out missing ones.
    
    Args:
        universe_symbols: List of symbols from universe file
        prices: DataFrame with price data (columns: timestamp, symbol, ...)
        target_date: Target date for filtering
    
    Returns:
        Tuple of (filtered_prices, missing_symbols)
        filtered_prices: Price data only for symbols that have data
        missing_symbols: List of symbols from universe that have no data
    """
    if prices.empty:
        return prices, universe_symbols
    
    # Filter prices to target date (last available)
    prices_filtered = filter_prices_for_date(prices, target_date, mode="last_available")
    
    if prices_filtered.empty:
        return prices_filtered, universe_symbols
    
    # Get symbols that have data
    available_symbols = set(prices_filtered["symbol"].str.upper().unique())
    universe_symbols_upper = [s.upper() for s in universe_symbols]
    
    # Find missing symbols
    missing_symbols = [s for s in universe_symbols_upper if s not in available_symbols]
    
    # Filter prices to only universe symbols that have data
    if available_symbols:
        valid_symbols = [s for s in universe_symbols_upper if s in available_symbols]
        if valid_symbols:
            prices_filtered = prices_filtered[
                prices_filtered["symbol"].str.upper().isin(valid_symbols)
            ].copy()
        else:
            prices_filtered = pd.DataFrame(columns=prices_filtered.columns)
    
    return prices_filtered, missing_symbols


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
    
    **Date Handling:**
    The `date_str` parameter specifies the target trading day for which orders should be generated.
    Price data is filtered to the last available data <= target_date (per symbol).
    This ensures that even if data for the exact target date is missing, we use the most recent
    available data up to that date.
    
    **Universe vs. Data:**
    If universe_file is provided, symbols from the universe are validated against available price data.
    Symbols without data are logged as warnings and dropped from the flow.
    If no symbols remain after filtering, the script exits cleanly with a clear message.
    
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
        ValueError: If date string is invalid or no valid symbols remain
        SystemExit: If no orders can be generated (no valid symbols)
    """
    # Setup logging
    logger = setup_logging(level="INFO")
    
    # Parse date
    try:
        target_date = parse_target_date(date_str)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
    
    # Determine output directory
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting EOD-MVP for {target_date.strftime('%Y-%m-%d')}")
    logger.info(f"Output directory: {out_dir}")
    
    # Step 1: Load universe symbols (if universe_file provided)
    universe_symbols = []
    if universe_file:
        universe_path = Path(universe_file)
        if not universe_path.exists():
            logger.error(f"Universe file not found: {universe_path}")
            sys.exit(1)
        
        # Read symbols from universe file
        try:
            with open(universe_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        universe_symbols.append(line.upper())
            
            if not universe_symbols:
                logger.error(f"No symbols found in universe file: {universe_path}")
                sys.exit(1)
            
            logger.info(f"Universe loaded: {len(universe_symbols)} symbols")
        except Exception as e:
            logger.error(f"Failed to read universe file: {e}", exc_info=True)
            sys.exit(1)
    
    # Step 2: Load EOD prices
    logger.info("Step 1: Loading EOD prices...")
    try:
        if price_file:
            price_path = Path(price_file)
            if not price_path.exists():
                logger.error(f"Price file not found: {price_path}")
                sys.exit(1)
            
            prices = load_eod_prices(price_file=price_path, freq="1d")
        else:
            prices = load_eod_prices_for_universe(
                universe_file=universe_file,
                freq="1d"
            )
        
        if prices.empty:
            logger.error("Price data is empty after loading")
            sys.exit(1)
        
        logger.info(f"Loaded prices: {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    except FileNotFoundError as e:
        logger.error(f"Price file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load prices: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 3: Validate universe vs. data and filter
    if universe_symbols:
        logger.info("Step 2: Validating universe symbols against price data...")
        prices_filtered, missing_symbols = validate_universe_vs_data(
            universe_symbols,
            prices,
            target_date
        )
        
        if missing_symbols:
            logger.warning(f"{len(missing_symbols)} symbols from universe have no price data: {', '.join(missing_symbols)}")
            logger.warning("These symbols will be dropped from the flow.")
        
        if prices_filtered.empty:
            logger.error("No valid symbols with price data remain after filtering.")
            logger.error("No orders will be generated.")
            sys.exit(1)
        
        prices = prices_filtered
        logger.info(f"Valid symbols: {prices['symbol'].nunique()} symbols with data")
    else:
        # No universe file: filter prices to target date
        logger.info("Step 2: Filtering prices to target date...")
        prices = filter_prices_for_date(prices, target_date, mode="last_available")
        
        if prices.empty:
            logger.error(f"No price data available for target date {target_date.strftime('%Y-%m-%d')}")
            sys.exit(1)
        
        logger.info(f"Filtered prices: {len(prices)} rows, {prices['symbol'].nunique()} symbols")
    
    # Step 4: Compute TA features
    logger.info("Step 3: Computing TA features...")
    try:
        prices_with_features = add_all_features(
            prices,
            ma_windows=(ma_fast, ma_slow),
            atr_window=14,
            rsi_window=14,
            include_rsi=True
        )
        logger.info(f"Features computed: {len(prices_with_features)} rows")
    except Exception as e:
        logger.error(f"Failed to compute features: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 5: Generate trend signals
    logger.info("Step 4: Generating trend signals...")
    try:
        signals = generate_trend_signals_from_prices(
            prices_with_features,
            ma_fast=ma_fast,
            ma_slow=ma_slow
        )
        long_signals = signals[signals["direction"] == "LONG"]
        logger.info(f"Signals generated: {len(long_signals)} LONG signals from {len(signals)} total")
    except Exception as e:
        logger.error(f"Failed to generate signals: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 6: Compute target positions
    logger.info("Step 5: Computing target positions...")
    try:
        targets = compute_target_positions_from_trend_signals(
            signals,
            total_capital=total_capital,
            top_n=top_n,
            min_score=min_score
        )
        
        if targets.empty:
            logger.warning("No target positions computed (no LONG signals or all filtered out)")
            logger.warning("No orders will be generated.")
            # Create empty SAFE file
            safe_path = write_safe_orders_csv(
                pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"]),
                date=target_date,
                output_path=None,
                price_type="MARKET",
                comment="EOD Strategy - Daily MVP (no signals)"
            )
            logger.info(f"Empty SAFE orders file written: {safe_path}")
            return safe_path
        
        logger.info(f"Target positions: {len(targets)} symbols")
        if not targets.empty:
            logger.info(f"Symbols: {', '.join(targets['symbol'].tolist())}")
    except Exception as e:
        logger.error(f"Failed to compute target positions: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 7: Generate orders
    logger.info("Step 6: Generating orders...")
    try:
        orders = generate_orders_from_signals(
            signals,
            total_capital=total_capital,
            top_n=top_n,
            timestamp=target_date,
            prices=prices_with_features
        )
        
        if orders.empty:
            logger.warning("No orders generated (no position changes)")
            # Create empty SAFE file
            safe_path = write_safe_orders_csv(
                orders,
                date=target_date,
                output_path=None,
                price_type="MARKET",
                comment="EOD Strategy - Daily MVP (no orders)"
            )
            logger.info(f"Empty SAFE orders file written: {safe_path}")
            return safe_path
        
        logger.info(f"Orders generated: {len(orders)} orders")
        if not orders.empty:
            buy_count = len(orders[orders["side"] == "BUY"])
            sell_count = len(orders[orders["side"] == "SELL"])
            logger.info(f"Order breakdown: {buy_count} BUY, {sell_count} SELL")
    except Exception as e:
        logger.error(f"Failed to generate orders: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 8: Write SAFE-Bridge CSV
    logger.info("Step 7: Writing SAFE-Bridge CSV...")
    try:
        safe_path = write_safe_orders_csv(
            orders,
            date=target_date,
            output_path=None,  # Use default: output/orders_YYYYMMDD.csv
            price_type="MARKET",
            comment="EOD Strategy - Daily MVP"
        )
        logger.info(f"SAFE orders written: {safe_path}")
        logger.info(f"Total orders: {len(orders)}")
    except Exception as e:
        logger.error(f"Failed to write SAFE orders: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"SUCCESS: EOD-MVP completed for {target_date.strftime('%Y-%m-%d')}")
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
        help="Date (YYYY-MM-DD), default: today. Orders are generated for this trading day using the last available price data <= date."
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
    
    # Setup logging for main
    logger = setup_logging(level="INFO")
    
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
        
        logger.info(f"Output file: {safe_path}")
        sys.exit(0)
    
    except SystemExit:
        # Re-raise SystemExit (from sys.exit() calls)
        raise
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
