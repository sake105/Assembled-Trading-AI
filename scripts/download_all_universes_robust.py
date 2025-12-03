"""Robust batch download script for all universe tickers.

This script downloads all symbols from all universe ticker files with:
- Robust retry logic for rate limits
- Per-symbol error handling
- Progress tracking
- Resume capability (skip existing files)

Usage:
    python scripts/download_all_universes_robust.py
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def read_symbols_from_file(symbols_file: Path) -> list[str]:
    """Read symbols from text file."""
    symbols = []
    if not symbols_file.exists():
        logger.error(f"Symbols file not found: {symbols_file}")
        return symbols
    
    with open(symbols_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())
    
    return symbols


def check_file_exists(symbol: str, interval: str, target_root: Path) -> bool:
    """Check if Parquet file already exists and has reasonable size."""
    file_path = target_root / interval / f"{symbol}.parquet"
    
    if not file_path.exists():
        return False
    
    # Check file size (should be at least 1KB for valid data)
    try:
        size = file_path.stat().st_size
        return size > 1024
    except Exception:
        return False


def download_single_symbol_with_retry(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str,
    target_root: Path,
    max_retries: int = 5,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
) -> tuple[bool, str]:
    """Download a single symbol with robust retry logic.
    
    Returns:
        (success: bool, message: str)
    """
    # Import here to avoid circular imports
    import sys
    from pathlib import Path as P
    
    # Add scripts directory to path
    scripts_dir = P(__file__).parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    
    # Import download functions
    from download_historical_snapshot import (
        download_symbol_history,
        normalize_dataframe,
        store_symbol_df,
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            # Check if file already exists
            if check_file_exists(symbol, interval, target_root):
                logger.info(f"  [{symbol}] File already exists, skipping")
                return True, "already_exists"
            
            # Download raw data
            hist = download_symbol_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                max_retries=2,  # Fewer retries per attempt, we retry the whole thing
                retry_delay=base_delay,
            )
            
            if hist.empty:
                return False, f"No data available for {symbol}"
            
            # Normalize and store
            df = normalize_dataframe(hist, symbol)
            if df.empty:
                return False, f"Normalization failed for {symbol}"
            
            store_symbol_df(symbol, df, interval, target_root)
            return True, f"Downloaded {len(df)} rows"
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limits
            is_rate_limit = any(
                keyword in error_str
                for keyword in ["rate limit", "too many requests", "429", "rate limited"]
            )
            
            if is_rate_limit:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    jitter = random.uniform(0, delay * 0.2)  # 20% jitter
                    total_delay = delay + jitter
                    
                    logger.warning(
                        f"  [{symbol}] Rate limit hit (attempt {attempt}/{max_retries}). "
                        f"Waiting {total_delay:.1f}s before retry..."
                    )
                    time.sleep(total_delay)
                    continue
                else:
                    return False, f"Rate limited after {max_retries} attempts"
            else:
                # Other errors - log and retry
                if attempt < max_retries:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"  [{symbol}] Error (attempt {attempt}/{max_retries}): {e}. "
                        f"Waiting {delay:.1f}s before retry..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    return False, f"Failed after {max_retries} attempts: {e}"
    
    return False, "Max retries exceeded"


def download_universe_robust(
    universe_name: str,
    symbols_file: Path,
    start_date: str,
    end_date: str,
    interval: str,
    target_root: Path,
    skip_existing: bool = True,
    sleep_between_symbols: float = 3.0,
    max_retries_per_symbol: int = 5,
) -> dict:
    """Download all symbols from a universe file with robust error handling.
    
    Returns:
        Dictionary with statistics: {successful, failed, skipped, rate_limits}
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Processing Universe: {universe_name}")
    logger.info(f"  Symbols File: {symbols_file}")
    logger.info("=" * 60)
    
    # Read symbols
    symbols = read_symbols_from_file(symbols_file)
    
    if not symbols:
        logger.error(f"  No symbols found in {symbols_file}")
        return {"successful": 0, "failed": 0, "skipped": 0, "rate_limits": 0}
    
    logger.info(f"  Found {len(symbols)} symbols")
    
    stats = {"successful": 0, "failed": 0, "skipped": 0, "rate_limits": 0}
    failed_symbols = []
    
    # Download each symbol
    for idx, symbol in enumerate(symbols, 1):
        logger.info(f"  [{idx}/{len(symbols)}] Downloading {symbol}...")
        
        # Check if should skip
        if skip_existing and check_file_exists(symbol, interval, target_root):
            logger.info(f"    [{symbol}] Already exists, skipping")
            stats["skipped"] += 1
            continue
        
        # Download with retry
        success, message = download_single_symbol_with_retry(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            target_root=target_root,
            max_retries=max_retries_per_symbol,
        )
        
        if success:
            if message != "already_exists":
                stats["successful"] += 1
                logger.info(f"    [{symbol}] ✓ {message}")
            else:
                stats["skipped"] += 1
        else:
            stats["failed"] += 1
            failed_symbols.append(symbol)
            if "rate limit" in message.lower():
                stats["rate_limits"] += 1
            logger.error(f"    [{symbol}] ✗ {message}")
        
        # Sleep between symbols (except after last)
        if idx < len(symbols) and success:
            time.sleep(sleep_between_symbols)
        
        # Extra delay after rate limit
        if "rate limit" in message.lower():
            logger.warning(f"    [{symbol}] Rate limit detected, waiting 10s before next symbol...")
            time.sleep(10.0)
    
    # Summary
    logger.info("")
    logger.info(f"  Universe {universe_name} Summary:")
    logger.info(f"    Successful: {stats['successful']}")
    logger.info(f"    Skipped: {stats['skipped']}")
    logger.info(f"    Failed: {stats['failed']}")
    if failed_symbols:
        logger.warning(f"    Failed symbols: {', '.join(failed_symbols[:10])}")
        if len(failed_symbols) > 10:
            logger.warning(f"    ... and {len(failed_symbols) - 10} more")
    
    return stats


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Robust batch download for all universe tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--target-root",
        type=str,
        default=r"F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025",
        help="Target root directory for Parquet files",
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-12-03",
        help="End date (YYYY-MM-DD)",
    )
    
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        help="Data interval (1d, 1h, etc.)",
    )
    
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=3.0,
        help="Sleep seconds between symbols",
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries per symbol",
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip symbols that already have files",
    )
    
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Download even if file exists",
    )
    
    args = parser.parse_args()
    
    target_root = Path(args.target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    
    # Universe definitions
    universes = [
        ("AI Tech", Path("config/universe_ai_tech_tickers.txt")),
        ("Healthcare Biotech", Path("config/healthcare_biotech_tickers.txt")),
        ("Energy Resources Cyclicals", Path("config/energy_resources_cyclicals_tickers.txt")),
        ("Defense Security Aero", Path("config/defense_security_aero_tickers.txt")),
        ("Consumer Financial Misc", Path("config/consumer_financial_misc_tickers.txt")),
    ]
    
    logger.info("=" * 60)
    logger.info("Robust Batch Download - All Universes")
    logger.info("=" * 60)
    logger.info(f"Target Root: {target_root}")
    logger.info(f"Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Sleep Between Symbols: {args.sleep_seconds}s")
    logger.info(f"Max Retries per Symbol: {args.max_retries}")
    logger.info(f"Skip Existing: {args.skip_existing}")
    logger.info("=" * 60)
    
    # Overall statistics
    overall_stats = {
        "total_symbols": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "rate_limits": 0,
    }
    
    start_time = time.time()
    
    # Process each universe
    for idx, (universe_name, symbols_file) in enumerate(universes, 1):
        logger.info("")
        logger.info(f"Universe {idx}/{len(universes)}: {universe_name}")
        
        if not symbols_file.exists():
            logger.error(f"  Symbols file not found: {symbols_file}, skipping")
            continue
        
        # Count symbols
        symbols = read_symbols_from_file(symbols_file)
        overall_stats["total_symbols"] += len(symbols)
        
        # Download universe
        stats = download_universe_robust(
            universe_name=universe_name,
            symbols_file=symbols_file,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            target_root=target_root,
            skip_existing=args.skip_existing,
            sleep_between_symbols=args.sleep_seconds,
            max_retries_per_symbol=args.max_retries,
        )
        
        # Accumulate stats
        overall_stats["successful"] += stats["successful"]
        overall_stats["failed"] += stats["failed"]
        overall_stats["skipped"] += stats["skipped"]
        overall_stats["rate_limits"] += stats["rate_limits"]
        
        # Pause between universes
        if idx < len(universes):
            logger.info(f"  Pausing 10 seconds before next universe...")
            time.sleep(10.0)
    
    # Final summary
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("Final Summary")
    logger.info("=" * 60)
    logger.info(f"Total Symbols: {overall_stats['total_symbols']}")
    logger.info(f"Successful: {overall_stats['successful']}")
    logger.info(f"Skipped: {overall_stats['skipped']}")
    logger.info(f"Failed: {overall_stats['failed']}")
    logger.info(f"Rate Limit Hits: {overall_stats['rate_limits']}")
    logger.info(f"Total Duration: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    logger.info("=" * 60)
    
    if overall_stats["failed"] > 0:
        logger.warning(f"Some downloads failed. You may want to re-run to retry failed symbols.")
        exit(1)
    else:
        logger.info("All downloads completed successfully!")
        exit(0)


if __name__ == "__main__":
    main()

