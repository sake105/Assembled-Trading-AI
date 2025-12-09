"""Download earnings and insider events from Finnhub API.

This script downloads earnings and insider transaction events from Finnhub
and saves them to Parquet files according to the B1 design:
- Raw events: data/raw/altdata/finnhub/{event_type}_events_raw.parquet
- Cleaned events: {output_dir}/events_{event_type}.parquet

Usage:
    # Download earnings events for symbols from file
    python scripts/download_altdata_finnhub_events.py \
        --symbols-file config/universe_ai_tech_tickers.txt \
        --start-date 2020-01-01 \
        --end-date 2025-12-03 \
        --event-types earnings

    # Download both earnings and insider events
    python scripts/download_altdata_finnhub_events.py \
        --symbols AAPL MSFT GOOGL \
        --start-date 2020-01-01 \
        --end-date 2025-12-03 \
        --event-types earnings insider
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.altdata.finnhub_events import (
    fetch_earnings_events,
    fetch_insider_events,
)


def load_symbols_from_file(symbols_file: Path) -> list[str]:
    """Load symbols from a text file (one per line, skip comments and empty lines).
    
    Args:
        symbols_file: Path to text file with symbols
    
    Returns:
        List of symbol strings (uppercase, stripped)
    """
    symbols = []
    with symbols_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper().strip())
    return symbols


def clean_events_df(df: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """Clean events DataFrame (deduplication, validation).
    
    Args:
        df: Raw events DataFrame
        event_type: Event type ("earnings" or "insider")
    
    Returns:
        Cleaned DataFrame (deduplicated, validated)
    """
    if df.empty:
        return df
    
    # Deduplicate by event_id (keep first occurrence)
    df = df.drop_duplicates(subset=["event_id"], keep="first")
    
    # Validate required columns
    required_cols = ["timestamp", "symbol", "event_type", "event_id"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure timestamp is UTC-aware
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Sort by symbol, then timestamp
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    
    return df


def main() -> int:
    """Main entry point for download script."""
    parser = argparse.ArgumentParser(
        description="Download earnings and insider events from Finnhub API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download earnings events for symbols from file
    python scripts/download_altdata_finnhub_events.py \\
        --symbols-file config/universe_ai_tech_tickers.txt \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --event-types earnings

    # Download both earnings and insider events for specific symbols
    python scripts/download_altdata_finnhub_events.py \\
        --symbols AAPL MSFT GOOGL \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --event-types earnings insider
        """
    )
    
    # Symbol arguments (mutually exclusive)
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbols-file",
        type=str,
        help="Path to text file with symbols (one per line)"
    )
    symbol_group.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="List of symbols directly (e.g., AAPL MSFT GOOGL)"
    )
    
    # Date arguments
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    
    # Event type arguments
    parser.add_argument(
        "--event-types",
        type=str,
        nargs="+",
        choices=["earnings", "insider"],
        default=["earnings", "insider"],
        help="Event types to download (default: both earnings and insider)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/altdata",
        help="Output directory for cleaned events (default: output/altdata)"
    )
    
    args = parser.parse_args()
    
    # Load settings
    settings = get_settings()
    
    # Validate API key
    if not settings.finnhub_api_key or not settings.finnhub_api_key.strip():
        print("ERROR: FINNHUB_API_KEY not set.")
        print("Set via ASSEMBLED_FINNHUB_API_KEY environment variable or in settings.")
        return 1
    
    # Load symbols
    if args.symbols:
        symbols = [s.upper().strip() for s in args.symbols]
    elif args.symbols_file:
        symbols_file = Path(args.symbols_file)
        if not symbols_file.exists():
            print(f"ERROR: Symbols file not found: {symbols_file}")
            return 1
        symbols = load_symbols_from_file(symbols_file)
    else:
        print("ERROR: Must specify either --symbols or --symbols-file")
        return 1
    
    if not symbols:
        print("ERROR: No symbols to process")
        return 1
    
    print(f"Processing {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    
    # Parse dates
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    except ValueError as e:
        print(f"ERROR: Invalid date format: {e}")
        print("Dates must be in YYYY-MM-DD format")
        return 1
    
    # Create output directories
    raw_output_dir = ROOT / "data" / "raw" / "altdata" / "finnhub"
    raw_output_dir.mkdir(parents=True, exist_ok=True)
    
    cleaned_output_dir = Path(args.output_dir)
    cleaned_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each event type
    for event_type in args.event_types:
        print(f"\n{'=' * 60}")
        print(f"Downloading {event_type} events...")
        print(f"{'=' * 60}")
        
        # Fetch events
        if event_type == "earnings":
            df = fetch_earnings_events(symbols, start_date, end_date, settings)
        elif event_type == "insider":
            df = fetch_insider_events(symbols, start_date, end_date, settings)
        else:
            print(f"ERROR: Unknown event type: {event_type}")
            continue
        
        if df.empty:
            print(f"WARNING: No {event_type} events found. Creating empty file anyway.")
        else:
            print(f"Downloaded {len(df)} {event_type} events")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Symbols: {df['symbol'].nunique()}")
        
        # Save raw events
        raw_output_path = raw_output_dir / f"{event_type}_events_raw.parquet"
        df.to_parquet(raw_output_path, index=False)
        print(f"Saved raw events to: {raw_output_path}")
        
        # Clean events
        df_cleaned = clean_events_df(df.copy(), event_type)
        
        # Save cleaned events
        cleaned_output_path = cleaned_output_dir / f"events_{event_type}.parquet"
        df_cleaned.to_parquet(cleaned_output_path, index=False)
        print(f"Saved cleaned events to: {cleaned_output_path}")
    
    print(f"\n{'=' * 60}")
    print("Download complete!")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    exit(main())

