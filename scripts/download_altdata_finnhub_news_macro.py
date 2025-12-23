"""Download news, news sentiment, and macro data from Finnhub API.

This script downloads news, sentiment, and macro-economic data from Finnhub
and saves them to Parquet files according to the B2 design:
- Raw data: data/raw/altdata/finnhub/{data_type}_raw.parquet
- Cleaned data: {output_dir}/{data_type}.parquet

Usage:
    # Download news for symbols from file
    python scripts/download_altdata_finnhub_news_macro.py \
        --symbols-file config/universe_ai_tech_tickers.txt \
        --start-date 2020-01-01 \
        --end-date 2025-12-03 \
        --download-news

    # Download news sentiment and macro indicators
    python scripts/download_altdata_finnhub_news_macro.py \
        --symbols-file config/universe_ai_tech_tickers.txt \
        --macro-codes-file config/macro_indicators.txt \
        --start-date 2020-01-01 \
        --end-date 2025-12-03 \
        --download-news-sentiment \
        --download-macro
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
from src.assembled_core.data.altdata.finnhub_news_macro import (
    fetch_macro_series,
    fetch_news,
    fetch_news_sentiment,
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


def load_macro_codes_from_file(codes_file: Path) -> list[str]:
    """Load macro indicator codes from a text file (one per line, skip comments and empty lines).

    Args:
        codes_file: Path to text file with macro codes

    Returns:
        List of macro code strings (uppercase, stripped)
    """
    codes = []
    with codes_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                codes.append(line.upper().strip())
    return codes


def clean_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean news events DataFrame (deduplication, validation).

    Args:
        df: Raw news events DataFrame

    Returns:
        Cleaned DataFrame (deduplicated, validated)
    """
    if df.empty:
        return df

    # Deduplicate by news_id (keep first occurrence)
    df = df.drop_duplicates(subset=["news_id"], keep="first")

    # Validate required columns
    required_cols = ["timestamp", "source", "headline", "news_id", "event_type"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp is UTC-aware
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by symbol (if present), then timestamp
    sort_cols = ["symbol", "timestamp"] if "symbol" in df.columns else ["timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def clean_sentiment_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean news sentiment daily DataFrame (validation).

    Args:
        df: Raw news sentiment DataFrame

    Returns:
        Cleaned DataFrame (validated)
    """
    if df.empty:
        return df

    # Validate required columns
    required_cols = ["timestamp", "sentiment_score", "sentiment_volume"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp is UTC-aware
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by symbol (if present), then timestamp
    sort_cols = ["symbol", "timestamp"] if "symbol" in df.columns else ["timestamp"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def clean_macro_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean macro series DataFrame (deduplication, validation).

    Args:
        df: Raw macro series DataFrame

    Returns:
        Cleaned DataFrame (deduplicated, validated)
    """
    if df.empty:
        return df

    # Deduplicate by macro_code + timestamp (keep first occurrence)
    df = df.drop_duplicates(subset=["macro_code", "timestamp"], keep="first")

    # Validate required columns
    required_cols = ["timestamp", "macro_code", "value", "country"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure timestamp is UTC-aware
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort by macro_code, then timestamp
    df = df.sort_values(["macro_code", "timestamp"]).reset_index(drop=True)

    return df


def main() -> int:
    """Main entry point for download script."""
    parser = argparse.ArgumentParser(
        description="Download news, sentiment, and macro data from Finnhub API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download news for symbols from file
    python scripts/download_altdata_finnhub_news_macro.py \\
        --symbols-file config/universe_ai_tech_tickers.txt \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --download-news

    # Download news sentiment and macro indicators
    python scripts/download_altdata_finnhub_news_macro.py \\
        --symbols-file config/universe_ai_tech_tickers.txt \\
        --macro-codes-file config/macro_indicators.txt \\
        --start-date 2020-01-01 \\
        --end-date 2025-12-03 \\
        --download-news-sentiment \\
        --download-macro
        """,
    )

    parser.add_argument(
        "--symbols-file",
        type=Path,
        help="Path to text file with symbols (one per line)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of symbols (e.g., AAPL MSFT GOOGL)",
    )
    parser.add_argument(
        "--macro-codes-file",
        type=Path,
        help="Path to text file with macro indicator codes (one per line, e.g., GDP CPI UNEMPLOYMENT)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--download-news",
        action="store_true",
        help="Download news events",
    )
    parser.add_argument(
        "--download-news-sentiment",
        action="store_true",
        help="Download and aggregate news sentiment",
    )
    parser.add_argument(
        "--download-macro",
        action="store_true",
        help="Download macro-economic indicators",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/altdata"),
        help="Output directory for cleaned data (default: output/altdata)",
    )

    args = parser.parse_args()

    # Validate that at least one download flag is set
    if not (args.download_news or args.download_news_sentiment or args.download_macro):
        parser.error(
            "At least one of --download-news, --download-news-sentiment, or --download-macro must be set"
        )

    # Load settings
    settings = get_settings()

    # Parse dates
    try:
        start_date = date.fromisoformat(args.start_date)
        end_date = date.fromisoformat(args.end_date)
    except ValueError as e:
        print(f"Error: Invalid date format: {e}")
        return 1

    # Load symbols
    symbols = None
    if args.symbols_file:
        if not args.symbols_file.exists():
            print(f"Error: Symbols file not found: {args.symbols_file}")
            return 1
        symbols = load_symbols_from_file(args.symbols_file)
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    elif args.symbols:
        symbols = [s.upper().strip() for s in args.symbols]
        print(f"Using {len(symbols)} symbols from command line")

    # Load macro codes
    macro_codes = None
    if args.macro_codes_file:
        if not args.macro_codes_file.exists():
            print(f"Error: Macro codes file not found: {args.macro_codes_file}")
            return 1
        macro_codes = load_macro_codes_from_file(args.macro_codes_file)
        print(f"Loaded {len(macro_codes)} macro codes from {args.macro_codes_file}")

    # Create output directories
    raw_dir = Path("data/raw/altdata/finnhub")
    raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download news
    if args.download_news:
        print("\n=== Downloading News Events ===")
        try:
            news_df = fetch_news(symbols, start_date, end_date, settings)

            if news_df.empty:
                print("Warning: No news found. Writing empty DataFrame.")
            else:
                print(f"Downloaded {len(news_df)} news articles")
                if "symbol" in news_df.columns:
                    unique_symbols = news_df["symbol"].dropna().unique()
                    print(f"  Symbols: {len(unique_symbols)} unique symbols")
                date_range = f"{news_df['timestamp'].min().date()} to {news_df['timestamp'].max().date()}"
                print(f"  Date range: {date_range}")

            # Save raw data
            raw_path = raw_dir / "news_raw.parquet"
            news_df.to_parquet(raw_path, index=False)
            print(f"Saved raw news to {raw_path}")

            # Clean and save
            news_clean = clean_news_df(news_df)
            clean_path = args.output_dir / "news_events.parquet"
            news_clean.to_parquet(clean_path, index=False)
            print(f"Saved cleaned news to {clean_path}")

        except Exception as e:
            print(f"Error downloading news: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Download news sentiment
    if args.download_news_sentiment:
        print("\n=== Downloading News Sentiment ===")
        try:
            sentiment_df = fetch_news_sentiment(symbols, start_date, end_date, settings)

            if sentiment_df.empty:
                print("Warning: No sentiment data found. Writing empty DataFrame.")
            else:
                print(f"Aggregated {len(sentiment_df)} daily sentiment records")
                if "symbol" in sentiment_df.columns:
                    unique_symbols = sentiment_df["symbol"].dropna().unique()
                    print(f"  Symbols: {len(unique_symbols)} unique symbols")
                date_range = f"{sentiment_df['timestamp'].min().date()} to {sentiment_df['timestamp'].max().date()}"
                print(f"  Date range: {date_range}")

            # Save raw data
            raw_path = raw_dir / "news_sentiment_raw.parquet"
            sentiment_df.to_parquet(raw_path, index=False)
            print(f"Saved raw sentiment to {raw_path}")

            # Clean and save
            sentiment_clean = clean_sentiment_df(sentiment_df)
            clean_path = args.output_dir / "news_sentiment_daily.parquet"
            sentiment_clean.to_parquet(clean_path, index=False)
            print(f"Saved cleaned sentiment to {clean_path}")

        except Exception as e:
            print(f"Error downloading news sentiment: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Download macro data
    if args.download_macro:
        print("\n=== Downloading Macro Indicators ===")
        if not macro_codes:
            print(
                "Warning: No macro codes provided. Use --macro-codes-file or skip --download-macro"
            )
        else:
            try:
                macro_df = fetch_macro_series(
                    macro_codes, start_date, end_date, settings
                )

                if macro_df.empty:
                    print("Warning: No macro data found. Writing empty DataFrame.")
                else:
                    print(f"Downloaded {len(macro_df)} macro data points")
                    unique_codes = macro_df["macro_code"].unique()
                    print(
                        f"  Macro codes: {len(unique_codes)} unique codes: {', '.join(unique_codes[:10])}"
                    )
                    if len(unique_codes) > 10:
                        print(f"  ... and {len(unique_codes) - 10} more")
                    date_range = f"{macro_df['timestamp'].min().date()} to {macro_df['timestamp'].max().date()}"
                    print(f"  Date range: {date_range}")

                # Save raw data
                raw_path = raw_dir / "macro_raw.parquet"
                macro_df.to_parquet(raw_path, index=False)
                print(f"Saved raw macro data to {raw_path}")

                # Clean and save
                macro_clean = clean_macro_df(macro_df)
                clean_path = args.output_dir / "macro_series.parquet"
                macro_clean.to_parquet(clean_path, index=False)
                print(f"Saved cleaned macro data to {clean_path}")

            except Exception as e:
                print(f"Error downloading macro data: {e}")
                import traceback

                traceback.print_exc()
                return 1

    print("\n=== Download Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
