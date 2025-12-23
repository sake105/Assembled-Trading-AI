# scripts/cli_factor_report.py
"""Factor Report CLI Runner.

This script provides a command-line interface for running factor analysis reports.
It loads price data, computes factors, and generates IC/IR statistics.

Example usage:
    python scripts/cli.py factor_report --freq 1d --symbols-file config/universe_ai_tech_tickers.txt --start-date 2005-01-01 --end-date 2025-12-02 --factor-set core --fwd-horizon-days 5
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.qa.factor_analysis import run_factor_report

import logging

logger = logging.getLogger(__name__)


def load_symbols_from_file(symbols_file: Path) -> list[str]:
    """Load symbols from a text file (one per line).

    Args:
        symbols_file: Path to text file with one symbol per line

    Returns:
        List of symbol strings (uppercased, no empty lines)
    """
    symbols = []
    with symbols_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())

    return symbols


def load_price_data(
    freq: str,
    symbols: list[str] | None = None,
    symbols_file: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    data_source: str | None = None,
) -> pd.DataFrame:
    """Load price data for factor report.

    Args:
        freq: Frequency string ("1d" or "5min")
        symbols: Optional list of symbols to load
        symbols_file: Optional path to file with symbols (one per line)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        data_source: Optional data source override ("local" or "yahoo")

    Returns:
        DataFrame with price data (columns: timestamp, symbol, close, ...)
    """
    settings = get_settings()

    # Determine data source
    if data_source is None:
        data_source = settings.data_source

    # Determine symbols
    if symbols is not None:
        symbol_list = symbols
    elif symbols_file is not None:
        symbol_list = load_symbols_from_file(symbols_file)
        logger.info(f"Loaded {len(symbol_list)} symbols from {symbols_file}")
    else:
        raise ValueError("Either --symbols or --symbols-file must be provided")

    if not symbol_list:
        raise ValueError("No symbols provided")

    logger.info(
        f"Loading price data: {len(symbol_list)} symbols, freq={freq}, data_source={data_source}"
    )

    # Get data source
    price_source = get_price_data_source(settings)

    # Load prices
    prices = price_source.get_history(
        symbols=symbol_list,
        start_date=start_date or "2000-01-01",
        end_date=end_date,
        freq=freq,
    )

    if prices.empty:
        raise ValueError(f"No price data loaded for symbols: {symbol_list}")

    # Filter by date range if provided
    if start_date or end_date:
        if "timestamp" in prices.columns:
            prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
            if start_date:
                start_dt = pd.to_datetime(start_date, utc=True)
                prices = prices[prices["timestamp"] >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date, utc=True)
                prices = prices[prices["timestamp"] <= end_dt]

    logger.info(
        f"Loaded {len(prices)} rows for {prices['symbol'].nunique()} symbols, "
        f"date range: {prices['timestamp'].min()} to {prices['timestamp'].max()}"
    )

    return prices


def run_factor_report_from_args(args) -> int:
    """Run factor report from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Load price data
        prices = load_price_data(
            freq=args.freq,
            symbols_file=args.symbols_file,
            start_date=args.start_date,
            end_date=args.end_date,
            data_source=args.data_source,
        )

        # Run factor report
        logger.info("=" * 60)
        logger.info("Factor Analysis Report")
        logger.info(f"Factor Set: {args.factor_set}")
        logger.info(f"Forward Horizon: {args.fwd_horizon_days} days")
        logger.info("=" * 60)

        report_results = run_factor_report(
            prices=prices,
            factor_set=args.factor_set,
            fwd_horizon_days=args.fwd_horizon_days,
        )

        # Extract results
        summary_ic = report_results["summary_ic"]
        summary_rank_ic = report_results["summary_rank_ic"]

        # Print summary tables
        logger.info("")
        logger.info("=" * 60)
        logger.info("IC Summary (Pearson Correlation)")
        logger.info("=" * 60)

        if not summary_ic.empty:
            # Format for display
            display_df = summary_ic[
                ["factor", "mean_ic", "std_ic", "ic_ir", "hit_ratio", "count"]
            ].copy()
            display_df["mean_ic"] = display_df["mean_ic"].round(4)
            display_df["std_ic"] = display_df["std_ic"].round(4)
            display_df["ic_ir"] = display_df["ic_ir"].round(4)
            display_df["hit_ratio"] = display_df["hit_ratio"].round(3)

            print(
                "\n"
                + tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)
            )
        else:
            logger.warning("No IC summary data available")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Rank-IC Summary (Spearman Rank Correlation)")
        logger.info("=" * 60)

        if not summary_rank_ic.empty:
            # Format for display
            display_df = summary_rank_ic[
                ["factor", "mean_ic", "std_ic", "ic_ir", "hit_ratio", "count"]
            ].copy()
            display_df["mean_ic"] = display_df["mean_ic"].round(4)
            display_df["std_ic"] = display_df["std_ic"].round(4)
            display_df["ic_ir"] = display_df["ic_ir"].round(4)
            display_df["hit_ratio"] = display_df["hit_ratio"].round(3)

            print(
                "\n"
                + tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)
            )
        else:
            logger.warning("No Rank-IC summary data available")

        # Save to CSV if requested
        if args.output_csv:
            output_path = Path(args.output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save IC summary
            summary_ic.to_csv(output_path, index=False)
            logger.info(f"Saved IC summary to {output_path}")

            # Optionally save Rank-IC summary to a separate file
            if output_path.suffix == ".csv":
                rank_ic_path = output_path.with_name(output_path.stem + "_rank_ic.csv")
                summary_rank_ic.to_csv(rank_ic_path, index=False)
                logger.info(f"Saved Rank-IC summary to {rank_ic_path}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Factor Report Completed Successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Factor report failed: {e}", exc_info=True)
        return 1
