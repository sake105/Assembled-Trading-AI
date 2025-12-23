# scripts/run_factor_analysis.py
"""Factor Analysis CLI Runner.

This script provides a command-line interface for comprehensive factor analysis,
including IC-based evaluation (C1) and portfolio-based evaluation (C2).

Example usage:
    python scripts/cli.py analyze_factors --freq 1d --symbols-file config/macro_world_etfs_tickers.txt --data-source local --start-date 2010-01-01 --end-date 2025-12-03 --factor-set core+vol_liquidity --horizon-days 20
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
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
)
from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)
from src.assembled_core.features.altdata_news_macro_factors import (
    build_news_sentiment_factors,
    build_macro_regime_factors,
)
from src.assembled_core.logging_config import setup_logging
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_ic,
    compute_rank_ic,
    summarize_ic_series,
    build_factor_portfolio_returns,
    build_long_short_portfolio_returns,
    summarize_factor_portfolios,
    compute_deflated_sharpe_ratio,
)

import logging

logger = logging.getLogger(__name__)


def list_available_factor_sets(
    with_descriptions: bool = False,
) -> list[str] | dict[str, str]:
    """List all available factor set names.

    This is the single source of truth for factor set names used across the codebase.

    Args:
        with_descriptions: If True, return a dict mapping factor set names to descriptions.
                          If False, return a list of factor set names only.

    Returns:
        If with_descriptions=False: List of factor set names (e.g., ["core", "vol_liquidity", ...])
        If with_descriptions=True: Dict mapping factor set names to descriptions
    """
    factor_sets = {
        "core": "TA/Price Factors (Multi-Horizon Returns, Trend Strength, Reversal)",
        "vol_liquidity": "Volatility & Liquidity Factors (RV, Vol-of-Vol, Turnover, Spread Proxies)",
        "core+vol_liquidity": "Core TA/Price + Volatility/Liquidity",
        "all": "Alle TA/Price/Vol/Liquidity Factors (inkl. Market Breadth)",
        "alt_earnings_insider": "Nur Earnings/Insider Factors (Earnings Surprise, Insider Activity)",
        "alt_news_macro": "Nur News/Macro Factors (News Sentiment, Macro Regime)",
        "core+alt": "Core TA/Price + Earnings/Insider (B1)",
        "core+alt_news": "Core TA/Price + News/Macro (B2)",
        "core+alt_full": "Core TA/Price + Earnings/Insider (B1) + News/Macro (B2)",
    }

    if with_descriptions:
        return factor_sets
    else:
        return list(factor_sets.keys())


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
    universe: Path | str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    data_source: str | None = None,
) -> pd.DataFrame:
    """Load price data for factor analysis.

    Args:
        freq: Frequency string ("1d" or "5min")
        symbols: Optional list of symbols to load
        symbols_file: Optional path to file with symbols (one per line)
        universe: Optional path to universe file (alias for symbols_file)
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        data_source: Optional data source override ("local", "yahoo", etc.)

    Returns:
        DataFrame with price data (columns: timestamp, symbol, close, ...)

    Raises:
        ValueError: If no symbols provided or data loading fails
    """
    settings = get_settings()

    # Determine data source (prefer local if ASSEMBLED_LOCAL_DATA_ROOT is set)
    if data_source is None:
        if settings.local_data_root:
            data_source = "local"
        else:
            data_source = settings.data_source

    # Determine symbols (priority: symbols > symbols_file > universe)
    if symbols is not None:
        symbol_list = symbols
    elif symbols_file is not None:
        symbol_list = load_symbols_from_file(Path(symbols_file))
        logger.info(f"Loaded {len(symbol_list)} symbols from {symbols_file}")
    elif universe is not None:
        symbol_list = load_symbols_from_file(Path(universe))
        logger.info(f"Loaded {len(symbol_list)} symbols from {universe}")
    else:
        raise ValueError(
            "Either --symbols, --symbols-file, or --universe must be provided"
        )

    if not symbol_list:
        raise ValueError("No symbols provided")

    logger.info(
        f"Loading price data: {len(symbol_list)} symbols, freq={freq}, data_source={data_source}"
    )

    # Get data source
    price_source = get_price_data_source(settings, data_source=data_source)

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


def load_altdata_events(
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load alt-data events from Parquet files.

    Args:
        output_dir: Optional output directory (default: output/altdata)

    Returns:
        Tuple of (events_earnings_df, events_insider_df)
        Both DataFrames may be empty if files don't exist
    """
    if output_dir is None:
        output_dir = Path("output") / "altdata"
    else:
        output_dir = Path(output_dir)

    earnings_path = output_dir / "events_earnings.parquet"
    insider_path = output_dir / "events_insider.parquet"

    events_earnings = pd.DataFrame()
    events_insider = pd.DataFrame()

    if earnings_path.exists():
        try:
            events_earnings = pd.read_parquet(earnings_path)
            logger.info(
                f"Loaded {len(events_earnings)} earnings events from {earnings_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load earnings events from {earnings_path}: {e}")
    else:
        logger.warning(f"Earnings events file not found: {earnings_path}")

    if insider_path.exists():
        try:
            events_insider = pd.read_parquet(insider_path)
            logger.info(
                f"Loaded {len(events_insider)} insider events from {insider_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load insider events from {insider_path}: {e}")
    else:
        logger.warning(f"Insider events file not found: {insider_path}")

    return events_earnings, events_insider


def load_altdata_news_macro(
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load alt-data news sentiment and macro series from Parquet files.

    Args:
        output_dir: Optional output directory (default: output/altdata)

    Returns:
        Tuple of (news_sentiment_daily_df, macro_series_df)
        Both DataFrames may be empty if files don't exist
    """
    if output_dir is None:
        output_dir = Path("output") / "altdata"
    else:
        output_dir = Path(output_dir)

    sentiment_path = output_dir / "news_sentiment_daily.parquet"
    macro_path = output_dir / "macro_series.parquet"

    news_sentiment_daily = pd.DataFrame()
    macro_series = pd.DataFrame()

    if sentiment_path.exists():
        try:
            news_sentiment_daily = pd.read_parquet(sentiment_path)
            logger.info(
                f"Loaded {len(news_sentiment_daily)} news sentiment daily records from {sentiment_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load news sentiment from {sentiment_path}: {e}")
    else:
        logger.warning(f"News sentiment file not found: {sentiment_path}")

    if macro_path.exists():
        try:
            macro_series = pd.read_parquet(macro_path)
            logger.info(
                f"Loaded {len(macro_series)} macro series records from {macro_path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load macro series from {macro_path}: {e}")
    else:
        logger.warning(f"Macro series file not found: {macro_path}")

    return news_sentiment_daily, macro_series


def compute_factors(
    prices: pd.DataFrame,
    factor_set: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute factors based on factor_set.

    Args:
        prices: Price DataFrame with timestamp, symbol, close, etc.
        factor_set: Factor set identifier:
            - "core": Core TA/Price factors only
            - "vol_liquidity": Volatility and liquidity factors only
            - "core+vol_liquidity": Core + Volatility/Liquidity
            - "all": All factors (core + vol/liquidity + market breadth)
            - "alt_earnings_insider": Alt-Data factors only (Earnings + Insider, B1)
            - "core+alt": Core TA/Price + Alt-Data factors (B1)
            - "alt_news_macro": Alt-Data factors only (News + Macro, B2)
            - "core+alt_news": Core TA/Price + Alt-Data factors (B2)
            - "core+alt_full": Core TA/Price + Alt-Data factors (B1 + B2)
        output_dir: Optional output directory for loading alt-data events (default: output/altdata)

    Returns:
        DataFrame with factors added
    """
    factors_df = prices.copy()

    # Compute core TA/Price factors
    if factor_set in (
        "core",
        "core+vol_liquidity",
        "all",
        "core+alt",
        "core+alt_news",
        "core+alt_full",
    ):
        logger.info("Computing core TA/Price factors...")
        factors_df = build_core_ta_factors(factors_df)

    # Compute volatility/liquidity factors
    if factor_set in ("vol_liquidity", "core+vol_liquidity", "all"):
        logger.info("Computing volatility factors...")
        factors_df = add_realized_volatility(factors_df, windows=[20, 60])

        if "volume" in factors_df.columns:
            logger.info("Computing liquidity factors...")
            factors_df = add_turnover_and_liquidity_proxies(factors_df)
        else:
            logger.warning("Skipping liquidity factors (no volume column)")

    # Compute market breadth factors
    if factor_set == "all":
        logger.info("Computing market breadth factors...")
        # Market breadth is universe-level, so we compute it separately
        # For now, we skip it in the main factor DataFrame
        # It can be added later if needed for factor analysis

    # Compute Alt-Data factors (Earnings + Insider) - B1
    if factor_set in ("alt_earnings_insider", "core+alt", "core+alt_full"):
        logger.info("Loading alt-data events (B1)...")
        events_earnings, events_insider = load_altdata_events(output_dir=output_dir)

        # Filter events to match price data symbols and date range
        if not events_earnings.empty:
            # Ensure timestamps are UTC-aware
            if "timestamp" in events_earnings.columns:
                events_earnings["timestamp"] = pd.to_datetime(
                    events_earnings["timestamp"], utc=True
                )

            # Filter to symbols in prices
            price_symbols = set(factors_df["symbol"].unique())
            events_earnings = events_earnings[
                events_earnings["symbol"].isin(price_symbols)
            ]

            # Filter to date range of prices
            if not factors_df.empty and "timestamp" in factors_df.columns:
                price_min_date = factors_df["timestamp"].min()
                price_max_date = factors_df["timestamp"].max()
                events_earnings = events_earnings[
                    (events_earnings["timestamp"] >= price_min_date)
                    & (events_earnings["timestamp"] <= price_max_date)
                ]

            if not events_earnings.empty:
                logger.info("Computing earnings surprise factors...")
                earnings_factors = build_earnings_surprise_factors(
                    events_earnings,
                    factors_df,
                    window_days=20,
                )

                # Merge earnings factors (only factor columns)
                earnings_factor_cols = [
                    col
                    for col in earnings_factors.columns
                    if col.startswith("earnings_") or col.startswith("post_earnings_")
                ]
                if earnings_factor_cols:
                    merge_cols = ["timestamp", "symbol"] + earnings_factor_cols
                    factors_df = factors_df.merge(
                        earnings_factors[merge_cols],
                        on=["timestamp", "symbol"],
                        how="left",
                    )
                    logger.info(
                        f"Added {len(earnings_factor_cols)} earnings factor columns"
                    )
            else:
                logger.warning(
                    "No earnings events found after filtering to price data symbols/date range"
                )
        else:
            logger.warning("No earnings events available (file missing or empty)")

        if not events_insider.empty:
            # Ensure timestamps are UTC-aware
            if "timestamp" in events_insider.columns:
                events_insider["timestamp"] = pd.to_datetime(
                    events_insider["timestamp"], utc=True
                )

            # Filter to symbols in prices
            price_symbols = set(factors_df["symbol"].unique())
            events_insider = events_insider[
                events_insider["symbol"].isin(price_symbols)
            ]

            # Filter to date range of prices
            if not factors_df.empty and "timestamp" in factors_df.columns:
                price_min_date = factors_df["timestamp"].min()
                price_max_date = factors_df["timestamp"].max()
                events_insider = events_insider[
                    (events_insider["timestamp"] >= price_min_date)
                    & (events_insider["timestamp"] <= price_max_date)
                ]

            if not events_insider.empty:
                logger.info("Computing insider activity factors...")
                insider_factors = build_insider_activity_factors(
                    events_insider,
                    factors_df,
                    lookback_days=60,
                )

                # Merge insider factors (only factor columns)
                insider_factor_cols = [
                    col for col in insider_factors.columns if col.startswith("insider_")
                ]
                if insider_factor_cols:
                    merge_cols = ["timestamp", "symbol"] + insider_factor_cols
                    factors_df = factors_df.merge(
                        insider_factors[merge_cols],
                        on=["timestamp", "symbol"],
                        how="left",
                    )
                    logger.info(
                        f"Added {len(insider_factor_cols)} insider factor columns"
                    )
            else:
                logger.warning(
                    "No insider events found after filtering to price data symbols/date range"
                )
        else:
            logger.warning("No insider events available (file missing or empty)")

    # Compute Alt-Data factors (News + Macro) - B2
    if factor_set in ("alt_news_macro", "core+alt_news", "core+alt_full"):
        logger.info("Loading alt-data news/macro (B2)...")
        news_sentiment_daily, macro_series = load_altdata_news_macro(
            output_dir=output_dir
        )

        # Filter news sentiment to match price data symbols and date range
        if not news_sentiment_daily.empty:
            # Ensure timestamps are UTC-aware
            if "timestamp" in news_sentiment_daily.columns:
                news_sentiment_daily["timestamp"] = pd.to_datetime(
                    news_sentiment_daily["timestamp"], utc=True
                )

            # Filter to date range of prices
            if not factors_df.empty and "timestamp" in factors_df.columns:
                price_min_date = factors_df["timestamp"].min()
                price_max_date = factors_df["timestamp"].max()
                news_sentiment_daily = news_sentiment_daily[
                    (news_sentiment_daily["timestamp"] >= price_min_date)
                    & (news_sentiment_daily["timestamp"] <= price_max_date)
                ]

            if not news_sentiment_daily.empty:
                logger.info("Computing news sentiment factors...")
                news_factors = build_news_sentiment_factors(
                    news_sentiment_daily,
                    factors_df,
                    lookback_days=20,
                )

                # Merge news sentiment factors (only factor columns)
                news_factor_cols = [
                    col
                    for col in news_factors.columns
                    if col.startswith("news_sentiment_")
                ]
                if news_factor_cols:
                    merge_cols = ["timestamp", "symbol"] + news_factor_cols
                    factors_df = factors_df.merge(
                        news_factors[merge_cols], on=["timestamp", "symbol"], how="left"
                    )
                    logger.info(
                        f"Added {len(news_factor_cols)} news sentiment factor columns"
                    )
            else:
                logger.warning(
                    "No news sentiment data found after filtering to price data date range"
                )
        else:
            logger.warning("No news sentiment data available (file missing or empty)")

        # Filter macro series to match price data date range
        if not macro_series.empty:
            # Ensure timestamps are UTC-aware
            if "timestamp" in macro_series.columns:
                macro_series["timestamp"] = pd.to_datetime(
                    macro_series["timestamp"], utc=True
                )

            # Filter to date range of prices
            if not factors_df.empty and "timestamp" in factors_df.columns:
                price_min_date = factors_df["timestamp"].min()
                price_max_date = factors_df["timestamp"].max()
                macro_series = macro_series[
                    (macro_series["timestamp"] >= price_min_date)
                    & (macro_series["timestamp"] <= price_max_date)
                ]

            if not macro_series.empty:
                logger.info("Computing macro regime factors...")
                macro_factors = build_macro_regime_factors(
                    macro_series,
                    factors_df,
                    country_filter=None,  # Use all countries by default
                )

                # Merge macro regime factors (only factor columns)
                macro_factor_cols = [
                    col for col in macro_factors.columns if col.startswith("macro_")
                ]
                if macro_factor_cols:
                    merge_cols = ["timestamp", "symbol"] + macro_factor_cols
                    factors_df = factors_df.merge(
                        macro_factors[merge_cols],
                        on=["timestamp", "symbol"],
                        how="left",
                    )
                    logger.info(
                        f"Added {len(macro_factor_cols)} macro regime factor columns"
                    )
            else:
                logger.warning(
                    "No macro series data found after filtering to price data date range"
                )
        else:
            logger.warning("No macro series data available (file missing or empty)")

    logger.info(f"Computed factors: {len(factors_df.columns)} total columns")

    return factors_df


def write_factor_analysis_report(
    summary_ic: pd.DataFrame,
    summary_rank_ic: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    output_dir: Path,
    factor_set: str,
    horizon_days: int,
    freq: str,
) -> None:
    """Write factor analysis report to Markdown and CSV files.

    Args:
        summary_ic: IC summary DataFrame
        summary_rank_ic: Rank-IC summary DataFrame
        portfolio_summary: Portfolio summary DataFrame
        output_dir: Output directory
        factor_set: Factor set identifier
        horizon_days: Forward return horizon
        freq: Frequency string
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate file names
    base_name = f"factor_analysis_{factor_set}_{horizon_days}d_{freq}"
    md_path = output_dir / f"{base_name}_report.md"
    ic_csv_path = output_dir / f"{base_name}_ic_summary.csv"
    rank_ic_csv_path = output_dir / f"{base_name}_rank_ic_summary.csv"
    portfolio_csv_path = output_dir / f"{base_name}_portfolio_summary.csv"

    # Write Markdown report
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Factor Analysis Report\n\n")
        f.write(f"**Factor Set:** {factor_set}\n")
        f.write(f"**Forward Horizon:** {horizon_days} days\n")
        f.write(f"**Frequency:** {freq}\n\n")

        f.write("## IC Summary (Pearson Correlation)\n\n")
        if not summary_ic.empty:
            f.write(summary_ic.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No IC data available.\n\n")

        f.write("## Rank-IC Summary (Spearman Rank Correlation)\n\n")
        if not summary_rank_ic.empty:
            f.write(summary_rank_ic.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No Rank-IC data available.\n\n")

        f.write("## Portfolio Performance Summary\n\n")
        if not portfolio_summary.empty:
            f.write(portfolio_summary.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("No portfolio data available.\n\n")

    logger.info(f"Saved report to {md_path}")

    # Write CSV files
    if not summary_ic.empty:
        summary_ic.to_csv(ic_csv_path, index=False)
        logger.info(f"Saved IC summary to {ic_csv_path}")

    if not summary_rank_ic.empty:
        summary_rank_ic.to_csv(rank_ic_csv_path, index=False)
        logger.info(f"Saved Rank-IC summary to {rank_ic_csv_path}")

    if not portfolio_summary.empty:
        portfolio_summary.to_csv(portfolio_csv_path, index=False)
        logger.info(f"Saved portfolio summary to {portfolio_csv_path}")


def run_factor_analysis_from_args(args) -> int:
    """Run factor analysis from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path("output") / "factor_analysis"

        # Load price data
        prices = load_price_data(
            freq=args.freq,
            symbols=getattr(args, "symbols", None),
            symbols_file=getattr(args, "symbols_file", None),
            universe=getattr(args, "universe", None),
            start_date=args.start_date,
            end_date=args.end_date,
            data_source=args.data_source,
        )

        # Compute factors
        logger.info("=" * 60)
        logger.info("Factor Analysis")
        logger.info(f"Factor Set: {args.factor_set}")
        logger.info(f"Forward Horizon: {args.horizon_days} days")
        logger.info(f"Quantiles: {args.quantiles}")
        logger.info("=" * 60)

        # Determine output directory for alt-data events/news/macro
        altdata_output_dir = None
        if args.factor_set in (
            "alt_earnings_insider",
            "core+alt",
            "alt_news_macro",
            "core+alt_news",
            "core+alt_full",
        ):
            # Use same output_dir if provided, otherwise default to output/altdata
            if args.output_dir:
                altdata_output_dir = Path(args.output_dir).parent / "altdata"
            else:
                altdata_output_dir = Path("output") / "altdata"

        factors_df = compute_factors(
            prices,
            factor_set=args.factor_set,
            output_dir=altdata_output_dir,
        )

        # Add forward returns
        logger.info(f"Adding forward returns (horizon: {args.horizon_days} days)...")
        factors_df = add_forward_returns(
            factors_df, horizon_days=args.horizon_days, return_type="log"
        )

        # Determine forward return column name
        if isinstance(args.horizon_days, list):
            fwd_return_col = f"fwd_ret_{args.horizon_days[0]}"
        else:
            fwd_return_col = f"fwd_return_{args.horizon_days}d"

        # Get factor columns (exclude metadata and forward return columns)
        exclude_cols = {
            "timestamp",
            "symbol",
            "close",
            "open",
            "high",
            "low",
            "volume",
            fwd_return_col,
        }
        factor_cols = [
            col
            for col in factors_df.columns
            if col not in exclude_cols and not col.startswith("fwd_")
        ]

        if not factor_cols:
            raise ValueError("No factor columns found after computation")

        logger.info(
            f"Analyzing {len(factor_cols)} factors: {factor_cols[:5]}..."
            + (
                f" (showing first 5 of {len(factor_cols)})"
                if len(factor_cols) > 5
                else ""
            )
        )

        # Compute IC and Rank-IC
        logger.info("Computing IC and Rank-IC...")
        ic_df = compute_ic(
            factors_df,
            forward_returns_col=fwd_return_col,
            group_col="symbol",
            method="pearson",
        )

        rank_ic_df = compute_rank_ic(
            factors_df, forward_returns_col=fwd_return_col, group_col="symbol"
        )

        # Summarize IC
        summary_ic = summarize_ic_series(ic_df) if not ic_df.empty else pd.DataFrame()
        summary_rank_ic = (
            summarize_ic_series(rank_ic_df) if not rank_ic_df.empty else pd.DataFrame()
        )

        # Build factor portfolios
        logger.info("Building factor portfolios...")
        portfolio_returns = build_factor_portfolio_returns(
            factors_df,
            factor_cols=factor_cols,
            forward_returns_col=fwd_return_col,
            quantiles=args.quantiles,
            min_obs=10,
        )

        # Build Long/Short portfolios
        ls_returns = (
            build_long_short_portfolio_returns(portfolio_returns)
            if not portfolio_returns.empty
            else pd.DataFrame()
        )

        # Summarize portfolios
        portfolio_summary = (
            summarize_factor_portfolios(ls_returns)
            if not ls_returns.empty
            else pd.DataFrame()
        )

        # Add deflated Sharpe Ratio if testing multiple factors
        if not portfolio_summary.empty and len(factor_cols) > 1:
            logger.info("Computing deflated Sharpe ratios...")
            portfolio_summary["deflated_sharpe"] = portfolio_summary.apply(
                lambda row: compute_deflated_sharpe_ratio(
                    sharpe=row["sharpe"],
                    n_obs=row["n_periods"],
                    n_trials=len(factor_cols),
                ),
                axis=1,
            )

        # Print summary tables
        logger.info("")
        logger.info("=" * 60)
        logger.info("IC Summary (Pearson Correlation)")
        logger.info("=" * 60)

        if not summary_ic.empty:
            display_df = summary_ic[
                ["factor", "mean_ic", "std_ic", "ic_ir", "hit_ratio", "count"]
            ].copy()
            display_df = display_df.round(4)
            print(
                "\n"
                + tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)
            )
        else:
            logger.warning("No IC summary data available")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Portfolio Performance Summary")
        logger.info("=" * 60)

        if not portfolio_summary.empty:
            display_cols = [
                "factor",
                "annualized_return",
                "annualized_vol",
                "sharpe",
                "t_stat",
                "win_ratio",
                "max_drawdown",
            ]
            if "deflated_sharpe" in portfolio_summary.columns:
                display_cols.append("deflated_sharpe")
            display_df = portfolio_summary[display_cols].copy()
            display_df = display_df.round(4)
            print(
                "\n"
                + tabulate(display_df, headers="keys", tablefmt="grid", showindex=False)
            )
        else:
            logger.warning("No portfolio summary data available")

        # Write reports
        write_factor_analysis_report(
            summary_ic=summary_ic,
            summary_rank_ic=summary_rank_ic,
            portfolio_summary=portfolio_summary,
            output_dir=output_dir,
            factor_set=args.factor_set,
            horizon_days=args.horizon_days,
            freq=args.freq,
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("Factor Analysis Completed Successfully")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Factor analysis failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point for direct script execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run comprehensive factor analysis (IC + Portfolio evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_factor_analysis.py --freq 1d --symbols-file config/macro_world_etfs_tickers.txt --data-source local --start-date 2010-01-01 --end-date 2025-12-03 --factor-set core+vol_liquidity --horizon-days 20
        """,
    )

    parser.add_argument(
        "--freq", type=str, required=True, choices=["1d", "5min"], help="Frequency"
    )
    parser.add_argument("--symbols", type=str, nargs="+", help="List of symbols")
    parser.add_argument("--symbols-file", type=str, help="Path to symbols file")
    parser.add_argument(
        "--universe", type=str, help="Path to universe file (alias for --symbols-file)"
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        choices=["local", "yahoo", "finnhub", "twelve_data"],
        help="Data source",
    )
    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--factor-set",
        type=str,
        default="core",
        choices=list_available_factor_sets(),
        help="Factor set",
    )
    parser.add_argument(
        "--horizon-days", type=int, default=20, help="Forward return horizon in days"
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="Number of quantiles for portfolio analysis",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: output/factor_analysis)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")

    return run_factor_analysis_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
