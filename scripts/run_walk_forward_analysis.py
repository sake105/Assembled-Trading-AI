"""Walk-Forward Analysis Runner (B3).

This script provides a lightweight command-line interface for running walk-forward
backtest analysis on strategies.

**Note:** This is primarily a research tool for systematic strategy validation.
It is not intended for daily EOD jobs but for:
- Strategy research and development
- Out-of-sample validation
- Parameter sensitivity analysis
- Model stability testing

Usage:
    python scripts/run_walk_forward_analysis.py \
        --freq 1d \
        --strategy multifactor_long_short \
        --start-date 2020-01-01 \
        --end-date 2023-12-31 \
        --train-window 252 \
        --test-window 63 \
        --mode rolling
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config.settings import get_settings
from src.assembled_core.costs import get_default_cost_model
from src.assembled_core.data.prices_ingest import load_eod_prices_for_universe
from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    make_engine_backtest_fn,
    run_walk_forward_backtest,
)
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.strategies.multifactor_long_short import (
    compute_multifactor_long_short_positions,
    generate_multifactor_long_short_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_signal_and_sizing_fns(args: argparse.Namespace):
    """Create signal and position sizing functions based on strategy argument.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (signal_fn, position_sizing_fn)
    """
    if args.strategy == "trend_baseline":
        from src.assembled_core.ema_config import get_default_ema_config
        from src.assembled_core.portfolio.position_sizing import (
            compute_target_positions_from_trend_signals,
        )

        ema_config = get_default_ema_config(args.freq)

        def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
            return generate_trend_signals_from_prices(
                prices_df,
                ma_fast=ema_config.fast,
                ma_slow=ema_config.slow,
            )

        def position_sizing_fn(
            signals_df: pd.DataFrame, capital: float
        ) -> pd.DataFrame:
            return compute_target_positions_from_trend_signals(
                signals_df,
                total_capital=capital,
                top_n=None,
                min_score=0.0,
            )

        return signal_fn, position_sizing_fn

    elif args.strategy == "multifactor_long_short":
        if not args.bundle_path:
            raise ValueError(
                "--bundle-path is required for multifactor_long_short strategy"
            )

        def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
            return generate_multifactor_long_short_signals(
                prices_df,
                bundle_path=args.bundle_path,
                rebalance_freq=args.rebalance_freq,
            )

        def position_sizing_fn(
            signals_df: pd.DataFrame, capital: float
        ) -> pd.DataFrame:
            return compute_multifactor_long_short_positions(
                signals_df,
                total_capital=capital,
                max_gross_exposure=args.max_gross_exposure,
            )

        return signal_fn, position_sizing_fn

    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")


def run_walk_forward_analysis(args: argparse.Namespace) -> int:
    """Run walk-forward analysis.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        # Load price data
        logger.info("Loading price data...")
        settings = get_settings()

        prices_df = load_eod_prices_for_universe(
            freq=args.freq,
            universe=args.universe,
            symbols_file=args.symbols_file,
            symbols=args.symbols,
            data_source=args.data_source,
            start_date=args.start_date,
            end_date=args.end_date,
            settings=settings,
        )

        if prices_df.empty:
            logger.error("No price data loaded. Check data source and date range.")
            return 1

        logger.info(
            f"Loaded prices: {len(prices_df)} rows, {prices_df['symbol'].nunique()} symbols"
        )

        # Create signal and position sizing functions
        logger.info(f"Creating {args.strategy} strategy functions...")
        signal_fn, position_sizing_fn = create_signal_and_sizing_fns(args)

        # Build cost model
        cost_model = get_default_cost_model()
        if args.commission_bps is not None:
            cost_model.commission_bps = args.commission_bps
        if args.spread_w is not None:
            cost_model.spread_w = args.spread_w
        if args.impact_w is not None:
            cost_model.impact_w = args.impact_w

        # Create backtest function factory
        logger.info("Creating backtest function...")
        backtest_fn = make_engine_backtest_fn(
            prices=prices_df,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            timestamp_col="timestamp",
            group_col="symbol",
            price_col="close",
            start_capital=args.start_capital,
            commission_bps=cost_model.commission_bps,
            spread_w=cost_model.spread_w,
            impact_w=cost_model.impact_w,
            include_costs=args.with_costs,
            include_trades=False,  # Not needed for walk-forward summary
            rebalance_freq=args.rebalance_freq
            if args.strategy == "multifactor_long_short"
            else args.freq,
            compute_features=args.strategy != "multifactor_long_short",
        )

        # Build Walk-Forward config
        config = WalkForwardConfig(
            start_date=pd.Timestamp(args.start_date, tz="UTC"),
            end_date=pd.Timestamp(args.end_date, tz="UTC"),
            train_window_days=args.train_window if args.mode == "rolling" else None,
            test_window_days=args.test_window,
            mode=args.mode,
            step_size_days=args.step_size,
            min_train_periods=args.min_train_periods,
            min_test_periods=args.min_test_periods,
            max_splits=args.max_splits,
        )

        # Run walk-forward analysis
        logger.info("Running walk-forward analysis...")
        result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)

        logger.info(
            f"Walk-forward analysis completed: {result.aggregated_metrics['n_successful_splits']}/{result.aggregated_metrics['n_splits']} splits successful"
        )

        # Write outputs
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else ROOT / "output" / "walk_forward"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        summary_csv_path = output_dir / "walk_forward_summary.csv"
        result.summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved walk-forward summary to {summary_csv_path}")

        # Aggregated metrics CSV
        metrics_df = pd.DataFrame([result.aggregated_metrics])
        metrics_csv_path = output_dir / "walk_forward_aggregated_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        logger.info(f"Saved aggregated metrics to {metrics_csv_path}")

        # Print summary
        logger.info("")
        logger.info("Walk-Forward Analysis Summary:")
        logger.info(f"  Total Splits: {result.aggregated_metrics['n_splits']}")
        logger.info(f"  Successful: {result.aggregated_metrics['n_successful_splits']}")
        logger.info(f"  Failed: {result.aggregated_metrics['n_failed_splits']}")

        if "mean_test_sharpe" in result.aggregated_metrics:
            logger.info(
                f"  Mean Test Sharpe: {result.aggregated_metrics['mean_test_sharpe']:.4f}"
            )
        if "std_test_sharpe" in result.aggregated_metrics:
            logger.info(
                f"  Std Test Sharpe: {result.aggregated_metrics['std_test_sharpe']:.4f}"
            )

        logger.info("")
        logger.info(f"Outputs saved to: {output_dir}")

        return 0

    except Exception as e:
        logger.error(f"Walk-forward analysis failed: {e}", exc_info=True)
        return 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest analysis (B3 - Research Tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rolling window: 1 year train, 3 months test
  python scripts/run_walk_forward_analysis.py \\
    --freq 1d \\
    --strategy trend_baseline \\
    --universe config/watchlist.txt \\
    --start-date 2020-01-01 \\
    --end-date 2023-12-31 \\
    --train-window 252 \\
    --test-window 63 \\
    --mode rolling

  # Expanding window
  python scripts/run_walk_forward_analysis.py \\
    --freq 1d \\
    --strategy multifactor_long_short \\
    --bundle-path config/factor_bundles/ai_tech_core_bundle.yaml \\
    --universe config/universe_ai_tech_tickers.txt \\
    --start-date 2020-01-01 \\
    --end-date 2023-12-31 \\
    --test-window 63 \\
    --mode expanding
        """,
    )

    # Data loading
    parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=["1d", "5min"],
        help="Trading frequency",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--universe", type=Path, help="Path to universe file (one symbol per line)"
    )
    group.add_argument("--symbols-file", type=Path, help="Path to symbols file")
    group.add_argument("--symbols", nargs="+", help="List of symbols")

    parser.add_argument(
        "--data-source",
        type=str,
        default="local",
        help="Data source ('local' or 'yahoo'). Default: local",
    )

    parser.add_argument(
        "--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date", type=str, required=True, help="End date (YYYY-MM-DD)"
    )

    # Strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="trend_baseline",
        choices=["trend_baseline", "multifactor_long_short"],
        help="Strategy type",
    )

    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=None,
        help="Path to factor bundle YAML (required for multifactor_long_short)",
    )

    parser.add_argument(
        "--rebalance-freq",
        type=str,
        default="M",
        help="Rebalancing frequency for multifactor strategy (default: M)",
    )

    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=1.0,
        help="Max gross exposure for multifactor strategy (default: 1.0)",
    )

    # Walk-Forward config
    parser.add_argument(
        "--mode",
        type=str,
        default="rolling",
        choices=["rolling", "expanding"],
        help="Window mode: rolling (fixed train size) or expanding (growing train size)",
    )

    parser.add_argument(
        "--train-window",
        type=int,
        default=252,
        help="Training window size in days (required for rolling mode, ignored for expanding)",
    )

    parser.add_argument(
        "--test-window", type=int, required=True, help="Test window size in days"
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Step size in days (default: test-window)",
    )

    parser.add_argument(
        "--min-train-periods",
        type=int,
        default=252,
        help="Minimum training periods (default: 252)",
    )

    parser.add_argument(
        "--min-test-periods",
        type=int,
        default=63,
        help="Minimum test periods (default: 63)",
    )

    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Maximum number of splits (default: no limit)",
    )

    # Backtest config
    parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000.0)",
    )

    parser.add_argument(
        "--with-costs",
        action="store_true",
        default=True,
        help="Include transaction costs (default: True)",
    )

    parser.add_argument(
        "--no-costs",
        action="store_false",
        dest="with_costs",
        help="Disable transaction costs",
    )

    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="Commission in basis points (override default)",
    )

    parser.add_argument(
        "--spread-w", type=float, default=None, help="Spread weight (override default)"
    )

    parser.add_argument(
        "--impact-w", type=float, default=None, help="Impact weight (override default)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/walk_forward)",
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    return run_walk_forward_analysis(args)


if __name__ == "__main__":
    sys.exit(main())
