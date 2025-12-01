# scripts/run_backtest_strategy.py
"""Strategy Backtest CLI Runner.

This script provides a command-line interface for running strategy backtests.
It uses the portfolio-level backtest engine with configurable signal and position sizing functions.

Example usage:
    python scripts/run_backtest_strategy.py --freq 1d --universe watchlist.txt --start-capital 10000 --generate-report
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS, get_base_dir
from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.data.prices_ingest import load_eod_prices, load_eod_prices_for_universe
from src.assembled_core.ema_config import EmaConfig, get_default_ema_config
from src.assembled_core.logging_utils import setup_logging
from src.assembled_core.portfolio.position_sizing import compute_target_positions_from_trend_signals
from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest
from src.assembled_core.qa.metrics import compute_all_metrics
from src.assembled_core.qa.qa_gates import QAResult, evaluate_all_gates
from src.assembled_core.reports.daily_qa_report import generate_qa_report
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices

logger = setup_logging(level="INFO")


def create_trend_baseline_signal_fn(ma_fast: int, ma_slow: int):
    """Create a signal function for trend baseline strategy.
    
    Args:
        ma_fast: Fast moving average window
        ma_slow: Slow moving average window
    
    Returns:
        Callable that takes prices DataFrame and returns signals DataFrame
    """
    def signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend signals from prices."""
        return generate_trend_signals_from_prices(
            prices_df,
            ma_fast=ma_fast,
            ma_slow=ma_slow
        )
    
    return signal_fn


def create_position_sizing_fn():
    """Create a position sizing function for trend baseline strategy.
    
    Returns:
        Callable that takes signals DataFrame and capital, returns target positions DataFrame
    """
    def position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
        """Compute target positions from trend signals."""
        return compute_target_positions_from_trend_signals(
            signals_df,
            total_capital=capital,
            top_n=None,  # No limit on number of positions
            min_score=0.0  # Accept all signals
        )
    
    return position_sizing_fn


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run strategy backtest using portfolio-level backtest engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic backtest with default settings
  python scripts/run_backtest_strategy.py --freq 1d

  # Backtest with universe file and report generation
  python scripts/run_backtest_strategy.py --freq 1d --universe watchlist.txt --generate-report

  # Backtest with custom price file and cost parameters
  python scripts/run_backtest_strategy.py --freq 1d --price-file data/sample/eod_sample.parquet --commission-bps 0.5 --spread-w 0.3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--freq",
        type=str,
        required=True,
        choices=SUPPORTED_FREQS,
        help=f"Trading frequency ({'/'.join(SUPPORTED_FREQS)})"
    )
    
    # Optional arguments
    parser.add_argument(
        "--price-file",
        type=Path,
        default=None,
        help="Explicit path to price file (overrides default path)"
    )
    
    parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Path to universe file (default: watchlist.txt in repo root)"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="trend_baseline",
        help="Strategy name (default: trend_baseline)"
    )
    
    parser.add_argument(
        "--start-capital",
        type=float,
        default=10000.0,
        help="Starting capital (default: 10000.0)"
    )
    
    parser.add_argument(
        "--with-costs",
        action="store_true",
        default=True,
        help="Include transaction costs in backtest (default: True)"
    )
    
    parser.add_argument(
        "--no-costs",
        action="store_false",
        dest="with_costs",
        help="Disable transaction costs (use cost-free simulation)"
    )
    
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="Commission in basis points (overrides default cost model)"
    )
    
    parser.add_argument(
        "--spread-w",
        type=float,
        default=None,
        help="Spread weight (overrides default cost model)"
    )
    
    parser.add_argument(
        "--impact-w",
        type=float,
        default=None,
        help="Market impact weight (overrides default cost model)"
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: config.OUTPUT_DIR)"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=False,
        help="Generate QA report after backtest"
    )
    
    return parser.parse_args()


def load_price_data(args: argparse.Namespace, output_dir: Path | None = None) -> pd.DataFrame:
    """Load price data based on CLI arguments.
    
    Args:
        args: Parsed command-line arguments
        output_dir: Output directory for price data (default: None, uses args.out or OUTPUT_DIR)
    
    Returns:
        DataFrame with price data (timestamp, symbol, open, high, low, close, volume)
    
    Raises:
        FileNotFoundError: If price file or universe file not found
        ValueError: If no data found for universe symbols
    """
    # Get output directory for price data loading
    if output_dir is None:
        output_dir = Path(args.out) if args.out else OUTPUT_DIR
    
    if args.price_file:
        logger.info(f"Loading prices from explicit file: {args.price_file}")
        prices = load_eod_prices(price_file=args.price_file, freq=args.freq)
    elif args.universe:
        logger.info(f"Loading prices for universe: {args.universe}")
        prices = load_eod_prices_for_universe(
            universe_file=args.universe, 
            data_dir=output_dir,
            freq=args.freq
        )
    else:
        # Default: use watchlist.txt
        logger.info("Loading prices for default universe (watchlist.txt)")
        prices = load_eod_prices_for_universe(
            universe_file=None, 
            data_dir=output_dir,
            freq=args.freq
        )
    
    logger.info(f"Loaded {len(prices)} rows for {prices['symbol'].nunique()} symbols")
    logger.info(f"Date range: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
    
    return prices


def get_cost_model(args: argparse.Namespace) -> CostModel:
    """Get cost model from CLI arguments or defaults.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        CostModel instance
    """
    if args.commission_bps is not None or args.spread_w is not None or args.impact_w is not None:
        # Use CLI overrides
        default = get_default_cost_model()
        return CostModel(
            commission_bps=args.commission_bps if args.commission_bps is not None else default.commission_bps,
            spread_w=args.spread_w if args.spread_w is not None else default.spread_w,
            impact_w=args.impact_w if args.impact_w is not None else default.impact_w
        )
    else:
        # Use default cost model
        return get_default_cost_model()


def main() -> int:
    """Main entry point for strategy backtest CLI."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set output directory
        output_dir = Path(args.out) if args.out else OUTPUT_DIR
        
        # Log start
        logger.info("=" * 60)
        logger.info("Strategy Backtest CLI")
        logger.info("=" * 60)
        logger.info(f"Frequency: {args.freq}")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Start Capital: {args.start_capital:.2f}")
        logger.info(f"With Costs: {args.with_costs}")
        logger.info(f"Output Dir: {output_dir}")
        
        # Load price data
        logger.info("")
        logger.info("Loading price data...")
        prices = load_price_data(args)
        
        if prices.empty:
            logger.error("No price data loaded")
            return 1
        
        # Get cost model
        cost_model = get_cost_model(args)
        logger.info(f"Cost Model: commission_bps={cost_model.commission_bps}, spread_w={cost_model.spread_w}, impact_w={cost_model.impact_w}")
        
        # Create signal and position sizing functions
        logger.info("")
        logger.info("Setting up strategy functions...")
        
        if args.strategy == "trend_baseline":
            # Get EMA defaults for frequency
            ema_config = get_default_ema_config(args.freq)
            logger.info(f"EMA Config: fast={ema_config.fast}, slow={ema_config.slow}")
            
            signal_fn = create_trend_baseline_signal_fn(
                ma_fast=ema_config.fast,
                ma_slow=ema_config.slow
            )
            position_sizing_fn = create_position_sizing_fn()
        else:
            logger.error(f"Unknown strategy: {args.strategy}")
            logger.info(f"Supported strategies: trend_baseline")
            return 1
        
        # Run backtest
        logger.info("")
        logger.info("Running backtest...")
        result: BacktestResult = run_portfolio_backtest(
            prices=prices,
            signal_fn=signal_fn,
            position_sizing_fn=position_sizing_fn,
            start_capital=args.start_capital,
            commission_bps=cost_model.commission_bps,
            spread_w=cost_model.spread_w,
            impact_w=cost_model.impact_w,
            include_costs=args.with_costs,
            include_trades=True,  # Always include trades for QA
            include_signals=False,
            include_targets=False,
            rebalance_freq=args.freq,
            compute_features=True
        )
        
        logger.info(f"Backtest completed: {len(result.equity)} equity points")
        
        # Compute comprehensive metrics using qa.metrics
        logger.info("")
        logger.info("Computing performance metrics...")
        metrics = compute_all_metrics(
            equity=result.equity,
            trades=result.trades,
            start_capital=args.start_capital,
            freq=args.freq,
            risk_free_rate=0.0
        )
        
        logger.info(f"Total Return: {metrics.total_return:.2%}")
        logger.info(f"CAGR: {metrics.cagr:.2%}" if metrics.cagr else "CAGR: N/A")
        logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}" if metrics.sharpe_ratio else "Sharpe Ratio: N/A")
        logger.info(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        logger.info(f"Total Trades: {metrics.total_trades if metrics.total_trades else 0}")
        
        # Evaluate QA gates
        logger.info("")
        logger.info("Evaluating QA gates...")
        gate_result = evaluate_all_gates(metrics)
        
        logger.info(f"QA Overall Result: {gate_result.overall_result.value.upper()}")
        logger.info(f"Gates: {gate_result.passed_gates} OK, {gate_result.warning_gates} WARNING, {gate_result.blocked_gates} BLOCK")
        
        # Log gate details
        for gate in gate_result.gate_results:
            status_icon = "✓" if gate.result == QAResult.OK else "⚠" if gate.result == QAResult.WARNING else "✗"
            logger.info(f"  {status_icon} {gate.gate_name}: {gate.result.value.upper()} - {gate.reason}")
        
        # Generate report if requested
        if args.generate_report:
            logger.info("")
            logger.info("Generating QA report...")
            
            # Build config info
            ema_config = get_default_ema_config(args.freq)
            config_info = {
                "strategy": args.strategy,
                "freq": args.freq,
                "start_capital": args.start_capital,
                "ema_fast": ema_config.fast,
                "ema_slow": ema_config.slow,
                "with_costs": args.with_costs,
                "commission_bps": cost_model.commission_bps,
                "spread_w": cost_model.spread_w,
                "impact_w": cost_model.impact_w
            }
            
            # Generate report
            report_path = generate_qa_report(
                metrics=metrics,
                gate_result=gate_result,
                strategy_name=args.strategy,
                freq=args.freq,
                equity_curve_path=None,  # Could save equity curve to file if needed
                data_start_date=metrics.start_date,
                data_end_date=metrics.end_date,
                config_info=config_info,
                output_dir=output_dir / "reports"
            )
            
            logger.info(f"QA Report written: {report_path}")
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("Backtest Summary")
        logger.info("=" * 60)
        logger.info(f"Final PF: {metrics.final_pf:.4f}")
        logger.info(f"Total Return: {metrics.total_return:.2%}")
        if metrics.cagr:
            logger.info(f"CAGR: {metrics.cagr:.2%}")
        if metrics.sharpe_ratio:
            logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
        logger.info(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        logger.info(f"Total Trades: {metrics.total_trades if metrics.total_trades else 0}")
        logger.info(f"QA Result: {gate_result.overall_result.value.upper()}")
        logger.info("=" * 60)
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

