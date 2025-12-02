"""Compare Trend Baseline vs Event Insider Shipping Strategy.

This script runs both strategies on the same price data and compares their performance metrics.

Example usage:
    python scripts/compare_strategies_trend_vs_event.py --freq 1d
    python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet --no-costs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Import core modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS
from src.assembled_core.costs import CostModel, get_default_cost_model
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.ema_config import get_default_ema_config
from src.assembled_core.logging_utils import setup_logging
from src.assembled_core.portfolio.position_sizing import compute_target_positions_from_trend_signals
from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest
from src.assembled_core.qa.metrics import PerformanceMetrics, compute_all_metrics

# Import strategy functions from run_backtest_strategy
from scripts.run_backtest_strategy import (
    create_event_insider_shipping_signal_fn,
    create_position_sizing_fn,
    create_trend_baseline_signal_fn,
)

logger = setup_logging(level="INFO")


def create_event_position_sizing_fn():
    """Create a position sizing function for event strategy (reuses trend logic for now)."""
    return create_position_sizing_fn()


def run_strategy_backtest(
    strategy_name: str,
    prices: pd.DataFrame,
    freq: str,
    start_capital: float,
    cost_model: CostModel,
    with_costs: bool = True,
) -> dict[str, Any]:
    """Run a single strategy backtest and return key metrics.
    
    Args:
        strategy_name: Strategy name ("trend_baseline" or "event_insider_shipping")
        prices: Price DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        freq: Trading frequency ("1d" or "5min")
        start_capital: Starting capital
        cost_model: Cost model configuration
        with_costs: Whether to include transaction costs
    
    Returns:
        Dictionary with key metrics:
        - strategy: Strategy name
        - total_return: Total return (as float, e.g., 0.05 for 5%)
        - cagr: CAGR (as float or None)
        - sharpe_ratio: Sharpe ratio (as float or None)
        - sortino_ratio: Sortino ratio (as float or None)
        - max_drawdown_pct: Max drawdown in percent (as float, negative)
        - volatility: Volatility (as float or None)
        - total_trades: Total number of trades (as int or None)
        - hit_rate: Win rate (as float or None)
        - profit_factor: Profit factor (as float or None)
        - turnover: Turnover (as float or None)
        - final_pf: Final performance factor (as float)
        - end_equity: Ending equity (as float)
    """
    logger.info(f"Running {strategy_name} strategy...")
    
    # Create signal and position sizing functions
    if strategy_name == "trend_baseline":
        ema_config = get_default_ema_config(freq)
        signal_fn = create_trend_baseline_signal_fn(
            ma_fast=ema_config.fast,
            ma_slow=ema_config.slow
        )
        position_sizing_fn = create_position_sizing_fn()
    elif strategy_name == "event_insider_shipping":
        signal_fn = create_event_insider_shipping_signal_fn()
        position_sizing_fn = create_event_position_sizing_fn()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    # Run backtest
    result: BacktestResult = run_portfolio_backtest(
        prices=prices,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        start_capital=start_capital,
        commission_bps=cost_model.commission_bps,
        spread_w=cost_model.spread_w,
        impact_w=cost_model.impact_w,
        include_costs=with_costs,
        include_trades=True,
        include_signals=False,
        include_targets=False,
        rebalance_freq=freq,
        compute_features=True
    )
    
    # Compute metrics
    metrics: PerformanceMetrics = compute_all_metrics(
        equity=result.equity,
        trades=result.trades,
        start_capital=start_capital,
        freq=freq,
        risk_free_rate=0.0
    )
    
    # Extract key metrics into dictionary
    return {
        "strategy": strategy_name,
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "sortino_ratio": metrics.sortino_ratio,
        "max_drawdown_pct": metrics.max_drawdown_pct,
        "volatility": metrics.volatility,
        "total_trades": metrics.total_trades,
        "hit_rate": metrics.hit_rate,
        "profit_factor": metrics.profit_factor,
        "turnover": metrics.turnover,
        "final_pf": metrics.final_pf,
        "end_equity": metrics.end_equity,
    }


def format_metric(value: float | None, format_type: str = "percent") -> str:
    """Format a metric value for display.
    
    Args:
        value: Metric value (may be None)
        format_type: Format type ("percent", "float", "int")
    
    Returns:
        Formatted string (or "N/A" if None)
    """
    if value is None:
        return "N/A"
    
    if format_type == "percent":
        # Handle very large or negative percentages
        if abs(value) > 1000:
            return f"{value:.2f}%"
        return f"{value:.2%}"
    elif format_type == "float":
        return f"{value:.4f}"
    elif format_type == "int":
        return f"{int(value)}"
    else:
        return str(value)


def write_comparison_markdown(
    trend_metrics: dict[str, Any],
    event_metrics: dict[str, Any],
    output_path: Path,
    freq: str,
    start_capital: float,
) -> None:
    """Write comparison results to Markdown file.
    
    Args:
        trend_metrics: Metrics dictionary for trend_baseline strategy
        event_metrics: Metrics dictionary for event_insider_shipping strategy
        output_path: Path to output Markdown file
        freq: Trading frequency
        start_capital: Starting capital
    """
    lines = [
        "# Strategy Comparison: Trend Baseline vs Event Insider Shipping",
        "",
        f"**Frequency:** {freq}",
        f"**Start Capital:** ${start_capital:,.2f}",
        "",
        "## Performance Metrics",
        "",
        "| Metric | Trend Baseline | Event Insider Shipping |",
        "|--------|----------------|------------------------|",
        f"| **Total Return** | {format_metric(trend_metrics['total_return'])} | {format_metric(event_metrics['total_return'])} |",
        f"| **CAGR** | {format_metric(trend_metrics['cagr'])} | {format_metric(event_metrics['cagr'])} |",
        f"| **Final PF** | {format_metric(trend_metrics['final_pf'], 'float')} | {format_metric(event_metrics['final_pf'], 'float')} |",
        f"| **End Equity** | ${format_metric(trend_metrics['end_equity'], 'float')} | ${format_metric(event_metrics['end_equity'], 'float')} |",
        "",
        "## Risk Metrics",
        "",
        "| Metric | Trend Baseline | Event Insider Shipping |",
        "|--------|----------------|------------------------|",
        f"| **Sharpe Ratio** | {format_metric(trend_metrics['sharpe_ratio'], 'float')} | {format_metric(event_metrics['sharpe_ratio'], 'float')} |",
        f"| **Sortino Ratio** | {format_metric(trend_metrics['sortino_ratio'], 'float')} | {format_metric(event_metrics['sortino_ratio'], 'float')} |",
        f"| **Max Drawdown** | {format_metric(trend_metrics['max_drawdown_pct'])} | {format_metric(event_metrics['max_drawdown_pct'])} |",
        f"| **Volatility** | {format_metric(trend_metrics['volatility'])} | {format_metric(event_metrics['volatility'])} |",
        "",
        "## Trade Metrics",
        "",
        "| Metric | Trend Baseline | Event Insider Shipping |",
        "|--------|----------------|------------------------|",
        f"| **Total Trades** | {format_metric(trend_metrics['total_trades'], 'int')} | {format_metric(event_metrics['total_trades'], 'int')} |",
        f"| **Hit Rate** | {format_metric(trend_metrics['hit_rate'])} | {format_metric(event_metrics['hit_rate'])} |",
        f"| **Profit Factor** | {format_metric(event_metrics['profit_factor'], 'float')} | {format_metric(event_metrics['profit_factor'], 'float')} |",
        f"| **Turnover** | {format_metric(trend_metrics['turnover'], 'float')} | {format_metric(event_metrics['turnover'], 'float')} |",
    ]
    
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Comparison summary written: {output_path}")


def write_comparison_csv(
    trend_metrics: dict[str, Any],
    event_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Write comparison results to CSV file.
    
    Args:
        trend_metrics: Metrics dictionary for trend_baseline strategy
        event_metrics: Metrics dictionary for event_insider_shipping strategy
        output_path: Path to output CSV file
    """
    df = pd.DataFrame([trend_metrics, event_metrics])
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison CSV written: {output_path}")


def print_comparison_table(
    trend_metrics: dict[str, Any],
    event_metrics: dict[str, Any],
) -> None:
    """Print formatted comparison table to console.
    
    Args:
        trend_metrics: Metrics dictionary for trend_baseline strategy
        event_metrics: Metrics dictionary for event_insider_shipping strategy
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Strategy Comparison: Trend Baseline vs Event Insider Shipping")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Performance Metrics:")
    logger.info(f"  Total Return:     Trend={format_metric(trend_metrics['total_return']):>10}  Event={format_metric(event_metrics['total_return']):>10}")
    logger.info(f"  CAGR:             Trend={format_metric(trend_metrics['cagr']):>10}  Event={format_metric(event_metrics['cagr']):>10}")
    logger.info(f"  Final PF:         Trend={format_metric(trend_metrics['final_pf'], 'float'):>10}  Event={format_metric(event_metrics['final_pf'], 'float'):>10}")
    logger.info(f"  End Equity:       Trend=${format_metric(trend_metrics['end_equity'], 'float'):>9}  Event=${format_metric(event_metrics['end_equity'], 'float'):>9}")
    logger.info("")
    logger.info("Risk Metrics:")
    logger.info(f"  Sharpe Ratio:     Trend={format_metric(trend_metrics['sharpe_ratio'], 'float'):>10}  Event={format_metric(event_metrics['sharpe_ratio'], 'float'):>10}")
    logger.info(f"  Sortino Ratio:    Trend={format_metric(trend_metrics['sortino_ratio'], 'float'):>10}  Event={format_metric(event_metrics['sortino_ratio'], 'float'):>10}")
    logger.info(f"  Max Drawdown:     Trend={format_metric(trend_metrics['max_drawdown_pct']):>10}  Event={format_metric(event_metrics['max_drawdown_pct']):>10}")
    logger.info(f"  Volatility:       Trend={format_metric(trend_metrics['volatility']):>10}  Event={format_metric(event_metrics['volatility']):>10}")
    logger.info("")
    logger.info("Trade Metrics:")
    logger.info(f"  Total Trades:     Trend={format_metric(trend_metrics['total_trades'], 'int'):>10}  Event={format_metric(event_metrics['total_trades'], 'int'):>10}")
    logger.info(f"  Hit Rate:         Trend={format_metric(trend_metrics['hit_rate']):>10}  Event={format_metric(event_metrics['hit_rate']):>10}")
    logger.info(f"  Profit Factor:    Trend={format_metric(trend_metrics['profit_factor'], 'float'):>10}  Event={format_metric(event_metrics['profit_factor'], 'float'):>10}")
    logger.info(f"  Turnover:         Trend={format_metric(trend_metrics['turnover'], 'float'):>10}  Event={format_metric(event_metrics['turnover'], 'float'):>10}")
    logger.info("")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Trend Baseline vs Event Insider Shipping Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_strategies_trend_vs_event.py --freq 1d
  python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet
  python scripts/compare_strategies_trend_vs_event.py --freq 1d --no-costs --start-capital 50000
        """
    )
    
    parser.add_argument(
        "--freq",
        type=str,
        default="1d",
        choices=SUPPORTED_FREQS,
        help=f"Trading frequency ({'/'.join(SUPPORTED_FREQS)}, default: 1d)"
    )
    
    parser.add_argument(
        "--price-file",
        type=Path,
        default=None,
        help="Path to price file (default: data/sample/eod_sample.parquet)"
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
        help="Include transaction costs (default: True)"
    )
    
    parser.add_argument(
        "--no-costs",
        action="store_false",
        dest="with_costs",
        help="Disable transaction costs"
    )
    
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: output/strategy_compare/trend_vs_event)"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    try:
        args = parse_args()
        
        # Determine output directory
        if args.out:
            output_dir = Path(args.out)
        else:
            output_dir = OUTPUT_DIR / "strategy_compare" / "trend_vs_event"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine price file
        if args.price_file:
            price_file = args.price_file
        else:
            price_file = ROOT / "data" / "sample" / "eod_sample.parquet"
        
        if not price_file.exists():
            logger.error(f"Price file not found: {price_file}")
            logger.info("Please provide --price-file or ensure data/sample/eod_sample.parquet exists")
            sys.exit(1)
        
        logger.info("=" * 80)
        logger.info("Strategy Comparison: Trend Baseline vs Event Insider Shipping")
        logger.info("=" * 80)
        logger.info(f"Frequency: {args.freq}")
        logger.info(f"Price File: {price_file}")
        logger.info(f"Start Capital: ${args.start_capital:,.2f}")
        logger.info(f"With Costs: {args.with_costs}")
        logger.info(f"Output Dir: {output_dir}")
        logger.info("")
        
        # Load price data
        logger.info("Loading price data...")
        prices = load_eod_prices(price_file=price_file, freq=args.freq)
        
        if prices.empty:
            logger.error("No price data loaded")
            sys.exit(1)
        
        logger.info(f"Loaded {len(prices)} rows for {prices['symbol'].nunique()} symbols")
        logger.info(f"Date range: {prices['timestamp'].min()} to {prices['timestamp'].max()}")
        logger.info("")
        
        # Get cost model
        cost_model = get_default_cost_model()
        logger.info(f"Cost Model: commission_bps={cost_model.commission_bps}, spread_w={cost_model.spread_w}, impact_w={cost_model.impact_w}")
        logger.info("")
        
        # Run both strategies
        logger.info("Running backtests...")
        logger.info("")
        
        trend_metrics = run_strategy_backtest(
            strategy_name="trend_baseline",
            prices=prices,
            freq=args.freq,
            start_capital=args.start_capital,
            cost_model=cost_model,
            with_costs=args.with_costs,
        )
        
        logger.info("")
        
        event_metrics = run_strategy_backtest(
            strategy_name="event_insider_shipping",
            prices=prices,
            freq=args.freq,
            start_capital=args.start_capital,
            cost_model=cost_model,
            with_costs=args.with_costs,
        )
        
        # Write comparison files
        logger.info("")
        logger.info("Writing comparison files...")
        
        markdown_path = output_dir / "comparison_summary.md"
        write_comparison_markdown(
            trend_metrics=trend_metrics,
            event_metrics=event_metrics,
            output_path=markdown_path,
            freq=args.freq,
            start_capital=args.start_capital,
        )
        
        csv_path = output_dir / "comparison_summary.csv"
        write_comparison_csv(
            trend_metrics=trend_metrics,
            event_metrics=event_metrics,
            output_path=csv_path,
        )
        
        # Print comparison table
        print_comparison_table(trend_metrics, event_metrics)
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

