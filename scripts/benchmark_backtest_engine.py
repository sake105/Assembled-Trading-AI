"""Performance benchmark script for backtest engine optimizations (P3).

This script measures the execution time of the optimized backtest engine
and logs timing information for performance analysis.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
from src.assembled_core.utils.timing import timed_block

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_benchmark_prices(n_symbols: int = 10, n_days: int = 252) -> pd.DataFrame:
    """Create synthetic price data for benchmarking.

    Args:
        n_symbols: Number of symbols to generate
        n_days: Number of trading days

    Returns:
        DataFrame with columns: timestamp, symbol, close
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D", tz="UTC")
    dates = dates[dates.weekday < 5]  # Only weekdays

    symbols = [f"SYM{i:02d}" for i in range(n_symbols)]
    rows = []

    for symbol in symbols:
        base_price = 100.0 + hash(symbol) % 100
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))

        for i, date in enumerate(dates):
            rows.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "close": prices[i],
                }
            )

    return (
        pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )


def simple_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Simple signal function for benchmarking."""
    signals = []
    for _, row in prices_df.iterrows():
        # Simple trend signal: LONG if price > 100-day moving average approximation
        direction = "LONG" if row["close"] > 100.0 else "FLAT"
        signals.append(
            {
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "direction": direction,
                "score": 1.0 if direction == "LONG" else 0.0,
            }
        )
    return pd.DataFrame(signals)


def simple_position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Simple position sizing function for benchmarking."""
    long_signals = signals_df[signals_df["direction"] == "LONG"]
    if long_signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])

    targets = []
    n = len(long_signals["symbol"].unique())
    for symbol in long_signals["symbol"].unique():
        targets.append(
            {
                "symbol": symbol,
                "target_weight": 1.0 / n if n > 0 else 0.0,
                "target_qty": (capital / n) / 100.0,  # Rough price estimate
            }
        )
    return pd.DataFrame(targets)


def run_benchmark(
    n_symbols: int = 10,
    n_days: int = 252,
    include_costs: bool = False,
    n_runs: int = 3,
    output_dir: Path | None = None,
) -> dict:
    """Run backtest benchmark and measure performance.

    Args:
        n_symbols: Number of symbols in universe
        n_days: Number of trading days
        include_costs: Whether to include transaction costs
        n_runs: Number of benchmark runs (for averaging)
        output_dir: Optional output directory for logs

    Returns:
        Dictionary with benchmark results
    """
    logger.info("=" * 60)
    logger.info("Backtest Engine Performance Benchmark (P3)")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  Symbols: {n_symbols}")
    logger.info(f"  Days: {n_days}")
    logger.info(f"  Include costs: {include_costs}")
    logger.info(f"  Runs: {n_runs}")
    logger.info("")

    # Create benchmark data
    logger.info("Generating benchmark price data...")
    prices = create_benchmark_prices(n_symbols=n_symbols, n_days=n_days)
    logger.info(f"  Generated {len(prices)} price records for {n_symbols} symbols")

    # Warm-up run (JIT compilation, cache warming)
    logger.info("Warm-up run...")
    _ = run_portfolio_backtest(
        prices=prices,
        signal_fn=simple_signal_fn,
        position_sizing_fn=simple_position_sizing_fn,
        start_capital=100000.0,
        include_costs=include_costs,
        include_trades=False,
    )

    # Benchmark runs
    logger.info(f"Running {n_runs} benchmark iterations...")
    run_times = []

    for run_id in range(n_runs):
        logger.info(f"  Run {run_id + 1}/{n_runs}...")

        start_time = time.perf_counter()
        with timed_block(f"benchmark_run_{run_id + 1}"):
            result = run_portfolio_backtest(
                prices=prices.copy(),
                signal_fn=simple_signal_fn,
                position_sizing_fn=simple_position_sizing_fn,
                start_capital=100000.0,
                include_costs=include_costs,
                include_trades=True,
            )
        end_time = time.perf_counter()

        run_time = end_time - start_time
        run_times.append(run_time)

        logger.info(f"    Time: {run_time:.3f}s")
        logger.info(f"    Final equity: {result.equity['equity'].iloc[-1]:.2f}")
        logger.info(f"    Trades: {result.metrics['trades']}")

    # Compute statistics
    avg_time = np.mean(run_times)
    std_time = np.std(run_times)
    min_time = np.min(run_times)
    max_time = np.max(run_times)

    logger.info("")
    logger.info("Benchmark Results:")
    logger.info(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    logger.info(f"  Min time: {min_time:.3f}s")
    logger.info(f"  Max time: {max_time:.3f}s")
    logger.info(f"  Throughput: {n_days / avg_time:.1f} days/second")

    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df = pd.DataFrame(
            {
                "run": range(1, n_runs + 1),
                "time_seconds": run_times,
            }
        )

        results_file = (
            output_dir / f"backtest_benchmark_{n_symbols}syms_{n_days}days.csv"
        )
        results_df.to_csv(results_file, index=False)
        logger.info("")
        logger.info(f"Results saved to: {results_file}")

    return {
        "n_symbols": n_symbols,
        "n_days": n_days,
        "include_costs": include_costs,
        "n_runs": n_runs,
        "avg_time": avg_time,
        "std_time": std_time,
        "min_time": min_time,
        "max_time": max_time,
        "run_times": run_times,
    }


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Performance benchmark for backtest engine optimizations (P3)"
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=10,
        help="Number of symbols in universe (default: 10)",
    )
    parser.add_argument(
        "--n-days", type=int, default=252, help="Number of trading days (default: 252)"
    )
    parser.add_argument(
        "--include-costs",
        action="store_true",
        help="Include transaction costs in benchmark",
    )
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Number of benchmark runs (default: 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for benchmark logs (default: output/perf_benchmarks)",
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = ROOT / "output" / "perf_benchmarks"

    # Run benchmark
    try:
        _ = run_benchmark(
            n_symbols=args.n_symbols,
            n_days=args.n_days,
            include_costs=args.include_costs,
            n_runs=args.n_runs,
            output_dir=args.output_dir,
        )
        logger.info("")
        logger.info("Benchmark completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
