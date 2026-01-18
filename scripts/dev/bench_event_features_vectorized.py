"""Benchmark script for vectorized vs legacy event features (Sprint 11.E1).

This script generates deterministic synthetic data and benchmarks both
implementations without flaky timing assertions.

Usage:
    python scripts/dev/bench_event_features_vectorized.py

Output:
    output/event_study_bench_<timestamp>.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.event_features import (
    add_disclosure_count_feature,
    build_event_feature_panel,
)


def generate_synthetic_data(
    n_symbols: int = 1000,
    n_days: int = 5 * 252,  # 5 years daily
    n_events_per_symbol: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic synthetic prices and events for benchmarking.

    Args:
        n_symbols: Number of symbols
        n_days: Number of trading days
        n_events_per_symbol: Number of events per symbol
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prices_df, events_df)
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Generate daily timestamps (5 years)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D", tz="UTC")

    # Generate prices panel
    symbols_list = [f"SYMBOL_{i:04d}" for i in range(n_symbols)]
    prices_list = []

    for symbol in symbols_list:
        for date in dates:
            prices_list.append({
                "timestamp": date,
                "symbol": symbol,
                "close": 100.0 + rng.uniform(-10, 10),
            })

    prices_df = pd.DataFrame(prices_list)

    # Generate events panel
    events_list = []

    for symbol in symbols_list:
        # Random event dates within the date range
        event_dates = rng.choice(dates, size=n_events_per_symbol, replace=True)
        # Disclosure dates are T+1 to T+5 after event dates
        disclosure_dates = [
            event_date + pd.Timedelta(days=int(1 + rng.integers(0, 5))) for event_date in event_dates
        ]

        for event_date, disclosure_date in zip(event_dates, disclosure_dates):
            # Ensure disclosure_date is within valid range
            if disclosure_date > dates[-1]:
                disclosure_date = dates[-1]

            events_list.append({
                "symbol": symbol,
                "event_date": event_date,
                "disclosure_date": disclosure_date,
                "effective_date": disclosure_date,
                "value": rng.uniform(100, 1000),
            })

    events_df = pd.DataFrame(events_list)

    return prices_df, events_df


def benchmark_function(func, *args, **kwargs) -> dict[str, float | int]:
    """Benchmark a function execution.

    Args:
        func: Function to benchmark
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Dictionary with timing and result info
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()

    elapsed_seconds = end_time - start_time

    return {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_ms": elapsed_seconds * 1000,
        "result_rows": len(result) if hasattr(result, "__len__") else None,
        "result_columns": list(result.columns) if hasattr(result, "columns") else None,
    }


def main() -> None:
    """Run benchmark and write results to JSON file."""
    print("Generating synthetic data...")
    prices_df, events_df = generate_synthetic_data(
        n_symbols=1000, n_days=5 * 252, n_events_per_symbol=200, seed=42
    )

    print(f"Generated {len(prices_df)} price rows, {len(events_df)} event rows")
    print(f"Symbols: {prices_df['symbol'].nunique()}, Date range: {prices_df['timestamp'].min()} to {prices_df['timestamp'].max()}")

    as_of = pd.Timestamp("2024-12-31", tz="UTC")
    lookback_days = 30

    benchmark_results = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "data_stats": {
            "n_prices": len(prices_df),
            "n_events": len(events_df),
            "n_symbols": prices_df["symbol"].nunique(),
            "date_range": {
                "start": str(prices_df["timestamp"].min()),
                "end": str(prices_df["timestamp"].max()),
            },
        },
        "parameters": {
            "as_of": str(as_of),
            "lookback_days": lookback_days,
        },
    }

    # Benchmark build_event_feature_panel
    print("\nBenchmarking build_event_feature_panel (legacy)...")
    legacy_panel_results = benchmark_function(
        build_event_feature_panel,
        events_df,
        prices_df,
        as_of=as_of,
        lookback_days=lookback_days,
        method="legacy",
    )

    print(f"  Elapsed: {legacy_panel_results['elapsed_ms']:.2f} ms")
    print(f"  Result rows: {legacy_panel_results['result_rows']}")

    print("\nBenchmarking build_event_feature_panel (vectorized)...")
    vectorized_panel_results = benchmark_function(
        build_event_feature_panel,
        events_df,
        prices_df,
        as_of=as_of,
        lookback_days=lookback_days,
        method="vectorized",
    )

    print(f"  Elapsed: {vectorized_panel_results['elapsed_ms']:.2f} ms")
    print(f"  Result rows: {vectorized_panel_results['result_rows']}")

    # Sanity checks
    assert legacy_panel_results["result_rows"] > 0, "Legacy result is empty"
    assert vectorized_panel_results["result_rows"] > 0, "Vectorized result is empty"
    assert legacy_panel_results["result_rows"] == vectorized_panel_results["result_rows"], (
        f"Row count mismatch: legacy={legacy_panel_results['result_rows']}, "
        f"vectorized={vectorized_panel_results['result_rows']}"
    )

    benchmark_results["build_event_feature_panel"] = {
        "legacy": legacy_panel_results,
        "vectorized": vectorized_panel_results,
        "speedup": legacy_panel_results["elapsed_seconds"] / vectorized_panel_results["elapsed_seconds"],
    }

    # Benchmark add_disclosure_count_feature
    print("\nBenchmarking add_disclosure_count_feature (legacy)...")
    legacy_count_results = benchmark_function(
        add_disclosure_count_feature,
        prices_df,
        events_df,
        window_days=lookback_days,
        as_of=as_of,
        method="legacy",
    )

    print(f"  Elapsed: {legacy_count_results['elapsed_ms']:.2f} ms")
    print(f"  Result rows: {legacy_count_results['result_rows']}")

    print("\nBenchmarking add_disclosure_count_feature (vectorized)...")
    vectorized_count_results = benchmark_function(
        add_disclosure_count_feature,
        prices_df,
        events_df,
        window_days=lookback_days,
        as_of=as_of,
        method="vectorized",
    )

    print(f"  Elapsed: {vectorized_count_results['elapsed_ms']:.2f} ms")
    print(f"  Result rows: {vectorized_count_results['result_rows']}")

    # Sanity checks
    assert legacy_count_results["result_rows"] > 0, "Legacy result is empty"
    assert vectorized_count_results["result_rows"] > 0, "Vectorized result is empty"
    assert legacy_count_results["result_rows"] == vectorized_count_results["result_rows"], (
        f"Row count mismatch: legacy={legacy_count_results['result_rows']}, "
        f"vectorized={vectorized_count_results['result_rows']}"
    )

    benchmark_results["add_disclosure_count_feature"] = {
        "legacy": legacy_count_results,
        "vectorized": vectorized_count_results,
        "speedup": legacy_count_results["elapsed_seconds"] / vectorized_count_results["elapsed_seconds"],
    }

    # Write results to JSON file
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    timestamp_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"event_study_bench_{timestamp_str}.json"

    with open(output_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nBenchmark results written to: {output_file}")
    print("\nSpeedup summary:")
    print(f"  build_event_feature_panel: {benchmark_results['build_event_feature_panel']['speedup']:.2f}x")
    print(f"  add_disclosure_count_feature: {benchmark_results['add_disclosure_count_feature']['speedup']:.2f}x")


if __name__ == "__main__":
    main()
