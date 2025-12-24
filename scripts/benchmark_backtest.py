"""Benchmark harness for backtest performance measurement.

This script runs BACKTEST_MEDIUM multiple times and logs median runtime.
Results are written to output/profiles/.../benchmark.json for tracking.

Usage:
    python scripts/benchmark_backtest.py [--runs N] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.assembled_core.config import OUTPUT_DIR
from scripts.profile_jobs import REFERENCE_JOBS, JOB_MAP, run_job_without_profiling


def run_benchmark_harness(
    job_name: str = "BACKTEST_MEDIUM",
    num_runs: int = 3,
    output_dir: Path | None = None,
) -> dict[str, float | int | str]:
    """Run benchmark harness: execute job multiple times and compute median runtime.

    Args:
        job_name: Name of reference job to benchmark (default: "BACKTEST_MEDIUM")
        num_runs: Number of runs to execute (default: 3, median of 3)
        output_dir: Output directory for benchmark results (default: OUTPUT_DIR/profiles/benchmark)

    Returns:
        Dictionary with benchmark results:
        - job_name: Name of benchmarked job
        - num_runs: Number of runs executed
        - runtimes: List of runtimes in seconds
        - median_runtime: Median runtime in seconds
        - mean_runtime: Mean runtime in seconds
        - min_runtime: Minimum runtime in seconds
        - max_runtime: Maximum runtime in seconds
        - timestamp: ISO timestamp of benchmark run
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR / "profiles" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BENCHMARK] Starting benchmark for {job_name} ({num_runs} runs)...")

    # Verify job exists
    if job_name not in JOB_MAP:
        print(f"[ERROR] Unknown job: {job_name}")
        print(f"[ERROR] Available jobs: {sorted(JOB_MAP.keys())}")
        sys.exit(1)

    # Set seed if this is a reference job (for determinism)
    for ref_job in REFERENCE_JOBS:
        if ref_job.name == job_name and ref_job.seed is not None:
            import numpy as np
            np.random.seed(ref_job.seed)
            break

    # Run benchmark
    runtimes = []
    for run_idx in range(num_runs):
        print(f"[BENCHMARK] Run {run_idx + 1}/{num_runs}...")
        start_time = time.time()

        try:
            # Run job (using run_job_without_profiling for clean execution)
            run_job_without_profiling(job_name, warm_cache=False, use_factor_store=False)
            elapsed = time.time() - start_time
            runtimes.append(elapsed)
            print(f"[BENCHMARK] Run {run_idx + 1} completed in {elapsed:.2f}s")
        except Exception as e:
            print(f"[ERROR] Run {run_idx + 1} failed: {e}")
            # Continue with remaining runs even if one fails
            continue

    if not runtimes:
        print("[ERROR] No successful runs completed")
        sys.exit(1)

    # Compute statistics
    median_runtime = statistics.median(runtimes)
    mean_runtime = statistics.mean(runtimes)
    min_runtime = min(runtimes)
    max_runtime = max(runtimes)

    # Prepare results
    results = {
        "job_name": job_name,
        "num_runs": num_runs,
        "runtimes": runtimes,
        "median_runtime": median_runtime,
        "mean_runtime": mean_runtime,
        "min_runtime": min_runtime,
        "max_runtime": max_runtime,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }

    # Write results to JSON
    benchmark_file = output_dir / "benchmark.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[BENCHMARK] Results written to {benchmark_file}")
    print(f"[BENCHMARK] Median runtime: {median_runtime:.2f}s")
    print(f"[BENCHMARK] Mean runtime: {mean_runtime:.2f}s")
    print(f"[BENCHMARK] Range: {min_runtime:.2f}s - {max_runtime:.2f}s")

    return results


def main() -> None:
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(description="Benchmark backtest performance")
    parser.add_argument(
        "--job",
        type=str,
        default="BACKTEST_MEDIUM",
        help="Reference job name to benchmark (default: BACKTEST_MEDIUM)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs to execute (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for benchmark results (default: output/profiles/benchmark)",
    )

    args = parser.parse_args()

    try:
        results = run_benchmark_harness(
            job_name=args.job,
            num_runs=args.runs,
            output_dir=args.output_dir,
        )
        print("\n[BENCHMARK] Benchmark completed successfully")
        print(f"[BENCHMARK] Median runtime: {results['median_runtime']:.2f}s")
    except KeyboardInterrupt:
        print("\n[BENCHMARK] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

