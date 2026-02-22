#!/usr/bin/env python3
"""Run a minimal backtest using synthetic EOD data (no external data or new deps).

Creates a small Parquet under output/analysis_run/smoke, runs run_backtest_strategy.py
with --price-file, then optionally runs analyze_backtest_results.py. For CI and
local smoke runs when no real price data is available.

Usage:
    py -3 scripts/dev/smoke_backtest_local.py
    py -3 scripts/dev/smoke_backtest_local.py --no-analyze
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke backtest with synthetic data.")
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip running analyze_backtest_results.py after backtest",
    )
    args = parser.parse_args()

    out_base = ROOT / "output" / "analysis_run" / "smoke"
    out_base.mkdir(parents=True, exist_ok=True)
    data_dir = out_base / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    price_file = data_dir / "eod_smoke.parquet"

    # Minimal synthetic EOD: 2 symbols, ~252 days
    import pandas as pd

    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    rows = []
    for sym in symbols:
        for i, d in enumerate(dates):
            close = 100.0 + i * 0.05 + (i % 20) * 0.5
            rows.append({
                "timestamp": d,
                "symbol": sym,
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(price_file, index=False)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_backtest_strategy.py"),
        "--freq", "1d",
        "--price-file", str(price_file),
        "--strategy", "trend_baseline",
        "--start-capital", "10000",
        "--out", str(out_base),
        "--no-ledger",
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), timeout=120)
    if r.returncode != 0:
        print("Backtest failed", file=sys.stderr)
        return r.returncode

    if not args.no_analyze:
        cmd2 = [
            sys.executable,
            str(ROOT / "scripts" / "dev" / "analyze_backtest_results.py"),
            "--out", str(out_base),
            "--summary-dir", str(ROOT / "output" / "analysis_run"),
        ]
        r2 = subprocess.run(cmd2, cwd=str(ROOT), timeout=30)
        if r2.returncode != 0:
            print("Analyze failed", file=sys.stderr)
            return r2.returncode

    print("Smoke backtest and analysis OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
