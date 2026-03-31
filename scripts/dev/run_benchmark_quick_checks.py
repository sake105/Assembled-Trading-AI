#!/usr/bin/env python3
"""Run the 3 quick diagnostic checks for 0-trades benchmark runs.

Usage:
  py -3 scripts/dev/run_benchmark_quick_checks.py [--slice PATH] [--variants PATH]

Defaults: slice = output/system_run/benchmark/filter_sweep/price_slice_1y.parquet
          (or trend_baseline/1y/price_slice.parquet if filter_sweep missing)
          variants = scripts/dev/benchmark_variants.json

Check A: Slice has movement and multiple symbols (min != max, symbols >= 1).
Check B: EMA(20)/EMA(60) state changes in slice (state_changes = 0 => no crossover => 0 trades).
Check C: Variants do not require a universe that mismatches synthetic symbols (AAPL, MSFT, GOOGL).

If A and B pass but you still get 0 trades: (1) Check anomalies.json for data_qc_fail (e.g. zero_volume
= slice had no volume column; benchmark now ensures OHLCV on slices). (2) Ensure slice has OHLCV.
(3) Run without --quick (more history) or use real parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_a(slice_path: Path) -> bool:
    import pandas as pd

    if not slice_path.exists():
        print(f"Check A: SLICE NOT FOUND: {slice_path}")
        return False
    df = pd.read_parquet(slice_path)
    print("Check A: Slice movement / symbols")
    print(f"  cols={list(df.columns)}, rows={len(df)}")
    if "symbol" not in df.columns:
        print("  symbols= NO symbol col")
        return False
    n = df["symbol"].nunique()
    print(f"  symbols={n}")
    if set(["symbol", "close"]).issubset(df.columns):
        agg = df.groupby("symbol")["close"].agg(["count", "min", "max"])
        print(agg.head().to_string())
        if (agg["min"] == agg["max"]).all():
            print("  => min==max: price constant, EMA cannot cross => 0 trades")
            return False
    print("  => OK (movement present)" if n >= 1 else "  => no symbols")
    return True


def check_b(slice_path: Path, strict_min_periods: bool = False) -> bool:
    import pandas as pd

    if not slice_path.exists():
        print("Check B: SLICE NOT FOUND")
        return False
    df = pd.read_parquet(slice_path)
    tcol = next(
        (
            c
            for c in df.columns
            if c.lower() in ("date", "datetime", "timestamp", "ts")
            or "date" in c.lower()
        ),
        None,
    )
    if not tcol:
        print("Check B: No date-like column found")
        return False
    df = df.sort_values([tcol, "symbol"])
    sym = df["symbol"].unique()[0]
    s = df[df.symbol == sym].sort_values(tcol)["close"].astype(float)
    if strict_min_periods:
        fast = s.ewm(span=20, adjust=False, min_periods=20).mean()
        slow = s.ewm(span=60, adjust=False, min_periods=60).mean()
        m = (fast > slow).dropna().astype(int)
        n_valid = len(m)
        changes = int((m.diff() != 0).sum())
        pct = float(m.mean()) if len(m) else None
        print("Check B (strict min_periods=20/60): EMA state after warmup")
        print(
            f"  symbol={sym}, n={len(s)}, n_valid={n_valid}, pct_fast_gt_slow={pct}, state_changes={changes}"
        )
    else:
        fast = s.ewm(span=20, adjust=False).mean()
        slow = s.ewm(span=60, adjust=False).mean()
        state = (fast > slow).astype(int)
        changes = int((state.diff() != 0).sum())
        pct = float(state.mean())
        n_valid = len(state)
        print("Check B: EMA(20)/EMA(60) state in slice")
        print(f"  symbol={sym}, pct_fast_gt_slow={pct:.4f}, state_changes={changes}")
    if changes == 0:
        print(
            "  => state_changes=0: no flip in slice => 0 trades (if strategy trades on crossover)"
        )
        return False
    if strict_min_periods and n_valid < 25:
        print(
            f"  => n_valid={n_valid} very small: warmup may kill signals in short series"
        )
    print("  => OK (at least one crossover)")
    return True


def check_c(variants_path: Path) -> bool:
    if not variants_path.exists():
        print(f"Check C: VARIANTS NOT FOUND: {variants_path}")
        return False
    d = json.loads(variants_path.read_text(encoding="utf-8"))
    print("Check C: Variant universes (must match synthetic: AAPL, MSFT, GOOGL)")
    synthetic_symbols = {"AAPL", "MSFT", "GOOGL"}
    ok = True
    for v in d.get("variants", []):
        u = v.get("params") or {}
        u = u.get("universe")
        vid = v.get("id", "?")
        if u is not None:
            uset = set(u) if isinstance(u, list) else {u}
            if not uset.intersection(synthetic_symbols):
                print(f"  {vid} -> {u} (no overlap with synthetic)")
                ok = False
            else:
                print(f"  {vid} -> {u}")
        else:
            print(f"  {vid} -> None")
    if ok:
        print("  => OK (no universe or universe matches)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark 0-trades quick checks")
    ap.add_argument(
        "--slice", type=Path, default=None, help="Path to 1y price slice parquet"
    )
    ap.add_argument(
        "--variants",
        type=Path,
        default=ROOT / "scripts" / "dev" / "benchmark_variants.json",
    )
    ap.add_argument("--output-root", type=Path, default=ROOT / "output" / "system_run")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Use min_periods=20/60 in Check B (like strict backtest warmup)",
    )
    args = ap.parse_args()
    output_root = args.output_root.resolve()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    bench = output_root / "benchmark"
    if args.slice is not None:
        slice_path = Path(args.slice)
    else:
        slice_path = bench / "filter_sweep" / "price_slice_1y.parquet"
        if not slice_path.exists():
            slice_path = bench / "trend_baseline" / "1y" / "price_slice.parquet"
    variants_path = (
        args.variants.resolve()
        if args.variants
        else (ROOT / "scripts" / "dev" / "benchmark_variants.json")
    )
    print("Slice:", slice_path)
    print("Variants:", variants_path)
    print()
    check_a(slice_path)
    print()
    check_b(slice_path, strict_min_periods=args.strict)
    print()
    check_c(variants_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
