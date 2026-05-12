"""Polars vs pandas microbenchmark for ta_features (audit B-001).

Run::

    python scripts/perf/bench_polars_vs_pandas.py

Generates a synthetic multi-symbol panel and times the pandas vs Polars
implementations of the four ported feature functions. Pure benchmark
— no data leaves the script.

Expected (per the audit B-001 acceptance criterion): 5y × 500 symbols
in < 10s, vs ~45s for the pandas path. Real numbers depend on host CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from anywhere via "python scripts/perf/bench_polars_vs_pandas.py".
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _synthetic_panel(n_symbols: int, n_days: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2020-01-01", periods=n_days, freq="D")
    frames = []
    for i in range(n_symbols):
        base = np.cumprod(1.0 + rng.normal(0.0005, 0.015, n_days))
        close = 100.0 * base
        high = close * (1.0 + rng.uniform(0.0, 0.012, n_days))
        low = close * (1.0 - rng.uniform(0.0, 0.012, n_days))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "symbol": f"SYM{i:03d}",
                    "close": close,
                    "high": high,
                    "low": low,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _time(name: str, fn, *args, **kwargs) -> tuple[str, float, pd.DataFrame]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {name:<28} {elapsed:8.3f}s  rows={len(out):>9,}")
    return name, elapsed, out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-symbols", type=int, default=100)
    p.add_argument("--n-days", type=int, default=1260)  # ~5y
    args = p.parse_args()

    print(f"[bench] panel: {args.n_symbols} symbols × {args.n_days} days")
    df = _synthetic_panel(args.n_symbols, args.n_days)
    print(f"[bench] total rows: {len(df):,}")

    from src.assembled_core.features.ta_features import (
        add_atr,
        add_log_returns,
        add_moving_averages,
        add_rsi,
    )
    from src.assembled_core.features.ta_features_polars import (
        add_atr_polars,
        add_log_returns_polars,
        add_moving_averages_polars,
        add_rsi_polars,
    )

    print()
    print("[bench] pandas path:")
    _, t_pd_lr, _ = _time("add_log_returns", add_log_returns, df)
    _, t_pd_ma, _ = _time(
        "add_moving_averages", add_moving_averages, df, windows=(20, 50, 200)
    )
    _, t_pd_atr, _ = _time("add_atr", add_atr, df, window=14)
    _, t_pd_rsi, _ = _time("add_rsi", add_rsi, df, window=14)
    pd_total = t_pd_lr + t_pd_ma + t_pd_atr + t_pd_rsi

    print()
    print("[bench] polars path:")
    _, t_pl_lr, _ = _time("add_log_returns_polars", add_log_returns_polars, df)
    _, t_pl_ma, _ = _time(
        "add_moving_averages_polars",
        add_moving_averages_polars,
        df,
        windows=(20, 50, 200),
    )
    _, t_pl_atr, _ = _time("add_atr_polars", add_atr_polars, df, window=14)
    _, t_pl_rsi, _ = _time("add_rsi_polars", add_rsi_polars, df, window=14)
    pl_total = t_pl_lr + t_pl_ma + t_pl_atr + t_pl_rsi

    speedup = pd_total / max(pl_total, 1e-9)
    print()
    print(f"[bench] pandas total: {pd_total:.3f}s")
    print(f"[bench] polars total: {pl_total:.3f}s")
    print(f"[bench] speedup: {speedup:.2f}x")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
