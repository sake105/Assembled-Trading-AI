#!/usr/bin/env python
"""Profile der Master-Pipeline — finde Bottlenecks.

Misst pro Pipeline-Schritt:
- Wall-Clock-Zeit
- cProfile-Top-Functions
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.strategies.cross_section_helpers import (  # noqa: E402
    cs_long_only_wide,
    long_format_to_wide,
)
from erweiterung.strategies.master_allocator import (  # noqa: E402
    MasterAllocator,
    MasterAllocatorConfig,
)


def time_block(name, fn):
    t0 = time.perf_counter()
    out = fn()
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {name:<40} {elapsed:>10.2f} ms")
    return out, elapsed


def _cs_long_only(panel, signal_col, quantile=0.3):
    out = panel.copy().sort_values(["symbol", "date"])
    out["sig_lag"] = out.groupby("symbol", group_keys=False)[signal_col].shift(1)
    by_d = out.groupby("date")["sig_lag"]
    out["sig_pct"] = by_d.rank(pct=True)
    out["position"] = 0.0
    out.loc[out["sig_pct"] >= 1 - quantile, "position"] = 1.0
    n_long = out.groupby("date")["position"].transform(lambda s: (s > 0).sum())
    long_mask = out["position"] > 0
    out.loc[long_mask, "position"] = 1.0 / n_long[long_mask]
    out["pnl"] = out["position"] * out["return"]
    return out


def run_pipeline():
    """End-to-End Master-Pipeline-Run (gleich wie run_master_long_history.py)."""
    stats = {}

    eq_panel, t = time_block(
        "Step 1: Load equity panel (parquet)",
        lambda: pd.read_parquet("data/sample/watchlist_2007_2026.parquet"),
    )
    stats["load_equity"] = t

    if "timestamp" in eq_panel.columns:
        eq_panel = eq_panel.rename(columns={"timestamp": "date"})

    def prepare():
        df = eq_panel.copy()
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["return"] = df.groupby("symbol")["close"].pct_change()
        return df

    eq_panel, t = time_block(
        "Step 2: Prepare equity panel (dt+sort+pct_change)", prepare
    )
    stats["prepare_equity"] = t

    def compute_mom():
        mom = momentum_12_1(eq_panel[["date", "symbol", "close"]])
        df = eq_panel.set_index(["date", "symbol"])
        df["mom_12_1"] = mom.reindex(df.index)
        return df.reset_index()

    eq_panel, t = time_block("Step 3: Compute Mom-12/1", compute_mom)
    stats["compute_mom"] = t

    # NEW: vectorized version
    def compute_factor_vec():
        sig_wide = long_format_to_wide(
            eq_panel[["date", "symbol", "mom_12_1"]], "mom_12_1"
        )
        eq_panel["return_clean"] = eq_panel["return"].fillna(0)
        ret_wide = long_format_to_wide(
            eq_panel[["date", "symbol", "return_clean"]], "return_clean"
        )
        pnl, _ = cs_long_only_wide(sig_wide, ret_wide, quantile=0.3, lag_days=1)
        return pnl

    eq_factor_ret, t = time_block(
        "Step 4+5: CS long-only + aggregate (vectorized)", compute_factor_vec
    )
    stats["cs_vectorized"] = t

    def load_xa():
        frames = []
        for sym in [
            "SPY",
            "QQQ",
            "IWM",
            "EFA",
            "EEM",
            "AGG",
            "TLT",
            "HYG",
            "GLD",
            "SLV",
            "DBC",
        ]:
            p = Path("data/cache/yfinance_long") / f"{sym}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p).reset_index()
            df["symbol"] = sym
            frames.append(df)
        panel = pd.concat(frames, ignore_index=True)
        panel["date"] = pd.to_datetime(panel["date"], utc=True)
        wide = panel.pivot_table(
            index="date", columns="symbol", values="close"
        ).sort_index()
        return wide.pct_change().dropna()

    xa_rets, t = time_block("Step 6: Load 11 ETFs (long history)", load_xa)
    stats["load_xa"] = t

    alloc = MasterAllocator(MasterAllocatorConfig(sa_weight=0.70))
    out, t = time_block(
        "Step 7: MasterAllocator.allocate()",
        lambda: alloc.allocate(eq_factor_ret, xa_rets),
    )
    stats["allocate"] = t

    total = sum(stats.values())
    print(f"\n{'TOTAL':<42} {total:>10.2f} ms ({total/1000:.2f} s)")

    # Pro-Step-Prozent
    print("\nTime distribution:")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>8.1f} ms  ({v/total:.1%})")

    return out, stats


def cprofile_run():
    """cProfile + top-20 functions."""
    print("\n" + "=" * 100)
    print("cProfile TOP-20 BY CUMULATIVE TIME")
    print("=" * 100)
    profiler = cProfile.Profile()
    profiler.enable()
    run_pipeline()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)


def main():
    print("=" * 100)
    print("MASTER PIPELINE PROFILING")
    print("=" * 100)
    print("\nWall-clock per step:")
    out, stats = run_pipeline()

    # Optional: cProfile run (heavier)
    if "--cprofile" in sys.argv:
        cprofile_run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
