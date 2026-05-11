#!/usr/bin/env python
"""LiveDecisionEngine Benchmark — Bootstrap + 252 Daily-Updates.

Misst:
1. Bootstrap (one-time) Latency
2. update_with_new_day() (per-bar) Latency-Distribution
3. decide_next() (per-bar) Latency-Distribution
4. Vergleich vs full-rebuild Master-Pipeline
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.live.live_decision_engine import (  # noqa: E402
    LiveDecisionEngine,
    LiveEngineConfig,
)


def _load_data():
    """Lade Equity (22 mega-caps, 19y) + Cross-Asset (11 ETFs, 19y)."""
    # Equity
    eq_panel = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    if "timestamp" in eq_panel.columns:
        eq_panel = eq_panel.rename(columns={"timestamp": "date"})
    eq_panel["date"] = pd.to_datetime(eq_panel["date"], utc=True)
    eq_panel = eq_panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    eq_panel["return"] = eq_panel.groupby("symbol")["close"].pct_change()
    eq_wide = eq_panel.pivot_table(
        index="date", columns="symbol", values="return"
    ).sort_index()

    # Cross-Asset
    xa_frames = []
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
        xa_frames.append(df)
    xa_panel = pd.concat(xa_frames, ignore_index=True)
    xa_panel["date"] = pd.to_datetime(xa_panel["date"], utc=True)
    xa_close_wide = xa_panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    xa_wide = xa_close_wide.pct_change()

    return eq_wide.fillna(0), xa_wide.fillna(0)


def main():
    print("=" * 100)
    print("LIVE-DECISION-ENGINE BENCHMARK")
    print("=" * 100)

    eq_wide, xa_wide = _load_data()
    print(f"Loaded eq_wide: {eq_wide.shape}, xa_wide: {xa_wide.shape}")
    print(f"Date range: {eq_wide.index.min().date()} -> {eq_wide.index.max().date()}")

    # Use first 4500 days for bootstrap, last 252 for live-test (1y of daily updates)
    split = -252
    eq_train, eq_test = eq_wide.iloc[:split], eq_wide.iloc[split:]
    xa_train, xa_test = xa_wide.iloc[:split], xa_wide.iloc[split:]
    print(f"\nBootstrap window: {len(eq_train)} days")
    print(f"Live-test window: {len(eq_test)} days")

    # === Bootstrap ===
    engine = LiveDecisionEngine(LiveEngineConfig(sa_weight=0.70))
    t0 = time.perf_counter()
    engine.bootstrap_from_history(eq_train, xa_train)
    bootstrap_ms = (time.perf_counter() - t0) * 1000
    print(f"\nBootstrap latency: {bootstrap_ms:.2f} ms")
    print(f"State summary: {engine.state_summary()}")

    # === Live-Loop: 252 daily updates + decisions ===
    print("\n" + "=" * 100)
    print("LIVE LOOP: 252 trading days, daily update + decide")
    print("=" * 100)

    update_latencies = []
    decide_latencies = []
    for i in range(len(eq_test)):
        date = eq_test.index[i]
        eq_row = eq_test.iloc[i]
        xa_row = xa_test.iloc[i]

        t0 = time.perf_counter()
        engine.update_with_new_day(date, eq_row, xa_row)
        update_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        decision = engine.decide_next()
        decide_latencies.append((time.perf_counter() - t0) * 1000)

    print(f"\nUpdate latency (n={len(update_latencies)}):")
    print(f"  median: {np.median(update_latencies):.3f} ms")
    print(f"  mean:   {np.mean(update_latencies):.3f} ms")
    print(f"  p95:    {np.percentile(update_latencies, 95):.3f} ms")
    print(f"  p99:    {np.percentile(update_latencies, 99):.3f} ms")
    print(f"  max:    {np.max(update_latencies):.3f} ms")

    print(f"\nDecide latency (n={len(decide_latencies)}):")
    print(f"  median: {np.median(decide_latencies):.3f} ms")
    print(f"  mean:   {np.mean(decide_latencies):.3f} ms")
    print(f"  p95:    {np.percentile(decide_latencies, 95):.3f} ms")
    print(f"  p99:    {np.percentile(decide_latencies, 99):.3f} ms")
    print(f"  max:    {np.max(decide_latencies):.3f} ms")

    total_per_bar = np.median(update_latencies) + np.median(decide_latencies)
    print(f"\nTotal per-bar median latency: {total_per_bar:.3f} ms")
    print(f"-> Theoretical max bars/second: {1000/total_per_bar:.0f}")

    # SLA-Check
    print("\nSLA Check:")
    sla = 10.0
    update_p99 = np.percentile(update_latencies, 99)
    decide_p99 = np.percentile(decide_latencies, 99)
    print(
        f"  update_with_new_day p99 = {update_p99:.2f} ms  "
        f"[{'OK' if update_p99 < sla else 'FAIL'} <{sla}ms]"
    )
    print(
        f"  decide_next p99         = {decide_p99:.2f} ms  "
        f"[{'OK' if decide_p99 < sla else 'FAIL'} <{sla}ms]"
    )

    # Final decision
    print("\n" + "=" * 100)
    print("FINAL DECISION (last bar)")
    print("=" * 100)
    final = engine.decide_next()
    print(f"  timestamp: {final['timestamp']}")
    print(f"  sa_leverage: {final['sa_leverage']:.3f}")
    print(f"  xa_ew_leverage: {final['xa_ew_leverage']:.3f}")
    print(
        f"  XA top picks (mom): "
        f"{final['xa_top_weights'][final['xa_top_weights'] > 0].index.tolist()}"
    )
    print(
        f"  EQ top picks (mom): "
        f"{final['eq_top_weights'][final['eq_top_weights'] > 0].index.tolist()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
