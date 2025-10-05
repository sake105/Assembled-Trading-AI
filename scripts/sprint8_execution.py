#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sprint 8 – Execution & Costs v2
- Reads engineered features (5min preferred)
- Generates example orders from simple signal (trend_regime)
- Applies cost model: spread + slippage (vol/liq aware) + commissions
- Simulates partial fills based on liquidity proxy
- Writes: orders.csv, fill_sim.md, cost_sensitivity.md

This is a pragmatic starter you can adapt to your broker/venue specifics.
Dependencies: pandas, numpy, pyarrow
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "features"
OUT  = ROOT / "output"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [EXEC] {msg}"
    print(line)
    with open(LOGS / "sprint8_execution.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_features(freq: str) -> pd.DataFrame:
    path = FEAT / f"regime_{freq}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    df = pd.read_parquet(path)
    # basic sanity
    need = {"timestamp","symbol","close","trend_regime","vol_regime","liq_regime","spread_proxy"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in features: {sorted(miss)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df


def simple_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Example signal using trend_regime transitions.
    +1 when trend_regime switches to 1, -1 when switches to -1, 0 otherwise.
    """
    df = df.sort_values(["symbol","timestamp"]).copy()
    grp = df.groupby("symbol")
    reg = grp["trend_regime"].apply(lambda s: s.fillna(0).astype(int))
    sw  = reg.groupby(level=0).diff().fillna(0)
    sig = np.sign(reg) * (sw != 0)
    df["signal"] = sig.values.astype(int)
    return df


def order_generator(df: pd.DataFrame, notional_per_trade: float = 10_000.0) -> pd.DataFrame:
    """Create orders from signals with simple sizing based on notional.
    Assumes we trade at the next bar's open.
    """
    df = df.copy()
    # next bar open as entry
    df["open_next"] = df.groupby("symbol")["open" if "open" in df.columns else "close"].shift(-1)
    # only take non-zero signals where we have a next bar
    trades = df[(df["signal"] != 0) & df["open_next"].notna()].copy()
    trades["side"] = trades["signal"].map({-1:"SELL", 0:"FLAT", 1:"BUY"})
    trades["qty"] = (notional_per_trade / trades["open_next"]).round(0)
    trades["px_ref"] = trades["open_next"]
    return trades[["timestamp","symbol","side","qty","px_ref","spread_proxy","vol_regime","liq_regime"]]


def cost_model(trades: pd.DataFrame, commission_bps: float = 0.5) -> pd.DataFrame:
    """Venue-agnostic cost model.
    execution_price = px_ref * (1 + slip), where
    slip = half_spread + impact
    half_spread ≈ 0.5 * spread_proxy
    impact ≈ alpha1*vol_regime + alpha2*(1-liq_regime) (scaled small)
    """
    t = trades.copy()
    half_spread = 0.5 * t["spread_proxy"].fillna(0.0005)
    impact = (0.0002 * t["vol_regime"].fillna(0)) + (0.0003 * (1 - t["liq_regime"].fillna(0)))
    slip = half_spread + impact
    # direction-aware
    signed_slip = np.where(t["side"] == "BUY", +slip, -slip)
    t["px_exec"] = t["px_ref"] * (1.0 + signed_slip)
    # commissions
    t["commission"] = commission_bps * 1e-4 * t["px_exec"] * t["qty"]
    # total cost in bps relative to ref
    t["cost_bps"] = (t["px_exec"] - t["px_ref"]) / t["px_ref"] * 1e4 + commission_bps
    return t


def partial_fill_sim(trades: pd.DataFrame) -> pd.DataFrame:
    """Toy partial fill model using liquidity regime.
    fill_ratio = 1.0 if liq_regime==1 else 0.7 (low liquidity → partial).
    """
    t = trades.copy()
    ratio = np.where(t["liq_regime"] == 1, 1.0, 0.7)
    t["qty_filled"] = (t["qty"] * ratio).round(0)
    t["qty_unfilled"] = t["qty"] - t["qty_filled"]
    return t


def write_reports(execdf: pd.DataFrame, freq: str) -> None:
    orders_path = OUT / "orders.csv"
    execdf.to_csv(orders_path, index=False)

    # simple sensitivity in markdown
    sens = execdf.groupby("side")["cost_bps"].describe().round(2)
    lines = ["# Cost Sensitivity (bps)", "", sens.to_markdown()]
    (OUT / "cost_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")

    fill_stats = execdf[["qty","qty_filled","qty_unfilled"]].sum()
    lines2 = ["# Partial Fill Summary", "", fill_stats.to_frame("sum").to_markdown()]
    (OUT / "fill_sim.md").write_text("\n".join(lines2), encoding="utf-8")

    log(f"[OK] written: {orders_path}")
    log(f"[OK] written: {OUT / 'cost_sensitivity.md'}")
    log(f"[OK] written: {OUT / 'fill_sim.md'}")


def main():
    ap = argparse.ArgumentParser(description="Sprint8 Execution & Costs v2")
    ap.add_argument("--freq", default="5min", choices=["1min","5min","15min"])
    ap.add_argument("--notional", type=float, default=10_000.0)
    ap.add_argument("--commission-bps", type=float, default=0.5)
    args = ap.parse_args()

    log(f"START Execution | freq={args.freq}")
    df = load_features(args.freq)
    df = simple_signal(df)
    trades = order_generator(df, notional_per_trade=args.notional)
    execdf = cost_model(trades, commission_bps=args.commission_bps)
    execdf = partial_fill_sim(execdf)
    write_reports(execdf, args.freq)
    log("DONE Execution")

if __name__ == "__main__":
    main()


