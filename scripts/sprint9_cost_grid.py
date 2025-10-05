#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grid over cost parameters using Sprint 8 execution & orders generation.
Reads: output/features/regime_<freq>.parquet
Writes: output/cost_grid_report.md
"""
from __future__ import annotations
import argparse
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
FEAT = OUT / "features"
LOGS = ROOT / "logs"
OUT.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] [GRID] {msg}"
    print(line)
    with open(LOGS / "sprint9_cost_grid.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def load_features(freq: str) -> pd.DataFrame:
    path = FEAT / f"regime_{freq}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"features not found: {path}")
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None)
    return df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

def simple_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grp = df.groupby("symbol")
    reg = grp["trend_regime"].apply(lambda s: s.fillna(0).astype(int))
    sw = reg.groupby(level=0).diff().fillna(0)
    sig = np.sign(reg) * (sw != 0)
    df["signal"] = sig.values.astype(int)
    df["open_next"] = df.groupby("symbol")[ "open" if "open" in df.columns else "close" ].shift(-1)
    trades = df[(df["signal"] != 0) & df["open_next"].notna()].copy()
    trades["side"] = trades["signal"].map({-1:"SELL",0:"FLAT",1:"BUY"})
    return trades

def generate_orders(trades: pd.DataFrame, notional: float=10_000.0) -> pd.DataFrame:
    t = trades.copy()
    t["qty"] = (notional / t["open_next"]).round(0)
    t["px_ref"] = t["open_next"]
    return t[["timestamp","symbol","side","qty","px_ref","spread_proxy","vol_regime","liq_regime"]]

def simulate(trades: pd.DataFrame, commission_bps: float, spread_w: float, impact_w: float) -> pd.DataFrame:
    t = trades.copy()
    half_spread = 0.5 * spread_w * t["spread_proxy"].fillna(0.0005)
    impact = (0.0002 * impact_w * t["vol_regime"].fillna(0)) + (0.0003 * impact_w * (1 - t["liq_regime"].fillna(0)))
    slip = half_spread + impact
    signed_slip = np.where(t["side"]=="BUY", +slip, -slip)
    t["px_exec"] = t["px_ref"] * (1.0 + signed_slip)
    t["qty_filled"] = (t["qty"] * np.where(t["liq_regime"]==1, 1.0, 0.7)).round(0)
    t["commission"] = commission_bps * 1e-4 * t["px_exec"] * t["qty_filled"]
    t["pnl"] = np.where(t["side"]=="BUY",
                        -t["px_exec"]*t["qty_filled"],
                        +t["px_exec"]*t["qty_filled"]) - t["commission"]
    return t

def pf_from_orders(orders: pd.DataFrame) -> float:
    pnl = orders["pnl"].sum()
    pos = orders["pnl"][orders["pnl"]>0].sum()
    neg = -orders["pnl"][orders["pnl"]<0].sum()
    return float(pos/neg) if neg>0 else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", default="5min", choices=["1min","5min","15min"])
    ap.add_argument("--notional", type=float, default=10_000.0)
    ap.add_argument("--commission-bps", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    ap.add_argument("--spread-w", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    ap.add_argument("--impact-w", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    args = ap.parse_args()

    log(f"START Grid | freq={args.freq}")
    df = load_features(args.freq)
    trades = simple_signal(df)
    orders_base = generate_orders(trades, args.notional)

    rows = []
    for cb, sw, iw in itertools.product(args.commission_bps, args.spread_w, args.impact_w):
        sim = simulate(orders_base, commission_bps=cb, spread_w=sw, impact_w=iw)
        pf = pf_from_orders(sim)
        rows.append({"commission_bps": cb, "spread_w": sw, "impact_w": iw,
                     "pf": pf, "trades": int(len(sim))})
    res = pd.DataFrame(rows).sort_values(["commission_bps","spread_w","impact_w"]).reset_index(drop=True)
    (OUT / "cost_grid_report.md").write_text("# Cost Grid Report\n\n" + res.to_markdown(index=False), encoding="utf-8")
    log("[OK] written: output/cost_grid_report.md")
    log("DONE Grid")

if __name__ == "__main__":
    main()


