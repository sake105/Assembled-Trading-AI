# scripts/dev/quick_daily_backtest.py
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import pathlib as pl

ROOT = pl.Path(__file__).resolve().parents[2]
OUT  = ROOT / "output"

def main(start_capital: float, fast: int, slow: int):
    p = OUT / "aggregates" / "daily.parquet"
    df = pd.read_parquet(p)[["timestamp","symbol","close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

    def per_symbol(d: pd.DataFrame) -> pd.DataFrame:
        px = d["close"].astype(float)
        f = px.ewm(span=fast, adjust=False, min_periods=fast).mean()
        s = px.ewm(span=slow, adjust=False, min_periods=slow).mean()
        sig = (f > s).astype(int) - (f < s).astype(int)
        return pd.DataFrame({"timestamp": d["timestamp"], "symbol": d["symbol"], "sig": sig})

    sig = df.groupby("symbol", group_keys=False).apply(per_symbol).reset_index(drop=True)
    df = df.merge(sig, on=["timestamp","symbol"], how="left").fillna({"sig":0})
    ret = np.log(df["close"]).groupby(df["symbol"]).diff().fillna(0.0)
    df = df.assign(ret=ret)

    piv_sig = df.pivot(index="timestamp", columns="symbol", values="sig").fillna(0.0)
    piv_ret = df.pivot(index="timestamp", columns="symbol", values="ret").fillna(0.0)
    w = (piv_sig != 0).astype(float)
    w = w.div(w.sum(axis=1).replace(0,1), axis=0)
    port = (piv_sig * piv_ret * w).sum(axis=1)

    eq = start_capital * np.exp(port.cumsum())
    out = pd.DataFrame({"timestamp": eq.index, "equity": eq.values})
    out.to_csv(OUT / "equity_curve_1d_quick.csv", index=False)
    print(f"[OK] output/equity_curve_1d_quick.csv rows={len(out)}  first={out['timestamp'].iloc[0]} last={out['timestamp'].iloc[-1]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-capital", type=float, default=10_000.0)
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=60)
    a = ap.parse_args()
    main(a.start_capital, a.fast, a.slow)
