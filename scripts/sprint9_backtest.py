# -*- coding: utf-8 -*-
"""
Sprint 9 Backtest – robust & zukunftssicher
Erzeugt Equity-Curve und Performance-Report aus den simulierten Orders.
"""

from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "output")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _load_orders() -> pd.DataFrame:
    path = os.path.join(OUT_DIR, "orders.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    lower = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=lower)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["qty", "price", "commission", "fill_price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp"])
    return df.reset_index(drop=True)

def _simulate_equity(orders: pd.DataFrame, start_capital: float, freq: str) -> pd.DataFrame:
    """Erstellt synthetische Equity-Kurve aus Orders."""
    if orders.empty:
        ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=120, freq=freq)
        return pd.DataFrame({"timestamp": ts, "equity": float(start_capital)})

    orders = orders.sort_values("timestamp")
    timeline = pd.date_range(start=orders["timestamp"].min(), end=orders["timestamp"].max(), freq=freq)
    cash = np.full(len(timeline), start_capital, dtype=np.float64)
    equity = pd.Series(cash, index=timeline)

    for _, row in orders.iterrows():
        ts = row["timestamp"]
        if pd.isna(ts):
            continue
        px = float(row.get("price", 0.0))
        qty = float(row.get("qty", 0.0))
        comm = float(row.get("commission", 0.0))
        side = str(row.get("side", "BUY")).upper()
        sign = 1.0 if side == "BUY" else -1.0
        delta = sign * px * qty + comm
        idx = equity.index.searchsorted(ts)
        if idx < len(equity):
            equity.iloc[idx:] = equity.iloc[idx:] - delta

    # Stabilisierung & Cleaning
    s = pd.Series(equity, index=timeline)
    s = s.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_capital)
    s = s.clip(0.0, 1e12)
    return pd.DataFrame({"timestamp": s.index, "equity": s.values})

def _performance_report(eq: pd.DataFrame) -> dict:
    if eq.empty:
        return {"PF": 1.0, "Sharpe": 0.0, "Trades": 0}
    ret = eq["equity"].pct_change().fillna(0.0)
    pf = eq["equity"].iloc[-1] / eq["equity"].iloc[0]
    sharpe = np.sqrt(252) * ret.mean() / (ret.std() + 1e-9)
    trades = len(ret[ret != 0])
    return {"PF": pf, "Sharpe": sharpe, "Trades": trades}

def _write_report(eq: pd.DataFrame, rep: dict, freq: str) -> None:
    _ensure_dir(OUT_DIR)
    curve_path = os.path.join(OUT_DIR, f"equity_curve_{freq}.csv")
    rep_path = os.path.join(OUT_DIR, "performance_report.md")
    eq.to_csv(curve_path, index=False)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"# Performance Report ({freq})\n\n")
        f.write(f"- Final PF: {rep['PF']:.2f}\n")
        f.write(f"- Sharpe: {rep['Sharpe']:.2f}\n")
        f.write(f"- Trades: {rep['Trades']}\n")
    print(f"[BT9] [OK] written: {curve_path}")
    print(f"[BT9] [OK] written: {rep_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min")
    args = p.parse_args()

    print(f"[BT9] START Backtest | freq={args.freq}")
    orders = _load_orders()
    eq = _simulate_equity(orders, start_capital=10000.0, freq=args.freq)
    rep = _performance_report(eq)
    _write_report(eq, rep, args.freq)
    print("[BT9] DONE Backtest")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"[BT9] ERROR {e}", file=sys.stderr)
        sys.exit(1)
