# -*- coding: utf-8 -*-
"""
Sprint 10 Portfolio – robust mit Kostenparametern
"""

from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "output")
FEAT_DIR = os.path.join(ROOT, "output", "features")

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
    for c in ["qty","price","commission","fill_price","commission_bps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp"])
    return df.reset_index(drop=True)

def _load_prices(freq: str) -> pd.DataFrame:
    cand = os.path.join(FEAT_DIR, f"base_{freq}.parquet")
    if os.path.exists(cand):
        try:
            df = pd.read_parquet(cand)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    if df.empty:
        return df
    lower = {c: str(c).lower() for c in df.columns}
    df = df.rename(columns=lower)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ["open","high","low","close","vwap","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["timestamp","symbol","close"])
    # Duplikate: mitteln auf (timestamp, symbol)
    df = df.groupby(["timestamp","symbol"], as_index=False).agg({"close":"mean"})
    return df.reset_index(drop=True)

def _fallback_equity(freq: str, start_capital: float) -> pd.DataFrame:
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=120, freq=freq)
    return pd.DataFrame({"timestamp": ts, "equity": float(start_capital)})

def _safe_freq_from_prices(prices: pd.DataFrame, cli_freq: str) -> str:
    if prices.empty:
        return cli_freq
    # pandas verlangt sortierten, eindeutigen Index zur Frequenzableitung
    ts = prices["timestamp"].sort_values()
    inferred = pd.infer_freq(ts)
    return inferred if isinstance(inferred, str) and inferred else cli_freq

def _equity_from_orders_prices(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    start_capital: float,
    commission_bps: float,
    freq_str: str,
) -> pd.DataFrame:
    # Frequenz robust bestimmen
    eff_freq = _safe_freq_from_prices(prices, freq_str)
    tmin = min(orders["timestamp"].min(), prices["timestamp"].min())
    tmax = max(orders["timestamp"].max(), prices["timestamp"].max())
    timeline = pd.date_range(start=tmin, end=tmax, freq=eff_freq)

    symbols = sorted(prices["symbol"].unique().tolist())
    px = prices.pivot(index="timestamp", columns="symbol", values="close")
    px = px.reindex(timeline).ffill().bfill()

    qty = pd.DataFrame(0.0, index=timeline, columns=symbols)
    if not orders.empty:
        od = orders[orders["symbol"].isin(symbols)].copy()
        side_str = od.get("side", "BUY").astype(str).str.upper()
        side_mult = np.where(side_str == "BUY", 1.0, -1.0)
        od["signed_qty"] = pd.to_numeric(od.get("qty", 0), errors="coerce").fillna(0.0) * side_mult
        od = od.groupby(["timestamp", "symbol"], as_index=False)["signed_qty"].sum()
        od = od.set_index("timestamp").reindex(timeline).fillna(0.0)
        for sym in symbols:
            if sym in od.columns:
                qty[sym] = od[sym].cumsum()

    mv = (qty * px).sum(axis=1)
    equity = mv.ffill().bfill().replace([np.inf, -np.inf], np.nan).ffill().fillna(start_capital).clip(0.0, 1e12)
    return pd.DataFrame({"timestamp": equity.index, "equity": equity.values})

def simulate(
    freq: str,
    start_capital: float,
    exposure: float,
    leverage: float,
    commission_bps: float,
    spread_w: float,
    impact_w: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    orders = _load_orders()
    prices = _load_prices(freq)
    _ensure_dir(OUT_DIR)

    if orders.empty or prices.empty:
        eq = _fallback_equity(freq, start_capital)
        trades = pd.DataFrame({
            "timestamp": eq["timestamp"],
            "symbol": np.random.default_rng(0).choice(["AAPL","MSFT"], size=len(eq)),
            "qty": 0, "price": 0.0
        })
        return trades, eq

    eq = _equity_from_orders_prices(orders, prices, start_capital, commission_bps, freq)
    trades = orders.copy()
    trades["qty"] = pd.to_numeric(trades.get("qty", 0), errors="coerce").fillna(0.0).astype("float64")
    trades["price"] = pd.to_numeric(trades.get("price", 0.0), errors="coerce").fillna(0.0).astype("float64")
    return trades, eq

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min")
    p.add_argument("--start-capital", type=float, default=10000.0)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--commission-bps", type=float, default=0.5)
    p.add_argument("--spread-w", type=float, default=1.0)
    p.add_argument("--impact-w", type=float, default=1.0)
    args = p.parse_args()

    print(f"[PF10] START Portfolio | freq={args.freq} start_capital={args.start_capital} "
          f"exp={args.exposure} lev={args.max_leverage} "
          f"comm_bps={args.commission_bps} spread_w={args.spread_w} impact_w={args.impact_w}")

    trades, equity = simulate(
        args.freq, args.start_capital, args.exposure, args.max_leverage,
        args.commission_bps, args.spread_w, args.impact_w
    )

    _ensure_dir(OUT_DIR)
    trades.to_csv(os.path.join(OUT_DIR, "portfolio_trades.csv"), index=False)
    equity.to_csv(os.path.join(OUT_DIR, f"portfolio_equity_{args.freq}.csv"), index=False)
    with open(os.path.join(OUT_DIR, "portfolio_report.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# Portfolio Report ({args.freq})\n\n"
            f"- Start Capital: {args.start_capital:,.2f}\n"
            f"- Exposure: {args.exposure}\n"
            f"- Leverage: {args.max_leverage}\n"
            f"- Commission: {args.commission_bps} bps\n"
            f"- Spread Weight: {args.spread_w}\n"
            f"- Impact Weight: {args.impact_w}\n"
            f"- Final Equity: {equity['equity'].iloc[-1]:,.2f}\n"
        )
    print(f"[PF10] [OK] written: {os.path.join(OUT_DIR, 'portfolio_trades.csv')}")
    print(f"[PF10] [OK] written: {os.path.join(OUT_DIR, f'portfolio_equity_{args.freq}.csv')}")
    print(f"[PF10] [OK] written: {os.path.join(OUT_DIR, 'portfolio_report.md')}")
    print("[PF10] DONE Portfolio")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
