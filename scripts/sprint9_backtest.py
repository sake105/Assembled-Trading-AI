# scripts/sprint9_backtest.py
from __future__ import annotations
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


OUT_DIR = "output"


def _read_prices_with_fallback(freq: str) -> pd.DataFrame:
    base = Path(OUT_DIR)
    if freq == "1d":
        candidates = [base / "aggregates" / "daily.parquet"]
    elif freq == "5min":
        candidates = [
            base / "aggregates" / "5min.parquet",          # bevorzugt
            base / "assembled_intraday" / "5min.parquet",  # falls vorhanden
            base / "features" / "base_5min.parquet",       # Fallback
        ]
    else:
        raise ValueError(f"Unbekannte freq '{freq}'")

    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        raise FileNotFoundError(f"Kein Preis-File gefunden. Versucht: {', '.join(map(str, candidates))}")

    df = pd.read_parquet(p)
    need = {"timestamp", "symbol", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"Preis-File {p} hat nicht alle Spalten (brauche {sorted(need)}), hat: {list(df.columns)}")

    df = df[["timestamp", "symbol", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    df = df.dropna(subset=["timestamp", "symbol", "close"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    print(f"[BT9] Prices: {p}  rows={len(df)}  symbols={df['symbol'].nunique()}")
    return df


def _read_prices(freq: str, price_file: str | None) -> pd.DataFrame:
    if price_file:
        p = Path(price_file)
        if not p.exists():
            raise FileNotFoundError(f"Preis-File nicht gefunden: {p}")
        df = pd.read_parquet(p)
        if not {"timestamp", "symbol", "close"}.issubset(df.columns):
            raise ValueError(f"{p} muss Spalten ['timestamp','symbol','close'] enthalten.")
        df = df[["timestamp", "symbol", "close"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
        df = df.dropna(subset=["timestamp", "symbol", "close"]).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        print(f"[BT9] Prices: {p}  rows={len(df)}  symbols={df['symbol'].nunique()}")
        return df
    return _read_prices_with_fallback(freq)


def _read_orders(freq: str) -> pd.DataFrame:
    p = Path(OUT_DIR) / f"orders_{freq}.csv"
    if not p.exists():
        # Kein Orders-File: backtest mit 0-Position (flach)
        print(f"[BT9] Orders nicht gefunden: {p}  → flache Equity (nur Startkapital).")
        return pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    df = pd.read_csv(p)
    # normalize
    if "timestamp" not in df.columns:
        raise ValueError(f"Orders-File {p} hat keine 'timestamp'-Spalte.")
    for c in ("symbol", "side"):
        if c not in df.columns:
            df[c] = ""  # toleranter
    for c in ("qty", "price"):
        if c not in df.columns:
            df[c] = 0.0

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["side"] = df["side"].astype(str).str.upper().str.strip()
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"[BT9] Reading orders: {p}  rows={len(df)}")
    return df


def _simulate_equity(prices: pd.DataFrame, orders: pd.DataFrame, start_capital: float, freq: str) -> pd.DataFrame:
    # Timeline & Price pivot
    prices = prices.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    timeline = prices["timestamp"].sort_values().drop_duplicates().to_list()
    symbols = sorted(prices["symbol"].unique().tolist())
    px = prices.pivot(index="timestamp", columns="symbol", values="close").sort_index()

    # Orders nach Timestamp gruppieren
    orders = orders.sort_values("timestamp").reset_index(drop=True)
    orders_by_ts = {}
    if len(orders):
        for ts, group in orders.groupby("timestamp"):
            orders_by_ts[ts] = group

    # Simulationszustand
    cash = float(start_capital)
    pos = {s: 0.0 for s in symbols}
    equity_series = []

    for ts in timeline:
        # Orders zum aktuellen Timestamp ausführen (Market zum angegebenen Orderpreis)
        if ts in orders_by_ts:
            g = orders_by_ts[ts]
            for _, row in g.iterrows():
                sym = row.get("symbol", "")
                side = row.get("side", "")
                qty = float(row.get("qty", 0.0))
                price = float(row.get("price", np.nan))
                if not sym or np.isnan(price) or qty == 0.0:
                    continue
                if side == "BUY":
                    cash -= qty * price
                    pos[sym] = pos.get(sym, 0.0) + qty
                elif side == "SELL":
                    cash += qty * price
                    pos[sym] = pos.get(sym, 0.0) - qty

        # Mark-to-Market
        if ts in px.index:
            mtm = 0.0
            row = px.loc[ts]
            for s in symbols:
                pr = float(row.get(s, np.nan))
                if not np.isnan(pr):
                    mtm += pos.get(s, 0.0) * pr
            equity = cash + mtm
        else:
            equity = cash  # falls Lücke

        equity_series.append((ts, float(equity)))

    eq = pd.DataFrame(equity_series, columns=["timestamp", "equity"])
    # Sanitizing
    s = pd.Series(eq["equity"].values, index=eq["timestamp"])
    s = s.replace([np.inf, -np.inf], np.nan).ffill().fillna(start_capital)
    eq["equity"] = s.values
    return eq


def _write_report(eq: pd.DataFrame, freq: str) -> None:
    curve_path = os.path.join(OUT_DIR, f"equity_curve_{freq}.csv")
    rep_path = os.path.join(OUT_DIR, f"performance_report_{freq}.md")

    eq.to_csv(curve_path, index=False)

    pf = float(eq["equity"].iloc[-1] / max(eq["equity"].iloc[0], 1e-12))
    # einfache Sharpe (daily/5min gleich behandelt, hier demo)
    ret = eq["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = float(ret.mean() / (ret.std() + 1e-12)) if not ret.empty else float("nan")

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(f"# Performance Report ({freq})\n\n")
        f.write(f"- Final PF: {pf:.4f}\n")
        f.write(f"- Sharpe: {sharpe:.4f}\n")
        f.write(f"- Rows: {len(eq)}\n")
        f.write(f"- First: {eq['timestamp'].iloc[0]}\n")
        f.write(f"- Last:  {eq['timestamp'].iloc[-1]}\n")

    print(f"[BT9] [OK] written: {curve_path}")
    print(f"[BT9] [OK] written: {rep_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, choices=["1d", "5min"], default="5min")
    p.add_argument("--start-capital", type=float, default=10000.0)
    p.add_argument("--price-file", type=str, default=None, help="Optional: Pfad zu einem Parquet mit ['timestamp','symbol','close']")
    args = p.parse_args()

    print(f"[BT9] START Backtest | freq={args.freq}")
    prices = _read_prices(args.freq, args.price_file)
    orders = _read_orders(args.freq)
    eq = _simulate_equity(prices, orders, start_capital=float(args.start_capital), freq=args.freq)
    _write_report(eq, args.freq)
    print("[BT9] DONE Backtest")


if __name__ == "__main__":
    main()
