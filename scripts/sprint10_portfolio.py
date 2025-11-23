# scripts/sprint10_portfolio.py
from __future__ import annotations
import argparse, os, pathlib as pl
import numpy as np
import pandas as pd

ROOT = pl.Path(__file__).resolve().parents[1]
OUT  = ROOT / "output"

def _load_orders(freq: str) -> pd.DataFrame:
    p = OUT / f"orders_{freq}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Orders nicht gefunden: {p} – erst sprint9_execute.py laufen lassen.")
    df = pd.read_csv(p)
    for c in ["timestamp", "symbol", "side", "qty", "price"]:
        if c not in df.columns: raise ValueError(f"Spalte '{c}' fehlt in {p}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp","symbol","side"])
    # normalisieren
    df["qty"]   = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["side"]  = df["side"].str.upper().str.strip()
    return df.sort_values(["timestamp","symbol"]).reset_index(drop=True)

def _simulate(eq0: float, orders: pd.DataFrame,
              commission_bps: float, spread_w: float, impact_w: float,
              freq: str) -> tuple[pd.DataFrame, dict]:
    # Timeline aus Orders ableiten – falls nur wenige Orders, erweitern wir leicht
    if orders.empty:
        ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=60, freq=freq)
        eq = pd.DataFrame({"timestamp": ts, "equity": float(eq0)})
        rep = {"final_pf": 1.0, "sharpe": float("nan"), "trades": 0}
        return eq, rep

    t0, t1 = orders["timestamp"].min(), orders["timestamp"].max()
    tl = pd.date_range(start=t0, end=t1, freq=freq)
    equity = np.full(len(tl), eq0, dtype=np.float64)

    # primitive, aber reproduzierbare Kostenmodellierung:
    # - Kommission in bps * notional-per-trade
    # - bid/ask via spread_w in bp Äquivalent
    # - 'impact' als zusätzl. Preisabschlag in bp
    # Wir haben keinen Notional im CSV; wir modellieren 1x Preis * qty als notional surrogate.
    k = commission_bps * 1e-4
    s = spread_w       * 1e-4
    im= impact_w       * 1e-4

    # Gruppiere Orders je Zeitstempel, summiere Cash-Delta
    orders = orders.copy()
    orders["sign"] = np.where(orders["side"].eq("BUY"), +1.0,
                       np.where(orders["side"].eq("SELL"), -1.0, 0.0))
    orders["notional"] = (orders["qty"].abs() * orders["price"].abs()).astype(np.float64)

    # effektiver Preisaufschlag/-abschlag
    # BUY zahlt: price * (1 + s + im) + kommission
    # SELL erhält: price * (1 - s - im) - kommission
    orders["cash_delta"] = np.where(
        orders["sign"] > 0,
        -(orders["qty"] * orders["price"] * (1.0 + s + im) + k * orders["notional"]),
        +(orders["qty"].abs() * orders["price"] * (1.0 - s - im) - k * orders["notional"])
    )

    ts_to_delta = (orders.groupby(pd.Grouper(key="timestamp", freq=freq))["cash_delta"]
                         .sum()
                         .reindex(tl, fill_value=0.0)
                         .to_numpy())

    # wende Cash-Deltas ab jeweiligem Index an (cum)
    equity = equity + np.cumsum(ts_to_delta)

    eq = pd.DataFrame({"timestamp": tl, "equity": equity})
    # simple Kennzahlen
    pf = float(equity[-1] / equity[0]) if equity.size else 1.0
    ret = pd.Series(equity).pct_change().replace([np.inf,-np.inf], np.nan).dropna()
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252 if freq=="1d" else 252*78)) if len(ret)>5 and ret.std()>0 else float("nan")
    rep = {"final_pf": pf, "sharpe": sharpe, "trades": int(len(orders))}
    return eq, rep

def _write(eq: pd.DataFrame, rep: dict, freq: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eq.to_csv(OUT / f"portfolio_equity_{freq}.csv", index=False)
    with open(OUT / "portfolio_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Portfolio Report ({freq})\n\n")
        f.write(f"- Final PF: {rep['final_pf']:.4f}\n")
        f.write(f"- Sharpe: {rep['sharpe']}\n")
        f.write(f"- Trades: {rep['trades']}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["1d","5min"], required=True)
    ap.add_argument("--start-capital", type=float, default=10000.0)
    ap.add_argument("--commission-bps", type=float, default=0.0)
    ap.add_argument("--spread-w",       type=float, default=0.25)
    ap.add_argument("--impact-w",       type=float, default=0.5)
    a = ap.parse_args()

    orders = _load_orders(a.freq)
    eq, rep = _simulate(a.start_capital, orders, a.commission_bps, a.spread_w, a.impact_w, a.freq)
    _write(eq, rep, a.freq)
    print(f"[PF10] DONE | PF={rep['final_pf']:.4f} Sharpe={rep['sharpe']} Trades={rep['trades']}")

if __name__ == "__main__":
    main()
