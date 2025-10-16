# -*- coding: utf-8 -*-
"""
Sprint 10 – Portfolio Simulator (robust gegen Duplikate & Lücken)
Erwartet Features (base_{freq}.parquet) und simuliert ein einfaches Portfolio.

CLI:
  python scripts/sprint10_portfolio.py \
    --freq 5min --start-capital 10000 --exposure 1 --max-leverage 1 \
    --commission-bps 0.5 --spread-w 1 --impact-w 1

Outputs:
  - output/portfolio_trades.csv
  - output/portfolio_equity_{freq}.csv
  - output/portfolio_report.md
"""

from __future__ import annotations
import os, sys, argparse, math
from dataclasses import dataclass
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEAT_DIR = os.path.join(ROOT, "output", "features")
OUT_DIR  = os.path.join(ROOT, "output")

def info(tag: str, msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{tag}] {msg}")

def _ensure_out() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min")
    p.add_argument("--start-capital", type=float, default=10_000.0)
    p.add_argument("--exposure", type=float, default=1.0)       # 1.0 = 100% Kapital investiert (ohne Leverage)
    p.add_argument("--max-leverage", type=float, default=1.0)   # Sicherheits-Cap
    p.add_argument("--commission-bps", type=float, default=0.5)
    p.add_argument("--spread-w", type=float, default=1.0)       # wird als Kostengewicht genutzt
    p.add_argument("--impact-w", type=float, default=1.0)       # wird als Kostengewicht genutzt
    return p.parse_args()

@dataclass
class Trade:
    timestamp: pd.Timestamp
    symbol: str
    side: str
    qty: float
    price: float
    commission: float

def _load_base(freq: str) -> pd.DataFrame:
    path = os.path.join(FEAT_DIR, f"base_{freq}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Normalisieren
    df.columns = [str(c).lower() for c in df.columns]
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    # nur relevante Spalten
    keep = [c for c in ["timestamp","symbol","open","high","low","close","volume","ret_1","ret_5"] if c in df.columns]
    df = df[keep].copy()
    # Duplikate je (timestamp, symbol) – letzte Zeile gewinnt (z.B. aus Aggregat-Dateien)
    df = df.sort_values(["symbol","timestamp"]).drop_duplicates(subset=["timestamp","symbol"], keep="last")
    return df.reset_index(drop=True)

def _costs(price: float, qty: float, commission_bps: float, spread_w: float, impact_w: float) -> float:
    notional = abs(price * qty)
    commission = notional * (commission_bps / 10_000.0)
    # sehr einfache Dummy-Modelle für Spread/Impact
    spread   = notional * (0.5 / 10_000.0) * float(spread_w)
    impact   = notional * (0.5 / 10_000.0) * float(impact_w)
    return commission + spread + impact

def _make_signals(df: pd.DataFrame) -> pd.Series:
    """Ein simples Momentum-Signal: sign(ret_1). Fallback: close-Momentum."""
    if "ret_1" not in df.columns and "close" in df.columns:
        g = df.groupby("symbol", group_keys=False)["close"]
        df = df.assign(ret_1=g.pct_change())
    sig = np.sign(df["ret_1"].fillna(0.0))
    return sig

def _position_target_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zielgewichte: gleichgewichtet LONG bei ret_1>0, SHORT bei ret_1<0; 0 sonst.
    Skaliert so, dass Summe |w| über Symbole = 1 (falls Signale vorhanden).
    """
    df = df.copy()
    df["sig"] = _make_signals(df)
    # Pro Timestamp normalisieren
    def _row_weights(g: pd.DataFrame) -> pd.DataFrame:
        s = g["sig"].values
        denom = np.sum(np.abs(s))
        if denom <= 0:
            g["w"] = 0.0
        else:
            g["w"] = s / denom
        return g
   df = (
    df.loc[:, ["timestamp", "symbol", "sig", "close"]]
      .groupby("timestamp", group_keys=False)[["symbol", "sig", "close"]]
      .apply(_row_weights)
)
    return df[["timestamp","symbol","w","close"]]

def _build_price_matrix(df: pd.DataFrame, symbols: list[str], timeline: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Baut Preis-Matrix (timestamp x symbol) robust:
    - Duplikate je (timestamp, symbol): last
    - pivot_table mit aggfunc="last" verhindert Fehler bei Rest-Duplikaten
    - Reindex auf timeline & columns (symbolliste), Füllen via ffill
    """
    tmp = df.sort_values(["symbol","timestamp"]).drop_duplicates(subset=["timestamp","symbol"], keep="last")
    px = pd.pivot_table(
        tmp,
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
        observed=False,
    )
    # robustes Reindexing
    px = px.reindex(index=pd.DatetimeIndex(timeline).unique().sort_values())
    # nur gewünschte Spalten, Reihenfolge fixieren
    px = px.reindex(columns=sorted(set(symbols)))
    # Lücken füllen (intraday: forward-fill reicht)
    px = px.ffill()
    return px

def _weights_matrix(df_w: pd.DataFrame, symbols: list[str], timeline: pd.DatetimeIndex) -> pd.DataFrame:
    tmp = df_w.sort_values(["symbol","timestamp"]).drop_duplicates(subset=["timestamp","symbol"], keep="last")
    W = pd.pivot_table(
        tmp,
        index="timestamp",
        columns="symbol",
        values="w",
        aggfunc="last",
        observed=False,
    )
    W = W.reindex(index=pd.DatetimeIndex(timeline).unique().sort_values())
    W = W.reindex(columns=sorted(set(symbols))).fillna(0.0)
    return W

def simulate(df: pd.DataFrame,
             start_capital: float,
             exposure: float,
             max_leverage: float,
             commission_bps: float,
             spread_w: float,
             impact_w: float):
    """
    Sehr einfacher Portfolio-Simulator mit Zielgewichten je Bar.
    """
    # Grundschutz für Parameter
    exposure = float(exposure or 0.0)
    max_leverage = float(max_leverage or 1.0)

    # Symbole & Zeitachse
    symbols = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    # Timeline direkt aus Daten ableiten (verhindert Duplikate von resample/assemble)
    timeline = pd.DatetimeIndex(sorted(df["timestamp"].unique()))

    # Preis- und Gewichtsmatrizen
    PX = _build_price_matrix(df, symbols, timeline)
    W  = _weights_matrix(_position_target_weights(df), symbols, timeline)

    # Startpreise für erste Positionsberechnung
    PX_next = PX.shift(-1)
    ret = PX_next / PX - 1.0
    ret = ret.fillna(0.0)

    # Positionsnotional pro Schritt (Zielgewichte * (exposure * Equity))
    equity = pd.Series(index=timeline, dtype=float)
    equity.iloc[0] = float(start_capital)

    # Trades-Log
    trade_rows = []

    # Startpositionen (0)
    pos_qty = pd.Series(0.0, index=symbols)

    for i in range(len(timeline) - 1):
        t  = timeline[i]
        t1 = timeline[i+1]

        eq  = float(equity.iloc[i])
        w   = W.loc[t].fillna(0.0)

        # clamp leverage
        total_abs_w = float(np.abs(w).sum())
        if total_abs_w > 0:
            scale = min(exposure / total_abs_w, max_leverage / total_abs_w)
            w = w * scale

        px = PX.loc[t]
        px1 = PX.loc[t1]

        # Ziel-Notional und Stückzahl
        target_notional = w * eq
        target_qty = (target_notional / px.replace(0, np.nan)).fillna(0.0)

        # Trades = delta zur aktuellen Position
        delta = target_qty - pos_qty

        # Kosten & Ausführung am Preis t (vereinfachtes Modell)
        step_costs = 0.0
        for sym, dq in delta.items():
            if dq == 0 or not np.isfinite(dq):
                continue
            price = float(px.get(sym, np.nan))
            if not np.isfinite(price):
                continue
            side = "BUY" if dq > 0 else "SELL"
            c = _costs(price, dq, commission_bps, spread_w, impact_w)
            step_costs += c
            trade_rows.append({
                "timestamp": t,
                "symbol": sym,
                "side": side,
                "qty": float(dq),
                "price": price,
                "commission": c
            })

        # Positionsupdate (nach Trade)
        pos_qty = target_qty

        # Equity-Update mit nachfolgender Rendite
        port_ret = float((pos_qty * (px1 / px - 1.0)).replace([np.inf,-np.inf], np.nan).fillna(0.0).sum())
        new_eq = eq * (1.0 + port_ret) - step_costs
        equity.iloc[i+1] = max(new_eq, 0.0)

    # Ergebnisse bauen
    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame({
        "timestamp": timeline,
        "equity": equity.values
    })

    return trades_df, equity_df

def write_outputs(trades_df: pd.DataFrame, equity_df: pd.DataFrame, freq: str) -> None:
    _ensure_out()
    trades_path = os.path.join(OUT_DIR, "portfolio_trades.csv")
    eq_path     = os.path.join(OUT_DIR, f"portfolio_equity_{freq}.csv")
    rep_path    = os.path.join(OUT_DIR, "portfolio_report.md")

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(eq_path, index=False)

    # kleines Reportchen
    eq = equity_df["equity"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    start = float(eq.iloc[0]) if len(eq) else np.nan
    end   = float(eq.iloc[-1]) if len(eq) else np.nan
    days  = max( (equity_df["timestamp"].iloc[-1] - equity_df["timestamp"].iloc[0]).days , 1) if len(equity_df) > 1 else 1
    try:
        cagr = (end / max(start, 1e-9)) ** (365.0 / days) - 1.0
    except Exception:
        cagr = np.nan
    pf = (end / max(start, 1e-9)) if np.isfinite(end) and np.isfinite(start) else np.nan

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Portfolio Report\n\n")
        f.write(f"- Start equity: {start:,.2f}\n")
        f.write(f"- End equity:   {end:,.2f}\n")
        f.write(f"- Profit factor: {pf:,.4f}\n")
        f.write(f"- Approx CAGR:   {cagr:,.4%}\n")

    info("PF10", f"[OK] written: {trades_path}")
    info("PF10", f"[OK] written: {eq_path}")
    info("PF10", f"[OK] written: {rep_path}")

def main() -> None:
    args = _parse_args()
    info("PF10", f"START Portfolio | freq={args.freq} start_capital={args.start_capital} exp={args.exposure} lev={args.max_leverage} comm_bps={args.commission_bps} spread_w={args.spread_w} impact_w={args.impact_w}")

    df = _load_base(args.freq)
    if df.empty:
        raise SystemExit("Keine Feature-Daten gefunden. Bitte Sprint8 laufen lassen.")

    trades_df, equity_df = simulate(
        df=df,
        start_capital=args.start_capital,
        exposure=args.exposure,
        max_leverage=args.max_leverage,
        commission_bps=args.commission_bps,
        spread_w=args.spread_w,
        impact_w=args.impact_w,
    )
    write_outputs(trades_df, equity_df, args.freq)
    info("PF10", "DONE Portfolio")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(e, file=sys.stderr)
        sys.exit(1)
