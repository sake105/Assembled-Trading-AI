# -*- coding: utf-8 -*-
"""
Sprint 8 – Execution & Kosten (robust + rückwärtskompatibel)
- Liest Features (benötigt close)
- Erzeugt Dummy-Orders, Kosten-Sensitivität und Fill-Sim
- Schreibt: output/{orders.csv, cost_sensitivity.md, fill_sim.md}
"""

from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEAT_DIR = os.path.join(ROOT, "output", "features")
OUT_DIR  = os.path.join(ROOT, "output")

def info(msg: str) -> None:
    ts = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[EXEC] {msg}")

def _ensure_out() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def _parse_args() -> argparse.Namespace:
    # --- Abwärtskompatible CLI ---
    # Entferne unbekanntes --notional <val> Flag, falls älteres Orchestrator-Script es setzt
    if "--notional" in sys.argv:
        try:
            i = sys.argv.index("--notional")
            # optional: notional_val = float(sys.argv[i+1])
            del sys.argv[i:i+2]
        except Exception:
            # nur Flag weg
            sys.argv.remove("--notional")

    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min", help="1min/5min/…")
    p.add_argument("--commission-bps", type=float, default=0.5, help="Kommission in Basispunkten")
    return p.parse_args()

def _load_features(freq: str) -> pd.DataFrame:
    # Nimmt 'base' als Quelle ( enthält close/ret_* )
    path = os.path.join(FEAT_DIR, f"base_{freq}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

    df = df.rename(columns={c: str(c).lower() for c in df.columns})
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

def _make_orders(df: pd.DataFrame, commission_bps: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp","symbol","side","qty","price","commission"])

    # einfache Signal-Heuristik: kaufe, wenn ret_1 > 0; verkaufe, wenn ret_1 < 0
    if "ret_1" not in df.columns and "close" in df.columns:
        df = df.copy()
        df["ret_1"] = df.groupby("symbol")["close"].pct_change()

    orders = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("timestamp")
        for _, row in g.iterrows():
            r = float(row.get("ret_1", 0.0))
            px = float(row.get("close", np.nan))
            if not np.isfinite(px):
                continue
            if r > 0:
                side = "BUY"
                qty  = 1.0
            elif r < 0:
                side = "SELL"
                qty  = 1.0
            else:
                continue
            comm = px * qty * (commission_bps / 10000.0)
            orders.append({
                "timestamp": row["timestamp"],
                "symbol": sym,
                "side": side,
                "qty": qty,
                "price": px,
                "commission": comm
            })
    if not orders:
        return pd.DataFrame(columns=["timestamp","symbol","side","qty","price","commission"])
    out = pd.DataFrame(orders)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def _write_reports(orders: pd.DataFrame, freq: str) -> None:
    _ensure_out()
    orders_path = os.path.join(OUT_DIR, "orders.csv")
    orders.to_csv(orders_path, index=False)
    info(f"[OK] written: {orders_path}")

    # Kosten-Sensitivität (einfaches Raster bps = 0, 0.5, 1.0)
    sens_rows = []
    for bps in [0.0, 0.5, 1.0]:
        cost = float((orders["price"] * orders["qty"]).sum() * (bps / 10000.0)) if not orders.empty else 0.0
        sens_rows.append({"commission_bps": bps, "total_cost": cost})
    sens = pd.DataFrame(sens_rows)
    sens_md = os.path.join(OUT_DIR, "cost_sensitivity.md")
    with open(sens_md, "w", encoding="utf-8") as f:
        f.write("# Cost Sensitivity (bps)\n\n")
        f.write(sens.to_markdown(index=False))
        f.write("\n")
    info(f"[OK] written: {sens_md}")

    # Fill-Simulation (Dummy)
    fill = pd.DataFrame({
        "metric": ["avg_slippage_bp", "fill_rate"],
        "value":  [0.2, 0.99]
    })
    fill_md = os.path.join(OUT_DIR, "fill_sim.md")
    with open(fill_md, "w", encoding="utf-8") as f:
        f.write("# Fill Simulation\n\n")
        f.write(fill.to_markdown(index=False))
        f.write("\n")
    info(f"[OK] written: {fill_md}")

def main() -> int:
    args = _parse_args()
    print(f"[COST] Start Execution & Costs | freq={args.freq} notional=10000 commission={args.commission_bps} bps")
    info(f"START Execution | freq={args.freq}")

    df = _load_features(args.freq)
    orders = _make_orders(df, args.commission_bps)
    _write_reports(orders, args.freq)

    info("DONE Execution")
    return 0

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        print(f"[EXEC] ERROR {e}", file=sys.stderr)
        sys.exit(1)
