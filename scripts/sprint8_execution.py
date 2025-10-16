# -*- coding: utf-8 -*-
"""
Sprint 8 Execution – stabil & kompatibel
"""

from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "output")

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _simulate_orders(freq: str, commission_bps: float, notional: float) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    ts = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq=freq)
    base_qty = max(int(round(notional / 100000.0)), 1)  # sehr grobe Skalierung
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": rng.choice(["AAPL", "MSFT"], size=n),
        "side": rng.choice(["BUY", "SELL"], size=n),
        "qty": rng.integers(1, 10, size=n).astype(int) * base_qty,
        "price": rng.uniform(100, 200, size=n)
    })
    df["commission_bps"] = float(commission_bps)
    df["commission"] = df["price"] * df["qty"] * (commission_bps / 10000.0)
    df["fill_price"] = df["price"] + np.where(df["side"] == "BUY", 0.01, -0.01)
    # sanitation
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in ["qty", "price", "commission", "fill_price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").ffill().bfill().fillna(0.0)
    return df

def _write_reports(commission_bps: float) -> None:
    _ensure_dir(OUT_DIR)
    with open(os.path.join(OUT_DIR, "cost_sensitivity.md"), "w", encoding="utf-8") as f:
        f.write("# Cost Sensitivity (bps)\n\n")
        f.write("|bps|impact|\n|-|-|\n")
        for b in [0.0, commission_bps, max(commission_bps*2, 1.0)]:
            f.write(f"|{b}|placeholder|\n")
    with open(os.path.join(OUT_DIR, "fill_sim.md"), "w", encoding="utf-8") as f:
        f.write("# Fill Simulation\n\nSynthetic fill simulation complete.\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=str, default="5min")
    p.add_argument("--commission-bps", type=float, default=0.5)
    # Für Kompatibilität zum PS1 (ignoriert die Details; nur zur Skalierung/Logging)
    p.add_argument("--notional", type=float, default=10000.0)
    args = p.parse_args()

    print(f"[EXEC] START Execution | freq={args.freq}")
    _ensure_dir(OUT_DIR)
    orders = _simulate_orders(args.freq, args.commission_bps, args.notional)
    orders.to_csv(os.path.join(OUT_DIR, "orders.csv"), index=False)
    print(f"[EXEC] [OK] written: {os.path.join(OUT_DIR, 'orders.csv')}")
    _write_reports(args.commission_bps)
    print(f"[EXEC] [OK] written: {os.path.join(OUT_DIR, 'cost_sensitivity.md')}")
    print(f"[EXEC] [OK] written: {os.path.join(OUT_DIR, 'fill_sim.md')}")
    print("[EXEC] DONE Execution")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
