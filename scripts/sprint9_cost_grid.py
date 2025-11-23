# scripts/sprint9_cost_grid.py
from __future__ import annotations
import argparse, pathlib as pl, itertools, numpy as np, pandas as pd

ROOT    = pl.Path(__file__).resolve().parents[1]
OUT     = ROOT / "output"
AGG     = ROOT / "output" / "aggregates"

def _prices(freq: str) -> pd.DataFrame:
    p = AGG / ("daily.parquet" if freq=="1d" else "5min.parquet")
    df = pd.read_parquet(p)[["timestamp","symbol","close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

def _signals(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    def per_symbol(d: pd.DataFrame) -> pd.DataFrame:
        px = d["close"].astype(float)
        f = px.ewm(span=fast, adjust=False, min_periods=fast).mean()
        s = px.ewm(span=slow, adjust=False, min_periods=slow).mean()
        sig = (f > s).astype(int) - (f < s).astype(int)
        return pd.DataFrame({"timestamp": d["timestamp"], "symbol": d["symbol"], "sig": sig})
    return df.groupby("symbol", group_keys=False).apply(per_symbol).reset_index(drop=True)

def _simulate(df: pd.DataFrame, commission_bps: float, spread_w: float, impact_w: float, notional: float) -> tuple[pd.DataFrame, dict]:
    # einfache Portfolio-Ret-Sim: port_ret = mean(sig * ret) – (Kosten)
    # Kosten ~ commission + spread + impact * |Δposition|
    ret = (np.log(df["close"]).groupby(df["symbol"]).diff()).fillna(0.0)
    df = df.assign(ret=ret)

    pivot_sig = df.pivot(index="timestamp", columns="symbol", values="sig").fillna(0.0)
    pivot_ret = df.pivot(index="timestamp", columns="symbol", values="ret").fillna(0.0)

    w = (pivot_sig != 0).astype(float)
    w = w.div(w.sum(axis=1).replace(0,1), axis=0)  # Gleichgewichtung aktivierter Signale

    gross = (pivot_sig * pivot_ret * w).sum(axis=1)

    # Transaktionen aus Signalwechseln: Δpos in {−2,0,+2} je Symbol → Kostenmaß
    pos = pivot_sig.copy()
    dpos = pos.diff().abs().fillna(0.0)
    # Kosten pro Schritt als lineare Kombination
    step_cost = (commission_bps/1e4) + spread_w*0.0001 + impact_w*0.0001
    costs = (dpos.sum(axis=1) * step_cost)

    net = gross - costs
    equity = notional * np.exp(net.cumsum())
    eq = pd.DataFrame({"timestamp": equity.index, "equity": equity.values})

    rr = np.log(equity).diff().dropna()
    rep = {
        "pf": float(equity.iloc[-1]/equity.iloc[0]),
        "sharpe": float(rr.mean()/(rr.std()+1e-12)) if len(rr) else float("nan"),
        "trades": int(dpos.sum().sum()/2),  # grobe Zählung
    }
    return eq, rep

def run_grid(freq: str, commissions: list[float], spread_ws: list[float], impact_ws: list[float], fast: int, slow: int, notional: float):
    px = _prices(freq)
    sig = _signals(px, fast, slow)
    df = px.merge(sig, on=["timestamp","symbol"], how="left").fillna({"sig":0.0})

    best = None
    best_tuple = None
    for c, sw, iw in itertools.product(commissions, spread_ws, impact_ws):
        eq, rep = _simulate(df, c, sw, iw, notional)
        if (best is None) or (rep["pf"] > best["pf"]):
            best = rep | {"commission":c, "spread":sw, "impact":iw}
            best_tuple = (eq, rep)

    # Speichern Kurzreport
    OUT.mkdir(parents=True, exist_ok=True)
    rpt = OUT / "cost_grid_report.md"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"# Cost Grid ({freq})\n\n")
        f.write(f"Bestes Grid → PF={best['pf']:.16f} | comm={best['commission']}bps | spread={best['spread']} | impact={best['impact']} | trades={best_tuple[1]['trades']}\n")
    print(f"[GRID] [OK] written: {rpt}")
    return best_tuple

def _parse_list_floats(vals: list[str]) -> list[float]:
    out = []
    for v in vals:
        if "," in v and " " not in v:
            # deutschsprachige Kommas → Punkte
            v = v.replace(",", ".")
        out.append(float(v))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", choices=["1d","5min"], required=True)
    ap.add_argument("--commission-bps", nargs="+", required=True)
    ap.add_argument("--spread-w", nargs="+", required=True)
    ap.add_argument("--impact-w", nargs="+", required=True)
    ap.add_argument("--ema-fast", type=int, default=20)
    ap.add_argument("--ema-slow", type=int, default=60)
    ap.add_argument("--notional", type=float, default=10_000.0)
    args = ap.parse_args()

    commissions = _parse_list_floats(args.commission_bps)
    spread_ws   = _parse_list_floats(args.spread_w)
    impact_ws   = _parse_list_floats(args.impact_w)

    print(f"[GRID] START Grid | freq={args.freq} | commission={', '.join(map(str,commissions))} | spread_w={', '.join(map(str,spread_ws))} | impact_w={', '.join(map(str,impact_ws))}")
    eq, rep = run_grid(
        args.freq, commissions, spread_ws, impact_ws,
        fast=args.ema_fast, slow=args.ema_slow, notional=args.notional
    )
    out = OUT / "cost_grid_equity.csv"
    eq.to_csv(out, index=False)
    print(f"[GRID] DONE Grid")

if __name__ == "__main__":
    main()
