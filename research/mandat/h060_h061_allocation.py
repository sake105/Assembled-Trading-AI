"""Welle 27 — H-060 robuste Gold-Quote (Szenario-Maximin), H-061 Rebalancing-Kadenz.

Historischer Pfad 2005–26, per-Sleeve-Basis, SPY 18,46 %, Gold/Silber §23 (>1J 0 %, <1J 44 %
mit konservativem Lot-Datum), End-Liquidation. Gold-Szenarien: Tagesrenditen ×{1.0,0.5,0.3,0.0}.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
TAX_EQ, TAX_PM_ST = 0.1846, 0.44  # precious metals long-term 0 %


def load(scale_gold: float = 1.0, with_silver: bool = False) -> pd.DataFrame:
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    if with_silver:
        slv = pd.read_parquet(DATA / "bond_SLV.parquet").set_index("timestamp")["close"]
        df["SLV"] = pd.Series(slv.values, index=pd.DatetimeIndex(slv.index)).reindex(
            df.index
        )
    df = df.dropna()
    df = df[
        df.index
        >= pd.Timestamp("2006-05-01" if with_silver else "2005-01-01", tz="UTC")
    ]
    if scale_gold != 1.0:
        for c in [c for c in ("GLD", "SLV") if c in df.columns]:
            r = df[c].pct_change().fillna(0.0) * scale_gold
            df[c] = float(df[c].iloc[0]) * (1 + r).cumprod()
    return df


def simulate(df: pd.DataFrame, weights: dict, *, cadence: str) -> dict:
    """cadence: never | annual | biennial | band20"""
    assets = list(weights)
    idx = df.index
    P = df[assets].values
    u = {a: weights[a] * START / P[0][i] for i, a in enumerate(assets)}
    basis = {a: weights[a] * START for a in assets}
    pm_lot = {a: idx[0] for a in assets}
    tax_paid, st_hits = 0.0, 0
    ye = set(idx.to_series().groupby(idx.to_period("Y")).max())
    me = set(idx.to_series().groupby(idx.to_period("M")).max())
    eq = []

    def sell(a, i, d, t):
        nonlocal tax_paid, st_hits
        px = P[i][assets.index(a)]
        val = u[a] * px
        d = min(d, val)
        if d <= 0:
            return 0.0
        fr = d / val
        gain = d - fr * basis[a]
        u[a] *= 1 - fr
        basis[a] *= 1 - fr
        if a == "SPY":
            rate = TAX_EQ
        else:
            lt = (t - pm_lot[a]).days >= 365
            rate = 0.0 if lt else TAX_PM_ST
            if not lt and gain > 0:
                st_hits += 1
        tx = max(gain, 0.0) * rate
        tax_paid += tx
        return d - tx

    def buy(a, i, d, t):
        px = P[i][assets.index(a)]
        u[a] += d / px
        basis[a] += d
        if a != "SPY":
            pm_lot[a] = t

    reb_years = set()
    for i, t in enumerate(idx):
        val = sum(u[a] * P[i][j] for j, a in enumerate(assets))
        eq.append((t, val))
        do = False
        if cadence == "annual" and t in ye:
            do = True
        elif (
            cadence == "biennial"
            and t in ye
            and t.year % 2 == 0
            and t.year not in reb_years
        ):
            do, _ = True, reb_years.add(t.year)
        elif cadence == "band20" and t in me:
            do = any(
                abs(u[a] * P[i][j] / val - weights[a]) > 0.20 * weights[a]
                for j, a in enumerate(assets)
                if weights[a] > 0
            )
        if do:
            for j, a in enumerate(assets):  # sells first
                cur, tgt = u[a] * P[i][j], weights[a] * val
                if cur > tgt + 1:
                    cash = sell(a, i, cur - tgt, t)
                    # distribute to underweights proportionally
                    unders = [
                        (b, weights[b] * val - u[b] * P[i][assets.index(b)])
                        for b in assets
                        if u[b] * P[i][assets.index(b)] < weights[b] * val - 1
                    ]
                    tot = sum(x for _, x in unders) or 1.0
                    for b, need in unders:
                        buy(b, i, cash * need / tot, t)

    last = len(idx) - 1
    t_end = idx[-1]
    net = sum(sell(a, last, u[a] * P[last][assets.index(a)], t_end) for a in assets)
    e = pd.Series(dict(eq))
    r = e.pct_change().dropna()
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net": round(net),
        "cagr": round(((net / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "tax": round(tax_paid),
        "st_hits": st_hits,
    }


def main() -> int:
    # ---- H-060 maximin sweep ----
    scenarios = {"x1.0": 1.0, "x0.5": 0.5, "x0.3": 0.3, "x0.0": 0.0}
    dfs = {k: load(v) for k, v in scenarios.items()}
    sweep = {}
    for gw in range(0, 55, 5):
        w = {"SPY": 1 - gw / 100, "GLD": gw / 100}
        nets = {}
        for k, d in dfs.items():
            nets[k] = simulate(d, w, cadence="annual")["net"]
        sweep[f"gold_{gw}pct"] = {**nets, "min": min(nets.values())}
        print(
            f"[H060] gold {gw:2d}%: "
            + " ".join(f"{k}={v:,}" for k, v in nets.items())
            + f" | min={min(nets.values()):,}",
            flush=True,
        )
    best = max(sweep, key=lambda k: sweep[k]["min"])
    print(
        f"[H060] MAXIMIN allocation: {best} (worst-case {sweep[best]['min']:,})",
        flush=True,
    )

    # silver split: 30% PM as 20 gold /10 silver vs 30 gold (common window 2006+)
    d_ag = load(1.0, with_silver=True)
    ag = {
        "70/30 gold only": simulate(d_ag, {"SPY": 0.7, "GLD": 0.3}, cadence="annual"),
        "70/20/10 gold+silver": simulate(
            d_ag, {"SPY": 0.7, "GLD": 0.2, "SLV": 0.1}, cadence="annual"
        ),
    }
    for k, v in ag.items():
        print(f"[H060-Ag] {k}: {v}", flush=True)

    # ---- H-061 cadence sweep (70/30, real gold) ----
    d = dfs["x1.0"]
    cad = {
        c: simulate(d, {"SPY": 0.7, "GLD": 0.3}, cadence=c)
        for c in ("never", "annual", "biennial", "band20")
    }
    for k, v in cad.items():
        print(f"[H061] {k}: {v}", flush=True)

    out = {
        "H060_sweep": sweep,
        "H060_maximin": best,
        "H060_silver": ag,
        "H061_cadence": cad,
    }
    (OUTD / "h060_h061_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
