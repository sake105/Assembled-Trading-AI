"""Welle 26 — H-057 Krisen-Rebalancing (historisch), H-058 Glide-Path (MC), H-059 Sparplan (MC).

Steuern: SPY 18,46 % auf realisierte Gewinne (Basis-Tracking), Gold §23 0 % (>1J; Lot-Alter wird
geprüft), Terminal-Liquidation. MC: Block-Bootstrap (60T) der Joint-Renditen SPY/GLD 2005–26,
1.000 Pfade, Seed 42.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
TAX_EQ, TAX_GOLD_LT, TAX_GOLD_ST = (
    0.1846,
    0.0,
    0.44,
)  # gold <1J: persönl. Satz (Annahme)


def load_pair() -> pd.DataFrame:
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].dropna()
    df.index = pd.DatetimeIndex(df.index)
    return df[df.index >= pd.Timestamp("2005-01-01", tz="UTC")]


# ---------------------------------------------------------------- H-057 historical
def crisis_rebalance(df: pd.DataFrame, *, revert_at_high: bool) -> dict:
    idx = df.index
    spy, gld = df["SPY"].values, df["GLD"].values
    u = {"SPY": 0.7 * START / spy[0], "GLD": 0.3 * START / gld[0]}
    basis = {"SPY": 0.7 * START, "GLD": 0.3 * START}
    gold_lot_date = idx[0]
    tax_paid = 0.0
    gold_st_hits = 0
    peak = spy[0]
    state = 0  # 0=normal(70/30) 1=dd20(80/20) 2=dd30(90/10)
    me = set(idx.to_series().groupby(idx.to_period("Y")).max())
    eq = []

    def sell(a, dollars, px, t):
        nonlocal tax_paid, gold_st_hits
        val = u[a] * px
        dollars = min(dollars, val)
        if dollars <= 0:
            return 0.0
        fr = dollars / val
        gain = dollars - fr * basis[a]
        u[a] *= 1 - fr
        basis[a] *= 1 - fr
        if a == "SPY":
            rate = TAX_EQ
        else:
            lt = (t - gold_lot_date).days >= 365
            rate = TAX_GOLD_LT if lt else TAX_GOLD_ST
            if not lt:
                gold_st_hits += 1
        tx = max(gain, 0.0) * rate
        tax_paid += tx
        return dollars - tx

    def buy(a, dollars, px, t):
        nonlocal gold_lot_date
        u[a] += dollars / px
        basis[a] += dollars
        if a == "GLD":
            gold_lot_date = t  # conservative: newest lot date governs (worst case)

    for i, t in enumerate(idx):
        ps, pg = spy[i], gld[i]
        peak = max(peak, ps)
        dd = ps / peak - 1
        val = u["SPY"] * ps + u["GLD"] * pg
        eq.append((t, val))
        tgt = None
        if state < 2 and dd <= -0.30:
            state, tgt = 2, 0.90
        elif state < 1 and dd <= -0.20:
            state, tgt = 1, 0.80
        elif state > 0 and ps >= peak:
            state = 0
            tgt = 0.70 if revert_at_high else None
        elif revert_at_high is False and t in me:
            tgt = 0.70 if state == 0 else None  # annual band only in normal state
        if tgt is not None:
            cur_eq = u["SPY"] * ps
            want = tgt * val
            if want > cur_eq + 1:
                cashd = sell("GLD", want - cur_eq, pg, t)
                buy("SPY", cashd, ps, t)
            elif want < cur_eq - 1:
                cashd = sell("SPY", cur_eq - want, ps, t)
                buy("GLD", cashd, pg, t)

    # terminal liquidation
    tl = idx[-1]
    net = sell("SPY", u["SPY"] * spy[-1], spy[-1], tl) + sell(
        "GLD", u["GLD"] * gld[-1], gld[-1], tl
    )
    e = pd.Series(dict(eq))
    r = e.pct_change().dropna()
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net_final": round(net),
        "cagr_net_pct": round(((net / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "tax_paid": round(tax_paid),
        "gold_shortterm_taxed_sales": gold_st_hits,
    }


def static_ref(df: pd.DataFrame, w_eq: float, *, annual_rebal: bool) -> dict:
    return crisis_rebalance_static(df, w_eq, annual_rebal)


def crisis_rebalance_static(df, w_eq, annual_rebal):
    idx = df.index
    spy, gld = df["SPY"].values, df["GLD"].values
    u = {"SPY": w_eq * START / spy[0], "GLD": (1 - w_eq) * START / gld[0]}
    basis = {"SPY": w_eq * START, "GLD": (1 - w_eq) * START}
    tax_paid = 0.0
    me = set(idx.to_series().groupby(idx.to_period("Y")).max())
    eq = []
    for i, t in enumerate(idx):
        ps, pg = spy[i], gld[i]
        val = u["SPY"] * ps + u["GLD"] * pg
        eq.append((t, val))
        if annual_rebal and t in me and (1 - w_eq) > 0:
            cur_eq = u["SPY"] * ps
            want = w_eq * val
            if abs(cur_eq - want) > val * 0.02:
                if want < cur_eq:
                    d = cur_eq - want
                    fr = d / cur_eq
                    gain = d - fr * basis["SPY"]
                    u["SPY"] *= 1 - fr
                    basis["SPY"] *= 1 - fr
                    tx = max(gain, 0) * TAX_EQ
                    tax_paid += tx
                    u["GLD"] += (d - tx) / pg
                    basis["GLD"] += d - tx
                else:
                    d = want - cur_eq
                    valg = u["GLD"] * pg
                    d = min(d, valg)
                    fr = d / valg if valg > 0 else 0
                    gain = d - fr * basis["GLD"]
                    u["GLD"] *= 1 - fr
                    basis["GLD"] *= 1 - fr
                    u["SPY"] += d / ps  # gold LT tax-free (held >1y in annual scheme)
                    basis["SPY"] += d
    ps, pg = spy[-1], gld[-1]
    gain_s = u["SPY"] * ps - basis["SPY"]
    gain_g = u["GLD"] * pg - basis["GLD"]
    net = u["SPY"] * ps - max(gain_s, 0) * TAX_EQ + u["GLD"] * pg  # gold LT 0 %
    tax_paid += max(gain_s, 0) * TAX_EQ
    e = pd.Series(dict(eq))
    r = e.pct_change().dropna()
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net_final": round(net),
        "cagr_net_pct": round(((net / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "tax_paid": round(tax_paid),
    }


# ---------------------------------------------------------------- MC engine
def bootstrap_paths(R: np.ndarray, n_paths: int, rng) -> list:
    T = len(R)
    out = []
    for _ in range(n_paths):
        idx_path = np.empty(T, dtype=np.int64)
        pos = 0
        while pos < T:
            si = rng.integers(0, T)
            blk = min(int(rng.geometric(1 / 60)), T - pos)
            idx_path[pos : pos + blk] = np.arange(si, si + blk) % T
            pos += blk
        out.append(R[idx_path])
    return out


def main() -> int:
    df = load_pair()
    print(f"[DATA] SPY/GLD {df.index[0].date()} -> {df.index[-1].date()}", flush=True)

    # ---- H-057 ----
    res57 = {
        "100% SPY": crisis_rebalance_static(df, 1.0, False),
        "static 70/30 annual": crisis_rebalance_static(df, 0.7, True),
        "crisis-rebal revert": crisis_rebalance(df, revert_at_high=True),
        "crisis-rebal noRevert": crisis_rebalance(df, revert_at_high=False),
    }
    for k, v in res57.items():
        print(f"[H057] {k}: {v}", flush=True)

    # ---- MC common ----
    R = df.pct_change().dropna().values
    rng = np.random.default_rng(42)
    paths = bootstrap_paths(R, 1000, rng)
    T = len(R)
    years = T / 252.0
    y_steps = int(years)

    # ---- H-058 glide path ----
    def run_glide(Rp, w0=0.9, w1=0.5):
        cum = np.cumprod(1 + Rp, axis=0)
        step = T // y_steps
        u_s = w0 * START
        u_g = (1 - w0) * START  # dollar positions scaled by cum relative to entry
        us = u_s / 1.0
        ug = u_g / 1.0
        bs, bg = u_s, u_g
        tax = 0.0
        prev_cs = prev_cg = 1.0
        for k in range(1, y_steps):
            i = k * step
            cs, cg = cum[i, 0], cum[i, 1]
            us *= cs / prev_cs
            ug *= cg / prev_cg
            prev_cs, prev_cg = cs, cg
            w_tgt = w0 + (w1 - w0) * k / y_steps
            tot = us + ug
            want = w_tgt * tot
            if want < us - 1:  # sell SPY -> gold
                d = us - want
                fr = d / us
                gain = d - fr * bs
                tx = max(gain, 0) * TAX_EQ
                tax += tx
                us -= d
                bs *= 1 - fr
                ug += d - tx
                bg += d - tx
        cs, cg = cum[-1, 0], cum[-1, 1]
        us *= cs / prev_cs
        ug *= cg / prev_cg
        net = us - max(us - bs, 0) * TAX_EQ + ug  # gold LT 0 %
        return net

    def run_static_mc(Rp, w_eq):
        cum = np.cumprod(1 + Rp, axis=0)
        vs = w_eq * START * cum[-1, 0]
        vg = (1 - w_eq) * START * cum[-1, 1]
        return vs - max(vs - w_eq * START, 0) * TAX_EQ + vg

    g_nets, s73_nets, s100_nets = [], [], []
    for Rp in paths:
        g_nets.append(run_glide(Rp))
        s73_nets.append(run_static_mc(Rp, 0.7))
        s100_nets.append(run_static_mc(Rp, 1.0))
    g, s73, s100 = map(np.array, (g_nets, s73_nets, s100_nets))
    res58 = {
        "glide 90->50": {
            "median": round(float(np.median(g))),
            "q05": round(float(np.quantile(g, 0.05))),
            "P_beat_SPY_pct": round(float((g > s100).mean() * 100), 1),
        },
        "static 70/30": {
            "median": round(float(np.median(s73))),
            "q05": round(float(np.quantile(s73, 0.05))),
            "P_beat_SPY_pct": round(float((s73 > s100).mean() * 100), 1),
        },
        "100% SPY": {
            "median": round(float(np.median(s100))),
            "q05": round(float(np.quantile(s100, 0.05))),
        },
    }
    print("[H058]", json.dumps(res58), flush=True)

    # ---- H-059 Sparplan (1000 EUR/Monat, kein Startkapital) ----
    CONTRIB = 1000.0
    step_m = 21  # ~Monat in Handelstagen

    def run_dca(Rp, w_eq):
        cum = np.cumprod(1 + Rp, axis=0)
        us = ug = bs = bg = 0.0
        for i in range(0, T, step_m):
            us += w_eq * CONTRIB / cum[i, 0]
            ug += (1 - w_eq) * CONTRIB / cum[i, 1]
            bs += w_eq * CONTRIB
            bg += (1 - w_eq) * CONTRIB
        vs, vg = us * cum[-1, 0], ug * cum[-1, 1]
        return vs - max(vs - bs, 0) * TAX_EQ + vg  # gold LT 0 %

    d100, d73 = [], []
    for Rp in paths:
        d100.append(run_dca(Rp, 1.0))
        d73.append(run_dca(Rp, 0.7))
    d100, d73 = np.array(d100), np.array(d73)
    contributed = CONTRIB * len(range(0, T, step_m))
    res59 = {
        "contributed_total": round(contributed),
        "DCA 100% SPY": {
            "median": round(float(np.median(d100))),
            "q05": round(float(np.quantile(d100, 0.05))),
        },
        "DCA 70/30": {
            "median": round(float(np.median(d73))),
            "q05": round(float(np.quantile(d73, 0.05))),
            "P_beat_SPY_pct": round(float((d73 > d100).mean() * 100), 1),
        },
    }
    print("[H059]", json.dumps(res59), flush=True)

    out = {
        "H057_crisis_rebal": res57,
        "H058_glide_MC": res58,
        "H059_sparplan_MC": res59,
        "mc": {"n_paths": 1000, "seed": 42, "years": round(years, 1)},
    }
    (OUTD / "h057_h059_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
