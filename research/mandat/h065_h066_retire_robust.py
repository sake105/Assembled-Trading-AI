"""Welle 29 — H-065 Entnahmephase (SWR unter dt. Steuer, MC), H-066 Rolling-Start-Robustheit.

H-065: MC-Bootstrap SPY/GLD (Block 60T, Seed 42), 500k Start, jährl. nominale Entnahme,
anteilige Steuer auf Gewinnanteil (Basis-Tracking), Gold §23 0 %. Ruin-P + Median-Rest.
H-066: alle Monats-Starts 2005–2016, 10-J-Fenster, 70/30 (2J-Rebal) vs 100 % SPY netto.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
TAX_EQ = 0.1846


def load_pair():
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].dropna()
    df.index = pd.DatetimeIndex(df.index)
    return df[df.index >= pd.Timestamp("2005-01-01", tz="UTC")]


def bootstrap_paths(R, n, rng):
    T = len(R)
    out = []
    for _ in range(n):
        idx = np.empty(T, dtype=np.int64)
        pos = 0
        while pos < T:
            si = rng.integers(0, T)
            blk = min(int(rng.geometric(1 / 60)), T - pos)
            idx[pos : pos + blk] = np.arange(si, si + blk) % T
            pos += blk
        out.append(R[idx])
    return out


# ---------------- H-065 ----------------
def run_withdraw(Rp, w_eq, wd_annual, start_cap=500_000.0):
    cum = np.cumprod(1 + Rp, axis=0)
    T = len(Rp)
    us = w_eq * start_cap
    ug = (1 - w_eq) * start_cap  # dollar sleeves, evolve with cum ratios
    bs, bg = us, ug
    prev = np.array([1.0, 1.0])
    for i in range(252, T, 252):  # annual withdrawal
        cs, cg = cum[i]
        us *= cs / prev[0]
        ug *= cg / prev[1]
        prev = cum[i]
        tot = us + ug
        if tot <= 0:
            return 0.0, True
        need = wd_annual
        # withdraw proportional; tax on gain fraction of the sold slice
        for sleeve, b, tax in (("s", bs, TAX_EQ), ("g", bg, 0.0)):
            v = us if sleeve == "s" else ug
            take = need * (v / tot)
            if v <= 0:
                continue
            gain_frac = max(1 - b / v, 0.0)
            gross = take / (1 - gain_frac * tax) if gain_frac * tax < 1 else take
            gross = min(gross, v)
            fb = gross / v
            if sleeve == "s":
                us -= gross
                bs *= 1 - fb
            else:
                ug -= gross
                bg *= 1 - fb
        if us + ug <= 0:
            return 0.0, True
    cs, cg = cum[-1]
    us *= cs / prev[0]
    ug *= cg / prev[1]
    return us + ug, False


# ---------------- H-066 ----------------
def window_run(df, start, years=10, w_eq=0.7):
    end = start + pd.DateOffset(years=years)
    d = df[(df.index >= start) & (df.index <= end)]
    if (d.index[-1] - d.index[0]).days < years * 360:
        return None
    spy, gld = d["SPY"], d["GLD"]
    # 100% SPY
    net_spy = 100_000 * (1 + (spy.iloc[-1] / spy.iloc[0] - 1) * (1 - TAX_EQ))
    # 70/30 biennial rebal with basis tracking
    us, ug = 70_000.0, 30_000.0
    bs, bg = us, ug
    tax = 0.0
    prev_s, prev_g = spy.iloc[0], gld.iloc[0]
    ye = d.index.to_series().groupby(d.index.to_period("Y")).max()
    reb_dates = [t for i, t in enumerate(sorted(ye)) if i % 2 == 1]
    for t in reb_dates:
        cs, cg = spy.loc[t], gld.loc[t]
        us *= cs / prev_s
        ug *= cg / prev_g
        prev_s, prev_g = cs, cg
        tot = us + ug
        tgt_s = 0.7 * tot
        if us > tgt_s + 1:
            dlt = us - tgt_s
            gain = dlt * max(1 - bs / us, 0)
            tx = gain * TAX_EQ
            tax += tx
            bs *= 1 - dlt / us
            us -= dlt
            ug += dlt - tx
            bg += dlt - tx
        elif us < tgt_s - 1:
            dlt = tgt_s - us
            gain = dlt * max(1 - bg / ug, 0)
            us += dlt  # gold LT 0 % (biennial > 1J)
            bs += dlt
            bg *= 1 - dlt / ug
            ug -= dlt
    us *= spy.iloc[-1] / prev_s
    ug *= gld.iloc[-1] / prev_g
    net_blend = us - max(us - bs, 0) * TAX_EQ + ug
    return net_spy, net_blend


def main() -> int:
    df = load_pair()
    R = df.pct_change().dropna().values
    rng = np.random.default_rng(42)
    paths = bootstrap_paths(R, 1000, rng)

    res65 = {}
    for wd_pct in (3.0, 3.5, 4.0, 5.0):
        wd = 500_000 * wd_pct / 100
        for name, w_eq in (("100% SPY", 1.0), ("70/30", 0.7)):
            finals, ruins = [], 0
            for Rp in paths:
                f, ruin = run_withdraw(Rp, w_eq, wd)
                finals.append(f)
                ruins += ruin
            res65[f"{wd_pct}%_{name}"] = {
                "ruin_pct": round(ruins / 10, 1),
                "median_rest": round(float(np.median(finals))),
                "q05_rest": round(float(np.quantile(finals, 0.05))),
            }
            print(
                f"[H065] {wd_pct}% {name}: ruin={ruins / 10:.1f}% median_rest={np.median(finals):,.0f}",
                flush=True,
            )

    starts = pd.date_range("2005-01-01", "2016-07-01", freq="MS", tz="UTC")
    wins, tot, rows = 0, 0, []
    for st in starts:
        r = window_run(df, st)
        if r is None:
            continue
        tot += 1
        net_spy, net_blend = r
        if net_blend >= net_spy:
            wins += 1
        rows.append((str(st.date()), round(net_spy), round(net_blend)))
    res66 = {
        "windows": tot,
        "blend_wins": wins,
        "win_rate_pct": round(100 * wins / tot, 1),
        "first": rows[0],
        "last": rows[-1],
    }
    print(
        f"[H066] 70/30 >= SPY in {wins}/{tot} 10J-Fenstern ({100 * wins / tot:.0f}%)",
        flush=True,
    )

    out = {
        "H065_withdrawal": res65,
        "H066_rolling": res66,
        "mc": {"paths": 1000, "seed": 42, "start_capital": 500000},
    }
    (OUTD / "h065_h066_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
