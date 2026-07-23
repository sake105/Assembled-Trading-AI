"""H-069 — Cash-Flow-Rebalancing: Null-Steuer-Rebalancing im Sparplan (MC).

(a) fixe 70/30-Raten ohne Rebal; (b) Rate -> untergewichteter Sleeve (nie verkaufen);
(c) fixe Raten + 2-J-Verkaufs-Rebal (Steuer auf SPY-Gewinnanteil). 1.000 €/Monat, 21,5 J,
1.000 Pfade, Seed 42. Terminal: SPY-Gewinn 18,46 %, Gold 0 %.
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
CONTRIB = 1000.0
W_EQ = 0.7


def load_R():
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].dropna()
    df.index = pd.DatetimeIndex(df.index)
    df = df[df.index >= pd.Timestamp("2005-01-01", tz="UTC")]
    return df.pct_change().dropna().values


def run(Rp, mode):
    cum = np.cumprod(1 + Rp, axis=0)
    T = len(Rp)
    us = ug = bs = bg = 0.0  # dollar sleeves + basis
    tax = 0.0
    prev = np.array([1.0, 1.0])
    vals = []
    for i in range(0, T, 21):
        cs, cg = cum[i]
        us *= cs / prev[0]
        ug *= cg / prev[1]
        prev = cum[i]
        tot = us + ug
        if mode == "fixed" or mode == "sellrebal" or tot <= 0:
            a_s, a_g = W_EQ * CONTRIB, (1 - W_EQ) * CONTRIB
        else:  # cashflow: alles in den untergewichteten Sleeve (bis Ziel erreicht)
            gap_s = W_EQ * (tot + CONTRIB) - us
            a_s = min(max(gap_s, 0.0), CONTRIB)
            a_g = CONTRIB - a_s
        us += a_s
        bs += a_s
        ug += a_g
        bg += a_g
        # biennial sell-rebalance for (c)
        if mode == "sellrebal" and i > 0 and (i // 21) % 24 == 0:
            tot = us + ug
            tgt = W_EQ * tot
            if us > tgt + 1:
                d = us - tgt
                gain = d * max(1 - bs / us, 0)
                t = gain * TAX_EQ
                tax += t
                bs *= 1 - d / us
                us -= d
                ug += d - t
                bg += d - t
            elif us < tgt - 1:
                d = tgt - us
                d = min(d, ug)
                bg *= 1 - d / ug if ug > 0 else 1
                ug -= d
                us += d  # gold LT steuerfrei
                bs += d
        vals.append(us + ug)
    cs, cg = cum[-1]
    us *= cs / prev[0]
    ug *= cg / prev[1]
    net = us - max(us - bs, 0) * TAX_EQ + ug
    v = np.array(vals)
    peak = np.maximum.accumulate(v)
    dd = float(((v - peak) / np.where(peak > 0, peak, 1)).min())
    w_end = us / (us + ug) if us + ug > 0 else np.nan
    return net, tax, dd, w_end


def main() -> int:
    R = load_R()
    T = len(R)
    rng = np.random.default_rng(42)
    modes = ("fixed", "cashflow", "sellrebal")
    res = {m: {"net": [], "tax": [], "dd": [], "w": []} for m in modes}
    for _ in range(1000):
        idx = np.empty(T, dtype=np.int64)
        pos = 0
        while pos < T:
            si = rng.integers(0, T)
            blk = min(int(rng.geometric(1 / 60)), T - pos)
            idx[pos : pos + blk] = np.arange(si, si + blk) % T
            pos += blk
        Rp = R[idx]
        for m in modes:
            net, tax, dd, w = run(Rp, m)
            res[m]["net"].append(net)
            res[m]["tax"].append(tax)
            res[m]["dd"].append(dd)
            res[m]["w"].append(w)
    out = {}
    for m in modes:
        n = np.array(res[m]["net"])
        out[m] = {
            "median_net": round(float(np.median(n))),
            "q05_net": round(float(np.quantile(n, 0.05))),
            "median_dd": round(float(np.median(res[m]["dd"])), 3),
            "median_tax": round(float(np.median(res[m]["tax"]))),
            "median_end_weight_eq": round(float(np.nanmedian(res[m]["w"])), 3),
        }
        print(f"[H069] {m}: {out[m]}", flush=True)
    (OUTD / "h069_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
