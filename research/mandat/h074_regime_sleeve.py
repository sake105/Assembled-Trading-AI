"""H-074 — VIX-Regime-Konditionierung der Gold-Quote (mildester Timing-Fall).

SPY/GLD 2005–26. Gold-Quote: statisch 25 % vs regime {15 % normal, 35 % wenn VIX>rolling-P80}.
Anpassung NUR im 2-J-Rebal-Raster (steuer-schonend, §23-Uhr sicher). Historisch + Robustheit
(Gold×0,5). Vergleich netto (End-Liq) + Sharpe/DD.
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
TAX_EQ = 0.1846


def load():
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    w = pd.read_parquet(DATA / "prices_w28.parquet")
    vix = w[w["symbol"] == "VIX"].set_index("timestamp")["close"]
    df["VIX"] = (
        pd.Series(vix.values, index=pd.DatetimeIndex(vix.index))
        .reindex(df.index)
        .ffill()
    )
    df = df.dropna()
    return df[df.index >= pd.Timestamp("2005-01-01", tz="UTC")]


def simulate(df, gold_w_fn, *, scale_gold=1.0):
    d = df.copy()
    if scale_gold != 1.0:
        r = d["GLD"].pct_change().fillna(0.0) * scale_gold
        d["GLD"] = float(d["GLD"].iloc[0]) * (1 + r).cumprod()
    idx = d.index
    spy, gld = d["SPY"].values, d["GLD"].values
    w0 = gold_w_fn(idx[0], d)
    us, ug = (1 - w0) * START, w0 * START
    bs, bg = us, ug
    tax = 0.0
    prev = (spy[0], gld[0])
    ye = sorted(idx.to_series().groupby(idx.to_period("Y")).max())
    reb = set(t for i, t in enumerate(ye) if i % 2 == 1)
    eq = []
    for i, t in enumerate(idx):
        us *= spy[i] / prev[0]
        ug *= gld[i] / prev[1]
        prev = (spy[i], gld[i])
        val = us + ug
        eq.append(val)
        if t in reb:
            gw = gold_w_fn(t, d)
            tgt_g = gw * val
            if ug > tgt_g + 1:  # sell gold (LT, 0 %)
                dlt = ug - tgt_g
                bg *= 1 - dlt / ug
                ug -= dlt
                us += dlt
                bs += dlt
            elif ug < tgt_g - 1:  # sell SPY (18,46 % auf Gewinnanteil)
                dlt = tgt_g - ug
                gain = dlt * max(1 - bs / us, 0)
                tx = gain * TAX_EQ
                tax += tx
                bs *= 1 - dlt / us
                us -= dlt
                ug += dlt - tx
                bg += dlt - tx
    net = us - max(us - bs, 0) * TAX_EQ + ug
    e = pd.Series(eq, index=idx)
    r = e.pct_change().dropna()
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net": round(net),
        "cagr": round(((net / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "tax": round(tax),
    }


def main() -> int:
    df = load()
    vix_p80 = df["VIX"].rolling(756, min_periods=252).quantile(0.8)

    def static25(t, d):
        return 0.25

    def regime(t, d):
        v = d["VIX"].loc[:t].iloc[-1]
        p = vix_p80.loc[:t].iloc[-1]
        if not np.isfinite(p):
            return 0.25
        return 0.35 if v > p else 0.15

    out = {}
    for lab, sg in (("gold_x1.0", 1.0), ("gold_x0.5", 0.5)):
        out[lab] = {
            "static_25": simulate(df, static25, scale_gold=sg),
            "regime_15_35_VIX": simulate(df, regime, scale_gold=sg),
        }
        print(
            f"[H074] {lab}: static={out[lab]['static_25']} | regime={out[lab]['regime_15_35_VIX']}",
            flush=True,
        )
    (OUTD / "h074_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
