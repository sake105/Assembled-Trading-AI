"""Welle 30 — H-067 BTC-Sizing (Kelly unter Haircut-Szenarien), H-068 Krisen-Replay EUR."""

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
R_F = 0.02


def load_all():
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    w = pd.read_parquet(DATA / "prices_w28.parquet")
    fx = w[w["symbol"] == "EURUSD"].set_index("timestamp")["close"]
    df["EURUSD"] = (
        pd.Series(fx.values, index=pd.DatetimeIndex(fx.index)).reindex(df.index).ffill()
    )
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    df["BTC"] = (
        pd.Series(btc.values, index=pd.DatetimeIndex(btc.index))
        .reindex(df.index)
        .ffill()
    )
    return df


# ---------------- H-067 ----------------
def kelly_analysis(df):
    b = df["BTC"].dropna()
    b = b[b.index >= pd.Timestamp("2016-01-01", tz="UTC")]
    r = b.pct_change().dropna()
    mu_ann = (
        float(r.mean()) * 365
    )  # crypto trades daily incl weekends in source? index is trading days -> use 252
    mu_ann = float(r.mean()) * 252
    sig_ann = float(r.std()) * sqrt(252)
    out = {
        "observed": {
            "mu_ann_pct": round(mu_ann * 100, 1),
            "sigma_ann_pct": round(sig_ann * 100, 1),
        }
    }
    rows = {}
    for hc in (1.0, 0.5, 0.25, 0.10, 0.0):
        mu = mu_ann * hc
        f_kelly = (mu - R_F) / (sig_ann**2) if sig_ann > 0 else 0.0
        rows[f"x{hc}"] = {
            "mu_pct": round(mu * 100, 1),
            "kelly_pct": round(max(f_kelly, 0) * 100, 1),
            "half_kelly_pct": round(max(f_kelly, 0) * 50, 1),
        }
    out["kelly_by_scenario"] = rows

    # MC: 70/30 base + BTC epsilon under scenarios (bootstrap SPY/GLD/BTC joint, 500 paths)
    j = df[["SPY", "GLD", "BTC"]].dropna()
    j = j[j.index >= pd.Timestamp("2016-01-01", tz="UTC")]
    Rj = j.pct_change().dropna().values
    T = len(Rj)
    rng = np.random.default_rng(42)
    res_mc = {}
    for hc in (1.0, 0.25, 0.0):
        for eps in (0.0, 0.02, 0.05, 0.10):
            nets = []
            for _ in range(500):
                idx = np.empty(T, dtype=np.int64)
                pos = 0
                while pos < T:
                    si = rng.integers(0, T)
                    blk = min(int(rng.geometric(1 / 60)), T - pos)
                    idx[pos : pos + blk] = np.arange(si, si + blk) % T
                    pos += blk
                Rp = Rj[idx].copy()
                Rp[:, 2] *= hc  # haircut BTC returns
                cum = np.cumprod(1 + Rp, axis=0)
                w_eq = 0.7 * (1 - eps)
                w_g = 0.3 * (1 - eps)
                vs = w_eq * START * cum[-1, 0]
                vg = w_g * START * cum[-1, 1]
                vb = eps * START * cum[-1, 2]
                net = (
                    vs - max(vs - w_eq * START, 0) * TAX_EQ + vg + vb
                )  # gold+BTC §23 0 %
                nets.append(net)
            nets = np.array(nets)
            res_mc[f"hc{hc}_btc{int(eps * 100)}pct"] = {
                "median": round(float(np.median(nets))),
                "q05": round(float(np.quantile(nets, 0.05))),
            }
    out["mc_10y_window"] = res_mc
    return out


# ---------------- H-068 ----------------
def crisis_replay(df):
    d = df[["SPY", "GLD", "EURUSD"]].dropna()
    spy_eur = d["SPY"] / d["EURUSD"]
    gld_eur = d["GLD"] / d["EURUSD"]
    crises = {
        "GFC_2007-2009": ("2007-10-09", "2010-12-31"),
        "COVID_2020": ("2020-02-19", "2021-12-31"),
        "Inflation_2022": ("2022-01-03", "2024-12-31"),
    }
    out = {}
    for name, (start, end) in crises.items():
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        w = d[(d.index >= s) & (d.index <= e)]
        if not len(w):
            continue
        res = {}
        for label, w_eq in (("100% SPY", 1.0), ("85/15", 0.85), ("70/30", 0.7)):
            port = (
                w_eq * spy_eur[w.index] / spy_eur[w.index][0]
                + (1 - w_eq) * gld_eur[w.index] / gld_eur[w.index][0]
            )
            trough = float(port.min())
            dd = trough - 1.0
            rec = port[port.index > port.idxmin()]
            rec_days = None
            back = rec[rec >= 1.0]
            if len(back):
                rec_days = int((back.index[0] - w.index[0]).days)
            res[label] = {
                "maxdd_eur_pct": round(dd * 100, 1),
                "recovered_after_days": rec_days,
            }
        out[name] = res
    return out


def main() -> int:
    df = load_all()
    r67 = kelly_analysis(df)
    print("[H067]", json.dumps(r67, indent=2), flush=True)
    r68 = crisis_replay(df)
    print("[H068]", json.dumps(r68, indent=2), flush=True)
    (OUTD / "h067_h068_results.json").write_text(
        json.dumps({"H067": r67, "H068": r68}, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
