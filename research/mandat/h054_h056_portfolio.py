"""Welle 25 — Portfolio-Konstruktion & Risiko (H-054 Risk-Parity, H-055 Vol-Target, H-056 Monte Carlo).

Steuern per Sleeve: Aktien-ETF 18,46 %, Gold/BTC §23 0 % (>1J), Bond-ETF 26,375 %, Cash 26,375 %
auf Zins. Per-Sleeve-Basis, End-Liquidation. Kein Hebel (Weights summieren <= 1; Rest Cash 2 %).
H-056: stationärer Block-Bootstrap (E[Block]=60T) der Joint-Renditen, 1.000 Pfade, Seed 42.
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
TAX = {"SPY": 0.1846, "EFA": 0.1846, "GLD": 0.0, "TLT": 0.26375, "BTC": 0.0}
CASH_RATE = 0.02
CASH_TAX = 0.26375


def load() -> pd.DataFrame:
    we = pd.read_parquet(DATA / "prices_world_etf.parquet")
    px = we.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "EFA"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    gld = pc[pc["symbol"] == "GLD"].set_index("timestamp")["close"]
    df["GLD"] = pd.Series(gld.values, index=pd.DatetimeIndex(gld.index)).reindex(
        df.index
    )
    tlt = pd.read_parquet(DATA / "bond_TLT.parquet").set_index("timestamp")["close"]
    df["TLT"] = pd.Series(tlt.values, index=pd.DatetimeIndex(tlt.index)).reindex(
        df.index
    )
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    df["BTC"] = (
        pd.Series(btc.values, index=pd.DatetimeIndex(btc.index))
        .reindex(df.index)
        .ffill()
    )
    return df


def simulate_dynamic(
    df: pd.DataFrame,
    assets: list,
    weight_fn,
    start: str,
    *,
    drift_band: float | None = None,
) -> dict:
    """Monthly target weights from weight_fn(t) (sum<=1, rest cash). Per-sleeve tax."""
    sub = df[df.index >= pd.Timestamp(start, tz="UTC")].dropna(subset=assets)
    idx = sub.index
    units = {a: 0.0 for a in assets}
    basis = {a: 0.0 for a in assets}
    cash, tax_paid = START, 0.0

    def buy(a, d, px):
        nonlocal cash
        d = min(d, cash)
        if d <= 0:
            return
        units[a] += d / px
        basis[a] += d
        cash -= d

    def sell(a, d, px):
        nonlocal cash, tax_paid
        val = units[a] * px
        if val <= 0 or d <= 0:
            return
        d = min(d, val)
        fr = d / val
        gain = d - fr * basis[a]
        units[a] *= 1 - fr
        basis[a] *= 1 - fr
        t = max(gain, 0.0) * TAX[a]
        tax_paid += t
        cash += d - t

    me = set(idx.to_series().groupby(idx.to_period("M")).max())
    cur_w = None
    eq = []
    for t in idx:
        p = sub.loc[t]
        # daily cash interest, taxed
        gross_int = cash * CASH_RATE / 252
        cash += gross_int * (1 - CASH_TAX)
        tax_paid += gross_int * CASH_TAX
        val = cash + sum(units[a] * float(p[a]) for a in assets)
        eq.append((t, val))
        if t not in me:
            continue
        w = weight_fn(t)
        if w is None:
            continue
        if (
            drift_band is not None
            and cur_w is not None
            and all(
                abs((units[a] * float(p[a])) / val - w.get(a, 0.0))
                <= drift_band * max(w.get(a, 1e-9), 1e-9)
                for a in assets
            )
        ):
            continue
        cur_w = w
        for a in assets:  # sells first
            cur = units[a] * float(p[a])
            tgt = w.get(a, 0.0) * val
            if cur > tgt + 1:
                sell(a, cur - tgt, float(p[a]))
        for a in assets:
            cur = units[a] * float(p[a])
            tgt = w.get(a, 0.0) * val
            if cur < tgt - 1:
                buy(a, tgt - cur, float(p[a]))

    pend = sub.iloc[-1]
    for a in assets:
        if units[a] > 0:
            sell(a, units[a] * float(pend[a]), float(pend[a]))
    e = pd.Series(dict(eq))
    r = e.pct_change().dropna()
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net_final": round(cash),
        "cagr_net_pct": round(((cash / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "tax_paid": round(tax_paid),
        "years": round(years, 1),
    }


def main() -> int:
    df = load()
    s = "2005-01-01"
    rets = df.pct_change()
    vol60 = rets.rolling(60).std()

    def iv_weights(assets):
        def fn(t):
            v = vol60.loc[:t].iloc[-1][assets]
            if v.isna().any() or (v <= 0).any():
                return None
            iv = 1.0 / v
            return dict(iv / iv.sum())

        return fn

    def static_w(w):
        return lambda t: w

    out = {
        "H054_riskparity": {
            "100% SPY (ref)": simulate_dynamic(df, ["SPY"], static_w({"SPY": 1.0}), s),
            "RP iv(SPY,EFA,GLD,TLT) monthly": simulate_dynamic(
                df,
                ["SPY", "EFA", "GLD", "TLT"],
                iv_weights(["SPY", "EFA", "GLD", "TLT"]),
                s,
            ),
            "RP iv +band20": simulate_dynamic(
                df,
                ["SPY", "EFA", "GLD", "TLT"],
                iv_weights(["SPY", "EFA", "GLD", "TLT"]),
                s,
                drift_band=0.20,
            ),
            "RP iv(SPY,GLD,TLT,BTC) 2016+": simulate_dynamic(
                df,
                ["SPY", "GLD", "TLT", "BTC"],
                iv_weights(["SPY", "GLD", "TLT", "BTC"]),
                "2016-01-01",
            ),
        }
    }
    for k, v in out["H054_riskparity"].items():
        print(f"[H054] {k}: {v}", flush=True)

    # H-055 vol targeting on 85/15 SPY/GLD
    base = {"SPY": 0.85, "GLD": 0.15}
    port_r = rets["SPY"] * 0.85 + rets["GLD"] * 0.15
    pvol = port_r.rolling(20).std() * sqrt(252)

    def vt_weights(target):
        def fn(t):
            v = pvol.loc[:t].iloc[-1]
            if not np.isfinite(v) or v <= 0:
                return None
            e = min(1.0, target / v)
            return {a: w * e for a, w in base.items()}

        return fn

    out["H055_voltarget"] = {
        "85/15 (ref, no VT)": simulate_dynamic(
            df, ["SPY", "GLD"], static_w(base), s, drift_band=0.20
        ),
        "VT 10%": simulate_dynamic(df, ["SPY", "GLD"], vt_weights(0.10), s),
        "VT 15%": simulate_dynamic(df, ["SPY", "GLD"], vt_weights(0.15), s),
    }
    for k, v in out["H055_voltarget"].items():
        print(f"[H055] {k}: {v}", flush=True)

    # ---- H-056 Monte Carlo (stationary block bootstrap, static allocations) ----
    assets = ["SPY", "EFA", "GLD", "TLT"]
    R = df[assets].pct_change().dropna()
    R = R[R.index >= pd.Timestamp(s, tz="UTC")].values
    T = len(R)
    rng = np.random.default_rng(42)
    allocs = {
        "100% SPY": {"SPY": 1.0},
        "70/30 SPY/GLD": {"SPY": 0.7, "GLD": 0.3},
        "60/40 SPY/TLT": {"SPY": 0.6, "TLT": 0.4},
        "50/30/20 SPY/GLD/TLT": {"SPY": 0.5, "GLD": 0.3, "TLT": 0.2},
        "40/25/20/15 SPY/EFA/GLD/TLT": {
            "SPY": 0.4,
            "EFA": 0.25,
            "GLD": 0.2,
            "TLT": 0.15,
        },
    }
    N_PATH = 1000
    p_block = 1.0 / 60.0
    years = T / 252.0
    stats = {k: {"net": [], "maxdd": []} for k in allocs}
    spy_nets = []
    ai = {a: i for i, a in enumerate(assets)}
    for _ in range(N_PATH):
        idx_path = np.empty(T, dtype=np.int64)
        pos = 0
        while pos < T:
            start_i = rng.integers(0, T)
            blk = min(int(rng.geometric(p_block)), T - pos)
            seg = np.arange(start_i, start_i + blk) % T
            idx_path[pos : pos + blk] = seg
            pos += blk
        Rp = R[idx_path]
        cum = np.cumprod(1 + Rp, axis=0)
        for name, w in allocs.items():
            path = sum(w[a] * cum[:, ai[a]] for a in w)
            terminal = float(path[-1])
            net = START
            for a, wa in w.items():
                g = wa * START * (float(cum[-1, ai[a]]) - 1.0)
                net += g * (1 - TAX[a]) if g > 0 else g
            dd = float((path / np.maximum.accumulate(path) - 1).min())
            stats[name]["net"].append(net)
            stats[name]["maxdd"].append(dd)
        spy_nets.append(stats["100% SPY"]["net"][-1])
    spy_nets = np.array(spy_nets)
    mc = {}
    for name in allocs:
        nets = np.array(stats[name]["net"])
        dds = np.array(stats[name]["maxdd"])
        mc[name] = {
            "median_net": round(float(np.median(nets))),
            "q05_net": round(float(np.quantile(nets, 0.05))),
            "q95_net": round(float(np.quantile(nets, 0.95))),
            "P_beat_SPY_pct": round(float((nets > spy_nets).mean() * 100), 1),
            "median_maxdd": round(float(np.median(dds)), 3),
            "median_cagr_net_pct": round(
                ((float(np.median(nets)) / START) ** (1 / years) - 1) * 100, 2
            ),
        }
        print(f"[H056] {name}: {mc[name]}", flush=True)
    out["H056_montecarlo"] = {
        "n_paths": N_PATH,
        "years_per_path": round(years, 1),
        "block_mean_days": 60,
        "seed": 42,
        "allocs": mc,
    }

    (OUTD / "h054_h056_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
