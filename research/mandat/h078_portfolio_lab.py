"""H-078 — PORTFOLIO-LABOR (Welle 39b): >=50 Konstruktionen × echtes Monte Carlo.

8 Assets (SPY/EFA/EEM/GLD/SLV/TLT/BTC/ETH), Joint-Block-Bootstrap (60T, Seed 42, 1.000 Pfade),
Fenster 2017+ (ETH-Constraint; Caveat: Krypto-Bull eingebacken -> zusätzlich BTC/ETH-Haircut ×0,25).
Terminal-Steuer je Sleeve: Aktien-ETF 18,46 %, TLT 26,375 %, Gold/Silber/BTC/ETH §23 0 % (>1J).
Metriken: Median, Floor (q05), Median-MaxDD, P(schlägt 100 % SPY).
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
START = 100_000.0
TAX = {
    "SPY": 0.1846,
    "EFA": 0.1846,
    "EEM": 0.1846,
    "TLT": 0.26375,
    "GLD": 0.0,
    "SLV": 0.0,
    "BTC": 0.0,
    "ETH": 0.0,
}
ASSETS = ["SPY", "EFA", "EEM", "GLD", "SLV", "TLT", "BTC", "ETH"]


def load():
    w = pd.read_parquet(DATA / "prices_world_etf.parquet")
    px = w.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "EFA", "EEM"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    gld = pc[pc["symbol"] == "GLD"].set_index("timestamp")["close"]
    df["GLD"] = pd.Series(gld.values, index=pd.DatetimeIndex(gld.index)).reindex(
        df.index
    )
    for name, f in (("SLV", "bond_SLV.parquet"), ("TLT", "bond_TLT.parquet")):
        s = pd.read_parquet(DATA / f).set_index("timestamp")["close"]
        df[name] = pd.Series(s.values, index=pd.DatetimeIndex(s.index)).reindex(
            df.index
        )
    for name, f in (
        ("BTC", "crypto_BTCUSDCC.parquet"),
        ("ETH", "crypto_ETHUSDCC.parquet"),
    ):
        s = pd.read_parquet(DATA / f)["close"]
        df[name] = (
            pd.Series(s.values, index=pd.DatetimeIndex(s.index))
            .reindex(df.index)
            .ffill()
        )
    df = df.dropna()
    return df[df.index >= pd.Timestamp("2017-01-01", tz="UTC")]


def constructions(df) -> dict:
    C = {}
    for wspy in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        rest = 1 - wspy
        if rest == 0:
            C["100 SPY"] = {"SPY": 1.0}
            continue
        for gname, (gg, bb) in (
            ("gold", (1.0, 0.0)),
            ("g75b25", (0.75, 0.25)),
            ("g50b50", (0.5, 0.5)),
        ):
            C[f"{int(wspy * 100)}spy_{gname}"] = {
                "SPY": wspy,
                "GLD": rest * gg,
                "BTC": rest * bb,
            }
    for name, w in {
        "60/20/20 spy/efa/gld": {"SPY": 0.6, "EFA": 0.2, "GLD": 0.2},
        "50/25/25 spy/efa/gld": {"SPY": 0.5, "EFA": 0.25, "GLD": 0.25},
        "40/30/30 spy/efa/gld": {"SPY": 0.4, "EFA": 0.3, "GLD": 0.3},
        "60/40 spy/tlt": {"SPY": 0.6, "TLT": 0.4},
        "50/30/20 spy/tlt/gld": {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2},
        "40/40/20 spy/tlt/gld": {"SPY": 0.4, "TLT": 0.4, "GLD": 0.2},
        "70/20/10 spy/gld/slv": {"SPY": 0.7, "GLD": 0.2, "SLV": 0.1},
        "70/15/15 spy/gld/slv": {"SPY": 0.7, "GLD": 0.15, "SLV": 0.15},
        "70/25/5 spy/gld/eth": {"SPY": 0.7, "GLD": 0.25, "ETH": 0.05},
        "65/25/5/5 spy/gld/btc/eth": {
            "SPY": 0.65,
            "GLD": 0.25,
            "BTC": 0.05,
            "ETH": 0.05,
        },
        "90/10 spy/eth": {"SPY": 0.9, "ETH": 0.1},
        "EW8": {a: 1 / 8 for a in ASSETS},
        "EW5 spy/efa/gld/tlt/btc": {
            a: 0.2 for a in ("SPY", "EFA", "GLD", "TLT", "BTC")
        },
        "kons 40/40/20 spy/gld/tlt": {"SPY": 0.4, "GLD": 0.4, "TLT": 0.2},
        "aggr 80/10/10 spy/btc/eth": {"SPY": 0.8, "BTC": 0.1, "ETH": 0.1},
        "maxcrypto 70/30 btc/eth": {"BTC": 0.7, "ETH": 0.3},
        "goldheavy 50/50 spy/gld": {"SPY": 0.5, "GLD": 0.5},
        "100 GLD": {"GLD": 1.0},
        "100 BTC": {"BTC": 1.0},
    }.items():
        C[name] = w
    vol = df.pct_change().std()
    for name, subset in (
        ("IV4 spy/efa/gld/tlt", ["SPY", "EFA", "GLD", "TLT"]),
        ("IV3 spy/gld/btc", ["SPY", "GLD", "BTC"]),
    ):
        iv = 1.0 / vol[subset]
        C[name] = dict((iv / iv.sum()).round(4))
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        C[f"btc_ladder_{int(eps * 100)}"] = {
            "SPY": 0.7 * (1 - eps),
            "GLD": 0.3 * (1 - eps),
            "BTC": eps,
        }
    return C


def main() -> int:
    df = load()
    R = df[ASSETS].pct_change().dropna().values
    T = len(R)
    years = T / 252.0
    ai = {a: i for i, a in enumerate(ASSETS)}
    C = constructions(df)
    print(
        f"[LAB] {len(C)} Konstruktionen, Fenster {years:.1f} J, 1.000 Pfade × 2 Szenarien",
        flush=True,
    )

    rng = np.random.default_rng(42)
    paths = []
    for _ in range(1000):
        idx = np.empty(T, dtype=np.int64)
        pos = 0
        while pos < T:
            si = rng.integers(0, T)
            blk = min(int(rng.geometric(1 / 60)), T - pos)
            idx[pos : pos + blk] = np.arange(si, si + blk) % T
            pos += blk
        paths.append(idx)

    out = {"n_constructions": len(C), "years": round(years, 1)}
    for scen, hc in (("crypto_x1.0", 1.0), ("crypto_x0.25", 0.25)):
        stats = {k: {"net": [], "dd": []} for k in C}
        spy_nets = []
        for idx in paths:
            Rp = R[idx].copy()
            Rp[:, ai["BTC"]] *= hc
            Rp[:, ai["ETH"]] *= hc
            cum = np.cumprod(1 + Rp, axis=0)
            spy_net = None
            for name, w in C.items():
                path = sum(wa * cum[:, ai[a]] for a, wa in w.items())
                net = START
                for a, wa in w.items():
                    g = wa * START * (cum[-1, ai[a]] - 1)
                    net += g * (1 - TAX[a]) if g > 0 else g
                dd = float((path / np.maximum.accumulate(path) - 1).min())
                stats[name]["net"].append(net)
                stats[name]["dd"].append(dd)
                if name == "100 SPY":
                    spy_net = net
            spy_nets.append(spy_net)
        spy_nets = np.array(spy_nets)
        rows = []
        for name in C:
            n = np.array(stats[name]["net"])
            rows.append(
                {
                    "name": name,
                    "median": round(float(np.median(n))),
                    "q05": round(float(np.quantile(n, 0.05))),
                    "median_dd": round(float(np.median(stats[name]["dd"])), 3),
                    "P_beat_SPY": round(float((n > spy_nets).mean() * 100), 1),
                }
            )
        rows.sort(key=lambda x: -x["q05"])
        out[scen] = rows
        print(f"\n[{scen}] Top-12 nach FLOOR (q05):", flush=True)
        for r in rows[:12]:
            print(
                f"  {r['name']:28s} med={r['median']:>9,} q05={r['q05']:>9,} "
                f"dd={r['median_dd']:.2f} P>{r['P_beat_SPY']}%",
                flush=True,
            )
    (OUTD / "h078_portfolio_lab.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
