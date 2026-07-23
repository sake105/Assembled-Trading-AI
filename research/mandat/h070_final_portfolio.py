"""H-070 — Integriertes Endportfolio (Synthese): 100%SPY vs 70/30 vs 70/25/5 (+BTC-Szenarien).

MC 1.000 Pfade (Block 60T, Seed 42), Fenster 2016+ (BTC-Constraint), Lump-Sum 100k,
Terminal-Steuer SPY 18,46 %, Gold/BTC §23 0 %. Auch MaxDD je Pfad (gross path).
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
TAX_EQ = 0.1846
ALLOCS = {
    "100% SPY": {"SPY": 1.0},
    "70/30": {"SPY": 0.7, "GLD": 0.3},
    "70/25/5": {"SPY": 0.7, "GLD": 0.25, "BTC": 0.05},
}


def main() -> int:
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    df = px[["SPY", "GLD"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    df["BTC"] = (
        pd.Series(btc.values, index=pd.DatetimeIndex(btc.index))
        .reindex(df.index)
        .ffill()
    )
    df = df.dropna()
    df = df[df.index >= pd.Timestamp("2016-01-01", tz="UTC")]
    R = df.pct_change().dropna().values
    T = len(R)
    years = T / 252.0
    ai = {"SPY": 0, "GLD": 1, "BTC": 2}
    rng = np.random.default_rng(42)

    out = {"window_years": round(years, 1), "n_paths": 1000, "seed": 42}
    for hc in (1.0, 0.25, 0.0):
        stats = {k: {"net": [], "dd": []} for k in ALLOCS}
        rng = np.random.default_rng(42)  # same paths per scenario
        for _ in range(1000):
            idx = np.empty(T, dtype=np.int64)
            pos = 0
            while pos < T:
                si = rng.integers(0, T)
                blk = min(int(rng.geometric(1 / 60)), T - pos)
                idx[pos : pos + blk] = np.arange(si, si + blk) % T
                pos += blk
            Rp = R[idx].copy()
            Rp[:, 2] *= hc
            cum = np.cumprod(1 + Rp, axis=0)
            for name, w in ALLOCS.items():
                path = sum(w[a] * cum[:, ai[a]] for a in w)
                vs = w["SPY"] * START * cum[-1, 0]
                net = vs - max(vs - w["SPY"] * START, 0) * TAX_EQ
                for a in ("GLD", "BTC"):
                    if a in w:
                        net += w[a] * START * cum[-1, ai[a]]  # §23 0 %
                dd = float((path / np.maximum.accumulate(path) - 1).min())
                stats[name]["net"].append(net)
                stats[name]["dd"].append(dd)
        sc = {}
        for name in ALLOCS:
            n = np.array(stats[name]["net"])
            sc[name] = {
                "median": round(float(np.median(n))),
                "q05": round(float(np.quantile(n, 0.05))),
                "median_dd": round(float(np.median(stats[name]["dd"])), 3),
            }
        out[f"btc_x{hc}"] = sc
        print(f"[H070] BTC×{hc}: " + json.dumps(sc), flush=True)

    (OUTD / "h070_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
