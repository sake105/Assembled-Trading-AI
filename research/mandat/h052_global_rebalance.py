"""H-052 — Global Tax-Aware Rebalanced Portfolio (Weltmarkt + §23-Rebalancing).

Assets: US(SPY) + Dev-ex-US(EFA) + EM(EEM) + §23-Gold(GLD) [+ §23-Krypto(BTC)].
Per-Sleeve-Basis-Tracking; jährlicher Rebalance; Aktien-ETF-Verkauf 18,46 %, §23-Sleeve 0 %
(>1 J, FIFO-alt = bei jährl. Rebalance eines dauerhaft gehaltenen Sleeves erfüllt). End-Liquidation.
Vergleich: 100%-SPY-BH vs Global-BH vs Global-rebalanced vs +Krypto. Guardrail 4: nur Spot.
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
TAX = {
    "SPY": 0.1846,
    "EFA": 0.1846,
    "EEM": 0.1846,
    "GLD": 0.0,
    "GLD_hc": 0.0,
    "BTC": 0.0,
}


def load() -> pd.DataFrame:
    we = pd.read_parquet(DATA / "prices_world_etf.parquet")
    px = we.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    gld = pc[pc["symbol"] == "GLD"].set_index("timestamp")["close"]
    df = px[["SPY", "EFA", "EEM"]].copy()
    df.index = pd.DatetimeIndex(df.index)
    df["GLD"] = pd.Series(gld.values, index=pd.DatetimeIndex(gld.index)).reindex(
        df.index
    )
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    df["BTC"] = (
        pd.Series(btc.values, index=pd.DatetimeIndex(btc.index))
        .reindex(df.index)
        .ffill()
    )
    rg = (
        df["GLD"].pct_change().fillna(0.0) * 0.3
    )  # gold ×0.3 (Norm-Rückkehr) — Robustheits-Stress
    df["GLD_hc"] = (1 + rg).cumprod()
    df.loc[df["GLD"].isna(), "GLD_hc"] = np.nan
    return df


def simulate(df: pd.DataFrame, weights: dict, start: str, *, rebalance: bool) -> dict:
    assets = list(weights)
    sub = df[df.index >= pd.Timestamp(start, tz="UTC")].dropna(subset=assets)
    idx = sub.index
    units = {a: 0.0 for a in assets}
    basis = {a: 0.0 for a in assets}  # cost basis total per sleeve
    cash = START
    tax_paid = 0.0

    def buy(a, dollars, px):
        nonlocal cash
        dollars = min(dollars, cash)
        if dollars <= 0:
            return
        units[a] += dollars / px
        basis[a] += dollars
        cash -= dollars

    def sell(a, dollars, px):  # returns nothing; realizes tax
        nonlocal cash, tax_paid
        val = units[a] * px
        if val <= 0 or dollars <= 0:
            return
        dollars = min(dollars, val)
        frac = dollars / val
        gain = dollars - frac * basis[a]
        units[a] *= 1 - frac
        basis[a] *= 1 - frac
        t = max(gain, 0.0) * TAX[a]
        tax_paid += t
        cash += dollars - t

    # initial allocation at first day
    p0 = sub.iloc[0]
    for a in assets:
        buy(a, weights[a] * START, float(p0[a]))

    eq = []
    years_seen = set()
    for t in idx:
        p = sub.loc[t]
        val = cash + sum(units[a] * float(p[a]) for a in assets)
        eq.append((t, val))
        if rebalance and t.year not in years_seen and t.year > idx[0].year:
            years_seen.add(t.year)
            total = val
            # sells first (overweight), then buys
            for a in assets:
                cur = units[a] * float(p[a])
                tgt = weights[a] * total
                if cur > tgt + 1:
                    sell(a, cur - tgt, float(p[a]))
            for a in assets:
                cur = units[a] * float(p[a])
                tgt = weights[a] * total
                if cur < tgt - 1:
                    buy(a, tgt - cur, float(p[a]))

    # terminal liquidation
    pend = sub.iloc[-1]
    for a in assets:
        if units[a] > 0:
            sell(a, units[a] * float(pend[a]), float(pend[a]))
    net_final = cash

    e = pd.Series(dict(eq))
    ret = e.pct_change().dropna()
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net_final": round(net_final),
        "cagr_net_pct": round(((net_final / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(ret.mean() / ret.std() * sqrt(252)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "tax_paid": round(tax_paid),
        "years": round(years, 1),
    }


def main() -> int:
    df = load()
    s = "2005-01-01"  # SPY/EFA/EEM/GLD common
    out = {
        "window_A_2005_noCrypto": {
            "100% SPY (BH)": simulate(df, {"SPY": 1.0}, s, rebalance=False),
            "Global-Equity BH (60/25/15)": simulate(
                df, {"SPY": 0.6, "EFA": 0.25, "EEM": 0.15}, s, rebalance=False
            ),
            "Global-Equity REBAL": simulate(
                df, {"SPY": 0.6, "EFA": 0.25, "EEM": 0.15}, s, rebalance=True
            ),
            "Global+Gold REBAL (55/20/10/15)": simulate(
                df,
                {"SPY": 0.55, "EFA": 0.2, "EEM": 0.1, "GLD": 0.15},
                s,
                rebalance=True,
            ),
            "US+Gold REBAL (85/15)": simulate(
                df, {"SPY": 0.85, "GLD": 0.15}, s, rebalance=True
            ),
            "US+Gold_hc REBAL (85/15, gold x0.3)": simulate(
                df, {"SPY": 0.85, "GLD_hc": 0.15}, s, rebalance=True
            ),
        }
    }
    sc = "2016-01-01"
    out["window_B_2016_withCrypto"] = {
        "100% SPY (BH)": simulate(df, {"SPY": 1.0}, sc, rebalance=False),
        "Global+Gold+BTC REBAL (55/15/10/12/8)": simulate(
            df,
            {"SPY": 0.55, "EFA": 0.15, "EEM": 0.1, "GLD": 0.12, "BTC": 0.08},
            sc,
            rebalance=True,
        ),
        "US+Gold+BTC REBAL (80/12/8)": simulate(
            df, {"SPY": 0.8, "GLD": 0.12, "BTC": 0.08}, sc, rebalance=True
        ),
    }
    (OUTD / "h052_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[RESULT]", json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
