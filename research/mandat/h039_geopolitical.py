"""H-039 — Geopolitik-News -> Crisis-Alpha (Registry Welle 14). Starker NULL-Prior (Fable R5).

Test A (Event-Study): forward 5/20/60T Crisis-Basket minus SPY nach Intensitaets-Spike.
Test B (Trial, 2): monatliche Rotation z>0.5 -> Crisis sonst SPY; und z>0 -> 50/50.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, START_CAPITAL, TaxedPortfolio  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
ETF_TAX = 0.185
START_WIN = pd.Timestamp("2016-01-01", tz="UTC")


def load():
    px = pd.read_parquet(DATA / "prices_crisis.parquet")
    close = px.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    inten = pd.read_parquet(DATA / "geopol_intensity.parquet").set_index("date")[
        "n_articles"
    ]
    inten = inten.reindex(close.index).fillna(0.0)
    z = (inten - inten.rolling(252, min_periods=120).mean()) / inten.rolling(
        252, min_periods=120
    ).std()
    return close, z.shift(1)  # PIT: yesterday's intensity known today


def event_study(close, z):
    crisis = close[["XLE", "GLD", "ITA"]].pct_change().mean(axis=1)
    spy = close["SPY"].pct_change()
    excess = crisis - spy
    spikes = z[z > 1.0].index
    spikes = spikes[spikes >= START_WIN]
    out = {}
    for h in (5, 20, 60):
        fwd = excess.rolling(h).sum().shift(-h)  # forward h-day cum excess
        vals = fwd.reindex(spikes).dropna()
        mean = float(vals.mean())
        t = (
            float(mean / (vals.std() / np.sqrt(len(vals))))
            if len(vals) > 2
            else float("nan")
        )
        out[f"{h}d"] = {
            "mean_excess_pct": round(mean * 100, 3),
            "t": round(t, 2),
            "n": int(len(vals)),
        }
    return out


def rotate(close, z, *, mode: str):
    idx = close.index
    month_ends = set(pd.Series(idx, index=idx).groupby(idx.to_period("M")).max())
    crisis_syms = ["XLE", "GLD", "ITA"]
    pf = TaxedPortfolio(START_CAPITAL)
    pending, eq = [], []
    for t in idx:
        px_t = close.loc[t]
        for act, sym, amt in pending:
            p = px_t.get(sym, np.nan)
            if not np.isfinite(p):
                continue
            if act == "sell_all":
                q = pf.qty(sym)
                if q > 0:
                    pf.sell(sym, q, float(p))
            else:
                d = amt - pf.qty(sym) * p
                if d > 1:
                    pf.buy(sym, d, float(p))
                elif d < -1:
                    pf.sell(sym, -d / p, float(p))
        pending = []
        v = pf.cash + sum(
            pf.qty(s) * px_t.get(s, 0)
            for s in ["XLE", "GLD", "ITA", "SPY"]
            if np.isfinite(px_t.get(s, np.nan)) and pf.qty(s) > 0
        )
        eq.append((t, v))
        if t not in month_ends or not np.isfinite(z.get(t, np.nan)):
            continue
        zt = z.at[t]
        if mode == "hard":
            targets = {s: 1 / 3 for s in crisis_syms} if zt > 0.5 else {"SPY": 1.0}
        else:  # tilt
            targets = (
                ({s: 1 / 6 for s in crisis_syms} | {"SPY": 0.5})
                if zt > 0
                else {"SPY": 1.0}
            )
        held = set(pf.lots.keys())
        for s in held - set(targets):
            pending.append(("sell_all", s, 0.0))
        for s, w in targets.items():
            pending.append(("trade_to", s, w * v))
    e = pd.Series(dict(eq)).sort_index()
    e = e[e.index >= START_WIN]
    r = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    return {
        "final": float(e.iloc[-1] / e.iloc[0] * START_CAPITAL),
        "cagr": float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
        "maxdd": float((e / e.cummax() - 1).min()),
        "tax": float(pf.tax_paid),
    }


def main() -> int:
    close, z = load()
    print(
        f"[DATA] {close.index[0].date()} -> {close.index[-1].date()}, spikes(z>1): {(z > 1).sum()}",
        flush=True,
    )
    ev = event_study(close, z)
    print("[EVENT]", ev, flush=True)

    hard = rotate(close, z, mode="hard")
    tilt = rotate(close, z, mode="tilt")
    print(f"[RUN] hard: {hard}", flush=True)
    print(f"[RUN] tilt: {tilt}", flush=True)

    # benchmarks same window
    spy = close["SPY"].dropna()
    spy = spy[spy.index >= START_WIN]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_r = spy.pct_change().dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    static = close[["XLE", "GLD", "ITA", "SPY"]].pct_change()
    static_r = (
        static[["XLE", "GLD", "ITA"]].mean(axis=1) * 0.5 + static["SPY"] * 0.5
    ).dropna()
    static_r = static_r[static_r.index >= START_WIN]
    static_eq = (1 + static_r).cumprod()

    a_pass = sum(1 for h in ev.values() if h["mean_excess_pct"] > 0 and h["t"] > 2) >= 2
    b_pass = (
        hard["final"] > etf_net and hard["final"] > static_eq.iloc[-1] * START_CAPITAL
    )
    verdict = {
        "test_A_event_study": ev,
        "test_A_pass": a_pass,
        "ETF_net": round(etf_net),
        "static_50_50_final": round(float(static_eq.iloc[-1] * START_CAPITAL)),
        "SPY_sharpe": round(float(spy_r.mean() / spy_r.std() * np.sqrt(252)), 3),
        "test_B_hard": hard,
        "test_B_tilt": tilt,
        "test_B_pass": bool(b_pass),
        "PASS": bool(a_pass and b_pass),
    }
    (OUTD / "h039_results.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(verdict, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
