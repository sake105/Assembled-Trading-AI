"""H-028 — GEM International (Original) mit deutscher Steuer inkl. Dividenden-Drag.

Registry Welle 6. 2 Laeufe: classic (SPY/EFA/IEF) | relative-only (SPY/EFA). N->86.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, START_CAPITAL, TaxedPortfolio, cscv_pbo  # noqa: E402
from verdict_engine import DATA  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
TOK = os.environ["EODHD_API_TOKEN"]
ETF_TAX = 0.185
TAX = 0.26375
N_TRIALS = 86
SYMS = ["SPY", "EFA", "IEF"]


def ensure_etf_prices() -> pd.DataFrame:
    p = DATA / "prices_etf_eodhd.parquet"
    if p.exists():
        return pd.read_parquet(p)
    frames = []
    for s in SYMS:
        url = (
            f"https://eodhd.com/api/eod/{s}.US?api_token={TOK}&fmt=json&from=1995-01-01"
        )
        rows = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "research"}),
                timeout=45,
            )
            .read()
            .decode()
        )
        df = pd.DataFrame(rows)[["date", "adjusted_close"]]
        df.columns = ["timestamp", "close"]
        df["symbol"] = s
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out.to_parquet(p, index=False)
    return out


def run_gem_intl(
    px: pd.DataFrame, divs: pd.DataFrame, *, absolute_gate: bool, label: str
):
    close = (
        px.pivot(index="timestamp", columns="symbol", values="close")
        .dropna()
        .sort_index()
    )
    r12 = close / close.shift(252) - 1.0
    month_ends = set(
        pd.Series(close.index, index=close.index)
        .groupby(close.index.to_period("M"))
        .max()
    )
    dv = divs[divs["symbol"].isin(SYMS)]
    pos = close.index.searchsorted(pd.DatetimeIndex(dv["ex_date"]))
    dv = dv.assign(t=close.index[np.clip(pos, 0, len(close.index) - 1)])
    div_map = dv.groupby(["t", "symbol"])["dividend"].sum()

    pf = TaxedPortfolio(START_CAPITAL)
    cur: str | None = None
    pend: str | None = None
    eq = []
    for t in close.index:
        if pend is not None and pend != cur:
            if cur is not None:
                q = pf.qty(cur)
                if q > 0:
                    pf.sell(cur, q, float(close.at[t, cur]))
            pf.buy(pend, pf.cash, float(close.at[t, pend]))
            cur = pend
        pend = None
        if cur is not None:
            d = div_map.get((t, cur), 0.0)
            if d > 0:
                tax = pf.qty(cur) * d * TAX
                pf.cash -= tax
                pf.tax_paid += tax
        v = pf.cash + (pf.qty(cur) * float(close.at[t, cur]) if cur else 0.0)
        eq.append((t, v))
        if (
            t in month_ends
            and np.isfinite(r12["SPY"].at[t])
            and np.isfinite(r12["EFA"].at[t])
        ):
            best = "SPY" if r12["SPY"].at[t] >= r12["EFA"].at[t] else "EFA"
            if absolute_gate and r12[best].at[t] <= 0:
                tgt = "IEF"
            else:
                tgt = best
            if tgt != cur:
                pend = tgt
    e = pd.Series(dict(eq)).sort_index()
    e = e[r12["SPY"].notna() & r12["EFA"].notna()]
    ret = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    return (
        {
            "label": label,
            "final_value": float(e.iloc[-1] / e.iloc[0] * START_CAPITAL),
            "cagr_net": float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1),
            "sharpe_net": float(ret.mean() / ret.std() * np.sqrt(252)),
            "maxdd_net": float((e / e.cummax() - 1).min()),
            "tax_paid": float(pf.tax_paid),
            "years": float(years),
        },
        e,
        ret,
    )


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    px = ensure_etf_prices()
    divs = pd.read_parquet(DATA / "dividends.parquet")
    results, rets = {}, {}
    for absolute, name in ((True, "H028_gem_classic"), (False, "H028_relative_only")):
        res, eq, ret = run_gem_intl(px, divs, absolute_gate=absolute, label=name)
        results[name] = res
        rets[name] = ret
        print(f"[RUN] {name}: {res}", flush=True)

    close = (
        px.pivot(index="timestamp", columns="symbol", values="close")
        .dropna()
        .sort_index()
    )
    spy = close["SPY"]
    spy = spy[spy.index >= rets["H028_gem_classic"].index[0]]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_r = spy.pct_change().dropna()
    # SPY B&H net incl. dividend-tax drag approximation is complex; report gross
    # SPY + ETF net path (thesaurierend, TFS) as the registry benchmark.
    gross_gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    results["SPY_bh"] = {
        "cagr_gross": float((spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(spy_r.mean() / spy_r.std() * np.sqrt(252)),
        "maxdd": float((spy / spy.cummax() - 1).min()),
    }
    results["ETF_net_path"] = {"final_value": float(etf_net)}

    best = max(rets, key=lambda k: results[k]["final_value"])
    dsr = deflated_sharpe(rets[best], n_trials=N_TRIALS)
    v = rets[best]
    mid = v.index[len(v) // 2]
    halves = [
        float(sl.mean() / sl.std() * np.sqrt(252))
        for sl in (v[v.index < mid], v[v.index >= mid])
    ]
    spy_halves = [
        float(sl.mean() / sl.std() * np.sqrt(252))
        for sl in (spy_r[spy_r.index < mid], spy_r[spy_r.index >= mid])
    ]
    results["_verdict"] = {
        "selected": best,
        "crit1_gt_etf": results[best]["final_value"] > etf_net,
        "crit2_dsr": {
            "prob": float(dsr.deflated_sharpe_probability),
            "pass": bool(dsr.passes_5pct),
        },
        "crit3_maxdd_ok": results[best]["maxdd_net"] >= results["SPY_bh"]["maxdd"],
        "crit5_halves": {
            "best": halves,
            "spy": spy_halves,
            "pass": all(h >= s - 0.05 for h, s in zip(halves, spy_halves)),
        },
        "PBO_info": float(cscv_pbo(pd.DataFrame(rets))),
    }
    results["_verdict"]["PASS"] = all(
        [
            results["_verdict"]["crit1_gt_etf"],
            results["_verdict"]["crit2_dsr"]["pass"],
            results["_verdict"]["crit3_maxdd_ok"],
            results["_verdict"]["crit5_halves"]["pass"],
        ]
    )
    (OUTD / "h028_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
