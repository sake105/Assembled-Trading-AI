"""H-053 — §4.6.1 Insider-Patrone auf BREITEM survivorship-freiem Universum.

Opportunistische Insider-Käufe (Cohen-Malloy-Pomorski) + CLUSTER (≥2 Insider), auf dem
Small-Cap-Broad-Universum (15.101 handelbare Namen inkl. delisted) statt nur S&P (H-031).
HANDELBARKEITS-FLOOR (Preis ≥ $5, ADV60 ≥ $1M) gegen das H-036-Illiquiditäts-Artefakt.
30 bps, 12M-Halten, EW-Cap 10 %. Vergleich vs SPY-ETF-Pfad + DSR/PBO/Fenster-Konsistenz.
Div-Steuer für Small Caps weggelassen (minimal, pro-Strategie-Bias benannt).
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

from h011_kandidat_a import OUTD, START_CAPITAL, cscv_pbo  # noqa: E402
from h031_insider import classify_opportunistic, load_purchases, run_insider  # noqa: E402
from smallcap_data import FLOOR_ADV, FLOOR_PRICE, load_smallcap  # noqa: E402

ETF_TAX = 0.185
SIGNAL_START = pd.Timestamp("2005-01-01", tz="UTC")


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close, adv = load_smallcap()
    idx = close.index
    month_ends = list(idx.to_series().groupby(idx.to_period("M")).max())
    empty_div = pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    p = load_purchases()
    opp = classify_opportunistic(p)
    print(
        f"[SIG] {len(p)} P-buys -> {len(opp)} opportunistic ({100 * len(opp) / len(p):.1f}%), "
        f"{opp['symbol'].nunique()} symbols",
        flush=True,
    )

    cols = set(close.columns)

    def tradable(me):
        if me not in adv.index:
            return set()
        pr = close.loc[me]
        av = adv.loc[me]
        ok = av[(av >= FLOOR_ADV)].index
        return {
            s
            for s in ok
            if s in cols
            and np.isfinite(pr.get(s, np.nan))
            and pr.get(s, 0.0) >= FLOOR_PRICE
        }

    # precompute tradable per month-end (>= SIGNAL_START)
    trad = {me: tradable(me) for me in month_ends if me >= SIGNAL_START}

    def build_sig(df, min_insiders):
        sig = {}
        for me in month_ends:
            if me < SIGNAL_START:
                continue
            recent = df[
                (df["available_at"] <= me)
                & (df["available_at"] > me - pd.DateOffset(months=3))
            ]
            if not len(recent):
                continue
            byc = recent.groupby("symbol")["reporting_owner_cik"].nunique()
            cand = set(byc[byc >= min_insiders].index)
            s = cand & trad[me]
            if s:
                sig[me] = s
        return sig

    variants = {
        "H053_all_opp": build_sig(opp, 1),
        "H053_cluster2": build_sig(opp, 2),
    }
    results, rets = {}, {}
    for name, sig in variants.items():
        med = int(np.median([len(v) for v in sig.values()])) if sig else 0
        print(
            f"[SIG] {name}: {len(sig)} signal-months, median {med} names/month",
            flush=True,
        )
        res, _e, ret = run_insider(close, empty_div, sig, month_ends, label=name)
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:,.0f} cagr={res['cagr_net'] * 100:.2f}% "
            f"sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net'] * 100:.1f}%",
            flush=True,
        )

    spy = close["SPY"].dropna()
    spy = spy[spy.index >= SIGNAL_START]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_r = spy.pct_change().dropna()
    spy_sharpe = float(spy_r.mean() / spy_r.std() * np.sqrt(252))
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )

    best = max(rets, key=lambda k: results[k]["final_value"])
    v = rets[best]
    dsr = deflated_sharpe(v, n_trials=141)
    win = {}
    for y0 in range(2005, 2026, 4):
        m = (v.index.year >= y0) & (v.index.year < y0 + 4)
        ms = (spy_r.index.year >= y0) & (spy_r.index.year < y0 + 4)
        if m.sum() > 200 and ms.sum() > 200:
            win[str(y0)] = {
                "ins": round(float(v[m].mean() / v[m].std() * np.sqrt(252)), 3),
                "spy": round(
                    float(spy_r[ms].mean() / spy_r[ms].std() * np.sqrt(252)), 3
                ),
            }
    n_win = sum(1 for w in win.values() if w["ins"] >= w["spy"])

    verdict = {
        "selected": best,
        "best_final": round(results[best]["final_value"]),
        "best_sharpe": round(results[best]["sharpe_net"], 3),
        "best_maxdd": round(results[best]["maxdd_net"], 3),
        "ETF_net_path": round(float(etf_net)),
        "SPY_sharpe": round(spy_sharpe, 3),
        "crit1_gt_ETF": bool(results[best]["final_value"] > etf_net),
        "crit2_sharpe_gt_SPY": bool(results[best]["sharpe_net"] > spy_sharpe),
        "crit3_DSR": {
            "prob": round(float(dsr.deflated_sharpe_probability), 3),
            "pass": bool(dsr.passes_5pct),
        },
        "crit4_windows": win,
        "crit4_pass": bool(win and n_win >= max(3, int(0.6 * len(win)))),
        "crit5_maxdd_ok": bool(
            results[best]["maxdd_net"] >= float((spy / spy.cummax() - 1).min())
        ),
        "PBO": round(float(cscv_pbo(pd.DataFrame(rets))), 3),
    }
    verdict["PASS"] = bool(
        verdict["crit1_gt_ETF"]
        and verdict["crit2_sharpe_gt_SPY"]
        and verdict["crit3_DSR"]["pass"]
        and verdict["crit4_pass"]
        and verdict["crit5_maxdd_ok"]
    )
    results["_verdict"] = verdict
    (OUTD / "h053_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(verdict, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
