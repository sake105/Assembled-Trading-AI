"""H-034 — Congress LETZTE Patrone: volles Universum (S&P + Non-S&P). Registry W10b.

Identische 3er-Familie wie H-033, Preisbasis erweitert um prices_congress_extra
(gleiche Datenhygiene: Kappung an unmoeglichen Spruengen, $1-Floor im Runner). N->104.
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
from h031_insider import run_insider  # noqa: E402
from verdict_engine import (
    DATA,
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)  # noqa: E402

ETF_TAX = 0.185
N_TRIALS = 104
SIGNAL_START = pd.Timestamp("2013-01-01", tz="UTC")


def load_full_close() -> pd.DataFrame:
    base = load_verdict_prices()  # hygiene included
    extra = pd.read_parquet(DATA / "prices_congress_extra.parquet")
    ex = extra.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    ex = ex.reindex(base.index)
    r = ex.pct_change(fill_method=None)
    bad = (r.abs() > 1.0) & (ex.shift(1) < 1.0)
    n = 0
    for sym in ex.columns[bad.any()]:
        ex.loc[bad.index[bad[sym]][0] :, sym] = np.nan
        n += 1
    print(f"[HYGIENE] extra: truncated {n} corrupt series", flush=True)
    ex = ex[[c for c in ex.columns if c not in base.columns]]
    return pd.concat([base, ex], axis=1)


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_full_close()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    month_ends = sorted(membership.index)
    print(f"[DATA] full universe: {close.shape[1]} symbols", flush=True)

    c = pd.read_parquet(
        ROOT / "data" / "raw" / "insider_congress" / "congress_trades_full.parquet"
    )
    c = c[c["type"] == "buy"].copy()
    c["available_at"] = pd.to_datetime(c["available_at"], utc=True, errors="coerce")
    c = c.dropna(subset=["available_at", "symbol"])
    cov = c["symbol"].isin(close.columns).mean()
    print(f"[DATA] buys coverage now: {cov * 100:.0f}%", flush=True)
    c = c[c["symbol"].isin(close.columns)]

    def monthly_sig(df, cluster_min=1):
        sig = {}
        for me in month_ends:
            if me < SIGNAL_START:
                continue
            recent = df[
                (df["available_at"] <= me)
                & (df["available_at"] > me - pd.DateOffset(months=3))
            ]
            if cluster_min > 1:
                counts = recent.groupby("symbol")["member"].nunique()
                syms = set(counts[counts >= cluster_min].index)
            else:
                syms = set(recent["symbol"])
            if syms:
                sig[me] = syms
        return sig

    variants = {
        "H034_copy_all": monthly_sig(c),
        "H034_big_buys": monthly_sig(
            c[pd.to_numeric(c["amount_low"], errors="coerce") >= 50000]
        ),
        "H034_cluster3": monthly_sig(c, cluster_min=3),
    }
    results, rets = {}, {}
    for name, sig in variants.items():
        res, _eq, ret = run_insider(close, divp, sig, month_ends, label=name)
        ret = ret[ret.index >= SIGNAL_START]
        eqr = (1 + ret).cumprod()
        yrs = (ret.index[-1] - ret.index[0]).days / 365.25
        res["final_value_window"] = float(eqr.iloc[-1] * START_CAPITAL)
        res["sharpe_window"] = float(ret.mean() / ret.std() * np.sqrt(252))
        res["maxdd_window"] = float((eqr / eqr.cummax() - 1).min())
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value_window']:.0f} sharpe={res['sharpe_window']:.3f} maxdd={res['maxdd_window'] * 100:.1f}%",
            flush=True,
        )

    spy = close["SPY"].dropna()
    spy = spy[spy.index >= SIGNAL_START]
    spy_r = spy.pct_change().dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    results["ETF_net_path"] = {"final_value": float(etf_net)}
    results["SPY_sharpe"] = float(spy_r.mean() / spy_r.std() * np.sqrt(252))
    results["SPY_maxdd"] = float((spy / spy.cummax() - 1).min())
    _r, _e, ret_ew = run_verdict(
        close[[cn for cn in close.columns]],
        membership,
        label="EW_ref",
        mode="ew",
        div_panel=divp,
    )
    ret_ew = ret_ew[ret_ew.index >= SIGNAL_START]
    ew_sharpe = float(ret_ew.mean() / ret_ew.std() * np.sqrt(252))
    results["EW_window_sharpe"] = ew_sharpe

    best = max(rets, key=lambda k: results[k]["final_value_window"])
    dsr = deflated_sharpe(rets[best], n_trials=N_TRIALS)
    v = rets[best]
    win = {}
    for y0 in range(2013, 2026, 2):
        m = (v.index.year >= y0) & (v.index.year < y0 + 2)
        me = (ret_ew.index.year >= y0) & (ret_ew.index.year < y0 + 2)
        if m.sum() > 100 and me.sum() > 100:
            win[str(y0)] = {
                "H034": round(float(v[m].mean() / v[m].std() * np.sqrt(252)), 3),
                "EW": round(
                    float(ret_ew[me].mean() / ret_ew[me].std() * np.sqrt(252)), 3
                ),
            }
    n_win = sum(1 for w in win.values() if w["H034"] >= w["EW"])
    results["_verdict"] = {
        "selected": best,
        "crit1_gt_etf": results[best]["final_value_window"] > etf_net,
        "crit2_sharpe_gt_ew": results[best]["sharpe_window"] > ew_sharpe,
        "crit3_dsr": {
            "prob": float(dsr.deflated_sharpe_probability),
            "pass": bool(dsr.passes_5pct),
        },
        "crit4_pass": n_win >= max(3, int(0.6 * len(win))) if win else False,
        "crit4_windows": win,
        "crit5_maxdd_ok": results[best]["maxdd_window"] >= results["SPY_maxdd"],
        "spy_corr_all": {
            k: round(float(rets[k].corr(spy_r.reindex(rets[k].index))), 3) for k in rets
        },
        "PBO_info": float(cscv_pbo(pd.DataFrame(rets))),
    }
    results["_verdict"]["PASS"] = all(
        [
            results["_verdict"]["crit1_gt_etf"],
            results["_verdict"]["crit2_sharpe_gt_ew"],
            results["_verdict"]["crit3_dsr"]["pass"],
            results["_verdict"]["crit4_pass"],
            results["_verdict"]["crit5_maxdd_ok"],
        ]
    )
    (OUTD / "h034_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
