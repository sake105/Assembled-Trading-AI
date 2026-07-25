"""H-029 — 13F-Top-Manager-Konsens (Best Ideas). Registry Welle 7, N->88.

Signal je Quartal: Top-10-Positionen (VALUE-Gewicht) je Top-100-Manager ->
Konsens-Zaehler je Ticker. Wirksam am Monatsultimo nach 45-Tage-Deadline.
Portfolio: EW aller Namen mit Konsens >= K, Cap 10 %, no-retrim, Exit < K/2.
Deutsche Steuern inkl. Dividenden-Drag. 2 Laeufe: K=5, K=10.
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

from h011_kandidat_a import OUTD, START_CAPITAL, TaxedPortfolio, cscv_pbo  # noqa: E402
from verdict_engine import (  # noqa: E402
    DATA,
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)

TAX = 0.26375
ETF_TAX = 0.185
N_TRIALS = 88
POS_CAP = 0.10


def build_consensus(top_pos: int = 10, m_top: int = 100) -> pd.DataFrame:
    """Quarter -> ticker -> #managers holding it in their top-`top_pos` (by weight)."""
    h = pd.read_parquet(
        DATA / "13f_top100.parquet",
        columns=["ACCESSION_NUMBER", "CUSIP", "VALUE", "PERIODOFREPORT", "quarter_zip"],
    )
    cm = pd.read_parquet(DATA / "cusip_ticker_map.parquet")
    cm = cm.sort_values("year").drop_duplicates("CUSIP", keep="last")[
        ["CUSIP", "SYMBOL"]
    ]
    # aggregate duplicate holdings per accession+cusip
    g = h.groupby(["PERIODOFREPORT", "ACCESSION_NUMBER", "CUSIP"], as_index=False)[
        "VALUE"
    ].sum()
    # DATA-HYGIENE FIX (runs 87-88 were contaminated): rank managers PER
    # PERIODOFREPORT across all zips (2024+ zips are 3-month spans that split a
    # quarter across files -> per-zip top-100 splinters the consensus), keep
    # top 100 per period; drop rump periods with < 50 managers (late-filing
    # amendments to old periods produced 2-114 rows/yr for 2009-2012).
    mgr_tot = g.groupby(["PERIODOFREPORT", "ACCESSION_NUMBER"], as_index=False)[
        "VALUE"
    ].sum()
    mgr_tot["mrank"] = mgr_tot.groupby("PERIODOFREPORT")["VALUE"].rank(
        ascending=False, method="first"
    )
    top_mgrs = mgr_tot[mgr_tot["mrank"] <= m_top]
    period_cov = top_mgrs.groupby("PERIODOFREPORT")["ACCESSION_NUMBER"].nunique()
    valid_periods = set(period_cov[period_cov >= 50].index)
    g = g[
        g["PERIODOFREPORT"].isin(valid_periods)
        & g.set_index(["PERIODOFREPORT", "ACCESSION_NUMBER"]).index.isin(
            top_mgrs.set_index(["PERIODOFREPORT", "ACCESSION_NUMBER"]).index
        )
    ]
    g["rank"] = g.groupby(["PERIODOFREPORT", "ACCESSION_NUMBER"])["VALUE"].rank(
        ascending=False, method="first"
    )
    top10 = (
        g[g["rank"] <= top_pos]
        .merge(cm, on="CUSIP", how="left")
        .dropna(subset=["SYMBOL"])
    )
    cons = (
        top10.groupby(["PERIODOFREPORT", "SYMBOL"])["ACCESSION_NUMBER"]
        .nunique()
        .rename("n_managers")
        .reset_index()
    )
    cons["period_end"] = pd.to_datetime(
        cons["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce"
    )
    bad = cons["period_end"].isna()
    if bad.any():
        cons.loc[bad, "period_end"] = pd.to_datetime(
            cons.loc[bad, "PERIODOFREPORT"], errors="coerce"
        )
    cons["period_end"] = pd.to_datetime(cons["period_end"], utc=True)
    return cons.dropna(subset=["period_end"])


def run_consensus(close, divp, cons, month_ends, *, k: int, label: str):
    # effective date: month-end AFTER period_end + 2 months (45d deadline, conservative)
    sig = {}
    for pe, grp in cons.groupby("period_end"):
        eff_candidates = [t for t in month_ends if t >= pe + pd.DateOffset(months=2)]
        if not eff_candidates:
            continue
        eff = min(eff_candidates)
        sig.setdefault(eff, {}).update(dict(zip(grp["SYMBOL"], grp["n_managers"])))
    pf = TaxedPortfolio(START_CAPITAL)
    pending, eq = [], []
    last_counts: dict[str, int] = {}
    last_valid = close.apply(lambda s: s.last_valid_index())
    global_last = close.index[-1]
    close_ff = close.ffill()  # valuation only; trades stay on real rows
    for t in close.index:
        px_t = close.loc[t]
        for action, sym, amount in pending:
            px = px_t.get(sym, np.nan)
            if not np.isfinite(px):
                lv = last_valid.get(sym)
                if lv is not None and lv < t:
                    px = close.at[lv, sym]
                else:
                    continue
            if action == "sell_all":
                q = pf.qty(sym)
                if q > 0:
                    pf.sell(sym, q, float(px))
            else:
                delta = amount - pf.qty(sym) * px
                if delta > 1.0:
                    pf.buy(sym, delta, float(px))
        pending = []
        for sym in list(pf.lots.keys()):
            lv = last_valid.get(sym)
            if lv is not None and lv < t and lv < global_last - pd.Timedelta(days=10):
                pending.append(("sell_all", sym, 0.0))
        if t in divp.index:
            drow = divp.loc[t]
            for sym in list(pf.lots.keys()):
                d = drow.get(sym, np.nan)
                if np.isfinite(d) and d > 0:
                    tax = pf.qty(sym) * d * TAX
                    pf.cash -= tax
                    pf.tax_paid += tax
        v = pf.cash
        ff_t = close_ff.loc[t]
        for sym, lots in pf.lots.items():
            px = ff_t.get(sym, np.nan)
            if np.isfinite(px):
                v += sum(q for q, _ in lots) * px
        eq.append((t, v))
        if t in sig:
            last_counts = sig[t]
            held = set(pf.lots.keys())
            targets = {
                s
                for s, n in last_counts.items()
                if n >= k
                and s in close.columns
                and np.isfinite(px_t.get(s, np.nan))
                and px_t.get(s, 0.0) >= 1.0
            }
            keep = {s for s in held if last_counts.get(s, 0) >= k / 2}
            # E-051-Determinismus-Fix 2026-07-24: Set-Differenz sortiert iterieren
            # (Pending-Reihenfolge -> Float-Summationsreihenfolge in Cash/Equity)
            for sym in sorted(held - keep - targets):
                pending.append(("sell_all", sym, 0.0))
            # E-051-Determinismus-Fix 2026-07-24: Entries deterministisch sortiert
            entries = sorted(s for s in targets if s not in held)
            basket = targets | keep
            if entries and basket:
                w = min(1.0 / len(basket), POS_CAP)
                for sym in entries:
                    pending.append(("trade_to", sym, w * v))
    e = pd.Series(dict(eq)).sort_index()
    first_sig = min(sig) if sig else e.index[0]
    e = e[e.index >= first_sig]
    ret = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    res = {
        "label": label,
        "final_value": float(e.iloc[-1] / e.iloc[0] * START_CAPITAL),
        "cagr_net": float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1),
        "sharpe_net": float(ret.mean() / ret.std() * np.sqrt(252)),
        "maxdd_net": float((e / e.cummax() - 1).min()),
        "tax_paid": float(pf.tax_paid),
        "years": float(years),
    }
    return res, e, ret


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    month_ends = sorted(membership.index)
    cons = build_consensus()
    print(
        f"[SIG] consensus rows: {len(cons)}, quarters: {cons['period_end'].nunique()}",
        flush=True,
    )

    results, rets = {}, {}
    for k in (5, 10):
        name = f"H029_k{k}"
        res, eq, ret = run_consensus(close, divp, cons, month_ends, k=k, label=name)
        results[name] = res
        rets[name] = ret
        print(f"[RUN] {name}: {res}", flush=True)

    # benchmarks on the same window (first signal onwards), div-tax versions
    start = rets["H029_k5"].index[0]
    spy = close["SPY"].dropna()
    spy = spy[spy.index >= start]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_r = spy.pct_change().dropna()
    gross_gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    results["SPY_bh"] = {
        "cagr_gross": float((spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(spy_r.mean() / spy_r.std() * np.sqrt(252)),
        "maxdd": float((spy / spy.cummax() - 1).min()),
    }
    results["ETF_net_path"] = {"final_value": float(etf_net)}
    res_ew, _eq, ret_ew = run_verdict(
        close[close.index >= start - pd.Timedelta(days=420)],
        membership[membership.index >= start - pd.Timedelta(days=420)],
        label="EW_PIT_divtax_window",
        mode="ew",
        div_panel=divp,
    )
    ret_ew = ret_ew[ret_ew.index >= start]
    ew_sharpe = float(ret_ew.mean() / ret_ew.std() * np.sqrt(252))
    results["EW_PIT_window"] = {"sharpe": ew_sharpe}

    best = max(rets, key=lambda kk: results[kk]["final_value"])
    v = rets[best]
    dsr = deflated_sharpe(v, n_trials=N_TRIALS)
    win = {}
    for y0 in range(v.index[0].year, v.index[-1].year, 2):
        m = (v.index.year >= y0) & (v.index.year < y0 + 2)
        me = (ret_ew.index.year >= y0) & (ret_ew.index.year < y0 + 2)
        if m.sum() > 100 and me.sum() > 100:
            win[f"{y0}"] = {
                "H029": round(float(v[m].mean() / v[m].std() * np.sqrt(252)), 3),
                "EW": round(
                    float(ret_ew[me].mean() / ret_ew[me].std() * np.sqrt(252)), 3
                ),
            }
    spy_corr = float(v.corr(spy_r.reindex(v.index)))
    results["_verdict"] = {
        "selected": best,
        "crit1_gt_etf": results[best]["final_value"] > etf_net,
        "crit2_sharpe_gt_ew": results[best]["sharpe_net"] > ew_sharpe,
        "crit3_dsr": {
            "prob": float(dsr.deflated_sharpe_probability),
            "pass": bool(dsr.passes_5pct),
        },
        "crit4_windows": win,
        "crit5_maxdd_ok": results[best]["maxdd_net"] >= results["SPY_bh"]["maxdd"],
        "spy_correlation_info": round(spy_corr, 3),
        "PBO_info": float(cscv_pbo(pd.DataFrame(rets))),
    }
    n_win = sum(1 for w in win.values() if w["H029"] >= w["EW"])
    results["_verdict"]["crit4_pass"] = (
        n_win >= max(3, int(0.6 * len(win))) if win else False
    )
    results["_verdict"]["PASS"] = all(
        [
            results["_verdict"]["crit1_gt_etf"],
            results["_verdict"]["crit2_sharpe_gt_ew"],
            results["_verdict"]["crit3_dsr"]["pass"],
            results["_verdict"]["crit4_pass"],
            results["_verdict"]["crit5_maxdd_ok"],
        ]
    )
    (OUTD / "h029_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
