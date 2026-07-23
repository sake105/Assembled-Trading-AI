"""H-030 — Confirmatory 13F-Konsens: 6 Parameter-Nachbarn + Beta-Kontrolle.

Registry Welle 8 (N->94). Original (Top10,K10,M100) ausgeschlossen.
Beta-Kontrolle: VALUE-gewichteter Basket ALLER Top-100-Holdings (13F-Markt-
Portfolio), gleiche Steuer-/Timing-Engine — kein Trial.
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
from h029_13f_consensus import TAX, build_consensus, run_consensus  # noqa: E402
from verdict_engine import (  # noqa: E402
    DATA,
    load_div_panel,
    load_membership,
    load_verdict_prices,
)

ETF_TAX = 0.185
EW_WINDOW_SHARPE = 0.55


def run_beta_control(close, divp, month_ends):
    """VALUE-weighted basket of ALL top-100-manager holdings, quarterly."""
    h = pd.read_parquet(
        DATA / "13f_top100.parquet",
        columns=["ACCESSION_NUMBER", "CUSIP", "VALUE", "PERIODOFREPORT"],
    )
    cm = pd.read_parquet(DATA / "cusip_ticker_map.parquet")
    cm = cm.sort_values("year").drop_duplicates("CUSIP", keep="last")[
        ["CUSIP", "SYMBOL"]
    ]
    g = h.groupby(["PERIODOFREPORT", "CUSIP"], as_index=False)["VALUE"].sum()
    g = g.merge(cm, on="CUSIP", how="left").dropna(subset=["SYMBOL"])
    g["period_end"] = pd.to_datetime(
        g["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce"
    )
    bad = g["period_end"].isna()
    if bad.any():
        g.loc[bad, "period_end"] = pd.to_datetime(
            g.loc[bad, "PERIODOFREPORT"], errors="coerce"
        )
    g["period_end"] = pd.to_datetime(g["period_end"], utc=True)
    g = g.dropna(subset=["period_end"])
    # weights per period (top 200 by value to keep it tradable)
    sig = {}
    for pe, grp in g.groupby("period_end"):
        cand = [t for t in month_ends if t >= pe + pd.DateOffset(months=2)]
        if not cand:
            continue
        eff = min(cand)
        top = grp.nlargest(200, "VALUE")
        w = top.set_index("SYMBOL")["VALUE"]
        sig[eff] = (w / w.sum()).to_dict()

    pf = TaxedPortfolio(START_CAPITAL)
    pending, eq = [], []
    close_ff = close.ffill()
    last_valid = close.apply(lambda s: s.last_valid_index())
    global_last = close.index[-1]
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
                elif delta < -1.0:
                    pf.sell(sym, -delta / px, float(px))
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
            w = sig[t]
            held = set(pf.lots.keys())
            for sym in held - set(w):
                pending.append(("sell_all", sym, 0.0))
            for sym, wi in w.items():
                if (
                    sym in close.columns
                    and np.isfinite(px_t.get(sym, np.nan))
                    and px_t.get(sym, 0) >= 1.0
                ):
                    pending.append(("trade_to", sym, wi * v))
    e = pd.Series(dict(eq)).sort_index()
    first_sig = min(sig) if sig else e.index[0]
    e = e[e.index >= first_sig]
    ret = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    return {
        "final_value": float(e.iloc[-1] / e.iloc[0] * START_CAPITAL),
        "cagr_net": float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1),
        "sharpe_net": float(ret.mean() / ret.std() * np.sqrt(252)),
    }


def main() -> int:
    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    month_ends = sorted(membership.index)

    fam = [
        ("H030_top8", dict(top_pos=8, m_top=100), 10),
        ("H030_top12", dict(top_pos=12, m_top=100), 10),
        ("H030_k7", dict(top_pos=10, m_top=100), 7),
        ("H030_k13", dict(top_pos=10, m_top=100), 13),
        ("H030_m50", dict(top_pos=10, m_top=50), 10),
        ("H030_m150", dict(top_pos=10, m_top=150), 10),
    ]
    results, rets = {}, {}
    for name, ckw, k in fam:
        cons = build_consensus(**ckw)
        res, _eq, ret = run_consensus(close, divp, cons, month_ends, k=k, label=name)
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net'] * 100:.1f}%",
            flush=True,
        )

    print("[RUN] beta control (13F market portfolio) ...", flush=True)
    beta = run_beta_control(close, divp, month_ends)
    results["beta_control"] = beta
    print(f"      {beta}", flush=True)

    # same-window ETF path
    start = rets["H030_top8"].index[0]
    spy = close["SPY"].dropna()
    spy = spy[spy.index >= start]
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    results["ETF_net_path"] = {"final_value": float(etf_net)}

    finals = [results[n]["final_value"] for n, _, _ in fam]
    crit1 = all(f > etf_net for f in finals)
    crit2 = float(np.median(finals)) > beta["final_value"]
    pbo = float(cscv_pbo(pd.DataFrame(rets)))
    crit3 = pbo <= 0.5
    crit4 = (
        sum(1 for n, _, _ in fam if results[n]["sharpe_net"] > EW_WINDOW_SHARPE) >= 4
    )
    results["_verdict"] = {
        "crit1_all_gt_etf": crit1,
        "crit2_median_gt_beta": crit2,
        "median_final": float(np.median(finals)),
        "crit3_pbo": {"value": pbo, "pass": crit3},
        "crit4_sharpe_4of6": crit4,
        "PASS": all([crit1, crit2, crit3, crit4]),
    }
    (OUTD / "h030_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
