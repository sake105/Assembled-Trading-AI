"""H-031 — Insider-Patrone (§4.6.1, Registry Welle 9): opportunistische Kaeufe.

Cohen/Malloy/Pomorski: Routine = Kauf im SELBEN Kalendermonat in allen 3 Vorjahren
(PIT aus Vergangenheit); opportunistisch = Rest. Basket: Titel mit >=1 opportunist.
Kauf (available_at-gated) in den letzten 3 Monaten; Halten 12M (frisches Signal
verlaengert); EW-Slots, Cap 10 %, $1-Floor; volle Steuern inkl. Div-Drag.
Varianten: (a) alle P-Kaeufe; (b) Officer/Director & >=10k$. Fenster 2005-2026. N->98.
"""

from __future__ import annotations

import glob
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
N_TRIALS = 98
SIGNAL_START = pd.Timestamp("2005-01-01", tz="UTC")


def load_purchases() -> pd.DataFrame:
    frames = [
        pd.read_parquet(f)
        for f in glob.glob(str(DATA / "form4_broad" / "tranche_*.parquet"))
    ]
    frames.append(
        pd.read_parquet(
            ROOT / "data" / "raw" / "insider_congress" / "form4_insider_full.parquet"
        )
    )
    f = pd.concat(frames, ignore_index=True)
    f = f[(f["transaction_code"] == "P") & (~f["is_derivative"].astype(bool))]
    for c in ("transaction_date", "available_at"):
        f[c] = pd.to_datetime(f[c], utc=True, errors="coerce")
    f = f.dropna(
        subset=["transaction_date", "available_at", "symbol", "reporting_owner_cik"]
    )
    f = f[f["transaction_date"] <= pd.Timestamp.now(tz="UTC")]  # 2050-junk out
    f = f[f["transaction_date"] >= "1996-01-01"]
    return f


def classify_opportunistic(p: pd.DataFrame) -> pd.DataFrame:
    """Routine = same owner bought in the SAME calendar month in each of the
    3 prior years (PIT: uses only past purchases)."""
    key = p[["reporting_owner_cik", "transaction_date"]].copy()
    key["y"] = key["transaction_date"].dt.year
    key["m"] = key["transaction_date"].dt.month
    seen = set(zip(key["reporting_owner_cik"], key["y"], key["m"]))
    routine = [
        all((o, y - k, m) in seen for k in (1, 2, 3))
        for o, y, m in zip(key["reporting_owner_cik"], key["y"], key["m"])
    ]
    p = p.copy()
    p["routine"] = routine
    return p[~p["routine"]]


def run_insider(close, divp, sig_by_month, month_ends, *, label: str):
    pf = TaxedPortfolio(START_CAPITAL)
    pending, eq = [], []
    hold_until: dict[str, pd.Timestamp] = {}
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
                    hold_until.pop(sym, None)
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
        if t in sig_by_month:
            fresh = {
                s
                for s in sig_by_month[t]
                if s in close.columns
                and np.isfinite(px_t.get(s, np.nan))
                and px_t.get(s, 0.0) >= 1.0
            }
            for s in fresh:
                hold_until[s] = t + pd.DateOffset(months=12)
            held = set(pf.lots.keys())
            for sym in held:
                if hold_until.get(sym, t) < t:
                    pending.append(("sell_all", sym, 0.0))
            entries = [s for s in fresh if s not in held]
            basket_n = max(20, len(held) + len(entries))
            for sym in entries:
                pending.append(("trade_to", sym, min(1.0 / basket_n, 0.10) * v))
    e = pd.Series(dict(eq)).sort_index()
    e = e[e.index >= SIGNAL_START]
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

    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    month_ends = sorted(membership.index)

    p = load_purchases()
    print(
        f"[DATA] {len(p)} open-market P-buys, {p['symbol'].nunique()} symbols, {p['reporting_owner_cik'].nunique()} insiders",
        flush=True,
    )
    opp = classify_opportunistic(p)
    print(
        f"[SIG] opportunistic buys: {len(opp)} ({100 * len(opp) / len(p):.1f}%)",
        flush=True,
    )

    variants = {
        "H031_all_opp": opp,
        "H031_officer_10k": opp[
            opp["role"]
            .astype(str)
            .str.contains("officer|director", case=False, na=False)
            & (pd.to_numeric(opp["value_usd"], errors="coerce") >= 10000)
        ],
    }
    results, rets = {}, {}
    for name, df in variants.items():
        sig: dict[pd.Timestamp, set] = {}
        for me in month_ends:
            if me < SIGNAL_START:
                continue
            recent = df[
                (df["available_at"] <= me)
                & (df["available_at"] > me - pd.DateOffset(months=3))
            ]
            if len(recent):
                sig[me] = set(recent["symbol"])
        res, _eq, ret = run_insider(close, divp, sig, month_ends, label=name)
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} cagr={res['cagr_net'] * 100:.2f}% sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net'] * 100:.1f}% tax={res['tax_paid']:.0f}",
            flush=True,
        )

    # benchmarks same window
    spy = close["SPY"].dropna()
    spy = spy[spy.index >= SIGNAL_START]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_r = spy.pct_change().dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    results["SPY_bh"] = {
        "cagr_gross": float((spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(spy_r.mean() / spy_r.std() * np.sqrt(252)),
        "maxdd": float((spy / spy.cummax() - 1).min()),
    }
    results["ETF_net_path"] = {"final_value": float(etf_net)}
    _res_ew, _e, ret_ew = run_verdict(
        close, membership, label="EW_ref", mode="ew", div_panel=divp
    )
    ret_ew = ret_ew[ret_ew.index >= SIGNAL_START]
    ew_sharpe = float(ret_ew.mean() / ret_ew.std() * np.sqrt(252))
    results["EW_PIT_window_sharpe"] = ew_sharpe

    best = max(rets, key=lambda k: results[k]["final_value"])
    v = rets[best]
    dsr = deflated_sharpe(v, n_trials=N_TRIALS)
    win = {}
    for y0 in range(2005, 2026, 4):
        m = (v.index.year >= y0) & (v.index.year < y0 + 4)
        me = (ret_ew.index.year >= y0) & (ret_ew.index.year < y0 + 4)
        if m.sum() > 200 and me.sum() > 200:
            win[str(y0)] = {
                "H031": round(float(v[m].mean() / v[m].std() * np.sqrt(252)), 3),
                "EW": round(
                    float(ret_ew[me].mean() / ret_ew[me].std() * np.sqrt(252)), 3
                ),
            }
    n_win = sum(1 for w in win.values() if w["H031"] >= w["EW"])
    results["_verdict"] = {
        "selected": best,
        "crit1_gt_etf": results[best]["final_value"] > etf_net,
        "crit2_sharpe_gt_ew": results[best]["sharpe_net"] > ew_sharpe,
        "crit3_dsr": {
            "prob": float(dsr.deflated_sharpe_probability),
            "pass": bool(dsr.passes_5pct),
        },
        "crit4_windows": win,
        "crit4_pass": n_win >= max(3, int(0.6 * len(win))) if win else False,
        "crit5_maxdd_ok": results[best]["maxdd_net"] >= results["SPY_bh"]["maxdd"],
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
    (OUTD / "h031_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
