"""Welle 28 — H-062 Covered Calls mit ECHTER VIX-IV, H-063 EUR-Realität, H-064 Faktor-ETFs.

H-062: wie h046, aber IV = VIX(Monatsstart)/100 × (1 − Skew-Haircut). Prämie sofort 26,375 %.
H-063: SPY/GLD in EUR (EURUSD), Kernvergleiche neu — kippt FX ein Verdict?
H-064: Faktor-ETFs vs SPY, gleiches Fenster, Buy&Hold, Terminal-Steuer 18,46 % beidseitig.
"""

from __future__ import annotations

import json
import sys
from math import erf, exp, log, sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
ETF_TAX, TAX = 0.1846, 0.26375
R = 0.02
OPT_COST_BPS = 3.0


def phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S, K, T, sig):
    if sig <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    return S * phi(d1) - K * exp(-R * T) * phi(d1 - sig * sqrt(T))


def load_w28():
    w = pd.read_parquet(DATA / "prices_w28.parquet")
    px = w.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    px.index = pd.DatetimeIndex(px.index)
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    px["SPY"] = pd.Series(spy.values, index=pd.DatetimeIndex(spy.index)).reindex(
        px.index
    )
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    gld = pc[pc["symbol"] == "GLD"].set_index("timestamp")["close"]
    px["GLD"] = pd.Series(gld.values, index=pd.DatetimeIndex(gld.index)).reindex(
        px.index
    )
    return px


def covered_call_vix(px, *, otm, skew_hc):
    d = px[["SPY", "VIX"]].dropna()
    me = d.groupby(d.index.to_period("M")).tail(1)
    idx = me.index
    overlay, T = 0.0, 21.0 / 252.0
    v0 = START
    for i in range(len(idx) - 1):
        S0, S1 = float(me["SPY"].iloc[i]), float(me["SPY"].iloc[i + 1])
        iv = float(me["VIX"].iloc[i]) / 100.0 * (1 - skew_hc)
        K = S0 * (1 + otm)
        v_stock = v0 * S0 / float(me["SPY"].iloc[0])
        sh = v_stock / S0
        pnl = (
            sh * (bs_call(S0, K, T, iv) - max(S1 - K, 0.0))
            - v_stock * OPT_COST_BPS / 1e4
        )
        overlay += pnl * (1 - TAX) if pnl > 0 else pnl
    S_first, S_last = float(me["SPY"].iloc[0]), float(me["SPY"].iloc[-1])
    stock_net = START + (START * S_last / S_first - START) * (1 - ETF_TAX)
    return {
        "overlay_net": round(overlay),
        "combined": round(stock_net + overlay),
        "buyhold_net": round(stock_net),
    }


def eur_reality(px):
    d = px[["SPY", "GLD", "EURUSD"]].dropna()
    spy_eur = d["SPY"] / d["EURUSD"]
    gld_eur = d["GLD"] / d["EURUSD"]
    years = (d.index[-1] - d.index[0]).days / 365.25

    def stats(series, tax):
        g = START * (series.iloc[-1] / series.iloc[0])
        net = START + (g - START) * (1 - tax)
        r = series.pct_change().dropna()
        return {
            "net": round(net),
            "cagr": round(((net / START) ** (1 / years) - 1) * 100, 2),
            "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
            "maxdd": round(float((series / series.cummax() - 1).min()), 3),
        }

    def blend_stats(w_eq):
        n = w_eq * START * (spy_eur.iloc[-1] / spy_eur.iloc[0])
        ng = (1 - w_eq) * START * (gld_eur.iloc[-1] / gld_eur.iloc[0])
        net = (n - w_eq * START) * (1 - ETF_TAX) + w_eq * START + ng  # gold §23 0 %
        port = w_eq * spy_eur / spy_eur.iloc[0] + (1 - w_eq) * gld_eur / gld_eur.iloc[0]
        r = port.pct_change().dropna()
        return {
            "net": round(net),
            "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
            "maxdd": round(float((port / port.cummax() - 1).min()), 3),
        }

    return {
        "window": [str(d.index[0].date()), str(d.index[-1].date())],
        "SPY_in_USD": stats(d["SPY"], ETF_TAX),
        "SPY_in_EUR": stats(spy_eur, ETF_TAX),
        "GLD_in_EUR": stats(gld_eur, 0.0),
        "70/30_in_EUR": blend_stats(0.7),
        "100SPY_in_EUR": blend_stats(1.0),
    }


def factor_etfs(px):
    out = {}
    for f in ("MTUM", "USMV", "QUAL", "VLUE", "SCHD", "NOBL", "SPMO"):
        d = px[[f, "SPY"]].dropna()
        years = (d.index[-1] - d.index[0]).days / 365.25
        res = {}
        for c in (f, "SPY"):
            g = START * (d[c].iloc[-1] / d[c].iloc[0])
            net = START + (g - START) * (1 - ETF_TAX)
            r = d[c].pct_change().dropna()
            res[c] = {
                "net": round(net),
                "cagr": round(((net / START) ** (1 / years) - 1) * 100, 2),
                "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
                "maxdd": round(float((d[c] / d[c].cummax() - 1).min()), 3),
            }
        ex = d[f].pct_change().dropna() - d["SPY"].pct_change().dropna()
        t = (
            float(ex.mean() / (ex.std() / sqrt(len(ex))))
            if ex.std() > 0
            else float("nan")
        )
        out[f] = {
            "years": round(years, 1),
            **res,
            "excess_t": round(t, 2),
            "beats": bool(
                res[f]["net"] > res["SPY"]["net"]
                and res[f]["sharpe"] > res["SPY"]["sharpe"]
                and t > 2
            ),
        }
    return out


def main() -> int:
    px = load_w28()
    r62 = {}
    for otm in (0.0, 0.03, 0.05):
        for hc in (0.0, 0.10, 0.20):
            key = f"otm{int(otm * 100)}_skew{int(hc * 100)}"
            r62[key] = covered_call_vix(px, otm=otm, skew_hc=hc)
            print(f"[H062] {key}: {r62[key]}", flush=True)
    r63 = eur_reality(px)
    print("[H063]", json.dumps(r63, indent=2), flush=True)
    r64 = factor_etfs(px)
    for k, v in r64.items():
        print(
            f"[H064] {k}: {f'{k}='}{v[k]['net']:,} SPY={v['SPY']['net']:,} t={v['excess_t']} beats={v['beats']}",
            flush=True,
        )
    out = {"H062_vix_covered_call": r62, "H063_eur": r63, "H064_factor_etfs": r64}
    (OUTD / "h062_h064_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
