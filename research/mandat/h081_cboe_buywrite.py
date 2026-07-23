"""H-081 — Stillhalter-Verdict via ECHTER CBOE-Historie (BXM/BXMD/PUT vs SPX-TR).

(1) Brutto: CAGR/Sharpe/MaxDD + Dekaden-Tabelle (Regimeabhängigkeit).
(2) Deutsches Steuer-Overlay (Approximation, benannt): Monats-Options-P&L ≈ idx_ret − spxtr_ret;
    positive Options-Monate sofort 26,375 % (Stillhalter), Aktien-Bein 18,46 % terminal auf
    SPX-TR-Gewinn; 5 bps/Mo Implementierung. Vergleich vs ETF-Pfad (SPX-TR × 18,46 % terminal).
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
TAX_OPT, TAX_ETF = 0.26375, 0.1846
COST_M = 5e-4


def monthly(s: pd.Series) -> pd.Series:
    m = s.resample(
        "ME"
    ).last()  # Kalendermonate; Lücken werden NaN (kein Fake-Multi-Monats-Return)
    return m


def stats(ret: pd.Series, label: str) -> dict:
    ret = ret.dropna()
    e = (1 + ret).cumprod()
    years = (ret.index[-1] - ret.index[0]).days / 365.25  # Kalender, nicht len/12
    return {
        "label": label,
        "cagr_pct": round((float(e.iloc[-1]) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(ret.mean() / ret.std() * sqrt(12)), 3),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "years": round(years, 1),
        "n_months": int(len(ret)),
    }


def german_overlay(idx_m: pd.Series, spx_m: pd.Series) -> dict:
    ri = idx_m.pct_change().dropna()
    rs = spx_m.pct_change().reindex(ri.index)
    V_opt = 0.0  # kumulierte Netto-Options-Cash (Prämiensaldo)
    pot = 0.0
    tax_paid = 0.0
    # Options-P&L wird auf das (mitwachsende) Aktien-Bein skaliert
    eq_path = (1 + rs).cumprod() * START
    eq_prev = START
    for t, r_i in ri.items():
        opt_pnl = (r_i - float(rs.loc[t])) * eq_prev - COST_M * eq_prev
        if opt_pnl > 0:
            off = min(opt_pnl, pot)
            pot -= off
            tx = (opt_pnl - off) * TAX_OPT
            tax_paid += tx
            V_opt += opt_pnl - tx
        else:
            pot += -opt_pnl
            V_opt += opt_pnl
        eq_prev = float(eq_path.loc[t])
    eq_final = float(eq_path.iloc[-1])
    eq_net = START + (eq_final - START) * (1 - TAX_ETF)
    etf_net = eq_net  # ETF-Pfad = dasselbe Aktien-Bein ohne Overlay
    combined = eq_net + V_opt
    return {
        "combined_net": round(combined),
        "ETF_net": round(etf_net),
        "overlay_contribution": round(V_opt),
        "opt_tax_paid": round(tax_paid),
        "beats_ETF": bool(combined > etf_net),
    }


def main() -> int:
    px = pd.read_parquet(DATA / "prices_cboe_buywrite.parquet")
    w = px.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    w.index = pd.DatetimeIndex(w.index).tz_localize(None)
    out = {}
    for name in ("BXMD", "PUT", "BXM"):
        d = w[[name, "SP500TR"]].dropna()
        im, sm = monthly(d[name]), monthly(d["SP500TR"])
        ri, rs = im.pct_change(fill_method=None), sm.pct_change(fill_method=None)
        both = ri.notna() & rs.notna()
        ri, rs = ri[both], rs[both]
        # Daten-Dichte-Check: nur zusammenhängende Ära nutzen (>= 90 % Monats-Abdeckung ab Start)
        span_m = (ri.index[-1].year - ri.index[0].year) * 12 + (
            ri.index[-1].month - ri.index[0].month
        )
        if len(ri) < 0.9 * span_m:
            # schneide auf die dichte jüngere Ära: erster Index ab dem 24 Folge-Monate lückenlos sind
            gaps = ri.index.to_series().diff().dt.days > 35
            last_gap = ri.index[gaps][-1] if gaps.any() else ri.index[0]
            ri, rs = ri[ri.index >= last_gap], rs[rs.index >= last_gap]
            print(
                f"  [{name}] Frühbereich lückenhaft -> Fenster ab {ri.index[0].date()} ({len(ri)} Monate)",
                flush=True,
            )
        res = {
            "gross_idx": stats(ri, name),
            "gross_spxtr": stats(rs, "SPXTR"),
            "excess_cagr_pp": round(
                stats(ri, "")["cagr_pct"] - stats(rs, "")["cagr_pct"], 2
            ),
        }
        dec = {}
        for d0 in (1990, 2000, 2010, 2020):
            m = (ri.index.year >= d0) & (ri.index.year < d0 + 10)
            if m.sum() > 24:
                dec[f"{d0}s"] = {
                    "idx_cagr": round(
                        (float((1 + ri[m]).prod()) ** (12 / m.sum()) - 1) * 100, 2
                    ),
                    "spx_cagr": round(
                        (float((1 + rs[m]).prod()) ** (12 / m.sum()) - 1) * 100, 2
                    ),
                }
        res["decades"] = dec
        res["german_tax_overlay"] = german_overlay(
            im.loc[ri.index[0] :], sm.loc[ri.index[0] :]
        )
        out[name] = res
        print(
            f"\n[{name}] {res['gross_idx']} vs SPXTR {res['gross_spxtr']}", flush=True
        )
        print(f"  Dekaden: {json.dumps(dec)}", flush=True)
        print(f"  DE-Steuer: {json.dumps(res['german_tax_overlay'])}", flush=True)
    (OUTD / "h081_cboe_results.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("\n[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
