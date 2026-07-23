"""Welle 4 — H-023/024/025: erste VERDICT-fähige Tests (Registry, VOR Lauf registriert).

Survivorship-freies PIT-Universum (EODHD + Konstituenten-Historie), 1997-2026.
8 Läufe + EW-PIT-Baseline + SPY/ETF-Pfad-Benchmarks. N→75.
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
from verdict_engine import load_membership, load_verdict_prices, run_verdict  # noqa: E402

ETF_TAX = 0.185


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_verdict_prices()
    membership = load_membership(close.index)
    print(
        f"[DATA] {close.shape[1]} symbols, {close.index[0].date()} -> {close.index[-1].date()}, {len(membership)} month-ends",
        flush=True,
    )

    results, rets = {}, {}

    def go(name, fam, **kw):
        res, _eq, ret = run_verdict(close, membership, label=name, **kw)
        results[name] = res
        rets.setdefault(fam, {})[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} cagr={res['cagr_net'] * 100:.2f}% sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net'] * 100:.1f}% tax={res['tax_paid']:.0f}",
            flush=True,
        )

    # benchmarks / baseline (not trials)
    go("EW_PIT_monthly", "_bench", mode="ew")
    # H-023 momentum family
    for out in (60, 80, 100):
        go(f"H023_out{out}", "H023", mode="momentum", top_out=out)
    # H-024 EW bands
    for b, nm in ((0.25, "H024_band25"), (0.50, "H024_band50")):
        go(nm, "H024", mode="ew", ew_band=b)
    # H-025 gates on out60
    for gm in ("sma", "vol", "both"):
        go(
            f"H025_gate_{gm}",
            "H025",
            mode="momentum",
            top_out=60,
            use_gate=True,
            gate_mode=gm,
        )

    # SPY benchmarks on same trimmed window
    spy = close["SPY"].dropna()
    spy = spy[spy.index >= close.index[0] + pd.Timedelta(days=400)]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_r = spy.pct_change().dropna()
    gross_gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net_final = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    results["SPY_gross"] = {
        "cagr": float((spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(spy_r.mean() / spy_r.std() * np.sqrt(252)),
        "maxdd": float((spy / spy.cummax() - 1).min()),
    }
    results["ETF_net_path"] = {
        "final_value": float(etf_net_final),
        "cagr_net": float((etf_net_final / START_CAPITAL) ** (1 / years) - 1),
    }

    # family metrics
    fam_n = {"H023": 70, "H024": 72, "H025": 75}
    for fam, n in fam_n.items():
        rr = rets[fam]
        best = max(rr, key=lambda k: results[k]["final_value"])
        dsr = deflated_sharpe(rr[best], n_trials=n)
        results[f"_family_{fam}"] = {
            "selected": best,
            "DSR_prob": float(dsr.deflated_sharpe_probability),
            "DSR_passes": bool(dsr.passes_5pct),
            "n_trials": n,
            "PBO_CSCV": float(cscv_pbo(pd.DataFrame(rr))),
        }

    # sub-period consistency: 4y windows, selected H023 vs EW_PIT
    v = rets["H023"][results["_family_H023"]["selected"]]
    ew = rets["_bench"]["EW_PIT_monthly"]
    win = {}
    for y0 in range(v.index[0].year, v.index[-1].year, 4):
        m = (v.index.year >= y0) & (v.index.year < y0 + 4)
        me = (ew.index.year >= y0) & (ew.index.year < y0 + 4)
        if m.sum() > 200:
            win[f"{y0}-{y0 + 3}"] = {
                "H023": round(float(v[m].mean() / v[m].std() * np.sqrt(252)), 3),
                "EW_PIT": round(float(ew[me].mean() / ew[me].std() * np.sqrt(252)), 3),
            }
    results["subperiods_H023_vs_EWPIT"] = win

    out = OUTD / "welle4_results.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
