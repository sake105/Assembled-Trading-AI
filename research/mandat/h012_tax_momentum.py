"""H-012 — Steuer-/Turnover-optimiertes Pure-Momentum (Registry: research/registry.md).

EXPLORATIV (gleiche Datenlage wie H-011). Familie: rank_out in {60, 80, 100},
Kauf Top 20, kein Quality, kein Gate, kein ATR-Backstop, no_retrim (Positionen
laufen bis Rang-Exit). Auswahl: bestes Netto-Endvermögen; Verdict auf der
gewählten Variante bei N=47; PBO über die 3er-Familie.
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

from h011_kandidat_a import (  # noqa: E402
    OUTD,
    START_CAPITAL,
    cscv_pbo,
    load_prices,
    run_variant,
)

ETF_TAX = 0.185
N_TRIALS = 47


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close, high, low = load_prices()
    close = close.dropna(axis=1, thresh=int(len(close) * 0.5))
    high, low = high[close.columns], low[close.columns]
    print(
        f"[DATA] {close.shape[1] - 1} symbols, {close.index[0].date()} -> {close.index[-1].date()}",
        flush=True,
    )

    results, rets = {}, {}
    for rank_out in (60, 80, 100):
        name = f"H012_out{rank_out}"
        print(f"[RUN] {name} ...", flush=True)
        res, eq, ret = run_variant(
            close,
            high,
            low,
            None,
            use_quality=False,
            use_gate=False,
            top_out=rank_out,
            label=name,
            use_atr_backstop=False,
            no_retrim=True,
        )
        results[name] = res
        rets[name] = ret
        print(f"      {res}", flush=True)

    # benchmarks (identical convention to H-011)
    spy = close["SPY"].dropna()
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    gross_gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net_final = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    results["ETF_net_path"] = {
        "final_value": float(etf_net_final),
        "cagr_net": float((etf_net_final / START_CAPITAL) ** (1 / years) - 1),
    }

    best = max((k for k in rets), key=lambda k: results[k]["final_value"])
    dsr = deflated_sharpe(rets[best], n_trials=N_TRIALS)
    results["selected"] = best
    results["DSR_selected"] = {
        "probability": float(dsr.deflated_sharpe_probability),
        "passes_5pct": bool(dsr.passes_5pct),
        "n_trials": N_TRIALS,
    }
    results["PBO_CSCV_3variants"] = float(cscv_pbo(pd.DataFrame(rets)))

    # sub-periods vs EW baseline (reuse stored H-011 baseline returns via rerun-free
    # comparison: EW Sharpe per window from h011_results.json is not stored as a
    # series, so recompute the EW baseline once here for the window comparison)
    print("[RUN] EW baseline (control, benchmark — not a trial) ...", flush=True)
    _, _, ret_ew = run_variant(
        close,
        high,
        low,
        None,
        use_quality=False,
        use_gate=False,
        top_out=40,
        label="EW_baseline",
        ew_baseline=True,
    )
    v = rets[best]
    win = {}
    for y0 in range(v.index[0].year, v.index[-1].year, 2):
        m = (v.index.year >= y0) & (v.index.year < y0 + 2)
        me = (ret_ew.index.year >= y0) & (ret_ew.index.year < y0 + 2)
        if m.sum() > 100:
            s1 = v[m].mean() / v[m].std() * np.sqrt(252)
            s2 = ret_ew[me].mean() / ret_ew[me].std() * np.sqrt(252)
            win[f"{y0}-{y0 + 1}"] = {
                "best": round(float(s1), 3),
                "EW": round(float(s2), 3),
            }
    results["subperiods_best_vs_EW"] = win
    ew_ann = ret_ew.mean() / ret_ew.std() * np.sqrt(252)
    results["EW_baseline_sharpe"] = float(ew_ann)

    out_path = OUTD / "h012_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] -> {out_path}", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
