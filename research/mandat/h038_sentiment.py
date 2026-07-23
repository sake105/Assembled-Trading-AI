"""H-038 — News-Sentiment-Tilt (Registry Welle 13, Sperrlisten-Override Hans).

Signal: trailing-21T-Mittel EODHD `normalized` je Symbol, am Monatsultimo (PIT);
Ausfuehrung T+1-Close (Engine). Top-30-Basket, no-retrim, Exit Rang>60.
3 Kostenstufen 5/10/20 bps (COST_BPS gepatcht). Volle Steuern inkl. Div-Drag.
Testet exakt die Sperr-Behauptung „Kollaps bei ~10 bps". N->112 (Ledger fuehrt Reihenfolge).
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

import h011_kandidat_a as base  # noqa: E402
from h011_kandidat_a import OUTD, START_CAPITAL, cscv_pbo  # noqa: E402
from verdict_engine import (  # noqa: E402
    DATA,
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)

ETF_TAX = 0.185
N_TRIALS = 112
SIGNAL_START = pd.Timestamp("2013-01-01", tz="UTC")


def build_score(close, membership):
    s = pd.read_parquet(DATA / "sentiment.parquet")
    piv = s.pivot_table(
        index="date", columns="symbol", values="normalized", aggfunc="mean"
    )
    piv = piv.reindex(close.index).ffill(limit=5)
    roll = piv.rolling(21, min_periods=5).mean()
    month_ends = [t for t in membership.index if t >= SIGNAL_START and t in roll.index]
    return roll.loc[month_ends]


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    score = build_score(close, membership)
    cov = score.notna().sum(axis=1)
    print(
        f"[SIG] sentiment score panel: {len(score)} months, median {cov.median():.0f} symbols/month",
        flush=True,
    )

    results, rets = {}, {}
    orig_cost = base.COST_BPS
    for bps in (5, 10, 20):
        base.COST_BPS = float(bps)
        name = f"H038_sent_{bps}bps"
        res, _eq, ret = run_verdict(
            close,
            membership,
            label=name,
            mode="momentum",
            top_in=30,
            top_out=60,
            div_panel=divp,
            score_panel=score,
        )
        ret = ret[ret.index >= SIGNAL_START]
        eqr = (1 + ret).cumprod()
        res["sharpe_window"] = float(ret.mean() / ret.std() * np.sqrt(252))
        res["final_window"] = float(eqr.iloc[-1] * START_CAPITAL)
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_window']:.0f} sharpe={res['sharpe_window']:.3f}",
            flush=True,
        )
    base.COST_BPS = orig_cost

    spy = close["SPY"].dropna()
    spy = spy[spy.index >= SIGNAL_START]
    spy_r = spy.pct_change().dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    results["ETF_net_path"] = {"final_value": float(etf_net)}
    _r, _e, ret_ew = run_verdict(
        close, membership, label="EW_ref", mode="ew", div_panel=divp
    )
    ret_ew = ret_ew[ret_ew.index >= SIGNAL_START]
    ew_sharpe = float(ret_ew.mean() / ret_ew.std() * np.sqrt(252))
    results["EW_window_sharpe"] = ew_sharpe

    r10 = rets["H038_sent_10bps"]
    dsr = deflated_sharpe(r10, n_trials=N_TRIALS)
    win = {}
    for y0 in range(2013, 2026, 2):
        m = (r10.index.year >= y0) & (r10.index.year < y0 + 2)
        me = (ret_ew.index.year >= y0) & (ret_ew.index.year < y0 + 2)
        if m.sum() > 100 and me.sum() > 100:
            win[str(y0)] = {
                "H038": round(float(r10[m].mean() / r10[m].std() * np.sqrt(252)), 3),
                "EW": round(
                    float(ret_ew[me].mean() / ret_ew[me].std() * np.sqrt(252)), 3
                ),
            }
    n_win = sum(1 for w in win.values() if w["H038"] >= w["EW"])
    results["_verdict"] = {
        "crit1_10bps_gt_etf": results["H038_sent_10bps"]["final_window"] > etf_net,
        "crit2_10bps_sharpe_gt_ew": results["H038_sent_10bps"]["sharpe_window"]
        > ew_sharpe,
        "crit3_dsr": {
            "prob": float(dsr.deflated_sharpe_probability),
            "pass": bool(dsr.passes_5pct),
        },
        "crit4_windows": win,
        "crit4_pass": n_win >= max(3, int(0.6 * len(win))) if win else False,
        "crit5_survives_20bps": results["H038_sent_20bps"]["sharpe_window"] > ew_sharpe,
        "sharpe_by_cost": {
            b: round(results[f"H038_sent_{b}bps"]["sharpe_window"], 3)
            for b in (5, 10, 20)
        },
        "spy_corr": round(float(r10.corr(spy_r.reindex(r10.index))), 3),
        "PBO_info": float(cscv_pbo(pd.DataFrame(rets))),
    }
    results["_verdict"]["PASS"] = all(
        [
            results["_verdict"]["crit1_10bps_gt_etf"],
            results["_verdict"]["crit2_10bps_sharpe_gt_ew"],
            results["_verdict"]["crit3_dsr"]["pass"],
            results["_verdict"]["crit4_pass"],
            results["_verdict"]["crit5_survives_20bps"],
        ]
    )
    (OUTD / "h038_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
