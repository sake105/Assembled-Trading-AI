"""H-040 (Low-Vol/BAB) + H-041 (Quality-Tilt) — netto-nach-Steuer, survivorship-frei.

Lokales PIT-S&P-Verdict-Universum (verdict_engine loaders). Reuse run_verdict mit
mode="ew" (EW-Band 50 %) auf einer SUB-Membership (Vol-Terzil bzw. ROE-Terzil je
Monatsende, PIT). Vergleich gegen full-S&P-EW-Band-Baseline + ETF-Netto-Pfad.

Div-Steuer AKTIV (div_panel): Low-Vol/High-Quality-Namen zahlen tendenziell MEHR
Dividende → Steuer-Drag ist gerade hier ehrlich einzurechnen (frisst einen Teil des
Low-Turnover-Vorteils).
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

import h011_kandidat_a as eng  # noqa: E402
from h011_kandidat_a import OUTD, START_CAPITAL, load_roe_panel  # noqa: E402
from verdict_engine import (  # noqa: E402
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)

ETF_TAX = 0.185


def sub_membership(
    membership: pd.Series, score: pd.DataFrame, *, pick: float, low_is_good: bool
) -> pd.Series:
    out: dict[pd.Timestamp, frozenset] = {}
    for me, members in membership.items():
        if me not in score.index:
            continue
        s = score.loc[me].reindex(list(members)).dropna()
        if len(s) < 15:
            continue
        pct = s.rank(pct=True)  # ascending: small value -> small pct
        sel = pct[pct <= pick].index if low_is_good else pct[pct >= (1 - pick)].index
        out[me] = frozenset(sel)
    return pd.Series(out)


def etf_and_spy(close: pd.DataFrame, r_idx) -> dict:
    spy = close["SPY"].dropna()
    spy = spy[(spy.index >= r_idx[0]) & (spy.index <= r_idx[-1])]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    sr = spy.pct_change().dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    return {
        "SPY_sharpe": float(sr.mean() / sr.std() * np.sqrt(252)),
        "ETF_net_final": float(etf_net),
        "years": float(years),
    }


def consistency(a: pd.Series, b: pd.Series, win: int = 504) -> float:
    idx = a.index.intersection(b.index)
    a, b = a.reindex(idx), b.reindex(idx)
    w = t = 0
    for st in range(0, len(idx) - win, win // 2):
        sl = slice(st, st + win)
        aa, bb = a.iloc[sl], b.iloc[sl]
        if aa.std() == 0 or bb.std() == 0:
            continue
        t += 1
        if aa.mean() / aa.std() >= bb.mean() / bb.std():
            w += 1
    return w / t if t else float("nan")


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    eng.COST_BPS = 10.0
    close = load_verdict_prices()
    membership = load_membership(close.index)
    div = load_div_panel(close.index)
    print(
        f"[DATA] {close.shape[1]} names, {len(membership)} rebalances, "
        f"{close.index[0].date()} -> {close.index[-1].date()}",
        flush=True,
    )

    results: dict = {}
    rets: dict = {}

    # full-S&P EW-band baseline
    res_base, eq_base, ret_base = run_verdict(
        close,
        membership,
        label="EWband_fullSP",
        mode="ew",
        ew_band=0.5,
        div_panel=div,
    )
    results["EWband_fullSP"] = res_base
    print(
        f"[RUN] EWband_fullSP: final={res_base['final_value']:.0f} "
        f"sharpe={res_base['sharpe_net']:.3f} maxdd={res_base['maxdd_net']:.3f}",
        flush=True,
    )

    # ---- H-040: Low-Vol terzile family ----
    vol63 = close.pct_change(fill_method=None).rolling(63).std()
    for pk in (0.20, 0.33, 0.50):
        mem = sub_membership(membership, vol63, pick=pk, low_is_good=True)
        lab = f"H040_lowvol_{int(pk * 100)}"
        res, eq, ret = run_verdict(
            close, mem, label=lab, mode="ew", ew_band=0.5, div_panel=div
        )
        results[lab] = res
        rets[lab] = ret
        print(
            f"[RUN] {lab}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} "
            f"maxdd={res['maxdd_net']:.3f}",
            flush=True,
        )

    # ---- H-041: Quality (ROE) terzile ----
    universe = [c for c in close.columns if c != "SPY"]
    roe = load_roe_panel(close.index, universe)
    print(
        f"[ROE] coverage median {roe.notna().sum(axis=1).median():.0f} symbols/month",
        flush=True,
    )
    for pk in (0.33, 0.50):
        mem = sub_membership(membership, roe, pick=pk, low_is_good=False)
        lab = f"H041_quality_{int(pk * 100)}"
        res, eq, ret = run_verdict(
            close, mem, label=lab, mode="ew", ew_band=0.5, div_panel=div
        )
        results[lab] = res
        rets[lab] = ret
        print(
            f"[RUN] {lab}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} "
            f"maxdd={res['maxdd_net']:.3f}",
            flush=True,
        )

    # ---- verdict variants ----
    v40_lab = "H040_lowvol_33"
    v40, v40r = results[v40_lab], rets[v40_lab]
    v41_lab = "H041_quality_33"
    v41, v41r = results[v41_lab], rets[v41_lab]

    bench = etf_and_spy(close, v40r.index)
    results["benchmarks"] = bench

    dsr40 = deflated_sharpe(v40r, n_trials=114)
    dsr41 = deflated_sharpe(v41r, n_trials=116)
    results["DSR_H040"] = {
        "prob": float(dsr40.deflated_sharpe_probability),
        "passes_5pct": bool(dsr40.passes_5pct),
        "n_trials": 114,
    }
    results["DSR_H041"] = {
        "prob": float(dsr41.deflated_sharpe_probability),
        "passes_5pct": bool(dsr41.passes_5pct),
        "n_trials": 116,
    }

    c40 = consistency(v40r, ret_base)

    h040 = {
        "1_sharpe_gt_base": bool(v40["sharpe_net"] > res_base["sharpe_net"]),
        "2_final_gt_ETF": bool(v40["final_value"] > bench["ETF_net_final"]),
        "3_DSR_passes": bool(dsr40.passes_5pct),
        "4_maxdd_better": bool(v40["maxdd_net"] > res_base["maxdd_net"]),
        "5_consistency_ge_60": bool(c40 >= 0.60),
    }
    results["H040_criteria"] = h040
    results["H040_PASS"] = bool(all(h040.values()))

    h041 = {
        "sharpe_gt_base_x105": bool(v41["sharpe_net"] > res_base["sharpe_net"] * 1.05),
        "final_gt_base_x105": bool(v41["final_value"] > res_base["final_value"] * 1.05),
        "final_gt_ETF": bool(v41["final_value"] > bench["ETF_net_final"]),
    }
    results["H041_criteria"] = h041
    results["H041_PASS"] = bool(all(h041.values()))

    (OUTD / "h040_h041_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]",
        json.dumps(
            {
                "base_final": res_base["final_value"],
                "base_sharpe": res_base["sharpe_net"],
                "base_maxdd": res_base["maxdd_net"],
                "H040_PASS": results["H040_PASS"],
                "H040": h040,
                "v40_final": v40["final_value"],
                "v40_sharpe": v40["sharpe_net"],
                "v40_maxdd": v40["maxdd_net"],
                "H041_PASS": results["H041_PASS"],
                "H041": h041,
                "v41_final": v41["final_value"],
                "v41_sharpe": v41["sharpe_net"],
                "ETF_net": bench["ETF_net_final"],
                "SPY_sharpe": bench["SPY_sharpe"],
                "DSR40_p": results["DSR_H040"]["prob"],
                "DSR41_p": results["DSR_H041"]["prob"],
                "consistency40": c40,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
