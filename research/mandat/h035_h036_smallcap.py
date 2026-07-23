"""H-035 (Small-Cap-Momentum netto) + H-036 (Size-Praemie via EW-Band-Mechanik).

Survivorship-freies Small-Cap-Universum (smallcap_data). Reuse run_verdict
(verdict_engine) mit Band-Membership. Kosten 30 bps/Seite (Small-Cap-realistisch).

H-035 Pass/Fail (ALLE): (1) > ETF-Pfad; (2) Sharpe > EW-Band-Kontrolle (SELBES
Band); (3) DSR passes; (4) >=60% 2J-Fenster Momentum-Sharpe >= EW-Band;
(5) MaxDD nicht schlechter als EW-Band.
H-036 Pass/Fail: Small(EW-Band) > Large(EW-Band) * 1.10 UND Small > ETF-Pfad.

Div-Steuer: dividends.parquet deckt nur S&P-Namen, nicht das Small-Cap-Universum
-> hier KEIN div_panel. adjusted_close reinvestiert Brutto-Div; das Weglassen der
Div-Steuer ueberschaetzt Small-Cap-Netto MINIMAL (Small Caps zahlen wenig/keine
Div) und wirkt PRO Strategie -> ein FAIL ist dadurch nur staerker, ein PASS
braucht den Caveat. Dokumentiert im Ledger.
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
from h011_kandidat_a import OUTD, START_CAPITAL, cscv_pbo  # noqa: E402
from smallcap_data import band_membership, load_smallcap  # noqa: E402
from verdict_engine import run_verdict  # noqa: E402

ETF_TAX = 0.185
N_LEDGER_AFTER = 114  # N=111 vor H-035; +3 Momentum-Laeufe


def etf_and_spy(close: pd.DataFrame, win_start, win_end) -> dict:
    spy = close["SPY"].dropna()
    spy = spy[(spy.index >= win_start) & (spy.index <= win_end)]
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    sr = spy.pct_change().dropna()
    gross_gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    return {
        "SPY_sharpe": float(sr.mean() / sr.std() * np.sqrt(252)),
        "SPY_cagr": float((spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1),
        "ETF_net_final": float(etf_net),
        "ETF_net_cagr": float((etf_net / START_CAPITAL) ** (1 / years) - 1),
        "years": float(years),
    }


def window_consistency(
    mom_ret: pd.Series, ew_ret: pd.Series, win_days: int = 504
) -> float:
    idx = mom_ret.index.intersection(ew_ret.index)
    m, e = mom_ret.reindex(idx), ew_ret.reindex(idx)
    wins, tot = 0, 0
    for start in range(0, len(idx) - win_days, win_days // 2):
        sl = slice(start, start + win_days)
        ms, es = m.iloc[sl], e.iloc[sl]
        if ms.std() == 0 or es.std() == 0:
            continue
        tot += 1
        if ms.mean() / ms.std() >= es.mean() / es.std():
            wins += 1
    return wins / tot if tot else float("nan")


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    eng.COST_BPS = 30.0  # 30 bps/side (module global read at call time)
    close, adv = load_smallcap()

    mem_small = band_membership(close, adv, side="small")
    mem_large = band_membership(close, adv, side="large")
    msz = int(pd.Series([len(s) for s in mem_small]).median())
    lsz = int(pd.Series([len(s) for s in mem_large]).median())
    print(
        f"[BAND] small median {msz} names/mo, large median {lsz} names/mo, "
        f"{len(mem_small)} rebalances",
        flush=True,
    )

    results: dict = {}
    rets: dict = {}

    # ---- H-035: momentum family on small band (top_in=30, top_out family) ----
    for to in (90, 120, 150):
        lab = f"H035_mom_top30_out{to}"
        res, eq, ret = run_verdict(
            close,
            mem_small,
            label=lab,
            mode="momentum",
            top_in=30,
            top_out=to,
            use_gate=False,
            div_panel=None,
        )
        results[lab] = res
        rets[lab] = ret
        print(
            f"[RUN] {lab}: final={res['final_value']:.0f} sharpe={res['sharpe_net']:.3f} "
            f"cagr={res['cagr_net']:.3f} maxdd={res['maxdd_net']:.3f}",
            flush=True,
        )

    # ---- No-Signal control: EW-band on SAME small band ----
    res_ews, eq_ews, ret_ews = run_verdict(
        close,
        mem_small,
        label="H035_EWband_small",
        mode="ew",
        ew_band=0.5,
    )
    results["H035_EWband_small"] = res_ews
    print(
        f"[RUN] EWband_small: final={res_ews['final_value']:.0f} "
        f"sharpe={res_ews['sharpe_net']:.3f} maxdd={res_ews['maxdd_net']:.3f}",
        flush=True,
    )

    # ---- H-036: EW-band on large band (pair vs small) ----
    res_ewl, eq_ewl, ret_ewl = run_verdict(
        close,
        mem_large,
        label="H036_EWband_large",
        mode="ew",
        ew_band=0.5,
    )
    results["H036_EWband_large"] = res_ewl
    print(
        f"[RUN] EWband_large: final={res_ewl['final_value']:.0f} "
        f"sharpe={res_ewl['sharpe_net']:.3f} maxdd={res_ewl['maxdd_net']:.3f}",
        flush=True,
    )

    # verdict variant = median-top_out momentum (out120)
    verdict_lab = "H035_mom_top30_out120"
    vres = results[verdict_lab]
    vret = rets[verdict_lab]

    # benchmarks over the verdict variant's window
    bench = etf_and_spy(close, vret.index[0], vret.index[-1])
    results["benchmarks"] = bench

    # mandatory metrics
    dsr = deflated_sharpe(vret, n_trials=N_LEDGER_AFTER)
    results["DSR_verdict"] = {
        "sharpe_observed_daily": float(dsr.sharpe_observed),
        "threshold": float(dsr.sharpe_threshold),
        "probability": float(dsr.deflated_sharpe_probability),
        "passes_5pct": bool(dsr.passes_5pct),
        "n_trials": N_LEDGER_AFTER,
    }
    results["PBO_CSCV_mom_family"] = float(cscv_pbo(pd.DataFrame(rets)))
    consist = window_consistency(vret, ret_ews)
    results["consistency_2y_mom_ge_ew"] = float(consist)

    # ---- H-035 pass/fail (ALL) ----
    h035 = {
        "1_gt_ETF": bool(vres["final_value"] > bench["ETF_net_final"]),
        "2_sharpe_gt_EWband": bool(vres["sharpe_net"] > res_ews["sharpe_net"]),
        "3_DSR_passes": bool(dsr.passes_5pct),
        "4_consistency_ge_60pct": bool(consist >= 0.60),
        "5_maxdd_not_worse": bool(vres["maxdd_net"] >= res_ews["maxdd_net"]),
    }
    results["H035_criteria"] = h035
    results["H035_PASS"] = bool(all(h035.values()))

    # ---- H-036 pass/fail ----
    h036 = {
        "small_gt_large_x110": bool(
            res_ews["final_value"] > res_ewl["final_value"] * 1.10
        ),
        "small_gt_ETF": bool(res_ews["final_value"] > bench["ETF_net_final"]),
    }
    results["H036_criteria"] = h036
    results["H036_PASS"] = bool(all(h036.values()))

    (OUTD / "h035_h036_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]",
        json.dumps(
            {
                "H035_PASS": results["H035_PASS"],
                "H035": h035,
                "H036_PASS": results["H036_PASS"],
                "H036": h036,
                "verdict_final": vres["final_value"],
                "verdict_sharpe": vres["sharpe_net"],
                "EWband_small_final": res_ews["final_value"],
                "EWband_large_final": res_ewl["final_value"],
                "ETF_net": bench["ETF_net_final"],
                "SPY_sharpe": bench["SPY_sharpe"],
                "DSR_p": results["DSR_verdict"]["probability"],
                "PBO": results["PBO_CSCV_mom_family"],
                "consistency": consist,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
