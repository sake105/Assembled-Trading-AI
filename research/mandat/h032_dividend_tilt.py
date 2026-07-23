"""H-032 — Dividend-Tilt: bestraft die deutsche Steuer Ausschuettung? (Registry W9).

Jaehrliches Signal (Januar-Ultimo): trailing-12M-Yield (PIT: vergangene Divs /
aktueller Kurs). low_div = Top 50 NIEDRIGSTER Yield (inkl. Null-Zahler),
high_div = Top 50 hoechster Yield. Beide identische Mechanik (no-retrim,
Exit ausserhalb Top 100 der eigenen Rangliste, Cap 10 %, volle Steuern inkl.
Div-Drag). Paar-Design. N->96.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402
from verdict_engine import (  # noqa: E402
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)

ETF_TAX = 0.185


def main() -> int:
    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)

    # trailing-12M dividend sum per symbol at each January month-end (PIT)
    div_daily = divp.reindex(close.index).fillna(0.0)
    trail12 = div_daily.rolling(252, min_periods=200).sum()
    jan_ends = [t for t in membership.index if t.month == 1]
    yld = (trail12.loc[jan_ends] / close.loc[jan_ends]).replace(
        [np.inf, -np.inf], np.nan
    )

    results, rets = {}, {}
    for name, score in (("H032_low_div", -yld), ("H032_high_div", yld)):
        res, _eq, ret = run_verdict(
            close,
            membership,
            label=name,
            mode="momentum",
            top_in=50,
            top_out=100,
            div_panel=divp,
            score_panel=score,
        )
        results[name] = res
        rets[name] = ret
        print(
            f"[RUN] {name}: final={res['final_value']:.0f} cagr={res['cagr_net'] * 100:.2f}% sharpe={res['sharpe_net']:.3f} tax={res['tax_paid']:.0f}",
            flush=True,
        )

    # benchmarks: same window EW-PIT (div-tax) + ETF path
    start = rets["H032_low_div"].index[0]
    res_ew, _eq, ret_ew = run_verdict(
        close, membership, label="EW_ref", mode="ew", div_panel=divp
    )
    ew_final_window = None  # EW runs full window; compare via its final directly
    spy = close["SPY"].dropna()
    spy_w = spy[spy.index >= start]
    etf_net = START_CAPITAL + START_CAPITAL * (spy_w.iloc[-1] / spy_w.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    results["EW_PIT_divtax_full"] = res_ew
    results["ETF_net_path_window"] = {"final_value": float(etf_net)}

    low, high = (
        results["H032_low_div"]["final_value"],
        results["H032_high_div"]["final_value"],
    )
    results["_verdict"] = {
        "low_div_final": low,
        "high_div_final": high,
        "ratio": round(low / high, 3),
        "crit1_low_gt_high_x110": low > high * 1.10,
        "crit2_low_gt_ewpit": low > res_ew["final_value"],
        "PASS": (low > high * 1.10) and (low > res_ew["final_value"]),
    }
    (OUTD / "h032_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(
        "[VERDICT]", json.dumps(results["_verdict"], indent=2, default=str), flush=True
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
