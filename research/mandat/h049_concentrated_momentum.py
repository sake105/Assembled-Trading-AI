"""H-049 — Konzentriertes Mega-Cap-Momentum (Copy-Trading-Archetyp) vs breiter ETF.

Top-N-12-1-Momentum, weites top_out (low turnover / no-retrim), KEIN Hebel,
terminal_liquidation (ehrliche Endsteuer 26,375 % minus Verlusttopf), Div-Steuer aktiv.
Vergleich vs window-matched ETF-Netto-Pfad (18,5 %, ebenfalls end-besteuert) + SPY-Sharpe.
Deterministische Engine (Frozenset-Fix). Explorativ: mega-cap-lastiges Fenster.
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
from verdict_engine import (
    load_div_panel,
    load_membership,
    load_verdict_prices,
    run_verdict,
)  # noqa: E402

ETF_TAX = 0.185


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)

    results, rets, eqs = {}, {}, {}
    for tin in (10, 20):
        lab = f"H049_conc_top{tin}"
        res, eq, ret = run_verdict(
            close,
            membership,
            label=lab,
            mode="momentum",
            top_in=tin,
            top_out=40,
            div_panel=divp,
            terminal_liquidation=True,
        )
        results[lab] = res
        rets[lab] = ret
        eqs[lab] = eq
        print(
            f"[RUN] {lab}: mtm={res['final_value']:,.0f} postliq={res['final_net_postliq']:,.0f} "
            f"sharpe={res['sharpe_net']:.3f} maxdd={res['maxdd_net']:.3f}",
            flush=True,
        )

    # window-matched ETF over the top10 book's equity window (both end-taxed)
    v_lab = "H049_conc_top10"
    v, vr, veq = results[v_lab], rets[v_lab], eqs[v_lab]
    spy = close["SPY"].dropna()
    spy_w = spy[(spy.index >= veq.index[0]) & (spy.index <= veq.index[-1])]
    etf_net = START_CAPITAL + START_CAPITAL * (spy_w.iloc[-1] / spy_w.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    spy_sharpe = float(
        spy_w.pct_change().dropna().mean()
        / spy_w.pct_change().dropna().std()
        * np.sqrt(252)
    )
    dsr = deflated_sharpe(vr, n_trials=133)

    crit = {
        "postliq_gt_ETF": bool(v["final_net_postliq"] > etf_net),
        "sharpe_gt_SPY": bool(v["sharpe_net"] > spy_sharpe),
        "DSR_passes": bool(dsr.passes_5pct),
    }
    out = {
        "top10_mtm": round(v["final_value"]),
        "top10_postliq": round(v["final_net_postliq"]),
        "top20_postliq": round(results["H049_conc_top20"]["final_net_postliq"]),
        "top10_sharpe": round(v["sharpe_net"], 3),
        "top10_maxdd": round(v["maxdd_net"], 3),
        "ETF_net_window": round(float(etf_net)),
        "SPY_sharpe": round(spy_sharpe, 3),
        "DSR_p": round(float(dsr.deflated_sharpe_probability), 3),
        "window": [str(veq.index[0].date()), str(veq.index[-1].date())],
        "criteria": crit,
        "PASS": bool(all(crit.values())),
    }
    (OUTD / "h049_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
