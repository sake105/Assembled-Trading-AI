"""H-032b — Re-Verifikation der PASS-Designs unter EHRLICHER End-Liquidation.

Frage (Crux des Mandats): Überlebt der einzige „schlägt-den-ETF"-PASS (H-032 low_div,
mark-to-market 2,70M vs ETF 1,59M) die End-Liquidation (Endgewinne zu 26,375 % minus
Verlusttopf/Sparerpauschbetrag)? Vergleich final_net_postliq vs ETF-Netto-Pfad (18,5 %).
KEIN neuer Trial — Robustheit/Methodik der bestehenden PASSes.
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
    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)

    div_daily = divp.reindex(close.index).fillna(0.0)
    trail12 = div_daily.rolling(252, min_periods=200).sum()
    jan_ends = [t for t in membership.index if t.month == 1]
    yld = (trail12.loc[jan_ends] / close.loc[jan_ends]).replace(
        [np.inf, -np.inf], np.nan
    )

    out = {}
    win_ref = None
    for name, score, mode, kw in (
        ("H032_low_div", -yld, "momentum", dict(top_in=50, top_out=100)),
        ("EW_full", None, "ew", dict(ew_band=0.5)),
    ):
        res_mtm, eq, _ = run_verdict(
            close,
            membership,
            label=name,
            mode=mode,
            div_panel=divp,
            score_panel=score,
            terminal_liquidation=False,
            **kw,
        )
        res_liq, _, _ = run_verdict(
            close,
            membership,
            label=name,
            mode=mode,
            div_panel=divp,
            score_panel=score,
            terminal_liquidation=True,
            **kw,
        )
        if name == "H032_low_div":
            win_ref = (eq.index[0], eq.index[-1])
        out[name] = {
            "final_mtm": round(res_mtm["final_value"]),
            "final_net_postliq": round(res_liq["final_net_postliq"]),
            "tax_mtm": round(res_mtm["tax_paid"]),
            "tax_postliq": round(res_liq["tax_paid"]),
            "sharpe": round(res_mtm["sharpe_net"], 3),
        }

    # window-matched ETF (over low-div's warmup-trimmed equity window)
    spy = close["SPY"].dropna()
    spy_w = spy[(spy.index >= win_ref[0]) & (spy.index <= win_ref[1])]
    etf_win = START_CAPITAL + START_CAPITAL * (spy_w.iloc[-1] / spy_w.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    etf_full = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    out["ETF_net_window_matched"] = round(float(etf_win))
    out["ETF_net_full_span"] = round(float(etf_full))
    out["window"] = [str(win_ref[0].date()), str(win_ref[1].date())]

    low = out["H032_low_div"]
    out["low_div_mtm_beats_ETFwin"] = bool(low["final_mtm"] > etf_win)
    out["low_div_postliq_beats_ETFwin"] = bool(low["final_net_postliq"] > etf_win)
    (OUTD / "h032b_terminal_liq.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(out, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
