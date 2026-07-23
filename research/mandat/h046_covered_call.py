"""H-046 — Covered-Call-Overlay („Aktien vermieten"). EXPLORATIV (BS-Modell-Prämien).

KEINE echten Optionsdaten im EODHD-Plan (403/404 belegt) -> Prämien via Black-Scholes
mit angenommener IV (Grid, da keine Markt-IV). Modell, kein Verdict.

Struktur: Stock-Bein voll investiert/gestundet (= ETF-artig, 18,5 % erst am Ende) PLUS
monatliches cash-settled Short-Call-Overlay. Overlay-P&L/Monat = Prämie − max(S1−K,0);
POSITIVE Monats-P&L SOFORT 26,375 % besteuert (Stillhalterprämie); negative voll (kein
garantierter Offset — konservativ). Frage: addiert das Overlay netto Endvermögen?

Grid: Strike {ATM, 3 %, 5 % OTM} × IV {realized, ×1,15, ×1,3} (bracket Vol-Risk-Prämie).
"""

from __future__ import annotations

import json
import sys
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
ETF_TAX = 0.185
TAX = 0.26375
R = 0.02
OPT_COST_BPS = (
    3.0  # per-month overlay transaction cost on underlying notional (optimistic)
)


def phi(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sig: float) -> float:
    if sig <= 0 or T <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S * phi(d1) - K * exp(-R * T) * phi(d2)


def run(close: pd.Series, *, otm: float, iv_mult: float) -> dict:
    close = close.dropna()
    r = close.pct_change()
    rv = r.rolling(21).std() * sqrt(252)  # trailing realized annualized vol
    me = close.groupby(close.index.to_period("M")).tail(1)  # month-end spots
    me = me[me.index.isin(close.index)]
    idx = me.index

    v_stock0 = START_CAPITAL
    overlay_cash = 0.0
    T = 21.0 / 252.0
    combined = []
    for i in range(len(idx) - 1):
        t0, t1 = idx[i], idx[i + 1]
        S0, S1 = float(close.at[t0]), float(close.at[t1])
        iv = float(rv.get(t0, np.nan)) * iv_mult
        if not np.isfinite(iv) or iv <= 0:
            v_stock = v_stock0 * S1 / float(close.at[idx[0]])
            combined.append((t1, v_stock + overlay_cash))
            continue
        K = S0 * (1.0 + otm)
        prem = bs_call(S0, K, T, iv)
        payout = max(S1 - K, 0.0)
        v_stock_t0 = v_stock0 * S0 / float(close.at[idx[0]])
        shares = v_stock_t0 / S0
        overlay_pnl = shares * (prem - payout)
        cost = v_stock_t0 * OPT_COST_BPS / 1e4
        overlay_pnl -= cost
        net = overlay_pnl * (1 - TAX) if overlay_pnl > 0 else overlay_pnl
        overlay_cash += net
        v_stock_t1 = v_stock0 * S1 / float(close.at[idx[0]])
        combined.append((t1, v_stock_t1 + overlay_cash))

    v_stock_final = v_stock0 * float(close.iloc[-1]) / float(close.at[idx[0]])
    v_stock_final_net = START_CAPITAL + (v_stock_final - START_CAPITAL) * (1 - ETF_TAX)
    cc_final = v_stock_final_net + overlay_cash

    c = pd.Series(dict(combined))
    cr = c.pct_change().dropna()
    return {
        "overlay_net_contribution": round(float(overlay_cash), 0),
        "buyhold_net_final": round(float(v_stock_final_net), 0),
        "covered_call_final": round(float(cc_final), 0),
        "cc_sharpe": round(float(cr.mean() / cr.std() * np.sqrt(12)), 3)
        if cr.std() > 0
        else None,
        "cc_maxdd": round(float((c / c.cummax() - 1).min()), 3),
    }


def main() -> int:
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()

    # buy&hold baseline metrics
    r = spy.pct_change().dropna()
    bh_final_net = START_CAPITAL + (
        START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    ) * (1 - ETF_TAX)
    bh_me = spy.groupby(spy.index.to_period("M")).tail(1)
    bh_mr = bh_me.pct_change().dropna()
    base = {
        "buyhold_ETF_net": round(float(bh_final_net), 0),
        "buyhold_sharpe_m": round(float(bh_mr.mean() / bh_mr.std() * np.sqrt(12)), 3),
        "buyhold_maxdd_m": round(float((bh_me / bh_me.cummax() - 1).min()), 3),
    }
    print("[BASE]", json.dumps(base, indent=2), flush=True)

    grid = {}
    any_beat = False
    for otm in (0.0, 0.03, 0.05):
        for ivm in (1.0, 1.15, 1.3):
            res = run(spy, otm=otm, iv_mult=ivm)
            key = f"otm{int(otm * 100)}_iv{ivm}"
            grid[key] = res
            beats = (
                res["overlay_net_contribution"] > 0
                and res["covered_call_final"] > base["buyhold_ETF_net"]
            )
            any_beat = any_beat or beats
            print(
                f"[RUN] {key}: overlay_contrib={res['overlay_net_contribution']:,.0f} "
                f"cc_final={res['covered_call_final']:,.0f} vs BH {base['buyhold_ETF_net']:,.0f} "
                f"| Sharpe {res['cc_sharpe']} DD {res['cc_maxdd']} {'BEAT' if beats else ''}",
                flush=True,
            )

    out = {"base": base, "grid": grid, "PASS_any": bool(any_beat)}
    (OUTD / "h046_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT] PASS_any:", any_beat, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
