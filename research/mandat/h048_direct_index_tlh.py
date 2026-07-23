"""H-048 — Direct-Indexing Tax-Loss-Harvesting vs ETF-Netto-Pfad.

Voll-investiertes EW-Direct-Index-Buch (breites Band, low turnover) + optionales TLH
(DE: keine Wash-Sale-Regel -> Verlust ernten + sofort zurueckkaufen, Exposure unveraendert,
Verlust in den Aktien-Verlusttopf). FAIRNESS: End-Liquidation auf beiden Seiten (Strategie
realisiert am Ende zu 26,375 % minus Verlusttopf; ETF 18,5 %).

Saubere Steuer-Alpha-Messung = TLH-Buch vs no-TLH-Buch (gleicher Brutto-Pfad).
TLH-vs-ETF = Deployment-Frage (durch EW-vs-CW-Brutto konfundiert -> ehrlich benannt).
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
from h011_kandidat_a import OUTD, START_CAPITAL, TaxedPortfolio  # noqa: E402
from verdict_engine import load_div_panel, load_membership, load_verdict_prices  # noqa: E402

ETF_TAX = 0.185
TAX = 0.26375


def run_direct_index(close, membership, div, *, tlh_pct, ew_band=0.5):
    idx = close.index
    close_ff = close.ffill()
    last_valid = close.apply(lambda s: s.last_valid_index())
    global_last = idx[-1]
    month_ends = set(membership.index)
    pf = TaxedPortfolio(START_CAPITAL)
    pending: list[tuple[str, str, float]] = []
    n_harvests = 0

    for t in idx:
        pf.set_date(t)
        px_t = close.loc[t]
        for action, sym, amount in pending:
            px = px_t.get(sym, np.nan)
            if not np.isfinite(px):
                lv = last_valid.get(sym)
                if lv is not None and lv < t:
                    px = close.at[lv, sym]
                else:
                    continue
            if action == "sell_all":
                q = pf.qty(sym)
                if q > 0:
                    pf.sell(sym, q, float(px))
            elif action == "trade_to":
                cur = pf.qty(sym) * px
                d = amount - cur
                if d > 1.0:
                    pf.buy(sym, d, float(px))
                elif d < -1.0:
                    pf.sell(sym, -d / px, float(px))
        pending = []

        for sym in list(pf.lots.keys()):
            lv = last_valid.get(sym)
            if lv is not None and lv < t and lv < global_last - pd.Timedelta(days=10):
                pending.append(("sell_all", sym, 0.0))

        if div is not None and t in div.index:
            drow = div.loc[t]
            for sym in list(pf.lots.keys()):
                dv = drow.get(sym, np.nan)
                if np.isfinite(dv) and dv > 0:
                    tax = pf.qty(sym) * dv * TAX
                    pf.cash -= tax
                    pf.tax_paid += tax

        v = pf.cash
        ff_t = close_ff.loc[t]
        for sym, lots in pf.lots.items():
            px = ff_t.get(sym, np.nan)
            if np.isfinite(px):
                v += sum(q for q, _ in lots) * px

        if t not in month_ends:
            continue

        # --- TLH harvest (same-day, exposure-neutral): sell underwater, rebuy ---
        if tlh_pct is not None:
            for sym in list(pf.lots.keys()):
                lots = pf.lots.get(sym, [])
                q = sum(lq for lq, _ in lots)
                if q <= 0:
                    continue
                avg = sum(lq * lpx for lq, lpx in lots) / q
                px = px_t.get(sym, np.nan)
                if np.isfinite(px) and px < avg * (1.0 - tlh_pct):
                    notional = q * px
                    pf.sell(sym, q, float(px))  # bank the loss
                    pf.buy(sym, notional, float(px))  # rebuy same exposure
                    n_harvests += 1

        # --- EW-band rebalance ---
        members = membership.loc[t]
        tradable = [
            s
            for s in members
            if s in close.columns
            and np.isfinite(px_t.get(s, np.nan))
            and px_t.get(s, 0.0) >= 1.0
        ]
        if not tradable:
            continue
        tgt = v / len(tradable)
        held = set(pf.lots.keys())
        for sym in held - set(tradable):
            pending.append(("sell_all", sym, 0.0))
        for sym in tradable:
            px = px_t.get(sym, np.nan)
            cur = pf.qty(sym) * px if np.isfinite(px) else 0.0
            if tgt > 0 and abs(cur - tgt) / tgt > ew_band:
                pending.append(("trade_to", sym, tgt))

    # --- terminal liquidation (realize remaining gains at 26,375% minus Verlusttopf) ---
    last_px = close_ff.loc[global_last]
    for sym in list(pf.lots.keys()):
        px = last_px.get(sym, np.nan)
        if np.isfinite(px):
            pf.sell(sym, pf.qty(sym), float(px))
    final_net = pf.cash
    return {
        "final_net_postliq": float(final_net),
        "tax_paid": float(pf.tax_paid),
        "loss_pot_left": float(pf.loss_pot),
        "n_harvests": int(n_harvests),
    }


def main() -> int:
    eng.COST_BPS = 10.0
    close = load_verdict_prices()
    membership = load_membership(close.index)
    div = load_div_panel(close.index)
    spy = close["SPY"].dropna()
    etf_net = START_CAPITAL + START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1) * (
        1 - ETF_TAX
    )
    print(
        f"[DATA] {len(membership)} rebalances; ETF-net-path {etf_net:,.0f}", flush=True
    )

    results = {"ETF_net_path": round(float(etf_net))}
    variants = {"noTLH": None, "TLH15": 0.15, "TLH30": 0.30}
    for lab, pct in variants.items():
        r = run_direct_index(close, membership, div, tlh_pct=pct)
        results[lab] = r
        print(
            f"[RUN] {lab}: final_net(postliq)={r['final_net_postliq']:,.0f} "
            f"tax={r['tax_paid']:,.0f} loss_pot_left={r['loss_pot_left']:,.0f} "
            f"harvests={r['n_harvests']}",
            flush=True,
        )

    base = results["noTLH"]["final_net_postliq"]
    best_tlh = max(
        results["TLH15"]["final_net_postliq"], results["TLH30"]["final_net_postliq"]
    )
    results["tax_alpha_TLH_vs_noTLH_best"] = round(best_tlh - base)
    results["PASS"] = bool(best_tlh > etf_net and best_tlh > base)
    (OUTD / "h048_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
