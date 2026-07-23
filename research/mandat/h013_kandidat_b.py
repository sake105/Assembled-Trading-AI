"""H-013 — Kandidat B: SMA200-Regime-Rotation auf SPY (Registry: research/registry.md).

WARNBOX §3.2 bestätigt (RORO-Liquidation). NUR Backtest — kein Paper, kein Live,
kein Hebelprodukt ohne separate schriftliche Freigabe (Guardrail 4).
Survivorship-IMMUN (SPY-only). Familie: B1 (1x/Cash), B2 (2x/1x), B3 (2x/Cash).
2x synthetisch: 2×daily-Rendite − 3,9 % p.a. Drag. Steuern: 26,375 % FIFO je Switch.
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

from h011_kandidat_a import OUTD, START_CAPITAL, TaxedPortfolio, cscv_pbo  # noqa: E402

ETF_TAX = 0.185
N_TRIALS = 50
DRAG_2X = 0.039 / 252.0  # financing + TER, daily


def run_rotation(spy: pd.Series, *, lev_on: float, lev_off: float, label: str):
    """Monthly SMA200 gate; the 'asset' is SPY (1x) or a synthetic 2x series.

    Implementation: we trade a single instrument whose daily return is
    lev*r_spy - drag (drag only for lev=2). Switching between regimes sells the
    entire position (FIFO tax) and buys the other instrument.
    """
    sma = spy.rolling(200).mean()
    ret = spy.pct_change().fillna(0.0)
    # synthetic price paths
    px = {
        1.0: spy / spy.iloc[0] * 100.0,
        2.0: (1.0 + 2.0 * ret - DRAG_2X).cumprod() * 100.0,
        0.0: pd.Series(1.0, index=spy.index),  # cash
    }
    month_ends = set(
        pd.Series(spy.index, index=spy.index).groupby(spy.index.to_period("M")).max()
    )

    pf = TaxedPortfolio(START_CAPITAL)
    cur_lev: float | None = None
    pending: float | None = None
    equity = []
    for t in spy.index:
        # execute pending switch at today's close
        if pending is not None and pending != cur_lev:
            if cur_lev is not None and cur_lev != 0.0:
                q = pf.qty("ASSET")
                if q > 0:
                    pf.sell("ASSET", q, float(px[cur_lev].at[t]))
            if pending != 0.0:
                pf.buy("ASSET", pf.cash, float(px[pending].at[t]))
            cur_lev = pending
            pending = None
        elif pending is not None:
            pending = None

        cur_px = float(px[cur_lev].at[t]) if cur_lev not in (None, 0.0) else 0.0
        v = pf.cash + (pf.qty("ASSET") * cur_px if cur_lev not in (None, 0.0) else 0.0)
        equity.append((t, v))

        if t in month_ends and np.isfinite(sma.at[t]):
            target = lev_on if spy.at[t] >= sma.at[t] else lev_off
            if cur_lev is None or target != cur_lev:
                pending = target

    eq = pd.Series(dict(equity)).sort_index()
    # warmup: start measuring once SMA exists
    eq = eq[spy.rolling(200).mean().notna()]
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    res = {
        "label": label,
        "final_value": float(eq.iloc[-1] / eq.iloc[0] * START_CAPITAL),
        "cagr_net": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
        "sharpe_net": float(r.mean() / r.std() * np.sqrt(252))
        if r.std() > 0
        else float("nan"),
        "maxdd_net": float((eq / eq.cummax() - 1).min()),
        "tax_paid": float(pf.tax_paid),
        "costs_paid": float(pf.costs_paid),
        "years": float(years),
    }
    return res, eq, r


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    prices = pd.read_parquet(
        Path(__file__).resolve().parent / "data" / "prices_sp500.parquet"
    )
    spy = (
        prices[prices["symbol"] == "SPY"]
        .set_index("timestamp")["close"]
        .sort_index()
        .dropna()
    )
    print(f"[DATA] SPY {spy.index[0].date()} -> {spy.index[-1].date()}", flush=True)

    fam = {
        "B1_1x_cash": dict(lev_on=1.0, lev_off=0.0),
        "B2_2x_1x": dict(lev_on=2.0, lev_off=1.0),
        "B3_2x_cash": dict(lev_on=2.0, lev_off=0.0),
    }
    results, rets = {}, {}
    for name, kw in fam.items():
        res, eq, r = run_rotation(spy, label=name, **kw)
        results[name] = res
        rets[name] = r
        print(f"[RUN] {name}: {res}", flush=True)

    # benchmarks on the SAME (post-warmup) window for fairness
    warm = spy[spy.rolling(200).mean().notna()]
    years = (warm.index[-1] - warm.index[0]).days / 365.25
    spy_r = warm.pct_change().dropna()
    gross_gain = START_CAPITAL * (warm.iloc[-1] / warm.iloc[0] - 1)
    etf_net_final = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    results["SPY_bh_gross"] = {
        "cagr": float((warm.iloc[-1] / warm.iloc[0]) ** (1 / years) - 1),
        "sharpe": float(spy_r.mean() / spy_r.std() * np.sqrt(252)),
        "maxdd": float((warm / warm.cummax() - 1).min()),
    }
    results["ETF_net_path"] = {
        "final_value": float(etf_net_final),
        "cagr_net": float((etf_net_final / START_CAPITAL) ** (1 / years) - 1),
    }

    best = max(rets, key=lambda k: results[k]["final_value"])
    dsr = deflated_sharpe(rets[best], n_trials=N_TRIALS)
    results["selected"] = best
    results["DSR_selected"] = {
        "probability": float(dsr.deflated_sharpe_probability),
        "passes_5pct": bool(dsr.passes_5pct),
        "n_trials": N_TRIALS,
    }
    results["PBO_CSCV_3variants"] = float(cscv_pbo(pd.DataFrame(rets)))

    v = rets[best]
    win = {}
    for y0 in range(v.index[0].year, v.index[-1].year, 2):
        m = (v.index.year >= y0) & (v.index.year < y0 + 2)
        mb = (spy_r.index.year >= y0) & (spy_r.index.year < y0 + 2)
        if m.sum() > 100:
            s1 = v[m].mean() / v[m].std() * np.sqrt(252)
            s2 = spy_r[mb].mean() / spy_r[mb].std() * np.sqrt(252)
            win[f"{y0}-{y0 + 1}"] = {
                "best": round(float(s1), 3),
                "SPY": round(float(s2), 3),
            }
    results["subperiods_best_vs_SPY"] = win

    out_path = OUTD / "h013_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] -> {out_path}", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
