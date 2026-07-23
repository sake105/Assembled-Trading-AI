"""Gross-vs-Net über ~30 J: was kostet die deutsche Steuer je Strategie?

Jede Kernstrategie brutto (TAX=0, DIV_TAX=0) und netto (volle Steuer + End-Liquidation),
Transaktionskosten in BEIDEN gleich (10 bps) — isoliert reinen Steuereffekt.
ETF passiv: 18,46 % Teilfreistellung; Direktaktien-Strategien: 26,375 % + Verlusttopf.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h011_kandidat_a as eng  # noqa: E402
import verdict_engine as ve  # noqa: E402
from h011_kandidat_a import OUTD, START_CAPITAL, load_roe_panel  # noqa: E402
from h040_h041_lowvol_quality import sub_membership  # noqa: E402
from verdict_engine import load_div_panel, load_membership, load_verdict_prices  # noqa: E402


def cagr(final: float, years: float) -> float:
    return (final / START_CAPITAL) ** (1 / years) - 1


def run(close, mem, *, mode, gross, div=None, **kw):
    eng.TAX = 0.0 if gross else 0.26375
    ve.DIV_TAX = 0.0 if gross else 0.26375
    eng.COST_BPS = 10.0
    res, _, _ = ve.run_verdict(
        close, mem, label="x", mode=mode, div_panel=div, terminal_liquidation=True, **kw
    )
    return float(res["final_net_postliq"])


def main() -> int:
    close = load_verdict_prices()
    membership = load_membership(close.index)
    divp = load_div_panel(close.index)
    years = (close.index[-1] - close.index[0]).days / 365.25

    # signals
    div_daily = divp.reindex(close.index).fillna(0.0)
    trail12 = div_daily.rolling(252, min_periods=200).sum()
    jan = [t for t in membership.index if t.month == 1]
    yld = (trail12.loc[jan] / close.loc[jan]).replace([np.inf, -np.inf], np.nan)
    vol63 = close.pct_change(fill_method=None).rolling(63).std()
    roe = load_roe_panel(close.index, [c for c in close.columns if c != "SPY"])
    mem_q = sub_membership(membership, roe, pick=0.33, low_is_good=False)
    mem_lv = sub_membership(membership, vol63, pick=0.33, low_is_good=True)

    # (name, membership, mode, kwargs, uses_div)
    strategies = [
        ("EW-Band (H-024)", membership, "ew", dict(ew_band=0.5), True),
        (
            "Low-Div Tilt (H-032)",
            membership,
            "momentum",
            dict(top_in=50, top_out=100, score_panel=-yld),
            True,
        ),
        (
            "Momentum Top-20 (H-049)",
            membership,
            "momentum",
            dict(top_in=20, top_out=40),
            True,
        ),
        ("Quality/ROE Top-33 (H-041)", mem_q, "ew", dict(ew_band=0.5), True),
        ("Low-Vol Top-33 (H-040)", mem_lv, "ew", dict(ew_band=0.5), True),
    ]

    rows = []
    # passive ETF benchmark
    spy = close["SPY"].dropna()
    g = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0])
    n = START_CAPITAL + (g - START_CAPITAL) * (1 - 0.1846)
    rows.append(("ETF passiv (thesaur., 18,46%)", g, n))

    for name, mem, mode, kw, uses_div in strategies:
        d = divp if uses_div else None
        gf = run(close, mem, mode=mode, gross=True, div=d, **kw)
        nf = run(close, mem, mode=mode, gross=False, div=d, **kw)
        rows.append((name, gf, nf))
        print(f"[OK] {name}: gross={gf:,.0f} net={nf:,.0f}", flush=True)

    out = {"years": round(years, 1), "principal": START_CAPITAL, "rows": []}
    for name, gf, nf in rows:
        out["rows"].append(
            {
                "name": name,
                "cagr_gross_pct": round(cagr(gf, years) * 100, 2),
                "cagr_net_pct": round(cagr(nf, years) * 100, 2),
                "final_gross": round(gf),
                "final_net": round(nf),
                "tax_cost_eur": round(gf - nf),
                "tax_cost_pct_of_gross": round((gf - nf) / gf * 100, 1),
                "cagr_drag_pp": round((cagr(gf, years) - cagr(nf, years)) * 100, 2),
            }
        )
    (OUTD / "h051_gross_vs_net.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("[TABLE]", json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
