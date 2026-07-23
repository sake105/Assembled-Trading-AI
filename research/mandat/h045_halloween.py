"""H-045 — Halloween / Sell-in-May (Bouman/Jacobsen). Low-Turnover-Saisonalität.

In-Markt Nov–Apr, Cash Mai–Okt. Jährliche Mai-Realisation → deutsche Steuer (Verlusttopf,
Jahres-Netting), Kosten pro Umschichtung. Vergleich vs ETF-Netto-Pfad + Buy&Hold-Sharpe.
Nutzt h042-Cache (prices_overnight_oc.parquet: SPY + Sektoren close).
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

from h011_kandidat_a import OUTD, START_CAPITAL  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
ETF_TAX = 0.185
TAX = 0.26375
COST_BPS = 5.0  # per switch (one-way)
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]


def halloween_book(close: pd.Series) -> dict:
    """Long Nov(11)-Apr(4), cash May(5)-Oct(10). Annual-netting German tax."""
    r = close.pct_change().dropna()
    in_mkt = r.index.month.isin([11, 12, 1, 2, 3, 4])
    # switch days: month transitions into/out of the in-market window
    invested = pd.Series(in_mkt, index=r.index)
    cap = START_CAPITAL
    year_pnl: dict[int, float] = {}
    eq = []
    prev_in = False
    for t, ri in r.items():
        now_in = bool(invested.at[t])
        if now_in != prev_in:  # switch (buy or sell) -> one-way cost
            cost = cap * COST_BPS / 1e4
            cap -= cost
            year_pnl[t.year] = year_pnl.get(t.year, 0.0) - cost
            prev_in = now_in
        if now_in:
            pnl = cap * ri
            cap += pnl
            year_pnl[t.year] = year_pnl.get(t.year, 0.0) + pnl
        eq.append((t, cap))
    tax_total = sum(max(p, 0.0) * TAX for p in year_pnl.values())
    e = pd.Series(dict(eq))
    er = e.pct_change().dropna()
    er = er[er != 0]  # ignore flat cash days for Sharpe of active exposure
    return {
        "final_pretax": float(e.iloc[-1]),
        "final_net": float(e.iloc[-1] - tax_total),
        "tax_total": float(tax_total),
        "sharpe_active": float(er.mean() / er.std() * np.sqrt(252))
        if er.std() > 0
        else float("nan"),
    }


def main() -> int:
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    close = oc.pivot(index="date", columns="symbol", values="close").sort_index()

    results = {}
    # SPY book
    spy = close["SPY"].dropna()
    book_spy = halloween_book(spy)
    gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net = START_CAPITAL + gain * (1 - ETF_TAX)
    spy_bh_sharpe = float(
        spy.pct_change().dropna().mean()
        / spy.pct_change().dropna().std()
        * np.sqrt(252)
    )
    results["SPY"] = {
        "book": book_spy,
        "ETF_net": round(float(etf_net)),
        "buyhold_sharpe": round(spy_bh_sharpe, 3),
        "PASS": bool(
            book_spy["final_net"] > etf_net
            and book_spy["sharpe_active"] > spy_bh_sharpe
        ),
    }

    # EW-sectors book (average of sector Halloween books, rebal-free proxy: mean equity path)
    sec = close[SECTORS].dropna()
    ew = sec.pct_change().mean(axis=1).dropna()  # EW daily return
    ew_close = (1 + ew).cumprod() * 100
    book_ew = halloween_book(ew_close)
    ew_gain = START_CAPITAL * (ew_close.iloc[-1] / ew_close.iloc[0] - 1)
    ew_etf_net = START_CAPITAL + ew_gain * (1 - ETF_TAX)
    ew_bh_sharpe = float(ew.mean() / ew.std() * np.sqrt(252))
    results["EW_sectors"] = {
        "book": book_ew,
        "ETF_net": round(float(ew_etf_net)),
        "buyhold_sharpe": round(ew_bh_sharpe, 3),
        "PASS": bool(
            book_ew["final_net"] > ew_etf_net
            and book_ew["sharpe_active"] > ew_bh_sharpe
        ),
    }

    results["PASS_any"] = bool(results["SPY"]["PASS"] or results["EW_sectors"]["PASS"])
    (OUTD / "h045_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("[VERDICT]", json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
