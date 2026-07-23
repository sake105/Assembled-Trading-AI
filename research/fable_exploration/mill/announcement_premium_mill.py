"""PHASE-3 mill for H4 — earnings-announcement premium as a tradeable basket.
PIT-tradeable version: predict the NEXT announcement from the prior one (+~91d
cadence) and hold the name in the [-5,-1] trading-day pre-window before the
PREDICTED date. Equal-weight basket, net of costs, vs SPY. Decaying-by-year was the
phase-1 flag; this checks whether anything survives costs + DSR deflation."""

from __future__ import annotations
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from insider_buy_mill import load_returns, metrics, run_portfolio  # noqa: E402
from src.assembled_core.features.pead_sue import build_quarterly_eps_panel  # noqa: E402
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402


def build_pre_window_eligibility(dates, universe, xbrl, pre=5):
    idx = pd.DatetimeIndex(dates)
    E = pd.DataFrame(False, index=idx, columns=universe)
    for sym in universe:
        panel = build_quarterly_eps_panel(xbrl, sym)
        if panel.empty:
            continue
        sub = xbrl[xbrl["symbol"].astype(str).str.upper() == sym]
        ann = (
            sub.dropna(subset=["period_end"])
            .groupby(sub["period_end"].dt.normalize())["available_at"]
            .min()
        )
        pes = set(pd.to_datetime(panel["period_end"]).dt.normalize())
        ann = ann[[pe in pes for pe in ann.index]].sort_values()
        adates = [
            pd.Timestamp(a).normalize() for a in ann.values
        ]  # .values already UTC-naive
        for a in adates:
            pred = a + pd.Timedelta(days=91)  # predict next quarter from THIS one
            ppos = idx.searchsorted(pred, side="left")
            lo, hi = ppos - pre, ppos - 1  # hold [-5,-1] before predicted date
            if hi < 0 or lo >= len(idx):
                continue
            lo = max(lo, 0)
            hi = min(hi, len(idx) - 1)
            E.iloc[lo : hi + 1, E.columns.get_loc(sym)] = True
    return E


def main():
    wide, R, deep, dvol = load_returns()
    dates = R.index
    spy = R["SPY"].reindex(dates).fillna(0.0)
    xbrl = pd.read_parquet("data/raw/fundamentals/fundamentals_xbrl_full.parquet")
    universe = sorted(set(deep) & set(xbrl["symbol"].astype(str).str.upper()) - {"SPY"})
    print(f"universe={len(universe)} dates {dates.min().date()}..{dates.max().date()}")

    E = build_pre_window_eligibility(dates, universe, xbrl)
    print(
        f"avg names in pre-window/day: {(E.sum(axis=1)[E.sum(axis=1) > 0]).mean():.1f} "
        f"active days: {int((E.sum(axis=1) > 0).sum())}/{len(dates)}"
    )

    print("\n=== basket net of costs (daily rebal) ===")
    print("  SPY: ", metrics(spy))
    for cb in (10, 20, 40):
        net, breadth, W, pnl = run_portfolio(dates, R, E, 1, cost_bps=cb)
        m = metrics(net)
        print(
            f"  pre-window @ {cb}bps: sharpe={m['sharpe']:.3f} cagr={m['cagr']:+.3%} maxdd={m['maxdd']:.3%}"
        )

    net, breadth, W, pnl = run_portfolio(dates, R, E, 1, cost_bps=20)
    print("\n=== per-year (basket sharpe vs SPY) @20bps ===")
    beats = 0
    n = 0
    for yr, g in net.groupby(net.index.year):
        si = metrics(g)["sharpe"]
        sb = metrics(spy.loc[g.index])["sharpe"]
        b = si > sb
        beats += b
        n += 1
        print(f"  {yr}: {si:+.2f} vs {sb:+.2f} {'BEAT' if b else '.'}")
    print(f"  folds beating SPY: {beats}/{n}")

    print("\n=== DSR (cumulative trial count ~33) ===")
    for nt in (1, 33):
        d = deflated_sharpe(net.dropna(), n_trials=nt)
        print(
            f"  n_trials={nt}: prob={d.deflated_sharpe_probability:.4f} "
            f"sr_pp={d.sharpe_observed:.4f} passes5%={d.passes_5pct}"
        )
    print("\n[DONE] announcement premium mill")


if __name__ == "__main__":
    main()
