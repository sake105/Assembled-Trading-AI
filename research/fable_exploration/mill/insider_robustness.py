"""PHASE-3 robustness battery for H1 — the cuts that decide real-vs-artifact.
Drop-top-names, small-cap-realistic costs, sub-period halves, and a proper
CSCV-PBO (Bailey-Lopez de Prado) across the 12-variant grid (the overfit test the
codebase lacks). Imports the mill engine; adds nothing it doesn't already use."""

from __future__ import annotations
import itertools
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from insider_buy_mill import (  # noqa: E402
    build_eligibility,
    load_pbuys,
    load_returns,
    metrics,
    run_portfolio,
)


def sr_pp(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std()) if len(r) > 2 and r.std() > 0 else float("nan")


def main():
    wide, R, deep, dvol = load_returns()
    buys = load_pbuys()
    dates = R.index
    spy = R["SPY"].reindex(dates).fillna(0.0)
    universe = sorted(set(deep) & set(buys["symbol"].unique()) - {"SPY"})

    E = build_eligibility(dates, universe, buys, 63, False)
    net, breadth, W, pnl = run_portfolio(dates, R, E, 5)
    base = metrics(net)
    print(
        f"BASE: sharpe={base['sharpe']:.3f} cagr={base['cagr']:+.3%}  SPY sharpe={metrics(spy)['sharpe']:.3f}"
    )

    # 1) drop-top-K PnL names — is the edge carried by a few winners?
    print("\n=== drop-top-K PnL names ===")
    ranked = pnl.sort_values(ascending=False)
    for K in (1, 3, 5):
        drop = set(ranked.head(K).index)
        sub = [u for u in universe if u not in drop]
        Es = build_eligibility(dates, sub, buys, 63, False)
        ns, _, _, _ = run_portfolio(dates, R, Es, 5)
        m = metrics(ns)
        print(
            f"  drop {K} ({','.join(list(drop))}): sharpe={m['sharpe']:.3f} cagr={m['cagr']:+.3%}"
        )

    # 2) small-cap-realistic costs (illiquid names have wide spreads, not 20bps)
    print("\n=== cost stress (round-trip bps on turnover) ===")
    for cb in (20, 40, 60, 100):
        ns, _, _, _ = run_portfolio(dates, R, E, 5, cost_bps=cb)
        print(
            f"  {cb:>3}bps: sharpe={metrics(ns)['sharpe']:.3f} cagr={metrics(ns)['cagr']:+.3%}"
        )

    # 3) liquidity slice at REALISTIC per-slice costs
    print("\n=== liquidity slices at slice-realistic costs ===")
    uvol = dvol.reindex(universe).dropna().sort_values()
    lo = list(uvol.index[: len(uvol) // 3])
    hi = list(uvol.index[-len(uvol) // 3 :])
    for label, sub, cb in (("LOW-liq @80bps", lo, 80), ("HIGH-liq @20bps", hi, 20)):
        Es = build_eligibility(dates, sub, buys, 63, False)
        ns, _, _, _ = run_portfolio(dates, R, Es, 5, cost_bps=cb)
        m = metrics(ns)
        print(
            f"  {label} ({len(sub)} names): sharpe={m['sharpe']:.3f} cagr={m['cagr']:+.3%}"
        )

    # 4) sub-period halves
    print("\n=== sub-period halves ===")
    mid = pd.Timestamp("2022-06-01")
    for label, mask in (("2018-2022H1", dates < mid), ("2022H2-2026", dates >= mid)):
        m = metrics(net[mask])
        ms = metrics(spy[mask])
        print(
            f"  {label}: port sharpe={m['sharpe']:.3f} (cagr {m['cagr']:+.3%}) | SPY {ms['sharpe']:.3f}"
        )

    # 5) CSCV-PBO across the 12-variant grid (Bailey-LdP overfit probability)
    print("\n=== CSCV-PBO across 12 variants ===")
    grid = [
        dict(lookback=lb, rebal=st, cluster=cl)
        for lb in (21, 63, 126)
        for st in (5, 21)
        for cl in (False, True)
    ]
    streams = []
    for cfg in grid:
        Eg = build_eligibility(dates, universe, buys, cfg["lookback"], cfg["cluster"])
        ng, _, _, _ = run_portfolio(dates, R, Eg, cfg["rebal"])
        streams.append(ng.reindex(dates).fillna(0.0).values)
    M = np.column_stack(streams)  # T x N
    T, N = M.shape
    S = 10
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for combo in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in combo])
        is_sr = np.array([sr_pp(M[is_idx, j]) for j in range(N)])
        oos_sr = np.array([sr_pp(M[oos_idx, j]) for j in range(N)])
        nstar = int(np.nanargmax(is_sr))
        # relative OOS rank of the IS-best (1=worst..N=best)
        rank = (np.sum(oos_sr <= oos_sr[nstar])) / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))
    print(
        f"  combinations={len(logits)}  PBO(prob IS-best is OOS-below-median)={pbo:.3f}  "
        f"median_logit={np.median(logits):+.2f}"
    )
    print("  (PBO<0.5 = selection not overfit; >0.5 = the best variant is likely luck)")
    print("\n[DONE] robustness battery")


if __name__ == "__main__":
    main()
