"""PHASE-3 FALSIFICATION MILL — H1: insider open-market-buy long portfolio.

Portfolio rule (PIT, parameter-free except the pre-registered grid):
  Hold an equal-weight long basket of every universe name that had >=1 open-market
  insider PURCHASE (Form 4 code 'P', non-derivative, price>0) accepted in the
  trailing LOOKBACK trading days. Eligibility starts the FIRST trading day STRICTLY
  AFTER `available_at` (no same-day leak). Rebalanced on a fixed cadence; cash (0)
  when no name qualifies. Costs charged on rebalance turnover.

Honesty rails:
  * Universe = survivorship survivors only -> the level is an UPPER BOUND (insiders
    buy falling stocks; bankruptcies are absent). Reported, not hidden.
  * Net of transaction costs (1x base + 2x stress).
  * Benchmarks: SPY and 60/40 (0.6 SPY + 0.4 AGG) over matched dates, risk-adjusted.
  * Multiple testing: Deflated Sharpe over the full variant grid (+ phase-1 looks).
  * Robustness: per-year folds, PnL concentration, liquidity-bucket split.

Run: python research/fable_exploration/mill/insider_buy_mill.py
"""

from __future__ import annotations
import json
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

ANN = 252
START, END = pd.Timestamp("2018-01-01"), pd.Timestamp("2026-06-10")
MIN_HISTORY = 1000  # >= ~4y of daily bars to enter the mill universe
BASE_COST_BPS = 20.0  # round-trip on traded notional (10 comm + ~10 spread/impact)


# --------------------------------------------------------------------- loaders
def load_returns():
    px = pd.read_parquet("output/aggregates/daily.parquet")
    px["date"] = (
        pd.to_datetime(px["timestamp"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    wide = px.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    wide = wide.loc[(wide.index >= START) & (wide.index <= END)]
    R = wide.pct_change(fill_method=None)
    # mill universe: deep-history names
    deep = [c for c in wide.columns if wide[c].notna().sum() >= MIN_HISTORY]
    dollar_vol = px.assign(dv=px["close"] * px["volume"]).groupby("symbol")["dv"].mean()
    return wide, R, deep, dollar_vol


def load_pbuys():
    ins = pd.read_parquet("data/raw/insider_congress/form4_insider_full.parquet")
    b = ins[
        (ins["transaction_type"] == "P")
        & (~ins["is_derivative"])
        & (ins["price"] > 0)
        & (ins["shares"] > 0)
    ].copy()
    b["avail"] = (
        pd.to_datetime(b["available_at"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    b["symbol"] = b["symbol"].astype(str).str.upper()
    return b[["symbol", "avail", "reporting_owner_cik", "value_usd"]]


# ----------------------------------------------------------------- eligibility
def build_eligibility(dates, universe, buys, lookback, cluster):
    """Boolean (date x symbol): name eligible on day t if a qualifying P-buy was
    available on a trading day in (t-lookback, t-1]. cluster=True requires >=2
    distinct insiders within the trailing 30 calendar days at the buy date."""
    idx = pd.DatetimeIndex(dates)
    E = pd.DataFrame(False, index=idx, columns=universe)
    for sym, g in buys.groupby("symbol"):
        if sym not in E.columns:
            continue
        g = g.sort_values("avail")
        avails = g["avail"].values
        for av in g["avail"].unique():
            if cluster:
                win = g[(g["avail"] > av - pd.Timedelta(days=30)) & (g["avail"] <= av)]
                if win["reporting_owner_cik"].nunique() < 2:
                    continue
            # eligible from first trading day strictly after av, for `lookback` trading days
            start_pos = idx.searchsorted(av, side="right")
            if start_pos >= len(idx):
                continue
            end_pos = min(start_pos + lookback, len(idx))
            E.iloc[start_pos:end_pos, E.columns.get_loc(sym)] = True
    return E


# ------------------------------------------------------------------- portfolio
def run_portfolio(dates, R, E, rebal_step, cost_bps=BASE_COST_BPS, max_names=None):
    """Each rebalance row fully specifies the weight vector (zeros included); weights
    are held between rebalances via reindex+ffill, so a name NOT re-selected is
    dropped to 0 (the earlier ffill-of-NaN bug silently kept + levered names)."""
    idx = pd.DatetimeIndex(dates)
    cols = E.columns
    rebal_pos = list(range(0, len(idx), rebal_step))
    reb_dates = idx[rebal_pos]
    Wreb = pd.DataFrame(0.0, index=reb_dates, columns=cols)
    cost = pd.Series(0.0, index=idx)
    prev_w = pd.Series(0.0, index=cols)
    for p in rebal_pos:
        rd = idx[p]
        elig = cols[E.iloc[p].values]
        w = pd.Series(0.0, index=cols)
        if len(elig) > 0:
            chosen = elig[:max_names] if (max_names and len(elig) > max_names) else elig
            w[chosen] = 1.0 / len(chosen)
        Wreb.loc[rd] = w.values
        turn = 0.5 * (w - prev_w).abs().sum()
        cost.loc[rd] = turn * cost_bps / 1e4
        prev_w = w
    W = Wreb.reindex(idx, method="ffill").fillna(0.0)
    Rf = R.reindex(columns=cols).reindex(index=idx).fillna(0.0)
    gross = (W * Rf).sum(axis=1)
    net = gross - cost
    breadth = (W > 0).sum(axis=1)
    pnl_by_name = (W * Rf).sum(axis=0)
    # leverage assertion: long-only equal-weight must sum to <=1 (+fp tolerance)
    lev = W.sum(axis=1)
    if lev.max() > 1.0 + 1e-9:
        raise AssertionError(f"gross leverage > 1: max={lev.max():.4f} (weight bug)")
    return net, breadth, W, pnl_by_name


# --------------------------------------------------------------------- metrics
def metrics(r):
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(
            sharpe=float("nan"),
            cagr=float("nan"),
            maxdd=float("nan"),
            sr_pp=float("nan"),
            n=len(r),
        )
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ANN / len(r)) - 1
    sr_pp = r.mean() / r.std()
    return dict(sharpe=sr_pp * np.sqrt(ANN), cagr=cagr, maxdd=dd, sr_pp=sr_pp, n=len(r))


def per_year(net, spy):
    out = {}
    for yr, g in net.groupby(net.index.year):
        s = metrics(g)["sharpe"]
        sb = metrics(spy.loc[g.index])["sharpe"]
        out[int(yr)] = (round(s, 3), round(sb, 3), s > sb)
    return out


# ------------------------------------------------------------------------ main
def main():
    wide, R, deep, dvol = load_returns()
    buys = load_pbuys()
    dates = R.index
    spy = (
        R["SPY"].reindex(dates).fillna(0.0)
        if "SPY" in R
        else pd.Series(0.0, index=dates)
    )
    has_agg = "AGG" in R.columns
    b6040 = (
        (0.6 * R["SPY"] + 0.4 * R["AGG"]).reindex(dates).fillna(0.0)
        if has_agg
        else None
    )
    universe = sorted(set(deep) & set(buys["symbol"].unique()) - {"SPY"})
    print(f"mill universe (deep-history & has insider buys): {len(universe)} names")
    print(
        f"dates: {dates.min().date()}..{dates.max().date()} n={len(dates)} | 60/40 avail={has_agg}"
    )

    # variant grid (the multiple-testing surface)
    grid = []
    for lb in (21, 63, 126):
        for step in (5, 21):  # weekly, monthly
            for clus in (False, True):
                grid.append(dict(lookback=lb, rebal=step, cluster=clus))

    results = []
    base = None
    for cfg in grid:
        E = build_eligibility(dates, universe, buys, cfg["lookback"], cfg["cluster"])
        net, breadth, W, pnl = run_portfolio(dates, R, E, cfg["rebal"])
        m = metrics(net)
        active = int((breadth > 0).sum())
        results.append(
            dict(
                **cfg,
                **m,
                active_days=active,
                avg_breadth=round(float(breadth[breadth > 0].mean()), 1)
                if active
                else 0,
            )
        )
        if cfg == dict(lookback=63, rebal=5, cluster=False):
            base = dict(net=net, breadth=breadth, W=W, pnl=pnl, E=E)

    res = pd.DataFrame(results)
    print("\n=== VARIANT GRID (net of base cost) ===")
    print(
        res[
            [
                "lookback",
                "rebal",
                "cluster",
                "sharpe",
                "cagr",
                "maxdd",
                "active_days",
                "avg_breadth",
            ]
        ].to_string(index=False)
    )

    bm = metrics(spy)
    print(
        f"\nSPY (full period):   sharpe={bm['sharpe']:.3f} cagr={bm['cagr']:+.3%} maxdd={bm['maxdd']:.3%}"
    )
    if b6040 is not None:
        m64 = metrics(b6040)
        print(
            f"60/40 (full period): sharpe={m64['sharpe']:.3f} cagr={m64['cagr']:+.3%} maxdd={m64['maxdd']:.3%}"
        )

    # --- focus on the pre-registered base variant ---
    net = base["net"]
    print("\n=== BASE variant (lookback=63, weekly, all-P-buys) ===")
    mb = metrics(net)
    print(
        f"full-period: sharpe={mb['sharpe']:.3f} cagr={mb['cagr']:+.3%} maxdd={mb['maxdd']:.3%} n={mb['n']}"
    )
    active_mask = base["breadth"] > 0
    ma = metrics(net[active_mask])
    print(
        f"active-only ({int(active_mask.sum())}d, {active_mask.mean():.0%} of days): "
        f"sharpe={ma['sharpe']:.3f} cagr={ma['cagr']:+.3%}"
    )
    # cost stress
    E63 = base["E"]
    net2x, _, _, _ = run_portfolio(dates, R, E63, 5, cost_bps=2 * BASE_COST_BPS)
    print(f"2x-cost: sharpe={metrics(net2x)['sharpe']:.3f}")

    print("\nper-year (port sharpe, SPY sharpe, beat?):")
    py = per_year(net, spy)
    for yr, (s, sb, beat) in py.items():
        print(f"  {yr}: {s:+.3f} vs {sb:+.3f}  {'BEAT' if beat else '.'}")
    print(f"  folds beating SPY: {sum(v[2] for v in py.values())}/{len(py)}")

    # DSR over the grid
    sr_pp = res["sr_pp"].dropna().values
    var_across = float(np.var(sr_pp, ddof=1))
    for nt in (1, len(grid), len(grid) + 11):  # pre-reg, grid, grid+phase1 looks
        d = deflated_sharpe(net, n_trials=nt, variance_across_trials=var_across)
        print(
            f"DSR n_trials={nt:3d}: prob={d.deflated_sharpe_probability:.4f} "
            f"thr={d.sharpe_threshold:.4f} sr_pp={d.sharpe_observed:.4f} passes5%={d.passes_5pct}"
        )

    # PnL concentration + liquidity split
    pnl = base["pnl"].sort_values(ascending=False)
    tot = pnl.sum()
    print(
        f"\nPnL concentration: top5 names = {pnl.head(5).sum() / tot:.0%} of gross PnL "
        f"({tot:+.2%} total); names contributing: {(pnl.abs() > 1e-9).sum()}"
    )
    print("  top names:", ", ".join(f"{k}:{v:+.2%}" for k, v in pnl.head(6).items()))
    # liquidity tercile of the universe; rebuild base signal restricted to each
    uvol = dvol.reindex(universe).dropna().sort_values()
    lo = list(uvol.index[: len(uvol) // 3])
    hi = list(uvol.index[-len(uvol) // 3 :])
    for label, sub in (("LOW-liquidity", lo), ("HIGH-liquidity", hi)):
        Es = build_eligibility(dates, sub, buys, 63, False)
        ns, _, _, _ = run_portfolio(dates, R, Es, 5)
        print(
            f"  {label} third ({len(sub)} names): sharpe={metrics(ns)['sharpe']:.3f} "
            f"cagr={metrics(ns)['cagr']:+.3%}"
        )

    out = dict(
        universe=len(universe),
        grid=results,
        base=mb,
        base_active=ma,
        spy=bm,
        per_year={str(k): v for k, v in py.items()},
        var_across_trials=var_across,
    )
    with open(
        os.path.join(os.path.dirname(__file__), "insider_mill_results.json"), "w"
    ) as f:
        json.dump(out, f, indent=2, default=str)
    print("\n[DONE] insider buy mill")


if __name__ == "__main__":
    main()
