"""PHASE-3 mill for H6 — long-low-short-flow tilt (FINRA RegSHO short-volume level).
Decisive cuts: long-low vs long-high vs equal-weight-universe (does the tilt ADD?),
beta/low-vol control (is it just low-beta in disguise?), costs, per-year, DSR.
Assembles all short_volume_*.parquet chunks. Run after the background pulls finish."""

from __future__ import annotations
import glob
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from insider_buy_mill import metrics, run_portfolio  # noqa: E402
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

ANN = 252
START, END = pd.Timestamp("2018-01-01"), pd.Timestamp("2026-06-10")
MIN_HISTORY = 1000


def load():
    files = sorted(glob.glob("research/fable_exploration/data/short_volume_*.parquet"))
    print(f"short-volume chunks: {len(files)}")
    sv = pd.concat(
        [pd.read_parquet(f) for f in files], ignore_index=True
    ).drop_duplicates(["date", "symbol"])
    sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
    sv["ratio"] = sv["short_volume"] / sv["total_volume"].where(sv["total_volume"] > 0)
    ratio = sv.pivot_table(
        index="date", columns="symbol", values="ratio", aggfunc="last"
    ).sort_index()

    px = pd.read_parquet("output/aggregates/daily.parquet")
    px["date"] = (
        pd.to_datetime(px["timestamp"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    close = px.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    close = close.loc[(close.index >= START) & (close.index <= END)]
    R = close.pct_change(fill_method=None)
    deep = [c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY]
    print(
        f"short-vol: {ratio.shape} {ratio.index.min().date()}..{ratio.index.max().date()}"
    )
    return ratio, close, R, deep


def tercile_eligibility(dates, universe, signal, which):
    """which in {'low','high'}: eligible = bottom/top third of cross-sectional signal."""
    idx = pd.DatetimeIndex(dates)
    sig = signal.reindex(index=idx, columns=universe)
    E = pd.DataFrame(False, index=idx, columns=universe)
    for d in idx:
        s = sig.loc[d].dropna()
        if s.size < 15:
            continue
        q = s.rank(pct=True)
        sel = (q <= 1 / 3) if which == "low" else (q >= 2 / 3)
        E.loc[d, s.index[sel.values]] = True
    return E


def main():
    ratio, close, R, deep = load()
    dates = R.index
    universe = sorted(set(deep) & set(ratio.columns) - {"SPY"})
    spy = R["SPY"].reindex(dates).fillna(0.0)
    has_agg = "AGG" in R.columns
    b6040 = (
        (0.6 * R["SPY"] + 0.4 * R["AGG"]).reindex(dates).fillna(0.0)
        if has_agg
        else None
    )
    print(
        f"universe={len(universe)} dates {dates.min().date()}..{dates.max().date()} n={len(dates)}"
    )

    # signal = trailing-21d mean short-flow level, lagged 1d (PIT: RegSHO t available t+1)
    signal = ratio.rolling(21, min_periods=10).mean().shift(1)

    print("\n=== baselines ===")
    print("  SPY:           ", metrics(spy))
    if b6040 is not None:
        print("  60/40:         ", metrics(b6040))
    Eall = pd.DataFrame(True, index=dates, columns=universe)
    ew, _, _, _ = run_portfolio(dates, R, Eall, 21)
    print("  EW-universe:   ", metrics(ew))

    print("\n=== short-flow tercile baskets (monthly rebal, 20bps) ===")
    nets = {}
    for which in ("low", "high"):
        E = tercile_eligibility(dates, universe, signal, which)
        net, breadth, W, pnl = run_portfolio(dates, R, E, 21)
        nets[which] = net
        m = metrics(net)
        print(
            f"  {which:>4}-short-flow: sharpe={m['sharpe']:.3f} cagr={m['cagr']:+.3%} "
            f"maxdd={m['maxdd']:.3%} breadth~{int((W > 0).sum(axis=1)[(W > 0).sum(axis=1) > 0].mean())}"
        )
    ls = nets["low"] - nets["high"]
    print(
        f"  L/S (low-high): sharpe={metrics(ls)['sharpe']:.3f} cagr={metrics(ls)['cagr']:+.3%}"
    )

    low = nets["low"]
    print("\n=== does the tilt ADD over EW-universe? ===")
    print(
        f"  low-short-flow Sharpe {metrics(low)['sharpe']:.3f}  vs  EW-universe {metrics(ew)['sharpe']:.3f}"
    )

    print("\n=== control: is low-short-flow just LOW-BETA? ===")
    betas = {}
    for s in universe:
        r = R[s].reindex(dates)
        df = pd.concat([r, spy], axis=1).dropna()
        if len(df) > 100:
            betas[s] = np.cov(df.iloc[:, 0], df.iloc[:, 1])[0, 1] / np.var(
                df.iloc[:, 1]
            )
    betas = pd.Series(betas)
    mean_sf = signal.reindex(columns=universe).mean()
    j = pd.concat([mean_sf.rename("sf"), betas.rename("beta")], axis=1).dropna()
    print(
        f"  corr(mean short-flow, beta) = {j['sf'].corr(j['beta'], method='spearman'):+.3f} "
        f"(if strongly +, the tilt ~ low-beta)"
    )

    print("\n=== cost stress (low-short-flow) ===")
    E = tercile_eligibility(dates, universe, signal, "low")
    for cb in (20, 40, 60):
        net, _, _, _ = run_portfolio(dates, R, E, 21, cost_bps=cb)
        print(f"  {cb}bps: sharpe={metrics(net)['sharpe']:.3f}")

    print("\n=== per-year (low-short-flow vs SPY) ===")
    beats = nyr = 0
    for yr, g in low.groupby(low.index.year):
        si = metrics(g)["sharpe"]
        sb = metrics(spy.loc[g.index])["sharpe"]
        b = si > sb
        beats += b
        nyr += 1
        print(f"  {yr}: {si:+.2f} vs {sb:+.2f} {'BEAT' if b else '.'}")
    print(f"  folds beating SPY: {beats}/{nyr}")

    print("\n=== DSR (cumulative ~34 trials) ===")
    for nt in (1, 34):
        d = deflated_sharpe(low.dropna(), n_trials=nt)
        print(
            f"  n_trials={nt}: prob={d.deflated_sharpe_probability:.4f} "
            f"sr_pp={d.sharpe_observed:.4f} passes5%={d.passes_5pct}"
        )
    print("\n[DONE] short-flow mill")


if __name__ == "__main__":
    main()
