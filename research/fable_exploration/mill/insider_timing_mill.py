"""PHASE-3 mill for H5 — aggregate insider net-buy ratio as a SPY market-timing /
defensive overlay. Survivorship-robust (trades SPY only) and testable over the FULL
2004-2026 span (needs only SPY prices since 1984 + insider timestamps since 2003).

Decisive cuts:
  * DETRENDED signal (rolling z-score, past-only) — the raw ratio is non-stationary
    (insider selling grew structurally; an expanding-median threshold is an artifact).
  * exclude-2020 (is the edge just the COVID insider-bottom?).
  * sub-period halves + per-year.
  * CONSTANT-exposure control (same average exposure, no timing) — proves timing adds
    beyond merely de-risking.
  * comparison + COMBINATION with a plain realized-vol overlay (the incumbent's family).
  * non-overlapping monthly regression t-stat; DSR with cumulative trial count.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

ANN = 252
START, END = pd.Timestamp("2004-01-01"), pd.Timestamp("2026-06-10")


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(sharpe=float("nan"), cagr=float("nan"), maxdd=float("nan"))
    eq = (1 + r).cumprod()
    return dict(
        sharpe=float(r.mean() / r.std() * np.sqrt(ANN)),
        cagr=float(eq.iloc[-1] ** (ANN / len(r)) - 1),
        maxdd=float((eq / eq.cummax() - 1).min()),
    )


def show(name, r):
    m = metrics(r)
    print(
        f"  {name:<28}: sharpe={m['sharpe']:.3f} cagr={m['cagr']:+.3%} maxdd={m['maxdd']:.3%}"
    )
    return m


def main():
    px = pd.read_parquet("output/aggregates/daily.parquet")
    px["date"] = (
        pd.to_datetime(px["timestamp"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    spy = px[px["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    spy = spy.loc[(spy.index >= START) & (spy.index <= END)]
    R = spy.pct_change(fill_method=None).dropna()
    tdays = R.index
    print(f"SPY {tdays.min().date()}..{tdays.max().date()} n={len(tdays)}")

    ins = pd.read_parquet("data/raw/insider_congress/form4_insider_full.parquet")
    ins["when"] = (
        pd.to_datetime(ins["available_at"], utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    ins = ins[(ins["when"] >= START - pd.Timedelta(days=200)) & (ins["when"] <= END)]
    p = ins[ins["transaction_type"] == "P"].groupby("when").size().rename("P")
    s = ins[ins["transaction_type"] == "S"].groupby("when").size().rename("S")
    daily = (
        pd.concat([p, s], axis=1)
        .reindex(pd.date_range(START - pd.Timedelta(days=200), END, freq="D"))
        .fillna(0.0)
    )
    roll = daily.rolling("63D").sum()
    ratio = roll["P"] / (roll["P"] + roll["S"]).replace(0, np.nan)
    # DETREND: past-only rolling z-score (2y window), then lag 1 day
    rmean = ratio.rolling("504D").mean()
    rstd = ratio.rolling("504D").std()
    z = ((ratio - rmean) / rstd).reindex(tdays, method="ffill").shift(1)

    # exposure mappings (parameter-light, capped long-only no-leverage)
    expo_ins = (0.5 + 0.5 * z).clip(0, 1)  # +1z->full, -1z->cash
    realized = R.rolling(20).std() * np.sqrt(ANN)
    expo_vt = (0.12 / realized).clip(0, 1).shift(1)  # 12% vol target, no leverage
    expo_combo = (expo_ins * expo_vt).clip(0, 1)

    print("\n=== full period 2004-2026 ===")
    show("SPY buy-hold", R)
    m_ins = show("insider-timed", R * expo_ins)
    show("vol-target only", R * expo_vt)
    show("insider x vol-target", R * expo_combo)
    # control: constant exposure = mean(expo_ins) -> isolates TIMING from de-risking
    const = pd.Series(float(expo_ins.reindex(tdays).fillna(0).mean()), index=tdays)
    show(f"CONTROL const-expo({const.iloc[0]:.2f})", R * const)

    print("\n=== exclude 2020 (is it just the COVID bottom?) ===")
    mask = tdays.year != 2020
    show("SPY ex-2020", R[mask])
    show("insider-timed ex-2020", (R * expo_ins)[mask])

    print("\n=== sub-period halves ===")
    mid = pd.Timestamp("2015-06-01")
    for label, mk in (("2004-2015", tdays < mid), ("2015-2026", tdays >= mid)):
        ms = metrics(R[mk])
        mi = metrics((R * expo_ins)[mk])
        print(
            f"  {label}: SPY sharpe={ms['sharpe']:.3f} | insider-timed={mi['sharpe']:.3f} "
            f"(maxdd {mi['maxdd']:.1%} vs {ms['maxdd']:.1%})"
        )

    print("\n=== per-year (insider-timed sharpe vs SPY) ===")
    beats = 0
    nyr = 0
    strat = R * expo_ins
    for yr, g in strat.groupby(strat.index.year):
        si = metrics(g)["sharpe"]
        sb = metrics(R.loc[g.index])["sharpe"]
        b = si > sb
        beats += b
        nyr += 1
        print(f"  {yr}: {si:+.2f} vs {sb:+.2f} {'BEAT' if b else '.'}")
    print(f"  folds beating SPY: {beats}/{nyr}")

    print("\n=== non-overlapping MONTHLY signal -> next-month SPY return ===")
    sig_m = z.reindex(tdays).resample("ME").last()
    spy_m = (spy.reindex(tdays).resample("ME").last().pct_change()).shift(-1)
    d = pd.concat([sig_m.rename("z"), spy_m.rename("fwd")], axis=1).dropna()
    x = d["z"].values
    y = d["fwd"].values
    b1 = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    resid = y - (y.mean() + b1 * (x - x.mean()))
    se = np.sqrt((resid @ resid) / (len(x) - 2)) / (
        np.std(x, ddof=1) * np.sqrt(len(x) - 1)
    )
    print(
        f"  n_months={len(d)} slope={b1:+.4f} t={b1 / se:+.2f} "
        f"Spearman={pd.Series(x).corr(pd.Series(y), method='spearman'):+.3f}"
    )

    print("\n=== DSR (cumulative trial count ~30) ===")
    for nt in (1, 30):
        dd = deflated_sharpe(strat.dropna(), n_trials=nt)
        print(
            f"  n_trials={nt}: prob={dd.deflated_sharpe_probability:.4f} "
            f"sr_pp={dd.sharpe_observed:.4f} passes5%={dd.passes_5pct}"
        )
    print("\n[DONE] insider timing mill")


if __name__ == "__main__":
    main()
