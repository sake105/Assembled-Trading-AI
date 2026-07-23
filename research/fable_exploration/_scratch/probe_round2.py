"""ROUND-2 probe: timing/market-level ideas that DODGE the killer biases
(concentration, illiquidity, survivorship) — because they time SPY or hold broad
predictable windows, not illiquid single names.

H4 earnings-announcement premium (pre-announcement window).
H5 aggregate insider net-buy ratio -> SPY market timing / defensive overlay.

Same honesty rails as round 1; cumulative trial count carried into any later DSR.
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.assembled_core.features.pead_sue import build_quarterly_eps_panel  # noqa: E402

ANN = 252
START, END = pd.Timestamp("2018-01-01"), pd.Timestamp("2026-06-10")


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def load_wide():
    px = pd.read_parquet("output/aggregates/daily.parquet")
    px["date"] = (
        pd.to_datetime(px["timestamp"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    wide = px.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    return wide.loc[(wide.index >= START) & (wide.index <= END)]


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return dict(sharpe=float("nan"), cagr=float("nan"), maxdd=float("nan"))
    eq = (1 + r).cumprod()
    return dict(
        sharpe=r.mean() / r.std() * np.sqrt(ANN),
        cagr=eq.iloc[-1] ** (ANN / len(r)) - 1,
        maxdd=(eq / eq.cummax() - 1).min(),
    )


# ----------------------------------------------------------------- H4
def probe_announcement_premium(wide, xbrl):
    banner("H4 earnings-announcement PREMIUM (pre-window [-5,-1], SPY-excess)")
    idx = wide.index
    R = wide.pct_change(fill_method=None)
    spy = R["SPY"]
    syms = sorted(
        set(xbrl["symbol"].astype(str).str.upper()) & set(wide.columns) - {"SPY"}
    )
    rows = []
    for sym in syms:
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
        for pe, av in ann.items():
            if pe not in pes:
                continue
            a = pd.Timestamp(av).tz_convert("UTC").normalize().tz_localize(None)
            pos = idx.searchsorted(a, side="left")
            if pos < 7 or pos >= len(idx):
                continue
            # pre-window: hold [-5,-1] trading days, exit the day BEFORE the report
            pre = wide[sym].iloc[pos - 1] / wide[sym].iloc[pos - 6] - 1
            pre_spy = wide["SPY"].iloc[pos - 1] / wide["SPY"].iloc[pos - 6] - 1
            # announcement gap [-1,+1]
            if pos + 1 < len(idx):
                gap = wide[sym].iloc[pos + 1] / wide[sym].iloc[pos - 1] - 1
                gap_spy = wide["SPY"].iloc[pos + 1] / wide["SPY"].iloc[pos - 1] - 1
            else:
                gap = gap_spy = np.nan
            rows.append(
                dict(sym=sym, year=idx[pos].year, pre=pre - pre_spy, gap=gap - gap_spy)
            )
    ev = pd.DataFrame(rows).dropna(subset=["pre"])
    if ev.empty:
        print("NO EVENTS")
        return
    n = len(ev)
    for c in ("pre", "gap"):
        s = ev[c].dropna()
        t = s.mean() / (s.std() / np.sqrt(len(s)))
        print(
            f"  {c:>4}: mean={s.mean():+.4%} median={s.median():+.4%} n={len(s)} t={t:+.2f}"
        )
    print(f"  n_events={n}, {ev['year'].min()}-{ev['year'].max()}")
    print("  pre-window mean by year:")
    for yr, g in ev.groupby("year"):
        if len(g) >= 20:
            t = g["pre"].mean() / (g["pre"].std() / np.sqrt(len(g)))
            print(f"    {yr}: n={len(g):4d} mean={g['pre'].mean():+.4%} t={t:+.2f}")


# ----------------------------------------------------------------- H5
def probe_aggregate_insider_timing(wide):
    banner("H5 aggregate insider net-buy ratio P/(P+S) -> SPY market timing")
    ins = pd.read_parquet("data/raw/insider_congress/form4_insider_full.parquet")
    ins["when"] = (
        pd.to_datetime(ins["available_at"], utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    ins = ins[(ins["when"] >= START - pd.Timedelta(days=120)) & (ins["when"] <= END)]
    p = ins[ins["transaction_type"] == "P"].groupby("when").size().rename("P")
    s = ins[ins["transaction_type"] == "S"].groupby("when").size().rename("S")
    daily = pd.concat([p, s], axis=1).fillna(0.0)
    daily = daily.reindex(pd.date_range(daily.index.min(), END, freq="D")).fillna(0.0)
    # trailing-63d counts -> net-buy ratio (past-only; uses only data up to t)
    roll = daily.rolling("63D").sum()
    ratio = roll["P"] / (roll["P"] + roll["S"]).replace(0, np.nan)
    # align to trading days
    R = wide.pct_change(fill_method=None)["SPY"].dropna()
    tdays = R.index
    sig = ratio.reindex(tdays, method="ffill")
    # signal as of yesterday drives today's exposure (no look-ahead)
    sig_lag = sig.shift(1)
    # forward 1m SPY return per day for correlation
    fwd1m = wide["SPY"].pct_change(21, fill_method=None).shift(-21).reindex(tdays)
    valid = sig_lag.notna() & fwd1m.notna()
    ic = sig_lag[valid].corr(fwd1m[valid], method="spearman")
    print(
        f"  Spearman(net-buy ratio_t-1, fwd-21d SPY) = {ic:+.4f}  n={int(valid.sum())}"
    )
    # contemporaneous fwd-1m by signal tercile
    df = pd.DataFrame({"sig": sig_lag, "fwd": fwd1m}).dropna()
    q = df["sig"].rank(pct=True)
    print(
        f"  fwd-21d SPY: top-third sig={df[q >= 2 / 3]['fwd'].mean():+.4%}  "
        f"bot-third={df[q <= 1 / 3]['fwd'].mean():+.4%}"
    )
    # timing overlays vs buy-hold SPY
    med = sig_lag.expanding(min_periods=252).median()
    print("\n  SPY buy-hold: ", metrics(R))
    for name, expo in (
        ("binary>median (1/0)", (sig_lag > med).astype(float)),
        ("binary>median (1/0.5)", (sig_lag > med).astype(float) * 0.5 + 0.5),
        (
            "percentile-scaled",
            sig_lag.expanding(min_periods=252).apply(
                lambda x: (x.rank(pct=True).iloc[-1]) if len(x) else np.nan, raw=False
            )
            if False
            else sig_lag.rank(pct=True),
        ),
    ):
        e = expo.reindex(tdays).fillna(0.0).clip(0, 1)
        strat = (R * e).dropna()
        m = metrics(strat)
        invested = float((e > 0).mean())
        print(
            f"  overlay [{name:>22}]: sharpe={m['sharpe']:.3f} cagr={m['cagr']:+.3%} "
            f"maxdd={m['maxdd']:.3%} invested={invested:.0%}"
        )


if __name__ == "__main__":
    wide = load_wide()
    print(
        f"prices {wide.shape} {wide.index.min().date()}..{wide.index.max().date()} SPY={'SPY' in wide.columns}"
    )
    xbrl = pd.read_parquet("data/raw/fundamentals/fundamentals_xbrl_full.parquet")
    probe_announcement_premium(wide, xbrl)
    probe_aggregate_insider_timing(wide)
    print("\n[DONE] round-2 probe")
