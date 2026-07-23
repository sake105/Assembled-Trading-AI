"""PHASE-1 raw-signal probe (cheap, pre-mill GO/NO-GO).

Honest caveats (ALL bias TOWARD finding signal — so a NULL here is robust):
  * Universe = survivorship-biased survivors only (delisted names absent).
  * PEAD panel uses latest-restated EPS values (tiny look-ahead vs as-reported).
  * Forward returns use total-return-adjusted close; entry = close of the FIRST
    trading day strictly AFTER availability (no same-bar fill).
  * SUE sigma is strictly PAST-only expanding (>=6 prior errors) -> the only part
    already done PIT-clean, because it's the cheapest place a leak inflates IC.
If a candidate shows IC here it earns the full mill; if not, it's dead.
"""

from __future__ import annotations
import os
import sys
import traceback
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.assembled_core.features.pead_sue import (  # noqa: E402
    build_quarterly_eps_panel,
    quarterly_seasonal_expected,
)

HORIZONS = [5, 20, 60]


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def load_wide_close():
    px = pd.read_parquet("output/aggregates/daily.parquet")
    px["date"] = (
        pd.to_datetime(px["timestamp"], utc=True)
        .dt.tz_convert("UTC")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    wide = px.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    return wide


def fwd_excess(wide, h):
    """h-trading-day forward return per symbol, minus benchmark (SPY if present else EW mean)."""
    fwd = wide.shift(-h) / wide - 1.0
    if "SPY" in wide.columns:
        bench = fwd["SPY"]
    else:
        bench = fwd.mean(axis=1)
    return fwd.sub(bench, axis=0), ("SPY" if "SPY" in wide.columns else "EW-mean")


def next_trading_pos(index_dates, when):
    """position of first trading day strictly > `when` (a tz-naive Timestamp)."""
    pos = index_dates.searchsorted(when, side="right")
    return pos if pos < len(index_dates) else -1


def announcement_avail(xbrl, sym):
    sub = xbrl[xbrl["symbol"].astype(str).str.upper() == sym]
    g = (
        sub.dropna(subset=["period_end"])
        .groupby(sub["period_end"].dt.normalize())["available_at"]
        .min()
    )
    return g  # index = period_end (naive), value = earliest acceptance (UTC)


# ---------------------------------------------------------------- PEAD / SUE
def probe_pead(wide, xbrl):
    banner("PEAD / SUE  (event = earnings availability, hold +5/+20/+60d, SPY-excess)")
    fwd = {h: fwd_excess(wide, h) for h in HORIZONS}
    bench_name = fwd[HORIZONS[0]][1]
    idx = wide.index
    rows = []
    syms = sorted(set(xbrl["symbol"].astype(str).str.upper()) & set(wide.columns))
    for sym in syms:
        try:
            panel = build_quarterly_eps_panel(xbrl, sym)
            if len(panel) < 8:
                continue
            panel = panel.sort_values("period_end")
            actual = pd.Series(
                panel["eps"].astype(float).values,
                index=pd.to_datetime(panel["period_end"]),
            )
            expected = quarterly_seasonal_expected(panel)
            df = pd.DataFrame({"actual": actual}).join(expected.rename("expected"))
            df = df.dropna()
            if len(df) < 8:
                continue
            fe = (df["actual"] - df["expected"]).sort_index()
            sigma_past = fe.shift(1).expanding(min_periods=6).std()
            sue = (fe / sigma_past).dropna()
            if sue.empty:
                continue
            avail = announcement_avail(xbrl, sym)  # period_end -> earliest acceptance
            for pe, s in sue.items():
                a = avail.get(pd.Timestamp(pe).normalize())
                if pd.isna(a):
                    continue
                when = pd.Timestamp(a).tz_convert("UTC").normalize().tz_localize(None)
                pos = next_trading_pos(idx, when)
                if pos < 0:
                    continue
                entry = idx[pos]
                rec = {"sym": sym, "date": entry, "year": entry.year, "sue": float(s)}
                ok = True
                for h in HORIZONS:
                    v = (
                        fwd[h][0].at[entry, sym]
                        if (entry in fwd[h][0].index and sym in fwd[h][0].columns)
                        else np.nan
                    )
                    if pd.isna(v):
                        ok = False
                    rec[f"r{h}"] = v
                if ok:
                    rows.append(rec)
        except Exception:
            print(f"[PEAD-FAIL] {sym}")
            traceback.print_exc()
    ev = pd.DataFrame(rows)
    if ev.empty:
        print("NO EVENTS")
        return ev
    print(
        f"benchmark={bench_name}  n_events={len(ev)}  n_symbols={ev['sym'].nunique()}  "
        f"years={ev['year'].min()}-{ev['year'].max()}"
    )
    for h in HORIZONS:
        col = f"r{h}"
        ic = ev["sue"].corr(ev[col], method="spearman")
        # tercile spread (top - bottom by SUE)
        q = ev["sue"].rank(pct=True)
        top = ev[q >= 2 / 3][col].mean()
        bot = ev[q <= 1 / 3][col].mean()
        print(
            f"  +{h:>2}d: SpearmanIC={ic:+.4f}  topT={top:+.4%}  botT={bot:+.4%}  spread={top - bot:+.4%}"
        )
    print("  by-year Spearman IC (r60):")
    for yr, g in ev.groupby("year"):
        if len(g) >= 20:
            print(
                f"    {yr}: n={len(g):4d}  IC={g['sue'].corr(g['r60'], method='spearman'):+.4f}  "
                f"meanR60={g['r60'].mean():+.4%}"
            )
    return ev


# ---------------------------------------------------------------- INSIDER
def probe_insider(wide):
    banner("INSIDER open-market BUYS (Form 4 'P'), hold +20/+60d, SPY-excess")
    ins = pd.read_parquet("data/raw/insider_congress/form4_insider_full.parquet")
    buys = ins[
        (ins["transaction_type"] == "P")
        & (~ins["is_derivative"])
        & (ins["price"] > 0)
        & (ins["shares"] > 0)
    ].copy()
    buys["when"] = (
        pd.to_datetime(buys["available_at"], utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    # cluster flag: >=2 distinct insiders buying same symbol within trailing 30d
    buys = buys.sort_values("when")
    fwd = {h: fwd_excess(wide, h)[0] for h in (20, 60)}
    idx = wide.index
    rows = []
    for sym, g in buys.groupby("symbol"):
        if sym not in wide.columns:
            continue
        g = g.sort_values("when")
        for when, gg in g.groupby("when"):
            pos = next_trading_pos(idx, when)
            if pos < 0:
                continue
            entry = idx[pos]
            # cluster: distinct owners in (when-30d, when]
            win = g[(g["when"] > when - pd.Timedelta(days=30)) & (g["when"] <= when)]
            n_owners = win["reporting_owner_cik"].nunique()
            rec = {
                "sym": sym,
                "date": entry,
                "year": entry.year,
                "cluster": int(n_owners >= 2),
                "n_owners": n_owners,
                "value": float(win["value_usd"].sum()),
            }
            ok = True
            for h in (20, 60):
                v = (
                    fwd[h].at[entry, sym]
                    if (entry in fwd[h].index and sym in fwd[h].columns)
                    else np.nan
                )
                if pd.isna(v):
                    ok = False
                rec[f"r{h}"] = v
            if ok:
                rows.append(rec)
    ev = pd.DataFrame(rows)
    if ev.empty:
        print("NO EVENTS")
        return ev
    # de-dup to one row per (sym, when-cluster) already grouped by when
    print(
        f"n_buy_events={len(ev)}  n_symbols={ev['sym'].nunique()}  years={ev['year'].min()}-{ev['year'].max()}"
    )
    for h in (20, 60):
        c = f"r{h}"
        print(
            f"  +{h}d ALL   : mean={ev[c].mean():+.4%} median={ev[c].median():+.4%} n={len(ev)} "
            f"t={ev[c].mean() / (ev[c].std() / np.sqrt(len(ev))):+.2f}"
        )
        cl = ev[ev["cluster"] == 1]
        if len(cl) >= 10:
            print(
                f"  +{h}d CLUST : mean={cl[c].mean():+.4%} median={cl[c].median():+.4%} n={len(cl)} "
                f"t={cl[c].mean() / (cl[c].std() / np.sqrt(len(cl))):+.2f}"
            )
    return ev


# ---------------------------------------------------------------- CONGRESS
def probe_congress(wide):
    banner("CONGRESS BUYS (STOCK-Act PTR), hold +20/+60d, SPY-excess")
    c = pd.read_parquet("data/raw/insider_congress/congress_trades_full.parquet")
    buys = c[c["type"] == "buy"].copy()
    buys["when"] = (
        pd.to_datetime(buys["available_at"], utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    fwd = {h: fwd_excess(wide, h)[0] for h in (20, 60)}
    idx = wide.index
    rows = []
    for _, r in buys.iterrows():
        sym = str(r["symbol"]).upper()
        if sym not in wide.columns:
            continue
        pos = next_trading_pos(idx, r["when"])
        if pos < 0:
            continue
        entry = idx[pos]
        rec = {
            "sym": sym,
            "date": entry,
            "year": entry.year,
            "amount": float(r.get("amount", np.nan)),
        }
        ok = True
        for h in (20, 60):
            v = (
                fwd[h].at[entry, sym]
                if (entry in fwd[h].index and sym in fwd[h].columns)
                else np.nan
            )
            if pd.isna(v):
                ok = False
            rec[f"r{h}"] = v
        if ok:
            rows.append(rec)
    ev = pd.DataFrame(rows)
    if ev.empty:
        print("NO EVENTS")
        return ev
    print(
        f"n_buy_events={len(ev)}  n_symbols={ev['sym'].nunique()}  years={ev['year'].min()}-{ev['year'].max()}"
    )
    for h in (20, 60):
        cc = f"r{h}"
        print(
            f"  +{h}d ALL  : mean={ev[cc].mean():+.4%} median={ev[cc].median():+.4%} n={len(ev)} "
            f"t={ev[cc].mean() / (ev[cc].std() / np.sqrt(len(ev))):+.2f}"
        )
        big = ev[ev["amount"] >= 50000]
        if len(big) >= 10:
            print(
                f"  +{h}d >50k : mean={big[cc].mean():+.4%} median={big[cc].median():+.4%} n={len(big)} "
                f"t={big[cc].mean() / (big[cc].std() / np.sqrt(len(big))):+.2f}"
            )
    return ev


if __name__ == "__main__":
    wide = load_wide_close()
    print(
        f"price wide: {wide.shape}, {wide.index.min().date()}..{wide.index.max().date()}, "
        f"SPY present={'SPY' in wide.columns}"
    )
    xbrl = pd.read_parquet("data/raw/fundamentals/fundamentals_xbrl_full.parquet")
    probe_pead(wide, xbrl)
    probe_insider(wide)
    probe_congress(wide)
    print("\n[DONE] raw-signal probe")
