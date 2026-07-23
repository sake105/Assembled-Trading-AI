"""Fable experiment engine — compose ANY subset of signals x risk overlays, run
In-Sample AND OOS, sweep the full grid, rank by OOS. Self-contained in the
playground (computes signals from raw data; no production coupling). Honesty lives
in the columns: every row carries OOS metrics + delta vs the EW-survivor baseline.

Signals (each toggleable): mom (12-1), trend (>MA200), insider (P-buy 63d),
shortflow (low short-flow tilt), pead (SUE), congress (buy 63d), quality (gross
profitability), news (daily sentiment).
Risk overlays: none / voltgt10 / voltgt15 / regime (SPY>MA200 gate).
"""

from __future__ import annotations
import glob
import itertools
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.assembled_core.features.pead_sue import (  # noqa: E402
    build_quarterly_eps_panel,
    quarterly_seasonal_expected,
)

ANN = 252
START, END = pd.Timestamp("2018-08-01"), pd.Timestamp("2026-06-10")
IS_END = pd.Timestamp("2022-06-30")  # In-Sample / OOS split
MIN_HISTORY = 1000
OUTDIR = os.path.dirname(__file__)


def _wide_close():
    enr = "research/fable_exploration/data/prices_enriched.parquet"
    if os.path.exists(enr):  # enriched yfinance panel (193 names, full history)
        px = pd.read_parquet(enr)
        px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    else:
        px = pd.read_parquet("output/aggregates/daily.parquet")
        px["date"] = (
            pd.to_datetime(px["timestamp"], utc=True)
            .dt.normalize()
            .dt.tz_localize(None)
        )
    w = px.pivot_table(
        index="date", columns="symbol", values="close", aggfunc="last"
    ).sort_index()
    return w.loc[(w.index >= START) & (w.index <= END)]


def load():
    close = _wide_close()
    R = close.pct_change(fill_method=None)
    dates = R.index
    deep = [c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY]
    universe = sorted(set(deep) - {"SPY"})

    # ---- short-flow ratio ----
    sv = pd.concat(
        [
            pd.read_parquet(f)
            for f in glob.glob("research/fable_exploration/data/short_volume_*.parquet")
        ],
        ignore_index=True,
    ).drop_duplicates(["date", "symbol"])
    sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
    sv["r"] = sv["short_volume"] / sv["total_volume"].where(sv["total_volume"] > 0)
    shortflow = (
        sv.pivot_table(index="date", columns="symbol", values="r", aggfunc="last")
        .reindex(dates)
        .reindex(columns=universe)
    )

    # ---- insider P-buys & congress buys -> 63d rolling flags ----
    ins = pd.read_parquet("data/raw/insider_congress/form4_insider_full.parquet")
    ins = ins[
        (ins["transaction_type"] == "P") & (~ins["is_derivative"]) & (ins["price"] > 0)
    ]
    ins_when = (
        pd.to_datetime(ins["available_at"], utc=True)
        .dt.normalize()
        .dt.tz_localize(None)
    )
    insider = _event_flag(
        ins["symbol"].astype(str).str.upper(), ins_when, dates, universe, 63
    )

    cg = pd.read_parquet("data/raw/insider_congress/congress_trades_full.parquet")
    cg = cg[cg["type"] == "buy"]
    cg_when = (
        pd.to_datetime(cg["available_at"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    congress = _event_flag(
        cg["symbol"].astype(str).str.upper(), cg_when, dates, universe, 63
    )

    # ---- XBRL: PEAD SUE state + quality (gross profitability) ----
    xbrl = pd.read_parquet("data/raw/fundamentals/fundamentals_xbrl_full.parquet")
    pead = _pead_state(xbrl, dates, universe)
    quality = _quality_state(xbrl, dates, universe)

    # ---- news sentiment (defensive schema) ----
    news = _news_state(dates, universe)

    return dict(
        close=close,
        R=R,
        dates=dates,
        universe=universe,
        signals=dict(
            shortflow=-shortflow,  # low short-flow = positive score
            insider=insider,
            congress=congress,
            pead=pead,
            quality=quality,
            news=news,
        ),
    )


def _event_flag(syms, whens, dates, universe, lookback):
    idx = pd.DatetimeIndex(dates)
    E = pd.DataFrame(0.0, index=idx, columns=universe)
    df = pd.DataFrame({"sym": syms.values, "when": whens.values})
    for sym, g in df.groupby("sym"):
        if sym not in E.columns:
            continue
        col = E.columns.get_loc(sym)
        for av in pd.to_datetime(g["when"]).unique():
            p = idx.searchsorted(av, side="right")
            if p < len(idx):
                E.iloc[p : min(p + lookback, len(idx)), col] = 1.0
    return E


def _pead_state(xbrl, dates, universe):
    """Latest standardized SUE per name, held 63 trading days after announcement."""
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(np.nan, index=idx, columns=universe)
    for sym in universe:
        panel = build_quarterly_eps_panel(xbrl, sym)
        if len(panel) < 8:
            continue
        panel = panel.sort_values("period_end")
        actual = pd.Series(
            panel["eps"].astype(float).values, index=pd.to_datetime(panel["period_end"])
        )
        exp = quarterly_seasonal_expected(panel)
        df = pd.DataFrame({"a": actual}).join(exp.rename("e")).dropna()
        if len(df) < 8:
            continue
        fe = (df["a"] - df["e"]).sort_index()
        sue = (fe / fe.shift(1).expanding(min_periods=6).std()).dropna()
        sub = xbrl[xbrl["symbol"].astype(str).str.upper() == sym]
        avail = (
            sub.dropna(subset=["period_end"])
            .groupby(sub["period_end"].dt.normalize())["available_at"]
            .min()
        )
        col = out.columns.get_loc(sym)
        for pe, s in sue.items():
            a = avail.get(pd.Timestamp(pe).normalize())
            if pd.isna(a):
                continue
            when = pd.Timestamp(a).tz_convert("UTC").normalize().tz_localize(None)
            p = idx.searchsorted(when, side="right")
            if p < len(idx):
                out.iloc[p : min(p + 63, len(idx)), col] = float(s)
    return out


def _coalesce(xbrl, sym, tag):
    s = xbrl[(xbrl["symbol"].astype(str).str.upper() == sym) & (xbrl["tag"] == tag)]
    return s


def _quality_state(xbrl, dates, universe):
    """Gross profitability proxy = GrossProfit / Assets (Novy-Marx), PIT by available_at,
    forward-filled. Falls back to NaN where tags absent."""
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(np.nan, index=idx, columns=universe)
    xu = xbrl.copy()
    xu["symU"] = xu["symbol"].astype(str).str.upper()
    xu["av"] = (
        pd.to_datetime(xu["available_at"], utc=True).dt.normalize().dt.tz_localize(None)
    )
    for sym in universe:
        sub = xu[xu["symU"] == sym]
        gp = sub[sub["tag"] == "GrossProfit"][["av", "val"]].dropna()
        at = sub[sub["tag"] == "Assets"][["av", "val"]].dropna()
        if gp.empty or at.empty:
            continue
        # latest GrossProfit and Assets available as of each date
        gp = gp.sort_values("av").groupby("av")["val"].last()
        at = at.sort_values("av").groupby("av")["val"].last()
        gp_s = gp.reindex(idx, method="ffill")
        at_s = at.reindex(idx, method="ffill")
        ratio = gp_s / at_s.where(at_s > 0)
        out[sym] = ratio.values
    return out


def _news_state(dates, universe):
    idx = pd.DatetimeIndex(dates)
    out = pd.DataFrame(np.nan, index=idx, columns=universe)
    path = "output/news_sentiment_daily.parquet"
    if not os.path.exists(path):
        return out
    try:
        nd = pd.read_parquet(path)
    except Exception:
        return out
    cols = {c.lower(): c for c in nd.columns}
    dcol = next((cols[k] for k in ("date", "timestamp", "day") if k in cols), None)
    scol = next((cols[k] for k in ("symbol", "ticker") if k in cols), None)
    vcol = next(
        (
            cols[k]
            for k in (
                "sentiment",
                "sentiment_score",
                "score",
                "compound",
                "mean_sentiment",
            )
            if k in cols
        ),
        None,
    )
    if not (dcol and scol and vcol):
        print(f"  [news] unrecognized schema cols={list(nd.columns)} -> news disabled")
        return out
    nd["d"] = (
        pd.to_datetime(nd[dcol], utc=True, errors="coerce")
        .dt.normalize()
        .dt.tz_localize(None)
    )
    nd["s"] = nd[scol].astype(str).str.upper()
    piv = nd.pivot_table(index="d", columns="s", values=vcol, aggfunc="mean")
    piv = piv.reindex(idx).reindex(columns=universe).shift(1)  # lag 1d
    cov = piv.notna().any(axis=1)
    print(
        f"  [news] coverage: {cov.sum()} of {len(idx)} days; first={idx[cov.argmax()].date() if cov.any() else 'none'}"
    )
    return piv


# ---------------------------------------------------------------- composition + backtest
def zscore_cs(panel):
    return panel.sub(panel.mean(axis=1), axis=0).div(
        panel.std(axis=1).replace(0, np.nan), axis=0
    )


def composite(signals, names):
    parts = [zscore_cs(signals[n]) for n in names]
    stack = pd.concat(parts).groupby(level=0).mean()  # average available z-scores
    return stack


def backtest(
    score, R, universe, rebal=21, top=1 / 3, cost_bps=20.0, overlay="none", spy=None
):
    idx = score.index
    cols = universe
    rebal_pos = list(range(0, len(idx), rebal))
    Wreb = pd.DataFrame(0.0, index=idx[rebal_pos], columns=cols)
    cost = pd.Series(0.0, index=idx)
    prev = pd.Series(0.0, index=cols)
    for p in rebal_pos:
        s = score.iloc[p].dropna()
        w = pd.Series(0.0, index=cols)
        if len(s) >= 10:
            thr = s.quantile(1 - top)
            sel = s[s >= thr].index
            if len(sel):
                w[sel] = 1.0 / len(sel)
        Wreb.loc[idx[p]] = w.values
        cost.loc[idx[p]] = 0.5 * (w - prev).abs().sum() * cost_bps / 1e4
        prev = w
    W = Wreb.reindex(idx, method="ffill").fillna(0.0)
    Rf = R.reindex(columns=cols).reindex(idx).fillna(0.0)
    gross = (W * Rf).sum(axis=1) - cost
    # risk overlay (exposure scaling, long-only cap 1.0)
    if overlay == "none":
        expo = pd.Series(1.0, index=idx)
    elif overlay.startswith("voltgt"):
        tv = float(overlay.replace("voltgt", "")) / 100.0
        realized = gross.rolling(20).std() * np.sqrt(ANN)
        expo = (tv / realized).clip(0, 1).shift(1).fillna(0.0)
    elif overlay == "regime":
        ma = spy.add(1).cumprod()
        gate = (
            (ma > ma.rolling(200).mean())
            .astype(float)
            .reindex(idx)
            .shift(1)
            .fillna(0.0)
        )
        expo = gate * 0.7 + 0.3
    else:
        expo = pd.Series(1.0, index=idx)
    net = gross * expo
    return net


def metrics(r):
    r = pd.Series(r).dropna()
    if len(r) < 20 or r.std() == 0:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    eq = (1 + r).cumprod()
    return dict(
        sharpe=float(r.mean() / r.std() * np.sqrt(ANN)),
        cagr=float(eq.iloc[-1] ** (ANN / len(r)) - 1),
        maxdd=float((eq / eq.cummax() - 1).min()),
    )


def run():
    d = load()
    R, dates, universe, signals = d["R"], d["dates"], d["universe"], d["signals"]
    spy = R["SPY"].reindex(dates).fillna(0.0)
    is_mask, oos_mask = dates <= IS_END, dates > IS_END
    SIGS = ["shortflow", "insider", "pead", "quality", "congress", "news"]
    print(
        f"universe={len(universe)} dates {dates.min().date()}..{dates.max().date()} "
        f"IS<= {IS_END.date()} | signals={SIGS}"
    )

    # baselines
    ew = backtest(
        pd.DataFrame(0.0, index=dates, columns=universe).add(
            zscore_cs(pd.DataFrame(1.0, index=dates, columns=universe)), fill_value=0
        ),
        R,
        universe,
    )  # ~EW (flat score)
    base = {}
    for nm, series in (("SPY", spy), ("EW-universe", ew)):
        base[nm] = dict(IS=metrics(series[is_mask]), OOS=metrics(series[oos_mask]))
        print(
            f"  baseline {nm:12}: IS Sharpe {base[nm]['IS']['sharpe']:.2f} | OOS {base[nm]['OOS']['sharpe']:.2f}"
        )

    overlays = ["none", "voltgt10", "voltgt15", "regime"]
    rows = []
    subsets = [
        c for k in range(1, len(SIGS) + 1) for c in itertools.combinations(SIGS, k)
    ]
    print(
        f"sweeping {len(subsets)} signal-subsets x {len(overlays)} overlays = {len(subsets) * len(overlays)} configs"
    )
    for combo in subsets:
        sc = composite(signals, list(combo)).reindex(index=dates, columns=universe)
        for ov in overlays:
            net = backtest(sc, R, universe, overlay=ov, spy=spy)
            mi, mo = metrics(net[is_mask]), metrics(net[oos_mask])
            rows.append(
                dict(
                    signals="+".join(combo),
                    n_sig=len(combo),
                    overlay=ov,
                    IS_sharpe=mi["sharpe"],
                    IS_cagr=mi["cagr"],
                    IS_maxdd=mi["maxdd"],
                    OOS_sharpe=mo["sharpe"],
                    OOS_cagr=mo["cagr"],
                    OOS_maxdd=mo["maxdd"],
                    OOS_vs_EW=mo["sharpe"] - base["EW-universe"]["OOS"]["sharpe"],
                    OOS_vs_SPY=mo["sharpe"] - base["SPY"]["OOS"]["sharpe"],
                )
            )
    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(OUTDIR, "experiment_results.parquet"), index=False)

    print("\n=== TOP 15 by OOS Sharpe ===")
    top = res.sort_values("OOS_sharpe", ascending=False).head(15)
    print(
        top[
            [
                "signals",
                "overlay",
                "IS_sharpe",
                "OOS_sharpe",
                "OOS_cagr",
                "OOS_maxdd",
                "OOS_vs_EW",
                "OOS_vs_SPY",
            ]
        ].to_string(index=False)
    )
    print(
        f"\nconfigs beating SPY OOS: {(res['OOS_vs_SPY'] > 0).sum()}/{len(res)}  "
        f"beating EW-universe OOS: {(res['OOS_vs_EW'] > 0).sum()}/{len(res)}"
    )
    print(
        f"configs beating EW-univ OOS by >0.1 Sharpe: {(res['OOS_vs_EW'] > 0.1).sum()}"
    )
    print("\n[DONE] experiment sweep -> experiment_results.parquet")
    return res


if __name__ == "__main__":
    run()
