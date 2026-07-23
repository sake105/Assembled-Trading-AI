"""H-077 — MEGA-STRATEGIE-SUCHE (Welle 39). Stufe-1-Screen über alle Stränge.

Je Strang ≥75 Configs (wo Daten tragen). Vektorisierter Monats-Screen; Steuer-Approximation
(Jahres-Netting; Satz je Asset-Typ) — SCREEN, kein Verdict. Kriterium: net > SPY-B&H-net
(window-matched) UND OOS-Hälfte-Sharpe > 0. Ausgabe je Strang: n_configs, n_survivors, top-5.
Guardrail-4-RESEARCH-OVERRIDE Hans: Short (borrow 3 %), Hebel (Finanzierung 4 %), FX, Optionen (VIX).
"""

from __future__ import annotations

import glob
import json
import sys
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
COST_M = 10e-4  # bps pro Monats-Rebalance (round-trip-Anteil, Screen-Niveau)


# ---------------------------------------------------------------- screen engine
def screen_eval(mret: pd.Series, tax_rate: float, bench_net: float) -> dict:
    """Monatsrenditen-Serie -> net (Jahres-Netting-Steuer), Sharpe, OOS, DD."""
    mret = mret.dropna()
    if len(mret) < 36:
        return {"net": 0, "skip": True}
    V = START
    pot = 0.0
    eq = []
    for y, g in mret.groupby(mret.index.year):
        v0 = V
        for r in g.values:
            V *= 1 + r
            eq.append(V)
        pnl = V - v0
        if pnl > 0:
            off = min(pnl, pot)
            pot -= off
            V -= (pnl - off) * tax_rate
        else:
            pot += -pnl
    e = pd.Series(eq)
    rr = e.pct_change().dropna()
    half = len(mret) // 2
    o = mret.iloc[half:]
    years = len(mret) / 12
    return {
        "net": round(V),
        "cagr": round(((V / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(mret.mean() / mret.std() * sqrt(12)), 3)
        if mret.std() > 0
        else 0,
        "oos_sharpe": round(float(o.mean() / o.std() * sqrt(12)), 3)
        if o.std() > 0
        else 0,
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "survives": bool(
            V > bench_net and (o.mean() / o.std() if o.std() > 0 else -1) > 0
        ),
    }


def month_panel():
    """Monats-Panel MIT kanonischer Hygiene-Trunkierung (wie load_verdict_prices).

    Ohne diese Trunkierung fangen Ganz-Universum-Baskets die impossible-jump-
    Artefakte delisteter Micro-Prices (+34.000x) -> 10^30-Fake-Compounding.
    """
    df = pd.read_parquet(DATA / "prices_verdict.parquet")
    close = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    r = close.pct_change(fill_method=None)
    bad = (r.abs() > 1.0) & (close.shift(1) < 1.0)
    for sym in close.columns[bad.any()]:
        close.loc[bad.index[bad[sym]][0] :, sym] = np.nan
    m = close.groupby(close.index.to_period("M")).last()
    m.index = m.index.to_timestamp(how="end").tz_localize("UTC")
    return m


def basket_returns(mclose: pd.DataFrame, sig: dict, hold_m: int) -> pd.Series:
    """sig: month_end -> set(symbols). EW, Halten hold_m Monate, monatliche Kohorten-Überlappung.

    fill_method=None: KEIN Pad über NaN-Lücken (delistete Serien erzeugten Fake-Sprünge).
    |Monatsrendite| > 100 % wird als Datenartefakt aus dem Basket-Mittel gedroppt (Screen-Regel).
    """
    fwd = mclose.pct_change(fill_method=None).shift(
        -1
    )  # Rendite Monat t->t+1 am Index t
    fwd = fwd.where(fwd.abs() <= 1.0)  # Artefakt-Drop (Screen)
    months = list(mclose.index)
    active: list[tuple[int, set]] = []
    out = {}
    for i, t in enumerate(months[:-1]):
        if t in sig and sig[t]:
            active.append((i + hold_m, set(sig[t])))
        active = [(e, s) for e, s in active if e > i]
        syms = set().union(*[s for _, s in active]) if active else set()
        syms = {s for s in syms if s in fwd.columns}
        if syms:
            r = fwd.loc[t, list(syms)].dropna()
            out[t] = float(r.mean()) - COST_M if len(r) else 0.0
        else:
            out[t] = 0.0
    return pd.Series(out)


def spy_bench(mclose, tax=0.1846):
    spy = mclose["SPY"].dropna()
    g = START * (spy.iloc[-1] / spy.iloc[0])
    return round(START + (g - START) * (1 - tax))


def report(strand: str, results: dict, out: dict):
    n = len(results)
    surv = {k: v for k, v in results.items() if v.get("survives")}
    top = sorted(results.items(), key=lambda kv: -kv[1].get("net", 0))[:5]
    print(f"\n[{strand}] configs={n} survivors={len(surv)}", flush=True)
    for k, v in top:
        print(
            f"  {k[:58]:58s} net={v.get('net', 0):>10,} oos={v.get('oos_sharpe', 0)}",
            flush=True,
        )
    out[strand] = {"n": n, "survivors": list(surv)[:20], "top5": {k: v for k, v in top}}


# ---------------------------------------------------------------- strands
def strand_insider(mclose, bench, out):
    frames = [
        pd.read_parquet(f)
        for f in glob.glob(str(DATA / "form4_broad" / "tranche_*.parquet"))
    ]
    frames.append(
        pd.read_parquet(
            ROOT / "data" / "raw" / "insider_congress" / "form4_insider_full.parquet"
        )
    )
    f = pd.concat(frames, ignore_index=True)
    f = f[(f["transaction_code"] == "P") & (~f["is_derivative"].astype(bool))]
    f["available_at"] = pd.to_datetime(f["available_at"], utc=True, errors="coerce")
    f = f.dropna(subset=["available_at", "symbol"])
    f["val"] = pd.to_numeric(f["value_usd"], errors="coerce").fillna(0)
    f["officer"] = (
        f["role"].astype(str).str.contains("officer|director", case=False, na=False)
    )
    res = {}
    months = list(mclose.index)
    for win in (1, 2, 3):
        for mins in (1, 2, 3):
            for mv in (0, 10_000, 50_000):
                for role in ("all", "off"):
                    for hold in (3, 6, 12):
                        d = f[f["val"] >= mv]
                        if role == "off":
                            d = d[d["officer"]]
                        sig = {}
                        for t in months:
                            r = d[
                                (d["available_at"] <= t)
                                & (d["available_at"] > t - pd.DateOffset(months=win))
                            ]
                            byc = r.groupby("symbol")["reporting_owner_cik"].nunique()
                            s = set(byc[byc >= mins].index)
                            if s:
                                sig[t] = s
                        key = f"INS_w{win}_n{mins}_v{mv // 1000}k_{role}_h{hold}"
                        res[key] = screen_eval(
                            basket_returns(mclose, sig, hold), 0.26375, bench
                        )
    report("INSIDER", res, out)


def strand_congress(mclose, bench, out):
    fp = ROOT / "data" / "raw" / "insider_congress" / "congress_trades_full.parquet"
    c = pd.read_parquet(fp)
    cols = {x.lower(): x for x in c.columns}
    sym = cols.get("ticker") or cols.get("symbol")
    dt = (
        cols.get("disclosure_date")
        or cols.get("available_at")
        or cols.get("transaction_date")
    )
    typ = cols.get("type") or cols.get("transaction_type") or cols.get("side")
    if not (sym and dt):
        print("[CONGRESS] SKIP: Spalten fehlen", list(c.columns), flush=True)
        return
    c[dt] = pd.to_datetime(c[dt], utc=True, errors="coerce")
    c = c.dropna(subset=[dt, sym])
    if typ:
        c = c[c[typ].astype(str).str.contains("purchase|buy", case=False, na=False)]
    res = {}
    months = list(mclose.index)
    cham = cols.get("chamber")
    for win in (1, 2, 3):
        for mins in (1, 2, 3):
            for ch in ("all", "house", "senate") if cham else ("all",):
                for hold in (1, 3, 6, 12):
                    d = (
                        c
                        if ch == "all"
                        else c[
                            c[cham].astype(str).str.contains(ch, case=False, na=False)
                        ]
                    )
                    sig = {}
                    for t in months:
                        r = d[(d[dt] <= t) & (d[dt] > t - pd.DateOffset(months=win))]
                        vc = r[sym].value_counts()
                        s = set(vc[vc >= mins].index)
                        if s:
                            sig[t] = s
                    res[f"CGR_w{win}_n{mins}_{ch}_h{hold}"] = screen_eval(
                        basket_returns(mclose, sig, hold), 0.26375, bench
                    )
    report("CONGRESS", res, out)


def strand_whale(mclose, bench, out):
    fp = DATA / "13f_top100.parquet"
    w = pd.read_parquet(fp)
    cols = {x.lower(): x for x in w.columns}
    sym = cols.get("symbol") or cols.get("ticker")
    per = cols.get("period") or cols.get("periodofreport") or cols.get("report_period")
    mgr = cols.get("manager") or cols.get("cik") or cols.get("manager_cik")
    if not (sym and per and mgr):
        print("[WHALE] SKIP: Spalten fehlen", list(w.columns), flush=True)
        return
    w[per] = pd.to_datetime(w[per], utc=True, errors="coerce")
    w = w.dropna(subset=[per, sym])
    res = {}
    months = list(mclose.index)
    for cons in (3, 5, 10, 15, 20):
        for lag_m in (2, 3):
            for hold in (3, 6, 12):
                for ex_mega in (False, True):
                    sig = {}
                    for t in months:
                        cut = t - pd.DateOffset(months=lag_m)
                        q = w[w[per] <= cut]
                        if not len(q):
                            continue
                        last_p = q[per].max()
                        qq = q[q[per] == last_p]
                        vc = qq.groupby(sym)[mgr].nunique()
                        s = set(vc[vc >= cons].index)
                        if ex_mega:
                            s -= {
                                "AAPL",
                                "MSFT",
                                "NVDA",
                                "AMZN",
                                "GOOGL",
                                "GOOG",
                                "META",
                                "TSLA",
                            }
                        if s:
                            sig[t] = s
                    res[
                        f"WHL_c{cons}_lag{lag_m}_h{hold}_{'xmega' if ex_mega else 'all'}"
                    ] = screen_eval(basket_returns(mclose, sig, hold), 0.26375, bench)
    report("WHALE_13F", res, out)


def strand_news(mclose, bench, out):
    s = pd.read_parquet(DATA / "sentiment.parquet")
    cols = {x.lower(): x for x in s.columns}
    sym = cols.get("symbol") or cols.get("ticker")
    dt = cols.get("date") or cols.get("timestamp")
    val = cols.get("sentiment") or cols.get("score") or cols.get("normalized")
    if not (sym and dt and val):
        print("[NEWS] SKIP:", list(s.columns), flush=True)
        return
    s[dt] = pd.to_datetime(s[dt], utc=True, errors="coerce")
    s = s.dropna(subset=[dt, sym, val])
    s["m"] = s[dt].dt.to_period("M")
    agg = s.groupby(["m", sym])[val].mean().unstack()
    agg.index = agg.index.to_timestamp(how="end").tz_localize("UTC")
    res = {}
    months = list(mclose.index)
    for q in (0.9, 0.8, 0.7):
        for look in (1, 3):
            for hold in (1, 3, 6):
                for mode in ("top", "bottom"):
                    sig = {}
                    for t in months:
                        rows = agg[
                            (agg.index <= t)
                            & (agg.index > t - pd.DateOffset(months=look))
                        ]
                        if not len(rows):
                            continue
                        sc = rows.mean()
                        thr = sc.quantile(q if mode == "top" else 1 - q)
                        sel = sc[sc >= thr] if mode == "top" else sc[sc <= thr]
                        if len(sel):
                            sig[t] = set(sel.index)
                    res[f"NWS_{mode}_q{int(q * 100)}_l{look}_h{hold}"] = screen_eval(
                        basket_returns(mclose, sig, hold), 0.26375, bench
                    )
    report("NEWS", res, out)


def strand_geo(out):
    inten = pd.read_parquet(DATA / "geopol_intensity.parquet").set_index("date")[
        "n_articles"
    ]
    inten.index = pd.DatetimeIndex(inten.index)
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    px = pc.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    px.index = pd.DatetimeIndex(px.index)
    z = (inten - inten.rolling(252, min_periods=120).mean()) / inten.rolling(
        252, min_periods=120
    ).std()
    z = z.reindex(px.index).ffill().shift(1)
    spy = px["SPY"].dropna()
    bench = round(START + (START * (spy.iloc[-1] / spy.iloc[0]) - START) * (1 - 0.1846))
    res = {}
    for asset in ("XLE", "GLD", "ITA"):
        for thr in (0.5, 1.0, 1.5, 2.0):
            for hold in (5, 10, 21, 63):
                for mode in ("long_crisis", "exit_spy"):
                    r_a = (
                        px[asset].pct_change()
                        if mode == "long_crisis"
                        else px["SPY"].pct_change()
                    )
                    active = z > thr
                    sig_days = active.rolling(hold, min_periods=1).max().fillna(0)
                    pos = sig_days if mode == "long_crisis" else (1 - sig_days)
                    dr = (pos.shift(1).fillna(0) * r_a).dropna()
                    switches = pos.diff().abs().fillna(0)
                    dr = dr - switches.shift(0).reindex(dr.index).fillna(0) * 5e-4
                    mr = (1 + dr).groupby(dr.index.to_period("M")).prod() - 1
                    mr.index = mr.index.to_timestamp(how="end").tz_localize("UTC")
                    res[f"GEO_{mode}_{asset}_z{thr}_h{hold}"] = screen_eval(
                        mr, 0.1846, bench
                    )
    report("GEOPOLITIK", res, out)


def strand_fx(out):
    fx = pd.read_parquet(DATA / "prices_fx_majors.parquet")
    px = fx.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    px.index = pd.DatetimeIndex(px.index)
    res = {}
    for pair in px.columns:
        s = px[pair].dropna()
        r = s.pct_change()
        bench = START  # FX hat keinen B&H-Anspruch; Screen-Hürde: > Cash + OOS>0
        for f_, sl in ((20, 100), (50, 200)):
            for mode in ("L", "S", "LS"):
                sig = (s.rolling(f_).mean() > s.rolling(sl).mean()).astype(float)
                pos = sig if mode == "L" else (1 - sig if mode == "S" else 2 * sig - 1)
                dr = (pos.shift(1).fillna(0) * r).dropna() - pos.diff().abs().reindex(
                    r.index
                ).fillna(0) * 2e-4
                if mode in ("S", "LS"):
                    dr -= (
                        0.03
                        / 252
                        * (pos.shift(1) < 0)
                        .reindex(dr.index)
                        .fillna(False)
                        .astype(float)
                    )
                mr = (1 + dr).groupby(dr.index.to_period("M")).prod() - 1
                mr.index = mr.index.to_timestamp(how="end").tz_localize("UTC")
                res[f"FX_{pair}_SMA{f_}_{sl}_{mode}"] = screen_eval(mr, 0.44, bench)
        for lb in (5, 20):
            sig = (r.rolling(lb).sum() < 0).astype(float)  # mean reversion long
            dr = (sig.shift(1).fillna(0) * r).dropna() - sig.diff().abs().reindex(
                r.index
            ).fillna(0) * 2e-4
            mr = (1 + dr).groupby(dr.index.to_period("M")).prod() - 1
            mr.index = mr.index.to_timestamp(how="end").tz_localize("UTC")
            res[f"FX_{pair}_MR{lb}_L"] = screen_eval(mr, 0.44, bench)
    report("FX", res, out)


def strand_leverage(out):
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    btc.index = pd.DatetimeIndex(btc.index)
    pc = pd.read_parquet(DATA / "prices_crisis.parquet")
    gld = pc[pc["symbol"] == "GLD"].set_index("timestamp")["close"].sort_index()
    gld.index = pd.DatetimeIndex(gld.index)
    res = {}
    for name, s, tax in (
        ("SPY", spy, 0.1846),
        ("BTC", btc, 0.26375),
        ("GLD", gld, 0.26375),
    ):
        r = s.pct_change()
        sigs = {
            "BH": pd.Series(1.0, index=s.index),
            "SMA200": (s > s.rolling(200).mean()).astype(float),
            "TSmom252": (s > s.shift(252)).astype(float),
            "SMA50_200": (s.rolling(50).mean() > s.rolling(200).mean()).astype(float),
            "Donch100": None,
        }
        up = s.rolling(100).max().shift(1)
        dn = s.rolling(50).min().shift(1)
        d = pd.Series(np.nan, index=s.index)
        d[s >= up] = 1.0
        d[s <= dn] = 0.0
        sigs["Donch100"] = d.ffill().fillna(0.0)
        for signame, sig in sigs.items():
            for lev in (1.5, 2.0, 3.0):
                pos = sig * lev
                fin = (lev - 1) * 0.04 / 252
                dr = (
                    pos.shift(1).fillna(0) * r - (pos.shift(1).fillna(0) > 0) * fin
                ).dropna()
                dr -= sig.diff().abs().reindex(dr.index).fillna(0) * 5e-4 * lev
                mr = (1 + dr).groupby(dr.index.to_period("M")).prod() - 1
                mr.index = mr.index.to_timestamp(how="end").tz_localize("UTC")
                g = START * float((1 + r.fillna(0)).prod())
                bench = round(START + (g - START) * (1 - tax))
                res[f"LEV_{name}_{signame}_x{lev}"] = screen_eval(mr, tax, bench)
    report("HEBEL", res, out)


def strand_options(out):
    w = pd.read_parquet(DATA / "prices_w28.parquet")
    px = w.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    px.index = pd.DatetimeIndex(px.index)
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    d = pd.DataFrame({"SPY": spy}).join(px[["VIX"]], how="inner").dropna()
    me = d.groupby(d.index.to_period("M")).tail(1)

    def phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    def bs(S, K, T, sig, put=False):
        if sig <= 0 or T <= 0:
            return max((K - S) if put else (S - K), 0.0)
        d1 = (log(S / K) + (0.02 + sig * sig / 2) * T) / (sig * sqrt(T))
        d2 = d1 - sig * sqrt(T)
        c = S * phi(d1) - K * exp(-0.02 * T) * phi(d2)
        return c - S + K * exp(-0.02 * T) if put else c

    res = {}
    T = 21 / 252
    for kind in ("CC", "CSP", "COLLAR"):
        for otm in (0.0, 0.02, 0.03, 0.05, 0.07):
            for skew in (0.0, 0.10, 0.20):
                for frac in (0.5, 1.0):
                    mrets = []
                    for i in range(len(me) - 1):
                        S0, S1 = float(me["SPY"].iloc[i]), float(me["SPY"].iloc[i + 1])
                        iv = float(me["VIX"].iloc[i]) / 100
                        base = S1 / S0 - 1
                        if kind == "CC":
                            K = S0 * (1 + otm)
                            prem = bs(S0, K, T, iv * (1 - skew))
                            pnl = frac * (prem - max(S1 - K, 0)) / S0
                            mrets.append(base + pnl - 3e-4)
                        elif kind == "CSP":
                            K = S0 * (1 - otm)
                            prem = bs(
                                S0, K, T, iv * (1 + skew), put=True
                            )  # Puts: Skew ERHÖHT IV
                            pnl = (prem - max(K - S1, 0)) / S0
                            mrets.append(frac * pnl + (1 - frac) * base - 3e-4)
                        else:  # collar: long stock + put(-otm) - call(+otm)
                            Kc, Kp = S0 * (1 + otm), S0 * (1 - otm)
                            prem_c = bs(S0, Kc, T, iv * (1 - skew))
                            prem_p = bs(S0, Kp, T, iv * (1 + skew), put=True)
                            pnl = (
                                prem_c - max(S1 - Kc, 0) - prem_p + max(Kp - S1, 0)
                            ) / S0
                            mrets.append(base + frac * pnl - 3e-4)
                    mr = pd.Series(mrets, index=me.index[:-1])
                    g = START * float((me["SPY"].iloc[-1] / me["SPY"].iloc[0]))
                    bench = round(START + (g - START) * (1 - 0.1846))
                    res[
                        f"OPT_{kind}_otm{int(otm * 100)}_skew{int(skew * 100)}_f{frac}"
                    ] = screen_eval(mr, 0.26375, bench)
    report("OPTIONEN(Modell)", res, out)


def strand_short_ls(out):
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    res = {}
    for sym in ("SPY", "XLK", "XLE", "XLF"):
        s = oc[oc["symbol"] == sym].set_index("date")["close"].sort_index()
        s.index = pd.DatetimeIndex(s.index)
        r = s.pct_change()
        base_sigs = {
            "SMA50_200": (s.rolling(50).mean() > s.rolling(200).mean()).astype(float),
            "TSmom252": (s > s.shift(252)).astype(float),
            "RSI50": None,
        }
        d = s.diff()
        up = d.clip(lower=0).rolling(14).mean()
        dn = (-d.clip(upper=0)).rolling(14).mean()
        base_sigs["RSI50"] = (
            (100 - 100 / (1 + up / dn.replace(0, np.nan))) > 50
        ).astype(float)
        g = START * float((1 + r.fillna(0)).prod())
        bench = round(START + (g - START) * (1 - 0.1846))
        for signame, sig in base_sigs.items():
            for mode in ("S", "LS"):
                pos = (1 - sig) * -1 if mode == "S" else 2 * sig - 1
                dr = (pos.shift(1).fillna(0) * r).dropna()
                dr -= (
                    (pos.shift(1) < 0).reindex(dr.index).fillna(False).astype(float)
                    * 0.03
                    / 252
                )
                dr -= pos.diff().abs().reindex(dr.index).fillna(0) * 5e-4
                mr = (1 + dr).groupby(dr.index.to_period("M")).prod() - 1
                mr.index = mr.index.to_timestamp(how="end").tz_localize("UTC")
                res[f"SHT_{sym}_{signame}_{mode}"] = screen_eval(mr, 0.26375, bench)
    report("SHORT_LS", res, out)


def main() -> int:
    mclose = month_panel()
    bench = spy_bench(mclose, tax=0.26375)  # Einzelaktien-Stränge vs Aktien-Steuer-B&H
    print(
        f"[BENCH] SPY B&H (26,375 %): {bench:,} | Fenster {mclose.index[0].date()}–{mclose.index[-1].date()}",
        flush=True,
    )
    out: dict = {"_bench_stock": bench}
    strand_insider(mclose, bench, out)
    strand_congress(mclose, bench, out)
    strand_whale(mclose, bench, out)
    strand_news(mclose, bench, out)
    strand_geo(out)
    strand_fx(out)
    strand_leverage(out)
    strand_options(out)
    strand_short_ls(out)
    n_total = sum(v["n"] for k, v in out.items() if isinstance(v, dict) and "n" in v)
    n_surv = sum(
        len(v["survivors"])
        for k, v in out.items()
        if isinstance(v, dict) and "survivors" in v
    )
    out["_total"] = {
        "configs": n_total,
        "survivors_stage1": n_surv,
        "N_cumulative": 1205 + n_total,
    }
    (OUTD / "h077_mega_search.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"\n[TOTAL] {n_total} Configs, Stage-1-Survivors: {n_surv}, N={1205 + n_total}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
