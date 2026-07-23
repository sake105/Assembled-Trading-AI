"""Verdict-Engine — survivorship-freies PIT-Universum (Welle 4+, Registry-Konventionen).

STATUS-DEKLARATION (W6, 2026-07-22 GESAMTBEWERTUNG): RESEARCH-ONLY.
Diese Engine (inkl. der aus h011_kandidat_a importierten TaxedPortfolio =
einzige vollstaendige deutsche Steuer-Engine des Projekts: FIFO, Verlusttopf,
Sparerpauschbetrag) ist NICHT in src/ integriert, hat KEINE Testabdeckung und
nutzt Modul-Globals mit Laufzeit-Monkeypatching als API (DIV_TAX/TAX). Der
produktive accounting/tax_lots.TaxLotStore ist davon unabhaengig und hat
seinerseits keinen Produktions-Caller (Anlage-KAP: spezifiziert, nicht
integriert — siehe KNOWN_ISSUES). Ergebnisse dieser Engine sind Research-
Verdicts (research/ledger.md), keine Steuerberechnungen fuer echte Konten.
Ein Lift nach src/ (TaxedPortfolio parametrisieren + Golden-Tests gegen
results/*.json) ist in docs/GESAMTBEWERTUNG.md §7.5 auf ~1-2 Tage geschaetzt
und wird erst bei realem Wiederverwendungsbedarf gemacht.

Unterschiede zur explorativen Engine (h011_kandidat_a.run_variant):
- Universum je Rebalance = PIT-S&P-500-Mitglieder (Snapshot <= as_of).
- Delisting: endet die Kurshistorie eines gehaltenen Titels, Zwangsverkauf zum
  letzten verfügbaren Kurs am FOLGETAG (kein Hindsight).
- Kein ATR-Backstop (keine H/L-Daten; Designs nutzen keinen).
- Steuer-/Kosten-Engine, next-close-Execution, Gates, no-retrim, EW-Band: identisch.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import (  # noqa: E402  (reuse tax engine + constants)
    GATE_EXPOSURE,
    POS_CAP,
    START_CAPITAL,
    TOP_IN,
    TaxedPortfolio,
)

DATA = Path(__file__).resolve().parent / "data"
DIV_TAX = 0.26375  # German dividend tax; monkeypatch to 0.0 for gross (no-tax) runs


def load_verdict_prices() -> pd.DataFrame:
    """Load verdict panel with data-hygiene truncation (Mandat §2.5).

    EODHD delisted series can end in broken adjusted micro-prices with
    impossible one-day jumps (observed: +34,000x from $0.005). Rule: the FIRST
    day with |return| > 100% AND previous close < $1 marks the series as
    corrupt from that point -> truncate (NaN) from that day on. Conservative:
    the name was dead/illiquid there anyway; truncation triggers the engine's
    delisting force-sell at the last clean price.
    """
    df = pd.read_parquet(DATA / "prices_verdict.parquet")
    close = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    r = close.pct_change(fill_method=None)
    bad = (r.abs() > 1.0) & (close.shift(1) < 1.0)
    n_trunc = 0
    for sym in close.columns[bad.any()]:
        first_bad = bad.index[bad[sym]][0]
        close.loc[first_bad:, sym] = np.nan
        n_trunc += 1
    print(
        f"[HYGIENE] truncated {n_trunc} corrupt series at first impossible jump",
        flush=True,
    )
    return close


def load_div_panel(close_index: pd.DatetimeIndex) -> pd.DataFrame:
    """dividends.parquet -> (trading_day x symbol) dividend/share panel.

    Ex-dates that fall on non-trading days snap to the NEXT trading day.
    """
    d = pd.read_parquet(DATA / "dividends.parquet")
    pos = close_index.searchsorted(pd.DatetimeIndex(d["ex_date"]))
    pos = np.clip(pos, 0, len(close_index) - 1)
    d = d.assign(t=close_index[pos])
    return d.groupby(["t", "symbol"])["dividend"].sum().unstack()


def load_membership(close_index: pd.DatetimeIndex) -> pd.Series:
    """Return a Series month_end -> frozenset(members), PIT (snapshot <= as_of)."""
    snaps: list[tuple[pd.Timestamp, frozenset]] = []
    with open(DATA / "sp500_historical_constituents.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            d = pd.Timestamp(row["date"], tz="UTC")
            snaps.append((d, frozenset(t.strip() for t in row["tickers"].split(","))))
    snaps.sort()
    snap_dates = [d for d, _ in snaps]
    month_ends = close_index.to_series().groupby(close_index.to_period("M")).max()
    out = {}
    for me in month_ends:
        i = np.searchsorted(snap_dates, me, side="right") - 1
        if i >= 0:
            out[me] = snaps[i][1]
    return pd.Series(out)


def run_verdict(
    close: pd.DataFrame,
    membership: pd.Series,
    *,
    label: str,
    mode: str = "momentum",  # momentum | ew
    top_out: int = 60,
    use_gate: bool = False,
    gate_mode: str = "sma",
    ew_band: float | None = None,
    spy_col: str = "SPY",
    gate_sma_win: int = 200,
    gate_vol_pct: float = 0.8,
    div_panel: pd.DataFrame | None = None,  # ex_date x symbol -> dividend/share
    score_panel: pd.DataFrame | None = None,  # custom ranking score (higher=buy);
    # rebalances ONLY on rows present in score_panel (e.g. yearly signals)
    top_in: int = TOP_IN,
    terminal_liquidation: bool = False,  # sell all lots at end -> tax terminal gains
) -> tuple[dict, pd.Series, pd.Series]:
    idx = close.index
    spy = close[spy_col]
    sma200 = spy.rolling(gate_sma_win).mean()
    mom = close.shift(21) / close.shift(252) - 1.0
    vol60 = close.pct_change(fill_method=None).rolling(60).std()
    spy_v20 = spy.pct_change().rolling(20).std()
    spy_v20_p80 = spy_v20.rolling(756, min_periods=252).quantile(gate_vol_pct)
    last_valid = close.apply(lambda s: s.last_valid_index())
    global_last = idx[-1]
    # Valuation uses forward-filled prices: single-symbol data gaps (e.g. rows
    # existing only for a subset on US holidays) must not value held positions
    # at 0 — that produced fake +-80% equity spikes. Trades still execute only
    # on real price rows (px_t checks stay on `close`).
    close_ff = close.ffill()

    month_ends = set(membership.index)
    pf = TaxedPortfolio(START_CAPITAL)
    exposure = 1.0
    pending: list[tuple[str, str, float]] = []
    equity = []

    for t in idx:
        pf.set_date(t)  # arm annual Sparerpauschbetrag (1000 EUR/yr)
        px_t = close.loc[t]
        for action, sym, amount in pending:
            px = px_t.get(sym, np.nan)
            if not np.isfinite(px):
                # delisted overnight: force-sell at last available price
                lv = last_valid.get(sym)
                if lv is not None and lv < t:
                    px = close.at[lv, sym]
                else:
                    continue
            if action == "sell_all":
                q = pf.qty(sym)
                if q > 0:
                    pf.sell(sym, q, float(px))
            elif action == "trade_to":
                cur = pf.qty(sym) * px
                delta = amount - cur
                if delta > 1.0:
                    pf.buy(sym, delta, float(px))
                elif delta < -1.0:
                    pf.sell(sym, -delta / px, float(px))
        pending = []

        # delisting force-sell: history ended yesterday (and not just data lag)
        for sym in list(pf.lots.keys()):
            lv = last_valid.get(sym)
            if lv is not None and lv < t and lv < global_last - pd.Timedelta(days=10):
                pending.append(("sell_all", sym, 0.0))

        # German dividend tax (26.375% on gross dividend at ex-date, cash out).
        # adjusted_close already reinvests the GROSS dividend in the price path;
        # withdrawing only the tax approximates net reinvestment. Dividends are
        # NOT offsettable against the Aktien-Verlusttopf (separate pot; we
        # conservatively ignore Sparerpauschbetrag and any pot netting).
        if div_panel is not None and t in div_panel.index:
            drow = div_panel.loc[t]
            for sym in list(pf.lots.keys()):
                d = drow.get(sym, np.nan)
                if np.isfinite(d) and d > 0:
                    q = pf.qty(sym)
                    tax = q * d * DIV_TAX
                    pf.cash -= tax
                    pf.tax_paid += tax

        # mark equity with forward-filled (stale-price) valuation
        v = pf.cash
        ff_t = close_ff.loc[t]
        for sym, lots in pf.lots.items():
            px = ff_t.get(sym, np.nan)
            if np.isfinite(px):
                v += sum(q for q, _ in lots) * px
        equity.append((t, v))

        if t not in month_ends:
            continue
        members = membership.loc[t]
        if use_gate:
            sma_on = np.isfinite(sma200.at[t]) and spy.at[t] >= sma200.at[t]
            vol_on = not (
                np.isfinite(spy_v20.at[t])
                and np.isfinite(spy_v20_p80.at[t])
                and spy_v20.at[t] > spy_v20_p80.at[t]
            )
            if gate_mode == "sma":
                exposure = 1.0 if sma_on else GATE_EXPOSURE
            elif gate_mode == "vol":
                exposure = 1.0 if vol_on else GATE_EXPOSURE
            else:
                n_off = (not sma_on) + (not vol_on)
                exposure = {0: 1.0, 1: 0.75, 2: GATE_EXPOSURE}[n_off]
        else:
            exposure = 1.0

        # sorted(members): `members` is a frozenset -> iteration order varies by
        # process (PYTHONHASHSEED). With tied scores (e.g. many zero-dividend names),
        # rank(method="first") would break ties by this non-deterministic order ->
        # non-reproducible selection. sorted() makes tie-breaking deterministic.
        tradable = [
            s
            for s in sorted(members)
            if s in close.columns
            and np.isfinite(px_t.get(s, np.nan))
            and px_t.get(s, 0.0) >= 1.0  # §3.1 min-liquidity/price floor
        ]
        if mode == "ew":
            nav = v
            tgt = nav / len(tradable) if tradable else 0.0
            held = set(pf.lots.keys())
            stale = sorted(
                held - set(tradable)
            )  # deterministic sell order (tax timing)
            if ew_band is not None and held:
                for sym in stale:
                    pending.append(("sell_all", sym, 0.0))
                for sym in tradable:
                    px = px_t.get(sym, np.nan)
                    cur = pf.qty(sym) * px if np.isfinite(px) else 0.0
                    if tgt > 0 and abs(cur - tgt) / tgt > ew_band:
                        pending.append(("trade_to", sym, tgt))
            else:
                for sym in stale:
                    pending.append(("sell_all", sym, 0.0))
                for sym in tradable:
                    pending.append(("trade_to", sym, tgt))
            continue

        # momentum mode (no-retrim, Welle-4 designs)
        if score_panel is not None:
            if t not in score_panel.index:
                continue  # custom signals rebalance only on their own dates
            m = score_panel.loc[t].reindex(tradable).dropna()
        else:
            m = mom.loc[t].reindex(tradable).dropna()
        order = m.rank(ascending=False, method="first")
        held = set(pf.lots.keys())
        keep = {s for s in held if order.get(s, 1e9) <= top_out}
        entries = [s for s in order[order <= top_in].index if s not in keep]
        for sym in sorted(
            held - keep - set(entries)
        ):  # deterministic sell order (tax timing)
            pending.append(("sell_all", sym, 0.0))
        if entries:
            iv = 1.0 / vol60.loc[t].reindex(entries)
            iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            w = (
                iv / iv.sum()
                if iv.sum() > 0
                else pd.Series(1.0 / len(entries), index=entries)
            )
            w = (w * len(entries) / top_in).clip(upper=POS_CAP)
            nav = v
            for sym in entries:
                pending.append(("trade_to", sym, float(w[sym]) * nav * exposure))

    # optional terminal liquidation: realize all remaining lots at the last
    # price (tax terminal gains at 26.375% minus Verlusttopf/Sparerpauschbetrag)
    # -> apples-to-apples with the end-taxed ETF path. Absolute EUR from START.
    final_net_postliq = None
    if terminal_liquidation:
        pf.set_date(global_last)
        last_px = close_ff.loc[global_last]
        for sym in list(pf.lots.keys()):
            px = last_px.get(sym, np.nan)
            if np.isfinite(px):
                pf.sell(sym, pf.qty(sym), float(px))
        final_net_postliq = float(pf.cash)

    eq = pd.Series(dict(equity)).sort_index()
    eq = eq[eq.index >= eq.index[0] + pd.Timedelta(days=400)]  # warmup trim
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    res = {
        "label": label,
        "final_value": float(eq.iloc[-1] / eq.iloc[0] * START_CAPITAL),
        "cagr_net": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
        "sharpe_net": float(ret.mean() / ret.std() * np.sqrt(252)),
        "maxdd_net": float((eq / eq.cummax() - 1).min()),
        "tax_paid": float(pf.tax_paid),
        "costs_paid": float(pf.costs_paid),
        "years": float(years),
        "final_net_postliq": final_net_postliq,
    }
    return res, eq, ret
