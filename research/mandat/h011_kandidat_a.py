"""H-011 — Kandidat A: Regime-Gated Momentum-Quality (Mandat §3.1).

EXPLORATIV, NICHT VERDICT-FÄHIG (Mandat §2.5): survivorship-biased Preisuniversum
(aktuelle S&P-500-Mitglieder via yfinance; Delistings fehlen). Registrierung:
research/registry.md H-011 (Parameter dort fixiert VOR diesem Lauf).

Varianten (Registry, genau 4):
  V1 Basis | V2 Momentum-only | V3 ohne Regime-Gate | V4 Rank-Puffer eng 20/25

Engine-Konventionen (vorab fixiert):
- Signale aus Daten bis Monatsultimo-Schluss; Ausführung zum Schluss des
  NÄCHSTEN Handelstags (kein Same-Bar-Look-ahead).
- Regime-Gate: monatliche Prüfung (SPY-Schluss vs. SMA200 am Ultimo) → Exposure
  100 % / 50 % für den Folgemonat.
- ATR-Backstop: täglicher Check; Exit zum NÄCHSTEN Tagesschluss, wenn Schluss <
  (Positions-Höchstschluss − 4×ATR20). Re-Entry nur über reguläres Rebalancing.
- Kosten: 10 bps je Seite auf gehandeltes Notional.
- Deutsche Steuern (Broker-Mechanik): bei jedem Verkauf FIFO-Gewinn; Gewinn wird
  zuerst gegen den Aktien-Verlusttopf verrechnet, Rest sofort mit 26,375 %
  belastet (Cash-Abfluss); Verluste füllen den Topf (Carry-Forward, unbegrenzt).
- ETF-Netto-Pfad: SPY TR Buy&Hold; Endwert = Kapital + Gewinn × (1 − 0,185).
  Vorabpauschale ignoriert → Benchmark wird eher ZU stark (konservativ gegen uns).
- EW-Baseline: alle verfügbaren Universumsnamen, monatlich equal-weight
  rebalanciert, gleiche Kosten-/Steuer-Engine, KEIN Gate/Signal — die
  Survivorship-Kontrolle (Fable-Befund: diese Baseline ≈ +0.35 Sharpe Gift).
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = Path(__file__).resolve().parent / "data"
OUTD = Path(__file__).resolve().parent / "results"
OUTD.mkdir(parents=True, exist_ok=True)

TAX = 0.26375
ETF_TAX = 0.185
COST_BPS = 10.0
TOP_IN, TOP_OUT = 20, 40
POS_CAP = 0.10
GATE_EXPOSURE = 0.5
START_CAPITAL = 100_000.0


# ---------------------------------------------------------------- data loading
def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA / "prices_sp500.parquet")
    close = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    high = df.pivot(index="timestamp", columns="symbol", values="high").sort_index()
    low = df.pivot(index="timestamp", columns="symbol", values="low").sort_index()
    return close, high, low


def load_roe_panel(close_index: pd.DatetimeIndex, symbols: list[str]) -> pd.DataFrame:
    """PIT ROE per (month-end, symbol): TTM NetIncome / latest StockholdersEquity.

    Uses available_at (filing acceptance) — a fact is visible only after it was
    actually filed. Fallback chain per symbol at as_of:
      NI: sum of the 4 most recent distinct quarterly (60–120d) NetIncomeLoss
          periods; else the most recent FY (330–400d) value.
      EQ: most recent StockholdersEquity (instant).
    """
    f = pd.read_parquet(DATA / "fundamentals_sp500.parquet")
    f = f[f["symbol"].isin(symbols)]
    ni = f[f["tag"].isin(["NetIncomeLoss", "ProfitLoss"])].copy()
    eq = f[f["tag"] == "StockholdersEquity"].copy()
    for d in (ni, eq):
        d["available_at"] = pd.to_datetime(d["available_at"], utc=True)
        d["period_end"] = pd.to_datetime(d["period_end"], utc=True)
        d["period_start"] = pd.to_datetime(d["period_start"], utc=True)
    ni["dur"] = (ni["period_end"] - ni["period_start"]).dt.days
    niq = ni[(ni["dur"] >= 60) & (ni["dur"] <= 120)]
    nify = ni[(ni["dur"] >= 330) & (ni["dur"] <= 400)]

    month_ends = close_index.to_series().groupby(close_index.to_period("M")).max()
    rows = []
    for as_of in month_ends:
        vq = niq[niq["available_at"] <= as_of]
        vy = nify[nify["available_at"] <= as_of]
        ve = eq[eq["available_at"] <= as_of]
        # latest restatement per (symbol, period_end): keep max available_at
        vq = vq.sort_values("available_at").groupby(["symbol", "period_end"]).tail(1)
        vy = vy.sort_values("available_at").groupby(["symbol", "period_end"]).tail(1)
        ve = ve.sort_values("available_at").groupby(["symbol", "period_end"]).tail(1)
        ttm = (
            vq.sort_values("period_end")
            .groupby("symbol")
            .tail(4)
            .groupby("symbol")
            .agg(ni=("val", "sum"), nq=("val", "size"))
        )
        ttm = ttm[ttm["nq"] == 4]["ni"]
        fy = vy.sort_values("period_end").groupby("symbol")["val"].last()
        ni_final = ttm.combine_first(fy)
        eq_final = ve.sort_values("period_end").groupby("symbol")["val"].last()
        roe = (ni_final / eq_final).replace([np.inf, -np.inf], np.nan)
        pos_eq = eq_final[eq_final > 0].index
        roe = roe.loc[roe.index.intersection(pos_eq)].dropna()
        for s, v in roe.items():
            rows.append((as_of, s, v))
    panel = pd.DataFrame(rows, columns=["timestamp", "symbol", "roe"])
    return panel.pivot(index="timestamp", columns="symbol", values="roe")


# ---------------------------------------------------------------- tax engine
class TaxedPortfolio:
    """FIFO lots + German Abgeltungsteuer with Aktien-Verlusttopf (broker style)."""

    def __init__(self, cash: float):
        self.cash = cash
        self.lots: dict[str, list[list[float]]] = {}  # sym -> [[qty, px], ...]
        self.loss_pot = 0.0
        self.tax_paid = 0.0
        self.costs_paid = 0.0
        # Sparerpauschbetrag (§20 Abs. 9 EStG): 1000 EUR/Jahr steuerfrei (ledig;
        # 2000 bei Zusammenveranlagung). Only applied when set_date() is called
        # per trading day -> backward-compatible: no set_date == exact old behavior.
        self.allowance_annual = 1000.0
        self.allowance_left = 0.0
        self._cur_year: int | None = None

    def set_date(self, date) -> None:
        """Arm the annual Sparerpauschbetrag; resets each calendar year."""
        yr = date.year
        if yr != self._cur_year:
            self._cur_year = yr
            self.allowance_left = self.allowance_annual

    def qty(self, sym: str) -> float:
        return sum(q for q, _ in self.lots.get(sym, []))

    def buy(self, sym: str, notional: float, px: float) -> None:
        if notional <= 0 or px <= 0:
            return
        cost = notional * COST_BPS / 1e4
        spend = min(notional, max(self.cash - cost, 0.0))
        if spend <= 0:
            return
        cost = spend * COST_BPS / 1e4
        q = (spend - cost) / px
        self.cash -= spend
        self.costs_paid += cost
        self.lots.setdefault(sym, []).append([q, px])

    def sell(self, sym: str, qty: float, px: float) -> None:
        lots = self.lots.get(sym, [])
        qty = min(qty, sum(q for q, _ in lots))
        if qty <= 0 or px <= 0:
            return
        proceeds = qty * px
        cost = proceeds * COST_BPS / 1e4
        gain = 0.0
        rest = qty
        while rest > 1e-12 and lots:
            lq, lpx = lots[0]
            take = min(rest, lq)
            gain += take * (px - lpx)
            lq -= take
            rest -= take
            if lq <= 1e-12:
                lots.pop(0)
            else:
                lots[0][0] = lq
        gain -= cost  # transaction cost reduces the taxable gain
        tax = 0.0
        if gain >= 0:
            offset = min(gain, self.loss_pot)
            self.loss_pot -= offset
            taxable = gain - offset
            used = min(taxable, self.allowance_left)  # Sparerpauschbetrag
            self.allowance_left -= used
            tax = (taxable - used) * TAX
        else:
            self.loss_pot += -gain
        self.cash += proceeds - cost - tax
        self.tax_paid += tax
        self.costs_paid += cost
        if not lots and sym in self.lots:
            del self.lots[sym]

    def value(self, prices: pd.Series) -> float:
        v = self.cash
        for sym, lots in self.lots.items():
            px = prices.get(sym, np.nan)
            if np.isfinite(px):
                v += sum(q for q, _ in lots) * px
        return v


# ---------------------------------------------------------------- backtest
def run_variant(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    roe: pd.DataFrame | None,
    *,
    use_quality: bool,
    use_gate: bool,
    top_out: int,
    label: str,
    ew_baseline: bool = False,
    use_atr_backstop: bool = True,
    no_retrim: bool = False,
    signal: str = "momentum",  # momentum | lowvol | high52
    gate_mode: str = "sma",  # sma | vol | both (only if use_gate)
    tlh_pct: float | None = None,  # e.g. 0.15 -> harvest losses >= 15%
    atr_mult: float = 4.0,
    ew_band: float | None = None,  # EW: rebalance only if relative drift > band
    rebal_months: set[int] | None = None,  # e.g. {1} = January only
) -> dict:
    idx = close.index
    spy = close["SPY"]
    sma200 = spy.rolling(200).mean()
    mom = close.shift(21) / close.shift(252) - 1.0
    vol60 = close.pct_change().rolling(60).std()
    vol252 = close.pct_change().rolling(252).std()
    high52 = close / close.rolling(252).max()
    # vol regime: SPY realized 20d vol vs its rolling past-only 80th percentile
    spy_v20 = spy.pct_change().rolling(20).std()
    spy_v20_p80 = spy_v20.rolling(756, min_periods=252).quantile(0.8)
    tr = (
        pd.concat(
            [
                (high - low),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ]
        )
        .groupby(level=0)
        .max()
    )
    atr20 = tr.rolling(20).mean()

    month_end_mask = pd.Series(idx, index=idx).groupby(idx.to_period("M")).max()
    month_ends = set(month_end_mask)
    universe = [c for c in close.columns if c != "SPY"]

    pf = TaxedPortfolio(START_CAPITAL)
    exposure = 1.0
    pending: list[tuple[str, str, float]] = []  # (action, sym, target_notional)
    peak_close: dict[str, float] = {}
    equity = []

    for t in idx:
        pf.set_date(t)  # arm annual Sparerpauschbetrag (1000 EUR/yr)
        px_t = close.loc[t]
        # 1) execute pending orders from the previous signal day at today's close
        for action, sym, amount in pending:
            px = px_t.get(sym, np.nan)
            if not np.isfinite(px):
                continue
            if action == "sell_all":
                self_q = pf.qty(sym)
                if self_q > 0:
                    pf.sell(sym, self_q, px)
                    peak_close.pop(sym, None)
            elif action == "trade_to":
                cur = pf.qty(sym) * px
                delta = amount - cur
                if delta > 1.0:
                    pf.buy(sym, delta, px)
                    peak_close.setdefault(sym, px)
                elif delta < -1.0:
                    pf.sell(sym, -delta / px, px)
        pending = []

        # 2) mark equity
        equity.append((t, pf.value(px_t)))

        # 3) ATR backstop signals (checked at close, executed next close)
        if not ew_baseline and use_atr_backstop:
            for sym in list(pf.lots.keys()):
                px = px_t.get(sym, np.nan)
                if not np.isfinite(px):
                    continue
                peak_close[sym] = max(peak_close.get(sym, px), px)
                a = atr20.at[t, sym] if sym in atr20.columns else np.nan
                if np.isfinite(a) and px < peak_close[sym] - atr_mult * a:
                    pending.append(("sell_all", sym, 0.0))

        # 4) monthly rebalance signal (at month-end close, executed next close)
        if t in month_ends and (rebal_months is None or t.month in rebal_months):
            if use_gate:
                sma_on = np.isfinite(sma200.at[t]) and spy.at[t] >= sma200.at[t]
                vol_on = not (
                    np.isfinite(spy_v20.at[t])
                    and np.isfinite(spy_v20_p80.at[t])
                    and spy_v20.at[t] > spy_v20_p80.at[t]
                )  # True = calm regime
                if gate_mode == "sma":
                    exposure = 1.0 if sma_on else GATE_EXPOSURE
                elif gate_mode == "vol":
                    exposure = 1.0 if vol_on else GATE_EXPOSURE
                else:  # both: two strikes -> 50%, one -> 75%, none -> 100%
                    n_off = (not sma_on) + (not vol_on)
                    exposure = {0: 1.0, 1: 0.75, 2: GATE_EXPOSURE}[n_off]
            else:
                exposure = 1.0

            if ew_baseline:
                avail = [s for s in universe if np.isfinite(px_t.get(s, np.nan))]
                nav = pf.value(px_t)
                tgt = nav / len(avail) if avail else 0.0
                held = set(pf.lots.keys())
                if ew_band is not None and held:
                    # banded EW (H-020/H-022): trade ONLY names whose current
                    # weight drifted more than ew_band (relative) from target
                    for sym in held - set(avail):
                        pending.append(("sell_all", sym, 0.0))
                    for sym in avail:
                        px = px_t.get(sym, np.nan)
                        cur = pf.qty(sym) * px if np.isfinite(px) else 0.0
                        if tgt > 0 and abs(cur - tgt) / tgt > ew_band:
                            pending.append(("trade_to", sym, tgt))
                else:
                    for sym in held - set(avail):
                        pending.append(("sell_all", sym, 0.0))
                    for sym in avail:
                        pending.append(("trade_to", sym, tgt))
                continue

            if signal == "lowvol":
                m = -vol252.loc[t]
            elif signal == "high52":
                m = high52.loc[t]
            else:
                m = mom.loc[t]
            ranks = m.rank(ascending=False)
            if use_quality and roe is not None and t in roe.index:
                q = roe.loc[t].reindex(m.index)
                qr = q.rank(ascending=False)
                combo = 0.5 * ranks + 0.5 * qr
                combo[q.isna()] = ranks[q.isna()]  # missing quality -> mom-only
            else:
                combo = ranks
            combo = combo.drop("SPY", errors="ignore").dropna()
            order = combo.rank(method="first")

            held = set(pf.lots.keys())

            # Tax-loss harvesting (H-014): sell losers >= tlh_pct below avg
            # cost NOW (fills the Verlusttopf); they leave `held` for this
            # month's target set, re-entry only via regular entry logic from
            # next month on (>= 1 month gap).
            tlh_sold: set[str] = set()
            if tlh_pct is not None:
                for sym in list(held):
                    px = px_t.get(sym, np.nan)
                    lots = pf.lots.get(sym, [])
                    tot_q = sum(q for q, _ in lots)
                    if not (np.isfinite(px) and tot_q > 0):
                        continue
                    avg_cost = sum(q * p for q, p in lots) / tot_q
                    if px < avg_cost * (1.0 - tlh_pct):
                        pending.append(("sell_all", sym, 0.0))
                        held.discard(sym)
                        tlh_sold.add(sym)
            keep = {s for s in held if order.get(s, 1e9) <= top_out}
            entries = [
                s
                for s in order[order <= TOP_IN].index
                if s not in keep and s not in tlh_sold
            ]
            targets = list(keep) + entries

            for sym in held - set(targets):
                pending.append(("sell_all", sym, 0.0))

            if targets:
                nav = pf.value(px_t)
                if no_retrim:
                    # H-012 convention: existing positions are left untouched
                    # (buy-and-hold until rank exit); only new entries are
                    # bought, each sized toward a 1/TOP_IN NAV slot,
                    # vol-tilted within the entry group, capped at POS_CAP.
                    if entries:
                        iv = 1.0 / vol60.loc[t].reindex(entries)
                        iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        w = (
                            iv / iv.sum()
                            if iv.sum() > 0
                            else pd.Series(1.0 / len(entries), index=entries)
                        )
                        w = (w * len(entries) / TOP_IN).clip(upper=POS_CAP)
                        for sym in entries:
                            pending.append(
                                ("trade_to", sym, float(w[sym]) * nav * exposure)
                            )
                else:
                    iv = 1.0 / vol60.loc[t].reindex(targets)
                    iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    w = (
                        iv / iv.sum()
                        if iv.sum() > 0
                        else pd.Series(1.0 / len(targets), index=targets)
                    )
                    w = w.clip(upper=POS_CAP)
                    w = w / w.sum()
                    for sym in targets:
                        pending.append(
                            ("trade_to", sym, float(w[sym]) * nav * exposure)
                        )

    eq = pd.Series(dict(equity)).sort_index()
    ret = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    out = {
        "label": label,
        "final_value": float(eq.iloc[-1]),
        "cagr_net": float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1),
        "sharpe_net": float(ret.mean() / ret.std() * np.sqrt(252)),
        "maxdd_net": float((eq / eq.cummax() - 1).min()),
        "tax_paid": float(pf.tax_paid),
        "costs_paid": float(pf.costs_paid),
        "loss_pot_end": float(pf.loss_pot),
        "years": float(years),
    }
    return out, eq, ret


# ---------------------------------------------------------------- CSCV-PBO
def cscv_pbo(returns_matrix: pd.DataFrame, n_blocks: int = 8) -> float:
    """Bailey/López de Prado CSCV-PBO over variants (columns = trials).

    Split time into n_blocks; for every C(n,n/2) IS/OOS combination pick the
    IS-best variant (Sharpe) and record its OOS rank; PBO = share of splits
    where the IS winner is below OOS median (logit < 0).
    """
    rm = returns_matrix.dropna()
    blocks = np.array_split(np.arange(len(rm)), n_blocks)
    neg = 0
    combos = list(combinations(range(n_blocks), n_blocks // 2))
    for is_idx in combos:
        is_rows = np.concatenate([blocks[i] for i in is_idx])
        oos_rows = np.concatenate(
            [blocks[i] for i in range(n_blocks) if i not in is_idx]
        )
        sr = lambda x: x.mean() / x.std() if x.std() > 0 else -np.inf  # noqa: E731
        is_sr = rm.iloc[is_rows].apply(sr)
        oos_sr = rm.iloc[oos_rows].apply(sr)
        winner = is_sr.idxmax()
        rank = (oos_sr < oos_sr[winner]).sum() / (len(oos_sr) - 1 + 1e-12)
        if rank < 0.5:
            neg += 1
    return neg / len(combos)


# ---------------------------------------------------------------- main
def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    close, high, low = load_prices()
    # liquidity floor: require median dollar volume proxy — here: non-NaN history
    close = close.dropna(axis=1, thresh=int(len(close) * 0.5))
    high, low = high[close.columns], low[close.columns]
    universe = [c for c in close.columns if c != "SPY"]
    print(
        f"[DATA] {len(universe)} symbols, {close.index[0].date()} -> {close.index[-1].date()}",
        flush=True,
    )

    print("[ROE] building PIT quality panel ...", flush=True)
    roe = load_roe_panel(close.index, universe)
    print(
        f"[ROE] coverage: {roe.notna().sum(axis=1).median():.0f} median symbols/month",
        flush=True,
    )

    variants = {
        "V1_basis": dict(use_quality=True, use_gate=True, top_out=40),
        "V2_mom_only": dict(use_quality=False, use_gate=True, top_out=40),
        "V3_no_gate": dict(use_quality=True, use_gate=False, top_out=40),
        "V4_tight_buffer": dict(use_quality=True, use_gate=True, top_out=25),
    }
    results, rets = {}, {}
    for name, kw in variants.items():
        print(f"[RUN] {name} ...", flush=True)
        res, eq, ret = run_variant(close, high, low, roe, label=name, **kw)
        results[name] = res
        rets[name] = ret
        print(f"      {res}", flush=True)

    print("[RUN] EW baseline (survivorship control) ...", flush=True)
    res_ew, eq_ew, ret_ew = run_variant(
        close,
        high,
        low,
        None,
        use_quality=False,
        use_gate=False,
        top_out=40,
        label="EW_baseline",
        ew_baseline=True,
    )
    results["EW_baseline"] = res_ew

    # SPY benchmarks
    spy = close["SPY"].dropna()
    spy_ret = spy.pct_change().dropna()
    years = (spy.index[-1] - spy.index[0]).days / 365.25
    spy_gross_cagr = (spy.iloc[-1] / spy.iloc[0]) ** (1 / years) - 1
    gross_gain = START_CAPITAL * (spy.iloc[-1] / spy.iloc[0] - 1)
    etf_net_final = START_CAPITAL + gross_gain * (1 - ETF_TAX)
    etf_net_cagr = (etf_net_final / START_CAPITAL) ** (1 / years) - 1
    results["SPY_gross"] = {
        "cagr": float(spy_gross_cagr),
        "sharpe": float(spy_ret.mean() / spy_ret.std() * np.sqrt(252)),
        "maxdd": float((spy / spy.cummax() - 1).min()),
    }
    results["ETF_net_path"] = {
        "final_value": float(etf_net_final),
        "cagr_net": float(etf_net_cagr),
    }

    # mandatory metrics on V1 (verdict variant per registry)
    N_TRIALS = 44  # ledger: N0=40 + 4 variant runs
    v1 = rets["V1_basis"]
    dsr = deflated_sharpe(v1, n_trials=N_TRIALS)
    results["DSR_V1"] = {
        "sharpe_observed_daily": float(dsr.sharpe_observed),
        "threshold": float(dsr.sharpe_threshold),
        "probability": float(dsr.deflated_sharpe_probability),
        "passes_5pct": bool(dsr.passes_5pct),
        "n_trials": N_TRIALS,
    }
    rm = pd.DataFrame(rets)
    results["PBO_CSCV_4variants"] = float(cscv_pbo(rm))

    # sub-period consistency: net Sharpe V1 vs EW baseline in 2y windows
    win = {}
    for y0 in range(v1.index[0].year, v1.index[-1].year, 2):
        m = (v1.index.year >= y0) & (v1.index.year < y0 + 2)
        me = (ret_ew.index.year >= y0) & (ret_ew.index.year < y0 + 2)
        if m.sum() > 100:
            s1 = v1[m].mean() / v1[m].std() * np.sqrt(252)
            s2 = ret_ew[me].mean() / ret_ew[me].std() * np.sqrt(252)
            win[f"{y0}-{y0 + 1}"] = {
                "V1": round(float(s1), 3),
                "EW": round(float(s2), 3),
            }
    results["subperiods_V1_vs_EW"] = win

    out_path = OUTD / "h011_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[DONE] -> {out_path}", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
