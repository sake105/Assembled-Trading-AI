"""H-071 — TECHNISCHES INDIKATOR-LABOR (Welle 33). Long/Cash, next-close, 5 bps/Seite.

Steuer je Asset korrekt: SPY-ETF 18,46 % (Teilfreistellung auch beim Timing), Gold/BTC §23
(>1J 0 %, <1J 44 %). Round-Trip-FIFO + Verlusttopf (aktienähnlicher Topf vereinfacht je Asset),
End-Liquidation. OOS = 2. Fensterhälfte. Ausgabe: alle Configs + Top-Tabelle + Kriterien-Zählung.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from h011_kandidat_a import OUTD, cscv_pbo  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
COST = 5e-4  # 5 bps/side


# ---------------------------------------------------------------- indicators
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_signals(close: pd.Series) -> dict[str, pd.Series]:
    sig = {}
    for f, s in ((10, 50), (20, 100), (50, 200), (100, 300)):
        sig[f"SMAx_{f}_{s}"] = (sma(close, f) > sma(close, s)).astype(float)
    for f, s in ((12, 26), (20, 100), (50, 200)):
        sig[f"EMAx_{f}_{s}"] = (ema(close, f) > ema(close, s)).astype(float)
    macd = ema(close, 12) - ema(close, 26)
    macds = ema(macd, 9)
    sig["MACD_12_26_9"] = (macd > macds).astype(float)
    for hi, lo in ((55, 20), (100, 50)):
        up = close.rolling(hi).max().shift(1)
        dn = close.rolling(lo).min().shift(1)
        pos = pd.Series(np.nan, index=close.index)
        pos[close >= up] = 1.0
        pos[close <= dn] = 0.0
        sig[f"Donch_{hi}_{lo}"] = pos.ffill().fillna(0.0)
    r = rsi(close, 14)
    for lo, hi in ((30, 55), (25, 50)):
        pos = pd.Series(np.nan, index=close.index)
        pos[r < lo] = 1.0
        pos[r > hi] = 0.0
        sig[f"RSIrev_{lo}_{hi}"] = pos.ffill().fillna(0.0)
    sig["RSItrend_50"] = (r > 50).astype(float)
    mid = sma(close, 20)
    sd = close.rolling(20).std()
    lo_b, up_b = mid - 2 * sd, mid + 2 * sd
    pos = pd.Series(np.nan, index=close.index)
    pos[close < lo_b] = 1.0
    pos[close > mid] = 0.0
    sig["BBrev_20_2"] = pos.ffill().fillna(0.0)
    pos = pd.Series(np.nan, index=close.index)
    pos[close > up_b] = 1.0
    pos[close < mid] = 0.0
    sig["BBbrk_20_2"] = pos.ffill().fillna(0.0)
    for m in (126, 252):
        sig[f"TSmom_{m}"] = (close > close.shift(m)).astype(float)
    vol = close.pct_change().rolling(20).std()
    volp = vol.rolling(756, min_periods=252).quantile(0.8)
    sig["VolFilter_p80"] = (vol < volp).astype(float)
    # combos
    s200 = sig["SMAx_50_200"]
    m12 = sig["TSmom_252"]
    sig["AND_SMA200_Mom12"] = ((s200 > 0) & (m12 > 0)).astype(float)
    sig["OR_SMA200_Mom12"] = ((s200 > 0) | (m12 > 0)).astype(float)
    sig["AND_MACD_RSI50"] = (
        (sig["MACD_12_26_9"] > 0) & (sig["RSItrend_50"] > 0)
    ).astype(float)
    sig["AND_Donch_Vol"] = (
        (sig["Donch_55_20"] > 0) & (sig["VolFilter_p80"] > 0)
    ).astype(float)
    sig["AND_SMA200_Vol"] = ((s200 > 0) & (sig["VolFilter_p80"] > 0)).astype(float)
    votes = s200 + m12 + sig["MACD_12_26_9"]
    sig["ENS_2of3"] = (votes >= 2).astype(float)
    sig["ENS_3of3"] = (votes >= 3).astype(float)
    return sig


# ---------------------------------------------------------------- engine
def backtest(close: pd.Series, pos: pd.Series, *, tax_kind: str) -> dict:
    """tax_kind: etf (18.46% flat) | p23 (>=365d 0%, else 44%). Round-trip tax + loss pot."""
    pos = pos.shift(1).fillna(0.0)  # execute next close
    px = close.values
    p = pos.values
    V = START
    pot = 0.0
    tax_paid = 0.0
    in_mkt = False
    v_entry = 0.0
    t_entry = None
    eq = np.empty(len(px))
    idx = close.index
    n_trades = 0
    for i in range(len(px)):
        if in_mkt and i > 0:
            V *= px[i] / px[i - 1]
        want = p[i] > 0.5
        if want and not in_mkt:
            V *= 1 - COST
            v_entry, t_entry = V, idx[i]
            in_mkt = True
            n_trades += 1
        elif not want and in_mkt:
            V *= 1 - COST
            gain = V - v_entry
            if tax_kind == "etf":
                rate = 0.1846
            else:
                rate = 0.0 if (idx[i] - t_entry).days >= 365 else 0.44
            if gain > 0:
                off = min(gain, pot)
                pot -= off
                t = (gain - off) * rate
                V -= t
                tax_paid += t
            else:
                pot += -gain
            in_mkt = False
        eq[i] = V
    if in_mkt:
        V *= 1 - COST
        gain = V - v_entry
        rate = (
            0.1846
            if tax_kind == "etf"
            else (0.0 if (idx[-1] - t_entry).days >= 365 else 0.44)
        )
        if gain > 0:
            off = min(gain, pot)
            pot -= off
            t = (gain - off) * rate
            V -= t
            tax_paid += t
        eq[-1] = V
    e = pd.Series(eq, index=idx)
    r = e.pct_change().dropna()
    half = len(e) // 2
    r2 = r.iloc[half:]
    years = (idx[-1] - idx[0]).days / 365.25
    return {
        "net": round(V),
        "cagr": round(((V / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3)
        if r.std() > 0
        else 0.0,
        "oos_sharpe": round(float(r2.mean() / r2.std() * sqrt(252)), 3)
        if r2.std() > 0
        else 0.0,
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "trades": n_trades,
        "tax": round(tax_paid),
    }, r


def bh_ref(close: pd.Series, tax_kind: str) -> dict:
    g = START * (close.iloc[-1] / close.iloc[0])
    rate = 0.1846 if tax_kind == "etf" else 0.0  # B&H §23 > 1J = 0 %
    net = START + (g - START) * (1 - rate)
    r = close.pct_change().dropna()
    half = len(r) // 2
    years = (close.index[-1] - close.index[0]).days / 365.25
    return {
        "net": round(net),
        "cagr": round(((net / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(252)), 3),
        "oos_sharpe": round(
            float(r.iloc[half:].mean() / r.iloc[half:].std() * sqrt(252)), 3
        ),
        "maxdd": round(float((close / close.cummax() - 1).min()), 3),
    }


def run_asset(name: str, close: pd.Series, tax_kind: str, results: dict, rets: dict):
    ref = bh_ref(close, tax_kind)
    results[f"_BH_{name}"] = ref
    print(
        f"\n[{name}] B&H ref: net={ref['net']:,} sharpe={ref['sharpe']} oos={ref['oos_sharpe']}",
        flush=True,
    )
    sigs = build_signals(close)
    rows = []
    for sname, pos in sigs.items():
        res, r = backtest(close, pos, tax_kind=tax_kind)
        key = f"{name}:{sname}"
        results[key] = res
        rets[key] = r
        beats = res["net"] > ref["net"] and res["oos_sharpe"] > ref["oos_sharpe"]
        rows.append(
            (
                sname,
                res["net"],
                res["sharpe"],
                res["oos_sharpe"],
                res["maxdd"],
                res["trades"],
                beats,
            )
        )
    rows.sort(key=lambda x: -x[1])
    print(f"[{name}] top-8 of {len(rows)} configs:", flush=True)
    for r_ in rows[:8]:
        print(
            f"  {r_[0]:22s} net={r_[1]:>10,} sh={r_[2]:.2f} oos={r_[3]:.2f} dd={r_[4]:.2f} tr={r_[5]} beats={r_[6]}",
            flush=True,
        )
    n_beats = sum(1 for r_ in rows if r_[6])
    print(f"[{name}] configs beating B&H (net+OOS): {n_beats}/{len(rows)}", flush=True)
    return len(rows)


def main() -> int:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    pcx = pd.read_parquet(DATA / "prices_crisis.parquet")
    gld = pcx[pcx["symbol"] == "GLD"].set_index("timestamp")["close"].sort_index()
    gld.index = pd.DatetimeIndex(gld.index)
    btc = pd.read_parquet(DATA / "crypto_BTCUSDCC.parquet")["close"]
    btc.index = pd.DatetimeIndex(btc.index)
    btc = btc[btc.index >= pd.Timestamp("2014-01-01", tz="UTC")]

    results: dict = {}
    rets: dict = {}
    n_total = 0
    n_total += run_asset("SPY", spy, "etf", results, rets)
    n_total += run_asset("GLD", gld, "p23", results, rets)
    n_total += run_asset("BTC", btc, "p23", results, rets)

    # best config overall vs DSR at new cumulative N
    N_NEW = 163 + n_total
    best_key = max(rets, key=lambda k: results[k]["net"])
    dsr = deflated_sharpe(rets[best_key], n_trials=N_NEW)
    rm = pd.DataFrame({k: v for k, v in rets.items() if k.startswith("SPY:")})
    pbo = float(cscv_pbo(rm.dropna()))
    summary = {
        "n_configs": n_total,
        "N_cumulative": N_NEW,
        "best": best_key,
        "best_res": results[best_key],
        "best_DSR_prob": round(float(dsr.deflated_sharpe_probability), 3),
        "best_DSR_pass": bool(dsr.passes_5pct),
        "PBO_SPY_family": round(pbo, 3),
    }
    results["_summary"] = summary
    (OUTD / "h071_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("\n[SUMMARY]", json.dumps(summary, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
