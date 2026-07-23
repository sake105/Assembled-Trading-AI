"""H-072 — INTRADAY-Indikator-Batterie (Welle 34). SPY 5m, 2020-10–2026-07, day-only.

Strategien: Opening-Range-Breakout (30/60 min), 5m-SMA-Cross (20/100 Bars), Intraday-RSI-Rev,
Gap-Fade & Gap-Follow, Prev-Day-Momentum-Dayhold. Kein Overnight. 4 bps/Seite.
Steuer 18,46 % (ETF) auf Jahres-Netting mit Verlusttopf. Quantifiziert Brutto- vs Netto-Kaskade.
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
from h011_kandidat_a import OUTD  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0
COST = 4e-4
TAX = 0.1846


def day_pnls(bars: pd.DataFrame, strat: str, **kw) -> pd.Series:
    """Return per-day strategy gross return (day-only)."""
    out = {}
    for d, g in bars.groupby(bars["datetime"].dt.normalize(), sort=True):
        g = g.sort_values("datetime").reset_index(drop=True)
        if len(g) < 30:
            continue
        o = g["open"].iloc[0]
        c = g["close"]
        ret = 0.0
        if strat == "orb":  # opening range breakout
            k = kw["k"]
            hi = c.iloc[:k].max()
            after = g.iloc[k:]
            trig = after[after["close"] > hi]
            if len(trig):
                entry = float(trig["close"].iloc[0])
                ret = float(c.iloc[-1]) / entry - 1 - 2 * COST
        elif strat == "smax":  # 5m sma cross, in while fast>slow
            f = c.rolling(20).mean()
            s = c.rolling(60).mean()
            pos = (f > s).astype(int).shift(1).fillna(0)
            r = c.pct_change().fillna(0)
            switches = int(pos.diff().abs().sum())
            ret = float((pos * r).sum()) - switches * COST
        elif strat == "rsirev":
            dlt = c.diff()
            up = dlt.clip(lower=0).rolling(14).mean()
            dn = (-dlt.clip(upper=0)).rolling(14).mean()
            rs = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            pos = pd.Series(np.nan, index=c.index)
            pos[rs < 25] = 1.0
            pos[rs > 50] = 0.0
            pos = pos.ffill().fillna(0).shift(1).fillna(0)
            r = c.pct_change().fillna(0)
            switches = int(pos.diff().abs().sum())
            ret = float((pos * r).sum()) - switches * COST
        elif strat in ("gapfade", "gapfollow"):
            prev_close = kw["prev_close"].get(d)
            if prev_close is None or not np.isfinite(prev_close):
                continue
            gap = o / prev_close - 1
            if abs(gap) < kw["min_gap"]:
                continue
            direction = -np.sign(gap) if strat == "gapfade" else np.sign(gap)
            if direction > 0:  # long only (Guardrail: kein Short)
                ret = float(c.iloc[-1]) / o - 1 - 2 * COST
            else:
                continue
        elif strat == "pdm":  # prev-day momentum day-hold
            pd_r = kw["prev_ret"].get(d)
            if pd_r is None or pd_r <= 0:
                continue
            ret = float(c.iloc[-1]) / o - 1 - 2 * COST
        out[d] = ret
    return pd.Series(out).sort_index()


def net_path(day_r: pd.Series) -> dict:
    """Annual netting with loss pot, 18.46% on positive annual net."""
    V = START
    pot = 0.0
    tax_paid = 0.0
    eq = []
    for year, g in day_r.groupby(day_r.index.year):
        v0 = V
        for r in g.values:
            V *= 1 + r
            eq.append(V)
        pnl = V - v0
        if pnl > 0:
            off = min(pnl, pot)
            pot -= off
            t = (pnl - off) * TAX
            V -= t
            tax_paid += t
        else:
            pot += -pnl
    e = pd.Series(eq)
    years = max((day_r.index[-1] - day_r.index[0]).days / 365.25, 1e-9)
    sr = day_r.mean() / day_r.std() * sqrt(252) if day_r.std() > 0 else 0.0
    return {
        "gross_daily_bps": round(float(day_r.mean()) * 1e4, 2),
        "days_traded": int(len(day_r)),
        "sharpe_gross": round(float(sr), 3),
        "net_final": round(V),
        "tax": round(tax_paid),
        "cagr_net": round(((V / START) ** (1 / years) - 1) * 100, 2),
    }


def main() -> int:
    bars = pd.read_parquet(DATA / "intraday_crisis_5m.parquet")
    bars = bars[bars["symbol"] == "SPY"].copy()
    daily_close = bars.groupby(bars["datetime"].dt.normalize())["close"].last()
    prev_close = daily_close.shift(1).to_dict()
    prev_ret = (
        daily_close.pct_change().shift(0).to_dict()
    )  # prev day's ret known at next open
    prev_ret = daily_close.pct_change().to_dict()
    prev_ret = {k: v for k, v in pd.Series(prev_ret).shift(1).items()}

    configs = {
        "ORB_30min": ("orb", {"k": 6}),
        "ORB_60min": ("orb", {"k": 12}),
        "SMA5m_20_60": ("smax", {}),
        "RSI5m_rev_25_50": ("rsirev", {}),
        "GapFollow_0.3pct": ("gapfollow", {"prev_close": prev_close, "min_gap": 0.003}),
        "GapFade_0.3pct": ("gapfade", {"prev_close": prev_close, "min_gap": 0.003}),
        "PrevDayMom": ("pdm", {"prev_ret": prev_ret}),
    }
    # B&H ref same window
    bh_g = START * (daily_close.iloc[-1] / daily_close.iloc[0])
    bh_net = START + (bh_g - START) * (1 - TAX)
    out = {"_BH_SPY_window": {"net": round(bh_net)}}
    print(f"[REF] SPY B&H net (window): {bh_net:,.0f}", flush=True)
    for name, (strat, kw) in configs.items():
        dr = day_pnls(bars, strat, **kw)
        if not len(dr):
            out[name] = {"days_traded": 0}
            continue
        res = net_path(dr)
        res["beats_BH"] = bool(res["net_final"] > bh_net)
        out[name] = res
        print(
            f"[H072] {name:18s} gross={res['gross_daily_bps']:+.1f}bps/d n={res['days_traded']} "
            f"shG={res['sharpe_gross']:.2f} net={res['net_final']:,} beats={res['beats_BH']}",
            flush=True,
        )
    (OUTD / "h072_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
