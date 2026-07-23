"""Welle 37 — H-075 Kalender-Anomalien (TOM/DoW), H-076 Sektor-Rotation im Wrapper.

Reuse h071-Engine (backtest: ETF-Steuer je Round-Trip + Verlusttopf, 5 bps/Seite).
Zusatz je Config: BRUTTO-Diagnose (Anomalie existent vor Kosten/Steuer?).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h011_kandidat_a import OUTD  # noqa: E402
from h071_indicator_lab import backtest, bh_ref  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
START = 100_000.0


def tom_positions(close: pd.Series, pre: int, post: int) -> pd.Series:
    idx = close.index
    months = idx.to_period("M")
    pos = np.zeros(len(idx))
    arr = np.arange(len(idx))
    for m in months.unique():
        mask = months == m
        ii = arr[mask]
        pos[ii[-pre:]] = 1.0  # letzte pre Tage
        nxt = arr[months > m]
        if len(nxt):
            pos[nxt[:post]] = 1.0  # erste post Tage Folgemonat
    return pd.Series(pos, index=idx)


def dow_positions(close: pd.Series, days: set[int]) -> pd.Series:
    return pd.Series(
        [1.0 if d in days else 0.0 for d in close.index.dayofweek], index=close.index
    )


def gross_edge(close: pd.Series, pos: pd.Series) -> dict:
    r = close.pct_change().fillna(0.0)
    p = pos.shift(1).fillna(0.0)
    in_r, out_r = r[p > 0.5], r[p < 0.5]
    return {
        "in_mean_bps": round(float(in_r.mean()) * 1e4, 2),
        "out_mean_bps": round(float(out_r.mean()) * 1e4, 2),
        "t_diff": round(
            float(
                (in_r.mean() - out_r.mean())
                / np.sqrt(in_r.var() / len(in_r) + out_r.var() / len(out_r))
            ),
            2,
        ),
    }


def sector_rotation(
    closes: pd.DataFrame, top_n: int, buffer_rank: int
) -> tuple[dict, pd.Series]:
    """Monatlich Top-N nach 12-1-Momentum; Wechsel-Steuer 18,46 % je Verkaufs-Gewinn (FIFO je Sektor)."""
    me = closes.groupby(closes.index.to_period("M")).tail(1)
    mom = me.shift(1) / me.shift(12) - 1.0
    V = START
    pot, tax_paid = 0.0, 0.0
    held: dict[str, float] = {}  # symbol -> entry value share
    eq = []
    n_sw = 0
    for i in range(12, len(me) - 1):
        t = me.index[i]
        ranks = mom.iloc[i].rank(ascending=False)
        target = set(ranks[ranks <= top_n].index)
        keep = {s for s in held if ranks.get(s, 99) <= buffer_rank}
        new_hold = list(keep | target)[: max(top_n, len(keep))]
        # monthly return of current holdings applied first
        if held:
            r_m = float(np.mean([me[s].iloc[i] / me[s].iloc[i - 1] - 1 for s in held]))
            V *= 1 + r_m
        # switches: taxed on proportional gain (approx: pot-level annual too complex -> per switch)
        sells = [s for s in held if s not in new_hold]
        if sells and held:
            frac = len(sells) / len(held)
            sold_val = V * frac
            entry_val = sum(held[s] for s in sells)
            gain = sold_val - entry_val
            V -= 2 * 5e-4 * sold_val  # round trip costs on turnover
            if gain > 0:
                off = min(gain, pot)
                pot -= off
                tx = (gain - off) * 0.1846
                V -= tx
                tax_paid += tx
            else:
                pot += -gain
            n_sw += len(sells)
        per = V / len(new_hold) if new_hold else 0.0
        held = {s: (held.get(s, per) if s in held else per) for s in new_hold}
        eq.append((t, V))
    e = pd.Series(dict(eq))
    r = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    # terminal tax
    gain = V - sum(held.values())
    if gain > 0:
        tx = max(gain - pot, 0) * 0.1846
        V -= tx
        tax_paid += tx
    half = len(r) // 2
    return {
        "net": round(V),
        "cagr": round(((V / START) ** (1 / years) - 1) * 100, 2),
        "sharpe": round(float(r.mean() / r.std() * sqrt(12)), 3),
        "oos_sharpe": round(
            float(r.iloc[half:].mean() / r.iloc[half:].std() * sqrt(12)), 3
        ),
        "maxdd": round(float((e / e.cummax() - 1).min()), 3),
        "switches": n_sw,
        "tax": round(tax_paid),
    }, r


def main() -> int:
    oc = pd.read_parquet(DATA / "prices_overnight_oc.parquet")
    spy = oc[oc["symbol"] == "SPY"].set_index("date")["close"].sort_index()
    spy.index = pd.DatetimeIndex(spy.index)
    ref = bh_ref(spy, "etf")
    print(f"[REF] SPY B&H: {ref['net']:,} oos={ref['oos_sharpe']}", flush=True)

    out = {"_BH": ref}
    configs = {
        "TOM_4_3": tom_positions(spy, 4, 3),
        "TOM_2_2": tom_positions(spy, 2, 2),
        "DoW_TueFri": dow_positions(spy, {1, 2, 3, 4}),
        "DoW_WedOnly": dow_positions(spy, {2}),
    }
    for name, pos in configs.items():
        res, _ = backtest(spy, pos, tax_kind="etf")
        res["gross"] = gross_edge(spy, pos)
        res["beats"] = bool(
            res["net"] > ref["net"] and res["oos_sharpe"] > ref["oos_sharpe"]
        )
        out[name] = res
        print(
            f"[H075] {name:12s} net={res['net']:>9,} gross_in={res['gross']['in_mean_bps']}bps "
            f"out={res['gross']['out_mean_bps']}bps t={res['gross']['t_diff']} beats={res['beats']}",
            flush=True,
        )

    sectors = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"]
    px = oc.pivot_table(index="date", columns="symbol", values="close")[
        sectors
    ].dropna()
    px.index = pd.DatetimeIndex(px.index)
    spy_w = spy[(spy.index >= px.index[0]) & (spy.index <= px.index[-1])]
    ref2 = bh_ref(spy_w, "etf")
    out["_BH_window_sectors"] = ref2
    for tn, br in ((1, 3), (3, 5), (3, 3), (1, 1)):
        res, _ = sector_rotation(px, tn, br)
        res["beats"] = bool(
            res["net"] > ref2["net"] and res["oos_sharpe"] > ref2["oos_sharpe"]
        )
        out[f"ROT_top{tn}_buf{br}"] = res
        print(
            f"[H076] ROT_top{tn}_buf{br}: net={res['net']:>9,} sh={res['sharpe']} oos={res['oos_sharpe']} "
            f"sw={res['switches']} vs BH {ref2['net']:,} beats={res['beats']}",
            flush=True,
        )

    (OUTD / "h075_h076_results.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
