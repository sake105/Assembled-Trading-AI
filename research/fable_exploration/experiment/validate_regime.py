"""Validate the experiment's lead finding HONESTLY: is the regime/trend overlay real
or just OOS-overfit from ranking 252 configs? Cleanest test = apply it to SPY ITSELF
(survivorship-immune, no signal, no selection). Report IS/OOS/full + MaxDD + DSR@252."""

from __future__ import annotations
import glob
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from engine import _wide_close, backtest, metrics, IS_END, MIN_HISTORY  # noqa: E402
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

ANN = 252
close = _wide_close()
R = close.pct_change(fill_method=None)
dates = R.index
spy = R["SPY"].reindex(dates).fillna(0.0)
universe = sorted(
    [c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY]
    and set(c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY) - {"SPY"}
)
is_m, oos_m = dates <= IS_END, dates > IS_END


def regime_gate(price_ret, lo=0.3, hi=1.0, ma=200):
    eq = (1 + price_ret).cumprod()
    g = (eq > eq.rolling(ma).mean()).astype(float).shift(1).fillna(0.0)
    return g * (hi - lo) + lo


def voltgt(ret, tv=0.15):
    realized = ret.rolling(20).std() * np.sqrt(ANN)
    return (tv / realized).clip(0, 1).shift(1).fillna(0.0)


def report(name, r):
    f, i, o = metrics(r), metrics(r[is_m]), metrics(r[oos_m])
    print(
        f"  {name:30} FULL Sh={f['sharpe']:.2f} DD={f['maxdd']:+.1%} | "
        f"IS Sh={i['sharpe']:.2f} DD={i['maxdd']:+.1%} | OOS Sh={o['sharpe']:.2f} DD={o['maxdd']:+.1%}"
    )
    return r


print(f"universe={len(universe)} dates {dates.min().date()}..{dates.max().date()}")
print("\n=== PURE OVERLAY TEST ON SPY (survivorship-immune, no signal/selection) ===")
report("SPY buy-hold", spy)
report("SPY x regime(MA200)", spy * regime_gate(spy))
report("SPY x voltgt15", spy * voltgt(spy))

print("\n=== same overlay on EW-survivor-universe ===")
ew = R.reindex(columns=universe).mean(axis=1)
report("EW-universe", ew)
report("EW-universe x regime", ew * regime_gate(spy))
report("EW-universe x voltgt15", ew * voltgt(ew))

print("\n=== shortflow basket (the top signal) +/- overlay ===")
sv = pd.concat(
    [
        pd.read_parquet(f)
        for f in glob.glob("research/fable_exploration/data/short_volume_*.parquet")
    ],
    ignore_index=True,
).drop_duplicates(["date", "symbol"])
sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
sv["r"] = sv["short_volume"] / sv["total_volume"].where(sv["total_volume"] > 0)
sf = (
    -sv.pivot_table(index="date", columns="symbol", values="r")
    .reindex(dates)
    .reindex(columns=universe)
)
sf_z = sf.sub(sf.mean(axis=1), axis=0).div(sf.std(axis=1).replace(0, np.nan), axis=0)
base = report(
    "shortflow basket (none)", backtest(sf_z, R, universe, overlay="none", spy=spy)
)
reg = report(
    "shortflow basket x regime", backtest(sf_z, R, universe, overlay="regime", spy=spy)
)

print("\n=== Is the overlay's gain just lower average exposure? (control) ===")
g = regime_gate(spy)
const = spy * float(g.mean())
report(f"SPY x CONST {g.mean():.2f} (no timing)", const)
print(
    "  (regime SPY Sharpe vs const-exposure SPY Sharpe -> if regime>const, the TIMING adds)"
)

print("\n=== DSR on top config (shortflow x regime), honest trial count ===")
for nt in (1, 252):
    d = deflated_sharpe(reg.dropna(), n_trials=nt)
    print(
        f"  n_trials={nt}: prob={d.deflated_sharpe_probability:.4f} sr_pp={d.sharpe_observed:.4f} passes5%={d.passes_5pct}"
    )
print("\n[DONE] regime validation")
