"""Before touching risk code: does the PRODUCTION vol_target_overlay already do what
my 'trend x vol' finding proposes? Run the REAL function (SPY+IEF) and compare to my
experiment variants + buy-hold. If incumbent >= mine, there is NOTHING to integrate."""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from engine import IS_END  # noqa: E402
from src.assembled_core.strategies.vol_target_overlay import (  # noqa: E402
    generate_vol_target_signals_from_prices,
)

ANN = 252
START, END = pd.Timestamp("2018-08-01"), pd.Timestamp("2026-06-10")

import yfinance as yf

px = yf.download(["SPY", "IEF"], period="max", auto_adjust=True, progress=False)[
    "Close"
]
px.index = pd.to_datetime(px.index).tz_localize(None)
px = px.loc[(px.index >= "2017-01-01") & (px.index <= END)]
ret = px.pct_change(fill_method=None)

# long-format for the production function
long = px.reset_index().melt(
    id_vars=px.index.name or "Date", var_name="symbol", value_name="close"
)
long.columns = ["timestamp", "symbol", "close"]
long["timestamp"] = pd.to_datetime(long["timestamp"])

sig = generate_vol_target_signals_from_prices(
    long,
    target_vol=0.12,
    vol_lookback=20,
    sma_window=200,
    defensive_asset="IEF",
    risk_asset="SPY",
)
W = sig.pivot_table(index="timestamp", columns="symbol", values="score").reindex(
    ret.index
)
# weight from bar t applies to next period's return -> shift(1)
incumbent = (W.shift(1) * ret).sum(axis=1)
incumbent = incumbent.loc[(incumbent.index >= START)]

spy = ret["SPY"].loc[incumbent.index].fillna(0.0)
# my experiment's regime x vol (cash de-risk, 0.3 floor, MA200) on SPY only
eq = (1 + spy).cumprod()
gate = (eq > eq.rolling(200).mean()).astype(float).shift(1).fillna(0.0) * 0.7 + 0.3
vt = (0.15 / (spy.rolling(20).std() * np.sqrt(ANN))).clip(0, 1).shift(1).fillna(0.0)
mine = spy * gate * vt

idx = incumbent.index
is_m, oos_m = idx <= IS_END, idx > IS_END


def m(r, msk=None):
    r = pd.Series(r).reindex(idx)
    if msk is not None:
        r = r[msk]
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return "n/a"
    eqq = (1 + r).cumprod()
    return f"Sh={r.mean() / r.std() * np.sqrt(ANN):.2f} DD={(eqq / eqq.cummax() - 1).min():+.0%}"


print(
    f"dates {idx.min().date()}..{idx.max().date()}  (IEF rows present: {ret['IEF'].notna().sum()})"
)
print(
    "\n=== PRODUCTION vol_target_overlay (SPY+IEF, real function) vs alternatives ==="
)
for name, r in (
    ("SPY buy-hold", spy),
    ("PROD vol_target_overlay (SPY+IEF)", incumbent),
    ("my regime x vol (SPY, cash)", mine),
):
    print(f"  {name:38} FULL {m(r)} | IS {m(r, is_m)} | OOS {m(r, oos_m)}")

print("\n  per-year FULL Sharpe:")
for y in sorted(set(idx.year)):
    msk = idx.year == y
    if msk.sum() < 60:
        continue
    print(f"   {y}: SPY {m(spy, msk)} | PROD {m(incumbent, msk)} | mine {m(mine, msk)}")
print("\n[DONE] incumbent overlay test")
