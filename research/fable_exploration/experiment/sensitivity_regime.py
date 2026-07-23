"""Is the trend/regime overlay ROBUST or a tuned MA window? Sweep MA in {100,150,200,250}
and exposure mappings, head-to-head vs vol-target, on SPY (survivorship-immune) and the
EW-universe. A robust effect should help across ALL windows/mappings, both periods."""

from __future__ import annotations
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from engine import _wide_close, metrics, IS_END, MIN_HISTORY  # noqa: E402

ANN = 252
close = _wide_close()
R = close.pct_change(fill_method=None)
dates = R.index
spy = R["SPY"].reindex(dates).fillna(0.0)
universe = sorted(
    set(c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY) - {"SPY"}
)
ew = R.reindex(columns=universe).mean(axis=1)
is_m, oos_m = dates <= IS_END, dates > IS_END


def gate(ret, ma, lo, hi):
    eq = (1 + ret).cumprod()
    g = (eq > eq.rolling(ma).mean()).astype(float).shift(1).fillna(0.0)
    return g * (hi - lo) + lo


def voltgt(ret, tv):
    return (tv / (ret.rolling(20).std() * np.sqrt(ANN))).clip(0, 1).shift(1).fillna(0.0)


def line(name, r):
    f, i, o = metrics(r), metrics(r[is_m]), metrics(r[oos_m])
    print(
        f"  {name:34} FULL Sh={f['sharpe']:.2f} DD={f['maxdd']:+.1%} | IS={i['sharpe']:.2f} | OOS={o['sharpe']:.2f}"
    )


for label, base in (("SPY", spy), ("EW-universe", ew)):
    print(f"\n=== {label} ===")
    line(f"{label} buy-hold", base)
    for ma in (100, 150, 200, 250):
        line(f"{label} x regime MA{ma} (0.3-1.0)", base * gate(spy, ma, 0.3, 1.0))
    print("  -- mapping sensitivity (MA200) --")
    for lo, hi in ((0.0, 1.0), (0.5, 1.0)):
        line(f"{label} x regime MA200 ({lo}-{hi})", base * gate(spy, 200, lo, hi))
    print("  -- vol-target head-to-head --")
    for tv in (0.10, 0.15, 0.20):
        line(f"{label} x voltgt{int(tv * 100)}", base * voltgt(base, tv))
    # combine trend + voltgt
    line(
        f"{label} x regime x voltgt15",
        base * gate(spy, 200, 0.3, 1.0) * voltgt(base, 0.15),
    )

print("\n[DONE] regime sensitivity")
