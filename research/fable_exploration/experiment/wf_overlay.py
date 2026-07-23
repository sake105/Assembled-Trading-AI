"""Per-fold (walk-forward) consistency of the risk overlays. The overlay is causal
(MA/vol use only past data) so each calendar year is a clean OOS fold. A deployable
overlay should help across folds, not just full-period. SPY (survivorship-immune)
+ enriched EW-universe."""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from engine import _wide_close, MIN_HISTORY  # noqa: E402

ANN = 252
close = _wide_close()
R = close.pct_change(fill_method=None)
dates = pd.DatetimeIndex(R.index)
spy = R["SPY"].reindex(dates).fillna(0.0)
universe = sorted(
    set(c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY) - {"SPY"}
)
ew = R.reindex(columns=universe).mean(axis=1)


def sharpe(r):
    r = pd.Series(r).dropna()
    return (
        float(r.mean() / r.std() * np.sqrt(ANN))
        if len(r) > 20 and r.std() > 0
        else np.nan
    )


def maxdd(r):
    eq = (1 + pd.Series(r).dropna()).cumprod()
    return float((eq / eq.cummax() - 1).min()) if len(eq) else np.nan


def gate(ma):
    eq = (1 + spy).cumprod()
    return (eq > eq.rolling(ma).mean()).astype(float).shift(1).fillna(0.0) * 0.7 + 0.3


def vt(ret, tv=0.15):
    return (tv / (ret.rolling(20).std() * np.sqrt(ANN))).clip(0, 1).shift(1).fillna(0.0)


def overlays(base):
    return {
        "buyhold": base,
        "regimeMA150": base * gate(150),
        "regimeMA200": base * gate(200),
        "voltgt15": base * vt(base),
        "regimexvol": base * gate(200) * vt(base),
    }


for label, base in (("SPY", spy), ("EW-universe", ew)):
    print(f"\n================= {label} — per-year Sharpe (MaxDD) =================")
    ov = overlays(base)
    years = sorted(set(dates.year))
    hdr = "year   " + "".join(f"{k:>14}" for k in ov)
    print(hdr)
    winsh = {k: 0 for k in ov if k != "buyhold"}
    windd = {k: 0 for k in ov if k != "buyhold"}
    nfold = 0
    for y in years:
        m = dates.year == y
        if m.sum() < 60:
            continue
        nfold += 1
        cells = []
        bh_sh, bh_dd = sharpe(base[m]), maxdd(base[m])
        for k, s in ov.items():
            cells.append(f"{sharpe(s[m]):+.2f}({maxdd(s[m]):+.0%})")
            if k != "buyhold":
                if sharpe(s[m]) > bh_sh:
                    winsh[k] += 1
                if maxdd(s[m]) > bh_dd:
                    windd[k] += 1
        print(f"{y}  " + "".join(f"{c:>14}" for c in cells))
    print(
        f"\n  folds where overlay BEATS buy-hold (Sharpe / smaller-MaxDD), n_folds={nfold}:"
    )
    for k in winsh:
        print(f"    {k:14} Sharpe-wins={winsh[k]}/{nfold}  DD-wins={windd[k]}/{nfold}")
print("\n[DONE] walk-forward overlay consistency")
