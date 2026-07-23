"""Is the intraday earnings-gap REVERSAL deployable? Split gap-up vs gap-down, net of
realistic earnings-day costs, long-only slice (gap-down buy only), + DSR on the fade.
Reads the saved event table (no re-pull)."""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe  # noqa: E402

ev = pd.read_parquet(
    "research/fable_exploration/experiment/first_minutes_events.parquet"
)
ev = ev.dropna(subset=["gap", "drift_close"])
up, dn = ev[ev["gap"] > 0], ev[ev["gap"] < 0]
print(f"events={len(ev)}  gap-up={len(up)}  gap-down={len(dn)}")


def tstat(s):
    s = pd.Series(s).dropna()
    return (
        s.mean() / (s.std() / np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else np.nan
    )


print("\n=== raw intraday drift (open->close) by gap sign ===")
for nm, d in (
    ("gap-UP (gapped up at open)", up),
    ("gap-DOWN (gapped down at open)", dn),
):
    s = d["drift_close"]
    print(
        f"  {nm:32} n={len(d)} mean={s.mean():+.3%} t={tstat(s):+.2f} "
        f"(reversal if sign opposes gap)"
    )

print("\n=== FADE strategy: trade AGAINST the gap at open, hold to close ===")
fade = -np.sign(ev["gap"]) * ev["drift_close"]  # +ve = fade profits
for cost in (0.0, 0.0016, 0.0030):  # 0 / 16bps / 30bps round-trip (earnings-day spread)
    net = fade - cost
    print(
        f"  cost={cost * 1e4:>2.0f}bps: mean={net.mean():+.3%}/trade t={tstat(net):+.2f} "
        f"win%={100 * (net > 0).mean():.0f} sr_pp={net.mean() / net.std():.3f}"
    )

print("\n=== LONG-ONLY deployable slice (buy gap-DOWN names at open, hold close) ===")
for cost in (0.0, 0.0016, 0.0030):
    net = dn["drift_close"] - cost
    print(
        f"  cost={cost * 1e4:>2.0f}bps: mean={net.mean():+.3%}/trade t={tstat(net):+.2f} n={len(dn)} "
        f"win%={100 * (net > 0).mean():.0f}"
    )

print("\n=== DSR on the FADE (per-trade stream, cumulative ~40 trials) ===")
net = (fade - 0.0016).dropna()
for nt in (1, 40):
    d = deflated_sharpe(net, n_trials=nt)
    print(
        f"  n_trials={nt}: prob={d.deflated_sharpe_probability:.4f} sr_pp={d.sharpe_observed:.4f} passes5%={d.passes_5pct}"
    )

print(
    "\nNOTE: fade needs SHORTING the gap-ups (not long-only-deployable); long-only slice"
)
print(
    "(gap-down buy) is the only directly usable piece. Single 2024-26 window, mega-cap names."
)
print("\n[DONE] first-minutes split")
