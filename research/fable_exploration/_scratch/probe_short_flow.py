"""H6 raw-signal gate: daily short-flow ratio (ShortVol/TotalVol) -> forward returns.
BJZ 2008: high relative shorting predicts NEGATIVE returns -> expect negative IC,
i.e. a LONG-low-shortflow / SHORT-high-shortflow tilt. Cross-sectional daily IC.
PIT: signal at EOD t is enter-able at close t+1; forward return measured from t+1."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)

sv = pd.read_parquet(
    "research/fable_exploration/data/short_volume_2023-01-01_2024-12-31.parquet"
)
sv["date"] = pd.to_datetime(sv["date"]).dt.normalize()
sv["ratio"] = sv["short_volume"] / sv["total_volume"].where(sv["total_volume"] > 0)
ratio = sv.pivot_table(
    index="date", columns="symbol", values="ratio", aggfunc="last"
).sort_index()

px = pd.read_parquet("output/aggregates/daily.parquet")
px["date"] = (
    pd.to_datetime(px["timestamp"], utc=True).dt.normalize().dt.tz_localize(None)
)
close = px.pivot_table(
    index="date", columns="symbol", values="close", aggfunc="last"
).sort_index()

# align
common_d = ratio.index.intersection(close.index)
common_s = sorted(set(ratio.columns) & set(close.columns) - {"SPY"})
ratio = ratio.loc[common_d, common_s]
close = close.loc[:, common_s]
print(f"aligned: {len(common_d)} days, {len(common_s)} symbols")

# abnormal short-flow = ratio - trailing 20d mean per symbol (cross-sectional structure removed)
abn = ratio - ratio.rolling(20, min_periods=10).mean()


def cs_ic(sig, h):
    """daily cross-sectional Spearman IC of signal_t vs return (t+1 -> t+1+h),
    market-neutralised (subtract cross-sectional mean fwd return)."""
    fwd = close.shift(-h) / close - 1.0
    fwd = fwd.sub(fwd.mean(axis=1), axis=0)  # cross-sectional demean
    fwd_entry = fwd.shift(-1).reindex(sig.index)  # enter t+1
    ics = []
    for d in sig.index:
        a = sig.loc[d]
        b = fwd_entry.loc[d] if d in fwd_entry.index else None
        if b is None:
            continue
        df = pd.concat([a, b], axis=1).dropna()
        if len(df) >= 20:
            ics.append(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))
    ics = pd.Series(ics)
    t = ics.mean() / (ics.std() / np.sqrt(len(ics))) if len(ics) > 2 else np.nan
    return ics.mean(), t, len(ics)


for name, sig in (("raw ratio", ratio), ("abnormal (vs 20d)", abn)):
    print(f"\n=== signal: {name} ===")
    for h in (1, 5, 20):
        ic, t, n = cs_ic(sig, h)
        print(
            f"  +{h:>2}d: mean daily IC={ic:+.4f}  t={t:+.2f}  (n_days={n})  "
            f"[BJZ expects NEGATIVE]"
        )

# tercile spread: low-shortflow minus high-shortflow, 5d fwd, entered t+1
fwd5 = close.shift(-5) / close - 1.0
fwd5 = fwd5.sub(fwd5.mean(axis=1), axis=0).shift(-1)
rows = []
for d in abn.index:
    s = abn.loc[d].dropna()
    f = fwd5.loc[d].reindex(s.index) if d in fwd5.index else None
    if f is None or s.size < 30:
        continue
    q = s.rank(pct=True)
    lo = f[q <= 1 / 3].mean()
    hi = f[q >= 2 / 3].mean()
    rows.append(lo - hi)  # long-low minus short-high
spread = pd.Series(rows).dropna()
print(
    f"\nL/S (low-minus-high abnormal short-flow, 5d): mean={spread.mean():+.4%}/period "
    f"t={spread.mean() / (spread.std() / np.sqrt(len(spread))):+.2f} n={len(spread)}"
)
print("\n[DONE] short-flow probe")
