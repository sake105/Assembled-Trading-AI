"""First-minutes post-earnings drift test (the intraday version of the dead daily PEAD).
For each earnings event: the overnight/after-hours move is already gapped by the liquid
regular open, so the TRADEABLE question is whether there is CONTINUATION after that open.
Enter at the first regular-session print after release (fully liquid), in the GAP's
direction; measure drift to +15/30/60min and the close. If positive gaps keep rising
intraday (and negatives keep falling) beyond costs -> a real (capacity-limited) edge."""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
IN = os.path.join(
    "research", "fable_exploration", "data", "intraday", "earnings_minute.parquet"
)

m = pd.read_parquet(IN)
m["ts"] = pd.to_datetime(m["ts"], utc=True)
m["release_ts"] = pd.to_datetime(m["release_ts"], utc=True)
m["et"] = m["ts"].dt.tz_convert("America/New_York")
m["reg"] = (
    (m["et"].dt.hour > 9) | ((m["et"].dt.hour == 9) & (m["et"].dt.minute >= 30))
) & (m["et"].dt.hour < 16)
print(f"events with bars: {m['event_id'].nunique()}  rows: {len(m)}")

rows = []
for eid, g in m.groupby("event_id"):
    g = g.sort_values("ts")
    rel = g["release_ts"].iloc[0]
    pre = g[g["ts"] < rel]
    if pre.empty:
        continue
    p_pre = pre["close"].iloc[-1]  # last print before release
    post = g[g["ts"] >= rel]
    if post.empty:
        continue
    p0 = post["close"].iloc[0]  # first print after release (instant, not tradeable)
    jump = p0 / p_pre - 1.0
    # first REGULAR-session print strictly after release = tradeable entry
    reg_after = g[(g["ts"] > rel) & (g["reg"])]
    if reg_after.empty:
        continue
    entry = reg_after.iloc[0]
    open_ts, open_px = entry["ts"], entry["close"]
    gap = open_px / p_pre - 1.0  # move captured BEFORE you can trade
    sess = reg_after[reg_after["et"].dt.date == entry["et"].date()]
    close_px = sess["close"].iloc[-1]
    rec = {
        "event_id": eid,
        "symbol": g["symbol"].iloc[0],
        "jump": jump,
        "gap": gap,
        "drift_close": close_px / open_px - 1.0,
    }
    for mn in (5, 15, 30, 60):
        nxt = sess[sess["ts"] >= open_ts + pd.Timedelta(minutes=mn)]
        rec[f"drift_{mn}m"] = (
            (nxt["close"].iloc[0] / open_px - 1.0) if len(nxt) else np.nan
        )
    rows.append(rec)

ev = pd.DataFrame(rows)
print(f"usable events: {len(ev)}  (median |gap|={ev['gap'].abs().median():.2%})")


def stats(col, mask, label):
    s = (ev[col] * np.sign(ev["gap"]))[
        mask
    ].dropna()  # signed by gap direction = continuation return
    if len(s) < 10:
        print(f"  {label:26} n={len(s)} (too few)")
        return
    t = s.mean() / (s.std() / np.sqrt(len(s)))
    print(
        f"  {label:26} n={len(s)} mean={s.mean():+.3%} median={s.median():+.3%} t={t:+.2f} "
        f"win%={100 * (s > 0).mean():.0f}"
    )


print(
    "\n=== CONTINUATION after the liquid open, signed by gap direction (tradeable) ==="
)
print("  (positive = the gap keeps going your way after you enter at the open)")
big = ev["gap"].abs() >= 0.02  # material earnings reaction (>=2% gap)
for col, lab in (
    ("drift_5m", "open->+5min"),
    ("drift_15m", "open->+15min"),
    ("drift_30m", "open->+30min"),
    ("drift_60m", "open->+60min"),
    ("drift_close", "open->close"),
):
    stats(col, slice(None), lab + " [all]")
print("  -- material gaps only (|gap|>=2%) --")
for col, lab in (
    ("drift_30m", "open->+30min"),
    ("drift_60m", "open->+60min"),
    ("drift_close", "open->close"),
):
    stats(col, big, lab + " [|gap|>=2%]")

print("\n=== reference: the GAP you CANNOT capture (pre-release -> open) ===")
print(
    f"  mean |gap|={ev['gap'].abs().mean():.2%}  mean signed jump(instant)={(ev['jump'] * np.sign(ev['gap'])).mean():+.2%}"
)
print(
    f"  corr(gap, drift_close) = {ev['gap'].corr(ev['drift_close']):+.3f}  "
    f"(positive=continuation, negative=intraday reversal)"
)
ev.to_parquet(
    os.path.join(
        "research", "fable_exploration", "experiment", "first_minutes_events.parquet"
    ),
    index=False,
)
print("\n[DONE] first-minutes test")
