"""Event-directional test (the §5.5 'Hormuz' thesis at the resolution we CAN measure):
GPR geopolitical-risk spike -> do energy/defense/gold outperform SPY over the following
days/weeks, per the news_alpha asset_router mapping? Daily GPR (try free Caldara-Iacoviello)
else our monthly GPR. ETFs via yfinance. Read-only research."""

from __future__ import annotations
import io
import os
import urllib.request
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)

# event assets per asset_router: energy, defense, gold, safety, oil + SPY benchmark
ETFS = ["SPY", "XLE", "XOM", "GLD", "ITA", "LMT", "NOC", "RTX", "TLT", "USO", "DBC"]

import yfinance as yf

px = yf.download(ETFS, period="max", auto_adjust=True, progress=False)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None)
ret = px.pct_change(fill_method=None)
print(
    f"ETFs: {[c for c in px.columns]}  range {px.index.min().date()}..{px.index.max().date()}"
)

# ---- daily GPR (try free) ----
gpr_daily = None
for url in (
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
    "https://www.matteoiacoviello.com/gpr_files/gpr_daily_recent.xls",
):
    try:
        raw = urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research"}),
            timeout=25,
        ).read()
        g = pd.read_excel(io.BytesIO(raw))
        dcol = next((c for c in g.columns if "date" in str(c).lower()), g.columns[0])
        gcol = next(
            (
                c
                for c in g.columns
                if str(c).upper() in ("GPRD", "GPR", "GPRD_MA7", "DAILY")
            ),
            None,
        )
        if gcol is None:
            gcol = [c for c in g.columns if "GPR" in str(c).upper()][0]
        gpr_daily = pd.Series(
            g[gcol].values, index=pd.to_datetime(g[dcol])
        ).sort_index()
        print(
            f"[daily GPR] fetched {url} -> {len(gpr_daily)} obs {gpr_daily.index.min().date()}..{gpr_daily.index.max().date()} col={gcol}"
        )
        break
    except Exception as e:
        print(f"[daily GPR] {url}: {type(e).__name__}: {e}")

# ---- monthly GPR (we have it) ----
gm = pd.read_parquet("data/cache/gpr/sheet1.parquet")
gm["month"] = pd.to_datetime(gm["month"])
gpr_m = pd.Series(gm["GPR"].values, index=gm["month"]).dropna().sort_index()
print(
    f"[monthly GPR] {len(gpr_m)} obs {gpr_m.index.min().date()}..{gpr_m.index.max().date()}"
)


def excess(sym, h_ret):
    return h_ret[sym] - h_ret["SPY"] if sym in h_ret else None


# ================= MONTHLY event study =================
print("\n=== MONTHLY: high-GPR-spike month -> NEXT-month excess return vs SPY ===")
mret = px.resample("ME").last().pct_change(fill_method=None)
gm_me = gpr_m.resample("ME").last()
# spike = GPR change in top quintile (past-only z of monthly change)
chg = gm_me.diff()
z = (chg - chg.expanding(min_periods=24).mean()) / chg.expanding(min_periods=24).std()
spike = z.shift(0) > 1.0  # this month's GPR jumped (known at month end)
fwd = mret.shift(-1)  # next month's return
common = gm_me.index.intersection(fwd.index)
sp = spike.reindex(common).fillna(False)
print(f"  n_months={len(common)} spike_months={int(sp.sum())} (2000+ where ETFs exist)")
for s in ("XLE", "XOM", "GLD", "ITA", "LMT", "TLT", "USO", "DBC"):
    if s not in fwd:
        continue
    ex = (fwd[s] - fwd["SPY"]).reindex(common)
    hi, lo = ex[sp].mean(), ex[~sp].mean()
    t = ex[sp].mean() / (ex[sp].std() / np.sqrt(max(sp.sum(), 1)))
    print(
        f"  {s:4}: spike-month next-mo excess={hi:+.3%} (t={t:+.2f}) | non-spike={lo:+.3%} | diff={hi - lo:+.3%}"
    )

# ================= DAILY event study (if available) =================
if gpr_daily is not None:
    print("\n=== DAILY: GPR spike day -> forward 5/10/21d excess vs SPY ===")
    gd = gpr_daily.reindex(px.index, method="ffill")
    zc = (gd - gd.rolling(252, min_periods=60).mean()) / gd.rolling(
        252, min_periods=60
    ).std()
    spk = (zc > 2.0) & (zc.shift(1) <= 2.0)  # fresh spike (cross above +2 sigma)
    spk = spk.shift(1).fillna(False)  # PIT: act day AFTER the spike is known
    print(f"  spike days: {int(spk.sum())}")
    for h in (5, 10, 21):
        fwdh = px.shift(-h) / px - 1.0
        print(f"  +{h}d:", end=" ")
        for s in ("XLE", "GLD", "ITA", "USO", "DBC"):
            if s not in px:
                continue
            ex = (fwdh[s] - fwdh["SPY"])[spk]
            print(
                f"{s}={ex.mean():+.2%}(t{ex.mean() / (ex.std() / np.sqrt(max(len(ex.dropna()), 1))):+.1f})",
                end="  ",
            )
        print()
print("\n[DONE] gpr event study")
