"""Fair re-examination: are insider & congress REALLY dead, or under-used?
1) Coverage proof (the data IS live in the panels, not a wiring bug).
2) Standalone long baskets (insider-only / congress-only / combined), best-shot
   encodings (binary, value-weighted, cluster), +/- regime overlay, head-to-head
   vs the EW-universe baseline (the honest bar). IS / OOS / full.
The question is NOT 'do insider buys precede positive returns' (they do, +4.3%/60d
in round 1) — it is 'does an insider basket BEAT just holding the survivor universe'.
"""

from __future__ import annotations
import os
import sys
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from engine import _wide_close, metrics, IS_END, MIN_HISTORY  # noqa: E402

ANN = 252
close = _wide_close()
R = close.pct_change(fill_method=None)
dates = pd.DatetimeIndex(R.index)
spy = R["SPY"].reindex(dates).fillna(0.0)
universe = sorted(
    set(c for c in close.columns if close[c].notna().sum() >= MIN_HISTORY) - {"SPY"}
)
is_m, oos_m = dates <= IS_END, dates > IS_END


def regime(lo=0.3, hi=1.0, ma=200):
    eq = (1 + spy).cumprod()
    g = (eq > eq.rolling(ma).mean()).astype(float).shift(1).fillna(0.0)
    return g * (hi - lo) + lo


# ---- build insider & congress event tables ----
ins = pd.read_parquet("data/raw/insider_congress/form4_insider_full.parquet")
ins = ins[
    (ins["transaction_type"] == "P")
    & (~ins["is_derivative"])
    & (ins["price"] > 0)
    & (ins["shares"] > 0)
].copy()
ins["when"] = (
    pd.to_datetime(ins["available_at"], utc=True).dt.normalize().dt.tz_localize(None)
)
ins["sym"] = ins["symbol"].astype(str).str.upper()
ins["val"] = ins["value_usd"].fillna(0.0)

cg = pd.read_parquet("data/raw/insider_congress/congress_trades_full.parquet")
cg = cg[cg["type"] == "buy"].copy()
cg["when"] = (
    pd.to_datetime(cg["available_at"], utc=True).dt.normalize().dt.tz_localize(None)
)
cg["sym"] = cg["symbol"].astype(str).str.upper()
cg["val"] = cg["amount"].fillna(0.0)


def panel(ev, lookback, mode="binary"):
    """date x symbol weight panel. mode: binary / value (sum $ in window) / cluster (#owners)."""
    P = pd.DataFrame(0.0, index=dates, columns=universe)
    ev = ev[ev["sym"].isin(universe)]
    for sym, g in ev.groupby("sym"):
        col = P.columns.get_loc(sym)
        g = g.sort_values("when")
        for av, gg in g.groupby("when"):
            p = dates.searchsorted(av, side="right")
            if p >= len(dates):
                continue
            end = min(p + lookback, len(dates))
            if mode == "binary":
                P.iloc[p:end, col] = 1.0
            elif mode == "value":
                P.iloc[p:end, col] += float(gg["val"].sum())
            elif mode == "cluster":
                n = (
                    gg["reporting_owner_cik"].nunique()
                    if "reporting_owner_cik" in gg
                    else len(gg)
                )
                P.iloc[p:end, col] += float(n)
    return P


def basket(weightpanel, rebal=21, cost_bps=20.0, overlay=None):
    """long names with weight>0, weighted by the (normalized) panel weight, monthly."""
    idx = dates
    pos = list(range(0, len(idx), rebal))
    Wreb = pd.DataFrame(0.0, index=idx[pos], columns=universe)
    cost = pd.Series(0.0, index=idx)
    prev = pd.Series(0.0, index=universe)
    for p in pos:
        row = weightpanel.iloc[p].clip(lower=0)
        w = pd.Series(0.0, index=universe)
        if row.sum() > 0:
            sel = row[row > 0]
            w[sel.index] = (sel / sel.sum()).values
        Wreb.loc[idx[p]] = w.values
        cost.loc[idx[p]] = 0.5 * (w - prev).abs().sum() * cost_bps / 1e4
        prev = w
    W = Wreb.reindex(idx, method="ffill").fillna(0.0)
    Rf = R.reindex(columns=universe).reindex(idx).fillna(0.0)
    net = (W * Rf).sum(axis=1) - cost
    if overlay is not None:
        net = net * overlay
    return net, (W > 0).sum(axis=1)


def show(name, r, brd=None):
    f, i, o = metrics(r), metrics(r[is_m]), metrics(r[oos_m])
    b = (
        f" breadth~{int(brd[brd > 0].mean())}"
        if brd is not None and (brd > 0).any()
        else ""
    )
    print(
        f"  {name:38} FULL Sh={f['sharpe']:.2f} DD={f['maxdd']:+.1%} | IS={i['sharpe']:.2f} | OOS={o['sharpe']:.2f}{b}"
    )


print(f"universe={len(universe)} dates {dates.min().date()}..{dates.max().date()}")

# ---- coverage proof ----
ib = panel(ins, 63, "binary")
cb = panel(cg, 63, "binary")
print("\n=== COVERAGE (data IS live, not a bug) ===")
for nm, P, ev in (("insider P-buy", ib, ins), ("congress buy", cb, cg)):
    flagged = P > 0
    print(
        f"  {nm}: events_in_universe={ev['sym'].isin(universe).sum()}  "
        f"distinct_names={int((flagged.any()).sum())}  "
        f"avg_names_flagged/day={flagged.sum(axis=1)[flagged.sum(axis=1) > 0].mean():.1f}  "
        f"days_with_>=1={int((flagged.sum(axis=1) > 0).sum())}/{len(dates)}"
    )

# ---- baseline ----
ewp = pd.DataFrame(1.0, index=dates, columns=universe)
gate = regime()
print("\n=== BASELINE (the bar insider/congress must beat) ===")
ew, ewb = basket(ewp)
show("EW-universe (no signal)", ew, ewb)
ewr, _ = basket(ewp, overlay=gate)
show("EW-universe x regime", ewr)

print("\n=== INSIDER-only baskets, best-shot encodings ===")
for mode in ("binary", "value", "cluster"):
    n, b = basket(panel(ins, 63, mode))
    show(f"insider [{mode}]", n, b)
nr, _ = basket(panel(ins, 63, "binary"), overlay=gate)
show("insider [binary] x regime", nr)

print("\n=== CONGRESS-only baskets ===")
for mode in ("binary", "value"):
    n, b = basket(panel(cg, 63, mode))
    show(f"congress [{mode}]", n, b)
nr, _ = basket(panel(cg, 63, "binary"), overlay=gate)
show("congress [binary] x regime", nr)
# large congress buys only (>50k)
cgl = cg[cg["val"] >= 50000]
n, b = basket(panel(cgl, 63, "binary"))
show("congress >$50k [binary]", n, b)

print("\n=== INSIDER+CONGRESS combined (either event) ===")
comb = (panel(ins, 63, "binary") + panel(cg, 63, "binary")).clip(upper=1.0)
n, b = basket(comb)
show("insider|congress", n, b)
nr, _ = basket(comb, overlay=gate)
show("insider|congress x regime", nr)

print(
    "\nINTERPRET: if these ~= EW-universe, the data is real but adds NO marginal edge"
)
print("(the events precede positive returns, but so does just holding the survivors).")
print("\n[DONE] insider/congress recheck")
