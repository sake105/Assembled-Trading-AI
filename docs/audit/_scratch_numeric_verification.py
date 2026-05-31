"""THROWAWAY scratch — Agent-4 numeric verification (2026-05-30).

Independent re-implementation of realized vol / Sharpe / drawdown / vol-target
scaling, compared against the production functions. Imports the installed
package ONLY to call the production code; the reference math below uses raw
numpy/pandas as a calculator (NO repo metric functions).

Safe to delete. Writes nothing to disk.
"""

from __future__ import annotations

import os
import sys

# Repo root = two levels up from docs/audit/. Add to path so `src.assembled_core`
# resolves regardless of editable-install state.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. Deterministic dataset: a fixed synthetic daily return series, length 252.
# ----------------------------------------------------------------------------
rng = np.random.default_rng(20260530)
# Drift + noise so Sharpe is non-trivial and there are real drawdowns.
daily_ret = rng.normal(loc=0.0006, scale=0.011, size=252)
# Inject a deliberate drawdown stretch (days 100-130 negative) for a clean MDD.
daily_ret[100:130] = rng.normal(loc=-0.004, scale=0.011, size=30)
daily_ret = np.round(daily_ret, 6)  # freeze to 6dp for reproducibility

START_CAP = 10_000.0
equity_vals = START_CAP * np.cumprod(1.0 + daily_ret)
equity_vals = np.insert(equity_vals, 0, START_CAP)  # prepend t0 = start capital
# -> 253 equity points, 252 returns via pct_change

dates = pd.date_range("2024-01-01", periods=len(equity_vals), freq="B")
equity_series = pd.Series(equity_vals, index=dates)
equity_df = pd.DataFrame({"timestamp": dates, "equity": equity_vals})

# Returns derived from the equity curve via pct_change (what production uses).
ret_series = equity_series.pct_change().dropna()

print("=" * 70)
print("DATASET")
print(
    f"  equity points : {len(equity_vals)}  (start={equity_vals[0]:.2f}, end={equity_vals[-1]:.4f})"
)
print(f"  returns (n)   : {len(ret_series)}")
print(f"  ret mean      : {ret_series.mean():.8f}")
print(f"  ret std ddof1 : {ret_series.std(ddof=1):.8f}")
print(f"  ret std ddof0 : {ret_series.std(ddof=0):.8f}")
print()

PPY = 252


def cmp(name, ref, prod, tol=1e-9):
    if prod is None:
        print(f"  {name:22s} ref={ref!r:>18}  prod=None  -> UNSURE(prod None)")
        return
    absd = abs(ref - prod)
    reld = absd / abs(ref) if ref != 0 else float("nan")
    verdict = "MATCH" if absd <= tol or (ref != 0 and reld <= 1e-9) else "MISMATCH"
    print(
        f"  {name:22s} ref={ref:18.10f}  prod={prod:18.10f}  absd={absd:.3e}  reld={reld:.3e}  {verdict}"
    )


# ----------------------------------------------------------------------------
# 2. REFERENCE math (first principles, raw numpy/pandas)
# ----------------------------------------------------------------------------
# Realized annualized volatility (sample std, ddof=1, annualize sqrt(252))
ref_vol = float(ret_series.std(ddof=1) * np.sqrt(PPY))

# Sharpe (rf=0): mean/std(ddof=1) * sqrt(252)
ref_sharpe_rf0 = float(ret_series.mean() / ret_series.std(ddof=1) * np.sqrt(PPY))

# Sharpe with rf=0.02 annual: subtract per-period rf = rf/252 from mean
RF = 0.02
ref_sharpe_rf2 = float(
    (ret_series.mean() - RF / PPY) / ret_series.std(ddof=1) * np.sqrt(PPY)
)

# Drawdown — ABSOLUTE on equity (production: equity - expanding().max())
running_peak = equity_series.cummax()
dd_abs = equity_series - running_peak
ref_mdd_abs = float(dd_abs.min())
# production max_drawdown_pct = (max_dd_abs / GLOBAL peak) * 100
ref_global_peak = float(running_peak.max())
ref_mdd_pct_prodstyle = float(ref_mdd_abs / ref_global_peak * 100.0)
# TEXTBOOK pct drawdown = min over t of (equity/peak_to_date - 1) * 100
dd_pct_textbook = (equity_series / running_peak - 1.0) * 100.0
ref_mdd_pct_textbook = float(dd_pct_textbook.min())

# ----------------------------------------------------------------------------
# 3. PRODUCTION calls
# ----------------------------------------------------------------------------
try:
    from assembled_core.qa.metrics import (  # noqa: E402
        compute_sharpe_ratio,
        compute_drawdown,
        compute_equity_metrics,
    )
    from assembled_core.risk.vol_targeting import (  # noqa: E402
        compute_realized_vol,
        compute_vol_scale_factor,
    )
    from assembled_core.strategies.vol_target_overlay import (  # noqa: E402
        generate_vol_target_signals_from_prices,
    )

    print("[import] via bare 'assembled_core'")
except ModuleNotFoundError:
    from src.assembled_core.qa.metrics import (  # noqa: E402
        compute_sharpe_ratio,
        compute_drawdown,
        compute_equity_metrics,
    )
    from src.assembled_core.risk.vol_targeting import (  # noqa: E402
        compute_realized_vol,
        compute_vol_scale_factor,
    )
    from src.assembled_core.strategies.vol_target_overlay import (  # noqa: E402
        generate_vol_target_signals_from_prices,
    )

    print("[import] via 'src.assembled_core'")

prod_sharpe_rf0 = compute_sharpe_ratio(ret_series, freq="1d", risk_free_rate=0.0)
prod_sharpe_rf2 = compute_sharpe_ratio(ret_series, freq="1d", risk_free_rate=RF)

# Production volatility comes out of compute_equity_metrics (uses ddof=1 std*sqrt)
# NOTE: compute_equity_metrics normalizes equity if start within 10% of start_cap.
# Our equity starts EXACTLY at start_cap so normalization is a no-op (scale=1).
em = compute_equity_metrics(
    equity_df, start_capital=START_CAP, freq="1d", risk_free_rate=0.0
)
prod_vol = em.volatility
prod_sharpe_em = em.sharpe_ratio

dd_series, prod_mdd_abs, prod_mdd_pct, prod_cur_dd = compute_drawdown(equity_series)

print("=" * 70)
print("METRIC COMPARISON  (ref = independent first-principles)")
print("-" * 70)
cmp("realized_vol (ann)", ref_vol, prod_vol)
cmp("sharpe rf=0", ref_sharpe_rf0, prod_sharpe_rf0)
cmp("sharpe rf=0 (em)", ref_sharpe_rf0, prod_sharpe_em)
cmp("sharpe rf=0.02", ref_sharpe_rf2, prod_sharpe_rf2)
cmp("max_drawdown (abs)", ref_mdd_abs, prod_mdd_abs)
print()
print("  drawdown-pct definitions:")
cmp("  mdd_pct prod-style", ref_mdd_pct_prodstyle, prod_mdd_pct)
print(f"    [reference textbook peak-to-date mdd_pct = {ref_mdd_pct_textbook:.10f}]")
print(
    f"    [delta prodstyle-vs-textbook            = {ref_mdd_pct_prodstyle - ref_mdd_pct_textbook:+.6f} pct-points]"
)
print()

# ----------------------------------------------------------------------------
# 4. compute_realized_vol (risk/vol_targeting.py) on the SAME returns tail
# ----------------------------------------------------------------------------
print("=" * 70)
print("risk/vol_targeting.compute_realized_vol  (lookback=20, ann=252)")
LB = 20
ref_rv20 = float(ret_series.tail(LB).std(ddof=1) * np.sqrt(252))
prod_rv20 = compute_realized_vol(ret_series, lookback_days=LB, annualize_factor=252.0)
cmp("realized_vol last-20", ref_rv20, prod_rv20)

# scale factor: target/realized clamped
TGT = 0.12
ref_scale = float(max(0.0, min(1.5, TGT / ref_rv20)))
prod_scale = compute_vol_scale_factor(prod_rv20, TGT, min_scale=0.0, max_scale=1.5)
cmp("vol_scale_factor", ref_scale, prod_scale)
print()

# ----------------------------------------------------------------------------
# 5. vol_target_overlay strategy — feed a known SPY price path, hand-derive the
#    last-bar w_spy and compare to the strategy's score.
# ----------------------------------------------------------------------------
print("=" * 70)
print("strategies/vol_target_overlay.generate_vol_target_signals_from_prices")
# Build a SPY price series long enough for sma_window. Use small windows so we
# can verify by hand: vol_lookback=20, sma_window=30.
VL, SW = 20, 30
n_bars = 60
spy_rng = np.random.default_rng(7)
spy_ret = np.round(spy_rng.normal(0.0005, 0.01, size=n_bars - 1), 6)
spy_close = 400.0 * np.cumprod(np.insert(1.0 + spy_ret, 0, 1.0))
spy_dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
prices = pd.DataFrame({"timestamp": spy_dates, "symbol": "SPY", "close": spy_close})
# add IEF rows (flat) so the panel has both
ief = pd.DataFrame({"timestamp": spy_dates, "symbol": "IEF", "close": 100.0})
prices = pd.concat([prices, ief], ignore_index=True)

out = generate_vol_target_signals_from_prices(
    prices,
    target_vol=0.12,
    vol_lookback=VL,
    sma_window=SW,
    defensive_asset="IEF",
    risk_asset="SPY",
)
# Hand-derive last-bar w_spy from the SPY series itself.
spy_df = (
    prices[prices["symbol"] == "SPY"].sort_values("timestamp").reset_index(drop=True)
)
r = spy_df["close"].pct_change()
rvol_full = r.rolling(VL, min_periods=VL).std() * np.sqrt(252)
sma_full = spy_df["close"].rolling(SW, min_periods=SW).mean()
w = np.minimum(1.0, 0.12 / rvol_full.clip(lower=1e-9))
below = spy_df["close"] < sma_full
w = w.copy()
w[below] = w[below] * 0.5
# last valid bar
valid_mask = rvol_full.notna() & sma_full.notna()
last_i = spy_df.index[valid_mask][-1]
ref_w_spy_last = float(w.iloc[last_i])
ref_w_ief_last = 1.0 - ref_w_spy_last

prod_spy_last = float(
    out[out["symbol"] == "SPY"].sort_values("timestamp")["score"].iloc[-1]
)
prod_ief_last = float(
    out[out["symbol"] == "IEF"].sort_values("timestamp")["score"].iloc[-1]
)

cmp("vt overlay w_spy(last)", ref_w_spy_last, prod_spy_last)
cmp("vt overlay w_ief(last)", ref_w_ief_last, prod_ief_last)
print(
    f"    [SPY last close={spy_df['close'].iloc[last_i]:.4f}  sma={sma_full.iloc[last_i]:.4f}  "
    f"rvol={rvol_full.iloc[last_i]:.6f}  below_sma={bool(below.iloc[last_i])}]"
)
print()
print("DONE")
