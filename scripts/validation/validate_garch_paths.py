"""GARCH path validation: does real Sharpe land in the simulated 5-95% band?

Procedure:
  1. Download SPX daily returns 2020-2024 (real data via yfinance).
  2. Fit GARCH(1,1) to those returns using the `arch` library.
  3. Run N_PATHS=1000 simulated paths with our generate_garch_returns() using
     the fitted omega/alpha/beta parameters.
  4. Apply a simple 60/200-day momentum strategy to each simulated path.
  5. Apply the same strategy to the real SPX data.
  6. Compare: is the real strategy Sharpe within the 5-95% percentile band?

Strategy (kept simple intentionally):
  - Signal: 1 if 60d rolling mean return > 0, else 0 (long or flat, no short).
  - Position: fully invested or fully flat.
  - Annual Sharpe = mean(daily_ret * signal_lag1) / std(...) * sqrt(252).

Interpretation:
  - If real Sharpe is *inside* the band: GARCH captures the return dynamics well.
  - If real Sharpe is *outside* the band: model is missing regime / dependency structure.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Step 1: Load SPX returns — local parquet first, yfinance fallback, then synthetic
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parents[2]
_LOCAL_PARQUET = _REPO_ROOT / "output" / "aggregates" / "eod_1d.parquet"

print("[1/5] Loading SPX returns ...")
spx_returns: pd.Series | None = None

if _LOCAL_PARQUET.exists():
    try:
        _df = pd.read_parquet(_LOCAL_PARQUET)
        # Try common column layouts produced by the EOD pipeline
        for _col in ("^SPX", "SPX", "spx"):
            if _col in _df.columns:
                spx_returns = _df[_col].dropna().pct_change().dropna()
                break
        if spx_returns is None and "close" in _df.columns:
            spx_returns = _df["close"].dropna().pct_change().dropna()
        if spx_returns is not None:
            print(f"    [local] {len(spx_returns)} daily returns from {_LOCAL_PARQUET.name}")
    except Exception as _e:
        print(f"    [WARN] local parquet failed ({_e}), trying yfinance ...")

if spx_returns is None:
    try:
        import yfinance as yf
        raw = yf.download("^SPX", start="2020-01-01", end="2024-12-31",
                          progress=False, auto_adjust=True)
        spx_close = raw["Close"].squeeze().dropna()
        spx_returns = spx_close.pct_change().dropna()
        print(f"    [yfinance] {len(spx_returns)} daily returns "
              f"({spx_returns.index[0].date()} to {spx_returns.index[-1].date()})")
    except Exception as e:
        print(f"    [WARN] yfinance failed ({e}). Using synthetic stand-in.")

if spx_returns is None or len(spx_returns) < 100:
    print("    [synthetic] no real data available — results are illustrative only")
    spx_returns = pd.Series(np.random.default_rng(0).normal(0.0004, 0.013, 1257))

# ---------------------------------------------------------------------------
# Step 2: Fit GARCH(1,1)
# ---------------------------------------------------------------------------
print("[2/5] Fitting GARCH(1,1) to real SPX returns ...")
from arch import arch_model  # type: ignore

ret_pct = spx_returns.values * 100  # arch expects percentage returns
garch_model = arch_model(ret_pct, vol="GARCH", p=1, q=1, dist="normal", mean="Zero")
res = garch_model.fit(disp="off")

omega = float(res.params["omega"]) / 10_000   # convert from pct² to decimal²
alpha = float(res.params["alpha[1]"])
beta  = float(res.params["beta[1]"])
unconditional_vol = float(np.sqrt(omega / max(1 - alpha - beta, 1e-8)))

print(f"    omega={omega:.6e}  alpha={alpha:.4f}  beta={beta:.4f}")
print(f"    Unconditional daily vol: {unconditional_vol*100:.3f}%  "
      f"({unconditional_vol*np.sqrt(252)*100:.1f}% annualised)")

# ---------------------------------------------------------------------------
# Step 3: Run 1000 simulated paths
# ---------------------------------------------------------------------------
N_PATHS  = 1000
N_DAYS   = len(spx_returns)
LOOKBACK = 60      # momentum signal window (days)

print(f"[3/5] Simulating {N_PATHS} GARCH paths ({N_DAYS} days each) ...")
from assembled_core.data.synthetic_generator import generate_garch_returns

sim_sharpes: list[float] = []
rng_global = np.random.default_rng(42)

for i in range(N_PATHS):
    seed = int(rng_global.integers(0, 2**31))
    df = generate_garch_returns(
        n_days=N_DAYS, n_assets=1,
        omega=omega, alpha=alpha, beta=beta,
        mean_annual=0.08,   # conservative drift
        seed=seed,
    )
    r = df["ASSET_0"].values

    # --- momentum strategy ---
    sig = pd.Series(r).rolling(LOOKBACK).mean().shift(1)  # lag 1 avoids look-ahead
    sig_binary = (sig > 0).astype(float)
    strat_ret = r * sig_binary.values

    mean_r = strat_ret.mean()
    std_r  = strat_ret.std()
    sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-8 else 0.0
    sim_sharpes.append(sharpe)

sim_sharpes_arr = np.array(sim_sharpes)
p5  = float(np.percentile(sim_sharpes_arr, 5))
p25 = float(np.percentile(sim_sharpes_arr, 25))
p50 = float(np.percentile(sim_sharpes_arr, 50))
p75 = float(np.percentile(sim_sharpes_arr, 75))
p95 = float(np.percentile(sim_sharpes_arr, 95))

print(f"    Simulated Sharpe band:  p5={p5:.2f}  p25={p25:.2f}  "
      f"p50={p50:.2f}  p75={p75:.2f}  p95={p95:.2f}")

# ---------------------------------------------------------------------------
# Step 4: Real SPX strategy Sharpe
# ---------------------------------------------------------------------------
print("[4/5] Computing real SPX strategy Sharpe ...")
r_real = spx_returns.values
sig_real = pd.Series(r_real).rolling(LOOKBACK).mean().shift(1)
sig_binary_real = (sig_real > 0).astype(float)
strat_real = r_real * sig_binary_real.values
real_sharpe = float(strat_real.mean() / strat_real.std() * np.sqrt(252)) if strat_real.std() > 0 else 0.0
real_buyhold_sharpe = float(r_real.mean() / r_real.std() * np.sqrt(252)) if r_real.std() > 0 else 0.0

print(f"    Real SPX strategy Sharpe:    {real_sharpe:.3f}")
print(f"    Real SPX buy-hold Sharpe:    {real_buyhold_sharpe:.3f}")

# ---------------------------------------------------------------------------
# Step 5: Is real Sharpe inside the band?
# ---------------------------------------------------------------------------
in_band_90 = p5 <= real_sharpe <= p95
in_band_50 = p25 <= real_sharpe <= p75

print("\n" + "=" * 60)
print("GARCH PATH VALIDATION REPORT")
print("=" * 60)
print("\nGARCH(1,1) parameters (fitted on SPX 2020-2024):")
print(f"  omega  = {omega:.6e}  (long-run variance)")
print(f"  alpha  = {alpha:.4f}        (ARCH — shock persistence)")
print(f"  beta   = {beta:.4f}        (GARCH — variance persistence)")
print(f"  alpha+beta = {alpha+beta:.4f}  (persistence; >0.98 = very sticky)")

print(f"\nSimulated Sharpe distribution (N={N_PATHS} paths, {LOOKBACK}d momentum strategy):")
print(f"  5th pct:   {p5:+.3f}")
print(f"  25th pct:  {p25:+.3f}")
print(f"  Median:    {p50:+.3f}")
print(f"  75th pct:  {p75:+.3f}")
print(f"  95th pct:  {p95:+.3f}")

print(f"\nReal SPX strategy Sharpe: {real_sharpe:+.3f}")
print(f"  In 90% band [{p5:.2f}, {p95:.2f}]: {'YES' if in_band_90 else 'NO'}")
print(f"  In 50% band [{p25:.2f}, {p75:.2f}]: {'YES' if in_band_50 else 'NO'}")

if in_band_90:
    print("\nVerdict: GARCH captures real-world return distribution well.")
    print("         Real strategy Sharpe is statistically plausible under the model.")
else:
    pct_rank = float(np.mean(sim_sharpes_arr < real_sharpe)) * 100
    print(f"\nVerdict: Real Sharpe ({real_sharpe:.2f}) is at percentile {pct_rank:.0f}")
    if pct_rank > 95:
        print("         Real outperformance exceeds model expectation — check survivorship bias.")
    else:
        print("         Real underperformance — possibly due to regime not captured (COVID crash).")
