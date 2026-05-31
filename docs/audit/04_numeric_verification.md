# Audit 04 — Independent Numeric Verification

**Agent:** 4 of 5 (read-only system audit)
**Date:** 2026-05-30
**Scope:** Independent re-implementation of the core metric math (realized
volatility, Sharpe, drawdown) and the simplest strategy (`vol_target_overlay`),
compared numerically against the production code on a shared deterministic
dataset.
**Production touched:** NONE. All compute ran from throwaway scratch files in
`docs/audit/` (see "Scratch files" at the bottom).

---

## 1. Production formulas (quoted, with file:line)

### Realized / annualized volatility
`src/assembled_core/qa/metrics.py:540-541` (inside `compute_equity_metrics`):
```python
periods_per_year = _get_periods_per_year(freq)   # "1d" -> 252  (line 22, 102)
volatility = float(returns.std() * np.sqrt(periods_per_year))
```
- `returns.std()` -> pandas default **ddof=1** (sample std).
- Returns are **simple** `pct_change()` (`_compute_returns`, line 119), inf/NaN dropped.
- Annualization factor **252** (`PERIODS_PER_YEAR_1D`, line 22).

Second realized-vol impl `src/assembled_core/risk/vol_targeting.py:44-45`
(`compute_realized_vol`): `float(tail.std(ddof=1)) * (annualize_factor**0.5)`,
default `annualize_factor=252.0`, lookback tail of 20, `min_observations=5`.
Same convention.

### Sharpe ratio
`src/assembled_core/qa/metrics.py:141-149` (`compute_sharpe_ratio`):
```python
mean_return = float(returns.mean())
std_return  = float(returns.std())          # ddof=1
periods_per_year = _get_periods_per_year(freq)   # 252
excess_return = mean_return - (risk_free_rate / periods_per_year)
sharpe = excess_return / std_return * np.sqrt(periods_per_year)
```
- Per-period risk-free = `rf_annual / 252` subtracted from the **mean** (correct).
- `std` ddof=1, annualize `sqrt(252)`. `risk_free_rate` defaults 0.0.
- Guards: `< 2` obs -> None; `std <= 0` -> None.

### Drawdown
`src/assembled_core/qa/metrics.py:211-219` (`compute_drawdown`):
```python
rolling_max     = equity.expanding().max()
drawdown_series = equity - rolling_max               # ABSOLUTE, negative
max_drawdown    = float(drawdown_series.min())       # abs MDD
peak_equity     = float(rolling_max.max())           # *** GLOBAL peak ***
max_drawdown_pct = float((max_drawdown / peak_equity) * 100) if peak_equity > 0 else 0.0
current_drawdown = float(drawdown_series.iloc[-1])
```
- `max_drawdown` (absolute) is textbook-correct: equity minus peak-to-date.
- `max_drawdown_pct` divides the absolute MDD by the **global** peak
  (`rolling_max.max()`), **not** by the peak-to-date at the trough. See finding F-1.

### vol_target_overlay (the strategy)
`src/assembled_core/strategies/vol_target_overlay.py:97-116`:
```python
spy["_ret"]  = spy["close"].pct_change()
spy["_rvol"] = spy["_ret"].rolling(vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
spy["_sma"]  = spy["close"].rolling(sma_window, min_periods=sma_window).mean()
spy["_w_spy"] = np.minimum(1.0, target_vol / spy["_rvol"].clip(lower=1e-9))
below_sma = spy["close"] < spy["_sma"]
spy.loc[below_sma, "_w_spy"] *= 0.5
spy["_w_def"] = 1.0 - spy["_w_spy"]
```
- `w_spy = min(1, target_vol / realized_vol)`, halved below the SMA trend filter.
- Strictly causal: `min_periods == window`, no partial-window leakage. Defaults
  `target_vol=0.12`, `vol_lookback=20`, `sma_window=200`. ddof=1, ann=252.

---

## 2. Reference (first-principles) implementation + dataset

Dataset (frozen, `np.random.default_rng(20260530)`): 252 daily simple returns
(drift 0.0006, sd 0.011) with a deliberate negative stretch days 100-129 to
force a real drawdown; equity = `10000 * cumprod(1+r)` with `t0 = 10000`
prepended (253 equity points -> 252 `pct_change` returns).

Reference numbers (raw numpy/pandas, NO repo functions):
- `ret mean = 0.00068619`, `ret std (ddof=1) = 0.01122844`, `std (ddof=0) = 0.01120614`.
- realized vol = `std(ddof=1) * sqrt(252)`.
- Sharpe = `(mean - rf/252) / std(ddof=1) * sqrt(252)`.
- MDD abs = `min(equity - cummax(equity))`; MDD% textbook = `min(equity/cummax - 1) * 100`.

For the strategy, a separate 60-bar synthetic SPY path (+ flat IEF leg) with
`vol_lookback=20, sma_window=30`; last-bar `w_spy` hand-derived from the same
rolling std / SMA.

---

## 3. Comparison tables

### Core metrics (main dataset, 252 returns)

| metric | reference | production | abs diff | rel diff | verdict |
|---|---|---|---|---|---|
| realized_vol (ann) | 0.1782459536 | 0.1782459536 | 0.0e+00 | 0.0e+00 | **MATCH** |
| Sharpe (rf=0) `compute_sharpe_ratio` | 0.9701145888 | 0.9701145888 | 0.0e+00 | 0.0e+00 | **MATCH** |
| Sharpe (rf=0) via `compute_equity_metrics` | 0.9701145888 | 0.9701145888 | 0.0e+00 | 0.0e+00 | **MATCH** |
| Sharpe (rf=0.02 annual) | 0.8579100781 | 0.8579100781 | 0.0e+00 | 0.0e+00 | **MATCH** |
| max_drawdown (absolute $) | -1855.6284 | -1855.6284 | 0.0e+00 | 0.0e+00 | **MATCH** |
| max_drawdown_pct (prod denominator) | -15.0285118 | -15.0285118 | 0.0e+00 | 0.0e+00 | **MATCH** |

(On this dataset the trough's peak-to-date equals the global peak, so prod-style
and textbook MDD% coincide at -15.0285%.)

### risk/vol_targeting.py + strategy

| metric | reference | production | abs diff | rel diff | verdict |
|---|---|---|---|---|---|
| `compute_realized_vol` (last 20) | 0.1395478261 | 0.1395478261 | 0.0e+00 | 0.0e+00 | **MATCH** |
| `compute_vol_scale_factor` (tgt 0.12) | 0.8599202389 | 0.8599202389 | 0.0e+00 | 0.0e+00 | **MATCH** |
| vol_target_overlay `w_spy` (last bar) | 0.8344187087 | 0.8344187087 | 0.0e+00 | 0.0e+00 | **MATCH** |
| vol_target_overlay `w_ief` (last bar) | 0.1655812913 | 0.1655812913 | 0.0e+00 | 0.0e+00 | **MATCH** |

All Sharpe / vol / drawdown-abs / vol-target arithmetic reproduces to **exact
float equality** (abs diff 0.0). ddof, annualization (252), simple-return
convention, and risk-free handling all match the textbook reference.

---

## 4. Finding F-1 — `max_drawdown_pct` uses the GLOBAL peak as denominator

`compute_drawdown` (metrics.py:215-217) computes
`max_drawdown_pct = max_drawdown_abs / rolling_max.max()`. `rolling_max.max()`
is the **highest equity over the whole curve**, not the peak that preceded the
trough. The standard definition normalizes the trough's loss by the peak that
immediately preceded it (`min(equity / cummax(equity) - 1)`).

**They diverge whenever the curve makes a NEW HIGH after recovering from the
worst drawdown.** Direct probe (`_scratch_dd_edgecase.py`), equity
`[100, 90, 80, 120, 200]`:

| quantity | value |
|---|---|
| max_drawdown (abs) | -20.0 (matches; correct) |
| **textbook** MDD% (÷ peak-to-date 100) | **-20.00 %** |
| **production** MDD% (÷ global peak 200) | **-10.00 %** |
| divergence | **+10.00 pct-points — production understates the drawdown** |

**Direction of bias: production reports a SMALLER (BETTER-looking) MDD%** the
moment the equity curve later prints a higher high than the pre-drawdown peak —
which is the *normal* case for any profitable long-running strategy. Magnitude
scales with how far the final peak exceeds the pre-trough peak; for a strategy
that, say, doubles over its life it can roughly halve the reported MDD%.

**Downstream contamination:** `calmar_ratio` (metrics.py:531-535) divides CAGR
by `|max_drawdown_pct/100|`, so an understated MDD% **inflates Calmar**
proportionally. `max_drawdown` (absolute $) is unaffected and correct.

**Not affected:** `compute_regime_segmented_performance` (metrics.py:1333-1335)
uses the correct `(cum - peak)/peak` peak-to-date form; the Sortino downside
deviation and all Sharpe variants are correct.

This is a real, repeatable definitional bug, not a rounding artifact. It does
not affect absolute-$ risk reporting, but any *percentage* MDD or Calmar shown
from `compute_drawdown` / `compute_equity_metrics` is optimistically biased on
profitable curves. Recommended (no change made here): normalize per peak-to-date,
e.g. `((equity / rolling_max - 1).min()) * 100`.

---

## 5. Verdict per metric

| metric | verdict |
|---|---|
| Realized / annualized volatility | **PASS** — exact match, ddof=1, ann 252 |
| Sharpe (rf=0 and rf=0.02) | **PASS** — exact match, correct per-period rf |
| `risk/vol_targeting` realized vol + scale | **PASS** — exact match |
| max_drawdown (absolute) | **PASS** — exact match |
| **max_drawdown_pct / Calmar** | **MISMATCH vs textbook (F-1)** — global-peak denominator understates MDD% (better-looking) on any curve that later makes a new high; matches its own implementation exactly but the implementation is non-standard |
| vol_target_overlay weights | **PASS** — exact match, causal windows verified |

**Bottom line:** the metric *arithmetic* (vol, Sharpe, vol-target scaling,
absolute drawdown) is numerically correct and reproduces to float precision.
The single substantive issue is the **percentage** drawdown denominator (F-1),
which biases reported MDD% and Calmar **optimistically** for profitable
strategies.

---

## 6. Scratch files (delete after audit)

Created by this agent under `docs/audit/` (no production file changed):

- `docs/audit/_scratch_numeric_verification.py` — main comparison harness.
- `docs/audit/_scratch_dd_edgecase.py` — F-1 drawdown-denominator probe.

Both are pure read/compute, write nothing to disk, and import the production
package via `src.assembled_core` (the editable install exposes the package under
the `src.` namespace; `vol_targeting.py:164` confirms this is the canonical
import path in-repo).
