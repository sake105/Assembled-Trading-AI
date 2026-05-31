# Audit 03 — Look-Ahead Bias & Backtest Correctness

- **Date:** 2026-05-30
- **Agent:** AGENT 3 (Look-Ahead Bias & Backtest Correctness)
- **Mode:** READ-ONLY. No source, script, workflow, or rule files were modified.
- **Scope:** All OOS / walk-forward backtest scripts (`scripts/_oos_wf_*.py`), their
  underlying strategy modules (`src/assembled_core/strategies/`), the shared
  walk-forward engine (`src/assembled_core/qa/walk_forward.py`,
  `backtest_engine.py`), the cost model (`src/assembled_core/pipeline/portfolio.py`),
  and the PIT / replay / leakage test suite.

## Method

For each OOS script the following five vectors were checked with file:line evidence:

1. **Look-ahead** — `.iloc[-N:]` / `.tail()` on full series, `.bfill()` on staggered
   inception, `.rolling()` without shift, signal-on-full-series-then-slice, scaler
   `fit` on whole sample.
2. **Train/test fold leakage** — scalers / cointegration vectors / normalizations
   fit on the FULL sample then applied per-fold.
3. **Lagged-weights / warmup carry-in defect** — weights lagged inside the test
   slice discarding warmup-end carry-in (anti-pattern E-027/E-028).
4. **Cost model actually applied** — the declared `commission_bps` really reaches PnL
   (a prior incident shipped 0 bps instead of 10 bps).
5. **Benchmark fairness** — SPY / 60-40 over the same period, cost, rebalance.

Cost-application was traced end-to-end:
`commission_bps` → `run_portfolio_backtest` (`backtest_engine.py:1182`) →
`_pb_simulate_equity` (`backtest_engine.py:972-982`, `if include_costs: simulate_with_costs(...)`) →
`simulate_with_costs` (`portfolio.py:20`); `total_cost_cash = commission + spread + slippage`
(`portfolio.py:164-166`) is **subtracted** from the BUY/SELL `cash_delta`
(`portfolio.py:177-190`). All scripts pass `commission_bps=10.0, include_costs=True`.

## Per-script findings

| Script | Look-ahead? | Fold leakage? | Cost bps applied? | Benchmark fair? | Evidence |
|---|---|---|---|---|---|
| `_oos_wf_trend_baseline.py` | **No** (same-bar exec caveat) | No (stateless) | Yes (10 bps) | Free pass (SPY cost-free) | signal_fn generates on warmup+test then filters `>= test_start` (`:164-170`); rolling MA causal (`rules_trend.py`, `ta_features.py:129` `min_periods`); `run_portfolio_backtest(commission_bps=10, include_costs=True)` (`:230-240`); SPY buy-hold no cost (`:240-261`) |
| `_oos_wf_mfv2.py` | **No** | No | Yes (10 bps) | Free pass | per-month slice `enriched[timestamp <= rebal_ts]` BEFORE compute (`:184`); `add_all_features` causal (`:166`); altdata all 0.0; cost wired (`:234`-region) |
| `_oos_wf_mfv_long_short.py` | **YES — winsorize pools all window dates** (mild) | Borderline (see note) | Yes (10 bps) | Free pass (SPY cost-free, `:279-299`) | Full warmup+test `window_prices` (`:166-168`) passed whole into `generate_multifactor_long_short_signals` (`:177`); that calls `build_multifactor_signal` with `winsorize=True` default (`config/factor_bundles.py:60`); `_winsorize_series` computes clip limits via `non_null.quantile()` over the ENTIRE series, all dates pooled (`multifactor_signal.py:80-81`) |
| `_oos_wf_etf_pairs_meanrev.py` | **No** | No (cointegration re-fit per trailing window) | Yes (10 bps) | Free pass | ffill only (`:244`); `pos_lag = pos_wide.shift(1)` (`:305`); returns on lagged pos (`:309`); cost on `pos_lag.diff()` (`:314-317`); strategy `eg_coint`/OLS re-fit per window (`strategies/etf_pairs_meanrev.py:142,157`), explicit no-bfill (`:299-300`) |
| `_oos_wf_dual_momentum.py` | **No** | No | Yes (10 bps) | Free pass | `holding_prev = shift(1)` on FULL frame = carry-in fix (`:212`); lagged port_ret (`:233-238`); switch cost (`:242-247`); strategy bfill omitted (`strategies/dual_momentum.py:90-91`) |
| `_oos_wf_vol_target_overlay.py` | **No** | No | Yes (10 bps) | Free pass | carry-in `w_spy_lag = shift(1)` on full frame (`:219-221`); lagged port_ret + cost (`:244-250`); rolling std + SMA causal (`strategies/vol_target_overlay.py:101-106`); SPY cost-free (`:358`) |
| `_oos_wf_low_max_lottery.py` | **No** (most conservative) | No | Yes (10 bps) | Free pass | `ret_window` EXCLUSIVE end idx — MAX strictly before rebal date (`:292`); `pos_lag = shift(1)` (`:322`); strategy "strictly causal, PIT-safe. No bfill (E-030)" (`strategies/low_max_lottery.py:25`) |

### Note on `_oos_wf_mfv_long_short.py` (the one real defect)

The signal path is: `compute_factors` (causal trailing-window TA, `run_factor_analysis.py:296`/`build_core_ta_factors`)
→ `build_multifactor_signal` → `select_top_bottom`.

- **`select_top_bottom`** groups by `timestamp` (`multifactor_signal.py:352`) → per-date
  quantile ranking. **PIT-safe.**
- **`_zscore_crosssectional`** groups by `timestamp` (`multifactor_signal.py:125`) →
  z-score per date across symbols. **PIT-safe.**
- **`_winsorize_series`** (`multifactor_signal.py:60-86`) computes clip bounds
  `non_null.quantile(lower/upper)` over the **whole factor series across all dates**,
  with `winsorize=True` the default. Because the OOS script feeds the full warmup+test
  window in one call (`:166-177`), the clip bounds for an early test date incorporate
  factor values from LATER test dates in the same window. This is a genuine — though
  modest (1%/99% clip) — look-ahead leak.

**Direction of bias:** optimistic (a hindsight-fitted winsorize slightly stabilizes
extreme factor values). **Magnitude:** small; only the tails are clipped, and the
actual long/short SELECTION (z-score + top/bottom quantile) is fully per-date causal.

**Divergence risk:** `_oos_wf_mfv2.py` and `_oos_wf_mfv_long_short.py` are near-identical
sibling scripts, yet mfv2 slices `enriched[timestamp <= rebal_ts]` per rebalance
(`_oos_wf_mfv2.py:184`) BEFORE computing signals — so even a pooled winsorize would only
pool over PAST dates — whereas mfv_long_short computes once over the whole window. The
two scripts therefore have **different PIT guarantees despite looking parallel.** mfv2 is
safe; mfv_long_short is not.

## PIT / replay / leakage test substance

| Test file | Real invariant? | Evidence / weakness |
|---|---|---|
| `test_trend_baseline_pit_safety.py` | **Yes — strong** | Manipulates future bars ×5 and to ~0, asserts as_of signal unchanged at `< 1e-10` (`:101,131`); slice-vs-full at bar 100 (`:259`). center=True / future leak would fail. |
| `test_low_max_lottery_pit_safety.py` | **Yes** (1 soft spot) | `test_causal_no_future_leak` compares portfolio symbol sets short-vs-full panel (`:122`). Soft spot: the core assert is inside `if not ...empty:` (`:112`) — vacuous-pass guarded by preceding `assert not sigs_full.empty` (`:102-103`). |
| `test_walk_forward_no_leakage.py` | **Yes — but geometry only** | Real boundary asserts: purge gap >= purge_days (`:38`), embargo gap (`:64`), `purge_days < max_label_horizon` MUST raise (`:85`), exactly-10-folds prod regression (`:172`), warmup-fits-in-train (`:258`). **Does NOT cover signal-computation leakage** → cannot catch the winsorize defect above. |
| `test_replay_determinism.py` | **Yes — exemplary** | SHA-256 byte-equality on kernels; explicitly includes anti-no-op guards `test_hash_changes_on_input_change` ("harness broken" — `:115`) and cross-seed sanity (`:195`). Opposite of the self-verifying-log anti-pattern. |
| `test_multifactor_long_short_pit_safety.py` | **DOES NOT EXIST** | No dedicated PIT test for the one leaky script. trend_baseline, low_max, dual_momentum, vol_target, etf_pairs all have `*_pit_safety.py`; multifactor_long_short does not. This is the gap that let the winsorize leak ship unguarded. |

No no-op / trivially-passing PIT test was found. No "silent default weights satisfy
weight-sum" pattern (the vol_target weight-sum assertion at `_oos_wf_vol_target_overlay.py:199-201`
raises on `err > 1e-6` — real). No "self-verifying log-test" pattern in the audited tests.

## Verdict — do the defects INFLATE the OOS numbers?

One genuine look-ahead defect exists: `_oos_wf_mfv_long_short.py` pools the winsorize
clip bounds across all window dates (optimistic bias, small magnitude). Every other
audited script (trend_baseline, mfv2, etf_pairs, dual_momentum, vol_target, low_max) is
PIT-clean: stateless/parameter-free transforms or per-trailing-window re-fits, ffill-only
(no bfill staggered-inception leak), explicit warmup carry-in via `shift(1)` on the full
frame before slicing, and the 10 bps cost is verifiably subtracted from PnL in every case.

**Could the mfv_long_short defect have FAKED an edge?** No. The defect is an optimistic
bias, but the strategy's documented OOS outcome was a **complete failure** in long-only
mode (memory obs 607; the comparison doc concluded none are go-live ready). A mild
hindsight advantage that still produced a losing result makes the negative conclusion
*more* robust, not less — a clean run would be at least as bad. The benchmark free-pass
(SPY buy-and-hold, cost-free, in every script) biases comparisons *against* the
strategies, so the "no strategy beats SPY" conclusion is conservative.

**Net:** The negative OOS results stand. The single defect (mfv_long_short winsorize
pooling) and the missing dedicated PIT test for that strategy are correctness debt to
fix before any *positive* mfv_long_short result could be trusted, but they do not
fabricate an edge in the current findings.
