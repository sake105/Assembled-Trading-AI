# dual_momentum through the LITERAL pipeline — OOS Walk-Forward

Run date (UTC): 2026-05-31  
Data: Alpaca daily bars (split-adjusted) for SPY, VEU, BIL, AGG — research-local cache  
Overlapping range: 2016-08-22 → 2025-02-27  
WF: 252/252/252 (train/test/step), monthly rebalance, single asset @ 98% (2% cash buffer — see methodology note)  
Execution: LITERAL `run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` → feature enrichment → `generate_orders_from_targets` → `simulate_with_costs`  
Config: `enable_risk_controls=False`, `include_costs=True`  
DSR multiple-testing deflation: n_trials = 16  
Pooled-OOS bars: 1764  

**Evidence tier (binding).** This is NOT Part A's 'Literal' tier. It shares Part A's EXECUTION realism (same cycle, real cost model, order-gen, enrichment, risk-controls OFF, monthly rebalance, 252/252/252 WF) but runs on a DIFFERENT data source: Alpaca SPY/VEU/BIL/AGG, not the 75-symbol offline survivor cache (which lacks VEU & BIL — verified: `load_eod_prices(None)` returns 220 symbols incl. SPY+AGG but not VEU/BIL). The FOLD COVERAGE also differs from Part A: the Alpaca menu starts earlier (2016-08) than the offline cache, so this WF yields 7 folds spanning 2017-08-22…2024-08-27, NOT Part A's 6 folds 2019-2024. The SPY benchmark below is the Alpaca SPY buy-and-hold over THESE windows — close to but not byte-identical to Part A's offline SPY bench (AnnSharpe +0.91 / CAGR +17.4%). Read this as comparable to Part A on the EXECUTION axis only, with its own folds and SPY series.

**Methodology note — cash buffer (material).** The book deploys 98% of capital into the held asset, not 100%. This is REQUIRED, not cosmetic: the literal fill model's non-negative-cash gate (`execution/fill_model.apply_cash_gate`) rejects any BUY whose `notional + cost` would drive cash below ~0, so a 100%-notional single-asset order (`weight 1.0`, `notional == capital`) is structurally un-fillable — the position would never establish and the equity path would be a phantom (verified: at weight 1.0, fold-4 realized -8.2% with the establishing BUY rejected, while SPY did +30.6%; at weight 0.98 the same fold realizes +20.7% with zero rejected trades). A real fully-invested book likewise must hold a small cash reserve for costs/slippage, so 0.98 is the honest deploy. (Implication for Part A: any research book that is gross-100% invested loses its last-alphabetical position to the same gate each rebalance — diluted across many names, but a known small drag; single-asset dual_momentum merely exposes it in full.)

**Data honesty.** Alpaca bar close ≈ price return (no dividend reinvestment): VEU ~3% yield and AGG ~3-4% coupon are NOT captured → the defensive/ex-US legs are return-UNDERSTATED. BIL price return ≈ 0 (correct cash-hurdle proxy). Long-only single-asset rotation → no short-borrow; the literal cost model charges commission+spread+impact on each switch. PIT: the cycle slices features to ≤ as_of; `dual_momentum.compute_signals` tags EOM bars causally and forward-fills the holding, so no look-ahead. CI: not run; local one-shot.

## Result

**[REJECTED]** pooled-OOS AnnSharpe **+0.65** vs SPY +0.72 · CAGR **+7.6%** vs SPY +12.7% · IR vs SPY -0.49 (t=-1.29) · DSR-prob 0.47 (pass5%=False) · beta +0.52 · vol-matched ann.ret +11.8%.

## Per-fold (literal pipeline)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Switches | SPY CAGR | SPY Sharpe |
|------|-------------|------|--------|-------|------|----------|----------|------------|
| 1 | 2017-08-22–2018-08-22 | +3.0% | +0.89 | -4.3% | +0.02 | 3 | +16.7% | +1.31 |
| 2 | 2018-08-22–2019-08-23 | +12.4% | +1.31 | -5.7% | +0.33 | 4 | +2.1% | +0.21 |
| 3 | 2019-08-23–2020-08-22 | +5.8% | +0.37 | -28.0% | +0.64 | 3 | +19.2% | +0.72 |
| 4 | 2020-08-24–2021-08-24 | +23.5% | +1.65 | -9.0% | +0.89 | 3 | +30.6% | +1.87 |
| 5 | 2021-08-24–2022-08-24 | -7.0% | -0.45 | -14.7% | +0.59 | 2 | -8.0% | -0.30 |
| 6 | 2022-08-24–2023-08-25 | -3.9% | -0.69 | -6.6% | +0.11 | 2 | +5.6% | +0.39 |
| 7 | 2023-08-25–2024-08-27 | +23.3% | +1.95 | -8.9% | +0.86 | 3 | +27.6% | +2.02 |
| **Ø (7/7)** | — | **+8.1%** | **+0.72** | **-11.0%** | **+0.49** | **3** | +13.4% | +0.89 |

## OOS-edge (pooled out-of-sample)

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dual_momentum (literal) | +0.65 | +1.72 | +7.6% | -28.0% | +0.52 | -0.49 | -1.29 | 0.47 | N | 0.42 | +11.8% |
| **SPY (bench)** | +0.72 | +1.92 | +12.7% | -34.2% | +1.00 | — | — | 0.55 | N | — | +12.7% |

## Verdict

**dual_momentum is REJECTED through the literal pipeline.** It does not clear SPY's pooled-OOS Sharpe with a DSR-deflated AND significant (IR t>1.96) edge. Consistent with the standalone vectorized study (`docs/results/2026_05_dual_momentum_real_oos.md`: 13-fold 2016-2025 Ø CAGR 9.7% / Sharpe 0.98 vs SPY 14.5% / 1.26) — the absolute-momentum trend filter cuts drawdowns but the defensive switches drag absolute return below buy-and-hold SPY in this bull-dominated sample, and the risk-adjusted edge is not significant.

---
_Script: `scripts/_oos_wf_dual_momentum_literal.py` (research harness; executes the real `run_trading_cycle`, reads `policy.yaml` read-only, EDITS no production module, forces crisis-overlay dry-run via `ASSEMBLED_NO_CRISIS_OVERLAY=1`, caches Alpaca bars to `output/research/` — mutates NO production state or price cache)._  
_Strategy: `src/assembled_core/strategies/dual_momentum.py` (`compute_signals`)._  
_Pipeline entry: `src/assembled_core/qa/backtest_engine.py` (`run_portfolio_backtest` / `make_cycle_fn`)._  
_Edge helpers reused from `scripts/_oos_wf_leverage_short.py` (DRY)._  