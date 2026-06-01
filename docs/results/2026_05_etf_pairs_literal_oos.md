# etf_pairs_meanrev through the LITERAL pipeline — OOS Walk-Forward

Run date (UTC): 2026-05-31  
Data: Alpaca daily bars (split-adjusted) for EWA, EWC, GDX, GDXJ, IVV, KBE, SPY, VDE, VGT, XLE, XLF, XLK — research-local cache  
Pairs: SPY/IVV, GDX/GDXJ, XLE/VDE, EWA/EWC, XLF/KBE, XLK/VGT  
Overlapping range: 2016-05-02 → 2025-02-27  
WF: 252/252/252 (train/test/step), **DAILY** rebalance, FULL long-short (gross ≈ 200%, market-neutral)  
Execution: LITERAL `run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` → feature enrichment → `generate_orders_from_targets` → `simulate_with_costs`  
Config: `enable_risk_controls=False`, `include_costs=True`, `long_only=False`  
DSR multiple-testing deflation: n_trials = 16  
Pooled-OOS bars: 2016  

**Evidence tier (binding).** Same THIRD tier as the dual_momentum literal study, NOT Part A's 'Literal' tier. It shares Part A's EXECUTION realism (same cycle, real cost model, order-gen, enrichment, risk-controls OFF, 252/252/252 WF) but: (a) **DAILY** rebalance — etf_pairs is a daily Z-score strategy, monthly rebalance would miss most entries/exits; (b) **FULL long-short**, gross ≈ 200%, NET ≈ 0 (market-neutral); (c) a DIFFERENT data source — Alpaca daily bars for the 12 pair ETFs, not the offline survivor cache. The SPY benchmark is the Alpaca SPY buy-and-hold over THESE windows (8 folds spanning 2017-01-03…2025-01-08). SPY/IVV is itself a traded pair, so SPY is both benchmark and (occasionally) a traded leg; the benchmark uses SPY close independently of any held position.

**Market-neutral caveat (read before the verdict).** This is a beta≈0 book; it is NOT built to out-CAGR a bull-market SPY (a neutral book structurally trails a rising tape on absolute return). The meaningful question is risk-adjusted: does the spread alpha clear SPY's Sharpe with a DSR-deflated AND significant (IR t>1.96) edge? The falsification bar is unchanged (beat SPY Sharpe AND DSR AND IR-t → PROSPECT), but a CAGR shortfall alone is expected and is NOT the interesting signal — the Sharpe/DSR/IR line is.

**Data honesty.** Alpaca bar close ≈ price return (no dividend reinvestment). For relative-value pairs this largely CANCELS in the spread (both legs are similar instruments with near-identical yields, e.g. SPY/IVV, XLK/VGT); residual divergence (e.g. GDX/GDXJ payout differences) is a second-order spread bias. Costs: the literal cost model charges commission+spread+impact on every leg change including SHORT legs, but NO explicit short-borrow/locate fee is modelled — a known OPTIMISTIC omission for the short side (disclosed). PIT: the signal panel is precomputed once per fold via `generate_etf_pairs_signals_from_prices` and looked up by as_of — PIT-identical to a per-as_of recompute because the state machine is strictly causal (verified by an explicit self-check at run start). statsmodels REQUIRED (Engle-Granger). CI: not run; local one-shot.

## Result

**[REJECTED]** pooled-OOS AnnSharpe **-0.06** vs SPY +0.76 · CAGR **-0.6%** vs SPY +12.9% · IR vs SPY -0.77 (t=-2.18) · DSR-prob 0.02 (pass5%=False) · beta +0.06 · vol-matched ann.ret -2.7%.

## Per-fold (literal pipeline, daily rebalance)

| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | Legs | Rej | SPY CAGR | SPY Sharpe |
|------|-------------|------|--------|-------|------|-------|------|-----|----------|------------|
| 1 | 2017-01-03–2018-01-03 | -5.9% | -1.78 | -6.3% | -0.02 | 1.97 | 2.3 | 0 | +19.5% | +2.72 |
| 2 | 2018-01-03–2019-01-04 | -2.3% | -0.38 | -4.9% | -0.02 | 1.39 | 2.0 | 0 | -9.8% | -0.51 |
| 3 | 2019-01-04–2020-01-04 | +4.8% | +1.03 | -1.9% | -0.04 | 1.64 | 2.0 | 0 | +27.9% | +2.15 |
| 4 | 2020-01-06–2021-01-05 | -1.3% | -0.02 | -8.8% | +0.18 | 1.57 | 2.3 | 0 | +14.0% | +0.57 |
| 5 | 2021-01-05–2022-01-04 | +5.2% | +2.10 | -1.4% | +0.01 | 1.66 | 2.0 | 0 | +28.8% | +2.01 |
| 6 | 2022-01-04–2023-01-05 | +1.4% | +2.14 | -0.1% | -0.00 | 1.24 | 2.0 | 0 | -19.7% | -0.78 |
| 7 | 2023-01-05–2024-01-06 | -9.6% | -1.07 | -13.7% | -0.12 | 1.49 | 2.0 | 0 | +23.4% | +1.67 |
| 8 | 2024-01-08–2025-01-08 | +3.9% | +1.03 | -1.4% | -0.01 | 1.77 | 2.4 | 0 | +24.1% | +1.78 |
| **Ø (8/8)** | — | **-0.5%** | **+0.38** | **-4.8%** | **-0.00** | **1.59** | **2.1** | **0** | +13.5% | +1.20 |

_Gross = avg Σ|target_weight| per rebalance (≈2.0 confirms full long-short); Legs = avg active legs per rebalance; Rej = rejected trades (fill-model gate / short handling sanity — expect ~0 since short SELLs credit cash before long BUYs)._

_Estimator note: the **Ø (8/8)** row above is the mean of per-fold metrics (each fold equal-weighted); the pooled-OOS edge table below and the verdict use the **pooled daily return series** (all OOS bars concatenated). The two can differ in magnitude AND sign — here fold-Ø Sharpe **+0.38** vs pooled **−0.06** — expected by construction, not an inconsistency. The verdict uses the pooled series._

## OOS-edge (pooled out-of-sample)

| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | VolMatchRet |
|---|---|---|---|---|---|---|---|---|---|---|---|
| etf_pairs_meanrev (literal) | -0.06 | -0.16 | -0.6% | -13.7% | +0.06 | -0.77 | -2.18 | 0.02 | N | 0.01 | -2.7% |
| **SPY (bench)** | +0.76 | +2.15 | +12.9% | -34.2% | +1.00 | — | — | 0.63 | N | — | +12.9% |

## Verdict

**etf_pairs_meanrev is REJECTED through the literal pipeline.** It does not clear SPY's pooled-OOS Sharpe with a DSR-deflated AND significant (IR t>1.96) edge. As a market-neutral book this is unsurprising on absolute CAGR, and the risk-adjusted line does not reach significance through the real cost model (commission+spread+impact on every leg change, gross 200% → cost drag on a thin relative-value edge). The cointegration filter and Z-score discipline keep drawdowns contained and beta near zero, but the net-of-cost spread alpha is not a deflated, significant SPY-beating edge in this sample.

---
_Script: `scripts/_oos_wf_etf_pairs_literal.py` (research harness; executes the real `run_trading_cycle`, reads `policy.yaml` read-only, EDITS no production module, forces crisis-overlay dry-run via `ASSEMBLED_NO_CRISIS_OVERLAY=1`, caches Alpaca bars to `output/research/` — mutates NO production state or price cache)._  
_Strategy: `src/assembled_core/strategies/etf_pairs_meanrev.py` (`generate_etf_pairs_signals_from_prices` / `compute_signals`)._  
_Pipeline entry: `src/assembled_core/qa/backtest_engine.py` (`run_portfolio_backtest` / `make_cycle_fn`)._  
_Edge helpers reused from `scripts/_oos_wf_leverage_short.py`; sizing fn from `scripts/_oos_wf_pipeline_realistic.py` (DRY)._  