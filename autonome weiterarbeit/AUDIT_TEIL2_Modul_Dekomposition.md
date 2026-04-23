# Audit Teil 2 — Modul-für-Modul-Dekomposition

**Audit-Datum:** 2026-04-23 (Fortsetzung von Teil 1)
**Scope:** Jede einzelne Datei in `src/assembled_core/` mit Verdikt
**Methodik:** Statische Analyse via AST-Parser über alle 1436 `.py`-Files im Repo. Für jede Datei wurden ermittelt:
- Zeilenzahl
- Kategorisierung der Nutzung (siehe unten)
- Anzahl und Art der importierenden Files

## Kategorien (Usage-Typen)

Jede Datei bekommt eine dieser Klassifikationen:

- **REAL_USE** — wird von Produktions-Code (irgendwo anders außer `trading_cycle.py` und Wiring-Tests) importiert. Diese Files haben echten Impact.
- **TRADING_CYCLE_ONLY** — wird **ausschließlich** von `trading_cycle.py` importiert. Das sind die "observability-wired"-Files — sie werden geladen, aber ihr Output fließt meist nur ins `result.meta`-Dict ein, nicht in Trading-Entscheidungen.
- **OBSERVABILITY_WIRING** — wird von `trading_cycle.py` UND von Wiring-Tests (`test_waveN_wiring.py`) importiert. Dasselbe wie oben, nur mit "Wiring-Tests als Beleg".
- **WIRING_TEST_ONLY** — wird nur von Wiring-Tests importiert. Das sind die reinsten Toten.
- **TEST_ONLY** — wird nur von echten Tests (nicht Wiring-Tests) benutzt.
- **INIT_ONLY** — wird nur im eigenen `__init__.py` als Re-Export referenziert.
- **ZERO** — wird nirgendwo importiert. Totaler Waisen.
- **is_init** — die Datei IST ein `__init__.py`. Wurde nicht weiter analysiert.

## Verdikts (pro Datei)

Jede Datei bekommt einen Verdikt:

- **KEEP** — bleibt, ist aktiv genutzt und notwendig.
- **REFACTOR** — bleibt, aber der Inhalt braucht Überarbeitung (zu groß, schlechte Struktur, Bugs).
- **MERGE** — Funktionalität sollte in ein anderes File konsolidiert werden (z.B. Duplikate).
- **ARCHIVE** — nicht im Produktionspfad, aber der Code hat konzeptionellen Wert. In `archive/` verschieben, nicht löschen.
- **DELETE** — wegwerfen. Kein echter Nutzen, keine Referenz.

## Zahlen zur Einordnung

Von den **551 Python-Files in `src/assembled_core/`**:
- 330 (60%) sind REAL_USE oder is_init/INIT_ONLY — echter Produktionscode
- 6 (1%) sind ZERO
- 215 (39%) sind TRADING_CYCLE_ONLY / OBSERVABILITY_WIRING / WIRING_TEST_ONLY / TEST_ONLY — zweifelhaft

Das heißt: **knapp 40% des Codes in `src/assembled_core/` leistet keine echte Arbeit.** Das ist der Maßstab der Konsolidierungsarbeit, die vor dir liegt.

---

## Modul: `src/assembled_core/_root/`

**Top-Level-Files direkt in `src/assembled_core/`. Enthält Core-Utilities und einige Duplikate.**

- Files: 7 · Zeilen gesamt: 740
- REAL_USE: 6 · Observability-wired: 0 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **0** (0% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `costs.py` | 203 | REAL USE | **KEEP** | Cost model configuration for portfolio simulation. |
| `logging_config.py` | 180 | REAL USE | **KEEP** | Central logging configuration for Assembled Trading AI Backend. |
| `errors.py` | 149 | REAL USE | **KEEP** | Assembled Trading AI — central error classification. |
| `logging_utils.py` | 74 | REAL USE | **KEEP** | Logging utilities for CLI scripts and core modules. |
| `ema_config.py` | 58 | REAL USE | **KEEP** | EMA (Exponential Moving Average) configuration for trading strategies. |
| `config.py` | 42 | REAL USE | **KEEP** | Central configuration for the trading pipeline. |
| `__init__.py` | 34 | is init | **KEEP** | Assembled Trading AI - Core Backend Package. |


## Modul: `src/assembled_core/pipeline/`

**Das Herz des Systems. Enthält `trading_cycle.py` (10.544 Zeilen, 309 Steps) und den EOD-Orchestrator.**

- Files: 14 · Zeilen gesamt: 13,942
- REAL_USE: 7 · Observability-wired: 6 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **873** (6% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `trading_cycle.py` | 10,544 | REAL USE | **REFACTOR** | Unified Trading Cycle Orchestrator (B1). |
| `orchestrator.py` | 1,351 | REAL USE | **REFACTOR** | Pipeline orchestration for EOD runs. |
| `backtest.py` | 443 | REAL USE | **KEEP** | Backtest simulation (equity curve without costs). |
| `portfolio.py` | 334 | REAL USE | **KEEP** | Portfolio simulation with cost model. |
| `precomputed_index.py` | 231 | TRADING CYCLE ONLY | **ARCHIVE** | Precomputed index for efficient snapshot extraction in backtests. |
| `event_bus.py` | 198 | OBSERVABILITY WIRING | **ARCHIVE** | Lightweight Event Bus for Event-Driven Architecture (M23 Task 23.2). |
| `io.py` | 184 | REAL USE | **KEEP** | Input/Output utilities for price and order data. |
| `graceful_degradation.py` | 160 | OBSERVABILITY WIRING | **ARCHIVE** | Graceful degradation for missing data sources (Plan 11.2). |
| `backtest_legacy.py` | 120 | OBSERVABILITY WIRING | **DELETE** | Legacy implementations of backtest functions for regression testing. |
| `run_metadata.py` | 102 | OBSERVABILITY WIRING | **DELETE** | Run Metadata and Reproducibility (Plan 11.4). |
| `orders.py` | 100 | REAL USE | **KEEP** | Order generation from trading signals. |
| `signals.py` | 78 | REAL USE | **KEEP** | Signal generation for trading strategies. |
| `pipeline_timing.py` | 62 | OBSERVABILITY WIRING | **DELETE** | Pipeline Timing (Plan 11.5). |
| `__init__.py` | 35 | is init | **KEEP** | Pipeline modules for trading strategy execution, backtesting, and portfolio simulation. |


## Modul: `src/assembled_core/execution/`

**Fill-Simulation, Broker-Adapter, Paper-Engine, Kill-Switch, Pre-Trade-Checks.**

- Files: 31 · Zeilen gesamt: 13,698
- REAL_USE: 23 · Observability-wired: 7 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **1,598** (11% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `unified_paper_engine.py` | 2,696 | REAL USE | **REFACTOR** | Unified Paper Trading Engine. |
| `pre_trade_checks.py` | 1,261 | REAL USE | **REFACTOR** | Pre-trade checks for order validation and risk limits. |
| `transaction_costs.py` | 1,008 | REAL USE | **REFACTOR** | Transaction costs computation for fills and trades. |
| `fill_model.py` | 967 | REAL USE | **KEEP** | Fill model: Schema and contract for trades/fills with partial fill support. |
| `broker_adapter.py` | 811 | REAL USE | **KEEP** | Broker Adapter — M12: Abstract broker interface for paper and live trading. |
| `algo_execution.py` | 487 | REAL USE | **KEEP** | Algorithmic Execution: TWAP and VWAP Order Schedulers. |
| `order_generation.py` | 468 | REAL USE | **KEEP** | Order generation module. |
| `broker_execution.py` | 464 | REAL USE | **KEEP** | Broker Execution Bridge — Submit orders to broker, poll fills, convert to ledger format. |
| `paper_trading_engine.py` | 373 | REAL USE | **KEEP** | Paper trading engine for in-memory order execution simulation. |
| `smart_order_router.py` | 367 | REAL USE | **KEEP** | Smart Order Router (M20.3) — Institutional-Grade Multi-Venue Routing. |
| `risk_controls.py` | 352 | REAL USE | **KEEP** | Risk controls integration for order filtering. |
| `almgren_chriss.py` | 347 | REAL USE | **KEEP** | Almgren-Chriss Optimal Execution Model (M20.2). |
| `intent_store.py` | 345 | REAL USE | **KEEP** | Intent store for M4 execution workers — idempotency and audit trail. |
| `ibkr_adapter.py` | 342 | OBSERVABILITY WIRING | **ARCHIVE** | Interactive Brokers (IBKR) Adapter (M24 Task 24.1). |
| `paper_monitoring.py` | 338 | OBSERVABILITY WIRING | **ARCHIVE** | Paper Trading Monitor. |
| `kill_switch.py` | 326 | REAL USE | **KEEP** | Kill switch for emergency order blocking. |
| `cost_model_calibrator.py` | 269 | REAL USE | **KEEP** | Offline cost-model calibration from TCA feedback. |
| `safe_bridge.py` | 253 | REAL USE | **KEEP** | SAFE-Bridge order file generation. |
| `adaptive_algo.py` | 249 | OBSERVABILITY WIRING | **ARCHIVE** | Adaptive Execution Algorithm (M20 Task 20.6). |
| `api_resilience.py` | 232 | REAL USE | **KEEP** | API Resilience — Retry logic, rate limiting, and error handling for broker APIs. |
| `order_lifecycle.py` | 232 | REAL USE | **KEEP** | Order lifecycle tracking with full audit trail. |
| `pre_open_signals.py` | 225 | OBSERVABILITY WIRING | **ARCHIVE** | Pre-Open Signal Generation (M20 Task 20.5). |
| `position_sync.py` | 218 | REAL USE | **KEEP** | Position Sync — Reconcile ledger state against broker positions. |
| `pre_live_gate.py` | 193 | OBSERVABILITY WIRING | **ARCHIVE** | Pre-Live Gate — 8 Mandatory Checks Before Going Live (M24 Task 24.3). |
| `symbol_kill_switch.py` | 193 | REAL USE | **KEEP** | Per-symbol kill switch (Sprint 4 / Plan C27). |
| `borrow_costs.py` | 169 | REAL USE | **KEEP** | Borrow cost computation for short positions. |
| `position_alignment.py` | 165 | TRADING CYCLE ONLY | **ARCHIVE** | Position alignment utilities for order generation. |
| `fat_finger_guard.py` | 143 | REAL USE | **KEEP** | Fat-finger guard (Sprint 4 / Plan C29). |
| `portfolio_execution.py` | 86 | TRADING CYCLE ONLY | **DELETE** | Portfolio Execution Optimizer (Plan 6.10). |
| `fill_model_pipeline.py` | 74 | REAL USE | **KEEP** | Fill model pipeline: central function for applying all fill model components. |
| `__init__.py` | 45 | is init | **KEEP** | Order execution and simulation modules. |


## Modul: `src/assembled_core/signals/`

**Alle Signal-Generatoren. Fragmentiert über 22 Files, größtenteils Adapter/Bridges.**

- Files: 22 · Zeilen gesamt: 6,409
- REAL_USE: 9 · Observability-wired: 11 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **2,866** (44% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `multifactor_signal.py` | 908 | REAL USE | **KEEP** | Multi-Factor Signal Generation Module. |
| `crash_prediction.py` | 577 | REAL USE | **KEEP** | Multi-signal crash prediction engine. |
| `rules_trend.py` | 491 | REAL USE | **KEEP** | Trend-following signal rules module. |
| `meta_model.py` | 453 | REAL USE | **KEEP** | Meta-Model for predicting setup success probability. |
| `intel_signal_adapter.py` | 421 | TRADING CYCLE ONLY | **ARCHIVE** | Intel signal adapter: converts DependencySignal objects to trading signals. |
| `behavioral_finance.py` | 378 | OBSERVABILITY WIRING | **ARCHIVE** | Behavioral Finance Signals — Exploiting Cognitive Biases (M28). |
| `signal_api.py` | 334 | OBSERVABILITY WIRING | **ARCHIVE** | Signal API for standardized signal representation and validation. |
| `short_signals.py` | 328 | TRADING CYCLE ONLY | **ARCHIVE** | Short signal generator: converts crash predictions into concrete short positions. |
| `ensemble.py` | 297 | REAL USE | **KEEP** | Ensemble layer for combining rule-based signals with meta-model confidence scores. |
| `earnings_integration.py` | 291 | TRADING CYCLE ONLY | **ARCHIVE** | Earnings calendar integration for signal pipeline (V18). |
| `sector_rotation.py` | 269 | REAL USE | **KEEP** | Sector rotation signals based on ETF momentum and relative strength (M16). |
| `risk_aware_combiner.py` | 259 | OBSERVABILITY WIRING | **ARCHIVE** | Risk-Aware Signal Combiner. |
| `ml_integration.py` | 238 | OBSERVABILITY WIRING | **ARCHIVE** | Integration der neuen ML-Stacks (Regime-Router, Nested-Meta, BMA) in den Signal-Layer. |
| `news_signal_bridge.py` | 230 | TRADING CYCLE ONLY | **ARCHIVE** | Part B deeper wiring: news → signal score bridge. |
| `signal_confidence.py` | 188 | REAL USE | **KEEP** | Bayesian Signal Confidence estimation (Plan 1.9). |
| `signal_diagnostics.py` | 174 | TRADING CYCLE ONLY | **ARCHIVE** | Signal Diagnostics and Real-Time Monitoring (Plan 1.10). |
| `rules_event_insider_shipping.py` | 166 | REAL USE | **KEEP** | Event-based signal rules module (Phase 6). |
| `mean_reversion.py` | 138 | OBSERVABILITY WIRING | **DELETE** | Mean-Reversion Signal Layer (Plan 1.8). |
| `hmm_posterior.py` | 108 | REAL USE | **KEEP** | F2 — Regime posterior + EWMA smoothing (Plan v3 Part F2). |
| `plugin_loader.py` | 75 | OBSERVABILITY WIRING | **DELETE** | Signal Plugin System (Plan 11.8). |
| `__init__.py` | 71 | is init | **KEEP** | Signal generation modules. |
| `__init__.py` | 15 | is init | **KEEP** | Regime-inference subpackage. |


## Modul: `src/assembled_core/strategies/`

**Strategie-Definitionen. Overlapping mit `signals/`. Multifactor V1/V2 parallel, Stat-Arb doppelt.**

- Files: 14 · Zeilen gesamt: 4,698
- REAL_USE: 7 · Observability-wired: 5 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **1,319** (28% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `multifactor_v2.py` | 1,034 | REAL USE | **REFACTOR** | STRATEGY-V2: 30-factor multi-factor strategy with regime-conditional weights. |
| `multifactor_v1.py` | 739 | REAL USE | **KEEP** | STRATEGY-V1: Production multi-factor strategy. |
| `multifactor_long_short.py` | 583 | REAL USE | **KEEP** | Multi-Factor Long/Short Strategy Module. |
| `stat_arb.py` | 366 | OBSERVABILITY WIRING | **ARCHIVE** | Statistical Arbitrage — Pairs Trading Strategy (M36). |
| `strategy_discovery.py` | 300 | OBSERVABILITY WIRING | **ARCHIVE** | Strategy Discovery Engine (M34). |
| `ema_trend_v0.py` | 260 | REAL USE | **KEEP** | BENCH-0: EOD benchmark strategy — EMA20/EMA60 with score-based sizing and exit signals. |
| `cointegration.py` | 245 | OBSERVABILITY WIRING | **ARCHIVE** | Cointegration Engine for Pairs Trading (M36.1). |
| `pair_signals.py` | 229 | OBSERVABILITY WIRING | **ARCHIVE** | Mean-Reversion Signal Generator for Pairs Trading (M36.2). |
| `base.py` | 224 | REAL USE | **KEEP** | Strategy Base Protocol and Registry (M21.1). |
| `__init__.py` | 215 | is init | **KEEP** | Statistical Arbitrage & Pairs Trading strategy modules. |
| `pca_arb.py` | 179 | OBSERVABILITY WIRING | **ARCHIVE** | PCA-Based Statistical Arbitrage (M36.3). |
| `ic_decay_weights.py` | 124 | REAL USE | **KEEP** | F1 — IC-decay-weighted factor combination (Plan v3 Part F1). |
| `signal_decay_gate.py` | 120 | REAL USE | **KEEP** | D5 — Signal-decay read-path for the multi-factor combiner. |
| `__init__.py` | 80 | is init | **KEEP** | Strategy modules for Assembled Trading AI Backend. |


## Modul: `src/assembled_core/qa/`

**Quality Assurance, Metriken, Backtest-Engine, Walk-Forward, Scenarios. 50 Files.**

- Files: 50 · Zeilen gesamt: 22,562
- REAL_USE: 34 · Observability-wired: 14 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **4,578** (20% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `factor_analysis.py` | 2,346 | REAL USE | **REFACTOR** | Factor Analysis and Information Coefficient (IC) Engine. |
| `backtest_engine.py` | 1,434 | REAL USE | **REFACTOR** | Portfolio-level backtest engine. |
| `robustness.py` | 1,403 | OBSERVABILITY WIRING | **ARCHIVE** | Robustness analysis for strategy validation (Sprint 12 RB2). |
| `walk_forward.py` | 1,248 | REAL USE | **REFACTOR** | Walk-Forward Analysis for Out-of-Sample Strategy Validation (B3). |
| `metrics.py` | 1,148 | REAL USE | **REFACTOR** | Performance metrics computation. |
| `scenario_engine.py` | 1,124 | REAL USE | **REFACTOR** | Scenario engine for risk analysis. |
| `data_qc.py` | 782 | REAL USE | **KEEP** | Data Quality Control (QC) Module. |
| `labeling.py` | 759 | REAL USE | **KEEP** | Trade and equity curve labeling for machine learning. |
| `qa_gates.py` | 657 | REAL USE | **KEEP** | QA gates for performance metrics evaluation. |
| `dataset_builder.py` | 586 | REAL USE | **KEEP** | ML Dataset Builder for Trading Features and Labels. |
| `validation.py` | 530 | REAL USE | **KEEP** | Model validation engine for performance, overfitting, and data quality checks. |
| `portfolio_analyzer.py` | 494 | REAL USE | **KEEP** | Portfolio Analyzer — comprehensive portfolio performance and risk analytics. |
| `post_trade_analyzer.py` | 446 | REAL USE | **KEEP** | Post-Trade Analyzer — M11: Post-Trade Learning Loop. |
| `drift_detection.py` | 441 | REAL USE | **KEEP** | Drift detection for features, labels, and performance. |
| `event_study.py` | 439 | REAL USE | **KEEP** | Event Study Engine for analyzing price reactions to events. |
| `e2e_integration.py` | 431 | OBSERVABILITY WIRING | **ARCHIVE** | End-to-End Integration Testing Framework (M32 Task 32.2). |
| `tca.py` | 411 | REAL USE | **KEEP** | Transaction Cost Analysis (TCA) Reporting. |
| `reverse_stress.py` | 407 | OBSERVABILITY WIRING | **ARCHIVE** | Reverse Stress Testing — Find scenarios that cause a target loss. |
| `experiment_tracking.py` | 394 | REAL USE | **KEEP** | Experiment tracking module for structured logging of research experiments. |
| `health.py` | 386 | REAL USE | **KEEP** | Health check and QA functions for pipeline outputs. |
| `signal_decay.py` | 382 | REAL USE | **KEEP** | Signal decay and alpha half-life analysis (V6). |
| `factor_report.py` | 339 | OBSERVABILITY WIRING | **ARCHIVE** | Factor Report Workflow Module. |
| `learning_store.py` | 329 | REAL USE | **KEEP** | Learning Store — M11: Append-only JSONL store for post-trade learning records. |
| `factor_ranking.py` | 313 | REAL USE | **KEEP** | Factor Ranking Module. |
| `monte_carlo.py` | 308 | REAL USE | **KEEP** | Monte Carlo / Bootstrap simulation for backtest confidence analysis. |
| `ab_testing.py` | 289 | OBSERVABILITY WIRING | **ARCHIVE** | A/B Testing Framework for Strategy Variants (M40 Task 40.2). |
| `performance_attribution.py` | 272 | REAL USE | **KEEP** | Performance Attribution — Faktor-Contribution-Dekomposition. |
| `adversarial_testing.py` | 269 | OBSERVABILITY WIRING | **ARCHIVE** | Adversarial Robustness Testing (M25 Task 25.5). |
| `numba_kernels.py` | 254 | REAL USE | **KEEP** | Optional Numba-accelerated kernels for backtest loops. |
| `__init__.py` | 243 | is init | **KEEP** | QA and health check modules for the trading pipeline. |
| `tca_arrival.py` | 243 | OBSERVABILITY WIRING | **ARCHIVE** | TCA: Implementation Shortfall vs Arrival Price (Sprint 2 / C11). |
| `backtest_comparison.py` | 240 | REAL USE | **KEEP** | Backtest-Comparison-Framework für Multi-Strategy-Vergleich. |
| `trade_tca.py` | 223 | REAL USE | **KEEP** | Trade-Level Transaction Cost Analysis (TCA). |
| `shipping_risk.py` | 222 | REAL USE | **KEEP** | Shipping and systemic risk analysis module. |
| `ml_evaluation.py` | 220 | REAL USE | **KEEP** | ML Model Evaluation Routines. |
| `benchmark_metrics.py` | 219 | REAL USE | **KEEP** | Benchmark-relative performance attribution (V7). |
| `deflated_sharpe.py` | 218 | REAL USE | **KEEP** | E4 — Deflated Sharpe Ratio (Bailey & López de Prado, 2014). |
| `regime_aware_wf.py` | 217 | OBSERVABILITY WIRING | **ARCHIVE** | Regime-aware walk-forward validation (V8). |
| `point_in_time_checks.py` | 216 | REAL USE | **KEEP** | Point-in-Time (PIT) safety checks for Alt-Data features. |
| `scenario_simulator.py` | 210 | REAL USE | **KEEP** | Monte-Carlo Scenario Stress-Testing für Portfolios. |
| `backtest_overfit.py` | 207 | OBSERVABILITY WIRING | **ARCHIVE** | Probability of Backtest Overfitting (PBO) — Bailey & Lopez de Prado. |
| `multiple_testing.py` | 179 | OBSERVABILITY WIRING | **ARCHIVE** | Multiple Testing Corrections for Factor Screening (M16.3). |
| `drawdown_decomposition.py` | 178 | REAL USE | **KEEP** | Drawdown-Decomposition — zerlegt Drawdown-Perioden nach Faktor-Contribution. |
| `risk_metrics.py` | 170 | REAL USE | **KEEP** | Portfolio risk metrics computation. |
| `altdata_leakage.py` | 167 | OBSERVABILITY WIRING | **ARCHIVE** | Leakage test helpers for alt-data features (PIT-safe validation). |
| `candidate_gate.py` | 163 | OBSERVABILITY WIRING | **ARCHIVE** | Candidate gate for strategy validation (Sprint 12 Final + Sprint 13). |
| `parallel_grid.py` | 141 | OBSERVABILITY WIRING | **DELETE** | C3 — Parallel parameter-grid runner for QA / research. |
| `backtest_engine_numba.py` | 126 | REAL USE | **KEEP** | Numba-accelerated functions for backtest engine performance optimization. |
| `capacity.py` | 123 | OBSERVABILITY WIRING | **DELETE** | Strategy Capacity Estimation (M16.5). |
| `__init__.py` | 16 | is init | **KEEP** | Leakage tests for PIT-safe feature validation. |


## Modul: `src/assembled_core/features/`

**Feature-Engineering. Von TA bis Alt-Data (Satellite, Patent, Supply-Chain ohne Ingest).**

- Files: 44 · Zeilen gesamt: 12,045
- REAL_USE: 24 · Observability-wired: 18 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **3,996** (33% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `ta_features.py` | 830 | REAL USE | **KEEP** | Technical analysis features module. |
| `altdata_earnings_insider_factors.py` | 714 | REAL USE | **KEEP** | Alt-Data Factors: Earnings and Insider Activity. |
| `altdata_news_macro_factors.py` | 702 | REAL USE | **KEEP** | Alt-Data Factors: News Sentiment and Macro Regime Indicators. |
| `market_breadth.py` | 665 | REAL USE | **KEEP** | Market Breadth and Risk-On/Risk-Off Indicators module. |
| `news_features.py` | 552 | OBSERVABILITY WIRING | **ARCHIVE** | News and sentiment features module (V13). |
| `ta_liquidity_vol_factors.py` | 512 | REAL USE | **KEEP** | Liquidity and Volatility Factors module. |
| `event_features_vectorized.py` | 440 | REAL USE | **KEEP** | Vectorized event feature builder module (Sprint 11.E1). |
| `registry.py` | 428 | OBSERVABILITY WIRING | **ARCHIVE** | Feature Registry (Sprint 5 / F2). |
| `supply_chain_features.py` | 358 | REAL USE | **KEEP** | Supply-chain and geopolitical network features per asset. |
| `ta_factors_core.py` | 308 | REAL USE | **KEEP** | Core TA/Price Factors module. |
| `correlation_features.py` | 305 | REAL USE | **KEEP** | Cross-asset correlation and dispersion features. |
| `event_features.py` | 303 | REAL USE | **KEEP** | Minimal event feature builder module (B2 Reference Implementation, Sprint 10.B). |
| `fundamental_factors.py` | 299 | OBSERVABILITY WIRING | **ARCHIVE** | Fundamental Factor Features (M17). |
| `ta_candlestick.py` | 290 | REAL USE | **KEEP** | Candlestick Pattern Recognition for OHLC Price Data. |
| `intermarket_factors.py` | 268 | REAL USE | **KEEP** | Intermarket Cross-Asset Factors. |
| `institutional_features.py` | 254 | OBSERVABILITY WIRING | **ARCHIVE** | Institutional Holdings Features (M26 Task 26.3). |
| `intraday_features.py` | 253 | OBSERVABILITY WIRING | **ARCHIVE** | Intraday Features for Alpha Generation (M20 Task 20.4). |
| `earnings_insider_wrapper.py` | 249 | REAL USE | **KEEP** | Cross-sectional Earnings Surprise and Insider Activity factors (B2.3). |
| `cross_asset_leads.py` | 244 | OBSERVABILITY WIRING | **ARCHIVE** | Cross-Asset Lead-Lag Signals (M26 Task 26.1). |
| `news_macro_wrapper.py` | 243 | REAL USE | **KEEP** | Cross-sectional News Sentiment and Macro Regime factors (B2.4). |
| `behavioral_features.py` | 232 | OBSERVABILITY WIRING | **ARCHIVE** | Behavioral Finance Factors (M41). |
| `vpin.py` | 224 | OBSERVABILITY WIRING | **ARCHIVE** | Volume-Synchronized Probability of Informed Trading (M26 Task 26.4). |
| `__init__.py` | 217 | is init | **KEEP** | Technical analysis features and feature engineering modules. |
| `volatility_features.py` | 216 | OBSERVABILITY WIRING | **ARCHIVE** | GARCH-based volatility features for factor models. |
| `options_derived_signals.py` | 209 | REAL USE | **KEEP** | Options-Derived Regime Signals — VIX Term Structure and Put/Call Ratio. |
| `congress_features.py` | 204 | REAL USE | **KEEP** | Congressional trading features module (Phase 6 Skeleton). |
| `buyback_features.py` | 201 | OBSERVABILITY WIRING | **ARCHIVE** | Buyback Announcement Alpha (M18 Task 18.5). |
| `fractional_diff.py` | 193 | OBSERVABILITY WIRING | **ARCHIVE** | Fractional Differentiation (Lopez de Prado, AIFML Chapter 5). |
| `factor_store_integration.py` | 179 | REAL USE | **KEEP** | Factor store integration for feature building. |
| `incremental_updates.py` | 174 | REAL USE | **KEEP** | Incremental Feature Updates (Sprint 5 / F3). |
| `mean_reversion_factors.py` | 174 | REAL USE | **KEEP** | Mean-reversion factor sidecar for multifactor_v2. |
| `geopolitical_features.py` | 170 | REAL USE | **KEEP** | Geopolitical Risk (GPR) features for factor models. |
| `insider_features.py` | 170 | REAL USE | **KEEP** | Insider trading features module (Phase 6 Skeleton). |
| `macro_features.py` | 161 | REAL USE | **KEEP** | Macro feature builder module (Sprint 11.E3). |
| `cross_sectional.py` | 159 | TEST ONLY | **ARCHIVE** | Cross-Sectional Feature Normalisierung. |
| `short_interest_features.py` | 159 | OBSERVABILITY WIRING | **ARCHIVE** | Short Interest Features — signals from FINRA short data. |
| `seasonal_features.py` | 155 | OBSERVABILITY WIRING | **ARCHIVE** | Seasonal and Calendar Effect Features. |
| `index_rebal_features.py` | 137 | OBSERVABILITY WIRING | **DELETE** | Index Rebalancing Front-Running Features. |
| `shipping_features.py` | 104 | REAL USE | **KEEP** | Shipping routes features module (Phase 6 Skeleton). |
| `interaction_features.py` | 102 | OBSERVABILITY WIRING | **DELETE** | Feature interaction terms (V10). |
| `weekly_alignment.py` | 100 | REAL USE | **KEEP** | F3 — Multi-timeframe (weekly) alignment filter (Plan v3 Part F3). |
| `feature_flag_audit.py` | 80 | OBSERVABILITY WIRING | **DELETE** | Feature Flag Audit (Plan 11.6). |
| `disclosure_features.py` | 63 | OBSERVABILITY WIRING | **DELETE** | Disclosure Text Complexity Features (Plan 3.10). |
| `satellite_proxy_features.py` | 45 | OBSERVABILITY WIRING | **DELETE** | Satellite Proxy Features (Plan 3.9 / 10.8). |


## Modul: `src/assembled_core/portfolio/`

**Position-Sizing, Kelly, HRP, Black-Litterman, Risk-Parity.**

- Files: 23 · Zeilen gesamt: 5,985
- REAL_USE: 10 · Observability-wired: 12 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **2,867** (47% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `position_sizing.py` | 780 | REAL USE | **KEEP** | Position sizing module. |
| `black_litterman.py` | 391 | REAL USE | **KEEP** | Black-Litterman Portfolio Optimizer. |
| `bl_sizing.py` | 386 | TRADING CYCLE ONLY | **ARCHIVE** | Black-Litterman sizing wrapper (Sprint 3 / Plan W11). |
| `robust_optimizer.py` | 315 | OBSERVABILITY WIRING | **ARCHIVE** | Robust Portfolio Optimization under Parameter Uncertainty. |
| `hierarchical_risk_parity.py` | 308 | REAL USE | **KEEP** | Hierarchical Risk Parity (HRP) — Lopez de Prado 2016. |
| `multi_period.py` | 305 | OBSERVABILITY WIRING | **ARCHIVE** | Multi-Period Portfolio Optimization. |
| `strategy_allocator.py` | 299 | OBSERVABILITY WIRING | **ARCHIVE** | Strategy Allocator — Multi-Strategy Ensemble Framework (M21.2). |
| `hrp_sizing.py` | 289 | TRADING CYCLE ONLY | **ARCHIVE** | HRP-based sizing wrapper (Sprint 3 / Plan W10). |
| `market_neutral_optimizer.py` | 276 | REAL USE | **KEEP** | Market-neutral portfolio construction (V17). |
| `risk_budgeting.py` | 270 | TRADING CYCLE ONLY | **ARCHIVE** | Risk Budgeting and Equal Risk Contribution (ERC) Portfolio. |
| `inverse_etf_selector.py` | 264 | REAL USE | **KEEP** | Inverse ETF selector: picks the optimal short instrument per sector. |
| `cost_aware_optimizer.py` | 256 | REAL USE | **KEEP** | Cost-aware portfolio optimizer with turnover penalty (V9). |
| `stress_test_constraints.py` | 249 | REAL USE | **KEEP** | Portfolio-level stress testing constraints for optimizer (V19). |
| `covariance.py` | 241 | TRADING CYCLE ONLY | **ARCHIVE** | Covariance Matrix Estimation Utilities. |
| `long_short_balance.py` | 233 | OBSERVABILITY WIRING | **ARCHIVE** | Long-short portfolio balance manager. |
| `barbell_strategy.py` | 216 | TRADING CYCLE ONLY | **ARCHIVE** | Barbell Strategy (Taleb) — convex portfolio in high tail-risk environments. |
| `multiasset_allocator.py` | 211 | REAL USE | **KEEP** | Regime-adaptive multi-asset allocator (M16). |
| `turnover_penalty.py` | 177 | REAL USE | **KEEP** | Turnover-Penalty-Wrapper für Positions-Smoothing. |
| `cost_aware_wrapper.py` | 144 | OBSERVABILITY WIRING | **DELETE** | Cost-aware sizing wrapper (Sprint 3 / Plan W12). |
| `kelly_uncertainty.py` | 127 | REAL USE | **KEEP** | Kelly Criterion mit Uncertainty-Penalty. |
| `mvo_optimizer.py` | 86 | TRADING CYCLE ONLY | **DELETE** | MVO with Cardinality Constraints (Plan 5.10). |
| `regime_portfolio.py` | 83 | OBSERVABILITY WIRING | **DELETE** | Regime-Conditional Portfolio Templates (Plan 5.9). |
| `__init__.py` | 79 | is init | **KEEP** | Portfolio management modules. |


## Modul: `src/assembled_core/risk/`

**Risk-Management-Module. 36 Files, 14 davon observability-only.**

- Files: 36 · Zeilen gesamt: 10,617
- REAL_USE: 14 · Observability-wired: 21 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **4,693** (44% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `risk_metrics.py` | 1,237 | REAL USE | **REFACTOR** | Advanced Risk Metrics and Attribution Module. |
| `regime_models.py` | 719 | REAL USE | **KEEP** | Regime Detection and State Classification Module. |
| `state_machine.py` | 609 | REAL USE | **KEEP** | Risk state machine: WATCH / ACTIVE / COOLDOWN / PAUSE. |
| `regime_analysis.py` | 505 | REAL USE | **KEEP** | Extended Regime Analysis for Performance Evaluation (B3). |
| `transaction_costs.py` | 483 | REAL USE | **KEEP** | Transaction Cost Analysis (TCA) Module. |
| `short_risk.py` | 418 | TRADING CYCLE ONLY | **ARCHIVE** | Short position risk controls. |
| `param_stability.py` | 414 | OBSERVABILITY WIRING | **ARCHIVE** | Parameter stability checks for risk controls. |
| `factor_risk_model.py` | 377 | TRADING CYCLE ONLY | **ARCHIVE** | Barra-Style Multi-Factor Risk Model. |
| `factor_exposures.py` | 349 | REAL USE | **KEEP** | Factor Exposure Analysis Module. |
| `var_methods.py` | 334 | REAL USE | **KEEP** | Portfolio Value-at-Risk methods (C5a + C5b). |
| `correlation_guard.py` | 313 | TRADING CYCLE ONLY | **ARCHIVE** | Correlation / cluster guard: detect and scale down concentrated correlated clusters. |
| `group_exposures.py` | 296 | REAL USE | **KEEP** | Group Exposure Calculation: Sector/Region/Currency aggregation (Sprint 9). |
| `stressed_var.py` | 290 | OBSERVABILITY WIRING | **ARCHIVE** | Stressed VaR and RMT Covariance Cleaning (M25 Tasks 25.6 + 25.7). |
| `tail_hedging.py` | 289 | TRADING CYCLE ONLY | **ARCHIVE** | Tail Hedging — Portfolio Insurance and Tail Risk Management (M26). |
| `attribution.py` | 287 | TRADING CYCLE ONLY | **ARCHIVE** | Portfolio return and volatility attribution: per-symbol contribution analysis. |
| `trailing_stops.py` | 269 | REAL USE | **KEEP** | Per-position trailing stops with regime-adaptive ATR multipliers (V16 + M16 E3/E4). |
| `tail_hedge.py` | 265 | OBSERVABILITY WIRING | **ARCHIVE** | Tail Risk Hedging — Collar Strategy and Put Spread Overlay (M38 Task 38.1). |
| `intraday_monitor.py` | 252 | OBSERVABILITY WIRING | **ARCHIVE** | Intraday Risk Monitoring (M23 Task 23.5). |
| `exposure_engine.py` | 246 | REAL USE | **KEEP** | Exposure Engine: Compute target portfolio state and exposure metrics. |
| `circuit_breaker.py` | 240 | TRADING CYCLE ONLY | **ARCHIVE** | Circuit breaker for flash-crash detection and automatic trading halts. |
| `evt_tail_var.py` | 235 | TRADING CYCLE ONLY | **ARCHIVE** | EVT Peaks-Over-Threshold tail VaR (C9 — diagnostic sidecar). |
| `liquidity_scoring.py` | 219 | REAL USE | **KEEP** | Liquidity scoring and liquidity-adjusted position sizing (V15). |
| `crowding_detector.py` | 214 | REAL USE | **KEEP** | Crowded-trade and factor-concentration detection (V20). |
| `regime_costs.py` | 211 | OBSERVABILITY WIRING | **ARCHIVE** | Regime-Conditional Transaction Cost Model (M39 Task 39.2). |
| `turnover_budget.py` | 204 | TRADING CYCLE ONLY | **ARCHIVE** | Turnover budget gate: cap realized turnover per run (daily/weekly). |
| `vol_targeting.py` | 172 | TRADING CYCLE ONLY | **ARCHIVE** | Portfolio-level volatility targeting: scale exposure to hit target annualized vol. |
| `georisk_overlay.py` | 168 | REAL USE | **KEEP** | — |
| `profit_targets.py` | 156 | REAL USE | **KEEP** | Tiered profit targets and partial exit logic (M16 E2). |
| `zombie_killer.py` | 155 | TRADING CYCLE ONLY | **ARCHIVE** | Time stop / zombie killer: flag positions held too long with insufficient gain. |
| `tail_dependence.py` | 139 | OBSERVABILITY WIRING | **DELETE** | Empirical tail-dependence diagnostic (C8 sidecar — pre-wiring). |
| `market_stress.py` | 136 | TRADING CYCLE ONLY | **DELETE** | Market stress signal (price-based, no external APIs). |
| `__init__.py` | 120 | is init | **KEEP** | Risk Management and Regime Detection Module. |
| `disclosures_confirm.py` | 105 | TRADING CYCLE ONLY | **DELETE** | Disclosures confirmation boost for NEWS geo_confidence (DISCL-4.2). |
| `profit_lock.py` | 85 | TRADING CYCLE ONLY | **DELETE** | Soft Profit Lock overlay: reduce exposure after strong gains (policy-driven). |
| `antifragility.py` | 60 | OBSERVABILITY WIRING | **DELETE** | Antifragility Score (Plan 7.10). |
| `systemic_risk.py` | 46 | OBSERVABILITY WIRING | **DELETE** | Systemic Risk via Network Analysis (Plan 7.9). |


## Modul: `src/assembled_core/ml/`

**Machine Learning. 55 Files, 26 observability-only. Der größte Friedhof.**

- Files: 55 · Zeilen gesamt: 18,538
- REAL_USE: 28 · Observability-wired: 26 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **7,584** (40% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `feedback_loop.py` | 1,667 | REAL USE | **REFACTOR** | Self-learning feedback loop controller for Assembled-Trading-AI. |
| `factor_models.py` | 1,263 | REAL USE | **REFACTOR** | Machine Learning Models for Factor-Based Return Prediction. |
| `stacking.py` | 569 | REAL USE | **KEEP** | Two-Level Ensemble Stacking for ML Factor Models. |
| `regime_weight_trainer.py` | 564 | OBSERVABILITY WIRING | **ARCHIVE** | Regime-conditional factor weight training -- library module. |
| `explainability.py` | 559 | REAL USE | **KEEP** | Model Explainability & Feature Importance for ML Factor Models. |
| `retraining_scheduler.py` | 553 | REAL USE | **KEEP** | Retraining Scheduler for Assembled-Trading-AI (Phase 8 autonomous improvement). |
| `causal_inference.py` | 545 | OBSERVABILITY WIRING | **ARCHIVE** | Causal Inference for Factor–Return Relationships (M29). |
| `feature_selection.py` | 467 | REAL USE | **KEEP** | Feature selection pipeline with stability filtering. |
| `rl_execution.py` | 420 | OBSERVABILITY WIRING | **ARCHIVE** | Reinforcement Learning for Optimal Execution (M31). |
| `regime_hmm.py` | 416 | REAL USE | **KEEP** | Hidden Markov Model for Regime Forecasting. |
| `factor_timing.py` | 413 | OBSERVABILITY WIRING | **ARCHIVE** | Factor Timing — Dynamic Factor Weight Adjustment (M25). |
| `news_ml_bridge.py` | 402 | REAL USE | **KEEP** | News → ML Feature Bridge. |
| `automl.py` | 393 | OBSERVABILITY WIRING | **ARCHIVE** | AutoML — Automated Model Selection and Feature Engineering (M32). |
| `graph_models.py` | 387 | OBSERVABILITY WIRING | **ARCHIVE** | Graph-Based Models — Cross-Asset Signal Propagation (M30). |
| `hyperopt.py` | 384 | REAL USE | **KEEP** | Hyperparameter Optimization for ML Factor Models using Optuna. |
| `garch_models.py` | 340 | REAL USE | **KEEP** | GARCH family: conditional volatility modelling. |
| `symbolic_regression.py` | 329 | OBSERVABILITY WIRING | **ARCHIVE** | Symbolic Regression for Alpha Formula Discovery (M19 Task 19.6). |
| `rl_portfolio.py` | 305 | OBSERVABILITY WIRING | **ARCHIVE** | Reinforcement Learning for Portfolio Optimization (M25 Task 25.1). |
| `nlp_sentiment.py` | 300 | REAL USE | **KEEP** | NLP Sentiment Analysis using FinBERT for financial news texts. |
| `experiment_tracking.py` | 296 | OBSERVABILITY WIRING | **ARCHIVE** | Experiment Tracking — Local MLflow-style experiment logger (M40). |
| `gaussian_process.py` | 292 | OBSERVABILITY WIRING | **ARCHIVE** | Gaussian Process Regression for Factor Return Prediction (M19b). |
| `nested_meta_labeling.py` | 290 | OBSERVABILITY WIRING | **ARCHIVE** | Nested Meta-Labeling: Zwei-Stufen-Klassifikation. |
| `gnn_stocks.py` | 289 | OBSERVABILITY WIRING | **ARCHIVE** | Graph Neural Network for Stock Relationships (M19 Task 19.5). |
| `conformal_prediction.py` | 288 | OBSERVABILITY WIRING | **ARCHIVE** | Conformal Prediction for Valid Prediction Intervals (M19a). |
| `__init__.py` | 286 | is init | **KEEP** | Machine Learning modules for the Assembled-Trading-AI system. |
| `temporal_attention.py` | 286 | OBSERVABILITY WIRING | **ARCHIVE** | Temporal Attention Model for Factor Prediction (M19 Task 19.4). |
| `copula_models.py` | 281 | REAL USE | **KEEP** | Copula models for tail dependence analysis. |
| `feature_importance_tracker.py` | 274 | OBSERVABILITY WIRING | **ARCHIVE** | Rolling Feature-Importance Tracking + Auto-Pruning. |
| `online_hpo.py` | 270 | REAL USE | **KEEP** | Online Hyperparameter Adaptation via Thompson Sampling (Multi-Armed Bandit). |
| `maml.py` | 264 | OBSERVABILITY WIRING | **ARCHIVE** | Model-Agnostic Meta-Learning (MAML) for Regime Adaptation (M25 Task 25.2). |
| `tda_regime.py` | 262 | OBSERVABILITY WIRING | **ARCHIVE** | Topological Data Analysis for Regime Detection (M25 Task 25.3). |
| `feature_clustering.py` | 257 | REAL USE | **KEEP** | Feature-Clustering für Multikollinearitäts-Reduktion + ClusteredMDA. |
| `regime_model_router.py` | 254 | OBSERVABILITY WIRING | **ARCHIVE** | Regime-bedingter Modell-Router. |
| `stacking_ensemble.py` | 250 | REAL USE | **KEEP** | Stacking Ensemble für Multi-Model-Blending (Level-2-Meta-Learner). |
| `model_registry.py` | 249 | REAL USE | **KEEP** | Versioned Model Registry mit Metadaten-Tracking. |
| `online_learning.py` | 248 | REAL USE | **KEEP** | Online learning and incremental model updates (Plan 2.7). |
| `purged_cv.py` | 248 | REAL USE | **KEEP** | Purged and Embargoing Cross-Validation for Financial Time Series. |
| `model_monitoring.py` | 247 | REAL USE | **KEEP** | Feature Drift Detection and Model Monitoring (Plan 2.10). |
| `signal_decay_tracker.py` | 243 | REAL USE | **KEEP** | Signal Decay Tracking — misst IC-Halbwertszeit von Signalen. |
| `meta_labeling.py` | 234 | REAL USE | **KEEP** | Meta-Labeling für Signal-Filterung (Lopez de Prado, AIFML Chapter 3). |
| `bayesian_nn.py` | 216 | OBSERVABILITY WIRING | **ARCHIVE** | Bayesian Neural Network via MC Dropout. |
| `calibration.py` | 209 | OBSERVABILITY WIRING | **ARCHIVE** | Probability Calibration for ML Predictions. |
| `triple_barrier.py` | 199 | REAL USE | **KEEP** | Triple-Barrier Labeling (Lopez de Prado, AIFML Chapter 3). |
| `evt_models.py` | 198 | REAL USE | **KEEP** | Extreme Value Theory (EVT): Peaks-Over-Threshold for tail risk. |
| `quantile_models.py` | 197 | TRADING CYCLE ONLY | **ARCHIVE** | Quantile Regression for prediction intervals (Plan 2.4). |
| `calibration_monitor.py` | 191 | OBSERVABILITY WIRING | **ARCHIVE** | Probability/Prediction Calibration Monitoring. |
| `online_hmm_regime.py` | 185 | OBSERVABILITY WIRING | **ARCHIVE** | Online HMM Regime Detection auf Returns/Vol. |
| `cpcv.py` | 180 | REAL USE | **KEEP** | Combinatorial Purged Cross-Validation (V12). |
| `conformal.py` | 178 | REAL USE | **KEEP** | Conformal Prediction für kalibrierte Unsicherheitsintervalle. |
| `bayesian_ensemble.py` | 177 | OBSERVABILITY WIRING | **ARCHIVE** | Bayesian Model Averaging (BMA) über mehrere ML-Modelle. |
| `lime_explainer.py` | 175 | REAL USE | **KEEP** | LIME (Local Interpretable Model-Agnostic Explanations) Wrapper. |
| `adversarial_validation.py` | 173 | REAL USE | **KEEP** | Adversarial Validation: Detektiert Distribution-Shift zwischen Train und Test. |
| `online_gradient_boosting.py` | 139 | OBSERVABILITY WIRING | **DELETE** | Online Gradient Boosting / Adaptive Tree für nichtlineare Online-Learning. |
| `combined_regime.py` | 119 | OBSERVABILITY WIRING | **DELETE** | Combined Regime Classifier — Ensemble aus News-Sentiment und HMM-Returns. |
| `signal_correlation.py` | 118 | REAL USE | **KEEP** | Signal-Correlation-Analyzer — erkennt redundante Signale. |


## Modul: `src/assembled_core/data/`

**Datenquellen, Ingest, Feature-Stores. 57 Files mit starker Observability-Schicht.**

- Files: 57 · Zeilen gesamt: 9,954
- REAL_USE: 24 · Observability-wired: 26 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **4,185** (42% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `corporate_actions.py` | 498 | REAL USE | **KEEP** | Corporate actions: splits, dividends, total return adjustment. |
| `factor_store.py` | 496 | REAL USE | **KEEP** | Factor store for caching computed factors (Parquet-based). |
| `ledger_store.py` | 407 | REAL USE | **KEEP** | SQLite-backed Paper Trading Ledger. |
| `prices_ingest.py` | 391 | REAL USE | **KEEP** | Price data ingestion module. |
| `cost_model_policy.py` | 326 | REAL USE | **KEEP** | Policy-driven cost model: estimate rebalancing cost for a weight change. |
| `universe.py` | 310 | REAL USE | **KEEP** | Universe management — symbol lists with start/end date ranges. |
| `earnings_calendar_source.py` | 287 | REAL USE | **KEEP** | Earnings Calendar Data Source. |
| `ws_client.py` | 273 | OBSERVABILITY WIRING | **ARCHIVE** | WebSocket Streaming Client for Real-Time Market Data (M23 Task 23.1). |
| `finnhub_news_macro.py` | 256 | REAL USE | **KEEP** | Finnhub news and macro data client. |
| `satellite_features.py` | 242 | OBSERVABILITY WIRING | **ARCHIVE** | Satellite / Geospatial Alternative Data Features (M38c). |
| `web_scraping.py` | 242 | OBSERVABILITY WIRING | **ARCHIVE** | Web Scraping Feature Extraction (M38d). |
| `pit_guard.py` | 235 | OBSERVABILITY WIRING | **ARCHIVE** | Point-in-Time (PIT) safety guard. |
| `calendar.py` | 228 | REAL USE | **KEEP** | Trading calendar utilities (NYSE-based). |
| `resample.py` | 221 | REAL USE | **KEEP** | Multi-Timeframe Resampling for Price Data. |
| `house_ptr_parser.py` | 204 | OBSERVABILITY WIRING | **ARCHIVE** | House Periodic Transaction Report (PTR) parser (T5.3). |
| `cboe_source.py` | 200 | REAL USE | **KEEP** | CBOE Public Data Source — VIX term structure and Put/Call Ratio. |
| `quality_checks.py` | 193 | OBSERVABILITY WIRING | **ARCHIVE** | Data quality checks for price and OHLCV data. |
| `realism_meta.py` | 189 | OBSERVABILITY WIRING | **ARCHIVE** | Realism metadata: label backtest/report outputs with realism assumptions. |
| `social_sentiment.py` | 189 | OBSERVABILITY WIRING | **ARCHIVE** | Social Sentiment Aggregation (M38a). |
| `finnhub_events.py` | 187 | REAL USE | **KEEP** | Finnhub corporate events client — earnings and insider transactions. |
| `contract.py` | 186 | REAL USE | **KEEP** | Alt-data contract: normalisation and PIT filtering for events (Sprint 10.A). |
| `universe_etf.py` | 186 | REAL USE | **KEEP** | ETF Universe Loader for M10 — Universe Upgrade. |
| `minute_bar_aggregator.py` | 184 | OBSERVABILITY WIRING | **ARCHIVE** | Minute Bar Aggregator (M23 Task 23.4). |
| `latency.py` | 182 | REAL USE | **KEEP** | Point-in-time latency helpers for alt-data events. |
| `data_source.py` | 181 | REAL USE | **KEEP** | Price data source abstraction. |
| `alphavantage_source.py` | 164 | OBSERVABILITY WIRING | **ARCHIVE** | Alpha Vantage market data source. |
| `shipping_routes_ingest.py` | 163 | REAL USE | **KEEP** | Shipping routes data ingestion module (Phase 6 Skeleton). |
| `polygon_source.py` | 161 | REAL USE | **KEEP** | Polygon.io OHLCV price data source. |
| `bls_source.py` | 158 | OBSERVABILITY WIRING | **ARCHIVE** | Bureau of Labor Statistics (BLS) data source. |
| `yfinance_source.py` | 158 | REAL USE | **KEEP** | yfinance-based OHLCV price data source. |
| `patent_features.py` | 156 | OBSERVABILITY WIRING | **ARCHIVE** | Patent Activity Features (M38b). |
| `newsapi_source.py` | 150 | OBSERVABILITY WIRING | **DELETE** | News headline source via newsapi.ai (EventRegistry). |
| `store.py` | 145 | OBSERVABILITY WIRING | **DELETE** | News persistence store — month-partitioned Parquet storage. |
| `news_ingest.py` | 143 | OBSERVABILITY WIRING | **DELETE** | News data ingestion module (Phase 6 Skeleton). |
| `panel_store.py` | 143 | OBSERVABILITY WIRING | **DELETE** | Panel store for price panel persistence (Parquet-based). |
| `edgar_source.py` | 142 | OBSERVABILITY WIRING | **DELETE** | SEC EDGAR data source — insider trades (Form 4) and company filings. |
| `worldbank_source.py` | 138 | OBSERVABILITY WIRING | **DELETE** | World Bank macro indicator source. |
| `fred_source.py` | 137 | OBSERVABILITY WIRING | **DELETE** | FRED macro data source (Federal Reserve Economic Data). |
| `insider_ingest.py` | 135 | REAL USE | **KEEP** | Insider trading data ingestion module (Phase 6 Skeleton). |
| `freshness_monitor.py` | 133 | OBSERVABILITY WIRING | **DELETE** | Data Freshness Monitoring (Plan 10.6). |
| `security_master.py` | 131 | REAL USE | **KEEP** | Lightweight security master for symbol metadata. |
| `congress_trades_ingest.py` | 124 | TRADING CYCLE ONLY | **DELETE** | Congress trading data ingestion module (Phase 6 Skeleton). |
| `entity_linking.py` | 118 | OBSERVABILITY WIRING | **DELETE** | News entity linking: map headlines to ticker symbols. |
| `contract.py` | 117 | OBSERVABILITY WIRING | **DELETE** | News data contract — normalisation and PIT filtering. |
| `contract.py` | 115 | REAL USE | **KEEP** | Macro data contract: normalisation and PIT filtering. |
| `contract.py` | 99 | OBSERVABILITY WIRING | **DELETE** | Shipping release data contract — normalisation and PIT filtering. |
| `synthetic_generator.py` | 98 | OBSERVABILITY WIRING | **DELETE** | Synthetic Data Generator (Plan 10.10). |
| `__init__.py` | 74 | is init | **KEEP** | Data ingestion, storage, quality, versioning, streaming and alt-data modules. |
| `snapshot.py` | 73 | REAL USE | **KEEP** | Data snapshot ID computation for reproducibility. |
| `data_versioning.py` | 69 | OBSERVABILITY WIRING | **DELETE** | Data Versioning (Plan 10.9). |
| `finnhub_common.py` | 61 | REAL USE | **KEEP** | Common utilities for Finnhub API clients. |
| `__init__.py` | 55 | is init | **KEEP** | Alt-data modules (insider trades, Congress trades, satellite, patents, social). |
| `__init__.py` | 36 | is init | **KEEP** | Free API data sources for Assembled-Trading-AI. |
| `__init__.py` | 25 | is init | **KEEP** | News data modules: entity linking + persistent store. |
| `__init__.py` | 15 | is init | **KEEP** | Shipping data modules. |
| `__init__.py` | 13 | is init | **KEEP** | Macro data modules. |
| `__init__.py` | 12 | is init | **KEEP** | Streaming data infrastructure for real-time market data. |


## Modul: `src/assembled_core/events/`

**Event-Processing für News, Disclosures, Crisis-Alpha. Die stärkste Komponente im Repo.**

- Files: 47 · Zeilen gesamt: 6,821
- REAL_USE: 39 · Observability-wired: 3 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **143** (2% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `pipeline.py` | 720 | REAL USE | **KEEP** | — |
| `fetch_house_ptr.py` | 523 | REAL USE | **KEEP** | House PTR fetch — index/RSS + optional PDF download (DISCL-2.1). |
| `state_machine.py` | 326 | REAL USE | **KEEP** | Crisis-Alpha state machine — persistent WATCH/ACTIVE/COOLDOWN/PAUSE. |
| `clustering.py` | 323 | REAL USE | **KEEP** | — |
| `pipeline.py` | 273 | REAL USE | **KEEP** | Disclosures pipeline: fetch (stub) -> normalize -> dedupe -> health -> emit. |
| `exit_rules.py` | 261 | REAL USE | **KEEP** | Crisis-Alpha exit rules and deactivation triggers — M5. |
| `fetch_gdelt.py` | 250 | REAL USE | **KEEP** | — |
| `fetch_edgar.py` | 240 | REAL USE | **KEEP** | SEC EDGAR fetch — Form 4 Atom feed (DISCL-1.1). |
| `sources.py` | 223 | REAL USE | **KEEP** | — |
| `normalize.py` | 220 | REAL USE | **KEEP** | — |
| `dedupe_store.py` | 214 | REAL USE | **KEEP** | — |
| `sources.py` | 211 | REAL USE | **KEEP** | Load disclosures source registry and pipeline config. |
| `trigger_scoring.py` | 210 | REAL USE | **KEEP** | Minimal keyword-based trigger scoring for NEWS v1 (Phase 4). |
| `entities.py` | 203 | REAL USE | **KEEP** | — |
| `risk_budget.py` | 191 | REAL USE | **KEEP** | Crisis-Alpha risk budget — daily loss guard and position size limits. |
| `gates.py` | 188 | REAL USE | **KEEP** | Activation and deactivation gate checks for Crisis-Alpha v1. |
| `pipeline.py` | 175 | REAL USE | **KEEP** | Crisis-Alpha v1 pipeline — M5. |
| `baseline.py` | 174 | REAL USE | **KEEP** | — |
| `triggers.py` | 165 | REAL USE | **KEEP** | Disclosure trigger scoring v1: events -> triggers with severity, confidence, TTL, decay. |
| `entry.py` | 157 | REAL USE | **KEEP** | Crisis-Alpha simple entry signals — M5. |
| `fetch_rss.py` | 137 | REAL USE | **KEEP** | — |
| `baskets.py` | 118 | REAL USE | **KEEP** | Crisis-Alpha ETF basket definitions — M5. |
| `burst.py` | 118 | REAL USE | **KEEP** | — |
| `normalize.py` | 104 | REAL USE | **KEEP** | Normalize raw disclosure items into DisclosureEvent. |
| `tfidf.py` | 97 | REAL USE | **KEEP** | — |
| `__init__.py` | 73 | is init | **KEEP** | Crisis-Alpha v1 subsystem — M5. |
| `__init__.py` | 73 | is init | **KEEP** | News v1 pipeline (MVP, free & robust). |
| `context.py` | 68 | REAL USE | **KEEP** | CrisisAlphaContext — input contract for the Crisis-Alpha v1 subsystem. |
| `misinfo_risk.py` | 62 | OBSERVABILITY WIRING | **DELETE** | Misinfo risk scorer for news cluster evidence. |
| `health.py` | 61 | REAL USE | **KEEP** | Compute DisclosuresHealth from counts and failures. |
| `health.py` | 59 | REAL USE | **KEEP** | — |
| `evidence.py` | 54 | REAL USE | **KEEP** | — |
| `models.py` | 54 | REAL USE | **KEEP** | Disclosure event and health models (v1 minimal). |
| `models.py` | 51 | REAL USE | **KEEP** | — |
| `evidence.py` | 49 | REAL USE | **KEEP** | Evidence summarization for disclosures (single-event or multi-event). |
| `state.py` | 48 | REAL USE | **KEEP** | — |
| `fingerprint.py` | 45 | REAL USE | **KEEP** | — |
| `grader.py` | 43 | OBSERVABILITY WIRING | **DELETE** | Evidence grader: derives EvidenceGrade from cluster evidence summary. |
| `__init__.py` | 42 | is init | **KEEP** | Disclosures v1 pipeline (contract & skeleton). |
| `dedupe.py` | 40 | REAL USE | **KEEP** | Dedupe disclosure events by fingerprint. |
| `dedupe.py` | 39 | REAL USE | **KEEP** | — |
| `action_gate.py` | 38 | OBSERVABILITY WIRING | **DELETE** | Action gate: controls which Crisis Alpha actions are permitted based on evidence grade. |
| `grades.py` | 27 | REAL USE | **KEEP** | Evidence grade definitions for the Evidence Engine. |
| `emit.py` | 25 | REAL USE | **KEEP** | — |
| `emit.py` | 24 | REAL USE | **KEEP** | Atomic JSON emit for disclosures artifacts. |
| `__init__.py` | 13 | is init | **KEEP** | Evidence Engine — M8: Fake-News Defense and Evidence Grading. |
| `__init__.py` | 12 | is init | **KEEP** | Events domain — news, disclosures, crisis-alpha, evidence engine. |


## Modul: `src/assembled_core/intel/`

**News-Anreicherung, Geo-Risk, Entity-Linking. 54 Files, viele Duplikate mit events/news.**

- Files: 54 · Zeilen gesamt: 12,163
- REAL_USE: 40 · Observability-wired: 12 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **2,525** (20% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `news_classifier.py` | 637 | REAL USE | **KEEP** | Rule-based news event classifier. |
| `shock_propagation.py` | 484 | REAL USE | **KEEP** | Shock propagation through the geopolitical dependency graph. |
| `models.py` | 483 | REAL USE | **KEEP** | Pydantic v2 models for the Intel/Crisis Alpha Pipeline. |
| `rss_fetcher.py` | 447 | REAL USE | **KEEP** | Generic RSS/Atom feed fetcher for the Intel pipeline. |
| `news_ingest.py` | 444 | REAL USE | **KEEP** | GDELT 2.0 GKG batch ingest — 15-minute polling, no API key required. |
| `news_dedupe.py` | 376 | REAL USE | **KEEP** | News event deduplication — prevents same event from triggering multiple times. |
| `shipping_lanes.py` | 363 | OBSERVABILITY WIRING | **ARCHIVE** | Shipping lane dependency modeling for geopolitical risk analysis. |
| `geo_trigger.py` | 340 | REAL USE | **KEEP** | Rules engine for scoring news events/clusters into geo triggers. |
| `sanctions_model.py` | 332 | OBSERVABILITY WIRING | **ARCHIVE** | Sanctions package modeling for geopolitical risk analysis. |
| `news_enricher.py` | 331 | REAL USE | **KEEP** | NewsEventEnricher — pipeline enrichment step for raw NewsEvent objects. |
| `news_entity_mapper.py` | 328 | REAL USE | **KEEP** | Lightweight news-to-ticker entity mapper. |
| `dependency_graph.py` | 312 | REAL USE | **KEEP** | Load and traverse the geopolitical dependency graph. |
| `weaponized_interdependence.py` | 304 | OBSERVABILITY WIRING | **ARCHIVE** | Weaponized Interdependence scoring (Farrell & Newman 2019). |
| `news_position_bridge.py` | 300 | REAL USE | **KEEP** | News-to-position bridge (Point 33). |
| `health_monitor.py` | 284 | REAL USE | **KEEP** | Simple freshness/health tracking for intel pipeline components. |
| `news_trade_attribution.py` | 269 | REAL USE | **KEEP** | News-zu-Trade-Attribution. |
| `currency_crisis.py` | 264 | OBSERVABILITY WIRING | **ARCHIVE** | Currency crisis modeling for geopolitical and macro risk analysis. |
| `central_bank_divergence.py` | 252 | OBSERVABILITY WIRING | **ARCHIVE** | Central bank policy divergence modeling. |
| `news_event_store.py` | 249 | REAL USE | **KEEP** | In-memory queryable NewsEvent store for the intel pipeline. |
| `pit_store.py` | 241 | REAL USE | **KEEP** | Full Point-in-Time (PIT) artifact store (X1). |
| `crisis_alpha_worker.py` | 236 | REAL USE | **KEEP** | Crisis Alpha Worker v0 — state machine for crisis mode detection and risk posture. |
| `entity_linker.py` | 211 | OBSERVABILITY WIRING | **ARCHIVE** | Entity → Ticker Linker (X2). |
| `bayesian_confidence.py` | 207 | REAL USE | **KEEP** | Bayesian confidence updating for the intel pipeline. |
| `news_entity_graph.py` | 206 | REAL USE | **KEEP** | Entity co-occurrence graph (lightweight, no NetworkX dependency). |
| `nation_profiles.py` | 202 | REAL USE | **KEEP** | Nation resource/vulnerability profiles for geopolitical risk modeling. |
| `news_signal_aggregator.py` | 201 | REAL USE | **KEEP** | Intel signal aggregator — combines multiple cluster signals into a unified view. |
| `news_alerts.py` | 197 | REAL USE | **KEEP** | Alert system for critical news events. |
| `news_impact_estimator.py` | 191 | REAL USE | **KEEP** | News impact estimator — maps event classification to expected return impact. |
| `news_macro_calendar.py` | 190 | REAL USE | **KEEP** | Macro calendar hooks for the news engine. |
| `news_cluster.py` | 186 | REAL USE | **KEEP** | News event clustering — groups related events into evidence clusters. |
| `sector_news_overlay.py` | 180 | REAL USE | **KEEP** | Sector-level news risk overlay for portfolio construction. |
| `news_impact_calibrator.py` | 179 | OBSERVABILITY WIRING | **ARCHIVE** | Skeleton calibrator for news impact priors. |
| `news_archive.py` | 172 | REAL USE | **KEEP** | JSONL archive for NewsEvents — raw append + chronological replay. |
| `news_archiver.py` | 171 | REAL USE | **KEEP** | JSONL event archiver for news replay and backtesting (Point 29). |
| `news_ticker_velocity.py` | 170 | REAL USE | **KEEP** | Per-ticker news velocity tracker. |
| `news_semantic_dedup.py` | 162 | REAL USE | **KEEP** | Semantic deduplication (gated). |
| `news_velocity.py` | 158 | REAL USE | **KEEP** | Breaking news velocity tracker. |
| `market_confirmation.py` | 155 | REAL USE | **KEEP** | Market confirmation signals for crisis state transitions. |
| `news_contradiction.py` | 151 | REAL USE | **KEEP** | News source contradiction detection. |
| `ic_loop.py` | 143 | REAL USE | **KEEP** | IC Feedback Loop (X3) — measures Information Coefficient per trigger type. |
| `news_sentiment_drift.py` | 140 | REAL USE | **KEEP** | News sentiment drift / trajectory tracker. |
| `news_corroboration.py` | 130 | REAL USE | **KEEP** | Cross-source corroboration tracker. |
| `news_newsapi_fetcher.py` | 124 | OBSERVABILITY WIRING | **DELETE** | NewsAPI.org fetcher (gated; requires NEWSAPI_KEY env var). |
| `news_source_voting.py` | 122 | REAL USE | **KEEP** | Tier-weighted source voting for news direction / event-type consensus. |
| `news_replay.py` | 121 | OBSERVABILITY WIRING | **DELETE** | News-Replay harness for backtesting (X5). |
| `news_decay.py` | 117 | REAL USE | **KEEP** | News-impact decay curves. |
| `news_language.py` | 107 | REAL USE | **KEEP** | Lightweight language detection for news headlines. |
| `trigger_snapshot_store.py` | 106 | OBSERVABILITY WIRING | **DELETE** | Trigger snapshot store — archives triggers_latest.json per run_id (T6.1 / X1-lite). |
| `evidence_grade_writer.py` | 96 | OBSERVABILITY WIRING | **DELETE** | Evidence grade artifact writer (T7.9). |
| `wild_card_detector.py` | 93 | OBSERVABILITY WIRING | **DELETE** | Wild-Card Event Detection (Plan 4.7). |
| `__init__.py` | 87 | is init | **KEEP** | Intel loaders and geopolitical/macro intel modules. |
| `news_triggers_loader.py` | 80 | TEST ONLY | **ARCHIVE** | Load news triggers snapshot for TradingContext (read-only, tolerant). |
| `disclosures_triggers_loader.py` | 79 | REAL USE | **KEEP** | Load disclosures triggers snapshot for TradingContext (read-only, tolerant). |
| `source_registry.py` | 53 | REAL USE | **KEEP** | Static source registry with tier and trust weight information. |


## Modul: `src/assembled_core/ops/`

**Operations, Scheduler, Health-Checks, Alert-Manager.**

- Files: 32 · Zeilen gesamt: 7,598
- REAL_USE: 25 · Observability-wired: 6 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **1,258** (16% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `daily_scheduler.py` | 826 | REAL USE | **KEEP** | daily_scheduler.py — Autonomous daily operations orchestrator. |
| `paper_runner.py` | 746 | REAL USE | **KEEP** | OPS-6: Paper daily runner — callable helper for one-day and range runs. |
| `kpi_artifacts.py` | 613 | REAL USE | **KEEP** | KPI artifact writer for paper/shadow runs (OPS-1). |
| `grafana_dashboards.py` | 378 | OBSERVABILITY WIRING | **ARCHIVE** | Grafana Dashboard Definitions (M31 Task 31.3). |
| `paper_ledger.py` | 341 | REAL USE | **KEEP** | OPS-4: Paper execution ledger — load/save state, simulate fills, apply to ledger, mark-to- |
| `replay_snapshot.py` | 330 | REAL USE | **KEEP** | Deterministic replay snapshots for paper runs. |
| `intel_activity_summary.py` | 315 | REAL USE | **KEEP** | OPS-13: Intel activity summary across experiment run days. |
| `alerts.py` | 272 | REAL USE | **KEEP** | OPS-3: Alerts/Anomalies v1 — deterministic alerts from run_kpis, reasons, diff artifacts. |
| `health_check.py` | 262 | REAL USE | **KEEP** | Health Check Core Module. |
| `self_healing.py` | 247 | OBSERVABILITY WIRING | **ARCHIVE** | Self-Healing + Autonomous Operations (M35). |
| `trade_journal.py` | 246 | REAL USE | **KEEP** | Trade Journal — Per-trade logging and daily summary generation. |
| `execution_cost_meta.py` | 219 | REAL USE | **KEEP** | Part B deeper wiring: pre-trade execution cost estimates. |
| `paper_summary.py` | 219 | REAL USE | **KEEP** | OPS-6: Paper range summary — aggregate metrics from daily run artifacts. |
| `alert_sinks.py` | 207 | OBSERVABILITY WIRING | **ARCHIVE** | Alert sinks — Slack webhook + email (Sprint 4 / Plan C14). |
| `heartbeat.py` | 200 | REAL USE | **KEEP** | Heartbeat + liveness helpers (Sprint 4 / Plan C16). |
| `experience_log.py` | 188 | REAL USE | **KEEP** | Experience Log — Append-only JSONL audit trail for trading cycles. |
| `experiment_runner.py` | 181 | REAL USE | **KEEP** | OPS-7: A/B paper experiment runner — run paper range with policy overrides and write summa |
| `dashboard_data.py` | 180 | OBSERVABILITY WIRING | **ARCHIVE** | Dashboard Data Provider — Backend for Streamlit/Dash UI. |
| `run_manifest.py` | 180 | REAL USE | **KEEP** | Run manifest for a single paper-engine day. |
| `metrics_exporter.py` | 177 | REAL USE | **KEEP** | Metrics exporter — Prometheus text format + optional push-gateway. |
| `certification.py` | 175 | OBSERVABILITY WIRING | **ARCHIVE** | Certification & Sign-Off Checklist for Go-Live. |
| `reconcile.py` | 165 | REAL USE | **KEEP** | OPS-5: Reconcile report and invariants for paper ledger runs. |
| `shadow_mode.py` | 125 | REAL USE | **KEEP** | Part D — Shadow-mode infrastructure for wiring activation (D1–D5). |
| `run_index.py` | 120 | REAL USE | **KEEP** | Cross-run aggregate index for paper-engine runs. |
| `intel_orchestrator.py` | 106 | REAL USE | **KEEP** | OPS-11: Intel orchestrator — run real NEWS + DISCLOSURES pipelines before trading cycle. |
| `__init__.py` | 93 | is init | **KEEP** | Operations & Monitoring Module. |
| `compare.py` | 92 | REAL USE | **KEEP** | OPS-7: Compare two paper experiment summaries (A/B). |
| `alert_manager.py` | 89 | REAL USE | **KEEP** | Alert Manager (Plan 11.9). |
| `inspect_data.py` | 83 | REAL USE | **KEEP** | OPS-8: EOD price coverage inspector — min/max timestamps and recommended experiment window |
| `intel_sim.py` | 79 | REAL USE | **KEEP** | BENCH-1/BENCH-2: Intel simulation harness for policy A/B — deterministic news_geo + disclo |
| `report_retention.py` | 73 | REAL USE | **KEEP** | Retention utility for date-stamped report files. |
| `shadow_recorder.py` | 71 | TRADING CYCLE ONLY | **DELETE** | Shadow-mode helper — thin wrapper over ``shadow_mode.write_shadow_snapshot``. |


## Modul: `src/assembled_core/paper/`

**Paper-Track-Runner (2060 Zeilen) und Paper-Engine-Adapter.**

- Files: 9 · Zeilen gesamt: 3,108
- REAL_USE: 8 · Observability-wired: 0 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **0** (0% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `paper_track.py` | 2,060 | REAL USE | **REFACTOR** | Paper Track Runner - Orchestrator + State IO. |
| `intel_context.py` | 395 | REAL USE | **KEEP** | Part B wiring: populate TradingContext intel attrs from artifacts. |
| `intel_runner.py` | 184 | REAL USE | **KEEP** | Minimal real-intel orchestration for paper track runner. |
| `strategy_adapters.py` | 134 | REAL USE | **KEEP** | Strategy adapters for Paper-Track. |
| `ranking_hysteresis.py` | 86 | REAL USE | **KEEP** | Ranking hysteresis for paper track runner. |
| `rebalance_filter.py` | 82 | REAL USE | **KEEP** | Minimal rebalance filter to suppress small order churn. |
| `deadzone_rebalance.py` | 79 | REAL USE | **KEEP** | Dead-zone rebalance filter for paper track runner. |
| `georisk_gate.py` | 63 | REAL USE | **KEEP** | Minimal GeoRisk gate for paper track runner. |
| `__init__.py` | 25 | is init | **KEEP** | Paper Trading Track Module. |


## Modul: `src/assembled_core/accounting/`

**Ledger, Position-Engine, Reconciliation, Evidence-Pack.**

- Files: 18 · Zeilen gesamt: 5,908
- REAL_USE: 14 · Observability-wired: 3 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **281** (4% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `evidence_pack.py` | 1,147 | REAL USE | **REFACTOR** | Evidence pack exporter for accounting artifacts. |
| `ledger.py` | 610 | REAL USE | **KEEP** | Ledger event generation and contract (Sprint 13). |
| `position_engine.py` | 512 | REAL USE | **KEEP** | Position engine: Build positions from ledger events (Sprint 13 L2). |
| `reconciliation_report.py` | 491 | REAL USE | **KEEP** | Reconciliation report writer (Sprint 13 L4). |
| `ledger_integration.py` | 489 | REAL USE | **KEEP** | Ledger integration helper for pipeline (Sprint 13 L5). |
| `broker_snapshot_importer.py` | 439 | REAL USE | **KEEP** | Broker snapshot importer (Sprint 13). |
| `accounting_report.py` | 397 | REAL USE | **KEEP** | Accounting report writer (Sprint 13). |
| `reconciliation.py` | 395 | REAL USE | **KEEP** | Reconciliation engine: Compare ledger state vs broker snapshots (Sprint 13 L3). |
| `ledger_store.py` | 257 | REAL USE | **KEEP** | Ledger storage module (Sprint 13). |
| `attribution.py` | 255 | REAL USE | **KEEP** | Attribution drilldowns for paper-engine runs. |
| `broker_snapshot_store.py` | 246 | REAL USE | **KEEP** | Broker snapshot storage (Sprint 13). |
| `evidence_index.py` | 135 | REAL USE | **KEEP** | Evidence index writer for accounting artifacts. |
| `round_trips.py` | 117 | OBSERVABILITY WIRING | **DELETE** | Round-Trip P&L Analysis (Plan 8.7). |
| `tax_lots.py` | 107 | OBSERVABILITY WIRING | **DELETE** | Tax Lot Tracking — FIFO (Plan 8.4). |
| `broker_snapshot.py` | 94 | REAL USE | **KEEP** | Broker snapshot normalization and contract (Sprint 13). |
| `__init__.py` | 83 | is init | **KEEP** | Accounting and ledger system for paper trading (Sprint 13). |
| `currency.py` | 77 | REAL USE | **KEEP** | Multi-Currency Support (Plan 8.8). |
| `decision_audit.py` | 57 | OBSERVABILITY WIRING | **DELETE** | Decision Audit Trail (Plan 8.10). |


## Modul: `src/assembled_core/compliance/`

**MiFID II / KWG-Schicht. Alle 3 Files observability-only.**

- Files: 4 · Zeilen gesamt: 762
- REAL_USE: 0 · Observability-wired: 3 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **719** (94% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `regulatory_reports.py` | 331 | OBSERVABILITY WIRING | **ARCHIVE** | Regulatory Report Generator. |
| `audit_log.py` | 229 | OBSERVABILITY WIRING | **ARCHIVE** | Hash-chained Tamper-Evident Audit Log. |
| `otr_monitor.py` | 159 | OBSERVABILITY WIRING | **ARCHIVE** | Order-to-Trade Ratio (OTR) Monitor — MiFID II compliance. |
| `__init__.py` | 43 | is init | **KEEP** | Compliance & Audit modules for regulatory readiness. |


## Modul: `src/assembled_core/api/`

**FastAPI-Backend. 13 Files, alle REAL_USE.**

- Files: 13 · Zeilen gesamt: 3,463
- REAL_USE: 11 · Observability-wired: 0 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **0** (0% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `models.py` | 876 | REAL USE | **KEEP** | Pydantic models for FastAPI endpoints (future implementation). |
| `monitoring.py` | 620 | REAL USE | **KEEP** | Monitoring endpoints for QA, Risk, and Drift status. |
| `qa.py` | 401 | REAL USE | **KEEP** | QA/Health check endpoints. |
| `paper_trading.py` | 355 | REAL USE | **KEEP** | Paper trading endpoints. |
| `portfolio.py` | 317 | REAL USE | **KEEP** | Portfolio endpoints. |
| `oms.py` | 196 | REAL USE | **KEEP** | OMS (Order Management System) endpoints for blotter and execution views. |
| `signals.py` | 195 | REAL USE | **KEEP** | Signals endpoints. |
| `performance.py` | 162 | REAL USE | **KEEP** | Performance endpoints. |
| `risk.py` | 121 | REAL USE | **KEEP** | Risk endpoints. |
| `app.py` | 110 | REAL USE | **KEEP** | FastAPI application factory. |
| `orders.py` | 91 | REAL USE | **KEEP** | Orders endpoints. |
| `__init__.py` | 17 | is init | **KEEP** | FastAPI backend modules. |
| `__init__.py` | 2 | is init | **KEEP** | API routers for FastAPI endpoints. |


## Modul: `src/assembled_core/reports/`

**Report-Writer. 3 Files.**

- Files: 3 · Zeilen gesamt: 745
- REAL_USE: 2 · Observability-wired: 0 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **0** (0% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `daily_qa_report.py` | 562 | REAL USE | **KEEP** | Daily QA Report generation. |
| `metrics_export.py` | 161 | REAL USE | **KEEP** | Metrics export utilities for backtest results. |
| `__init__.py` | 22 | is init | **KEEP** | Report generation modules. |


## Modul: `src/assembled_core/config/`

**Settings, Policy-Schema, Factor-Bundles.**

- Files: 10 · Zeilen gesamt: 1,487
- REAL_USE: 6 · Observability-wired: 2 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **477** (32% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `settings.py` | 342 | REAL USE | **KEEP** | Central settings configuration using Pydantic Settings. |
| `models.py` | 338 | TEST ONLY | **ARCHIVE** | Pydantic models for strict configuration validation. |
| `factor_bundles.py` | 232 | REAL USE | **KEEP** | Factor Bundle Configuration Module. |
| `policy_schema.py` | 193 | REAL USE | **KEEP** | Pydantic schema for policy.yaml validation (MEDIUM-6.3). |
| `__init__.py` | 114 | is init | **KEEP** | Configuration package for Assembled Trading AI. |
| `secrets_loader.py` | 85 | OBSERVABILITY WIRING | **DELETE** | Secret-key loader — reads env vars, optionally from .env file. |
| `policy_loader.py` | 56 | REAL USE | **KEEP** | — |
| `logging_config.py` | 54 | OBSERVABILITY WIRING | **DELETE** | Structured JSON Logging (Plan 11.7). |
| `config.py` | 43 | REAL USE | **KEEP** | Central configuration for the trading pipeline. |
| `constants.py` | 30 | REAL USE | **KEEP** | Central constants for the trading system. |


## Modul: `src/assembled_core/experiments/`

**Batch-Runner und Experiment-Config.**

- Files: 3 · Zeilen gesamt: 2,049
- REAL_USE: 1 · Observability-wired: 1 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **1,623** (79% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `batch_runner.py` | 1,623 | OBSERVABILITY WIRING | **ARCHIVE** | Batch runner for executing multiple backtest runs (serial and parallel). |
| `batch_config.py` | 401 | REAL USE | **KEEP** | Batch configuration for systematic backtest runs. |
| `__init__.py` | 25 | is init | **KEEP** | Experiment configuration and batch runner modules. |


## Modul: `src/assembled_core/utils/`

**Timing, Dataframe-Utils, Paths. 5 Files, klein und fokussiert.**

- Files: 5 · Zeilen gesamt: 307
- REAL_USE: 4 · Observability-wired: 0 · Wiring-Test-only: 0 · Zero: 0
- Zeilen, die in DELETE/ARCHIVE fallen: **0** (0% des Moduls)

### File-Liste (sortiert nach Größe)

| Datei | Zeilen | Usage | Verdict | Zweck |
|-------|-------:|-------|---------|-------|
| `timing.py` | 154 | REAL USE | **KEEP** | Timing utilities for profiling pipeline steps. |
| `random_state.py` | 59 | REAL USE | **KEEP** | Global random state utilities for deterministic backtests and experiments. |
| `dataframe.py` | 42 | REAL USE | **KEEP** | DataFrame utility functions (shared across layers). |
| `paths.py` | 28 | REAL USE | **KEEP** | Path utility functions (shared across layers). |
| `__init__.py` | 24 | is init | **KEEP** | Utility modules for assembled-trading-ai. |


---

# Kritische Strukturbefunde (über Tabellen hinaus)

Die obigen Tabellen zeigen Verdicts pro Datei. Aber es gibt strukturelle Befunde, die quer zu Modulgrenzen gehen. Diese sind meist gefährlicher als einzelne Datei-Probleme.

## A. Namenskonflikte im Package

### A.1 `config.py` existiert DOPPELT
**Schwer · Datenintegritätsrisiko**

Das Repo hat:
- `src/assembled_core/config.py` (43 Zeilen, 35 Importer)
- `src/assembled_core/config/` (Package-Ordner)
- `src/assembled_core/config/config.py` (43 Zeilen, 7 Importer)

Die beiden `config.py`-Files sind **fast identisch**, unterscheiden sich nur in einer einzigen Zeile:
- Root-Version: `_BASE_DIR = Path(__file__).resolve().parents[2]`
- Package-Version: `_BASE_DIR = Path(__file__).resolve().parents[3]`

Beide exportieren `OUTPUT_DIR` und `get_output_path()`. Je nachdem, welche importiert wird, zeigt `OUTPUT_DIR` auf **unterschiedliche Verzeichnisse**. Das ist die Datenintegritäts-Zeitbombe. Wenn ein Script die eine Version, ein anderes die andere Version importiert, schreiben sie ihre Outputs in unterschiedliche Pfade — mit zufälligem Pattern.

**Zusätzlich:** `logging_config.py` existiert auch doppelt — `src/assembled_core/logging_config.py` (180 Zeilen) und `src/assembled_core/config/logging_config.py` (54 Zeilen). Verschiedener Inhalt, ähnlicher Zweck.

**Verdict:** DELETE einer der beiden `config.py`-Files. Konsolidiere alle Importer auf die Package-Variante (`config/config.py`), oder umgekehrt. Der parallele Existenzzustand ist nicht haltbar.

### A.2 `strategies/stat_arb.py` UND `strategies/stat_arb/` Ordner
**Mittel · Unklare Primärquelle**

- `src/assembled_core/strategies/stat_arb.py` (366 Zeilen, OBSERVABILITY_WIRING)
- `src/assembled_core/strategies/stat_arb/` (Ordner mit 4 Files: `__init__.py`, `cointegration.py`, `pair_signals.py`, `pca_arb.py`)

Der flache File ist observability-wired. Der Ordner hat die eigentliche Implementation. Das ist das Resultat einer halben Migration, wo der Code von `strategies/stat_arb.py` in einen Subfolder verschoben wurde — aber das alte File wurde nie gelöscht.

**Verdict:** DELETE des flachen `stat_arb.py`. Der Sub-Ordner ist die Wahrheit.

### A.3 `intel/news_*` dupliziert mit `events/news/*`
**Schwer · Funktionalitätsüberlappung**

Beide Module haben Files zu demselben Thema:

| Datei in `intel/` | Äquivalent in `events/news/` |
|---|---|
| `news_dedupe.py` (376 Z.) | `dedupe.py` + `dedupe_store.py` |
| `news_cluster.py` | `clustering.py` |
| `news_enricher.py` | (nur entity-basiert in events) |
| `news_event_store.py` | `dedupe_store.py` |
| `news_archive.py`, `news_archiver.py` | — |
| `news_classifier.py` | `trigger_scoring.py` |

Das deutet auf zwei parallele News-Architekturen hin. `events/news/` ist moderner (meine Analyse: echte Pipeline mit RSS+GDELT). `intel/news_*` scheint älter und hat Enrichment-Stufen, die nicht in `events/news/` sind.

**Verdict:** Eine der beiden Hierarchien muss primär werden. Mein Vorschlag: `events/news/` ist die Ingest+Clustering-Schicht, `intel/news_*` ist die Enrichment+Classification-Schicht. Dann explicit dokumentieren. Duplikate (z.B. News-Dedupe an beiden Stellen) konsolidieren.

## B. Verdeckte Module-Hierarchien (Sub-Pakete)

Innerhalb der Haupt-Module gibt es Sub-Pakete, die in den Tabellen oben nicht sichtbar waren. Hier die wichtigsten:

### B.1 `src/assembled_core/api/routers/` (10 Files)
Alle REAL_USE, alle werden von `api/app.py` registriert. Diese Schicht ist gut strukturiert und **die einzige**, die komplett ohne Observability-Overhead arbeitet.

- `monitoring.py` (620 Z.) — **REFACTOR**: Liefert laut `KNOWN_ISSUES.md` Dummy-Drift-Daten
- `qa.py` (401 Z.) — KEEP
- `paper_trading.py` (355 Z.) — KEEP
- `portfolio.py` (317 Z.) — KEEP
- `oms.py` (196 Z.) — KEEP
- `signals.py` (195 Z.) — KEEP
- `performance.py` (162 Z.) — KEEP
- `risk.py` (121 Z.) — KEEP
- `orders.py` (91 Z.) — KEEP

### B.2 `src/assembled_core/data/altdata/` (10 Files)
Der Observability-Friedhof der Alt-Data-Module:
- `finnhub_news_macro.py` (256 Z.) — REAL_USE, KEEP
- `satellite_features.py` (242 Z.) — OBSERVABILITY, **ARCHIVE** (kein Ingest)
- `web_scraping.py` (242 Z.) — OBSERVABILITY, **ARCHIVE** (kein Ingest)
- `house_ptr_parser.py` (204 Z.) — OBSERVABILITY, **ARCHIVE**
- `finnhub_events.py` (190 Z.) — REAL_USE, KEEP
- `patent_features.py` (175 Z.) — OBSERVABILITY, **ARCHIVE** (kein Ingest)
- `social_sentiment.py` (108 Z.) — OBSERVABILITY, **ARCHIVE** (kein Source)
- `finnhub_common.py` — REAL_USE, KEEP
- `contract.py` — REAL_USE, KEEP

**Verdict für das gesamte Sub-Modul:** 6 von 10 Files sind Alibi-Code. Archive, bis echter Ingest geplant ist.

### B.3 `src/assembled_core/data/sources/` (11 Files)
Data-Source-Clients. Gemischtes Bild:
- `earnings_calendar_source.py` (287 Z.) — REAL_USE, KEEP
- `cboe_source.py` (200 Z.) — REAL_USE, KEEP
- `polygon_source.py` — REAL_USE, KEEP
- `alphavantage_source.py` — REAL_USE, KEEP
- `fred_source.py` — REAL_USE, KEEP
- `newsapi_source.py` — REAL_USE, KEEP
- `edgar_source.py` — REAL_USE, KEEP
- `worldbank_source.py` — REAL_USE, KEEP
- `bls_source.py` — REAL_USE, KEEP
- `yfinance_source.py` — REAL_USE, KEEP — **aber:** kommerzielle Nutzung verboten

**Verdict:** KEEP alle, aber `yfinance_source.py` muss kommerziell durch Polygon oder Alpha Vantage ersetzt werden, falls du Richtung SaaS gehst.

### B.4 `src/assembled_core/events/news/` (20 Files)
Die stärkste Sub-Hierarchie im Repo. Alle REAL_USE außer zweier Edge-Cases:
- `pipeline.py` (720 Z.) — KEEP (orchestriert alles)
- `sources.py` (223 Z.) — KEEP
- `fetch_gdelt.py` (250 Z.) — KEEP
- `fetch_rss.py` — KEEP
- `normalize.py` (220 Z.) — KEEP
- `clustering.py` — KEEP
- `dedupe.py` + `dedupe_store.py` — KEEP
- `fingerprint.py` — KEEP
- `trigger_scoring.py` — KEEP
- `baseline.py` — KEEP
- `burst.py` — KEEP
- `emit.py` — KEEP
- `entities.py` — KEEP
- `evidence.py` — KEEP
- `health.py` — KEEP
- `models.py` — KEEP
- `state.py` — KEEP
- `tfidf.py` — KEEP

**Verdict:** Das gesamte Sub-Modul ist KEEP und produktionsreif. Das ist **die** Referenz-Qualität, an der andere Module sich messen lassen müssen.

### B.5 `src/assembled_core/events/crisis_alpha/` (9 Files)
Sauber strukturiert, komplett:
- `state_machine.py` (326 Z.) — KEEP
- `exit_rules.py` (261 Z.) — KEEP
- `risk_budget.py` (191 Z.) — KEEP
- `gates.py` (188 Z.) — KEEP
- `pipeline.py` (175 Z.) — KEEP
- `entry.py` (157 Z.) — KEEP
- `baskets.py` (118 Z.) — KEEP
- `context.py` (68 Z.) — KEEP

**Verdict:** KEEP alle, aber: **nie gegen echte Krisen backtested**. Der Code ist sauber, die Validierung fehlt komplett (siehe Teil 1, Befund 2.5).

### B.6 `src/assembled_core/events/disclosures/` (12 Files)
SEC-EDGAR-basierte Disclosure-Verarbeitung:
- `pipeline.py` — KEEP
- `normalize.py` — KEEP
- `triggers.py` — KEEP
- `health.py` — KEEP
- `models.py` — KEEP
- `fetch_edgar.py` — KEEP
- `fetch_house_ptr.py` — KEEP (Congress-Trading)
- `emit.py` — KEEP
- `evidence.py` — KEEP
- `dedupe.py` — KEEP
- `sources.py` — KEEP

**Verdict:** KEEP alle. Gut strukturiert, parallel zu `events/news/`. Auch hier: nie gegen echte Event-Studies validiert.

### B.7 `src/assembled_core/events/evidence_engine/` (5 Files)
Evidenzsammlung für Trading-Entscheidungen. Praktisch nur via observability-wired:

- `evidence_grade.py` — OBSERVABILITY_WIRING
- `signal_evidence.py` — OBSERVABILITY_WIRING
- `tier_classifier.py` — OBSERVABILITY_WIRING

**Verdict:** ARCHIVE das ganze Sub-Modul. Wenn du "Evidence-Grade"-Logik brauchst, wieder reinholen und richtig integrieren.

## C. Die echten produktiven Kern-Module

Wenn wir nur die Files mit ≥5 Importern und REAL_USE nehmen, kristallisiert sich die **echte** Codebase heraus — die Module, die tatsächlich die Arbeit machen:

### C.1 Pipeline (7 Files, ~13.000 Zeilen)
- `trading_cycle.py` (10.544 Z., 41 Importer) — **muss zerlegt werden**, siehe Teil 1, Befund 1.1
- `orchestrator.py` (1.351 Z., 12 Importer) — REFACTOR (zu groß)
- `backtest.py` (443 Z., 12 Importer) — KEEP
- `portfolio.py` (334 Z., 8 Importer) — KEEP
- `io.py` (184 Z., 13 Importer) — KEEP
- `orders.py` (100 Z., 5 Importer) — KEEP (aber Bug: qty=1.0 festcodiert)
- `signals.py` (78 Z., 7 Importer) — KEEP (aber: ist EMA-Crossover, zu simpel)

### C.2 Execution (10 Files, ~10.000 Zeilen)
- `unified_paper_engine.py` (2.696 Z., 26 Importer) — REFACTOR (zu groß)
- `pre_trade_checks.py` (1.261 Z., 15 Importer) — KEEP
- `transaction_costs.py` (1.008 Z., 12 Importer) — KEEP
- `fill_model.py` (967 Z., 17 Importer) — KEEP
- `broker_adapter.py` (811 Z., 11 Importer) — KEEP
- `order_generation.py` (468 Z., 14 Importer) — KEEP
- `intent_store.py` (345 Z., 10 Importer) — KEEP
- `kill_switch.py` (326 Z., 11 Importer) — KEEP
- `safe_bridge.py` (253 Z., 8 Importer) — KEEP
- `fill_model_pipeline.py` (74 Z., 7 Importer) — KEEP

**Rest von `execution/`:** 21 weitere Files, davon 7 observability-wired. Darunter `ibkr_adapter.py` (342 Z., OBSERVABILITY — nie real benutzt), `paper_monitoring.py` (338 Z.), `broker_execution.py` (266 Z.).

### C.3 QA (10 Kern-Files, ~10.000 Zeilen)
- `backtest_engine.py` (1.434 Z., 35 Importer) — REFACTOR (zu groß)
- `walk_forward.py` (1.248 Z., 10 Importer) — REFACTOR
- `metrics.py` (1.148 Z., 29 Importer) — REFACTOR (zu groß)
- `scenario_engine.py` (1.124 Z., 7 Importer) — REFACTOR
- `factor_analysis.py` (2.346 Z., 13 Importer) — REFACTOR (zu groß)
- `qa_gates.py` (657 Z., 12 Importer) — KEEP
- `tca.py` (411 Z., 8 Importer) — KEEP
- `learning_store.py` (329 Z., 8 Importer) — KEEP
- `point_in_time_checks.py` (216 Z., 9 Importer) — KEEP
- `benchmark_metrics.py` (219 Z., 9 Importer) — KEEP

**Rest von `qa/`:** 40 weitere Files, davon 14 observability-wired. Darunter `robustness.py` (1.403 Z., **OBSERVABILITY_WIRING** — 1403 Zeilen!), `e2e_integration.py` (431 Z.), `reverse_stress.py` (407 Z.), `factor_report.py` (339 Z.).

### C.4 Features (10 Kern-Files, ~5.000 Zeilen)
- `ta_features.py` (830 Z., 28 Importer) — KEEP
- `ta_liquidity_vol_factors.py` (512 Z., 10 Importer) — KEEP
- `ta_factors_core.py` (308 Z., 9 Importer) — KEEP
- `altdata_earnings_insider_factors.py` (714 Z., 8 Importer) — KEEP
- `market_breadth.py` (665 Z., 8 Importer) — KEEP
- `altdata_news_macro_factors.py` (702 Z., 6 Importer) — KEEP
- `event_features.py` (303 Z., 7 Importer) — KEEP
- `factor_store_integration.py` (179 Z., 7 Importer) — KEEP
- `insider_features.py` (170 Z., 6 Importer) — KEEP
- `intermarket_factors.py` (268 Z., 6 Importer) — KEEP

**Rest von `features/`:** 34 weitere Files, davon 18 observability-wired. Darunter `news_features.py` (552 Z., OBSERVABILITY), `registry.py` (428 Z., OBSERVABILITY).

### C.5 Signals (9 Kern-Files, ~3.500 Zeilen)
- `rules_trend.py` (491 Z., 27 Importer) — KEEP
- `multifactor_signal.py` (908 Z., 10 Importer) — KEEP
- `meta_model.py` (453 Z., 11 Importer) — KEEP
- `crash_prediction.py` (577 Z., 6 Importer) — KEEP
- `rules_event_insider_shipping.py` (166 Z., 6 Importer) — KEEP (auf Dummy-Daten)
- `hmm_posterior.py` (108 Z., 5 Importer) — KEEP
- `ensemble.py` (297 Z., 4 Importer) — KEEP
- `sector_rotation.py` (269 Z., 4 Importer) — KEEP
- `signal_confidence.py` (188 Z., 3 Importer) — KEEP

**Rest von `signals/`:** 13 weitere Files, darunter die 11 observability-wired und 2 klein. Besonders problematisch: `intel_signal_adapter.py` (421 Z., TRADING_CYCLE_ONLY), `short_signals.py` (328 Z., TRADING_CYCLE_ONLY), `signal_api.py` (334 Z., OBSERVABILITY_WIRING), `behavioral_finance.py` (378 Z., OBSERVABILITY_WIRING).

### C.6 Strategies (7 Kern-Files, ~3.100 Zeilen)
- `multifactor_v2.py` (1.034 Z., 8 Importer) — KEEP
- `multifactor_v1.py` (739 Z., 10 Importer) — **MERGE/DEPRECATE** (zwei parallele Versionen)
- `multifactor_long_short.py` (583 Z., 7 Importer) — KEEP
- `ema_trend_v0.py` (260 Z., 7 Importer) — KEEP
- `base.py` (224 Z., 5 Importer) — KEEP
- `signal_decay_gate.py` (120 Z., 5 Importer) — KEEP
- `ic_decay_weights.py` (124 Z., 4 Importer) — KEEP

**Rest:** `stat_arb.py` flat (DELETE, siehe A.2), `strategy_discovery.py` (OBSERVABILITY).

### C.7 ML (10 Kern-Files, ~5.500 Zeilen, das eigentlich Produktive)
- `feedback_loop.py` (1.667 Z., 6 Importer) — REFACTOR (zu groß)
- `factor_models.py` (1.263 Z., 11 Importer) — REFACTOR
- `regime_hmm.py` (416 Z., 6 Importer) — KEEP
- `explainability.py` (559 Z., 6 Importer) — KEEP
- `news_ml_bridge.py` (402 Z., 6 Importer) — KEEP
- `nlp_sentiment.py` (300 Z., 7 Importer) — KEEP (aber: kein FinBERT-Modell geladen)
- `purged_cv.py` (248 Z., 6 Importer) — KEEP
- `model_registry.py` (249 Z., 6 Importer) — KEEP
- `cpcv.py` (180 Z., 10 Importer) — KEEP
- `adversarial_validation.py` (173 Z., 6 Importer) — KEEP

**Rest von `ml/`:** 45 weitere Files, davon 26 observability-wired. **Der größte einzelne Teil-Abfall im Repo.**

### C.8 Risk (10 Kern-Files, ~4.500 Zeilen)
- `risk_metrics.py` (1.237 Z., 12 Importer) — REFACTOR
- `regime_models.py` (719 Z., 9 Importer) — KEEP
- `state_machine.py` (609 Z., 6 Importer) — KEEP
- `regime_analysis.py` (505 Z., 5 Importer) — KEEP (hat aber viele TODOs, siehe Teil 1)
- `transaction_costs.py` (483 Z., 5 Importer) — KEEP (aber: Duplikation mit `execution/transaction_costs.py`!)
- `factor_exposures.py` (349 Z., 5 Importer) — KEEP
- `trailing_stops.py` (269 Z., 7 Importer) — KEEP
- `exposure_engine.py` (246 Z., 5 Importer) — KEEP
- `liquidity_scoring.py` (219 Z., 7 Importer) — KEEP
- `georisk_overlay.py` (168 Z., 4 Importer) — KEEP

**Rest von `risk/`:** 26 weitere Files, davon 21 observability-wired. **Darunter 2000+ Zeilen, die nicht im Entscheidungspfad sind.**

### C.9 Portfolio (10 Kern-Files, ~3.000 Zeilen)
- `position_sizing.py` (780 Z., 33 Importer) — KEEP (der Hot-Spot für Sizing)
- `black_litterman.py` (391 Z., 7 Importer) — KEEP
- `hierarchical_risk_parity.py` (308 Z., 5 Importer) — KEEP
- `cost_aware_optimizer.py` (256 Z., 6 Importer) — KEEP
- `market_neutral_optimizer.py` (276 Z., 4 Importer) — KEEP
- `stress_test_constraints.py` (249 Z., 4 Importer) — KEEP
- `multiasset_allocator.py` (211 Z., 4 Importer) — KEEP
- `inverse_etf_selector.py` (264 Z., 3 Importer) — KEEP
- `turnover_penalty.py` (177 Z., 4 Importer) — KEEP
- `kelly_uncertainty.py` (127 Z., 4 Importer) — KEEP

**Rest:** 13 weitere Files, 12 observability-wired. Darunter `bl_sizing.py` (386 Z., TRADING_CYCLE_ONLY), `robust_optimizer.py` (315 Z., OBSERVABILITY).

## D. Duplikationen, die quer über Module gehen

### D.1 `transaction_costs` existiert in zwei Modulen
- `execution/transaction_costs.py` (1.008 Zeilen) — REAL_USE, 12 Importer
- `risk/transaction_costs.py` (483 Zeilen) — REAL_USE, 5 Importer

Beide existieren parallel. `execution/` scheint die Implementierung zu sein, `risk/` enthält wahrscheinlich Risk-adjustierte Kosten. Aber der Overlap ist nicht dokumentiert, und wenn jemand "Transaction Costs" sucht, weiß er nicht welche.

**Verdict:** MERGE — einer sollte den anderen nutzen. Klarer Boundary definieren: `execution/` = operative Kosten, `risk/` = Risk-Metric-Kosten.

### D.2 `state_machine` in drei Modulen
- `risk/state_machine.py` (609 Z.)
- `events/crisis_alpha/state_machine.py` (326 Z.)
- (plus weitere in intel/ und anderen)

Das sind drei unabhängige State-Machine-Implementierungen. Jeder macht sein eigenes Ding. **Keine** von ihnen nutzt eine Shared-Base-Class. 

**Verdict:** Ein gemeinsames `utils/state_machine.py` oder `core/state_machine.py` mit einer Base-Class. Die drei Spezialisierungen erben davon.

### D.3 `reconciliation.py` und `reconciliation_report.py` separate Files
- `accounting/reconciliation.py` (395 Z., 18 Importer)
- `accounting/reconciliation_report.py` (491 Z., 8 Importer)

Das ist ok strukturiert (Logik vs. Writer), aber die Namensgebung lässt das nicht erkennen.

### D.4 `fill_model.py` vs `fill_model_pipeline.py`
- `execution/fill_model.py` (967 Z., 17 Importer) — eigentliches Modell
- `execution/fill_model_pipeline.py` (74 Z., 7 Importer) — Wrapper

Der Wrapper ist sinnvoll, aber 74 Zeilen Wrapper für 967 Zeilen Kern-Code ist ein Code-Smell.

### D.5 `backtest.py`, `backtest_legacy.py`, `backtest_engine.py`, `backtest_engine_numba.py`
- `pipeline/backtest.py` (443 Z., REAL_USE) — aktuelle Implementation
- `pipeline/backtest_legacy.py` (120 Z., OBSERVABILITY_WIRING) — **DELETE**
- `qa/backtest_engine.py` (1.434 Z., REAL_USE) — noch eine Implementation
- `qa/backtest_engine_numba.py` (kleiner) — Numba-optimierte Variante
- `qa/backtest_comparison.py` — Vergleichsschicht
- `qa/backtest_overfit.py` — Overfitting-Checks

Das sind **sechs** Backtest-Module. Ein gesundes Projekt hat eines mit optionalen Erweiterungen.

**Verdict:** Konsolidierung: Eine primäre Backtest-Engine (`pipeline/backtest.py`), optional-schnell via Numba, Overfit-Checks als separates QA-Tool. `backtest_legacy.py` löschen.

## E. Strukturelle Anti-Patterns, die ich im Einzel-File-Review gefunden habe

### E.1 `experiments/batch_runner.py` — 1623 Zeilen, OBSERVABILITY_WIRING
- **Schwer** · größtes observability-only File im Repo
- Ein Batch-Runner, der nur von trading_cycle.py und Wiring-Tests referenziert wird
- 1623 Zeilen Code, die im Produktivpfad **nicht laufen**
- **Verdict:** ARCHIVE sofort. Der Code hat Wert, aber im aktuellen Zustand ist er Ballast.

### E.2 `qa/robustness.py` — 1403 Zeilen, OBSERVABILITY_WIRING
- **Schwer** · zweitgrößtes observability-only File
- 1403 Zeilen Robustness-Testing-Code, der nur observability-wired ist
- Robustness ist ein Kern-Thema für Quants — sollte im Entscheidungspfad sein, nicht im Log
- **Verdict:** Entweder in `qa/walk_forward.py` integrieren, oder ARCHIVE

### E.3 `ml/regime_weight_trainer.py` — 564 Zeilen, OBSERVABILITY_WIRING
- **Mittel** · ein Regime-Weight-Trainer wird nie aufgerufen produktiv
- `configs/factor_weights_by_regime.json` ist statisch — vermutlich wurde einmal trainiert, dann handgepflegt
- **Verdict:** Entscheide: entweder aktiv machen (Retrain-Scheduler!), oder ARCHIVE

### E.4 `features/news_features.py` — 552 Zeilen, OBSERVABILITY_WIRING
- **Mittel** · News-Features für Signal-Generierung — aber nicht im Pfad
- Das ist einer der Faktoren, warum deine News-Pipeline keinen Signal-Impact hat
- **Verdict:** Integrieren! Diese Datei hat realen Wert, wenn sie in den Signal-Pfad eingebaut wird

### E.5 `ml/causal_inference.py` — 545 Zeilen, OBSERVABILITY_WIRING
- **Mittel** · Causal Inference ist ein cooles Thema, aber schwer zu operationalisieren
- **Verdict:** ARCHIVE. Wenn du später ernst machst, reaktivieren.

### E.6 `signals/intel_signal_adapter.py` — 421 Zeilen, TRADING_CYCLE_ONLY
- **Schwer** · Der Adapter zwischen Intel-Schicht und Signal-Schicht wird nur observability-gewired
- Das ist einer der Haupt-Gründe, warum deine News-Pipeline keinen Entscheidungseinfluss hat
- **Verdict:** INTEGRIEREN oder DELETEN. Halbe Verdrahtung ist am schlechtesten.

### E.7 `risk/short_risk.py` — 418 Zeilen, TRADING_CYCLE_ONLY
- **Schwer** · Short-Risk-Modul ist nur observability
- `policy.yaml` erlaubt Shorts, aber das Risk-Modul für Shorts wird nicht geprüft
- **Verdict:** Wenn Shorts aktiv werden, MUSS das integriert werden. Aktuell ARCHIVE.

### E.8 `risk/factor_risk_model.py` — 377 Zeilen, TRADING_CYCLE_ONLY
- **Mittel** · Ein Faktor-Risiko-Modell, das nie zu Entscheidungen führt
- **Verdict:** ARCHIVE, bis klar ist, wie es genutzt werden soll

### E.9 `portfolio/bl_sizing.py` — 386 Zeilen, TRADING_CYCLE_ONLY
- **Mittel** · Black-Litterman-Sizing ist trading-cycle-only
- `portfolio/black_litterman.py` ist REAL_USE, aber der Sizing-Wrapper nicht
- **Verdict:** Entweder direkt in `black_litterman.py` integrieren oder ARCHIVE

### E.10 `execution/ibkr_adapter.py` — 342 Zeilen, OBSERVABILITY_WIRING
- **Mittel** · IBKR-Adapter nie real benutzt
- **Verdict:** ARCHIVE bis IBKR aktiv wird. Aktuell Alpaca-Only.

## F. Die `archive/`-Ecke

`archive/intel_research_2026q2/` enthält 10+ Files zum Thema Geopolitik/Strategie:
- `escalation_tracker.py`, `escalation_model.py`
- `multichannel_propagation.py`
- `structural_cycles.py`
- `sensitivity_analysis.py`
- `feedback_loops.py`
- `shock_correlation.py`
- `hegemonic_dynamics.py`
- `analyst_features.py`
- `earnings_call_nlp.py`

Das ist Zeihan/Dalio/Allison-inspirierter Code, der einmal in `src/` war und nach `archive/` verschoben wurde (Stand 2026-04-19 laut Git).

**Verdict:** Das Archive ist das richtige Ziel für Code, den man nicht löschen will, aber der nicht produktiv ist. Der Ordner sollte als Template für weitere Konsolidierung dienen.

## G. Test-Ordner-Struktur

Habe die 733 Test-Files nicht einzeln aufgelistet (würde diesen Audit sprengen), aber die kritischen Cluster:

### G.1 `tests/test_wave*_wiring.py` — 147 Files
Alle folgen dem Pattern aus Teil 1, Sektion 2.1.
**Verdict:** Nach Konsolidierung von `trading_cycle.py` sind alle diese Tests überflüssig und müssen DELETE werden.

### G.2 `tests/regression/` — Golden-Equity-Tests
**Verdict:** KEEP und ERWEITERN auf längere Zeiträume

### G.3 `tests/test_phase*.py`-Muster
**Verdict:** KEEP, aber Marker-Migration fertigstellen (laut `docs/tech_debt/markers_migration.md` bis 2026-07-01)

## H. Scripts-Ordner (95 Top-Level-Files)

Hier gruppiert nach Zweck (die vollständige Einzel-Liste wäre 3 Seiten):

### H.1 Echte CLI-Entries (KEEP)
- `cli.py` — kanonische CLI
- `run_eod_pipeline.py` — EOD-Pipeline
- `run_backtest_strategy.py` — Backtest
- `run_live_paper.py` — Live-Paper-Runner
- `run_paper_track.py` — Paper-Track
- `run_news_worker.py` — News-Worker
- `run_api.py` — FastAPI-Server

### H.2 Redundante/Sprint-Artefakte (DELETE/ARCHIVE)
- `sprint9_backtest.py`, `sprint9_execute.py`, `sprint10_portfolio.py` — DELETE
- `run_grand_backtest.py` — das "Aktiviert-alle-Module"-Script, ARCHIVE (siehe Teil 1, Befund 13.2)
- `run_final_optimized.py` — ARCHIVE (Name suggeriert "final", ist's aber nicht)
- `run_improvement_cycle.py` — ARCHIVE
- `run_daily.py` vs. `run_eod_pipeline.py` vs. `run_daily_scheduler.py` — MERGE auf einen

### H.3 Operative Utilities (KEEP)
- `check_health.py`, `liveness_check.py`
- `ack_halt.py`
- `verify_evidence_pack.py`, `export_evidence_pack.py`
- `import_broker_snapshot.py`
- `snapshot_alpaca_balance.py`
- `train_meta_model.py`, `train_regime_weights.py`

### H.4 Download/Fetch-Scripts (KEEP, aber konsolidieren)
Viele `download_*` und `fetch_*` Scripts, die meiste sollten in die Data-Ingest-Schicht von `src/assembled_core/data/` wandern.

### H.5 Scripts, die ich nicht einordnen kann (30+)
Benötigen individuelle Review. Die Masse ist ein klarer Indikator: Script-Ordner wurde als Dumping-Ground benutzt, nicht als Entry-Point-Collection.

---

# Action-Plan nach diesem Audit

Dieses Dokument hat dir für jede der 551 Python-Files in `src/assembled_core/` einen Verdikt gegeben. Wenn du das jetzt abarbeitest:

## Schritt 1: Sicherheit (Woche 1)
1. `.env`-Keys finally rotieren, falls noch nicht (siehe Teil 1 §10.1)
2. Repo privat setzen
3. `git config user.email` korrekt setzen (weg vom Placeholder)

## Schritt 2: Die Namenskonflikte killen (Woche 1)
1. Einen der zwei `config.py`-Files löschen, alle Imports umziehen
2. `strategies/stat_arb.py` (flat) löschen
3. `logging_config.py`-Duplikate konsolidieren

## Schritt 3: Die observability-Wiring-Schicht zerschlagen (Woche 2-3)
Für jede der ~215 observability-wired Files:
- Entscheidung: ARCHIVE oder DELETE?
- Bei ARCHIVE: Verschiebe nach `archive/observability_graveyard_2026q2/<modul>/`
- Bei DELETE: `git rm` ohne Ersatz
- In einem einzigen Commit pro Modul, damit rückverfolgbar

## Schritt 4: `trading_cycle.py` zerlegen (Woche 4-6)
Der Megamonolith muss in 7 Funktionen à <500 Zeilen aufgeteilt werden:
```
cycle() = 
  ingest_data() →
  build_features() →
  generate_signals() →
  size_positions() →
  check_risk() →
  route_orders() →
  book_fills()
```
Alle 309 Steps einem dieser 7 zuordnen oder entfernen.

## Schritt 5: Die Wiring-Tests entfernen (Woche 6)
Nach Schritt 4 sind die 147 `test_waveN_wiring.py`-Files inkompatibel mit der neuen Struktur. Das ist ok — sie werden nicht mehr gebraucht. Löschen.

## Schritt 6: Eine Strategie End-to-End validieren (Woche 7-10)
Wie in Teil 1 beschrieben: EMA-Trend + News-Overlay, auf echter Historie, mit realen Kosten, mit Position-Sizing ≠ 1.0, auf Alpaca-Paper-Account über mindestens 8 Wochen.

## Schritt 7: Crisis-Alpha gegen 2008/2020 backtesten (Woche 10-11)
Reproduzierbarer Report, ob Crisis-Alpha-State-Machine korrekt getriggert hätte.

## Schritt 8: News → Signal-Pfad schließen (Woche 11-14)
Die Observability-Wiring-Todschleife zwischen News-Pipeline und Signal-Generation schließen. `signals/intel_signal_adapter.py` und `features/news_features.py` aktiv machen.

---

# Schlussbemerkung Teil 2

Was du jetzt in der Hand hast:

- Teil 1 (~1460 Zeilen): **Was** ist schlecht, in 24 thematischen Sektionen
- Teil 2 (dieses Dokument, ~1000 Zeilen): **Wo genau** ist es schlecht, Datei für Datei

Zusammen: rund 2500 Zeilen konkrete, adressbare Befunde.

Das ist der vollständige Audit, den du gefordert hast. Mehr Tiefe würde nicht mehr nützen — der nächste sinnvolle Schritt ist das Tun, nicht das Suchen.

Wenn ich eine einzelne Zahl zusammenfassen müsste, was das Repo heute ist: von 551 Python-Files in `src/assembled_core/` sind
- **rund 180** wirklich produktiv (gut!),
- **rund 215** observability-wired (muss weg oder ins Archiv),
- **rund 156** is_init/utility/edge (sind ok, aber niemand weiß was davon was ist).

Wenn du Schritt 3 (Observability-Schicht zerschlagen) **sauber** durchziehst, schrumpft dein Repo von 1971 auf ~800 Files. Die produktive Substanz bleibt erhalten. Das ist die Operation am offenen Herzen, aber sie ist machbar — und notwendig, bevor du weiterbauen kannst.

Sag mir, ob du als Nächstes:
1. Teil 3 willst (was ich nicht tiefer analysiert habe: Tests im Detail, Scripts im Detail, Docs-Konsolidierung),
2. **oder** ob wir in die konkrete Umsetzung einsteigen (z.B. Schritt 2 oder Schritt 4 des Action-Plans mit mir zusammen durchziehen).

Ich empfehle Option 2. Der Audit ist jetzt lang genug.

---

**Ende Audit Teil 2.**
