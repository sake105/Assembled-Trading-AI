# Top-Level Scripts Index

Generated: 2026-05-17. Maintain manually when adding/removing top-level scripts.

For scripts in subdirectories (`architecture/`, `calibration/`, `ci/`, `commands/`,
`comparison/`, `data/`, `dev/`, `ops/`, `analysis/`), see each subdirectory's README or
`ls scripts/<subdir>/`.

---

## Entry-Point CLIs (installed via pyproject.toml [project.scripts])

- `cli.py` — central dispatcher CLI (assembled-cli)
- `cli_factor_report.py` — factor report CLI runner
- `run_api.py` — CLI entry point for FastAPI server

---

## Backtest Runners

- `run_backtest_strategy.py` — single-strategy backtest with full options (primary runner)
- `batch_backtest.py` — official wrapper for batch_runner; entry point for batch backtests
- `batch_runner.py` — reproducible batch backtests with manifest (P4)
- `run_experiments.py` — systematic parameter experimentation across strategy dimensions
- `run_full_experiments.py` — full experiment suite (Long/Short, News-Sentiment, Multi-Signal, HHI-Fix)
- `run_analysis_1y.py` — 1-year backtest analysis using V1–V20 improvements (Phase 6)
- `run_ab_experiment.py` — A/B experiment runner for paper track
- `ab_compare_strategies.py` — generic A/B strategy comparison with statistical significance
- `compare_strategies_trend_vs_event.py` — compare Trend Baseline vs Event Insider Shipping strategy
- `backtest_pairs_trading.py` — isolated A/B backtest for pairs_trading_v1
- `benchmark_backtest.py` — benchmark harness for backtest performance measurement
- `benchmark_backtest_engine.py` — performance benchmark for backtest engine optimizations
- `benchmark_compare.py` — benchmark comparison report
- `walk_forward_w4.py` — walk-forward stability test (4 rolling windows)
- `run_walk_forward_analysis.py` — walk-forward analysis runner (B3)
- `release_gate_walk_forward.py` — release-gate: walk-forward OOS Sharpe + Deflated Sharpe (E3+E4)
- `forward_test.py` — forward test with known outcomes
- `run_stress_test.py` — stress-test across historical crisis windows
- `run_stress_replay.py` — stress scenario replay runner
- `profile_backtest.py` — cProfile wrapper for backtest performance investigation
- `summarize_backtest_experiments.py` — summarize backtest experiment runs
- `leaderboard.py` — rank and display best runs from batch backtest results
- `build_golden_equity_baseline.py` — build the golden equity baseline JSON (B0)
- `compare_equity_curves.py` — equity-curve regression diff (B0)

---

## Paper Trading

- `run_paper_track.py` — paper track runner, daily execution script
- `run_paper_pilot.py` — 30-day paper-live pilot, pre-flight check for live trading
- `run_paper_live.py` — paper-trading live runner, real-time Alpaca bars
- `run_live_paper.py` — live paper trading runner via Alpaca broker API
- `evaluate_pilot_v2.py` — pilot GO/NO-GO evaluation (Backlog 129/148)
- `daily_pilot_review.py` — generates structured markdown summary for paper pilot monitoring
- `smoke_test_paper.py` — 72-hour pre-paper-pilot smoke test; validates full stack integration
- `paper_trading_scheduler.py` — autonomous paper trading scheduler
- `run_daily_scheduler.py` — CLI runner for the autonomous daily operations cycle
- `run_preflight_checks.py` — pre-flight check automation script (Checks 1–6)
- `snapshot_alpaca_balance.py` — Alpaca EOD balance snapshot (A4)
- `import_broker_snapshot.py` — CLI tool for importing external broker snapshots
- `inspect_paper_track_data.py` — inspect available price data for paper track runner
- `validate_backtest_paper_parity.py` — backtest vs paper-trade parity check
- `report_shadow_delta.py` — shadow vs applied delta reporter (Part D)
- `daily_paper_trading.bat` — Windows Task Scheduler entry point for daily paper trading
- `run_paper_scheduler.bat` — alternative batch scheduler entry point

---

## EOD / Daily Pipeline

- `run_daily.py` — EOD-MVP runner: daily order generation (core pipeline)
- `run_eod_pipeline.py` — EOD pipeline orchestration script
- `run_daily.py` — EOD-MVP runner (also listed above; primary entry point)

---

## Data Ingest / Refresh

- `download_all_market_data.py` — download all free market data (FRED macro, fundamentals, earnings, …)
- `download_all_universes_robust.py` — robust batch download for all universe tickers
- `download_master_universe_data.py` — bulk yfinance downloader for full_us_universe.yaml
- `download_historical_snapshot.py` — download historical price snapshot for one or multiple symbols
- `download_altdata_finnhub_events.py` — download earnings and insider events from Finnhub API
- `download_altdata_finnhub_news_macro.py` — download news/sentiment/macro from Finnhub API
- `fetch_earnings_calendar.py` — refresh cached earnings calendar for paper cycle
- `fetch_missing_data.py` — fetch missing market data (SH, PSQ, VIX, VIX3M, UUP, …)
- `fetch_news_alphavantage.py` — Alpha Vantage news & sentiment fetcher
- `fetch_news_newsapi.py` — NewsAPI.org fetcher for company/ticker news
- `fetch_news_polygon.py` — Polygon.io ticker news fetcher
- `fetch_real_daily.py` — fetch real daily OHLCV from Yahoo Finance for runner universe
- `update_prices.py` — update local price cache from Polygon or yfinance
- `backfill_news_sentiment_gdelt.py` — backfill news_sentiment_daily.parquet using free GDELT 2.0 API
- `fuse_news_sentiment.py` — fuse news sentiment from all sources into unified parquet
- `convert_rss_events_to_sentiment.py` — convert RSS events to sentiment format
- `run_rss_fetch.py` — standalone RSS feed fetcher; saves to JSON
- `build_pre2020_panel.py` — download pre-2020 EOD data and merge into extended watchlist panel
- `prewarm_factor_store.py` — offline factor-store pre-warm (C2)
- `seed_questdb_from_csv.py` — seed QuestDB tick store from historical CSV/Parquet price data
- `00_seed_demo_data.py` — one-time zero-data seeder for a fresh project after a move

---

## PowerShell Download Helpers

- `DOWNLOAD_COMMANDS_OPTIMIZED.ps1` — optimized download commands batch
- `RUN_DOWNLOADS_ALL_UNIVERSES.ps1` — run downloads for all universes
- `download_all_altdata_9h.ps1` — download all alt-data with 9h window
- `download_all_universes.ps1` — download all universes
- `download_all_universes_batch.ps1` — batch variant
- `download_all_universes_safe.ps1` — safe/throttled variant
- `download_all_universes_twelve_data.ps1` — Twelve Data API variant
- `download_all_universes_with_long_delays.ps1` — variant with long delays for rate-limited APIs
- `download_macro_etfs_conservative.ps1` — macro ETF download (conservative rate)
- `download_missing_symbols_sequential.ps1` — sequential fetch for missing symbols
- `download_one_by_one_robust.ps1` — one-by-one robust fetch
- `download_one_by_one_safe.ps1` — one-by-one safe fetch
- `download_single_symbol.ps1` — single symbol download
- `check_all_universes_completeness.ps1` — check completeness across all universe downloads
- `test_problem_symbols.ps1` — test/debug problem symbols

---

## Audits & Checks

- `audit_dependencies.py` — dependency audit script (Plan 11.10)
- `audit_trading_cycle_dead_imports.py` — audit dead imports in trading_cycle.py
- `check_data_completeness.py` — check completeness and quality of downloaded historical data
- `check_data_sources_health.py` — data-source health check (audit C3-061)
- `check_health.py` — health check script for backend operations
- `check_phantom_imports.py` — finds imports of archived/missing modules
- `check_scheduler_health.py` — scheduler heartbeat health monitor (A2)
- `health_check.py` — health check script (Plan 11.3)
- `liveness_check.py` — liveness check CLI (Plan C16)
- `detect_secrets_baseline_diff.py` — diff detect-secrets scan output against committed baseline
- `run_system_check.py` — CLI entry point for the System-Check Tournament
- `validate_altdata_snapshot.py` — validate alt-data snapshot directory
- `validate_download.py` — quick validation of downloaded Parquet files
- `validate_cpcv.py` — combinatorial purged cross-validation for strategy validation
- `validate_edcl_conviction.py` — validate EDCL conviction score distribution

---

## Factor / Signal Analysis

- `run_factor_analysis.py` — factor analysis CLI runner
- `run_factor_analysis_smoketests.py` — end-to-end smoketests for factor analysis pipeline
- `run_ml_factor_validation.py` — ML factor validation CLI runner
- `summarize_factor_rankings.py` — summarize factor rankings from multiple analysis outputs
- `compute_signal_decay_profile.py` — offline signal-decay profile writer (D5/R4)
- `run_leakage_audit.py` — leakage audit for ML features; saves JSON report
- `run_validation_and_drift_checks.py` — validation and drift checks on ML datasets
- `explain_trade.py` — why did the system trade X? (Plan 11/10 §5.1.2)
- `performance_attribution.py` — performance attribution report (Backlog 29)
- `run_post_trade_analysis.py` — post-trade analysis runner (M11)
- `generate_tca_report.py` — generate transaction cost analysis (TCA) report

---

## ML Training

- `train_meta_model.py` — train meta-model (Plan W7 / Sprint 2)
- `train_regime_weights.py` — train regime-conditional factor weights (Plan B3.3 / Sprint 3)
- `train_rl_executor.py` — train RL order-execution agent using PPO

---

## Worker Processes (long-running / background)

- `run_news_worker.py` — NEWS v1 worker (M1-T13)
- `run_kill_switch_worker.py` — kill switch worker (M4 Execution Workers / Ops v1)
- `run_stop_worker.py` — STOP worker (M4 Execution Workers / Ops v1)
- `run_reconcile_worker.py` — RECONCILE worker (M4 Execution Workers / Ops v1)
- `run_disclosures_worker.py` — DISCLOSURES v1 worker entry point (M2)
- `run_crisis_alpha_worker.py` — Crisis-Alpha v1 worker (M5)
- `run_intel_cycle.py` — intel cycle runner; fetches GDELT every 15 minutes, updates crisis state

---

## Risk & Ops

- `ack_halt.py` — manual halt acknowledgement CLI (E0.4 / A2)
- `release_sanity_halt.py` — release a sanity-halted order (Plan 11/10 §5.2.3)
- `run_alert_drill.py` — weekly synthetic alert-drill (P1 A13)
- `generate_risk_report.py` — generate risk report from backtest results
- `quantify_realism_delta.py` — post-E0 Sharpe-drop quantification (E1)
- `compare_real_vs_synthetic_fills.py` — real-vs-synthetic fill calibration (E5)
- `run_cost_calibration.py` — offline cost-model calibration runner (E5-loop)
- `cleanup_old_outputs.py` — storage cleanup (Backlog 71)
- `backup_databases.py` — backup DuckDB and SQLite databases

---

## Reporting & QA

- `generate_daily_qa_report.py` — daily QA report: Bayesian Sharpe + risk-parity weights + LLM-RAG news digest
- `generate_performance_profile_report.py` — generate performance profile report from profiling outputs
- `generate_review_bundle.py` — create single text file with project structure and file contents
- `generate_risk_report.py` — generate risk report from backtest results (also listed under Risk & Ops)
- `export_evidence_pack.py` — CLI tool for exporting evidence packs (Sprint 13)
- `verify_evidence_pack.py` — validate an evidence pack ZIP offline
- `daily_decision_log.py` — daily decision log (Item 103)
- `news_coverage_report.py` — news coverage report: feed count by tier and focus category
- `run_premarket_digest.py` — pre-market news digest generator (Point 34)

---

## Profiling

- `profile_backtest.py` — cProfile wrapper for backtest (also listed under Backtest)
- `profile_job.py` — profile reference benchmark jobs (EOD_SMALL, BACKTEST_MEDIUM, ML_JOB)
- `profile_jobs.py` — profile common jobs (backtests, factor+ML runs, playbooks)
- `memory_profile.py` — memory profiling script for the pilot (Item 27)

---

## Event Studies / Disclosures

- `run_event_study.py` — event study CLI workflow
- `run_disclosure_event_study.py` — disclosure event study (T6.2)
- `generate_sample_event_data.py` — generate sample event data matching existing price sample data

---

## Demo / Seed / Dev Utilities

- `00_seed_demo_data.py` — one-time zero-data seeder for fresh project (also listed under Data Ingest)
- `generate_demo_daily.py` — generate synthetic daily price data for paper track demo/testing
- `plot_equity_drawdown.py` — equity curve + drawdown visualization (2-panel)
- `set_gh_secrets.py` — set GitHub Actions secrets via REST API using stored git credential
- `regenerate_agents_stats.py` — regenerate AGENTS.md repo statistics (C5)
- `test_pipeline_integration.py` — test pipeline integration with downloaded alt-data

---

## PowerShell Pipeline / Operations

- `run_live_pipeline.ps1` — run live pipeline
- `run_all_sprint10.ps1` — run all Sprint 10 tasks
- `run_phase4_tests.ps1` — run Phase 4 tests
- `setup_pipeline_integration.ps1` — set up pipeline integration
- `start_pilot_v2.ps1` — start pilot v2

---

## Other / One-Off

- `debug_event_signals.py` — debug script for event signal generation (German docstring; categorized here)
- `run_disclosure_event_study.py` — (also listed under Event Studies)
- `walk_forward_w4.py` — (also listed under Backtest Runners)

> If this section grows large, add a new category above.
