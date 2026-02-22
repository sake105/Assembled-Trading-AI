# Strategy Current Behavior (from code)

ASCII-only. File pointers: module + function names. No speculation.

---

## 1. Repo inventory

### 1.1 src/assembled_core

| Subsystem | Purpose |
|-----------|--------|
| accounting/ | Ledger, broker_snapshot, evidence_index, evidence_pack, reconciliation_report, accounting_report, ledger_integration, position_engine |
| pipeline/ | orchestrator, trading_cycle, backtest, backtest_legacy, portfolio, orders, signals, io, precomputed_index |
| qa/ | backtest_engine, backtest_engine_numba, candidate_gate, metrics, robustness, walk_forward, tca, data_qc, numba_kernels, health, dataset_builder, labeling, factor_analysis, ml_evaluation, qa_gates, scenario_engine, shipping_risk |
| execution/ | order_generation, transaction_costs, fill_model, fill_model_pipeline, pre_trade_checks, risk_controls, safe_bridge, position_alignment |
| features/ | ta_features, ta_factors_core, ta_liquidity_vol_factors, registry, factor_store_integration, event_features, insider_features, congress_features, macro_features |
| strategies/ | multifactor_long_short (MultiFactorStrategyConfig, generate_multifactor_long_short_signals, compute_multifactor_long_short_positions) |
| signals/ | rules_trend, rules_event_insider_shipping, multifactor_signal, signal_api, ensemble, meta_model |
| portfolio/ | position_sizing (compute_target_positions, compute_target_positions_from_trend_signals) |
| risk/ | transaction_costs, risk_metrics, regime_analysis, regime_models, factor_exposures, exposure_engine, group_exposures |
| config/ | settings, models, factor_bundles, config, constants |
| reports/ | metrics_export, daily_qa_report |
| paper/ | paper_track, strategy_adapters |
| experiments/ | batch_runner, batch_config |
| api/ | app, routers (qa, performance, risk, portfolio, monitoring) |
| data/ | factor_store, panel_store, prices_ingest, security_master, etc. |
| costs/ | CostModel, get_default_cost_model (used by backtest_engine) |

### 1.2 scripts/

| Script | Purpose |
|--------|--------|
| run_backtest_strategy.py | CLI: backtest with trend_baseline, event_insider_shipping, multifactor_long_short; --freq, --universe, --price-file, --start-capital, --commission-bps, --spread-w, --generate-report, --write-evidence-pack |
| run_eod_pipeline.py | EOD pipeline: --freq, --universe, --price-file, --start-date, --end-date, --data-source (local/yahoo), --skip-backtest/portfolio/qa |
| run_daily.py | Daily run orchestration; watchlist, broker snapshot |
| cli.py | Unified CLI: run_backtest, run_phase4_tests, walk_forward, etc. |
| export_evidence_pack.py, verify_evidence_pack.py | Evidence pack export/verify |
| dev/run_checks.py | py_compile, ruff, pytest presets (evidence_pack, ops_evidence, broker_snapshot, accounting, release_sprint13) |
| dev/release_sprint13.py | Merge-gate: run_checks + evidence_pack preset |

### 1.3 tests/ (grouped by subsystem)

- accounting: test_accounting_*, test_reconcile_*, test_broker_snapshot_*, test_ledger_*, test_ops_evidence_pack_e2e, test_evidence_*
- pipeline/backtest: test_backtest_*, test_pipeline_*
- qa: test_qa_*, test_deflated_sharpe_*, test_robustness_*
- execution: test_execution_*, test_fill_model_*
- signals/strategies: test_signals_*, test_strategies_*
- config/features: test_config_*, test_features_*
- api: test_api_*
- reports: test_reports_*
- smoke/CI: test_ci_workflows_inventory_smoke, test_docs_*, test_paths_posix_*, test_verify_evidence_pack_*

---

## 2. Hard invariants (from code/docs)

- **Evidence index paths schema** (src/assembled_core/accounting/evidence_index.py): PATHS_KEYS = ("broker_snapshot_path", "ledger_pack_path", "reconcile_report_path", "accounting_report_path", "manifest_path"). All keys always present (value None if missing). JSON: sort_keys=True, indent=2, trailing newline. as_of_date date-only (YYYY-MM-DD).
- **Relative POSIX paths**: Ledger integration and evidence writers use .as_posix() / relative_to(output_dir). No backslashes in outputs (Windows-safe).
- **Atomic writes**: Ledger/orchestrator use temp file + replace where documented. Factor store _write_parquet_atomic.
- **Compile excludes SSOT**: scripts/dev/run_checks.py _COMPILE_EXCLUDE_SUBDIRS, _COMPILE_EXCLUDE_NAMES. tests/test_compile_excludes_smoke.py asserts them.
- **Optional test deps**: pytest.importorskip("scipy"/"fastapi"/"sklearn") in affected tests; robustness.py scipy optional.

---

## 3. Strategy logic (from code)

### 3.1 Where strategy is defined

- **Trend baseline**: signals/rules_trend.py `generate_trend_signals_from_prices` (EMA crossover: ma_fast > ma_slow => LONG). portfolio/position_sizing.py `compute_target_positions_from_trend_signals`.
- **Event insider/shipping**: signals/rules_event_insider_shipping.py `generate_event_signals` (insider net buy/sell thresholds, shipping congestion thresholds). Used via run_backtest_strategy.py strategy "event_insider_shipping".
- **Multi-factor long/short**: strategies/multifactor_long_short.py
  - `generate_multifactor_long_short_signals`: factors from bundle YAML, build_multifactor_signal, select_top_bottom by quantile.
  - `compute_multifactor_long_short_positions`: target weights by quantile, optional regime overlay (max_gross_exposure, target_net_exposure per regime).

Entrypoint: scripts/run_backtest_strategy.py (--strategy trend_baseline | event_insider_shipping | multifactor_long_short), which builds signal_fn and position_sizing_fn and calls qa/backtest_engine.run_portfolio_backtest.

### 3.2 Universe selection

- Settings (config/settings.py): default_universe list (e.g. ["AAPL", "MSFT", "GOOGL"]), watchlist_file (default base_dir / "watchlist.txt").
- run_backtest_strategy.py / run_eod_pipeline.py: --universe <path> or settings.watchlist_file. Prices: load_eod_prices_for_universe(universe_path) or load_eod_prices(..., universe=...) (data/prices_ingest).
- Multi-factor: bundle YAML defines universe (config/factor_bundles: FactorBundle.universe).

### 3.3 Position sizing

- portfolio/position_sizing.py:
  - `compute_target_positions(signals, total_capital, top_n, equal_weight)`: LONG-only; equal weight 1/N or score-based; optional top_n by score.
  - `compute_target_positions_from_trend_signals(signals, total_capital, top_n, min_score)`.
- strategies/multifactor_long_short: positions from quantile weights, max_gross_exposure cap, optional regime_risk_map (max_gross_exposure, target_net_exposure per regime).

### 3.4 Transaction costs / slippage

- execution/transaction_costs.py: CommissionModel (bps, fixed, bps_plus_fixed), SpreadModel, SlippageModel; add_cost_columns_to_trades.
- pipeline/backtest.py: _simulate_fills_per_order(spread_w, impact_w, commission_bps); optional use_numba.
- qa/backtest_engine: uses CostModel (costs/), commission_bps, spread_w, impact_w; simulate_with_costs (pipeline/portfolio); add_cost_columns_to_trades on trades.
- risk/transaction_costs.py: also present (commission, slippage, spread).

### 3.5 Execution model

- execution/order_generation.py: generate_orders_from_targets, generate_orders_from_targets_fast (target vs current positions => BUY/SELL orders). Orders: timestamp, symbol, side, qty, price.
- pipeline/backtest.py: simulate_equity (order execution bar-by-bar); fills at order price with optional spread/impact/commission. Next-bar execution (orders at timestamp T filled at T).
- Fill model: execution/fill_model.py; trades get fill_qty, fill_price, commission_cash, spread_cash, slippage_cash.

---

## 4. What can we tune without changing concept

- **Trend baseline**: ma_fast, ma_slow (rules_trend), top_n, min_score (position_sizing_from_trend).
- **Event strategy**: insider_weight, shipping_weight, insider_net_buy_threshold, insider_net_sell_threshold, shipping_congestion_low/high_threshold.
- **Multi-factor**: bundle_path, top_quantile, bottom_quantile, rebalance_freq (M/W/D), max_gross_exposure, max_leverage, transaction_cost_bps, use_regime_overlay, regime_config, regime_risk_map.
- **Global backtest**: start_capital, commission_bps, spread_w, impact_w (run_backtest_strategy.py and backtest_engine); use_numba (settings).
- **Universe**: --universe file or default_universe / watchlist_file.
- **Data range**: --start-date, --end-date (run_eod_pipeline / run_backtest_strategy where supported).

---

## 5. Data locations (from settings/code)

- data_dir: base/data. output_dir: base/output. sample_data_dir: base/data/sample. sample_eod_file: base/data/sample/eod_sample.parquet. sample_events_dir: base/data/sample/events. watchlist_file: base/watchlist.txt. local_data_root: optional override for price parquet root.
