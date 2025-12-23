# Use Cases & Roles - Assembled Trading AI Backend

**Last Updated:** 2025-01-XX  
**Status:** Overview of Backend Capabilities by Role

---

## Overview

The Assembled Trading AI backend provides a comprehensive research and trading infrastructure that serves multiple roles in a quantitative trading organization. From a **Quant Portfolio Manager** perspective, the system enables systematic strategy evaluation, risk assessment, and performance attribution through backtests, walk-forward analysis, and regime-aware risk reports. The backend supports decision-making with transaction cost analysis (TCA), deflated Sharpe ratios for multiple testing protection, and comprehensive factor ranking.

For **Quant Researchers**, the platform offers extensive factor development capabilities, including technical analysis factors (Phase A), alternative data integration (Phase B), and systematic factor evaluation through IC analysis, portfolio-based factor ranking, and ML model validation. Researchers can run model zoo comparisons, validate factors across multiple time periods, and export factor panels for further analysis.

**Quant Dev/Backend Engineers** leverage the modular architecture to extend functionality, integrate new data sources, and optimize performance. The system provides deterministic backtests (B1), point-in-time safety for alt-data (B2), walk-forward frameworks (B3), and batch processing capabilities (P4) for systematic testing. The backend emphasizes reproducibility, testability, and clear separation of concerns.

**Data Engineers** work with the data ingestion layer, managing price data (EOD and intraday), alternative data sources (insider trading, earnings, news, shipping), and ensuring data quality through validation and QC checks. The system supports local-first data storage (Parquet/CSV), configurable data sources, and structured data contracts for factor panels and backtest inputs.

---

## Roles & Goals

### Quant Portfolio Manager (PM)

**Role Description:**  
The Quant PM uses the backend to evaluate trading strategies, assess risk-adjusted returns, and make deployment decisions. The PM focuses on production-ready strategies with robust out-of-sample validation, regime-aware performance, and realistic cost assumptions.

**Typical Goals:**
- Evaluate strategy performance across different market regimes
- Assess risk-adjusted returns (Sharpe, Sortino, Max Drawdown)
- Validate strategies with walk-forward analysis (out-of-sample stability)
- Understand transaction costs and their impact on net returns
- Compare multiple strategy variants (batch backtests)
- Generate risk reports for compliance and reporting

**Wichtigste Artefakte/Ergebnisse:**
- Backtest Reports (`output/performance_report_*.md`)
- Risk Reports (`output/risk_report_*.md`)
- TCA Reports (`output/tca_report_*.md`)
- Walk-Forward Summaries (`output/walk_forward_*.md`)
- Portfolio Equity Curves (`output/portfolio_equity_*.csv`)
- Batch Backtest Results (`output/batch_backtests/`)

---

### Quant Researcher

**Role Description:**  
The Quant Researcher develops and evaluates new factors, tests ML models, and conducts systematic research on factor effectiveness. The researcher uses IC analysis, portfolio-based evaluation, and ML validation to identify promising factors and model configurations.

**Typical Goals:**
- Develop new factors (TA, Alt-Data, Market Breadth)
- Evaluate factor predictive power (IC, Rank-IC, IC-IR)
- Test factor portfolios (Long/Short, Quantile-based)
- Validate ML models on factor panels
- Compare multiple models (Model Zoo)
- Protect against Factor Zoo (Deflated Sharpe, Multiple Testing)
- Export factor panels for further analysis

**Wichtigste Artefakte/Ergebnisse:**
- Factor Reports (`output/factor_reports/`)
- Factor Ranking Tables (`output/factor_ranking_*.csv`)
- ML Validation Reports (`output/ml_validation/`)
- Model Zoo Summaries (`output/ml_model_zoo/`)
- Factor Panels (`output/factor_panels/`)
- IC Analysis Results (CSV/Markdown)

---

### Quant Dev/Backend Engineer

**Role Description:**  
The Quant Dev/Backend Engineer extends the system, integrates new features, and ensures code quality and performance. The engineer works with the modular architecture, implements new strategies, optimizes backtests, and maintains the infrastructure.

**Typical Goals:**
- Implement new strategies (signal functions, position sizing)
- Extend factor libraries (Phase A, B, C)
- Optimize backtest performance (P3, P4)
- Integrate new data sources
- Ensure deterministic backtests (B1)
- Implement point-in-time safety (B2)
- Add new validation frameworks (B3, B4)

**Wichtigste Artefakte/Ergebnisse:**
- Source Code (`src/assembled_core/`)
- Test Suites (`tests/`)
- Design Documents (`docs/*_DESIGN.md`)
- Performance Profiles (`output/profiling/`)
- Batch Runner Results (`output/batch_backtests/`)
- Integration Test Results

---

### Data Engineer

**Role Description:**  
The Data Engineer manages data ingestion, quality, and storage. The engineer ensures data availability, validates data contracts, and maintains data pipelines for prices, alternative data, and factor panels.

**Typical Goals:**
- Ingest price data (EOD, intraday) from multiple sources
- Manage alternative data (insider trading, earnings, news, shipping)
- Validate data quality and contracts
- Ensure point-in-time safety for alt-data
- Export factor panels for research
- Maintain data storage (Parquet/CSV)

**Wichtigste Artefakte/Ergebnisse:**
- Price Data Files (`data/raw/`, `output/aggregates/`)
- Alt-Data Files (`output/altdata/`)
- Factor Panels (`output/factor_panels/`)
- Data Validation Reports
- Data Source Configuration

---

## Use Cases by Role

### Quant Portfolio Manager

| Use-Case | Inputs | Aktion (CLI/Script) | Outputs/Reports |
|----------|--------|-------------------|-----------------|
| **Single Strategy Backtest** | Universe file, price data, strategy config | `python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt --start-capital 100000` | `output/performance_report_1d.md`, `output/portfolio_equity_1d.csv`, `output/orders_1d.csv` |
| **Batch Strategy Comparison** | Batch config YAML (multiple strategies) | `python scripts/cli.py batch_backtest --config config/batch_strategies.yaml` | `output/batch_backtests/summary.csv`, individual backtest results per strategy |
| **Walk-Forward Analysis** | Price data, strategy config, walk-forward config | `python scripts/cli.py walk_forward --freq 1d --train-size 252 --test-size 63 --step-size 21` | `output/walk_forward_*.md`, OOS metrics per window, stability analysis |
| **Risk Report Generation** | Backtest results (equity curve, trades) | `python scripts/cli.py risk_report --freq 1d --benchmark SPY` | `output/risk_report_1d.md`, VaR, CVaR, volatility, drawdown analysis |
| **TCA Report** | Backtest results (trades, prices) | `python scripts/cli.py tca_report --freq 1d --commission-bps 0.5 --spread-w 0.3` | `output/tca_report_1d.md`, cost breakdown, impact analysis |
| **Regime-Aware Risk Analysis** | Backtest results, regime classification | `python scripts/cli.py risk_report --freq 1d --regime-analysis` | `output/risk_report_1d.md` with regime-specific metrics (Bull/Bear/Sideways) |
| **Paper-Track Evaluation** | Backtest results, paper-track metrics | Manual evaluation using Paper-Track-Playbook criteria | Paper-Track-Report, Gate-Decision-Log. Details zu Kriterien und Ablauf siehe [Paper-Track-Playbook](PAPER_TRACK_PLAYBOOK.md) |

---

### Quant Researcher

| Use-Case | Inputs | Aktion (CLI/Script) | Outputs/Reports |
|----------|--------|-------------------|-----------------|
| **Factor Report (IC Analysis)** | Price data, universe, factor set | `python scripts/cli.py factor_report --freq 1d --symbols-file config/universe_ai_tech_tickers.txt --factor-set core --fwd-horizon-days 20` | `output/factor_reports/`, IC/Rank-IC summaries, factor ranking |
| **Comprehensive Factor Analysis** | Price data, universe, factor set | `python scripts/cli.py analyze_factors --freq 1d --symbols-file config/universe_ai_tech_tickers.txt --factor-set core+vol_liquidity` | `output/factor_analysis/`, IC + Portfolio summaries, deflated Sharpe |
| **ML Model Validation** | Factor panel (Parquet), model config | `python scripts/cli.py ml_validate_factors --factor-panel-file output/factor_panels/ai_tech_core_1d.parquet --label-col fwd_return_20d --model-type ridge` | `output/ml_validation/`, portfolio metrics, IC, test R², deflated Sharpe |
| **Model Zoo Comparison** | Factor panel, multiple model configs | `python scripts/cli.py ml_model_zoo --factor-panel-file output/factor_panels/ai_tech_core_1d.parquet --label-col fwd_return_20d` | `output/ml_model_zoo/`, model comparison summary (CSV/Markdown), deflated Sharpe per model |
| **Factor Panel Export** | Price data, universe, factor set | `python research/factors/export_factor_panel_for_ml.py --freq 1d --symbols-file config/universe_ai_tech_tickers.txt --factor-set core+vol_liquidity --horizon-days 20` | `output/factor_panels/`, Parquet file with factors + forward returns |
| **Factor Ranking** | Factor analysis results | (Automatic in `analyze_factors`) | `output/factor_ranking_*.csv`, sorted by deflated Sharpe, IC-IR, or other metrics |

---

### Quant Dev/Backend Engineer

| Use-Case | Inputs | Aktion (CLI/Script) | Outputs/Reports |
|----------|--------|-------------------|-----------------|
| **Run Full EOD Pipeline** | Universe, price data, config | `python scripts/cli.py run_daily --date 2025-01-15 --universe watchlist.txt` | Complete pipeline: orders, backtest, portfolio, QA reports |
| **Performance Profiling** | Backtest config | `python scripts/cli.py run_backtest --freq 1d --profile` | `output/profiling/`, performance profiles for optimization |
| **Batch Backtest Optimization** | Batch config, optimization targets | `python scripts/cli.py batch_backtest --config config/batch_optimization.yaml --parallel` | `output/batch_backtests/`, parallel execution results |
| **Integration Testing** | Test suite | `python scripts/cli.py run_phase4_tests` | Test results, coverage reports |
| **Factor Store Operations** | Factor data, metadata | (Via `src/assembled_core/features/` modules) | Factor store (P2), cached factors, metadata |
| **Point-in-Time Validation** | Alt-data, price data, as_of dates | (Via `build_*_factors(..., as_of=...)` APIs) | PIT-safe factors, validation reports |

---

### Data Engineer

| Use-Case | Inputs | Aktion (CLI/Script) | Outputs/Reports |
|----------|--------|-------------------|-----------------|
| **Ingest EOD Price Data** | Data source (Yahoo, Alpha Vantage, local) | `python scripts/live/pull_eod.ps1` or via data source config | `data/raw/1d/*.parquet`, `output/aggregates/1d.parquet` |
| **Ingest Intraday Price Data** | Data source, symbols, date range | `python scripts/live/pull_intraday.ps1 --symbols AAPL,MSFT --start-date 2025-01-01` | `data/raw/1min/*.parquet`, `output/aggregates/5min.parquet` |
| **Resample Intraday Data** | 1min Parquet files | (Automatic in pipeline) | `output/aggregates/5min.parquet` (resampled from 1min) |
| **Load Alt-Data (Insider/Earnings)** | Alt-data files, price data | (Via `build_earnings_surprise_factors()`, `build_insider_activity_factors()`) | Alt-data factors in factor panels |
| **Export Factor Panel** | Price data, factor set | `python research/factors/export_factor_panel_for_ml.py --freq 1d --factor-set core` | `output/factor_panels/`, Parquet file with factors + forward returns |
| **Validate Data Contracts** | Data files, expected schema | (Via validation in ingest scripts) | Validation reports, error logs |

---

## Component Map

| Rolle | Kern-Workflow | Wichtige Scripts/Module | Relevante Docs |
|-------|--------------|------------------------|----------------|
| **Quant PM** | Strategy Evaluation -> Risk Assessment -> Deployment Decision | `scripts/cli.py run_backtest`, `batch_backtest`, `risk_report`, `tca_report`, `walk_forward` | `BACKTEST_B1_UNIFIED_PIPELINE_DESIGN.md`, `WALK_FORWARD_AND_REGIME_B3_DESIGN.md`, `TRANSACTION_COSTS_E4_DESIGN.md`, `RISK_2_0_D2_DESIGN.md` |
| **Quant Researcher** | Factor Development -> IC Analysis -> Portfolio Evaluation -> ML Validation | `scripts/cli.py factor_report`, `analyze_factors`, `ml_validate_factors`, `ml_model_zoo`, `research/factors/export_factor_panel_for_ml.py` | `ADVANCED_ANALYTICS_FACTOR_LABS.md`, `ML_VALIDATION_E1_DESIGN.md`, `ML_ALPHA_E3_DESIGN.md`, `DEFLATED_SHARPE_B4_DESIGN.md`, `ML_VALIDATION_EXPERIMENTS.md` |
| **Quant Dev/Backend** | Feature Development -> Testing -> Integration -> Optimization | `src/assembled_core/`, `tests/`, `scripts/cli.py run_phase4_tests`, `scripts/run_all_sprint10.ps1` | `BACKEND_ROADMAP.md`, `ARCHITECTURE_BACKEND.md`, `BACKEND_MODULES.md`, `BATCH_BACKTEST_P4_DESIGN.md`, `PERFORMANCE_PROFILING_P1_DESIGN.md`, `BACKTEST_OPTIMIZATION_P3_DESIGN.md` |
| **Data Engineer** | Data Ingestion -> Validation -> Storage -> Export | `scripts/live/pull_*.ps1`, `src/assembled_core/data/`, `research/factors/export_factor_panel_for_ml.py` | `DATA_SOURCES_BACKEND.md`, `POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md`, `FACTOR_STORE_P2_DESIGN.md`, `eod_pipeline.md` |

---

## Open Questions & Future Work

**Web-Frontend & APIs:**
- Web-Frontend for Signal API (Phase 10+): Interactive dashboard for strategy monitoring, signal visualization, and order management
- REST API for Backtest Execution: Programmatic access to backtest engine via HTTP API
- Real-time Monitoring Dashboard: Live performance metrics, risk alerts, and regime indicators

**Simplified Research Playbooks:**
- One-Click Factor Research: Automated workflow from factor idea to portfolio evaluation
- Automated Model Selection: Auto-select best model from Model Zoo based on deflated Sharpe and OOS metrics
- Strategy Comparison Dashboard: Visual comparison of multiple strategies with regime-aware performance

**Advanced Analytics:**
- Interactive Factor Correlation Matrix: Web-based visualization of factor relationships
- Regime Transition Alerts: Automated notifications when market regime changes
- Factor Decay Analysis: Time-series analysis of factor predictive power over time

**Infrastructure:**
- Distributed Batch Processing: Scale batch backtests across multiple machines
- Incremental Backtest Updates: Only recompute changed periods
- Factor Store with Versioning: Track factor changes over time, rollback capabilities

**Integration:**
- Broker API Integration: Connect to live trading APIs (beyond SAFE-Bridge CSV)
- External Data Source Plugins: Easier integration of new data providers
- ML Model Registry: Centralized storage and versioning of trained models

---

## References

**Design Documents:**
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Complete roadmap for Phase A-E
- [Backtest B1 Design](BACKTEST_B1_UNIFIED_PIPELINE_DESIGN.md) - Deterministic backtests
- [Point-in-Time B2 Design](POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md) - PIT-safe alt-data
- [Walk-Forward B3 Design](WALK_FORWARD_AND_REGIME_B3_DESIGN.md) - Out-of-sample validation
- [Deflated Sharpe B4 Design](DEFLATED_SHARPE_B4_DESIGN.md) - Multiple testing protection
- [Batch Backtest P4 Design](BATCH_BACKTEST_P4_DESIGN.md) - Systematic strategy comparison
- [TCA E4 Design](TRANSACTION_COSTS_E4_DESIGN.md) - Transaction cost analysis
- [Risk 2.0 D2 Design](RISK_2_0_D2_DESIGN.md) - Advanced risk metrics

**Workflow Documentation:**
- [ML Validation Experiments](ML_VALIDATION_EXPERIMENTS.md) - ML validation quick start
- [Research Roadmap](RESEARCH_ROADMAP.md) - Overall research strategy
- [CLI Reference](CLI_REFERENCE.md) - Complete CLI command reference

**Architecture:**
- [Backend Architecture](ARCHITECTURE_BACKEND.md) - System architecture overview
- [Backend Modules](BACKEND_MODULES.md) - Module-level documentation

