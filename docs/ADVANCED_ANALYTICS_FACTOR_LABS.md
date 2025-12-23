# Advanced Analytics & Factor Labs

**Last Updated:** 2025-12-03  
**Status:** Roadmap for Advanced Research & Factor Development

---

## Overview

**Advanced Analytics & Factor Labs** represents a comprehensive roadmap for extending the trading system's analytical capabilities beyond the core backend. This roadmap focuses on:

- **Technical Analysis & Price Factor Library** (Phase A): Building a comprehensive library of price-based indicators and technical analysis factors
- **Alternative Data Factors** (Phase B): Advanced integration of insider trading, congressional trading, shipping, and news sentiment data
- **Factor Analysis & Event Study Engine** (Phase C): Systematic evaluation of factor effectiveness and event-driven research
- **Regime Models & Risk 2.0** (Phase D): Advanced risk modeling with market regime detection and adaptive risk management
- **ML Validation, Explainability & TCA** (Phase E): Enhanced model validation, explainability tools, and transaction cost analysis

**Relationship to Existing Backend:**

The Factor Labs roadmap builds upon and extends the existing backend infrastructure:

- **Phase 4** (Backtesting & QA): Provides the foundation for strategy testing and performance evaluation
- **Phase 6** (Event Features): Current insider/shipping features form the base for Phase B enhancements
- **Phase 7** (ML Meta-Layer): Existing meta-models are extended with explainability and advanced validation
- **Phase 8** (Risk Engine): Current risk metrics are enhanced with regime-aware modeling
- **Phase 9** (Model Governance): Current validation framework is extended with advanced explainability tools

---

## Phases & Sprints

### Phase A: TA/Price Factor Library

**Goal:** Build a comprehensive, production-ready library of technical analysis indicators and price-based factors.

**Sprints:**

#### A1: Core TA/Price Factors ✅ (Completed)

**Module:** `src/assembled_core/features/ta_factors_core.py`  
**API:** `build_core_ta_factors(prices, price_col="close", group_col="symbol", timestamp_col="timestamp")`

**Implementation:**
- **Multi-Horizon Returns**: Forward returns for 1/3/6/12 months (21/63/126/252 trading days)
  - `returns_1m`, `returns_3m`, `returns_6m`, `returns_12m`
  - `momentum_12m_excl_1m`: 12-month momentum excluding last month
- **Time-Series Trend Strength**: Normalized price vs. moving averages
  - `trend_strength_20`, `trend_strength_50`, `trend_strength_200`
  - Formula: `(price - MA_lookback) / ATR_20`
  - Works with or without high/low columns (falls back to volatility proxy)
- **Short-Term Reversal**: Z-scored returns over 1-3 days
  - `reversal_1d`, `reversal_2d`, `reversal_3d`
  - Rolling z-score with 60-day window to capture mean-reversion patterns

**Integration:**
- Builds on existing TA features (`ta_features.py`): reuses `add_moving_averages()` and `add_atr()`
- Compatible with standard price data format (timestamp, symbol, close)
- Designed for factor research and ML feature engineering
- Tests: `tests/test_features_ta_factors_core.py` (marked with `@pytest.mark.advanced`)

**Usage Example:**
```python
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
import pandas as pd

prices = pd.read_parquet("output/aggregates/1d.parquet")
factors = build_core_ta_factors(prices)
```

**Note:** Sprint A1 focuses on core price-based factors. Additional TA indicators (RSI, MACD, Bollinger Bands, etc.) will be added in Sprint A2.

#### A2: Liquidity & Volatility Factors ✅ (Completed)

**Module:** `src/assembled_core/features/ta_liquidity_vol_factors.py`

**Implementation:**
- **Realized Volatility**: Rolling standard deviation of log returns
  - `add_realized_volatility()`: Computes RV for multiple windows (default: 20, 60 days)
  - Columns: `rv_{window}` (annualized, e.g., `rv_20`, `rv_60`)
  - Formula: `std(log_returns) * sqrt(252)` for annualization
  
- **Volatility of Volatility**: Stability of volatility over time
  - `add_vol_of_vol()`: Rolling std of realized volatility over longer period
  - Columns: `vov_{rv_window}_{vol_window}` (e.g., `vov_20_60`)
  - Captures volatility clustering and regime changes
  
- **Turnover & Liquidity Proxies**: Multiple liquidity measures
  - `add_turnover_and_liquidity_proxies()`: Computes various liquidity factors
  - `turnover`: volume / freefloat (if freefloat column provided)
  - `volume_zscore`: Normalized volume per symbol (rolling z-score)
  - `spread_proxy`: (high - low) / close (simple spread approximation)

**Integration:**
- Works with standard price data format (timestamp, symbol, close)
- Optional columns: high, low, volume, freefloat
- Compatible with `build_core_ta_factors()` from Sprint A1
- Tests: `tests/test_features_ta_liquidity_vol_factors.py` (marked with `@pytest.mark.advanced`)

**Usage Example:**
```python
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_vol_of_vol,
    add_turnover_and_liquidity_proxies,
)

# Add realized volatility
df_rv = add_realized_volatility(prices, windows=[20, 60])

# Add Vol-of-Vol
df_vov = add_vol_of_vol(df_rv)

# Add liquidity proxies
df_final = add_turnover_and_liquidity_proxies(df_vov, freefloat_col="freefloat")
```

**Note:** Sprint A2 focuses on volatility and liquidity factors. Additional advanced price factors (multi-timeframe, adaptive indicators, pattern recognition) will be added in Sprint A3.

#### A4: Signal API & Factor Exposures ✅ (Completed)

**Module:** 
- `src/assembled_core/signals/signal_api.py` (Signal API)
- `src/assembled_core/risk/factor_exposures.py` (Factor Exposure Analysis)

**Implementation:**
- **Signal API**: Unified SignalFrame contract for strategy signals
  - `SignalMetadata` dataclass for strategy metadata (name, freq, universe, as_of, etc.)
  - `normalize_signals()`: Z-score, rank, or none normalization (cross-sectional per timestamp)
  - `make_signal_frame()`: Create standardized SignalFrame from raw scores
  - `validate_signal_frame()`: PIT-safety checks and structural validation
  
- **Factor Exposure Analysis**: Rolling regression of strategy returns against factor returns
  - `FactorExposureConfig`: Configuration for regression (window_size, mode, regression_method, etc.)
  - `compute_factor_exposures()`: Rolling/expanding window OLS or Ridge regression
  - `summarize_factor_exposures()`: Aggregate statistics (mean_beta, std_beta, mean_r2, etc.)

**Integration:**
- Signal API integrates with PIT-checks from `qa/point_in_time_checks.py`
- Factor Exposures integrated into Risk Reports via `scripts/generate_risk_report.py`
- CLI arguments: `--enable-factor-exposures`, `--factor-returns-file`, `--factor-exposures-window`
- Outputs: `factor_exposures_detail.csv`, `factor_exposures_summary.csv`, extended `risk_report.md`

**Tests:**
- `tests/test_signals_signal_api.py` (10 tests)
- `tests/test_risk_factor_exposures.py` (9 tests)
- `tests/test_cli_risk_report_factor_exposures.py` (3 tests)

**Documentation:**
- Design Document: `docs/SIGNAL_API_AND_FACTOR_EXPOSURES_A2_DESIGN.md`
- Workflow Integration: `docs/WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md` (Factor Exposures section)

#### A3: Market Breadth & Risk-On/Risk-Off Indicators ✅ (Completed)

**Module:** `src/assembled_core/features/market_breadth.py`

**Implementation:**
- **Market Breadth (MA-based)**: Fraction of symbols above moving average
  - `compute_market_breadth_ma()`: Computes fraction of stocks above MA per timestamp
  - Columns: `fraction_above_ma_{ma_window}`, `count_above_ma`, `count_total`
  - Returns universe-level DataFrame (one row per timestamp)
  - High values (>0.7): Broad participation, strong market
  - Low values (<0.3): Narrow participation, weak market
  
- **Advance/Decline Line**: Cumulative net advances/declines
  - `compute_advance_decline_line()`: Counts advances vs. declines per day
  - Columns: `advances`, `declines`, `net_advances`, `ad_line` (cumulative), `ad_line_normalized`
  - Rising A/D Line indicates broad market participation
  - Returns universe-level DataFrame (one row per timestamp)
  
- **Risk-On/Risk-Off Indicator**: Simple ratio-based proxy
  - `compute_risk_on_off_indicator()`: Ratio of advancing to declining stocks
  - Columns: `risk_on_ratio`, `risk_off_ratio`, `risk_on_off_score` (-1 to +1)
  - Placeholder implementation (sector-based classification not yet implemented)
  - Returns universe-level DataFrame (one row per timestamp)

**Integration:**
- Works with panel price data (multiple symbols over time)
- Returns universe-level time-series (aggregated across all symbols)
- Designed for regime detection and market state analysis
- Primary use: Research notebooks, factor analysis, regime detection workflows
- Can be merged/joined by timestamp for combined market state analysis
- Tests: `tests/test_features_market_breadth.py` (marked with `@pytest.mark.advanced`)

**Usage Example:**
```python
from src.assembled_core.features.market_breadth import (
    compute_market_breadth_ma,
    compute_advance_decline_line,
    compute_risk_on_off_indicator,
)

# Compute market breadth
breadth = compute_market_breadth_ma(prices, ma_window=50)

# Compute A/D Line
ad_line = compute_advance_decline_line(prices)

# Compute Risk-On/Risk-Off
risk_indicator = compute_risk_on_off_indicator(prices)

# Combine for regime detection
market_state = breadth.merge(ad_line, on="timestamp").merge(risk_indicator, on="timestamp")
```

**Note:** Sprint A3 focuses on universe-level market state indicators. Factor Engineering Pipeline (automated batch processing, caching, metadata tracking) will be added in a future sprint.

**Integration with Existing Backend:**
- Extends Phase 4 (Backtesting) with richer feature sets
- Enhances Phase 7 (ML Meta-Layer) with additional predictive features
- Complements existing trend baseline strategy in `signals/rules_trend.py`

#### A1: Use Cases & Roles Documentation ✅ (Completed)

**Document:** [Use Cases & Roles](USE_CASES_AND_ROLES_A1.md)

**Description:** Comprehensive documentation of backend capabilities organized by user roles (Quant PM, Quant Researcher, Quant Dev/Backend, Data Engineer). Includes use cases with CLI commands, component map, and references to design documents.

**Content:**
- Overview of backend from different role perspectives
- Role-specific goals and key artifacts
- Use cases per role (3-6 per role) with inputs, CLI actions, and outputs
- Component map linking roles to workflows, scripts, and documentation
- Open questions and future work

#### A3: Operations & Monitoring ✅ (Completed)

**Status:** Implemented and tested

**Module:** `src/assembled_core/ops/health_check.py`  
**CLI:** `scripts/check_health.py`, `scripts/cli.py check_health`  
**Profile Jobs:** Optional `OPERATIONS_HEALTH_CHECK` job type in `scripts/profile_jobs.py`

**Implementation:**
- **Health Check Core**: Data structures and utilities for health checks
  - `HealthCheck` and `HealthCheckResult` dataclasses
  - Status aggregation (OK, WARN, CRITICAL, SKIP)
  - JSON serialization (`health_result_to_dict()`, `health_result_from_dict()`)
  - Text rendering (`render_health_summary_text()`)
- **Health Check Script**: Read-only health checks for backend operations
  - Existence checks (backtest runs, risk reports, TCA reports)
  - Plausibility checks (drawdown, Sharpe, turnover, benchmark correlation)
  - Backtest freshness validation (last equity timestamp within lookback window)
  - Status interpretation and exit codes (0=OK/SKIP, 1=WARN, 2=CRITICAL)

**Integration:**
- Standalone script: `scripts/check_health.py` (direct execution)
- CLI subcommand: `scripts/cli.py check_health` (with all arguments: `--backtests-root`, `--days`, `--min-sharpe`, etc.)
- Outputs: `health_summary.json` (machine-readable), `health_summary.md` (human-readable)
- Designed for daily automation (cron/scheduler integration, see Runbook)

#### A4: Paper-Track-Playbook ✅ (Completed)

**Document:** [Paper Track Playbook](PAPER_TRACK_PLAYBOOK.md)

**Description:** Standardized process definition for moving trading strategies from backtest to paper trading (simulation) and eventually to live trading. Defines quality criteria, operational processes, monitoring metrics, and gate decisions.

**Content:**
- **Phase 0 - Candidate-Backtest**: Gate criteria for entry into paper track (minimum backtest duration >= 5 years, regime coverage >= 3 regimes, deflated Sharpe >= 0.5, max drawdown <= -30%, PIT-safety, optional walk-forward and factor exposures)
- **Phase 1 - Paper-Track Setup**: Required artifacts (backtest reports, model cards, configs), technical setup (EOD pipeline, SAFE-Bridge outputs), roles (Researcher, Operator, Risk/QA) and their tasks
- **Phase 2 - Paper-Track Runtime**: Minimum duration (6-12 months), monitored metrics (Sharpe, deflated Sharpe, MaxDD, hit-rate, turnover, slippage, factor exposures, regime performance), acceptable deviations vs. backtest (with example thresholds), handling of WARN/CRITICAL scenarios (pause, parameter review, kill)
- **Phase 3 - Go/No-Go to Live**: Clarification that live trading requires additional professional and regulatory checks, example Go/No-Go criteria, documentation requirements (updated model card, decision log)
- **Checklists & Templates**: Two compact tables (Backtest → Paper Gate, Paper → Live Gate) with fields: Sharpe/DSR, MaxDD, regime coverage, QA status, comments

**Integration:**
- Builds on existing backtest infrastructure (`scripts/run_backtest_strategy.py`, `scripts/batch_backtest.py`)
- Uses risk reports (`scripts/generate_risk_report.py`) for regime analysis and factor exposures
- Integrates with walk-forward analysis (B3) and deflated Sharpe (B4) for gate criteria
- Links to PIT-safety (B2) for validation requirements
- Optional job profiling: `python scripts/profile_jobs.py --job OPERATIONS_HEALTH_CHECK`

**Tests:**
- `tests/test_ops_health_check_core.py` (7 tests): Core functionality (status aggregation, serialization, rendering)
- `tests/test_cli_check_health.py` (7 tests): CLI integration, smoke tests, help commands

**Documentation:**
- Design Document: `docs/OPERATIONS_BACKEND_A3_DESIGN.md`
- Runbook: `docs/OPERATIONS_BACKEND.md` (daily/weekly checklists, troubleshooting, automation)

---

### Phase B: Alt-Data Factors 2.0

**Goal:** Advanced integration and enhancement of alternative data sources (insider trading, congressional trading, shipping, news).

**Sprints:**

#### B1: Earnings & Insider Alt-Data ✅ (Completed)

**Status:** Implemented and tested

**Module:** `src/assembled_core/features/altdata_earnings_insider_factors.py`  
**API:** 
- `build_earnings_surprise_factors(events_earnings, prices, window_days=20, as_of=None)`
- `build_insider_activity_factors(events_insider, prices, lookback_days=60, as_of=None)`

**Implementation:**
- **Earnings Surprise Factors**: Transform earnings events into time-series factors
  - `earnings_eps_surprise_last`: Last EPS surprise percentage (most recent earnings event)
  - `earnings_revenue_surprise_last`: Last revenue surprise percentage
  - `earnings_positive_surprise_flag`: Binary flag (1 if last surprise was positive)
  - `earnings_negative_surprise_flag`: Binary flag (1 if last surprise was negative)
  - `post_earnings_drift_return_{window_days}d`: Forward return after earnings announcement
  
- **Insider Activity Factors**: Aggregate insider transactions over rolling windows
  - `insider_net_notional_{lookback_days}d`: Net insider notional (buy - sell) over lookback window
  - `insider_buy_count_{lookback_days}d`: Number of insider buy transactions
  - `insider_sell_count_{lookback_days}d`: Number of insider sell transactions
  - `insider_buy_sell_ratio_{lookback_days}d`: Ratio of buys to sells (count-based)
  - `insider_net_notional_normalized_{lookback_days}d`: Net notional normalized by market cap proxy

**Data Contracts:**
- **events_earnings_df**: `timestamp`, `symbol`, `event_type`, `event_id`, `eps_actual`, `eps_estimate`, `revenue_actual`, `revenue_estimate`
- **events_insider_df**: `timestamp`, `symbol`, `event_type`, `event_id`, `usd_notional`, `direction`, `shares`, `price`

**Integration:**
- Compatible with `build_core_ta_factors()` and other Phase A factors
- Can be merged with price DataFrame using `timestamp` & `symbol`
- Designed for use in Phase C1/C2 factor analysis workflows
- Price data comes from `LocalParquetPriceDataSource` (not Finnhub)
- Events loaded from `output/altdata/events_earnings.parquet` and `events_insider.parquet`

**Finnhub Events Client:**
- **Module:** `src/assembled_core/data/altdata/finnhub_events.py`
- **Functions:** `fetch_earnings_events()`, `fetch_insider_events()`
- Robust error handling (4xx/5xx → empty DataFrame, no crash)
- Rate limit handling (60 calls/minute for free tier)
- Data contract compliance (normalized to events_earnings_df/events_insider_df)

**Point-in-Time Safety (B2):**
- All Alt-Data feature builders support `as_of` parameter for PIT-safe factor computation
- Events are filtered by `disclosure_date <= as_of` to prevent look-ahead bias
- See [Point-in-Time and Latency Documentation](POINT_IN_TIME_AND_LATENCY.md) for details

**Download Script:**
- **Module:** `scripts/download_altdata_finnhub_events.py`
- CLI interface for downloading events from Finnhub
- Supports `--symbols-file`, `--event-types` (earnings, insider), `--start-date`, `--end-date`
- Stores raw events in `data/raw/altdata/finnhub/` and cleaned events in `output/altdata/`

**Factor Columns Summary:**

| Factor Name | Category | Description |
|------------|----------|-------------|
| `earnings_eps_surprise_last` | Earnings | Last EPS surprise percentage (forward-filled until next event) |
| `earnings_revenue_surprise_last` | Earnings | Last revenue surprise percentage |
| `earnings_positive_surprise_flag` | Earnings | Binary flag (1 if last EPS surprise > 0) |
| `earnings_negative_surprise_flag` | Earnings | Binary flag (1 if last EPS surprise < 0) |
| `post_earnings_drift_return_{window_days}d` | Earnings | Forward return after earnings announcement |
| `insider_net_notional_{lookback_days}d` | Insider | Net insider notional (buy - sell) over rolling window |
| `insider_buy_count_{lookback_days}d` | Insider | Number of buy transactions in window |
| `insider_sell_count_{lookback_days}d` | Insider | Number of sell transactions in window |
| `insider_buy_sell_ratio_{lookback_days}d` | Insider | Ratio of buys to sells (count-based) |
| `insider_net_notional_normalized_{lookback_days}d` | Insider | Net notional normalized by market cap proxy (if volume available) |

**Tests:**
- `tests/test_features_altdata_earnings_insider_factors.py`: Factor computation tests (marked with `@pytest.mark.advanced`)
- `tests/test_data_finnhub_events_client.py`: API client tests with mocks (marked with `@pytest.mark.advanced`)

**Usage Example:**
```python
from src.assembled_core.features.altdata_earnings_insider_factors import (
    build_earnings_surprise_factors,
    build_insider_activity_factors,
)
import pandas as pd

# Load prices (from LocalParquetPriceDataSource)
prices = price_source.get_history(...)

# Load events (from Parquet files)
events_earnings = pd.read_parquet("output/altdata/events_earnings.parquet")
events_insider = pd.read_parquet("output/altdata/events_insider.parquet")

# Build factors
earnings_factors = build_earnings_surprise_factors(events_earnings, prices, window_days=20)
insider_factors = build_insider_activity_factors(events_insider, prices, lookback_days=60)

# Merge with other factors
from src.assembled_core.features import build_core_ta_factors
ta_factors = build_core_ta_factors(prices)

all_factors = ta_factors.merge(
    earnings_factors[["timestamp", "symbol", "earnings_eps_surprise_last", ...]],
    on=["timestamp", "symbol"],
    how="left"
).merge(
    insider_factors[["timestamp", "symbol", "insider_net_notional_60d", ...]],
    on=["timestamp", "symbol"],
    how="left"
)
```

**CLI Integration:**
- Factor analysis workflow supports `--factor-set alt_earnings_insider` and `--factor-set core+alt`
- See `docs/WORKFLOWS_FACTOR_ANALYSIS.md` for usage examples

#### B2: News, Sentiment & Macro Alt-Data ✅ (Completed)

**Status:** Implemented and tested

**Design Document:** `docs/ALT_DATA_FACTORS_B2_DESIGN.md`

**Module:** `src/assembled_core/features/altdata_news_macro_factors.py`  
**API:**
- `build_news_sentiment_factors(news_sentiment_daily, prices, lookback_days=20, as_of=None)`
- `build_macro_regime_factors(macro_series, prices, country_filter=None)`

**Implementation:**
- **News Sentiment Factors**: Transform daily sentiment data into time-series factors
  - `news_sentiment_mean_{lookback_days}d`: Rolling mean of daily sentiment scores
  - `news_sentiment_trend_{lookback_days}d`: Trend in sentiment (slope over lookback window)
  - `news_sentiment_shock_flag`: Binary flag (1 if sentiment change exceeds threshold)
  - `news_sentiment_volume_{lookback_days}d`: Rolling mean of news volume
  
- **Macro Regime Factors**: Transform macro-economic indicators into regime indicators
  - `macro_growth_regime`: Growth regime (+1 = expansion, -1 = recession, 0 = neutral)
  - `macro_inflation_regime`: Inflation regime (+1 = high inflation, -1 = low/deflation, 0 = neutral)
  - `macro_risk_aversion_proxy`: Risk-on/risk-off indicator (+1 = risk-off, -1 = risk-on, 0 = neutral)

**Data Contracts:**
- **news_events_df**: `timestamp`, `symbol` (or None for market-wide), `source`, `headline`, `news_id`, `event_type="news"`, optional `sentiment_score`, `category`
- **news_sentiment_daily_df**: `timestamp`, `symbol` (or `"__MARKET__"`), `sentiment_score`, `sentiment_volume`
- **macro_series_df**: `timestamp`, `macro_code`, `value`, `country`, `release_time`

**Client Module:** `src/assembled_core/data/altdata/finnhub_news_macro.py`
- `fetch_news()`: Fetches company news or market-wide news from Finnhub
- `fetch_news_sentiment()`: Aggregates news sentiment on daily basis
- `fetch_macro_series()`: Fetches economic calendar and indicator values

**Download Script:** `scripts/download_altdata_finnhub_news_macro.py`
- CLI for downloading news, news sentiment, and macro data
- Supports symbol lists and macro code lists
- Saves raw data to `data/raw/altdata/finnhub/` and cleaned data to `output/altdata/`

**Integration:**
- Integrated into `analyze_factors` CLI workflow
- Factor sets: `alt_news_macro`, `core+alt_news`, `core+alt_full`
- Compatible with Phase C1/C2 factor analysis workflows
- Macro factors are market-wide (same value for all symbols on same date)

**Storage:**
- Raw: `data/raw/altdata/finnhub/news_raw.parquet`, `news_sentiment_raw.parquet`, `macro_raw.parquet`
- Clean: `output/altdata/news_events.parquet`, `news_sentiment_daily.parquet`, `macro_series.parquet`

**Point-in-Time Safety (B2):**
- All Alt-Data feature builders support `as_of` parameter for PIT-safe factor computation
- Events are filtered by `disclosure_date <= as_of` to prevent look-ahead bias
- See [Point-in-Time and Latency Documentation](POINT_IN_TIME_AND_LATENCY.md) for details

**Tests:**
- `tests/test_features_altdata_news_macro_factors.py` (marked with `@pytest.mark.advanced`)
- `tests/test_data_finnhub_news_macro_client.py` (marked with `@pytest.mark.advanced`)
- `tests/test_point_in_time_altdata.py` (PIT safety tests, marked with `@pytest.mark.advanced`)

**Factor Table:**

| Factor Name | Category | Description |
|------------|----------|-------------|
| `news_sentiment_mean_{lookback_days}d` | News Sentiment | Rolling mean of daily sentiment scores over lookback window |
| `news_sentiment_trend_{lookback_days}d` | News Sentiment | Trend in sentiment (slope over lookback window) |
| `news_sentiment_shock_flag` | News Sentiment | Binary flag (1 if sentiment change > 1.5 std) |
| `news_sentiment_volume_{lookback_days}d` | News Sentiment | Rolling mean of news volume |
| `macro_growth_regime` | Macro Regime | Growth regime indicator (+1 = expansion, -1 = recession, 0 = neutral) |
| `macro_inflation_regime` | Macro Regime | Inflation regime indicator (+1 = high inflation, -1 = low/deflation, 0 = neutral) |
| `macro_risk_aversion_proxy` | Macro Regime | Risk-on/risk-off indicator (+1 = risk-off, -1 = risk-on, 0 = neutral) |

#### B3: Insider Trading 2.0 (Future)
- Enhanced insider transaction types classification (Open Market, Private Placement, Exercise, etc.)
- Insider clustering analysis (multiple insiders trading simultaneously)
- Insider historical performance tracking (success rate of individual insiders)
- Insider sentiment scoring (aggregate buy/sell signals with confidence weights)
- Integration with Phase 6 existing insider features

#### B4: Congressional Trading & Public Disclosures
- STOCK Act data integration (congressional member trades)
- Timing analysis (trades before important policy decisions/announcements)
- Sector exposure analysis (which sectors do members trade most?)
- Public disclosure parsing (8-K filings, Form 4 filings)
- Event-driven signals based on congressional activity

#### B3: Shipping & News Sentiment Enhancement
- Advanced shipping congestion scoring (multiple ports, time-series patterns)
- News sentiment scoring using NLP (e.g., FinBERT, VADER)
- News volume spike detection (abnormal news activity)
- Earnings announcement integration (pre/post earnings drift analysis)
- Multi-source signal aggregation (combining shipping, news, insider data)

**Integration with Existing Backend:**
- Extends Phase 6 (Event Features) with enhanced data quality and new sources
- Provides richer features for event-based strategies in `signals/rules_event_insider_shipping.py`

---

### Phase C: Factor Analysis & Event Study Engine

**Goal:** Systematic evaluation of factor effectiveness, correlation analysis, and event-driven research tools.

**Sprints:**

#### C1: Information Coefficient (IC) Engine ✅ (Completed)

**Module:** `src/assembled_core/qa/factor_analysis.py`

##### C1 – Factor-IC/IR-Engine: Zielbild & Data Contract

**DataFrame-Format (Input für `qa/factor_analysis.py`):**

Die IC-Engine erwartet ein **Panel-DataFrame** im folgenden Format:

- **Index:** Standard Integer-Index (kein MultiIndex)
- **Spalten:**
  - `timestamp`: UTC-Zeitstempel (datetime64[ns, UTC])
  - `symbol`: Symbol-Name (string)
  - `factor_*`: Faktor-Spalten (z.B. `returns_12m`, `trend_strength_200`, `rv_20`, `volume_zscore`)
  - `fwd_return_*`: Forward-Return-Spalten (z.B. `fwd_return_1d`, `fwd_return_5d`, `fwd_return_21d`)
  - Optional: Original-Preis-Spalten (`close`, `open`, `high`, `low`, `volume`)

**Beispiel-Struktur:**
```
   timestamp              symbol  close  returns_12m  trend_strength_200  fwd_return_5d
0  2020-01-01 00:00:00+00:00  AAPL  100.0      0.15             0.5           0.02
1  2020-01-01 00:00:00+00:00  MSFT  200.0      0.12             0.3           0.01
2  2020-01-02 00:00:00+00:00  AAPL  101.0      0.16             0.6           0.03
3  2020-01-02 00:00:00+00:00  MSFT  201.0      0.13             0.4           0.02
...
```

**Wichtige Eigenschaften:**
- **Panel-Format:** Mehrere Symbole pro Timestamp (cross-sectional analysis)
- **Sortierung:** Nach `symbol`, dann `timestamp` (für korrekte Forward-Return-Berechnung)
- **Kein MultiIndex:** Normale Spalten, kein hierarchischer Index
- **UTC-Zeitzone:** Alle Timestamps müssen UTC-aware sein

**Faktor-Berechnung (Phase A):**

Faktoren werden von den Phase-A-Modulen erzeugt:

- **`ta_factors_core.build_core_ta_factors()`:**
  - Input: Panel mit `timestamp`, `symbol`, `close` (+ optional: `high`, `low`, `volume`)
  - Output: Gleiche Struktur + Faktor-Spalten (`returns_1m`, `returns_3m`, `returns_6m`, `returns_12m`, `momentum_12m_excl_1m`, `trend_strength_20/50/200`, `reversal_1d/2d/3d`)
  - Format: Panel (kein MultiIndex), sortiert nach `symbol`, dann `timestamp`

- **`ta_liquidity_vol_factors.add_realized_volatility()`:**
  - Input: Panel mit `timestamp`, `symbol`, `close`
  - Output: Gleiche Struktur + `rv_20`, `rv_60` (annualized realized volatility)
  - Format: Panel (kein MultiIndex)

- **`ta_liquidity_vol_factors.add_turnover_and_liquidity_proxies()`:**
  - Input: Panel mit `timestamp`, `symbol`, `volume` (+ optional: `high`, `low`, `close`, `freefloat`)
  - Output: Gleiche Struktur + `volume_zscore`, `spread_proxy` (+ optional: `turnover`)
  - Format: Panel (kein MultiIndex)

**Forward-Returns/Labels (für IC-Berechnung):**

Forward-Returns werden von `add_forward_returns()` erzeugt:

- **Input:** Panel mit `timestamp`, `symbol`, `close` (oder `price_col`)
- **Output:** Gleiche Struktur + `fwd_return_{horizon_days}d` (z.B. `fwd_return_5d`)
- **Berechnung:** Log-Returns oder Simple-Returns (konfigurierbar)
  - Log: `ln(price[t+h] / price[t])`
  - Simple: `(price[t+h] / price[t]) - 1`
- **Look-Ahead-Bias:** Verhindert durch `shift(-horizon_days)` (keine zukünftigen Daten)
- **Format:** Panel (kein MultiIndex), sortiert nach `symbol`, dann `timestamp`
- **NaN-Handling:** Letzte `horizon_days` Zeilen pro Symbol haben NaN (keine zukünftigen Daten verfügbar)

**IC-Berechnung (Cross-Sectional):**

Die IC-Engine berechnet pro Timestamp die Korrelation zwischen Faktoren und Forward-Returns:

- **`compute_factor_ic()`:**
  - Input: Panel mit `timestamp`, `symbol`, `factor_*`, `fwd_return_*`
  - Gruppierung: Pro `timestamp` (cross-sectional)
  - Berechnung: Pearson-Korrelation (linear) oder Spearman-Korrelation (rank)
  - Output: DataFrame mit Spalten: `timestamp`, `factor`, `ic`, `count`
  - Format: Eine Zeile pro (timestamp, factor) Kombination

- **`compute_rank_ic()`:**
  - Wrapper um `compute_factor_ic()` mit `method="spearman"`
  - Gleiche Input/Output-Struktur wie `compute_factor_ic()`

**IC-Aggregation:**

- **`summarize_factor_ic()`:**
  - Input: IC-DataFrame von `compute_factor_ic()` oder `compute_rank_ic()`
  - Gruppierung: Pro `factor` (aggregiert über alle Timestamps)
  - Output: DataFrame mit Spalten: `factor`, `mean_ic`, `std_ic`, `ic_ir`, `hit_ratio`, `count`, `min_ic`, `max_ic`
  - Format: Eine Zeile pro Faktor, sortiert nach `ic_ir` (absteigend)

**Factor-Report-Workflow:**

Die High-Level-Funktion `run_factor_report()` orchestriert den kompletten Workflow:

1. **Faktor-Berechnung:** Ruft Phase-A-Module auf (je nach `factor_set`)
2. **Forward-Returns:** Fügt `fwd_return_{horizon_days}d` hinzu
3. **IC-Berechnung:** Berechnet IC und Rank-IC für alle Faktoren
4. **Zusammenfassung:** Aggregiert IC-Statistiken pro Faktor

**Output-Struktur:**
```python
{
    "factors": DataFrame,      # Panel mit Faktoren + Forward-Returns
    "ic": DataFrame,            # IC pro (timestamp, factor)
    "rank_ic": DataFrame,       # Rank-IC pro (timestamp, factor)
    "summary_ic": DataFrame,   # Aggregierte IC-Statistiken pro Faktor
    "summary_rank_ic": DataFrame # Aggregierte Rank-IC-Statistiken pro Faktor
}
```

**Kernfunktionen (Sprint C1 - Implementiert):**

1. **`add_forward_returns()`:** Forward-Returns berechnen (kein Look-Ahead-Bias)
   - Unterstützt einzelne oder mehrere Horizons (z.B. `horizon_days=[20, 60, 252]`)
   - Funktioniert mit Preis- und Factor-DataFrames
   - Erzeugt Spalten: `fwd_return_{horizon}d` (einzelner Horizon) oder `fwd_ret_{horizon}` (mehrere Horizons)

2. **`compute_ic()`:** Cross-Sectional IC pro Timestamp und Faktor (Pearson/Spearman)
   - Auto-Detektion von Factor-Spalten
   - Output: DataFrame mit Index=timestamp, Spalten=ic_<factor_name>
   - MultiIndex-Handling (wird intern normalisiert)
   - Robust gegenüber NaNs

3. **`compute_rank_ic()`:** Rank-IC (Spearman-Korrelation, robust gegen Outliers)
   - Wrapper um `compute_ic()` mit `method="spearman"`
   - Gleiche Output-Struktur wie `compute_ic()`

4. **`summarize_ic_series()`:** IC-Aggregation zu Faktor-Level-Statistiken
   - Output: `mean_ic`, `std_ic`, `ic_ir`, `hit_ratio`, `q05`, `q95`, `min_ic`, `max_ic`, `count`
   - Sortiert nach `ic_ir` (absteigend)

5. **`compute_rolling_ic()`:** Rolling-IC-Statistiken für Stabilitäts-Analyse
   - Rolling-Mean und Rolling-IR pro Faktor
   - Konfigurierbares Fenster (default: 60 Tage)

6. **`example_factor_analysis_workflow()`:** High-Level-Workflow (Faktoren + IC + Zusammenfassung)
   - Orchestriert kompletten Workflow: Faktoren → Forward-Returns → IC → Summary → Rolling-IC

**Zukünftige Erweiterungen (C1 extension):**
- Rolling-IC (IC über rollierendes Fenster, z.B. 60 Tage)
- IC-Decay-Analyse (wie lange bleiben Faktoren prädiktiv?)
- IC-Verteilungs-Analyse (Skew, Kurtosis)
- Visualisierungs-Tools (IC-Zeitreihen-Plots, IC-Verteilungs-Histogramme)

**Implementation:**
- **Forward Returns Computation**: `add_forward_returns()`
  - Computes forward-looking returns (no look-ahead bias)
  - Supports log and simple returns
  - Customizable horizons (1d, 5d, 21d, etc.)
  - Returns per symbol with proper handling of edge cases (NaN for last rows)

- **Cross-Sectional IC**: `compute_ic()`
  - Computes Information Coefficient per timestamp (cross-sectional correlation)
  - Auto-detects factor columns (excludes timestamp, symbol, forward return columns)
  - Supports Pearson (linear) and Spearman (rank) correlation methods
  - Handles multiple factors simultaneously
  - Returns DataFrame with Index=timestamp, Columns=ic_<factor_name>
  - MultiIndex-Handling: automatically resets to regular columns if needed
  - Robust to NaNs: rows with NaN in factor or forward return are ignored
  - IC measures how well a factor predicts forward returns at each point in time

- **Rank-IC**: `compute_rank_ic()`
  - Convenience wrapper for Spearman rank correlation
  - More robust to outliers than Pearson correlation
  - Captures monotonic relationships better
  - Same output format as `compute_ic()` (Index=timestamp, Columns=ic_<factor_name>)

- **IC Summary Statistics**: `summarize_ic_series()`
  - Aggregates IC time-series to factor-level statistics:
    - `mean_ic`: Average IC across all timestamps
    - `std_ic`: Standard deviation of IC
    - `ic_ir`: Information Ratio = mean_ic / std_ic
    - `hit_ratio`: Percentage of days with positive IC (0.0 to 1.0)
    - `q05`, `q95`: 5% and 95% quantiles
    - `min_ic`, `max_ic`: Range of IC values
    - `count`: Number of timestamps with valid IC values
  - Results sorted by IC-IR (descending) for factor ranking
  - Handles both index-based (timestamp as index) and column-based formats

- **Rolling IC Statistics**: `compute_rolling_ic()`
  - Computes rolling mean IC and rolling IR over configurable window (default: 60 days)
  - Useful for analyzing factor stability over time
  - Output: DataFrame with `rolling_mean_<factor_name>` and `rolling_ir_<factor_name>` columns
  - First `window-1` rows have NaN (not enough data for rolling window)

**Integration:**
- Works with factors from Phase A (ta_factors_core, ta_liquidity_vol_factors)
- Designed for factor research and evaluation
- Compatible with backtest engine and research workflows
- Primary use: Research notebooks, factor evaluation, factor selection
- No external dependencies (uses pandas for correlation, no scipy required)

**Usage Example (New Phase C1 API):**
```python
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_ic,
    compute_rank_ic,
    summarize_ic_series,
    compute_rolling_ic,
)
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
from src.assembled_core.features.ta_liquidity_vol_factors import (
    add_realized_volatility,
    add_turnover_and_liquidity_proxies,
)

# Build factors and add forward returns (multiple horizons)
factors_df = build_core_ta_factors(prices)
factors_df = add_realized_volatility(factors_df, windows=[20, 60])
factors_df = add_turnover_and_liquidity_proxies(factors_df)
factors_df = add_forward_returns(factors_df, horizon_days=[20, 60], return_type="log")

# Compute IC (auto-detects all factor columns)
ic_df = compute_ic(
    factors_df,
    forward_returns_col="fwd_ret_20",  # Use 20-day forward returns
    group_col="symbol"
)

# Compute Rank-IC
rank_ic_df = compute_rank_ic(
    factors_df,
    forward_returns_col="fwd_ret_20",
    group_col="symbol"
)

# Summarize IC statistics
summary_ic = summarize_ic_series(ic_df)
# Returns DataFrame with mean_ic, std_ic, ic_ir, hit_ratio, q05, q95, etc.
# Sorted by IC-IR (descending)

# Rolling IC analysis
rolling_ic = compute_rolling_ic(ic_df, window=60)
```

**Complete Workflow Example:**
See `research/factors/IC_analysis_core_factors.py` for a complete example that:
- Loads price data from local Parquet files
- Computes core TA/Price factors and volatility/liquidity factors
- Adds forward returns for multiple horizons
- Computes IC and Rank-IC
- Summarizes IC statistics
- Creates simple visualizations

**Tests:** 
- `tests/test_factor_analysis.py` (Legacy functions, marked with `@pytest.mark.advanced`)
- `tests/test_qa_factor_analysis.py` (New Phase C1 functions, 18 tests, marked with `@pytest.mark.advanced`)

**Future Enhancements (C1 extension):**
- IC decay analysis (how long do factors remain predictive?)
- IC distribution analysis (skew, kurtosis)
- Visualization tools (IC time-series plots, IC distribution histograms)

#### C2: Factor Report Workflow ✅ (Completed)

**Module:** `src/assembled_core/qa/factor_analysis.py`

**Implementation:**
- **High-Level Factor Report Function**: `run_factor_report()`
  - Orchestrates complete factor analysis pipeline:
    1. Compute factors based on factor_set ("core", "vol_liquidity", "all")
    2. Add forward returns for evaluation
    3. Compute IC and Rank-IC
    4. Generate summary statistics
  - Returns dictionary with factors DataFrame, IC DataFrames, and summaries
  - Supports custom forward horizons and factor sets

**CLI Integration:**
- **Subcommand**: `scripts/cli.py factor_report`
  - Loads price data from local Parquet files or Yahoo Finance
  - Supports universe files (`--symbols-file`)
  - Configurable factor sets and forward horizons
  - Optional CSV output for summary statistics
  - Table-formatted output for easy viewing

**Usage Example:**
```powershell
# Set environment variables for local alt-data
$env:ASSEMBLED_DATA_SOURCE = "local"
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

# Generate factor report for AI Tech universe
python scripts/cli.py factor_report `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2005-01-01 `
  --end-date 2025-12-02 `
  --factor-set core `
  --fwd-horizon-days 5 `
  --output-csv output/factor_reports/ai_tech_core_5d_ic.csv
```

**Integration:**
- Works with all Phase A factors (core TA/Price, volatility, liquidity, market breadth)
- Uses Phase C1 IC Engine for evaluation
- Designed for research notebooks and systematic factor evaluation
- Primary use: Factor ranking, factor selection, research workflow

**Tests:** `tests/test_factor_report_workflow.py` (marked with `@pytest.mark.advanced`)

**Note:** Event Study Framework (original C2) will be implemented in a future sprint as C3 or later.

---

#### C2: Factor Ranking & Selection ✅ (Completed)

**Status:** ✅ Completed  
**Module:** `src/assembled_core/qa/factor_analysis.py`

##### C2 – Factor Ranking & Selection: Implementation

**Ziel:**
Erweitere die IC-Engine (C1) um Portfolio-basierte Faktor-Evaluierung. Während IC die Korrelation zwischen Faktor und Forward-Return misst, evaluiert C2 die tatsächlichen Portfolio-Returns, wenn man nach Faktor-Werten sortiert und in Quantilen investiert.

**Input-Format:**
- **Panel-DataFrame** (gleiches Format wie C1):
  - **Spalten:** `timestamp`, `symbol`, `factor_*`, `fwd_return_*`
  - **Format:** Kein MultiIndex, sortiert nach `symbol`, dann `timestamp`
  - **Beispiel:**
    ```
    timestamp              symbol  returns_12m  trend_strength_200  fwd_return_20d
    2020-01-01 00:00:00+00:00  AAPL      0.15             0.5           0.02
    2020-01-01 00:00:00+00:00  MSFT      0.12             0.3           0.01
    2020-01-01 00:00:00+00:00  GOOG      0.18             0.7           0.03
    ...
    ```

**Zusätzliche Kennzahlen (C2):**

1. **Factor-Portfolio-Returns (Quantil-basiert):**
   - Pro Timestamp: Sortiere Symbole nach Faktor-Wert
   - Teile in Quantile (z.B. Q1–Q5, Q1 = niedrigster Faktor-Wert, Q5 = höchster)
   - Berechne gleichgewichtete Portfolio-Returns pro Quantil
   - Output: Zeitreihe der Portfolio-Returns pro Quantil

2. **Long/Short-Portfolio:**
   - Long: Top-Quantil (Q5) – höchste Faktor-Werte
   - Short: Bottom-Quantil (Q1) – niedrigste Faktor-Werte
   - Portfolio-Return = Q5-Return - Q1-Return
   - Output: Zeitreihe der Long/Short-Returns

3. **Performance-Kennzahlen:**
   - **Sharpe Ratio:** Annualisiert, basierend auf Portfolio-Returns
   - **t-Statistik:** t-Test für Signifikanz (H0: Mean Return = 0)
   - **Deflated Sharpe Ratio:** Sharpe Ratio adjustiert für Multiple Testing (False Discovery Rate)
   - **CAGR:** Compound Annual Growth Rate (für Long/Short-Portfolio)
   - **Max Drawdown:** Maximum Drawdown (für Long/Short-Portfolio)
   - **Win Rate:** Anteil positiver Perioden

**Implementierte Funktionen:**

| Funktion | Beschreibung | Output |
|----------|-------------|--------|
| `build_factor_portfolio_returns()` | Erstellt Quantil-basierte Portfolio-Returns | DataFrame: `timestamp`, `factor`, `quantile`, `mean_return`, `n` |
| `build_long_short_portfolio_returns()` | Berechnet Long/Short-Portfolio-Returns (Q5 - Q1) | DataFrame: `timestamp`, `factor`, `ls_return`, `gross_exposure`, `n_long`, `n_short` |
| `summarize_factor_portfolios()` | Aggregiert Portfolio-Performance-Metriken | DataFrame: `factor`, `annualized_return`, `annualized_vol`, `sharpe`, `t_stat`, `p_value`, `win_ratio`, `max_drawdown`, `n_periods` |
| `compute_deflated_sharpe_ratio()` | Berechnet Deflated Sharpe Ratio (Multiple Testing Adjustment) | Float: Deflated Sharpe Ratio |

**Wichtigste Kennzahlen (C2):**

1. **Sharpe Ratio** (annualisiert)
   - Maß für risikoadjustierte Rendite
   - Sharpe > 1.0 gilt als gut, > 2.0 als sehr gut

2. **t-Statistik** (t-Test für Mean Return)
   - Testet Signifikanz: H0: Mean Return = 0
   - |t| > 2.0 deutet auf statistische Signifikanz hin (p < 0.05)

3. **Deflated Sharpe Ratio (DSR)**
   - Adjustiert für Multiple Testing (False Discovery Rate)
   - DSR > 0: Faktor ist signifikant nach Adjustierung
   - DSR < 0: Faktor könnte durch Overfitting/Multiple Testing entstanden sein

4. **Long/Short-Returns**
   - Q5 (Top-Quantil) - Q1 (Bottom-Quantil)
   - Zeigt Spread zwischen besten und schlechtesten Faktor-Werten

5. **Max Drawdown**
   - Größter Rückgang vom Peak
   - Negativer Wert, kleiner ist besser

6. **Win Ratio**
   - Anteil positiver Perioden (0.0 bis 1.0)
   - Win Ratio > 0.5 ist wünschenswert

**Funktions-Signaturen:**

```python
def build_factor_portfolio_returns(
    data: pd.DataFrame,
    factor_cols: str | list[str],
    forward_returns_col: str,
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    quantiles: int = 5,
    min_obs: int = 10,
) -> pd.DataFrame:
    """
    Build factor portfolio returns based on quantile sorting.
    
    Args:
        factor_df: Panel DataFrame with timestamp, symbol, factor_col, forward_returns_col
        factor_col: Column name of the factor to rank by
        forward_returns_col: Column name of forward returns (e.g., "fwd_return_20d")
        quantiles: Number of quantiles (default: 5, i.e., Q1-Q5)
        group_col: Column name for grouping (default: "symbol")
        timestamp_col: Column name for timestamp (default: "timestamp")
        rebalance_freq: Rebalancing frequency (default: "1d")
        equal_weight: Whether to use equal weights (default: True)
    
    Returns:
        DataFrame with columns:
        - timestamp: Timestamp
        - quantile: Quantile number (1, 2, ..., quantiles)
        - portfolio_return: Equal-weighted portfolio return for this quantile
        - n_symbols: Number of symbols in this quantile
        - long_short_return: Q5 - Q1 return (only for Q5 rows)
    
    Example:
        timestamp              quantile  portfolio_return  n_symbols  long_short_return
        2020-01-01 00:00:00+00:00  1        0.01            10         NaN
        2020-01-01 00:00:00+00:00  2        0.02            10         NaN
        ...
        2020-01-01 00:00:00+00:00  5        0.05            10         0.04
    """
    pass


def summarize_factor_portfolios(
    portfolio_returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    quantile_col: str = "quantile",
    return_col: str = "portfolio_return",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Summarize factor portfolio performance metrics.
    
    Args:
        portfolio_returns_df: DataFrame from build_factor_portfolio_returns()
        risk_free_rate: Risk-free rate (annualized, default: 0.0)
        periods_per_year: Trading periods per year (default: 252 for daily)
        quantile_col: Column name for quantile (default: "quantile")
        return_col: Column name for portfolio return (default: "portfolio_return")
        timestamp_col: Column name for timestamp (default: "timestamp")
    
    Returns:
        DataFrame with one row per quantile, columns:
        - quantile: Quantile number
        - mean_return: Mean portfolio return (annualized)
        - std_return: Standard deviation of returns (annualized)
        - sharpe_ratio: Sharpe Ratio (annualized)
        - t_stat: t-statistic for mean return (H0: mean = 0)
        - p_value: p-value for t-test
        - cagr: Compound Annual Growth Rate
        - max_drawdown: Maximum drawdown
        - win_rate: Percentage of positive returns
        - n_periods: Number of periods
        - n_symbols_avg: Average number of symbols per period
    
    For Long/Short (Q5 - Q1):
        - Additional row with quantile="long_short"
        - Same metrics computed for long_short_return
    """
    pass


def compute_deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_observations: int,
    n_factors_tested: int = 1,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> float:
    """
    Compute Deflated Sharpe Ratio (DSR) to adjust for multiple testing.
    
    The Deflated Sharpe Ratio adjusts the observed Sharpe Ratio for:
    - Multiple testing (False Discovery Rate)
    - Non-normal return distributions (skewness, kurtosis)
    
    Formula (simplified):
        DSR = (SR - E[SR]) / std(SR)
        where E[SR] and std(SR) account for multiple testing and distribution
    
    Args:
        sharpe_ratio: Observed Sharpe Ratio
        n_observations: Number of observations (time periods)
        n_factors_tested: Number of factors tested (for multiple testing adjustment)
        skewness: Optional skewness of returns (for distribution adjustment)
        kurtosis: Optional kurtosis of returns (for distribution adjustment)
    
    Returns:
        Deflated Sharpe Ratio (float)
    
    References:
        - Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio:
          Correcting for selection bias, backtest overfitting and non-normality.
          Journal of Portfolio Management, 40(5), 94-107.
    """
    pass
```

**Workflow-Beispiel:**

```python
from src.assembled_core.qa.factor_analysis import (
    build_factor_portfolio_returns,
    build_long_short_portfolio_returns,
    summarize_factor_portfolios,
    compute_deflated_sharpe_ratio,
)

# 1. Build factor portfolios (Q1-Q5)
portfolio_returns = build_factor_portfolio_returns(
    data=factors_with_returns,
    factor_cols="returns_12m",  # Can be list of factors
    forward_returns_col="fwd_return_20d",
    quantiles=5,
    min_obs=10
)

# 2. Build Long/Short portfolios
ls_returns = build_long_short_portfolio_returns(
    portfolio_returns,
    low_quantile=1,
    high_quantile=5  # Default: highest quantile
)

# 3. Summarize portfolio performance
summary = summarize_factor_portfolios(
    ls_returns,
    risk_free_rate=0.02,  # 2% risk-free rate
    periods_per_year=252
)

# 4. Display results
print(summary[["factor", "annualized_return", "sharpe", "t_stat", "win_ratio"]].head(10))

# 5. Compute Deflated Sharpe (if testing multiple factors)
if len(factor_cols) > 1:
    summary["deflated_sharpe"] = summary.apply(
        lambda row: compute_deflated_sharpe_ratio(
            sharpe=row["sharpe"],
            n_obs=row["n_periods"],
            n_trials=len(factor_cols),
        ),
        axis=1
    )
```

**CLI-Integration:**

Der neue `analyze_factors`-Command kombiniert C1 (IC-Analyse) und C2 (Portfolio-Analyse) in einem Workflow:

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+vol_liquidity `
  --horizon-days 20 `
  --quantiles 5
```

**Integration mit C1:**
- Nutzt gleiches Input-Format wie C1 (Panel-DF mit `timestamp`, `symbol`, `factor_*`, `fwd_return_*`)
- Ergänzt IC-basierte Evaluierung um Portfolio-basierte Evaluierung
- Kombiniert IC-IR (C1) und Portfolio-Sharpe (C2) für umfassende Faktor-Rankings
- Beide Metriken werden im `analyze_factors`-Workflow parallel berechnet

**Tests:** 
- `tests/test_qa_factor_analysis_c2.py` (21 Tests, marked with `@pytest.mark.advanced`)
- `tests/test_cli_analyze_factors.py` (3 Tests, marked with `@pytest.mark.advanced`)

**Beispiel-Strategie auf Basis der Factor-Rankings:**

Die Ergebnisse von C1/C2 werden in der **Multi-Factor Long/Short Strategy** genutzt:
- **Factor Bundles**: Vordefinierte Kombinationen von Faktoren mit Gewichten basierend auf IC-IR und DSR
- **Multi-Factor Scores**: Gewichtete, normalisierte Faktor-Kombinationen
- **Quantil-basierte Auswahl**: Top/Bottom-Quantile für Long/Short-Positionen
- **Rebalancing**: Konfigurierbare Rebalancing-Frequenzen (täglich, wöchentlich, monatlich)

Siehe [Workflows – Multi-Factor Long/Short Strategy](WORKFLOWS_STRATEGIES_MULTIFACTOR.md) für vollständige Dokumentation des Strategie-Workflows.

#### C3: Event Study Framework ✅ (Completed)

**Status:** ✅ Completed  
**Module:** `src/assembled_core/qa/event_study.py`

##### C3 – Event-Study-Engine: Zielbild & Data Contract

**Ziel:**
Systematische Analyse von Preisreaktionen auf Events (Earnings, Insider-Trades, News, etc.) durch Berechnung von Abnormal Returns, Cumulative Abnormal Returns (CAR) und statistische Signifikanz-Tests.

**Anwendungsfälle:**
- **Earnings Announcements**: Wie reagieren Preise auf Earnings-Releases?
- **Insider Trading**: Gibt es signifikante Preismuster nach Insider-Käufen/Verkäufen?
- **News Events**: Welche News-Typen führen zu abnormalen Returns?
- **Regulatory Events**: Wie wirken sich regulatorische Änderungen aus?

---

**Input-Format für Events:**

**DataFrame-Struktur:**
```python
events_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware) - Event-Zeitpunkt
    - symbol: str - Ticker-Symbol
    - event_type: str - Event-Kategorie (z.B. "earnings", "insider_buy", "insider_sell", "news", "regulatory")
    - event_id: str - Eindeutige Event-ID (z.B. "earnings_2024Q1_AAPL", "insider_12345")
    - payload: dict | str | None - Zusätzliche Event-Metadaten (JSON-serialisierbar)
        ODER: Zusätzliche Spalten je nach event_type (z.B. "earnings_surprise", "insider_role", "news_sentiment")
```

**Beispiel:**
```
timestamp              symbol  event_type    event_id                    payload
2024-01-15 16:00:00+00:00  AAPL   earnings      earnings_2024Q1_AAPL      {"surprise": 0.05, "eps": 2.10}
2024-01-20 09:30:00+00:00  MSFT   insider_buy   insider_12345             {"role": "CEO", "shares": 10000}
2024-02-01 10:00:00+00:00  GOOG   news          news_67890                {"sentiment": "positive", "source": "Reuters"}
```

**Hinweise:**
- `payload` kann ein dict sein (wird als JSON gespeichert) oder als separate Spalten (`earnings_surprise`, `insider_role`, etc.)
- `event_id` muss eindeutig sein (kann später für Deduplizierung verwendet werden)
- `event_type` sollte konsistent sein (empfohlene Werte: `earnings`, `insider_buy`, `insider_sell`, `news`, `regulatory`, `custom`)
- Events können **provider-agnostisch** sein (später aus Finnhub/Fundamentals, jetzt erstmal synthetisch oder CSV)

---

**Input-Format für Preise/Faktoren:**

**Gleiches Panel-Format wie C1/C2:**
```python
prices_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str
    - close: float (oder price_col)
    - Optional: open, high, low, volume
    - Optional: factor_* Spalten (falls Faktor-basierte Abnormal-Return-Berechnung gewünscht)
```

**Sortierung:** Nach `symbol`, dann `timestamp` (aufsteigend)

**Hinweise:**
- Gleicher Data Contract wie bei Factor Analysis (C1/C2)
- Kann direkt aus `LocalParquetPriceDataSource` oder anderen Data Sources geladen werden
- Forward-Returns werden intern berechnet (analog zu C1)

---

**Kern-Funktionen (geplante Interfaces):**

**1. `build_event_window_prices()`**
```python
def build_event_window_prices(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    window_pre: int = 20,  # Tage vor Event
    window_post: int = 40,  # Tage nach Event
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """
    Extrahiert Preis- und Return-Windows um Events.
    
    Args:
        events_df: Events DataFrame (timestamp, symbol, event_type, event_id, ...)
        prices_df: Panel mit Preisen (timestamp, symbol, close, ...)
        window_pre: Anzahl Tage vor Event (default: 20)
        window_post: Anzahl Tage nach Event (default: 40)
        price_col: Spaltenname für Preis (default: "close")
        timestamp_col: Spaltenname für Timestamp (default: "timestamp")
        symbol_col: Spaltenname für Symbol (default: "symbol")
    
    Returns:
        DataFrame mit Spalten:
        - event_id: Event-ID
        - event_type: Event-Typ
        - symbol: Symbol
        - event_timestamp: Event-Zeitpunkt
        - relative_day: Relativer Tag (-window_pre bis +window_post, 0 = Event-Tag)
        - timestamp: Tatsächlicher Timestamp
        - price: Preis an diesem Tag
        - return: Tages-Return (log oder simple)
        - forward_return_*: Forward-Returns für verschiedene Horizonte (optional)
    
    Beispiel:
        event_id              event_type  symbol  relative_day  timestamp              price   return
        earnings_2024Q1_AAPL  earnings    AAPL   -20           2023-12-26 00:00:00+00:00  150.0   0.01
        earnings_2024Q1_AAPL  earnings    AAPL   -19           2023-12-27 00:00:00+00:00  151.5   0.01
        ...
        earnings_2024Q1_AAPL  earnings    AAPL    0            2024-01-15 00:00:00+00:00  160.0   0.02
        ...
        earnings_2024Q1_AAPL  earnings    AAPL    +40           2024-02-24 00:00:00+00:00  165.0   0.01
    """
    pass
```

**2. `compute_event_returns()`**
```python
def compute_event_returns(
    event_windows_df: pd.DataFrame,
    benchmark: str | pd.DataFrame | None = None,  # "market", "sector", oder custom DataFrame
    method: str = "market_model",  # "market_model", "mean_adjust", "factor_model"
    estimation_window: int = 250,  # Tage für Beta-Schätzung (market model)
    risk_free_rate: float = 0.0,
    return_type: str = "log",  # "log" oder "simple"
) -> pd.DataFrame:
    """
    Berechnet Normal- und Abnormal-Returns für Event-Windows.
    
    Args:
        event_windows_df: Output von build_event_window_prices()
        benchmark: Benchmark für Normal-Return-Berechnung
            - "market": Markt-Return (z.B. SPY)
            - "sector": Sektor-Return (falls verfügbar)
            - pd.DataFrame: Custom Benchmark (timestamp, return)
            - None: Mean-Adjustment (Durchschnitts-Return als Normal-Return)
        method: Berechnungsmethode
            - "market_model": CAPM-basiert (R_i = alpha + beta * R_m + epsilon)
            - "mean_adjust": Normal-Return = Durchschnitts-Return (estimation window)
            - "factor_model": Multi-Factor-Modell (falls factor_cols vorhanden)
        estimation_window: Anzahl Tage für Beta-Schätzung (nur bei market_model)
        risk_free_rate: Risk-free Rate (annualisiert, default: 0.0)
        return_type: "log" oder "simple" (default: "log")
    
    Returns:
        DataFrame mit Spalten:
        - event_id, event_type, symbol, event_timestamp, relative_day (wie Input)
        - price, return (wie Input)
        - normal_return: Erwarteter Normal-Return (basierend auf Benchmark/Methode)
        - abnormal_return: Abnormal Return = return - normal_return
        - cumulative_abnormal_return: Kumulierter Abnormal Return (CAR) ab Event-Tag
        - beta: Beta (falls market_model verwendet)
        - alpha: Alpha (falls market_model verwendet)
    
    Beispiel:
        event_id              relative_day  return  normal_return  abnormal_return  cumulative_abnormal_return
        earnings_2024Q1_AAPL  -1            0.01    0.005          0.005           0.005
        earnings_2024Q1_AAPL   0            0.02    0.005          0.015           0.020
        earnings_2024Q1_AAPL  +1            0.01    0.005          0.005           0.025
        ...
    """
    pass
```

**3. `aggregate_event_study()`**
```python
def aggregate_event_study(
    event_returns_df: pd.DataFrame,
    group_by: str | list[str] | None = None,  # "event_type", ["event_type", "symbol"], etc.
    confidence_level: float = 0.95,  # Für Konfidenzintervalle
    test_method: str = "t_test",  # "t_test", "bootstrap", "wilcoxon"
) -> pd.DataFrame:
    """
    Aggregiert Event-Returns über Events hinweg.
    
    Args:
        event_returns_df: Output von compute_event_returns()
        group_by: Gruppierung für Aggregation
            - None: Aggregation über alle Events
            - "event_type": Separate Aggregation pro Event-Typ
            - ["event_type", "symbol"]: Separate Aggregation pro Event-Typ und Symbol
        confidence_level: Konfidenzniveau für Konfidenzintervalle (default: 0.95)
        test_method: Statistische Test-Methode
            - "t_test": t-Test für Mean Abnormal Return (H0: mean = 0)
            - "bootstrap": Bootstrap-basierter Test (robust gegen Non-Normalität)
            - "wilcoxon": Wilcoxon Signed-Rank Test (non-parametric)
    
    Returns:
        DataFrame mit Spalten:
        - group_key: Gruppierungsschlüssel (z.B. event_type oder Kombination)
        - relative_day: Relativer Tag (-window_pre bis +window_post)
        - mean_abnormal_return: Durchschnittlicher Abnormal Return (AAR)
        - std_abnormal_return: Standardabweichung der Abnormal Returns
        - cumulative_abnormal_return: Kumulierter AAR (CAAR)
        - std_caar: Standardabweichung des CAAR
        - n_events: Anzahl Events in dieser Gruppe
        - t_stat: t-Statistik (falls t_test)
        - p_value: p-Wert (Signifikanz-Test)
        - ci_lower: Untere Grenze des Konfidenzintervalls
        - ci_upper: Obere Grenze des Konfidenzintervalls
        - is_significant: Boolean (p < 0.05)
    
    Beispiel:
        group_key    relative_day  mean_abnormal_return  cumulative_abnormal_return  t_stat  p_value  is_significant
        earnings     -1            0.002                0.002                        1.5     0.13     False
        earnings      0            0.015                0.017                        3.2     0.001    True
        earnings     +1            0.003                0.020                        2.1     0.04     True
        ...
    """
    pass
```

**4. Zusätzliche Utility-Funktionen (optional):**
```python
def detect_event_clustering(
    events_df: pd.DataFrame,
    max_window_days: int = 5,
) -> pd.DataFrame:
    """
    Erkennt Event-Clustering (mehrere Events in kurzem Zeitfenster).
    
    Returns DataFrame mit event_id, cluster_id, cluster_size, etc.
    """

def compute_event_correlation(
    event_returns_df: pd.DataFrame,
    event_types: list[str],
) -> pd.DataFrame:
    """
    Berechnet Korrelation zwischen verschiedenen Event-Typen.
    """
```

---

**Integration mit bestehenden Modulen:**

**Events können aus verschiedenen Quellen kommen:**
- **Synthetisch**: Für Testing/Development
- **CSV/Parquet**: Manuell erstellte Event-Dateien
- **Finnhub API**: Earnings, Insider Trades, News (später)
- **Fundamentals**: Earnings-Daten aus Fundamentals-Datenbanken (später)
- **Bestehende Features**: `insider_features.py`, `shipping_features.py` können als Event-Quellen dienen

**Preise/Faktoren:**
- Gleicher Data Contract wie C1/C2
- Kann direkt aus `LocalParquetPriceDataSource` geladen werden
- Forward-Returns werden intern berechnet (analog zu `add_forward_returns()`)

---

**Workflow-Beispiel (geplant):**

```python
from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)

# 1. Events laden (z.B. aus CSV oder API)
events_df = pd.read_csv("data/events/earnings_2024.csv")

# 2. Preise laden (gleicher Data Contract wie C1/C2)
prices_df = load_prices_from_local(...)

# 3. Event-Windows extrahieren
event_windows = build_event_window_prices(
    events_df,
    prices_df,
    window_pre=20,
    window_post=40
)

# 4. Abnormal Returns berechnen
event_returns = compute_event_returns(
    event_windows,
    benchmark="market",  # oder custom DataFrame
    method="market_model"
)

# 5. Aggregieren und statistische Tests
aggregated = aggregate_event_study(
    event_returns,
    group_by="event_type",
    test_method="t_test"
)

# 6. Ergebnisse analysieren
print(aggregated[aggregated["relative_day"] == 0])  # Event-Tag
print(aggregated[aggregated["relative_day"].between(0, 5)])  # Erste 5 Tage nach Event
```

---

**Nächste Schritte:**
1. Implementierung von `build_event_window_prices()`
2. Implementierung von `compute_event_returns()` (mit market_model, mean_adjust, factor_model)
3. Implementierung von `aggregate_event_study()` (mit t_test, bootstrap, wilcoxon)
4. Tests für alle drei Funktionen
5. Integration mit bestehenden Event-Features (`insider_features`, `shipping_features`)
6. CLI-Integration (optional, z.B. `analyze_events` Command)

#### C4: Factor Correlation & Selection Tools (Renumbered)
- Factor correlation matrix computation (pairwise correlations)
- Factor redundancy detection (highly correlated factors)
- Factor selection algorithms (mutual information, univariate selection, RFE)
- Factor attribution analysis (which factors contribute most to returns?)
- Factor stability metrics (factor performance across different market regimes)

**Integration with Existing Backend:**
- Extends Phase 4 (Backtesting) with factor evaluation capabilities
- Enhances Phase 9 (Model Governance) with systematic factor validation
- Supports research workflow in `research/` directory

---

### Phase D: Regime Models & Risk 2.0

**Goal:** Build advanced risk modeling capabilities with market regime detection and adaptive risk management. Regime-aware strategies can adjust exposure, position sizing, and factor selection based on market conditions.

**Sprints:**

#### D1: Regime Models & Risk Overlay (Design) 📋

**Status:** 📋 Design Phase  
**Design Document:** [Regime Models D1 Design](REGIME_MODELS_D1_DESIGN.md)

**Goal:** Develop a regime detection system that identifies market regimes (bull, bear, sideways, crisis, reflation) and provides a risk overlay for adaptive strategies.

**Scope:**
- **No new APIs**: Uses existing local alt-data snapshots and factors
- **Regime Detection**: Daily regime labels based on macro factors, market breadth, and volatility
- **Risk Overlay**: Regime → risk parameter mapping (exposure limits, net exposure targets, position sizing)

**Available Inputs:**
- Macro factors: `macro_growth_regime`, `macro_inflation_regime`, `macro_risk_aversion_proxy`
- Market breadth: `fraction_above_ma_50`, `ad_line`, `risk_on_off_score`
- Volatility: `rv_20`, `vov_20_60`
- Trend indicators: `trend_strength_50`, `trend_strength_200` (optional)

**Planned Functions:**
- `build_regime_state()`: Compute daily regime labels and sub-scores
- `compute_regime_transition_stats()`: Analyze regime transitions and durations
- `evaluate_factor_by_regime()`: Evaluate factor effectiveness (IC, Sharpe) by regime
- `apply_risk_overlay()`: Adjust positions based on regime-specific risk limits

**Risk Overlay Concept:**
- Mapping from regime to risk parameters:
  - `max_gross_exposure`: Maximum gross exposure (long + short)
  - `target_net_exposure`: Target net exposure (long - short)
  - `max_single_position_weight`: Maximum weight per position
  - `allow_trading`: Whether to allow new positions
- Example: `bull` regime → high exposure, net long; `crisis` regime → minimal exposure, defensive

**Integration Points:**
1. **Multi-Factor Strategy**: Regime-adaptive exposure control and position sizing
2. **Event Studies**: CAAR analysis separated by regime (e.g., earnings events in bull vs. bear markets)
3. **Factor Analysis**: IC/Sharpe evaluation by regime (which factors work in which regimes?)

**Implementation Steps (D1.1-D1.4):**
- D1.1: Core regime module + data contracts ✅
- D1.2: Regime evaluation vs. factors ✅
- D1.3: Integration in multi-factor strategy as risk overlay ✅
- D1.4: Tests + documentation ✅

**See:**
- [Regime Models D1 Design Document](REGIME_MODELS_D1_DESIGN.md) – Detailed design, data contracts, and implementation plan
- [Regime Models & Risk Overlay Workflow](WORKFLOWS_REGIME_MODELS_AND_RISK.md) – Step-by-step workflow guide with examples

#### D2: Risk 2.0 & Attribution ✅ (Completed)

**Status:** ✅ Completed  
**Design Document:** [Risk 2.0 & Attribution D2 Design](RISK_2_0_D2_DESIGN.md)  
**Workflow Guide:** [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md)

**Goal:** Erweiterte Risk-Analyse und Performance-Attribution für Backtests mit Segmentierung nach Regime, Faktor-Gruppen und Universes.

**Scope:**
- **Erweiterte Risk-Metriken**: Skewness, Kurtosis, Expected Shortfall (ES), Tail Ratio (zusätzlich zu bestehenden Metriken in `qa/metrics.py`)
- **Exposure-Analyse**: Gross/Net Exposure-Zeitreihen, HHI Concentration, Turnover über Zeit
- **Risiko nach Regime**: Sharpe, Volatilität, Max Drawdown, Calmar pro identifiziertem Regime (Verknüpfung mit `regime_state_df` aus D1)
- **Risiko nach Faktor-Gruppen**: Performance-Attribution nach Faktor-Kategorien (Trend, Vol/Liq, Earnings, Insider, News/Macro)
- **Risk Report**: Zentrale Sammlung aller Risk-Analysen als Markdown-Report und CSVs

**Data Contracts:**
- Input: `equity_curve_df`, `positions_df`, `trades_df` (optional), `regime_state_df` (optional), `factor_panel_df` (optional)
- Output: `exposure_timeseries_df`, `risk_by_regime_df`, `risk_by_factor_group_df`, Risk-Report (Markdown + CSVs)

**Planned Functions:**
- `compute_basic_risk_metrics()`: Erweiterte Risk-Metriken (Skewness, Kurtosis, ES, Tail Ratio)
- `compute_exposure_timeseries()`: Exposure-Zeitreihen (Gross/Net Exposure, HHI, Turnover)
- `compute_risk_by_regime()`: Risk-Metriken pro Regime
- `compute_risk_by_factor_group()`: Performance-Attribution nach Faktor-Gruppen
- `generate_risk_report()`: Zentrale Report-Generierung

**Integration:**
- Post-Processing-Schritt nach `run_backtest_strategy.py` (optionaler CLI-Flag `--with-risk-report`)
- Separates Script `scripts/generate_risk_report.py` für nachträgliche Analyse
- Erweiterung von `BacktestResult` um optionales `risk_report`-Feld

**Implementation Steps (D2.1-D2.4):**
- D2.1: Risk-Metrics-Core-Modul (`risk_metrics_advanced.py`)
- D2.2: Risk-by-Regime & Risk-by-Factor-Group
- D2.3: Risk-Report-Generierung (`generate_risk_report()`, `scripts/generate_risk_report.py`)
- D2.4: CLI-Integration & Workflows

**Implementation:**
- ✅ D2.1: Risk-Metrics-Core-Modul (`src/assembled_core/risk/risk_metrics.py`)
- ✅ D2.2: Risk-by-Regime & Risk-by-Factor-Group
- ✅ D2.3: Risk-Report-Generierung (`scripts/generate_risk_report.py`, CLI-Integration)
- ✅ D2.4: Tests (`tests/test_risk_risk_metrics.py`, `tests/test_cli_risk_report.py`)

**See:**
- [Risk 2.0 & Attribution D2 Design Document](RISK_2_0_D2_DESIGN.md) for detailed design, data contracts, and implementation plan.
- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) for usage guide and examples.

#### D3: Adaptive Factor Selection (Future)

- Select factors based on current regime
- Regime-specific factor bundles
- Dynamic factor weighting by regime

#### D4: Regime-Aware Risk Models (Future)

- Regime-conditional VaR/CVaR (different risk models per regime)
- Dynamic position sizing based on regime (higher risk in low-vol regimes)
- Regime-aware portfolio constraints (reduce leverage in high-vol regimes)
- Stress testing with regime-specific scenarios
- Integration with existing risk engine (`qa/risk_metrics.py`, `qa/scenario_engine.py`)

**Integration with Existing Backend:**
- Extends Phase 8 (Risk Engine) with regime-aware modeling
- Enhances `qa/scenario_engine.py` with regime-specific scenarios
- Provides adaptive risk management for portfolio layer

---

### Phase E: ML Validation, Explainability & TCA

**Goal:** Enhanced model validation, explainability tools, and transaction cost analysis for production readiness.

**Sprints:**

#### E1: ML Validation & Model Comparison ✅ (Completed)

**Status:** Implemented and tested

**Module:** `src/assembled_core/ml/factor_models.py`  
**CLI:** `scripts/cli.py ml_validate_factors`

**Implementation:**
- **Time-Series Cross-Validation:** Expanding and rolling window splits to prevent data leakage
- **Model Types:** Linear, Ridge, Lasso, Random Forest regression models
- **Comprehensive Metrics:**
  - Classical ML metrics (MSE, MAE, R², Directional Accuracy)
  - Factor-specific metrics (IC, Rank-IC, IC-IR, Rank-IC-IR)
  - Portfolio metrics (Long/Short Sharpe, Return, Volatility)
- **Auto Feature Detection:** Automatically detects `factor_*` columns from factor panels
- **Robust Error Handling:** Clear error messages for missing data, invalid configurations

**Integration:**
- Works with factor panels from Phase C1/C2 (factor analysis workflows)
- Supports all factor types (Core TA/Price, Vol/Liquidity, Alt-Data)
- Outputs standardized CSV and Markdown reports for easy comparison

**Usage:**
```powershell
python scripts/cli.py ml_validate_factors `
  --factor-panel-file output/factor_analysis/core_20d_factors.parquet `
  --label-col fwd_return_20d `
  --model-type ridge `
  --n-splits 5
```

**Documentation:**
- Design document: [ML Validation E1 Design](ML_VALIDATION_E1_DESIGN.md)
- Workflow guide: [Workflows – ML Validation & Model Comparison](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md)
- Tests: `tests/test_ml_factor_models.py`, `tests/test_cli_ml_validation.py`

**Next Steps (Future):**
- Walk-forward analysis framework (rolling window validation)
- Out-of-sample testing protocols (strict train/test separation)
- Overfitting detection (performance degradation on OOS data)
- Model stability metrics (performance consistency over time)
- Automated model ranking and selection

#### E2: Explainability & Feature Importance ✅ (Completed)

**Status:** Implemented and integrated with E1

**Module:** `src/assembled_core/ml/explainability.py`

**Features:**
- Model-based feature importance (coefficients for linear models, `feature_importances_` for tree models)
- Permutation importance (model-agnostic)
- Global feature importance aggregation across multiple models
- Automatic integration in ML validation reports

**Integration:**
- Automatically computed after `run_time_series_cv()` in `run_ml_factor_validation.py`
- Included in Model Zoo comparisons
- Outputs: Feature importance CSV, permutation importance CSV, enhanced Markdown reports

**References:**
- Design: [ML Validation & Model Comparison Design (E1)](ML_VALIDATION_E1_DESIGN.md) (E2 section)
- Workflows: [ML Validation & Model Comparison Workflows](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md) (Feature Importance section)

#### E3: ML Alpha Factor & Strategy Integration ✅ (Completed)

**Status:** Implemented and tested

**Scope:**
- Generate ML alpha factors (`ml_alpha_{model_type}_{horizon}d`) from trained ML models
- Integrate ML alpha factors into factor bundles
- Enable ML alpha factors in multi-factor strategies

**Module:** `research/ml/export_ml_alpha_factor.py`  
**Bundles:** `config/factor_bundles/ai_tech_ml_alpha_bundle.yaml`, `config/factor_bundles/ai_tech_core_ml_bundle.yaml`

**Features:**
- Trains ML models via time-series cross-validation (reuses E1 infrastructure)
- Merges predictions (`y_pred`) back into factor panels as `ml_alpha_*` columns
- Supports all model types: Linear, Ridge, Lasso, Random Forest
- Preserves original factor columns (enables mixed bundles)
- Only test samples have predictions (training samples remain NaN to prevent look-ahead bias)

**Integration:**
- Uses factor panels from `export_factor_panel_for_ml.py`
- Uses ML models and validation from `run_ml_factor_validation.py` / `model_zoo_factor_validation.py`
- Factor bundles work with existing `multifactor_long_short` strategy (no code changes)
- Pure ML bundles (100% ML alpha) and mixed bundles (Core + ML alpha) available

**Usage:**
```bash
# Export ML alpha factor
python research/ml/export_ml_alpha_factor.py \
  --factor-panel-file output/factor_panels/factor_panel_ai_tech_core_20d_1d.parquet \
  --label-col fwd_return_20d \
  --model-type ridge \
  --model-param alpha=0.1

# Use in strategy
python scripts/run_backtest_strategy.py \
  --strategy multifactor_long_short \
  --bundle-path config/factor_bundles/ai_tech_ml_alpha_bundle.yaml \
  --factor-file output/ml_alpha_factors/ml_alpha_panel_ridge_20d.parquet
```

**References:**
- Design: [ML Alpha Factor & Strategy Integration Design (E3)](ML_ALPHA_E3_DESIGN.md)
- ML Validation Workflows: [ML Validation & Model Comparison Workflows](WORKFLOWS_ML_VALIDATION_AND_MODEL_COMPARISON.md) (includes ML Alpha export workflow)
- Strategy Workflows: [Multi-Factor Strategy Workflows](WORKFLOWS_STRATEGIES_MULTIFACTOR.md) (includes ML Alpha usage section)

#### E4: Transaction Cost Analysis (TCA) ✅ (Completed)

**Status:** Implemented and tested

**Module:** `src/assembled_core/risk/transaction_costs.py`

**Goal:** Einfache Transaction Cost Analysis für Backtest-Strategien mit Schätzung von Execution-Kosten und deren Auswirkung auf Net-Returns und Performance-Metriken.

**Features:**
- Per-Trade Cost Estimation (Commission + Spread/2 + Slippage)
- Cost aggregation per period (daily, weekly, monthly)
- Cost-adjusted risk metrics (net returns, net Sharpe, net Sortino)
- Comparison of gross vs. net performance metrics
- Integration with backtest outputs (trades.csv/parquet)
- Integration with equity curves for net return calculation

**Integration:**
- Uses trades from `backtest_engine.py` (BacktestResult.trades)
- Integrates with `risk_metrics.py` for net return metrics
- CLI script: `scripts/generate_tca_report.py` / `scripts/cli.py tca_report`

**Usage:**
```bash
# Generate TCA report
python scripts/cli.py tca_report --backtest-dir output/backtests/experiment_123/
```

**Outputs:**
- `tca_trades.csv`: Detailed TCA for each trade
- `tca_summary.csv`: Daily aggregated TCA metrics
- `tca_risk_summary.csv`: Cost-adjusted risk metrics (if equity curve available)
- `tca_report.md`: Comprehensive Markdown report

**References:**
- Design: [Transaction Cost Analysis Design (E4)](TRANSACTION_COSTS_E4_DESIGN.md)
- Workflows: [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) (includes TCA section)

**Future Enhancements:**
- Adaptive cost models based on volatility/liquidity factors
- Regime-specific cost estimates
- Market impact models (based on trade size relative to volume)
- Integration with live execution layer (post-trade TCA)

#### P4: Batch Runner & Parallelisierung ✅ (Design implemented, Runner available)

**Status:** Design und erste Implementierung abgeschlossen, Batch-Runner produktiv nutzbar (seriell), Parallelisierung vorbereitet.

**Module / Scripts:**
- Batch-Runner Script: `scripts/batch_backtest.py`
- CLI-Subcommand: `python scripts/cli.py batch_backtest --config-file <config.yaml>`
- Profiling-Integration: `scripts/profile_jobs.py` (Job `BATCH_BACKTEST`)

**Goal:** Viele Backtests (Parameter-Sweeps, Bundle-/Universe-Vergleiche, Regime-On/Off) systematisch und reproduzierbar ausführen, basierend auf einer YAML/JSON-Config.

**Features:**
- YAML/JSON Batch-Config mit Defaults + Liste von Runs (Strategie, Bundle, Zeitraum, Universe).
- Serieller Batch-Runner mit vorbereitetem Interface für spätere Parallelisierung (P4+).
- Saubere Output-Struktur je Run (`runs/{run_id}/backtest/...`) und zentrale Summary:
  - `batch_summary.csv`
  - `batch_summary.md`
- Integration mit optimierter Backtest-Engine (P3) und Risiko-/TCA-Workflows (D2/E4).

**Workflow-Dokumentation:**
- [Batch Backtests & Parallelization Workflow](WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md)

---

#### B1: Deterministic Backtests & Seeds ✅ (Completed)

**Status:** Implemented and tested

**Goal:** Sicherstellen, dass Backtests bei gleicher Codebasis, gleichen Daten und gleicher Konfiguration vollständig reproduzierbar sind.

**Features:**
- Zentrales Seed-Management in `src/assembled_core/utils/random_state.py`
- Utilities: `set_global_seed(seed: int)` und `seed_context(seed: int)`
- Deterministische Backtests für gegebene Seeds:
  - Wiederholte Runs mit gleichem Seed liefern identische Equity-Kurven und Trades
  - Verschiedene Seeds können unterschiedliche Pfade erzeugen, sind aber jeweils deterministisch
- Integration mit der optimierten Backtest-Engine (P3) und Batch-Runner (P4)

**Testing:**
- `tests/test_utils_random_state.py`: Verifiziert deterministische Zufallsfolgen für `random` und `numpy`
- `tests/test_backtest_determinism.py`: Verifiziert deterministische Backtests für feste Seeds

**Guarantee:**
- Gleiches Repo + gleiche Daten + gleiche Backtest-Konfiguration + gleicher Seed ⇒ reproduzierbarer Backtest (Equity, Trades, Metriken)

**References:**
- Design: [Backtest B1 Unified Pipeline Design](BACKTEST_B1_UNIFIED_PIPELINE_DESIGN.md)

---

#### B2: Point-in-Time Safety & Latency for Alt-Data ✅ (Completed)

**Status:** Implemented and tested

**Goal:** Sicherstellen, dass Alt-Data-Faktoren nur Informationen verwenden, die zum jeweiligen Backtest-Datum tatsächlich verfügbar waren, um Look-Ahead-Bias zu verhindern.

**Features:**
- Explizite `event_date` und `disclosure_date` Felder in allen Alt-Data-Event-Contracts
- Feature-Builder unterstützen `as_of` Parameter für PIT-sichere Faktor-Berechnung
- Automatische Filterung: Events mit `disclosure_date > as_of` werden ausgeschlossen
- Konservative Defaults: Falls `disclosure_date` fehlt, wird `event_date` verwendet (kein Look-Ahead, aber möglicherweise keine echte Latenz-Modellierung)

**Implementation:**
- Daten-Layer: `src/assembled_core/data/altdata/finnhub_events.py`, `finnhub_news_macro.py`
  - Erzeugen `event_date` und `disclosure_date` Spalten in Event-DataFrames
- Feature-Builder: `src/assembled_core/features/altdata_earnings_insider_factors.py`, `altdata_news_macro_factors.py`
  - `build_earnings_surprise_factors(..., as_of=None)`
  - `build_insider_activity_factors(..., as_of=None)`
  - `build_news_sentiment_factors(..., as_of=None)`
  - Alle Builder filtern Events nach `disclosure_date <= as_of`, falls `as_of` gesetzt ist

**Latency Models:**
- Insider Trades: `disclosure_date` = Form 4 filing date (typisch T+2 nach Trade)
- Earnings: `disclosure_date` = Announcement timestamp (meist same day)
- News Sentiment: `disclosure_date` = Daily aggregation date (end-of-day snapshot)
- Congress Trades: `disclosure_date` = PTR publication date (kann Wochen dauern)

**Testing:**
- `tests/test_point_in_time_altdata.py`: Umfassende PIT-Tests für alle Alt-Data-Kategorien
  - Verifiziert, dass Events vor `disclosure_date` nicht in Faktoren auftauchen
  - Verhindert Look-Ahead-Bias durch Vergleich mit/ohne verzögerte Events
  - Mini-Backtest-Szenario: Faktoren erscheinen erst nach `disclosure_date`

**Guarantee:**
- Features berechnet mit `as_of=T` enthalten keine Informationen von Events mit `disclosure_date > T`
- Gleiche Events + gleicher `as_of` = identische Faktoren (deterministisch)

**References:**
- Design: [Point-in-Time and Latency B2 Design](POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md)
- Workflow: [Point-in-Time and Latency Documentation](POINT_IN_TIME_AND_LATENCY.md)
- Tests: `tests/test_point_in_time_altdata.py`

---

#### B3: Walk-Forward Analysis & Regime-Based Performance Evaluation ✅ (Completed)

**Status:** ✅ Completed  
**Design Document:** [Walk-Forward & Regime Analysis B3 Design](WALK_FORWARD_AND_REGIME_B3_DESIGN.md)

#### B4: Deflated Sharpe & Factor-Zoo Protection ✅ (Completed)

**Status:** ✅ Completed  
**Design Document:** [Deflated Sharpe B4 Design](DEFLATED_SHARPE_B4_DESIGN.md)

**Goal:** Protect against the "Factor Zoo" problem by adjusting Sharpe Ratios for multiple testing and selection bias.

**Implementation:**
- **Core Functions:** `deflated_sharpe_ratio()` and `deflated_sharpe_ratio_from_returns()` in `src/assembled_core/qa/metrics.py`
- **Formula:** Based on Bailey & Lopez de Prado (2014), adjusts observed Sharpe for:
  - Multiple testing (False Discovery Rate)
  - Non-normal return distributions (skewness, kurtosis)
- **Integration:** All experiment pipelines now compute and report `n_tests`, `sharpe_raw`, and `sharpe_deflated`

**Experiment Pipelines with Deflated Sharpe:**
1. **ML Model Zoo** (`research/ml/model_zoo_factor_validation.py`):
   - `ml_model_zoo_summary.csv`: Contains `n_tests`, `ls_sharpe_raw`, `ls_sharpe_deflated`
   - `n_tests` = number of models tested in the zoo
   - `n_obs` = number of unique timestamps in predictions (or `n_samples` as fallback)

2. **ML Factor Validation** (`scripts/run_ml_factor_validation.py`):
   - `ml_portfolio_metrics_*.csv`: Contains `n_tests`, `ls_sharpe_raw`, `ls_sharpe_deflated`
   - `n_tests` = 1 (conservative, single model validation)
   - `n_obs` = number of unique timestamps in predictions (or `n_samples` as fallback)

3. **Factor Portfolio Summaries** (`src/assembled_core/qa/factor_analysis.py::summarize_factor_portfolios`):
   - Portfolio summary DataFrames contain `deflated_sharpe` column
   - `n_tests` = number of factors tested (computed after aggregation)
   - `n_obs` = `n_periods` (number of periods per factor)

**Tests:**
- `tests/test_qa_deflated_sharpe.py`: 8 comprehensive tests covering edge cases, monotonicity, and integration
- All tests marked with `@pytest.mark.advanced`

**References:**
- Bailey, D. H., & Lopez de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting and non-normality. Journal of Portfolio Management, 40(5), 94-107.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. Review of Financial Studies, 29(1), 5-68.

**Goal:** Out-of-sample stability testing (Walk-Forward) and regime-aware performance evaluation to ensure strategies work across different market conditions.

**Features:**
- **Walk-Forward Framework:**
  - Rolling and expanding window strategies
  - Train/test split generation with configurable step sizes
  - Aggregated summary metrics (OOS Sharpe, win rate, stability measures)
  - Integration with `backtest_engine.py`
- **Regime-Based Analysis (Skeleton):**
  - Simplified regime classification from index returns
  - Extended metrics summarization by regime (including trade-level metrics)
  - Regime transition analysis

**Implementation:**
- Walk-Forward Module: `src/assembled_core/qa/walk_forward.py` (fully implemented)
  - `WalkForwardConfig`: Configuration dataclass
  - `run_walk_forward_backtest()`: Main execution function
  - `WalkForwardResult`: Aggregated results with summary metrics
- Regime Analysis Module: `src/assembled_core/risk/regime_analysis.py` (skeleton)
  - `RegimeConfig`: Configuration for simplified regime classification
  - `classify_regimes_from_index()`: TODO - Simplified regime detection
  - `summarize_metrics_by_regime()`: TODO - Extended metrics by regime
  - `compute_regime_transitions()`: TODO - Transition analysis
- Existing Integration:
  - `build_regime_state()` in `regime_models.py` (D1) - Full regime detection
  - `compute_risk_by_regime()` in `risk_metrics.py` (D2) - Basic regime metrics

**Usage:**
```python
from src.assembled_core.qa.walk_forward import WalkForwardConfig, run_walk_forward_backtest

cfg = WalkForwardConfig(
    train_size=252,  # 1 year
    test_size=63,    # 1 quarter
    step_size=21,    # 1 month
    window_type="rolling",
)

result = run_walk_forward_backtest(
    prices=prices,
    signal_fn=my_signal_fn,
    position_sizing_fn=my_sizing_fn,
    config=cfg,
    start_capital=100000.0,
)
```

**Testing:**
- Walk-Forward: Unit tests for window generation, integration tests with backtest engine
- Regime Analysis: Tests for regime classification and metrics summarization

---

### Research Playbooks

#### R1: AI/Tech Multi-Factor + ML Alpha + Regime + Risk – End-to-End Research Playbook ✅ (Completed)

**Status:** Implemented and tested

**Module:** `research/playbooks/ai_tech_multifactor_mlalpha_regime_playbook.py`

**Goal:** Automatisiert den vollständigen Research-Workflow von der Factor-Panel-Erstellung bis zum Risk-Report für AI/Tech-Universe mit ML-Alpha-Integration und Regime-Analyse.

**Workflow:**
1. Factor Panel Export (mit Forward-Returns)
2. ML Model Zoo Comparison (systematischer Modellvergleich)
3. Best Model Selection (basierend auf IC-IR, Test-R²)
4. ML Alpha Factor Export (mit bestem Modell)
5. Backtests mit Multiple Bundles (Core-only, Core+ML, ML-only)
6. Risk Reports Generation (mit optionaler Regime-Attribution)
7. Research Summary (konsolidierter Markdown-Report)

**Features:**
- Vollständige Automatisierung des Research-Prozesses
- Konfigurierbar via `PlaybookConfig` Dataclass
- Robustes Error-Handling und Logging
- Unterstützt Regime-Attribution (optional)
- Generiert konsolidierten Research-Summary-Report

**Usage:**
```python
from research.playbooks.ai_tech_multifactor_mlalpha_regime_playbook import main

# Run with default config (AI/Tech universe)
main()
```

**Outputs:**
- Factor Panel: `output/factor_panels/factor_panel_*.parquet`
- ML Model Zoo: `output/ml_validation/model_zoo/ml_model_zoo_summary.csv`
- ML Alpha Panel: `output/ml_alpha_factors/ml_alpha_panel_*.parquet`
- Backtest Results: `output/backtests/{bundle_name}_{timestamp}/`
- Risk Reports: `output/backtests/{bundle_name}_{timestamp}/risk_report.md`
- Research Summary: `output/risk_reports/research_summaries/research_summary_*.md`

**References:**
- Implementation: `research/playbooks/ai_tech_multifactor_mlalpha_regime_playbook.py`
- Tests: `tests/test_research_playbook_ai_tech.py`
- Real-time TCA (estimate costs before placing orders)
- Historical TCA analysis (analyze actual execution costs)
- Cost-aware position sizing (reduce position size if costs are high)
- Integration with execution layer (`execution/pre_trade_checks.py`)

**Integration with Existing Backend:**
- Extends Phase 7 (ML Meta-Layer) with explainability and advanced validation
- Enhances Phase 9 (Model Governance) with rigorous validation protocols
- Improves Phase 10 (Paper-Trading & OMS) with realistic cost modeling

---

## Integration with Existing Roadmap

**Mapping to Backend Phases:**

| Factor Labs Phase | Related Backend Phases | Integration Points |
|-------------------|------------------------|-------------------|
| **Phase A** (TA/Price Factors) | Phase 4 (Backtesting), Phase 7 (ML Meta-Layer) | Extends `features/ta_features.py`, provides richer features for backtests and ML models |
| **Phase B** (Alt-Data Factors) | Phase 6 (Event Features) | Enhances existing insider/shipping features, adds congressional trading and news sentiment |
| **Phase C** (Factor Analysis) | Phase 4 (Backtesting), Phase 9 (Model Governance) | Provides factor evaluation tools for research and validation |
| **Phase D** (Regime & Risk) | Phase 8 (Risk Engine) | Extends `qa/risk_metrics.py` and `qa/scenario_engine.py` with regime-aware modeling |
| **Phase E** (ML Validation & TCA) | Phase 7 (ML Meta-Layer), Phase 9 (Model Governance), Phase 10 (Paper-Trading) | Enhances model validation, adds explainability, improves cost modeling |

**Shared Infrastructure:**
- Backtest engine (`qa/backtest_engine.py`) used for factor evaluation
- Experiment tracking (`qa/experiment_tracking.py`) for factor research experiments
- Data source abstraction (`data/data_source.py`) for loading historical and live data
- Research folder structure (`research/`) for notebooks and experiments

---

## Prioritization & Recommended Order

**Recommended Implementation Sequence:**

1. **Phase A (A1–A3)**: Start with core technical indicators library
   - Provides immediate value for strategy development
   - Foundation for all other phases (factors are used everywhere)
   - Relatively straightforward implementation (well-defined TA indicators)

2. **Phase C1 (IC Engine)**: Early factor evaluation capability
   - Enables systematic evaluation of existing and new factors
   - Critical for research workflow (which factors actually work?)
   - Relatively lightweight (correlation analysis)

3. **Phase B1/B2 (Insider/Congress)**: Enhanced alternative data
   - Builds on existing Phase 6 features
   - High potential alpha (alternative data is often underutilized)
   - Congressional trading is a relatively new data source

4. **Phase D1 (Regime Detection)**: Advanced risk modeling foundation
   - Enables adaptive strategies and risk management
   - Important for production readiness
   - Builds on existing risk metrics

5. **Phase E (ML Validation & Explainability)**: Production readiness
   - Critical for model governance and regulatory compliance
   - Explainability is increasingly important for ML models
   - TCA improves realistic backtesting and execution

**Dependencies:**
- Phase A should be completed before Phase C (need factors to evaluate)
- Phase C1 (IC Engine) should be completed before Phase C2 (Factor Report Workflow) and Phase C4 (Factor Selection)
- Phase D1 (Regime Detection) should be completed before Phase D2/D3 (regime identification before adaptive models)
- Phase E1 (Model Validation) should be completed before E2/E3 (validation before explainability/costs)

---

## Research Workflow Integration

**Factor Labs experiments follow the standard research workflow:**

1. **Hypothesis**: "Does RSI mean-reversion outperform trend-following in sideways markets?"
2. **Setup**: Define factors, time periods, universe, backtest parameters
3. **Experiment**: Implement factor in research notebook, run backtests
4. **Evaluation**: Use IC Engine (Phase C1) to evaluate factor effectiveness
5. **Documentation**: Track experiment with experiment tracking system

**Integration with Research Roadmap:**
- See `docs/RESEARCH_ROADMAP.md` for overall research strategy
- Factor Labs provides systematic tools for factor development
- Research Roadmap covers broader strategy development (Factor Labs is a subset)

---

## Related Documentation

- [Research Roadmap](RESEARCH_ROADMAP.md) - Overall research strategy and backlog
- [Backend Roadmap](BACKEND_ROADMAP.md) - Core backend development phases
- [Backend Architecture](ARCHITECTURE_BACKEND.md) - System architecture overview
- [Workflows – Backtests & Meta-Model Ensemble](WORKFLOWS_BACKTEST_AND_ENSEMBLE.md) - How to run backtests and use meta-models

---

**Next Steps:**
1. ✅ Phase A1 (Core TA/Price Factors) - **Completed**
2. ✅ Phase A2 (Liquidity & Volatility Factors) - **Completed**
3. ✅ Phase A3 (Market Breadth & Risk-On/Risk-Off Indicators) - **Completed**
4. ✅ Phase C1 (Information Coefficient Engine) - **Completed**
5. ✅ Phase C2 (Factor Report Workflow) - **Completed**
6. ✅ Phase C3 (Event Study Framework) - **Completed**
7. ⏳ Start with Phase C4 (Factor Correlation & Selection Tools) or Factor Engineering Pipeline (A3 extension)
8. ⏳ Enhance alternative data sources (Phase B1/B2)

