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

---

### Phase B: Alt-Data Factors 2.0

**Goal:** Advanced integration and enhancement of alternative data sources (insider trading, congressional trading, shipping, news).

**Sprints:**

#### B1: Insider Trading 2.0
- Enhanced insider transaction types classification (Open Market, Private Placement, Exercise, etc.)
- Insider clustering analysis (multiple insiders trading simultaneously)
- Insider historical performance tracking (success rate of individual insiders)
- Insider sentiment scoring (aggregate buy/sell signals with confidence weights)
- Integration with Phase 6 existing insider features

#### B2: Congressional Trading & Public Disclosures
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

#### C3: Event Study Framework (Renumbered)
- Event definition and matching (earnings, insider trades, news events)
- Pre/post event return analysis (cumulative abnormal returns, CAR)
- Statistical significance testing (t-tests, bootstrap methods)
- Event clustering detection (multiple events in short time windows)
- Integration with backtest engine for event-driven strategy testing

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

**Goal:** Advanced risk modeling with market regime detection and adaptive risk management.

**Sprints:**

#### D1: Market Regime Detection
- Regime classification (Bull, Bear, High Volatility, Low Volatility)
- Hidden Markov Models (HMM) for regime identification
- Macro indicators integration (VIX, yield curve, economic indicators)
- Regime transition probabilities and persistence
- Regime-aware performance attribution (strategy performance by regime)

#### D2: Adaptive Risk Models
- Regime-conditional VaR/CVaR (different risk models per regime)
- Dynamic position sizing based on regime (higher risk in low-vol regimes)
- Regime-aware portfolio constraints (reduce leverage in high-vol regimes)
- Stress testing with regime-specific scenarios
- Integration with existing risk engine (`qa/risk_metrics.py`, `qa/scenario_engine.py`)

#### D3: Risk Attribution & Decomposition
- Factor-based risk attribution (which factors drive portfolio risk?)
- Sector risk attribution (concentration risk analysis)
- Time-varying risk decomposition (risk changes over time)
- Risk budgeting tools (allocate risk budget across factors/strategies)
- Risk-adjusted performance metrics (Information Ratio, Risk-Adjusted Returns)

**Integration with Existing Backend:**
- Extends Phase 8 (Risk Engine) with regime-aware modeling
- Enhances `qa/scenario_engine.py` with regime-specific scenarios
- Provides adaptive risk management for portfolio layer

---

### Phase E: ML Validation, Explainability & TCA

**Goal:** Enhanced model validation, explainability tools, and transaction cost analysis for production readiness.

**Sprints:**

#### E1: Advanced Model Validation
- Walk-forward analysis framework (rolling window validation)
- Time-series cross-validation (avoiding data leakage)
- Out-of-sample testing protocols (strict train/test separation)
- Overfitting detection (performance degradation on OOS data)
- Model stability metrics (performance consistency over time)

#### E2: Model Explainability
- SHAP values integration for meta-models (feature importance, interaction effects)
- Partial dependence plots (marginal effect of individual features)
- Feature importance rankings (which features drive predictions?)
- Model decision trees visualization (for tree-based models)
- Counterfactual analysis (what would prediction be if feature X changed?)

#### E3: Transaction Cost Analysis (TCA)
- Execution cost modeling (slippage, market impact, spread costs)
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
6. ⏳ Start with Phase C3 (Event Study Framework, renumbered) or Factor Engineering Pipeline (A3 extension)
7. ⏳ Enhance alternative data sources (Phase B1/B2)

