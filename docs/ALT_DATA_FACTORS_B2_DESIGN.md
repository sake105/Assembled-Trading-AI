# Alt-Data Factors – Sprint B2 Design: News, Sentiment & Macro Data

**Last Updated:** 2025-12-09  
**Status:** Design Phase  
**Sprint:** B2 – News, Sentiment & Macro Alt-Daten Integration

---

## Overview

Sprint B2 extends the Alt-Data integration from B1 (Earnings & Insider) to include **News**, **Sentiment**, and **Macro-Economic** data from Finnhub API. These data sources will serve as:

1. **News/Sentiment Factors** (Phase C1/C2): Time-series factors derived from news sentiment and volume
2. **Macro/Regime Factors** (Phase D): Economic indicators for regime detection and risk modeling
3. **Event Studies** (Phase C3): Analyze price reactions to news events and macro announcements

**Key Design Principles:**
- **Finnhub is used exclusively for News/Macro/Event data**, not for price history
- **Price series always come from existing Alt-Daten Parquets** via `LocalParquetPriceDataSource`
- **Provider-agnostic design**: News/Macro data can later be extended to other sources (CSV, other APIs)
- **Compatibility with existing systems**: News/Macro data must work with `qa/event_study.py` and factor analysis workflows

---

## Scope B2 (Concrete)

### Alt-Data Sources from Finnhub (No Price APIs)

**1. News & Company News:**
- Company-specific news articles
- Market-wide news
- Source: Finnhub `/company-news` and `/news` endpoints
- **No price/candle endpoints** (prices come from local Parquets)

**2. News Sentiment:**
- Sentiment scores for news articles
- Aggregated sentiment per symbol/day
- Source: Finnhub `/news-sentiment` endpoint
- Optional: `/stock/social-sentiment` for social media sentiment

**3. Economic Calendar & Macro Indicators:**
- Economic calendar events (Fed meetings, CPI releases, etc.)
- Economic indicator values (GDP, unemployment, inflation, etc.)
- Source: Finnhub `/calendar/economic` and `/economic` endpoints
- Optional: `/economic/code` for specific indicator codes

**4. Optional Extensions (Future):**
- Stock social sentiment (`/stock/social-sentiment`)
- Sector metrics (`/sector/metrics`)

### Goals

**News/Sentiment Factors per Symbol:**
- Daily sentiment scores aggregated from news articles
- News volume spikes (abnormal news activity)
- Sentiment momentum (change in sentiment over time)
- News event flags (binary indicators for significant news days)

**Macro/Regime Factors as Time Series:**
- Economic indicator values (GDP, CPI, unemployment, etc.)
- Economic calendar events (Fed meetings, policy announcements)
- Regime indicators (expansion vs. contraction, risk-on vs. risk-off)
- **Later use in Phase D (Regime Models & Risk 2.0)**

**Price Series:**
- Continue using `LocalParquetPriceDataSource` only
- No price data fetching from Finnhub

---

## Data Contracts

### News Events DataFrame (`news_events_df`)

**Required Columns:**
- `timestamp`: `pd.Timestamp` (UTC-aware) - News publication date/time
- `symbol`: `str | None` - Stock ticker symbol (None for market-wide news)
- `source`: `str` - News source (e.g., `"Bloomberg"`, `"Reuters"`)
- `headline`: `str` - News headline/title
- `news_id`: `str` - Unique news identifier (e.g., `"news_20241209_AAPL_001"`)
- `event_type`: `str` - Always `"news"` for news events

**Optional Columns:**
- `category`: `str | None` - News category (e.g., `"earnings"`, `"product"`, `"management"`)
- `sentiment_score`: `float | None` - Sentiment score from Finnhub (if available, range typically -1 to +1)
- `sentiment_label`: `str | None` - Sentiment label (e.g., `"positive"`, `"negative"`, `"neutral"`)
- `summary`: `str | None` - News summary/description
- `url`: `str | None` - News article URL
- `raw_payload`: `str | dict | None` - Raw API response (JSON-serialized) for debugging/audit

**Example:**
```
timestamp              symbol  source     headline                                    news_id                    event_type  sentiment_score  category
2024-12-09 10:30:00+00:00  AAPL   Bloomberg  Apple Announces Record Q4 Earnings         news_20241209_AAPL_001     news        0.75            earnings
2024-12-09 14:00:00+00:00  None   Reuters    Fed Raises Interest Rates by 0.25%         news_20241209_MARKET_001  news        -0.50           macro
```

**Compatibility with `qa/event_study.py`:**
- Required columns (`timestamp`, `symbol`, `event_type`, `event_id`) match exactly (using `news_id` as `event_id`)
- UTC-aware timestamps ensure proper alignment with price data
- `symbol=None` for market-wide news (can be filtered or handled separately)

---

### News Sentiment Daily DataFrame (`news_sentiment_daily_df`)

**Required Columns:**
- `timestamp`: `pd.Timestamp` (UTC-aware) - Trading day (daily aggregation)
- `symbol`: `str | None` - Stock ticker symbol (None or `"__MARKET__"` for market-wide sentiment)
- `sentiment_score`: `float` - Aggregated sentiment score (e.g., mean or weighted mean)
- `sentiment_volume`: `int` - Number of news articles aggregated

**Optional Columns:**
- `sentiment_std`: `float | None` - Standard deviation of sentiment scores (volatility of sentiment)
- `positive_count`: `int | None` - Number of positive news articles
- `negative_count`: `int | None` - Number of negative news articles
- `neutral_count`: `int | None` - Number of neutral news articles
- `sentiment_momentum`: `float | None` - Change in sentiment vs. previous day/week

**Example:**
```
timestamp       symbol  sentiment_score  sentiment_volume  sentiment_std  positive_count  negative_count
2024-12-09      AAPL    0.65            15                 0.20           10              2
2024-12-09      MSFT    0.45            8                  0.15           5               1
2024-12-09      __MARKET__  -0.10       50                 0.30           15              20
```

**Usage in Factor Analysis:**
- Can be merged with price DataFrame using `timestamp` and `symbol`
- Can be used as a factor in Phase C1/C2 workflows
- Market-wide sentiment (`symbol="__MARKET__"`) can be joined to all symbols for regime detection

---

### Macro Series DataFrame (`macro_series_df`)

**Required Columns:**
- `timestamp`: `pd.Timestamp` (UTC-aware) - Release date/time of economic indicator
- `macro_code`: `str` - Economic indicator code (e.g., `"GDP"`, `"CPI"`, `"UNEMPLOYMENT"`, `"FED_RATE"`)
- `value`: `float` - Economic indicator value
- `country`: `str` - Country code (e.g., `"US"`, `"EU"`, `"CN"`)
- `release_time`: `pd.Timestamp | None` - Scheduled release time (if available)

**Optional Columns:**
- `indicator_name`: `str | None` - Human-readable indicator name (e.g., `"Gross Domestic Product"`)
- `unit`: `str | None` - Unit of measurement (e.g., `"percent"`, `"index"`, `"billion USD"`)
- `previous_value`: `float | None` - Previous period value (for comparison)
- `forecast_value`: `float | None` - Forecasted value (if available)
- `surprise`: `float | None` - Surprise (actual - forecast)
- `raw_payload`: `str | dict | None` - Raw API response (JSON-serialized) for debugging/audit

**Example:**
```
timestamp              macro_code    value    country  release_time          indicator_name              unit
2024-12-09 08:30:00+00:00  CPI         3.2      US      2024-12-09 08:30:00+00:00  Consumer Price Index      percent
2024-12-09 14:00:00+00:00  FED_RATE    5.25     US      2024-12-09 14:00:00+00:00  Federal Funds Rate        percent
2024-12-10 10:00:00+00:00  GDP         2.1      US      2024-12-10 10:00:00+00:00  Gross Domestic Product    percent
```

**Usage in Phase D (Regime Models):**
- Can be used to detect market regimes (expansion vs. contraction)
- Can be used for risk-on/risk-off indicators
- Can be joined to price data for regime-aware factor analysis

---

## Storage Layout

### Raw Data (Finnhub Response Normalized)

**Location:** `data/raw/altdata/finnhub/`

**Files:**
- `news_raw.parquet` - Raw news events (normalized from Finnhub API)
- `news_sentiment_raw.parquet` - Raw news sentiment data (if available as separate endpoint)
- `macro_raw.parquet` - Raw economic calendar and macro indicators

**Characteristics:**
- **Detail-rich**: Contains all fields from Finnhub API response
- **Normalized**: Consistent schema across all news/macro data
- **Append-only**: New data appended (with deduplication by `news_id` or `macro_code` + `timestamp`)
- **Audit trail**: `raw_payload` column preserves original API response

**Schema Evolution:**
- Schema changes should be versioned (e.g., `news_raw_v1.parquet`)
- Migration scripts can transform old schemas to new ones

---

### Cleaned Data Tables (For Analysis)

**Location:** `output/altdata/`

**Files:**
- `news_events.parquet` - Cleaned news events (ready for analysis)
- `news_sentiment_daily.parquet` - Aggregated daily sentiment per symbol
- `macro_series.parquet` - Cleaned macro indicators (ready for regime analysis)

**Characteristics:**
- **Analysis-ready**: Filtered, validated, and standardized
- **Symbol-filtered**: Only news for symbols with local price data (where applicable)
- **Deduplicated**: One row per unique `news_id` or `macro_code` + `timestamp`
- **Validated**: Timestamps within expected range, required columns present
- **Optimized**: Sorted by `symbol` (or `macro_code`), then `timestamp` for efficient joins
- **Aggregated**: Daily sentiment aggregated from individual news articles

**Cleaning Steps:**
1. **Deduplication**: Remove duplicate `news_id` entries (keep most recent)
2. **Symbol filtering**: Only keep news for symbols that exist in `ASSEMBLED_LOCAL_DATA_ROOT` (where applicable)
3. **Timestamp validation**: Ensure timestamps are UTC-aware and within date range
4. **Missing value handling**: Standardize `None` vs. `NaN` for optional columns
5. **Type validation**: Ensure numeric columns are proper types (float, int)
6. **Sentiment aggregation**: Aggregate individual news sentiment to daily sentiment per symbol

---

## API Usage & Provider Plan

### Finnhub API Integration

**Endpoints Used (B2):**

1. **Company News**: `/company-news`
   - Parameters: `symbol`, `from`, `to`, `token`
   - Returns: List of company-specific news articles with headlines, dates, sources, sentiment
   - **Note**: No price/candle data (prices come from local Parquets)

2. **General News**: `/news`
   - Parameters: `category`, `from`, `to`, `token`
   - Returns: Market-wide news articles
   - **Note**: Can be used for market sentiment (symbol=None or `"__MARKET__"`)

3. **News Sentiment**: `/news-sentiment`
   - Parameters: `symbol`, `from`, `to`, `token`
   - Returns: Aggregated sentiment scores for news articles
   - **Note**: May be available as separate endpoint or embedded in news response

4. **Economic Calendar**: `/calendar/economic`
   - Parameters: `from`, `to`, `token`
   - Returns: Economic calendar events (Fed meetings, CPI releases, etc.)
   - **Note**: Scheduled events with release times

5. **Economic Indicators**: `/economic`
   - Parameters: `code`, `from`, `to`, `token`
   - Returns: Historical economic indicator values (GDP, CPI, unemployment, etc.)
   - **Note**: Can fetch specific indicators by code

6. **Economic Indicator by Code**: `/economic/code` (Optional)
   - Parameters: `code`, `token`
   - Returns: Specific economic indicator metadata and values

**Optional Endpoints (Future):**
- `/stock/social-sentiment`: Social media sentiment for stocks
- `/sector/metrics`: Sector-level metrics

**Rate Limits:**
- Free tier: 60 calls/minute
- Paid tiers: Higher limits (see Finnhub pricing)
- **Strategy**: Batch requests with delays, cache responses locally

**API Key Management:**
- Stored in `Settings.finnhub_api_key` (from `ASSEMBLED_FINNHUB_API_KEY` environment variable)
- Never logged or exposed in error messages
- Validated before making API calls

**Error Handling:**
- **Rate limit exceeded (429)**: Exponential backoff, retry after delay
- **Invalid API key (401)**: Clear error message, skip symbol/indicator
- **Symbol not found (404)**: Log warning, skip symbol (tolerated for missing European tickers)
- **Network errors**: Retry with exponential backoff (max 3 retries)

---

### Price Data Strategy

**Always Use Local Parquet Files:**
- Price data is **never** fetched from Finnhub (only news/macro data)
- All price series loaded via `LocalParquetPriceDataSource`
- Path structure: `<ASSEMBLED_LOCAL_DATA_ROOT>/<freq>/<symbol>.parquet`

**Symbol Filtering:**
- Before fetching news, check which symbols have price data available
- Only fetch news for symbols that exist in `ASSEMBLED_LOCAL_DATA_ROOT`
- Missing European tickers are known limitation (marked in universe files, tolerated)

**Date Range Alignment:**
- News/macro date ranges should align with available price data
- If price data starts at 2010-01-01, fetch news/macro from 2010-01-01 onwards
- If price data ends at 2025-12-03, fetch news/macro up to 2025-12-03

**Join Strategy:**
- News/sentiment joined with price/factor data via:
  - `timestamp` (UTC, daily alignment)
  - `symbol` (where applicable)
- Macro data joined with price/factor data via:
  - `timestamp` (UTC, daily alignment)
  - No symbol join (macro is market-wide, can be joined to all symbols)

---

## Integration with Existing System

### Integration with Phase C1/C2 (Factor Analysis)

**News Sentiment as Factor:**
- `news_sentiment_daily_df` can be merged with price DataFrame using `timestamp` and `symbol`
- Can be used as a factor in IC and portfolio analysis
- Example factors:
  - `news_sentiment_score`: Daily sentiment score (from `news_sentiment_daily_df`)
  - `news_sentiment_momentum`: Change in sentiment vs. previous day/week
  - `news_volume_spike`: Abnormal news volume (z-score of `sentiment_volume`)

**Workflow Example:**
```python
from src.assembled_core.features.altdata_news_macro_factors import (
    build_news_sentiment_factors,
    build_news_volume_factors,
)
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_ic,
    summarize_ic_series,
)

# Load prices
prices = price_source.get_history(...)

# Load news sentiment (from cleaned Parquet)
news_sentiment = pd.read_parquet("output/altdata/news_sentiment_daily.parquet")

# Create news factors
news_factors = build_news_sentiment_factors(
    prices,
    news_sentiment,
    lookback_days=20,  # Rolling window for momentum
)

# Combine with other factors
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
ta_factors = build_core_ta_factors(prices)

all_factors = ta_factors.merge(
    news_factors,
    on=["timestamp", "symbol"],
    how="left"
)

# Use in factor analysis
factors_with_returns = add_forward_returns(all_factors, horizon_days=20)
ic_df = compute_ic(factors_with_returns, forward_returns_col="fwd_return_20d")
summary = summarize_ic_series(ic_df)
```

**Market-Wide Sentiment:**
- `news_sentiment_daily_df` with `symbol="__MARKET__"` can be joined to all symbols
- Useful for regime detection and risk-on/risk-off indicators
- Can be used as a market-wide factor in factor analysis

---

### Integration with Phase C3 (Event Study Engine)

**News Events as Events:**
- `news_events_df` can be passed directly to `build_event_window_prices()`
- Required columns (`timestamp`, `symbol`, `event_type`, `event_id`) match exactly (using `news_id` as `event_id`)
- UTC-aware timestamps ensure proper alignment with price data

**Workflow Example:**
```python
from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)

# Load prices
prices = price_source.get_history(...)

# Load news events (from cleaned Parquet)
news_events = pd.read_parquet("output/altdata/news_events.parquet")
news_events_filtered = news_events[
    (news_events["symbol"].isin(["AAPL", "MSFT"])) &
    (news_events["timestamp"] >= "2020-01-01") &
    (news_events["timestamp"] <= "2025-12-03")
]

# Build event windows
windows = build_event_window_prices(
    prices,
    news_events_filtered,
    window_before=5,  # 5 days before news
    window_after=10,  # 10 days after news
)

# Compute returns
returns = compute_event_returns(windows, return_type="log")

# Aggregate results (e.g., by sentiment category)
aggregated = aggregate_event_study(returns, use_abnormal=False)
```

**Sentiment-Based Event Studies:**
- Can filter news events by `sentiment_score` or `sentiment_label`
- Separate analysis for positive vs. negative news
- Compare price reactions to different sentiment categories

---

### Integration with Phase D (Regime Models & Risk 2.0)

**Macro Indicators for Regime Detection:**
- `macro_series_df` can be used to detect market regimes
- Example regimes:
  - **Expansion vs. Contraction**: Based on GDP growth, unemployment
  - **Risk-On vs. Risk-Off**: Based on VIX, Fed rate, economic uncertainty
  - **Inflation Regime**: Based on CPI, PPI, inflation expectations

**Workflow Example (Future - Phase D):**
```python
from src.assembled_core.risk.regime_detection import (
    detect_market_regime,
    build_regime_factors,
)

# Load macro indicators
macro_series = pd.read_parquet("output/altdata/macro_series.parquet")

# Detect regimes
regimes = detect_market_regime(
    macro_series,
    indicators=["GDP", "CPI", "UNEMPLOYMENT", "FED_RATE"],
    method="hmm",  # Hidden Markov Model
)

# Build regime factors
regime_factors = build_regime_factors(
    prices,
    regimes,
    lookback_days=60,
)

# Use in risk models
from src.assembled_core.risk.risk_engine import compute_regime_aware_risk
risk_metrics = compute_regime_aware_risk(
    portfolio_returns,
    regime_factors,
    regime="expansion",  # Current regime
)
```

**Economic Calendar Events:**
- Can be used as events in event studies (e.g., Fed rate announcements)
- Can be used to trigger regime changes
- Can be used for risk-on/risk-off indicators

---

## Implementation Plan

### Sprint B2.1: Design (Current Step)

**Tasks:**
- [x] Create design document (`docs/ALT_DATA_FACTORS_B2_DESIGN.md`)
- [x] Define data contracts for `news_events_df`, `news_sentiment_daily_df`, `macro_series_df`
- [x] Plan storage layout (raw and cleaned Parquet files)
- [x] Document API usage and provider plan
- [x] Document integration with Phase C1/C2/C3/D

---

### Sprint B2.2: Finnhub News/Macro Client + Download Script

**Tasks:**
1. Extend `src/assembled_core/data/altdata/finnhub_events.py` module
   - `fetch_company_news()`: Fetch company-specific news from Finnhub API
   - `fetch_market_news()`: Fetch market-wide news from Finnhub API
   - `fetch_news_sentiment()`: Fetch news sentiment scores (if available as separate endpoint)
   - `fetch_economic_calendar()`: Fetch economic calendar events
   - `fetch_economic_indicators()`: Fetch economic indicator values
   - `normalize_news_response()`: Normalize Finnhub news response to DataFrame
   - `normalize_macro_response()`: Normalize Finnhub macro response to DataFrame
   
2. Create `scripts/download_altdata_finnhub_news_macro.py` script
   - CLI interface for downloading news and macro data
   - Symbol filtering (only symbols with local price data, where applicable)
   - Date range configuration
   - Rate limit handling
   - Save to `data/raw/altdata/finnhub/`

3. Create `src/assembled_core/data/altdata/clean_news_macro.py` module
   - `clean_news_events()`: Deduplicate, validate, filter news events
   - `aggregate_news_sentiment()`: Aggregate individual news sentiment to daily sentiment per symbol
   - `clean_macro_series()`: Deduplicate, validate, filter macro indicators
   - `filter_symbols_by_price_data()`: Filter news to only symbols with price data
   - Save cleaned data to `output/altdata/`

4. Tests: `tests/test_data_finnhub_news_macro.py`
   - Mock Finnhub API responses
   - Test normalization functions
   - Test cleaning and aggregation functions
   - Test symbol filtering

---

### Sprint B2.3: Factor Module for News/Macro Factors

**Tasks:**
1. Create `src/assembled_core/features/altdata_news_macro_factors.py` module
   - `build_news_sentiment_factors()`: Daily sentiment score, sentiment momentum
   - `build_news_volume_factors()`: News volume spikes, abnormal news activity
   - `build_news_event_factors()`: Binary flags for significant news days
   - `build_macro_regime_factors()`: Macro-based regime indicators (for Phase D)
   - Output: Panel DataFrame compatible with Phase C1/C2

2. Integration with `run_factor_analysis.py`
   - Add `--factor-set news_sentiment` option
   - Add `--factor-set all` to include news/macro factors
   - Load news sentiment and compute factors
   - Include in IC and portfolio analysis

3. Tests: `tests/test_features_altdata_news_macro_factors.py`
   - Test factor computation on synthetic news/macro data
   - Test alignment with price data
   - Test edge cases (no news, missing data)

---

### Sprint B2.4: Integration in analyze_factors + Optional Event Study Variants

**Tasks:**
1. Extend `scripts/run_factor_analysis.py`
   - Add `--factor-set news_sentiment` and `--factor-set all` options
   - Load news sentiment from `output/altdata/news_sentiment_daily.parquet`
   - Compute news factors and merge with other factors
   - Handle missing news data gracefully (warnings, not crashes)

2. Extend `research/events/event_study_template_core.py`
   - Add option to load real news events from `output/altdata/news_events.parquet`
   - Support for sentiment-based event filtering
   - Separate analysis for positive vs. negative news

3. Create `research/events/event_study_news.py` workflow
   - Load news events from `output/altdata/news_events.parquet`
   - Filter by sentiment category (positive, negative, neutral)
   - Run event study workflow
   - Visualize results (AAR, CAAR by sentiment category)

4. Documentation updates
   - `docs/WORKFLOWS_FACTOR_ANALYSIS.md`: Add news sentiment factors section
   - `docs/WORKFLOWS_EVENT_STUDIES.md`: Add news event studies section

---

### Sprint B2.5: Tests & Documentation

**Tasks:**
1. Comprehensive test suite
   - `tests/test_data_finnhub_news_macro.py`: API client tests with mocks
   - `tests/test_features_altdata_news_macro_factors.py`: Factor computation tests
   - `tests/test_cli_analyze_factors_news.py`: CLI integration tests
   - All tests marked with `@pytest.mark.advanced`

2. Documentation updates
   - `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`: Mark B2 as completed
   - `docs/WORKFLOWS_FACTOR_ANALYSIS.md`: Add news sentiment examples
   - `docs/WORKFLOWS_EVENT_STUDIES.md`: Add news event study examples
   - `docs/RESEARCH_ROADMAP.md`: Update with B2 completion status

3. Integration testing
   - End-to-end workflow: Download → Clean → Factor Analysis → Event Study
   - Verify compatibility with Phase C1/C2/C3
   - Verify no regressions in existing B1 functionality

---

## TODO List

### Immediate (Sprint B2.1 - Design)
- [x] Create design document
- [x] Define data contracts
- [x] Plan storage layout
- [x] Document API usage
- [x] Document integration points

### Short-term (Sprint B2.2)
- [ ] Extend `finnhub_events.py` with news/macro endpoints
- [ ] Create `download_altdata_finnhub_news_macro.py` script
- [ ] Create `clean_news_macro.py` module
- [ ] Implement news sentiment aggregation
- [ ] Create tests for data ingestion

### Medium-term (Sprint B2.3)
- [ ] Create `altdata_news_macro_factors.py` module
- [ ] Implement news sentiment factors
- [ ] Implement news volume factors
- [ ] Integrate with factor analysis workflow
- [ ] Create tests for news/macro factors

### Long-term (Sprint B2.4-B2.5)
- [ ] CLI integration for news/macro download
- [ ] CLI integration for news/macro factors
- [ ] Event study workflows for news events
- [ ] Comprehensive test suite
- [ ] Documentation updates

---

## References

- **B1 Design**: `docs/ALT_DATA_FACTORS_B1_DESIGN.md`
- **Event Study Engine**: `src/assembled_core/qa/event_study.py`
- **Factor Analysis**: `src/assembled_core/qa/factor_analysis.py`
- **Price Data Source**: `src/assembled_core/data/data_source.py`
- **Settings**: `src/assembled_core/config/settings.py`
- **Finnhub API Docs**: https://finnhub.io/docs/api
- **Event Study Workflow**: `research/events/event_study_template_core.py`
- **Factor Analysis Workflow**: `docs/WORKFLOWS_FACTOR_ANALYSIS.md`
- **Event Study Workflow**: `docs/WORKFLOWS_EVENT_STUDIES.md`
- **Advanced Analytics Roadmap**: `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`

