# Alt-Data Factors – Sprint B1 Design: Earnings & Insider Events

**Last Updated:** 2025-12-09  
**Status:** Design Phase  
**Sprint:** B1 – Earnings & Insider Alt-Daten Integration

---

## Overview

Sprint B1 focuses on integrating **Earnings** and **Insider Activity** events from Finnhub API into the trading system. These events will serve as:

1. **Input for Event Studies** (Phase C3): Analyze price reactions to earnings announcements and insider transactions
2. **Alt-Data Factors** (Phase C1/C2): Use event signals as factors in factor analysis workflows
3. **Research Data**: Enable systematic analysis of event-driven trading strategies

**Key Design Principles:**
- **Finnhub is used exclusively for Events**, not for price history
- **Price series always come from existing Alt-Daten Parquets** via `LocalParquetPriceDataSource`
- **Provider-agnostic design**: Events can later be extended to other sources (CSV, other APIs)
- **Compatibility with existing systems**: Events must work with `qa/event_study.py` and factor analysis workflows

---

## Scope B1 (Concrete)

### Event Types

**1. Earnings Events:**
- Earnings calendar data (announcement dates)
- Earnings report data (EPS actual vs. estimate, revenue, surprise)
- Source: Finnhub `/calendar/earnings` and `/stock/earnings` endpoints

**2. Insider Activity Events:**
- Insider transactions (buys, sells, option exercises)
- Insider sentiment indicators (aggregate insider activity)
- Source: Finnhub `/stock/insider-transactions` endpoint

### Data Source Strategy

**Price Data:**
- Always loaded via `LocalParquetPriceDataSource` from `ASSEMBLED_LOCAL_DATA_ROOT`
- Format: `<local_data_root>/<freq>/<symbol>.parquet` (e.g., `1d/AAPL.parquet`)
- No price data fetching from Finnhub (only events)

**Event Data:**
- Fetched from Finnhub API (using `finnhub_api_key` from settings)
- Stored in normalized Parquet format
- Filtered to only include symbols that have price data available locally

**Date Range & Symbol Filtering:**
- Configurable date ranges (e.g., 2010-2025)
- Symbol filtering: Only process symbols that have Parquet price files in `ASSEMBLED_LOCAL_DATA_ROOT`
- Missing European tickers are tolerated (known limitation, will be handled separately)

---

## Data Contracts

### Events Earnings DataFrame (`events_earnings_df`)

**Required Columns:**
- `timestamp`: `pd.Timestamp` (UTC-aware) - Earnings announcement date/time
- `symbol`: `str` - Stock ticker symbol
- `event_type`: `str` - Always `"earnings"` for earnings events
- `event_id`: `str` - Unique event identifier (e.g., `"earnings_2024Q1_AAPL"`)

**Additional Columns:**
- `fiscal_period`: `str` - Fiscal period (e.g., `"2024Q1"`, `"2024FY"`)
- `eps_actual`: `float | None` - Actual EPS (if reported)
- `eps_estimate`: `float | None` - Estimated EPS (consensus)
- `eps_surprise`: `float | None` - EPS surprise (actual - estimate)
- `eps_surprise_pct`: `float | None` - EPS surprise percentage
- `revenue_actual`: `float | None` - Actual revenue (if reported)
- `revenue_estimate`: `float | None` - Estimated revenue (consensus)
- `revenue_surprise`: `float | None` - Revenue surprise
- `revenue_surprise_pct`: `float | None` - Revenue surprise percentage
- `source`: `str` - Data source (e.g., `"finnhub"`)
- `raw_payload`: `str | dict | None` - Raw API response (JSON-serialized) for debugging/audit

**Example:**
```
timestamp              symbol  event_type  event_id              fiscal_period  eps_actual  eps_estimate  eps_surprise  source   raw_payload
2024-01-15 16:00:00+00:00  AAPL   earnings    earnings_2024Q1_AAPL  2024Q1        2.10        2.05         0.05         finnhub  {...}
2024-04-15 16:00:00+00:00  AAPL   earnings    earnings_2024Q2_AAPL  2024Q2        2.15        2.10         0.05         finnhub  {...}
```

**Compatibility with `qa/event_study.py`:**
- Required columns (`timestamp`, `symbol`, `event_type`, `event_id`) match exactly
- UTC-aware timestamps ensure proper alignment with price data
- `event_id` allows deduplication and event-specific analysis

---

### Events Insider DataFrame (`events_insider_df`)

**Required Columns:**
- `timestamp`: `pd.Timestamp` (UTC-aware) - Transaction date
- `symbol`: `str` - Stock ticker symbol
- `event_type`: `str` - Event category: `"insider_buy"`, `"insider_sell"`, `"insider_exercise"`, etc.
- `event_id`: `str` - Unique event identifier (e.g., `"insider_20240115_AAPL_CEO_12345"`)

**Additional Columns:**
- `insider_name`: `str` - Name of insider (e.g., `"Tim Cook"`)
- `position`: `str` - Insider position/role (e.g., `"CEO"`, `"CFO"`, `"Director"`)
- `transaction_type`: `str` - Transaction type code (Finnhub format: `"P"`, `"S"`, `"A"`, etc.)
- `shares`: `int | float | None` - Number of shares transacted
- `usd_notional`: `float | None` - Transaction value in USD
- `price`: `float | None` - Transaction price per share
- `direction`: `str` - Normalized direction: `"buy"`, `"sell"`, `"exercise"`, `"other"`
- `source`: `str` - Data source (e.g., `"finnhub"`)
- `raw_payload`: `str | dict | None` - Raw API response (JSON-serialized) for debugging/audit

**Example:**
```
timestamp              symbol  event_type    event_id                        insider_name  position  transaction_type  shares   usd_notional  direction  source   raw_payload
2024-01-20 09:30:00+00:00  AAPL   insider_buy   insider_20240120_AAPL_CEO_001  Tim Cook     CEO       P                 10000    1500000.0    buy       finnhub  {...}
2024-02-15 09:30:00+00:00  MSFT   insider_sell  insider_20240215_MSFT_CFO_001  Amy Hood    CFO       S                 5000     2000000.0    sell      finnhub  {...}
```

**Compatibility with `qa/event_study.py`:**
- Required columns match exactly
- `event_type` allows filtering by transaction type (e.g., only `insider_buy` events)
- `event_id` ensures unique identification for event study aggregation

---

## Storage Layout

### Raw Events (Finnhub Response Normalized)

**Location:** `data/raw/altdata/finnhub/`

**Files:**
- `earnings_events_raw.parquet` - Raw earnings events (normalized from Finnhub API)
- `insider_events_raw.parquet` - Raw insider events (normalized from Finnhub API)

**Characteristics:**
- **Detail-rich**: Contains all fields from Finnhub API response
- **Normalized**: Consistent schema across all events
- **Append-only**: New events appended (with deduplication by `event_id`)
- **Audit trail**: `raw_payload` column preserves original API response

**Schema Evolution:**
- Schema changes should be versioned (e.g., `earnings_events_raw_v1.parquet`)
- Migration scripts can transform old schemas to new ones

---

### Cleaned Event Tables (For Analysis)

**Location:** `output/altdata/`

**Files:**
- `events_earnings.parquet` - Cleaned earnings events (ready for analysis)
- `events_insider.parquet` - Cleaned insider events (ready for analysis)

**Characteristics:**
- **Analysis-ready**: Filtered, validated, and standardized
- **Symbol-filtered**: Only events for symbols with local price data
- **Deduplicated**: One row per unique `event_id`
- **Validated**: Timestamps within expected range, required columns present
- **Optimized**: Sorted by `symbol`, then `timestamp` for efficient joins

**Cleaning Steps:**
1. **Deduplication**: Remove duplicate `event_id` entries (keep most recent)
2. **Symbol filtering**: Only keep events for symbols that exist in `ASSEMBLED_LOCAL_DATA_ROOT`
3. **Timestamp validation**: Ensure timestamps are UTC-aware and within date range
4. **Missing value handling**: Standardize `None` vs. `NaN` for optional columns
5. **Type validation**: Ensure numeric columns are proper types (float, int)

---

## API Usage & Provider Plan

### Finnhub API Integration

**Endpoints Used:**
1. **Earnings Calendar**: `/calendar/earnings`
   - Parameters: `symbol`, `from`, `to`, `token`
   - Returns: List of earnings announcements with dates, EPS estimates/actuals, revenue
   
2. **Stock Earnings**: `/stock/earnings`
   - Parameters: `symbol`, `token`
   - Returns: Historical earnings data for a symbol
   
3. **Insider Transactions**: `/stock/insider-transactions`
   - Parameters: `symbol`, `from`, `to`, `token`
   - Returns: List of insider transactions with dates, names, positions, transaction types, shares

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
- **Invalid API key (401)**: Clear error message, skip symbol
- **Symbol not found (404)**: Log warning, skip symbol (tolerated for missing European tickers)
- **Network errors**: Retry with exponential backoff (max 3 retries)

---

### Price Data Strategy

**Always Use Local Parquet Files:**
- Price data is **never** fetched from Finnhub (only events)
- All price series loaded via `LocalParquetPriceDataSource`
- Path structure: `<ASSEMBLED_LOCAL_DATA_ROOT>/<freq>/<symbol>.parquet`

**Symbol Filtering:**
- Before fetching events, check which symbols have price data available
- Only fetch events for symbols that exist in `ASSEMBLED_LOCAL_DATA_ROOT`
- Missing European tickers are known limitation (marked in universe files, tolerated)

**Date Range Alignment:**
- Event date ranges should align with available price data
- If price data starts at 2010-01-01, fetch events from 2010-01-01 onwards
- If price data ends at 2025-12-03, fetch events up to 2025-12-03

---

## Integration with Existing System

### Integration with Phase C3 (Event Study Engine)

**Direct Compatibility:**
- `events_earnings_df` and `events_insider_df` can be passed directly to `build_event_window_prices()`
- Required columns (`timestamp`, `symbol`, `event_type`, `event_id`) match exactly
- UTC-aware timestamps ensure proper alignment with price data

**Workflow Example:**
```python
from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.config.settings import get_settings

# Load prices
settings = get_settings()
price_source = get_price_data_source(settings, data_source="local")
prices = price_source.get_history(
    symbols=["AAPL", "MSFT"],
    start_date="2020-01-01",
    end_date="2025-12-03",
    freq="1d",
)

# Load events (from cleaned Parquet files)
events_earnings = pd.read_parquet("output/altdata/events_earnings.parquet")
events_earnings_filtered = events_earnings[
    (events_earnings["symbol"].isin(["AAPL", "MSFT"])) &
    (events_earnings["timestamp"] >= "2020-01-01") &
    (events_earnings["timestamp"] <= "2025-12-03")
]

# Build event windows
windows = build_event_window_prices(
    prices,
    events_earnings_filtered,
    window_before=20,
    window_after=40,
)

# Compute returns
returns = compute_event_returns(windows, return_type="log")

# Aggregate results
aggregated = aggregate_event_study(returns, use_abnormal=False)
```

**Event Type Filtering:**
- Can filter by `event_type` before event study (e.g., only `insider_buy` events)
- Can group by `event_type` in aggregation (e.g., separate analysis for `insider_buy` vs. `insider_sell`)

---

### Integration with Phase C1/C2 (Factor Analysis)

**Alt-Data Factors from Events:**
- Events can be transformed into time-series factors for factor analysis
- Example factors:
  - **Earnings Surprise Factor**: `eps_surprise_pct` (positive = beat, negative = miss)
  - **Insider Buy/Sell Ratio**: Ratio of insider buys to sells over rolling window
  - **Earnings Frequency**: Number of earnings announcements per quarter
  - **Insider Activity Intensity**: Total USD notional of insider transactions per month

**Factor Creation Workflow:**
```python
from src.assembled_core.features.altdata_factors import (
    build_earnings_surprise_factor,
    build_insider_activity_factor,
)

# Load events
events_earnings = pd.read_parquet("output/altdata/events_earnings.parquet")
events_insider = pd.read_parquet("output/altdata/events_insider.parquet")

# Load prices (for alignment)
prices = price_source.get_history(...)

# Create alt-data factors
earnings_factor = build_earnings_surprise_factor(
    prices,
    events_earnings,
    window=90,  # Rolling 90-day window
)

insider_factor = build_insider_activity_factor(
    prices,
    events_insider,
    window=30,  # Rolling 30-day window
    direction="buy",  # Only count buys
)

# Combine with other factors
from src.assembled_core.features.ta_factors_core import build_core_ta_factors
ta_factors = build_core_ta_factors(prices)

# Merge factors
all_factors = ta_factors.merge(
    earnings_factor,
    on=["timestamp", "symbol"],
    how="left"
).merge(
    insider_factor,
    on=["timestamp", "symbol"],
    how="left"
)

# Use in factor analysis
from src.assembled_core.qa.factor_analysis import (
    add_forward_returns,
    compute_ic,
    summarize_ic_series,
)

factors_with_returns = add_forward_returns(all_factors, horizon_days=20)
ic_df = compute_ic(factors_with_returns, forward_returns_col="fwd_return_20d")
summary = summarize_ic_series(ic_df)
```

**Factor Module (Future):**
- New module: `src/assembled_core/features/altdata_factors.py`
- Functions: `build_earnings_surprise_factor()`, `build_insider_activity_factor()`, etc.
- Output format: Panel DataFrame (same as Phase A factors) with `timestamp`, `symbol`, `factor_*` columns

---

## Implementation Plan

### Sprint B1.1: Data Ingestion (Current Sprint)

**Tasks:**
1. Create `src/assembled_core/data/finnhub_events.py` module
   - `fetch_earnings_events()`: Fetch earnings from Finnhub API
   - `fetch_insider_events()`: Fetch insider transactions from Finnhub API
   - `normalize_earnings_response()`: Normalize Finnhub earnings response to DataFrame
   - `normalize_insider_response()`: Normalize Finnhub insider response to DataFrame
   
2. Create `scripts/download_finnhub_events.py` script
   - CLI interface for downloading events
   - Symbol filtering (only symbols with local price data)
   - Date range configuration
   - Rate limit handling
   - Save to `data/raw/altdata/finnhub/`

3. Create `src/assembled_core/data/clean_events.py` module
   - `clean_earnings_events()`: Deduplicate, validate, filter earnings events
   - `clean_insider_events()`: Deduplicate, validate, filter insider events
   - `filter_symbols_by_price_data()`: Filter events to only symbols with price data
   - Save cleaned events to `output/altdata/`

4. Tests: `tests/test_data_finnhub_events.py`
   - Mock Finnhub API responses
   - Test normalization functions
   - Test cleaning functions
   - Test symbol filtering

---

### Sprint B1.2: Event Study Integration (Planned)

**Tasks:**
1. Create `research/events/event_study_earnings.py` workflow
   - Load earnings events from `output/altdata/events_earnings.parquet`
   - Load prices via `LocalParquetPriceDataSource`
   - Run event study workflow (build windows, compute returns, aggregate)
   - Visualize results (AAR, CAAR by earnings surprise)
   
2. Create `research/events/event_study_insider.py` workflow
   - Load insider events from `output/altdata/events_insider.parquet`
   - Separate analysis for `insider_buy` vs. `insider_sell`
   - Run event study workflow
   - Visualize results (AAR, CAAR by transaction type)

3. Extend `research/events/event_study_template_core.py`
   - Add option to load real events instead of synthetic
   - Support for earnings and insider events
   - Benchmark selection (market, sector)

---

### Sprint B1.3: Alt-Data Factors (Planned)

**Tasks:**
1. Create `src/assembled_core/features/altdata_factors.py` module
   - `build_earnings_surprise_factor()`: Rolling earnings surprise factor
   - `build_insider_activity_factor()`: Insider buy/sell ratio, intensity
   - `build_earnings_frequency_factor()`: Earnings announcement frequency
   - Output: Panel DataFrame compatible with Phase C1/C2

2. Integration with `run_factor_analysis.py`
   - Add `--factor-set altdata` option
   - Load events and compute alt-data factors
   - Include in IC and portfolio analysis

3. Tests: `tests/test_features_altdata_factors.py`
   - Test factor computation on synthetic events
   - Test alignment with price data
   - Test edge cases (no events, missing data)

---

### Sprint B1.4: CLI Integration (Planned)

**Tasks:**
1. Extend `scripts/cli.py` with `download_events` subcommand
   - `--event-type` (earnings, insider, or both)
   - `--symbols-file` or `--symbols`
   - `--start-date`, `--end-date`
   - `--output-dir` (default: `data/raw/altdata/finnhub/`)

2. Extend `scripts/cli.py` with `clean_events` subcommand
   - `--event-type` (earnings, insider, or both)
   - `--input-dir` (default: `data/raw/altdata/finnhub/`)
   - `--output-dir` (default: `output/altdata/`)
   - `--validate-symbols` (check against local price data)

3. Documentation updates
   - `docs/WORKFLOWS_EVENT_STUDIES.md`: Add real events examples
   - `docs/WORKFLOWS_FACTOR_ANALYSIS.md`: Add alt-data factors section

---

### Sprint B1.5: Advanced Features (Planned)

**Tasks:**
1. **Event Clustering Detection**
   - Detect events that occur close together (e.g., earnings + insider transaction)
   - Handle clustered events in event studies (exclude or group)

2. **Event Quality Scoring**
   - Score events by data quality (completeness, timeliness)
   - Filter low-quality events in analysis

3. **Multi-Event Analysis**
   - Analyze interactions between earnings and insider events
   - Combined event study (e.g., earnings + insider buy)

4. **Historical Backfill**
   - Backfill historical events (if API supports)
   - Incremental updates (only fetch new events)

---

## TODO List

### Immediate (Sprint B1.1)
- [ ] Create `src/assembled_core/data/finnhub_events.py` module
- [ ] Implement `fetch_earnings_events()` function
- [ ] Implement `fetch_insider_events()` function
- [ ] Implement normalization functions
- [ ] Create `scripts/download_finnhub_events.py` script
- [ ] Create `src/assembled_core/data/clean_events.py` module
- [ ] Implement cleaning functions
- [ ] Create tests for data ingestion
- [ ] Document API usage and rate limits

### Short-term (Sprint B1.2)
- [ ] Create event study workflows for earnings
- [ ] Create event study workflows for insider
- [ ] Extend template workflow with real events
- [ ] Add visualization for event study results

### Medium-term (Sprint B1.3)
- [ ] Create `altdata_factors.py` module
- [ ] Implement earnings surprise factor
- [ ] Implement insider activity factor
- [ ] Integrate with factor analysis workflow
- [ ] Create tests for alt-data factors

### Long-term (Sprint B1.4-B1.5)
- [ ] CLI integration for event download
- [ ] CLI integration for event cleaning
- [ ] Event clustering detection
- [ ] Event quality scoring
- [ ] Multi-event analysis
- [ ] Historical backfill support

---

## References

- **Event Study Engine**: `src/assembled_core/qa/event_study.py`
- **Factor Analysis**: `src/assembled_core/qa/factor_analysis.py`
- **Price Data Source**: `src/assembled_core/data/data_source.py`
- **Settings**: `src/assembled_core/config/settings.py`
- **Finnhub API Docs**: https://finnhub.io/docs/api
- **Event Study Workflow**: `research/events/event_study_template_core.py`
- **Factor Analysis Workflow**: `docs/WORKFLOWS_FACTOR_ANALYSIS.md`
- **Event Study Workflow**: `docs/WORKFLOWS_EVENT_STUDIES.md`

