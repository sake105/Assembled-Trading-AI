# Workflows – Event Studies

**Last Updated:** 2025-12-09  
**Status:** Active Workflows for Event Study Analysis

---

## Overview

This document describes workflows for event study analysis using the Phase C3 Event Study Engine. Event studies enable systematic analysis of price reactions to specific events such as:

- **Earnings Announcements**: How do prices react to earnings releases?
- **Insider Trading**: Are there significant price patterns after insider buys/sells?
- **News Events**: Which news types lead to abnormal returns?
- **Regulatory Events**: How do regulatory changes affect prices?

**Prerequisites:**
- Local alt-data available (Parquet files in `ASSEMBLED_LOCAL_DATA_ROOT`)
- Event data (synthetic for testing, or real events from APIs/CSV)
- Python environment with dependencies installed

---

## Quick Start

### Basic Event Study with Synthetic Events

```powershell
# Set environment variable for local data root
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

# Run the research workflow
python research/events/event_study_template_core.py
```

**Output:**
- `output/event_studies/event_study_synthetic_earnings.png` - Visualization (AAR, CAAR)
- `output/event_studies/event_study_synthetic_earnings.csv` - Aggregated results
- Optional: `experiments/.../` (if experiment tracking enabled)

---

## Standard Workflows

### 1. Research Workflow with Synthetic Events

The template workflow (`research/events/event_study_template_core.py`) provides a complete example:

**Features:**
- Loads prices from local Parquet files
- Generates synthetic events (pseudo-earnings every 60 days)
- Builds event windows (-20 to +40 days)
- Computes normal and abnormal returns
- Aggregates results across events
- Visualizes Average/Cumulative Abnormal Returns
- Optional experiment tracking

**Configuration:**
```python
# In event_study_template_core.py
symbols_file = ROOT / "config" / "macro_world_etfs_tickers.txt"
freq = "1d"
start_date = "2010-01-01"
end_date = "2025-12-03"
window_before = 20
window_after = 40
event_interval_days = 60  # Generate pseudo-earnings every 60 days
```

**Customization:**
- Change `event_interval_days` to adjust event frequency
- Modify `window_before` and `window_after` to change event window
- Add benchmark column for abnormal return calculation
- Enable experiment tracking for structured logging

**Using Real Earnings Events:**
The template now supports loading real earnings events from `output/altdata/events_earnings.parquet`:

```python
# In event_study_template_core.py, set:
USE_REAL_EVENTS = True  # Set to True to use real events instead of synthetic
```

**Prerequisites for Real Events:**
1. Download earnings events using `download_altdata_finnhub_events.py`:
   ```powershell
   python scripts/download_altdata_finnhub_events.py `
     --symbols-file config/macro_world_etfs_tickers.txt `
     --start-date 2010-01-01 `
     --end-date 2025-12-03 `
     --event-types earnings `
     --output-dir output/altdata
   ```

2. Ensure `output/altdata/events_earnings.parquet` exists

3. Set `USE_REAL_EVENTS = True` in the template script

**Behavior:**
- If `USE_REAL_EVENTS = True` and events file exists: Loads real events
- Events are automatically filtered to match the symbols and date range of the price data

**Combining Events with News Sentiment (Conceptual):**
For advanced event studies, you can combine earnings events with news sentiment factors:

1. **Load Earnings Events**: Use `load_real_earnings_events()` or load from `output/altdata/events_earnings.parquet`
2. **Load News Sentiment**: Load from `output/altdata/news_sentiment_daily.parquet`
3. **Filter by Sentiment**: Only analyze earnings events that occurred during periods of high/low news sentiment
4. **Stratify Results**: Compare AAR/CAAR for earnings events with positive vs. negative sentiment

**Example Conceptual Workflow (Future Enhancement):**
```python
# Load earnings events
events_earnings = pd.read_parquet("output/altdata/events_earnings.parquet")

# Load news sentiment
news_sentiment = pd.read_parquet("output/altdata/news_sentiment_daily.parquet")

# Join sentiment to events (merge_asof on timestamp)
events_with_sentiment = pd.merge_asof(
    events_earnings.sort_values("timestamp"),
    news_sentiment.sort_values("timestamp"),
    on="timestamp",
    direction="backward"
)

# Stratify by sentiment
high_sentiment_events = events_with_sentiment[
    events_with_sentiment["sentiment_score"] > 0.5
]
low_sentiment_events = events_with_sentiment[
    events_with_sentiment["sentiment_score"] < -0.5
]

# Run separate event studies for each group
# (This would require extending event_study_template_core.py)
```

**Note:** This is a conceptual workflow. A full implementation would require extending `event_study_template_core.py` to support sentiment-based stratification.
- If `USE_REAL_EVENTS = True` but file missing: Falls back to synthetic events with warning
- If `USE_REAL_EVENTS = False`: Uses synthetic events (default)

---

### 2. Integration with Real Event Data

The event study engine is **provider-agnostic** and can work with events from various sources:

#### Option A: Finnhub Earnings Calendar

```python
import requests
import pandas as pd
from src.assembled_core.config.settings import get_settings

def load_earnings_events(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load earnings events from Finnhub API."""
    settings = get_settings()
    events = []
    
    for symbol in symbols:
        url = "https://finnhub.io/api/v1/calendar/earnings"
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
            "token": settings.finnhub_api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        for earnings in data.get("earningsCalendar", []):
            events.append({
                "timestamp": pd.Timestamp(earnings["date"], tz="UTC"),
                "symbol": symbol,
                "event_type": "earnings",
                "event_id": f"earnings_{earnings['date']}_{symbol}",
                "eps_estimate": earnings.get("epsEstimate"),
                "eps_actual": earnings.get("epsActual"),
                "surprise": earnings.get("surprise"),
            })
    
    return pd.DataFrame(events)
```

#### Option B: Finnhub Insider Transactions

```python
def load_insider_events(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Load insider transaction events from Finnhub API."""
    settings = get_settings()
    events = []
    
    for symbol in symbols:
        url = f"https://finnhub.io/api/v1/stock/insider-transactions"
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
            "token": settings.finnhub_api_key
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        for transaction in data.get("data", []):
            events.append({
                "timestamp": pd.Timestamp(transaction["transactionDate"], tz="UTC"),
                "symbol": symbol,
                "event_type": "insider_buy" if transaction["transactionCode"] in ["P", "A"] else "insider_sell",
                "event_id": f"insider_{transaction['transactionDate']}_{symbol}_{transaction['name']}",
                "name": transaction.get("name"),
                "transaction_code": transaction.get("transactionCode"),
                "shares": transaction.get("share"),
            })
    
    return pd.DataFrame(events)
```

#### Option C: CSV/Parquet Files

```python
def load_events_from_file(file_path: Path) -> pd.DataFrame:
    """Load events from CSV or Parquet file."""
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Ensure required columns
    required_cols = ["timestamp", "symbol", "event_type"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {file_path}")
    
    # Ensure UTC-aware timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Generate event_id if missing
    if "event_id" not in df.columns:
        df["event_id"] = (
            df["event_type"].astype(str) + "_" +
            df["symbol"].astype(str) + "_" +
            df["timestamp"].dt.strftime("%Y%m%d")
        )
    
    return df
```

**Event DataFrame Format:**
- **Required columns:** `timestamp` (UTC-aware), `symbol`, `event_type`
- **Optional columns:** `event_id` (auto-generated if missing), `payload` (dict or additional columns)
- **Event types:** `earnings`, `insider_buy`, `insider_sell`, `news`, `regulatory`, `custom`

---

### 3. Custom Event Study Workflow

```python
from pathlib import Path
import pandas as pd
from src.assembled_core.config.settings import get_settings
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)

# 1. Load prices
settings = get_settings()
price_source = get_price_data_source(settings, data_source="local")
prices = price_source.get_history(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2025-12-03",
    freq="1d",
)

# 2. Load events (from your preferred source)
events = load_earnings_events(["AAPL", "MSFT", "GOOGL"], "2020-01-01", "2025-12-03")
# OR: events = load_events_from_file(Path("data/events/earnings_2024.csv"))

# 3. Build event windows
windows = build_event_window_prices(
    prices,
    events,
    window_before=20,
    window_after=40,
)

# 4. Compute returns
returns = compute_event_returns(
    windows,
    price_col="close",
    return_type="log",
)

# 5. Aggregate results
aggregated = aggregate_event_study(
    returns,
    use_abnormal=False,  # Use normal returns (no benchmark)
    confidence_level=0.95,
)

# 6. Analyze results
print(f"Event Day Return: {aggregated[aggregated['rel_day'] == 0]['avg_ret'].iloc[0]:.4f}")
print(f"Cumulative Return (Day +5): {aggregated[aggregated['rel_day'] == 5]['cum_ret'].iloc[0]:.4f}")
```

---

## Interpretation Guide

### Key Metrics

**Average Abnormal Return (AAR):**
- Average return across all events for a specific relative day
- Positive AAR on event day (rel_day = 0) suggests positive market reaction
- Significant AAR (outside confidence interval) indicates systematic pattern

**Cumulative Abnormal Return (CAAR):**
- Cumulative sum of AAR from event day onwards
- Positive CAAR suggests persistent positive effect
- Negative CAAR suggests overreaction followed by correction

**Confidence Intervals:**
- 95% CI around AAR indicates statistical significance
- If CI excludes zero, the return is statistically significant
- Wider CI with fewer events indicates higher uncertainty

### Common Patterns

**Earnings Announcements:**
- **Positive Surprise**: Often positive AAR on event day, may persist for 1-5 days
- **Negative Surprise**: Negative AAR on event day, may reverse over 5-10 days
- **Pre-Announcement Drift**: Sometimes positive AAR in days -5 to -1 (information leakage)

**Insider Trading:**
- **Insider Buys**: Often positive AAR over 20-60 days (insiders have information advantage)
- **Insider Sells**: May show negative AAR, but less reliable (multiple reasons for selling)

**News Events:**
- **Positive News**: Positive AAR on event day, may fade over 5-10 days
- **Negative News**: Negative AAR on event day, may persist or reverse depending on severity

---

## Downloading News & Macro Alt-Data (B2)

**Important:** Price data continues to come exclusively from `LocalParquetPriceDataSource` (local Parquet files). Finnhub is used only for news, sentiment, and macro data.

### Download News Events for Event Studies

News events can be used in event studies to analyze price reactions to news announcements:

```powershell
# Download news for symbols from file
python scripts/download_altdata_finnhub_news_macro.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --download-news
```

**Output:**
- Raw: `data/raw/altdata/finnhub/news_raw.parquet`
- Clean: `output/altdata/news_events.parquet`

The cleaned `news_events.parquet` file follows the `news_events_df` data contract and can be used directly with `qa/event_study.py`:

- Required columns: `timestamp`, `symbol` (or None for market-wide), `source`, `headline`, `news_id`, `event_type="news"`
- Optional columns: `category`, `sentiment_score`, `sentiment_label`

**Note:** News events can be filtered by sentiment (positive, negative, neutral) for separate event study analyses.

### Download Macro Indicators for Event Studies

Macro-economic calendar events (e.g., Fed meetings, CPI releases) can also be used in event studies:

```powershell
# Create macro codes file (e.g., config/macro_indicators.txt)
# Contents:
# GDP
# CPI
# UNEMPLOYMENT
# FED_RATE

# Download macro indicators
python scripts/download_altdata_finnhub_news_macro.py `
  --macro-codes-file config/macro_indicators.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --download-macro
```

**Output:**
- Raw: `data/raw/altdata/finnhub/macro_raw.parquet`
- Clean: `output/altdata/macro_series.parquet`

**Note:** Macro indicators are primarily intended for Phase D (Regime Models), but can also be used in event studies for specific economic announcements.

---

## Downloading Real Events from Finnhub (B1)

**Prerequisites:**
- Finnhub API key set via `ASSEMBLED_FINNHUB_API_KEY` environment variable
- Symbols with local price data available (events are filtered to symbols with price data)

**Download Earnings and Insider Events:**

```powershell
# Set Finnhub API key
$env:ASSEMBLED_FINNHUB_API_KEY = "your_finnhub_api_key_here"

# Download earnings events for symbols from file
python scripts/download_altdata_finnhub_events.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings

# Download both earnings and insider events
python scripts/download_altdata_finnhub_events.py `
  --symbols AAPL MSFT GOOGL `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings insider

# Download to custom output directory
python scripts/download_altdata_finnhub_events.py `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings insider `
  --output-dir output/custom_altdata
```

**Output Files:**
- **Raw events**: `data/raw/altdata/finnhub/earnings_events_raw.parquet`, `insider_events_raw.parquet`
- **Cleaned events**: `output/altdata/events_earnings.parquet`, `events_insider.parquet`

**Using Downloaded Events in Event Studies:**

```python
from src.assembled_core.qa.event_study import (
    build_event_window_prices,
    compute_event_returns,
    aggregate_event_study,
)
from src.assembled_core.data.data_source import get_price_data_source
from src.assembled_core.config.settings import get_settings
import pandas as pd

# Load prices (from local Parquet files, NOT from Finnhub)
settings = get_settings()
price_source = get_price_data_source(settings, data_source="local")
prices = price_source.get_history(
    symbols=["AAPL", "MSFT"],
    start_date="2020-01-01",
    end_date="2025-12-03",
    freq="1d",
)

# Load events (from downloaded Parquet files)
events_earnings = pd.read_parquet("output/altdata/events_earnings.parquet")
events_earnings_filtered = events_earnings[
    (events_earnings["symbol"].isin(["AAPL", "MSFT"])) &
    (events_earnings["timestamp"] >= "2020-01-01") &
    (events_earnings["timestamp"] <= "2025-12-03")
]

# Run event study
windows = build_event_window_prices(prices, events_earnings_filtered, window_before=20, window_after=40)
returns = compute_event_returns(windows, return_type="log")
aggregated = aggregate_event_study(returns, use_abnormal=False)
```

**Important Notes:**
- **Finnhub is used ONLY for events**, not for price data
- Price data always comes from local Parquet files via `LocalParquetPriceDataSource`
- Events are automatically filtered to symbols that have local price data
- Rate limits: 60 calls/minute (free tier) - script includes delays between requests

---

## Future CLI Integration

A future CLI command `analyze_events` is planned (see `scripts/run_event_study.py` skeleton):

**Planned Features:**
- Load events from CSV/JSON files
- Use event_study.py functions
- Generate Markdown report + CSV output
- Support for multiple event types
- Benchmark selection (market, sector, custom)

**Example (planned):**
```powershell
python scripts/cli.py analyze_events `
  --events-file data/events/earnings_2024.csv `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --window-before 20 `
  --window-after 40 `
  --output-dir output/event_studies
```

---

## Troubleshooting

### No Events Generated

**Problem:** `generate_synthetic_events()` returns empty DataFrame

**Solutions:**
- Check that price data exists for the specified symbols
- Verify date range overlaps with price data
- Reduce `event_interval_days` if date range is too short

### Missing Price Data for Events

**Problem:** `build_event_window_prices()` returns empty or incomplete windows

**Solutions:**
- Ensure event timestamps fall within price data range
- Check that symbols in events match symbols in prices
- Verify timestamps are UTC-aware and correctly formatted

### Abnormal Returns Always Zero

**Problem:** Abnormal returns equal normal returns (no benchmark effect)

**Solutions:**
- Add benchmark column to prices (e.g., market index)
- Use `benchmark_col` parameter in `compute_event_returns()`
- Check that benchmark prices are correctly aligned with event windows

---

## References

- **Module:** `src/assembled_core/qa/event_study.py`
- **Tests:** `tests/test_qa_event_study.py`
- **Research Template:** `research/events/event_study_template_core.py`
- **Documentation:** `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` (Phase C3)

