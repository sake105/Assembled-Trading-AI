# Data Sources — Status & Usage Guide

Quick reference for all data sources in `src/assembled_core/data/sources/`.
Use this to find the right source for a given data type.

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Wired into production pipeline (ingest_data / trading cycle) |
| 🧪 | Test-covered, not yet in production pipeline |
| 📚 | Single caller outside pipeline (script or util) |
| ⚠️ | No active callers — library-ready but unwired |

---

## Price / OHLCV Data

| Source | File | Status | API Key | Notes |
|--------|------|--------|---------|-------|
| **yfinance** | `yfinance_source.py` | ✅ Pipeline | None (free) | Primary EOD source, no key needed |
| **Polygon.io** | `polygon_source.py` | ✅ Pipeline | `POLYGON_API_KEY` | Paid tier for intraday; free for EOD delayed |
| **Stooq** | `stooq_source.py` | 🧪 Test-only | None (free) | Used in tests; functional alternative to yfinance |

## Macro / Economic Data

| Source | File | Status | API Key | Notes |
|--------|------|--------|---------|-------|
| **FRED** | `fred_source.py` | 📚 Single caller | `FRED_API_KEY` | 800k+ series; free at fred.stlouisfed.org |
| **World Bank** | `worldbank_source.py` | 📚 Single caller | None (free) | Country-level macro, long history |
| **BLS** | `bls_source.py` | 📚 Single caller | None (free) | US labor stats (CPI, jobs, PPI) |

## Corporate Events / Filings

| Source | File | Status | API Key | Notes |
|--------|------|--------|---------|-------|
| **Earnings Calendar** | `earnings_calendar_source.py` | ✅ Pipeline | None (free) | Used by earnings guard in signal step |
| **EDGAR** | `edgar_source.py` | 📚 Single caller | None (free) | SEC filings via `edgartools`; rate-limited |

## News / Sentiment

| Source | File | Status | API Key | Notes |
|--------|------|--------|---------|-------|
| **NewsAPI** | `newsapi_source.py` | 📚 Single caller | `NEWSAPI_KEY` | 100 req/day free; used by news pipeline |
| **Finnhub** | (via altdata/) | 📚 Single caller | `FINNHUB_API_KEY` | Events + news; see `data/altdata/` |
| **Wikipedia Views** | `wikipedia_views_source.py` | 🧪 Test-only | None (free) | Retail attention proxy |

## Alternative / Market-Specific

| Source | File | Status | API Key | Notes |
|--------|------|--------|---------|-------|
| **CBOE** | `cboe_source.py` | 📚 Single caller | None (free) | VIX term structure, put/call ratios |
| **FINRA** | `finra_source.py` | 🧪 Test-only | None (free) | Short interest data |
| **Alpha Vantage** | `alphavantage_source.py` | 📚 Single caller | `ALPHAVANTAGE_API_KEY` | Free tier: 25 req/day; broad coverage |
| **Weather / Energy** | `weather_source.py` | 🧪 Test-only | `NOAA_CDO_TOKEN` | HDD/CDD anomalies for energy sector |

## Prediction Markets (T1.5 source tier)

| Source | File | Status | API Key | Notes |
|--------|------|--------|---------|-------|
| **Polymarket** | `polymarket_source.py` | ✅ Pipeline (live/paper) | None (public API) | CFTC-regulated; geo-risk signal |
| **Kalshi** | `kalshi_source.py` | ✅ Pipeline (live/paper) | None (public API) | CFTC-regulated; blended with Polymarket |

---

## Which Source Should I Use?

| I need... | Use this |
|-----------|---------|
| Daily OHLCV for backtests | `yfinance_source.py` or `polygon_source.py` |
| Real-time / intraday prices | `polygon_source.py` (paid tier) |
| Macro time series (inflation, rates) | `fred_source.py` |
| Earnings dates | `earnings_calendar_source.py` |
| SEC filings | `edgar_source.py` |
| News sentiment | `newsapi_source.py` + `intel/news_rag.py` |
| Geo-risk signal (live) | `polymarket_source.py` + `kalshi_source.py` via `risk/georisk_overlay.py` |
| Short interest | `finra_source.py` |
| VIX data | `cboe_source.py` |

---

## Sources Removed (2026-04-29)

The following files were deleted as they had zero callers and were superseded:

| File | Reason for removal |
|------|-------------------|
| `bluesky_source.py` | No callers; alt-social data not integrated |
| `coinmetrics_source.py` | No callers; crypto metrics not in scope |
| `gdelt_gcam_source.py` | Superseded by `events/news/fetch_gdelt.py` + `events/news/sources.py` |

---

## Style Note

Sources use two patterns (both valid):

- **Functional** (`fetch_*` functions) — majority of sources
- **Class-based** (`CBOESource`, `EarningsCalendarSource`) — 2 older sources

New sources should use the functional pattern for consistency with the majority.
