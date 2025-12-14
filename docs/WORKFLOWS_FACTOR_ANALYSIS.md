# Workflows – Factor Analysis

**Last Updated:** 2025-12-09  
**Status:** Active Workflows for Factor Research & Evaluation

---

## Overview

This document describes workflows for comprehensive factor analysis using the Advanced Analytics & Factor Labs tools (Phase C1 and C2). These workflows enable systematic evaluation of factor effectiveness through:

- **IC-based evaluation (C1)**: Information Coefficient (IC) and Information Ratio (IC-IR)
- **Portfolio-based evaluation (C2)**: Quantile portfolios, Long/Short returns, Sharpe ratios, and Deflated Sharpe ratios

**Prerequisites:**
- Local alt-data available (Parquet files in `ASSEMBLED_LOCAL_DATA_ROOT`)
- Universe files with symbol lists (e.g., `config/macro_world_etfs_tickers.txt`)
- Python environment with dependencies installed

---

## Quick Start

### Basic Factor Analysis

```powershell
# Set environment variables for local alt-data
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
$env:ASSEMBLED_DATA_SOURCE = "local"

# Run comprehensive factor analysis
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+vol_liquidity `
  --horizon-days 20
```

**Output:**
- `output/factor_analysis/factor_analysis_{factor_set}_{horizon_days}d_{freq}_report.md` - Markdown report
- `output/factor_analysis/factor_analysis_{factor_set}_{horizon_days}d_{freq}_ic_summary.csv` - IC statistics
- `output/factor_analysis/factor_analysis_{factor_set}_{horizon_days}d_{freq}_rank_ic_summary.csv` - Rank-IC statistics
- `output/factor_analysis/factor_analysis_{factor_set}_{horizon_days}d_{freq}_portfolio_summary.csv` - Portfolio performance metrics

---

## Factor-Set Übersicht

Die folgenden Factor-Sets sind verfügbar:

| Factor-Set | Inhalt | Phase |
|------------|--------|-------|
| `core` | TA/Price Factors (Multi-Horizon Returns, Trend Strength, Reversal) | Phase A1 |
| `vol_liquidity` | Volatility & Liquidity Factors (RV, Vol-of-Vol, Turnover, Spread Proxies) | Phase A2 |
| `core+vol_liquidity` | Core TA/Price + Volatility/Liquidity | Phase A1 + A2 |
| `all` | Alle TA/Price/Vol/Liquidity Factors (inkl. Market Breadth) | Phase A |
| `alt_earnings_insider` | Nur Earnings/Insider Factors (Earnings Surprise, Insider Activity) | Phase B1 |
| `alt_news_macro` | Nur News/Macro Factors (News Sentiment, Macro Regime) | Phase B2 |
| `core+alt` | Core TA/Price + Earnings/Insider (B1) | Phase A1 + B1 |
| `core+alt_news` | Core TA/Price + News/Macro (B2) | Phase A1 + B2 |
| `core+alt_full` | Core TA/Price + Earnings/Insider (B1) + News/Macro (B2) | Phase A1 + B1 + B2 |

**Hinweis:** Die Factor-Set-Namen sind historisch gewachsen. Eine spätere Umbenennung für bessere UX ist möglich (z.B. `core+alt_b1`, `core+alt_b2`, `core+alt_all`).

**Single Source of Truth:** Die Factor-Set-Namen werden zentral in `scripts/run_factor_analysis.py` durch die Funktion `list_available_factor_sets()` definiert. Diese Funktion wird automatisch von der CLI verwendet (`scripts/cli.py`), um die gültigen `--factor-set` Werte zu validieren.

**Tabelle automatisch generieren:**

Die obige Tabelle kann mit folgendem Python-Snippet automatisch generiert werden:

```python
from scripts.run_factor_analysis import list_available_factor_sets

factor_sets = list_available_factor_sets(with_descriptions=True)

# Mapping von Factor-Set zu Phase
phase_mapping = {
    "core": "Phase A1",
    "vol_liquidity": "Phase A2",
    "core+vol_liquidity": "Phase A1 + A2",
    "all": "Phase A",
    "alt_earnings_insider": "Phase B1",
    "alt_news_macro": "Phase B2",
    "core+alt": "Phase A1 + B1",
    "core+alt_news": "Phase A1 + B2",
    "core+alt_full": "Phase A1 + B1 + B2",
}

print("| Factor-Set | Inhalt | Phase |")
print("|------------|--------|-------|")
for name, desc in factor_sets.items():
    phase = phase_mapping.get(name, "N/A")
    print(f"| `{name}` | {desc} | {phase} |")
```

---

## End-to-End Smoketests

Bevor du mit der Forschung beginnst, solltest du drei konkrete End-to-End-Läufe durchführen, um sicherzustellen, dass alle Komponenten korrekt zusammenarbeiten:

### 1. Plain Core-Factors (nur Alt-Prices)

**Zweck:** Prüft, ob die Basis-Factor-Pipeline funktioniert.

```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core `
  --horizon-days 20
```

**Erwartetes Ergebnis:**
- `output/factor_analysis/factor_ic_summary.csv` mit IC-Statistiken für Core-Faktoren
- `output/factor_analysis/factor_portfolio_summary.csv` mit Portfolio-Performance-Metriken

### 2. Core + B1 (Earnings/Insider)

**Zweck:** Prüft, ob Alt-Data B1 (Earnings/Insider) korrekt integriert ist.

**Voraussetzung:** Alt-Data Events müssen heruntergeladen sein:
```powershell
# Falls noch nicht geschehen:
python scripts/download_altdata_finnhub_events.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings insider
```

**Lauf:**
```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+alt `
  --horizon-days 20
```

**Erwartetes Ergebnis:**
- IC-Statistiken für Core-Faktoren + Earnings/Insider-Faktoren
- Portfolio-Performance für alle Faktoren
- Earnings/Insider-Faktoren sollten in den Reports sichtbar sein

### 3. Core + B1 + B2 (Full Alt-Stack)

**Zweck:** Prüft, ob der komplette Alt-Data-Stack (B1 + B2) korrekt integriert ist.

**Voraussetzung:** Alle Alt-Data müssen heruntergeladen sein:
```powershell
# Events (B1)
python scripts/download_altdata_finnhub_events.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings insider

# News & Macro (B2)
python scripts/download_altdata_finnhub_news_macro.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --download-news-sentiment `
  --download-macro
```

**Lauf:**
```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+alt_full `
  --horizon-days 20
```

**Erwartetes Ergebnis:**
- IC-Statistiken für Core-Faktoren + Earnings/Insider + News/Macro
- Portfolio-Performance für alle Faktoren
- Alle Alt-Data-Faktoren sollten in den Reports sichtbar sein

**Wenn alle drei Reports erfolgreich generiert werden:** Das System ist faktisch "Research-ready" und kann für systematische Factor-Evaluation verwendet werden.

### Smoketests automatisiert

Ein automatisiertes Smoketest-Script führt alle drei Tests nacheinander aus:

```powershell
# Alle drei Tests ausführen
python scripts/run_factor_analysis_smoketests.py
```

**Optionen:**
```powershell
# Einzelne Tests überspringen
python scripts/run_factor_analysis_smoketests.py --skip-test-1
python scripts/run_factor_analysis_smoketests.py --skip-test-2 --skip-test-3

# Custom output directory
python scripts/run_factor_analysis_smoketests.py --output-dir output/custom_smoketests
```

**Das Script:**
- Führt die drei Tests nacheinander aus
- Prüft Exit-Codes
- Validiert, dass alle erwarteten Report-Dateien erstellt wurden
- Gibt eine Zusammenfassung aus

**Voraussetzungen:**
- `ASSEMBLED_LOCAL_DATA_ROOT` Umgebungsvariable gesetzt
- Universe-Dateien vorhanden (`config/macro_world_etfs_tickers.txt`, `config/universe_ai_tech_tickers.txt`)
- Für Tests 2 & 3: Alt-Data-Dateien in `output/altdata/` (falls nicht vorhanden, werden Warnungen geloggt, aber Tests laufen weiter)

**Wichtig:** Das Script verwendet **nur lokale Parquet-Daten** (`data_source="local"`) und macht **keine direkten API-Calls**.

---

## Standard Commands

### 1. Macro ETF Universe Analysis

**Universe:** `config/macro_world_etfs_tickers.txt`  
**Use Case:** Broad market factor evaluation across major ETFs

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

**Typical Results:**
- **High IC-IR factors**: `returns_12m`, `trend_strength_200` (momentum and trend factors)
- **High Sharpe factors**: Factors with consistent predictive power
- **Deflated Sharpe**: Helps identify factors that survive multiple testing adjustment

### 2. AI/Tech Universe Analysis

**Universe:** `config/universe_ai_tech_tickers.txt`  
**Use Case:** Sector-specific factor evaluation for technology stocks

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2005-01-01 `
  --end-date 2025-12-02 `
  --factor-set all `
  --horizon-days 21 `
  --quantiles 10 `
  --output-dir output/factor_analysis/ai_tech
```

**Typical Results:**
- **Volatility factors**: May show higher predictive power in tech sector
- **Liquidity factors**: Volume-based factors may be more relevant for liquid tech stocks
- **Market breadth**: Can be computed separately for sector-level analysis

### 3. Core Factors Only (Fast Analysis)

**Use Case:** Quick evaluation of price-based factors only

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --factor-set core `
  --horizon-days 5 `
  --quantiles 5
```

**Factors Included:**
- Multi-horizon returns (1M, 3M, 6M, 12M)
- Momentum (12M excluding last month)
- Trend strength (20, 50, 200-day lookbacks)
- Short-term reversal (1-3 days)

---

## Downloading News & Macro Alt-Data (B2)

**Important:** Price data continues to come exclusively from `LocalParquetPriceDataSource` (local Parquet files). Finnhub is used only for news, sentiment, and macro data.

### Prerequisites

1. **Finnhub API Key**: Set `ASSEMBLED_FINNHUB_API_KEY` environment variable
2. **Symbol Files**: Text files with symbols (one per line, e.g., `config/universe_ai_tech_tickers.txt`)
3. **Macro Codes File** (optional): Text file with macro indicator codes (e.g., `GDP`, `CPI`, `UNEMPLOYMENT`, `FED_RATE`)

### Download News Events

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

### Download News Sentiment (Daily Aggregated)

```powershell
# Download and aggregate news sentiment
python scripts/download_altdata_finnhub_news_macro.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --download-news-sentiment
```

**Output:**
- Raw: `data/raw/altdata/finnhub/news_sentiment_raw.parquet`
- Clean: `output/altdata/news_sentiment_daily.parquet`

### Download Macro Indicators

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

### Download All (News + Sentiment + Macro)

```powershell
python scripts/download_altdata_finnhub_news_macro.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --macro-codes-file config/macro_indicators.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --download-news `
  --download-news-sentiment `
  --download-macro
```

### Notes

- **Rate Limits**: Finnhub free tier allows 60 calls/minute. The script includes delays between requests.
- **Missing Data**: If no data is found, empty Parquet files are still written (for consistency).
- **Market-Wide News**: Use `--symbols` with empty list or omit `--symbols-file` to fetch market-wide news (symbol=None).
- **Error Handling**: HTTP errors (4xx/5xx) are logged but do not crash the script. Missing symbols/codes are skipped.

---

## Alt-Data 2.0 – News & Macro Factors (B2)

**Overview:**
The Alt-Data 2.0 factors module (`altdata_news_macro_factors.py`) transforms news sentiment and macro-economic indicators into time-series factors that can be evaluated alongside traditional TA/Price factors and B1 Alt-Data factors.

**News Sentiment Factors:**
- `news_sentiment_mean_{lookback_days}d`: Rolling mean of daily sentiment scores
- `news_sentiment_trend_{lookback_days}d`: Trend in sentiment (slope over lookback window)
- `news_sentiment_shock_flag`: Binary flag (1 if sentiment change exceeds threshold)
- `news_sentiment_volume_{lookback_days}d`: Rolling mean of news volume

**Macro Regime Factors:**
- `macro_growth_regime`: Growth regime indicator (+1 = expansion, -1 = recession, 0 = neutral)
- `macro_inflation_regime`: Inflation regime indicator (+1 = high inflation, -1 = low/deflation, 0 = neutral)
- `macro_risk_aversion_proxy`: Risk-on/risk-off indicator based on macro conditions

**Prerequisites:**
Before using News/Macro factors, you must download the data using `download_altdata_finnhub_news_macro.py`:

```powershell
# Download news sentiment and macro data
python scripts/download_altdata_finnhub_news_macro.py `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --macro-codes-file config/macro_indicators.txt `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --download-news-sentiment `
  --download-macro `
  --output-dir output/altdata
```

**Important:** 
- **Price data** continues to come exclusively from `LocalParquetPriceDataSource` (local Parquet files), NOT from Finnhub.
- **News/Macro data** comes from Finnhub API (via downloaded Parquet files in `output/altdata/`).
- Finnhub is used **only** for news, sentiment, and macro data, NOT for price/candle data.

**Example: Core + News/Macro Factors**

```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+alt_news `
  --horizon-days 20 `
  --quantiles 5
```

**Example: All Alt-Data Factors (B1 + B2)**

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+alt_full `
  --horizon-days 20
```

**Example: News/Macro Factors Only**

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --factor-set alt_news_macro `
  --horizon-days 20
```

**Notes:**
- If news/macro files (`output/altdata/news_sentiment_daily.parquet` or `macro_series.parquet`) are missing, the workflow will log warnings and continue without those factors
- News sentiment data is filtered to match the date range of the price data
- Macro regime factors are market-wide (all symbols on the same date get the same values)
- Missing values (NaN) occur when no news/macro data is available for a given symbol/date

**⚠️ Symbol-Mismatches:**
- News-Sentiment für Symbole ohne Price-Historie werden beim Join stillschweigend verworfen
- Market-wide Sentiment (symbol=None oder "__MARKET__") wird an alle Symbole gejoint, auch wenn keine symbol-spezifischen News vorhanden sind
- **Empfehlung:** Siehe Hinweis zu Symbol-Mismatches unter "Using Alt-Data Earnings & Insider Factors (B1)"

---

## Using Alt-Data Earnings & Insider Factors (B1)

**Overview:**
The Alt-Data factors module (`altdata_earnings_insider_factors.py`) transforms earnings and insider events into time-series factors that can be evaluated alongside traditional TA/Price factors.

**Earnings Factors:**
- `earnings_eps_surprise_last`: Last EPS surprise percentage (most recent earnings event)
- `earnings_revenue_surprise_last`: Last revenue surprise percentage
- `earnings_positive_surprise_flag`: Binary flag (1 if last surprise was positive)
- `earnings_negative_surprise_flag`: Binary flag (1 if last surprise was negative)
- `post_earnings_drift_return_{window_days}d`: Forward return after earnings announcement

**Insider Activity Factors:**
- `insider_net_notional_{lookback_days}d`: Net insider notional (buy - sell) over lookback window
- `insider_buy_count_{lookback_days}d`: Number of insider buy transactions
- `insider_sell_count_{lookback_days}d`: Number of insider sell transactions
- `insider_buy_sell_ratio_{lookback_days}d`: Ratio of buys to sells (count-based)
- `insider_net_notional_normalized_{lookback_days}d`: Net notional normalized by market cap proxy

**Prerequisites:**
Before using Alt-Data factors, you must download events using `download_altdata_finnhub_events.py`:

```powershell
# Download earnings and insider events
python scripts/download_altdata_finnhub_events.py `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --event-types earnings insider `
  --output-dir output/altdata
```

**Example: Core + Alt-Data Factors**

```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2015-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+alt `
  --horizon-days 20 `
  --quantiles 5
```

**Example: Alt-Data Factors Only**

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --factor-set alt_earnings_insider `
  --horizon-days 20
```

**Notes:**
- If event files (`output/altdata/events_earnings.parquet` or `events_insider.parquet`) are missing, the workflow will log warnings and continue without those factors
- Events are automatically filtered to match the symbols and date range of the price data
- Factors are computed per symbol and aligned with price timestamps
- Missing values (NaN) occur when no events are available for a given symbol/date

---

## Interpreting Results

### IC Summary (C1 Metrics)

**Key Metrics:**
- **mean_ic**: Average Information Coefficient (correlation between factor and forward returns)
  - Positive: Factor predicts positive returns
  - Negative: Factor predicts negative returns (inverse relationship)
  - Typical range: -0.1 to +0.1 for daily data
- **ic_ir**: Information Ratio = mean_ic / std_ic
  - Higher is better (more consistent predictive power)
  - IC-IR > 0.5 is considered good, > 1.0 is excellent
- **hit_ratio**: Percentage of days with positive IC
  - > 0.5 indicates factor is more often right than wrong
  - > 0.6 is very good

**Example Interpretation:**
```
factor          mean_ic  std_ic  ic_ir   hit_ratio
returns_12m     0.08     0.05    1.60    0.65
trend_strength_200  0.05  0.04    1.25    0.60
```

- `returns_12m` has higher IC-IR (1.60) and hit ratio (0.65) → stronger factor
- Both factors have positive mean_ic → both predict positive forward returns

### Portfolio Summary (C2 Metrics)

**Key Metrics:**
- **annualized_return**: Mean return of Long/Short portfolio (annualized)
  - Positive: Top quantile outperforms bottom quantile
  - Typical range: 5-20% for good factors
- **sharpe**: Sharpe Ratio (risk-adjusted return)
  - Sharpe > 1.0: Good
  - Sharpe > 2.0: Excellent
  - Sharpe < 0.5: Weak factor
- **t_stat**: t-statistic for mean return significance
  - |t| > 2.0: Statistically significant (p < 0.05)
  - |t| > 3.0: Highly significant (p < 0.01)
- **deflated_sharpe**: Deflated Sharpe Ratio (adjusted for multiple testing)
  - DSR > 0: Factor remains significant after adjustment
  - DSR < 0: Factor may be due to luck/overfitting
- **win_ratio**: Percentage of positive Long/Short returns
  - > 0.5: Factor is more often profitable than not
  - > 0.6: Very consistent factor
- **max_drawdown**: Maximum drawdown (negative value)
  - Smaller (less negative) is better
  - Typical range: -10% to -30% for good factors

**Example Interpretation:**
```
factor          annualized_return  sharpe  t_stat  deflated_sharpe  win_ratio
returns_12m     0.15               1.80    3.45    1.20             0.65
trend_strength_200  0.10           1.25    2.50    0.80             0.60
```

- `returns_12m` has higher Sharpe (1.80) and t-stat (3.45) → stronger factor
- Both factors have positive deflated_sharpe → both survive multiple testing adjustment
- `returns_12m` has higher win_ratio (0.65) → more consistent

### Factor Ranking Strategy

**Recommended Approach:**
1. **Start with IC-IR (C1)**: Identifies factors with consistent predictive power
2. **Validate with Sharpe (C2)**: Confirms factors are profitable in portfolio context
3. **Check Deflated Sharpe**: Ensures factors survive multiple testing adjustment
4. **Consider t-stat**: Ensures statistical significance
5. **Review win_ratio**: Ensures consistency (not just a few large wins)

**Factor Selection Criteria:**
- IC-IR > 0.5 AND Sharpe > 1.0 AND Deflated Sharpe > 0 AND |t-stat| > 2.0
- Prefer factors with win_ratio > 0.55 for consistency

---

## Common Factor Patterns

### Typically High IC-IR Factors

**Momentum Factors:**
- `returns_12m`: 12-month returns (long-term momentum)
- `momentum_12m_excl_1m`: 12M momentum excluding last month (avoids short-term reversal)
- **Typical IC-IR**: 0.8 - 1.5
- **Typical Sharpe**: 1.0 - 2.0

**Trend Strength Factors:**
- `trend_strength_200`: Price vs. 200-day moving average (long-term trend)
- `trend_strength_50`: Price vs. 50-day moving average (medium-term trend)
- **Typical IC-IR**: 0.5 - 1.2
- **Typical Sharpe**: 0.8 - 1.5

**Volatility Factors:**
- `rv_20`: 20-day realized volatility (mean-reversion signal)
- `rv_60`: 60-day realized volatility
- **Typical IC-IR**: 0.3 - 0.8 (often negative - high vol predicts lower returns)
- **Typical Sharpe**: 0.5 - 1.2

**Liquidity Factors:**
- `volume_zscore`: Normalized volume (high volume may predict reversals)
- `spread_proxy`: (high - low) / close (illiquidity proxy)
- **Typical IC-IR**: 0.2 - 0.6
- **Typical Sharpe**: 0.3 - 1.0

### Factors to Watch Out For

**Short-Term Reversal:**
- `reversal_1d`, `reversal_2d`, `reversal_3d`
- Often show high IC but low Sharpe (transaction costs matter)
- May have negative deflated_sharpe (overfitting risk)

**Price Columns (Not Factors):**
- `open`, `high`, `low`, `close` may appear in factor list
- These are not factors - ignore or exclude from analysis
- Typically show negative IC (price predicts price, but with look-ahead bias)

---

## Advanced Usage

### Custom Quantiles

Use more quantiles for finer granularity:

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core `
  --horizon-days 20 `
  --quantiles 10  # Use deciles instead of quintiles
```

### Custom Output Directory

```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --data-source local `
  --start-date 2005-01-01 `
  --end-date 2025-12-02 `
  --factor-set all `
  --horizon-days 21 `
  --output-dir output/custom_factor_analysis
```

### Different Forward Horizons

Test different prediction horizons:

```powershell
# Short-term (5 days)
python scripts/cli.py analyze_factors ... --horizon-days 5

# Medium-term (20 days ≈ 1 month)
python scripts/cli.py analyze_factors ... --horizon-days 20

# Long-term (60 days ≈ 3 months)
python scripts/cli.py analyze_factors ... --horizon-days 60
```

**Typical Patterns:**
- **Short-term (5d)**: Reversal factors may work better
- **Medium-term (20d)**: Momentum and trend factors excel
- **Long-term (60d)**: Value and fundamental factors may dominate

---

## Troubleshooting

### No Portfolio Summary Generated

**Symptom:** `portfolio_summary.csv` is missing or empty

**Causes:**
- Insufficient data (fewer than `min_obs` symbols per timestamp)
- Too short time period (need at least 20-30 days for meaningful statistics)
- All factors have NaN values

**Solution:**
- Use longer time period (e.g., 1+ year of data)
- Use larger universe (more symbols)
- Check data quality (ensure factors are computed correctly)

### Low IC-IR Across All Factors

**Symptom:** All factors have IC-IR < 0.3

**Possible Causes:**
- Market regime change (factors may not work in all regimes)
- Data quality issues (missing data, incorrect timestamps)
- Wrong forward horizon (try different `--horizon-days`)

**Solution:**
- Check data completeness: `python scripts/check_data_completeness.py`
- Try different time periods (split into bull/bear markets)
- Experiment with different forward horizons

### Negative Deflated Sharpe

**Symptom:** Deflated Sharpe < 0 for factors with high raw Sharpe

**Interpretation:**
- Factor may be due to luck/multiple testing
- Overfitting risk (factor works in-sample but may not work out-of-sample)
- Need more data or stricter selection criteria

**Solution:**
- Use longer time periods (more observations)
- Test on out-of-sample data
- Combine multiple factors (diversification)

---

## Integration with Research Notebooks

The factor analysis results can be loaded into Jupyter notebooks for further analysis:

```python
import pandas as pd
from pathlib import Path

# Load IC summary
ic_summary = pd.read_csv("output/factor_analysis/factor_analysis_core_20d_1d_ic_summary.csv")

# Load portfolio summary
portfolio_summary = pd.read_csv("output/factor_analysis/factor_analysis_core_20d_1d_portfolio_summary.csv")

# Combine for comprehensive ranking
combined = ic_summary.merge(
    portfolio_summary,
    on="factor",
    suffixes=("_ic", "_portfolio")
)

# Rank by combined score (IC-IR + Sharpe)
combined["combined_score"] = combined["ic_ir"] + combined["sharpe"]
combined = combined.sort_values("combined_score", ascending=False)

print(combined[["factor", "ic_ir", "sharpe", "deflated_sharpe", "combined_score"]].head(10))
```

---

## References

- **Factor Analysis Documentation**: `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`
- **Phase C1 (IC Engine)**: IC/IR-based factor evaluation
- **Phase C2 (Portfolio Engine)**: Portfolio-based factor evaluation
- **Factor Modules**: `src/assembled_core/features/ta_factors_core.py`, `ta_liquidity_vol_factors.py`

---

## Next Steps

1. **Factor Selection**: Use IC-IR and Sharpe to select top factors
2. **Factor Combination**: Test combinations of top factors
3. **Regime Analysis**: Analyze factor performance across market regimes (Phase D)
4. **ML Integration**: Use factors as features for ML models (Phase E)

