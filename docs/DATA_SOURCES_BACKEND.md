# Data Sources - Assembled Trading AI Backend

## Übersicht

Dieses Dokument beschreibt alle vorhandenen und geplanten Datenquellen für das Backend von Assembled Trading AI.

**Prinzip:** Offline-first, lokale Daten bevorzugt. Netz-Calls nur in Pull-Skripten.

---

## Aktuelle Datenquellen

### 1. Yahoo Finance (`yfinance`)

**Status:** ✅ Implementiert

**Script:** `scripts/live/pull_intraday.py`

**Daten:**
- Intraday-Preise (1-Minuten, 5-Minuten)
- EOD-Preise (Täglich)
- Volumen-Daten

**Format:**
- Output: `data/raw/1min/{SYMBOL}.parquet`
- Spalten: `timestamp` (UTC), `symbol`, `open`, `high`, `low`, `close`, `volume`

**Limits:**
- Rate-Limits bei zu vielen Requests
- Fallback auf Alpha Vantage empfohlen

**Verwendung:**
```bash
python scripts/live/pull_intraday.py --symbols AAPL,MSFT --days 10
```

---

### 2. Alpha Vantage

**Status:** ✅ Implementiert (Fallback)

**Script:** `scripts/live/pull_intraday_av.py`

**Daten:**
- Intraday-Preise (1-Minuten, 5-Minuten)
- EOD-Preise

**Format:**
- Output: `data/raw/1min/{SYMBOL}.parquet`
- Spalten: `timestamp` (UTC), `symbol`, `open`, `high`, `low`, `close`, `volume`

**Limits:**
- Free-Tier: 5 Requests/Minute, 500 Requests/Tag
- API-Key erforderlich (siehe `config/datasource.psd1`)

**Verwendung:**
```bash
# Setze API-Key als Umgebungsvariable
$env:ALPHAVANTAGE_API_KEY = "your-key"
python scripts/live/pull_intraday_av.py --symbols AAPL --interval 5min
```

---

### 3. Lokale Dateien

**Status:** ✅ Implementiert

**Formate:**
- Parquet (`.parquet`) - Bevorzugt für große Datenmengen
- CSV (`.csv`) - Für kleine Datensätze und Kompatibilität

**Pfade:**
- Rohdaten: `data/raw/1min/*.parquet`
- Aggregierte Daten: `output/aggregates/5min.parquet`, `output/aggregates/daily.parquet`
- Orders: `output/orders_{freq}.csv`
- Equity-Kurven: `output/equity_curve_{freq}.csv`, `output/portfolio_equity_{freq}.csv`

**Verwendung:**
```python
from src.assembled_core.pipeline.io import load_prices, load_orders

# Automatischer Fallback auf verschiedene Pfade
prices = load_prices_with_fallback("5min")
orders = load_orders("1d")
```

---

## Geplante Datenquellen

### 4. Fundamentals (SEC Filings)

**Status:** 🔄 Geplant (Phase 2 - Sprint 8)

**Daten:**
- 10-K, 10-Q Reports
- Finanzkennzahlen (Revenue, Earnings, etc.)
- Bilanzdaten

**Integration:**
- `data/fundamentals.py` - SEC-Filings-Ingestion
- `features/fundamental_features.py` - Fundamental-Analyse-Features

**Quellen:**
- SEC EDGAR API
- Alternative: Drittanbieter-APIs (z. B. Financial Modeling Prep)

---

### 5. Insider-Transaktionen

**Status:** 🔄 Geplant (Phase 2 - Sprint 8)

**Daten:**
- Form 4 Filings (Insider-Käufe/Verkäufe)
- Insider-Positionen
- Executive-Transaktionen

**Integration:**
- `data/insider.py` - Insider-Daten-Ingestion
- `signals/rules_insider.py` - Insider-basierte Signale

**Quellen:**
- SEC EDGAR API (Form 4)
- Alternative: Drittanbieter-APIs

---

### 6. Congress Trading

**Status:** 🔄 Geplant (Phase 3 - Sprint 9)

**Daten:**
- STOCK Act Filings (Congress-Mitglieder)
- Politiker-Portfolio-Transaktionen
- Congress-Trading-Patterns

**Integration:**
- `data/congress.py` - Congress-Daten-Ingestion
- `signals/rules_congress.py` - Congress-basierte Signale

**Quellen:**
- House Stock Watcher
- Senate Financial Disclosures
- Drittanbieter-APIs (z. B. Quiver Quantitative)

---

### 7. Shipping-Daten

**Status:** 🔄 Geplant (Phase 3 - Sprint 10)

**Daten:**
- Container-Schiffsbewegungen
- Hafen-Aktivitäten
- Supply-Chain-Indikatoren

**Integration:**
- `data/shipping.py` - Shipping-Daten-Ingestion
- `features/shipping_features.py` - Shipping-basierte Features

**Quellen:**
- MarineTraffic API
- Port-APIs
- Alternative: Drittanbieter-APIs

---

### 8. News-Feeds

**Status:** 🔄 Geplant (Phase 3 - Sprint 10)

**Daten:**
- Finanznachrichten
- Pressemitteilungen
- Social Media Sentiment

**Integration:**
- `data/news.py` - News-Ingestion
- `features/sentiment.py` - Sentiment-Analyse

**Quellen:**
- NewsAPI
- Finnhub News
- Twitter/X API (für Sentiment)
- Alternative: Drittanbieter-APIs

**Konfiguration:**
- `news_whitelist.yaml` - Erlaubte News-Quellen
- `news_blacklist.yaml` - Blockierte Quellen

---

## Datenfluss

### Ingestion → Processing → Storage

```
┌─────────────────┐
│  Data Source    │  (Yahoo, Alpha Vantage, etc.)
│  (Pull Script)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Raw Data       │  data/raw/1min/{SYMBOL}.parquet
│  (Parquet)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Resampling     │  scripts/run_all_sprint10.ps1
│  (1m → 5m)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Aggregated     │  output/aggregates/5min.parquet
│  Data           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pipeline       │  src/assembled_core/pipeline/
│  Processing     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Files   │  output/orders_*.csv
│  (CSV/Parquet)  │  output/equity_curve_*.csv
└─────────────────┘
```

---

## Datenformate

### Preis-Daten

**Schema:**
- `timestamp` - UTC-Zeitstempel (datetime)
- `symbol` - Ticker-Symbol (string)
- `open`, `high`, `low`, `close` - Preise (float)
- `volume` - Volumen (int/float)

**Format:** Parquet (bevorzugt) oder CSV

---

### Orders

**Schema:**
- `timestamp` - UTC-Zeitstempel (datetime)
- `symbol` - Ticker-Symbol (string)
- `side` - BUY oder SELL (string)
- `qty` - Menge (float)
- `price` - Preis (float)

**Format:** CSV

---

### Equity-Kurven

**Schema:**
- `timestamp` - UTC-Zeitstempel (datetime)
- `equity` - Portfolio-Equity (float)

**Format:** CSV

---

## Konfiguration

### API-Keys

**Datei:** `config/datasource.psd1`

**Umgebungsvariablen:**
- `ALPHAVANTAGE_API_KEY` - Alpha Vantage API-Key
- `FINNHUB_API_KEY` - Finnhub API-Key (geplant)
- `NEWSAPI_KEY` - NewsAPI-Key (geplant)

**Verwendung:**
```powershell
# PowerShell
$env:ALPHAVANTAGE_API_KEY = "your-key"
```

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur
- [Backend Modules](BACKEND_MODULES.md) - Modulübersicht
- [Backend Roadmap](BACKEND_ROADMAP.md) - Entwicklungs-Roadmap

