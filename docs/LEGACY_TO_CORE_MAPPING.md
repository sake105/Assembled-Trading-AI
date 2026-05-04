# Legacy to Core Mapping - Assembled Trading AI

**Letzte Aktualisierung:** 2026-05-05 (ECB FX + CoinGecko OHLC implementiert)

## Ziel

Dieses Dokument mappt Legacy-Flows (alte PowerShell-Jobs, Skripte, etc.) auf die neue Core-Architektur und zeigt den Migrations-Status.

## Mapping-Tabelle

| Legacy-Name | Beschreibung | Neuer Core-Einstiegspunkt | Status | Notizen |
|-------------|--------------|---------------------------|--------|---------|
| **Täglicher EOD-Lauf** | Täglicher End-of-Day-Pipeline-Lauf | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Verwendet `src/assembled_core/pipeline/orchestrator.py` |
| **Backtest-Einmal-Run** | Einmaliger Strategy-Backtest-Run | `scripts/run_backtest_strategy.py` | ✅ **Fertig** | Verwendet `src/assembled_core/qa/backtest_engine.py` |
| **Phase-4-Tests** | Regression-Tests für Phase-4-Kern | `pytest -m phase4` | ✅ **Fertig** | 5400+ Tests gesammelt |
| **Sprint-9-Backtest** | Legacy Sprint-9-Backtest | `scripts/run_backtest_strategy.py` | ✅ **Fertig** | Ersetzt durch neue Backtest-Engine |
| **Sprint-9-Execute** | Legacy Sprint-9-Execute | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Sprint-10-Portfolio** | Legacy Sprint-10-Portfolio | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Run-Daily (Legacy)** | Legacy täglicher Run | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Sprint-10-All-in-One** | Legacy Sprint-10-All-in-One | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Stooq EOD-Pull** | Legacy Stooq EOD-Daten-Pull | `src/assembled_core/data/prices_ingest.py` — `load_eod_prices()` | ✅ **Fertig** | Yahoo Finance Fallback via yfinance integriert |
| **AlphaVantage Intraday-Pull** | Legacy AlphaVantage Intraday-Pull | `scripts/live/pull_intraday.py` | ⚠️ **In Migration** | Pull-Skript existiert; API-Key via `.env` |
| **Intraday-Resampling** | Resampling 1m → 5m/15m/1h/1d | `src/assembled_core/data/resample.py` | ✅ **Fertig** | Multi-Timeframe-Resampling vollständig implementiert (2026-05-05) |
| **Intraday-QC-Gaps** | Quality-Check für Intraday-Gaps | `src/assembled_core/qa/health.py` — `check_prices()` | ✅ **Fertig** | Gap-Detection + OHLC-Sanity in health.py |
| **Daily-Features-Build** | Legacy Daily-Features-Build | `src/assembled_core/features/ta_features.py` — `add_all_features()` | ✅ **Fertig** | TA + Volatility + Volume + Alt-Daten-Features |
| **Cost-Model-Tests** | Legacy Cost-Model-Tests | `src/assembled_core/costs.py` — `CostModel` | ✅ **Fertig** | Ersetzt durch neue Cost-Model-Klasse |
| **Dashboard-Generierung** | Legacy Dashboard-Generierung | `src/assembled_core/reports/daily_qa_report.py` | ✅ **Fertig** | QA-Report + Metrics-Export implementiert |
| **Parameter-Sweep** | Legacy Parameter-Sweep | `scripts/batch_runner.py --max-workers N` | ✅ **Fertig** | ProcessPoolExecutor, YAML-Config, Manifest-Tracking |
| **Cost-Grid** | Legacy Cost-Grid | `scripts/batch_runner.py` mit `cost_model`-Parameter | ✅ **Fertig** | Cost-Varianten via YAML-Batch-Config sweepbar |
| **Rehydrate** | Legacy Rehydrate (Feature-Reload) | `src/assembled_core/data/factor_store.py` — `load_factors()` | ✅ **Fertig** | Append-Mode + Partition-basiertes Laden |
| **CoinGecko OHLC-Pull** | Legacy CoinGecko OHLC-Pull | `src/assembled_core/data/crypto.py` — `fetch_coingecko_ohlc()` | ✅ **Fertig** | Free-Tier Public API; kein Key nötig; rate-limited; OHLC + market_chart + coin list (2026-05-05) |
| **ECB FX-Pull** | Legacy ECB FX-Pull | `src/assembled_core/data/fx.py` — `fetch_ecb_fx_rates()` | ✅ **Fertig** | ECB SDMX-CSV API; kein Key nötig; long + wide format; 22 EUR-Paare (2026-05-05) |
| **Congress-Daten** | Congress-Member-Trades als Feature | `src/assembled_core/features/congress_features.py` | ✅ **Fertig** | Quiver Quant API-kompatibel (2026-04-29) |
| **Insider-Daten** | Insider-Trades als Feature | `src/assembled_core/data/altdata/` + `output/insider_trading.parquet` | ✅ **Fertig** | Via yfinance/Finnhub; Datenqualität: alle `transaction_type='unknown'` |
| **News-Feeds** | News-Sentiment + Trigger-Signale | `src/assembled_core/intel/` (rss_fetcher, news_sentiment_drift, finbert_sentiment) | ✅ **Fertig** | FinBERT-Wrapper mit VADER/Keyword-Fallback (2026-05-05) |
| **Shipping-Daten** | Baltic-Dry-Index / Container-Rates | `src/assembled_core/features/shipping_features.py` | ✅ **Fertig** | BDI + CCFI + Drewry-WCI als Features |

## Status-Legende

- ✅ **Fertig**: Vollständig migriert, in Betrieb
- ⚠️ **In Migration**: Teilweise migriert, noch in Arbeit
- ❓ **Nicht migriert**: Bewusst außerhalb des aktuellen Scope

## Migrations-Roadmap

### Phase 4 (Abgeschlossen) ✅

- ✅ Backtest-Engine → `src/assembled_core/qa/backtest_engine.py`
- ✅ QA-Metriken → `src/assembled_core/qa/metrics.py`
- ✅ QA-Gates → `src/assembled_core/qa/qa_gates.py`
- ✅ TA-Features → `src/assembled_core/features/ta_features.py`
- ✅ EOD-Pipeline → `scripts/run_eod_pipeline.py`
- ✅ Strategy-Backtest → `scripts/run_backtest_strategy.py`
- ✅ Cost-Model → `src/assembled_core/costs.py`

### Phase 5 (Abgeschlossen) ✅

- ✅ Intraday-Resampling → `src/assembled_core/data/resample.py`
- ✅ QC-Gaps → `src/assembled_core/qa/health.py`
- ✅ Dashboard/Reports → `src/assembled_core/reports/`
- ✅ Parameter-Sweep → `scripts/batch_runner.py --max-workers`
- ✅ Feature-Caching → `src/assembled_core/data/factor_store.py`

### Phase 6 (Abgeschlossen) ✅

- ✅ Congress-Daten → `src/assembled_core/features/congress_features.py`
- ✅ Insider-Daten → `src/assembled_core/data/altdata/`
- ✅ Shipping-Daten → `src/assembled_core/features/shipping_features.py`
- ✅ News-Feeds → `src/assembled_core/intel/`
- ✅ ECB FX → `src/assembled_core/data/fx.py` — `fetch_ecb_fx_rates()` + `fetch_ecb_fx_wide()` (2026-05-05)
- ✅ CoinGecko OHLC → `src/assembled_core/data/crypto.py` — `fetch_coingecko_ohlc()` + market_chart (2026-05-05)

## Legacy → Core Ersetzungs-Strategie

### 1. CLI-Commands ersetzen PowerShell-Jobs

**Vorher (Legacy)**:
```powershell
# Task Scheduler startet:
.\scripts\run_all_sprint10.ps1 -Symbols "AAPL,MSFT" -Days 2
```

**Nachher (Core)**:
```powershell
# Task Scheduler startet:
python scripts\run_eod_pipeline.py --freq 1d --universe watchlist.txt
```

### 2. Batch-Backtests mit Parallelisierung

**Vorher (Legacy)**:
```powershell
# Manueller Parameter-Sweep mit PowerShell-Schleifen
.\scripts\sprint10_param_sweep.ps1 -Windows "20,50,100"
```

**Nachher (Core)**:
```bash
# YAML-gesteuert, parallel, mit Manifest-Tracking
python scripts/batch_runner.py --config-file configs/batch_example.yaml --max-workers 4
```

### 3. Feature-Caching für schnelle Re-Runs

**Vorher (Legacy)**:
```python
# Kein Caching — jeder Run berechnet alles neu
features_df = add_all_features(prices_df)
```

**Nachher (Core)**:
```python
# Factor-Store mit Append-Mode
from src.assembled_core.data.factor_store import store_factors, load_factors
existing = load_factors(universe_key, factors_root)
if existing is None:
    store_factors(new_factors, universe_key, factors_root)
```

### 4. News-Sentiment mit FinBERT-Fallback

**Vorher (Legacy)**:
```python
# Kein strukturiertes Sentiment
```

**Nachher (Core)**:
```python
from src.assembled_core.intel.finbert_sentiment import get_sentiment_scorer
scorer = get_sentiment_scorer()  # auto-detects best backend
result = scorer.score("Earnings beat expectations by 15%.")
# SentimentResult(score=0.82, label='positive', backend='finbert')
```
