# Legacy to Core Mapping - Assembled Trading AI

**Letzte Aktualisierung:** 2025-01-15

## Ziel

Dieses Dokument mappt Legacy-Flows (alte PowerShell-Jobs, Skripte, etc.) auf die neue Core-Architektur und zeigt den Migrations-Status.

## Mapping-Tabelle

| Legacy-Name | Beschreibung | Neuer Core-Einstiegspunkt | Status | Notizen |
|-------------|--------------|---------------------------|--------|---------|
| **Täglicher EOD-Lauf** | Täglicher End-of-Day-Pipeline-Lauf (Execute → Backtest → Portfolio → QA) | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Phase 4 abgeschlossen. Verwendet `src/assembled_core/pipeline/orchestrator.py` |
| **Backtest-Einmal-Run** | Einmaliger Strategy-Backtest-Run | `scripts/run_backtest_strategy.py` | ✅ **Fertig** | Phase 4 abgeschlossen. Verwendet `src/assembled_core/qa/backtest_engine.py` |
| **Phase-4-Tests** | Regression-Tests für Phase-4-Kern | `scripts/run_phase4_tests.ps1`<br/>oder<br/>`pytest -m phase4` | ✅ **Fertig** | Phase 4 abgeschlossen. 110 Tests, ~13s |
| **Sprint-9-Backtest** | Legacy Sprint-9-Backtest | `scripts/run_backtest_strategy.py` | ✅ **Fertig** | Ersetzt durch neue Backtest-Engine |
| **Sprint-9-Execute** | Legacy Sprint-9-Execute | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Sprint-10-Portfolio** | Legacy Sprint-10-Portfolio | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Run-Daily (Legacy)** | Legacy täglicher Run | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Sprint-10-All-in-One** | Legacy Sprint-10-All-in-One | `scripts/run_eod_pipeline.py` | ✅ **Fertig** | Ersetzt durch neue EOD-Pipeline |
| **Stooq EOD-Pull** | Legacy Stooq EOD-Daten-Pull | `src/assembled_core/data/prices_ingest.py`<br/>`load_eod_prices()` | ⚠️ **In Migration** | Daten-Ingest in Core integriert, aber Pull-Logik noch in Legacy |
| **AlphaVantage Intraday-Pull** | Legacy AlphaVantage Intraday-Pull | `scripts/live/pull_intraday.py`<br/>oder<br/>`scripts/live/pull_intraday_av.py` | ⚠️ **In Migration** | Pull-Skript existiert, aber noch nicht vollständig in Core integriert |
| **Intraday-Resampling** | Resampling 1m → 5m | `TODO: Phase 5`<br/>`src/assembled_core/data/resample.py` (geplant) | ⚠️ **Geplant** | Phase 5 |
| **Intraday-QC-Gaps** | Quality-Check für Intraday-Gaps | `src/assembled_core/qa/health.py`<br/>`check_prices()` | ⚠️ **In Migration** | Health-Check existiert, aber Gap-Detection noch nicht vollständig |
| **Daily-Features-Build** | Legacy Daily-Features-Build | `src/assembled_core/features/ta_features.py`<br/>`add_all_features()` | ✅ **Fertig** | Ersetzt durch neue TA-Features |
| **Cost-Model-Tests** | Legacy Cost-Model-Tests | `src/assembled_core/costs.py`<br/>`CostModel` | ✅ **Fertig** | Ersetzt durch neue Cost-Model-Klasse |
| **Dashboard-Generierung** | Legacy Dashboard-Generierung | `TODO: Phase 5/6` | ❓ **Unbekannt** | Zu klären in Phase 5/6 |
| **Parameter-Sweep** | Legacy Parameter-Sweep | `TODO: Phase 5/6` | ❓ **Unbekannt** | Zu klären in Phase 5/6 |
| **Cost-Grid** | Legacy Cost-Grid | `TODO: Phase 5/6` | ❓ **Unbekannt** | Zu klären in Phase 5/6 |
| **Rehydrate** | Legacy Rehydrate | `TODO: Phase 5/6` | ❓ **Unbekannt** | Zu klären in Phase 5/6 |
| **CoinGecko OHLC-Pull** | Legacy CoinGecko OHLC-Pull | `TODO: Phase 5/6` | ❓ **Unbekannt** | Zu klären in Phase 5/6 |
| **ECB FX-Pull** | Legacy ECB FX-Pull | `TODO: Phase 5/6` | ❓ **Unbekannt** | Zu klären in Phase 5/6 |
| **TODO: [Legacy-Name]** | TODO: [Beschreibung] | `TODO: [Core-Einstiegspunkt]` | ❓ **Unbekannt** | Bitte ergänzen |

## Status-Legende

- ✅ **Fertig**: Vollständig migriert, in Betrieb (Phase 4)
- ⚠️ **In Migration**: Teilweise migriert, noch in Arbeit
- ⚠️ **Geplant**: Geplant für Phase 5/6
- ❓ **Unbekannt**: Status unklar, muss geklärt werden

## Migrations-Roadmap

### Phase 4 (Abgeschlossen) ✅

- ✅ Backtest-Engine → `src/assembled_core/qa/backtest_engine.py`
- ✅ QA-Metriken → `src/assembled_core/qa/metrics.py`
- ✅ QA-Gates → `src/assembled_core/qa/qa_gates.py`
- ✅ TA-Features → `src/assembled_core/features/ta_features.py`
- ✅ EOD-Pipeline → `scripts/run_eod_pipeline.py`
- ✅ Strategy-Backtest → `scripts/run_backtest_strategy.py`
- ✅ Cost-Model → `src/assembled_core/costs.py`
- ✅ Phase-4-Tests → `scripts/run_phase4_tests.ps1` / `pytest -m phase4`

### Phase 5 (Geplant) ⚠️

- ⚠️ Intraday-Pull → `src/assembled_core/data/intraday_ingest.py` (geplant)
- ⚠️ Resampling → `src/assembled_core/data/resample.py` (geplant)
- ⚠️ QC-Gaps → `src/assembled_core/qa/health.py` (erweitern)
- ⚠️ Dashboard-Generierung → `src/assembled_core/reports/dashboard.py` (geplant)
- ⚠️ Parameter-Sweep → `src/assembled_core/qa/param_sweep.py` (geplant)

### Phase 6 (Geplant) ⚠️

- ⚠️ Insider-Daten → `src/assembled_core/data/insider.py` (geplant)
- ⚠️ Congress-Daten → `src/assembled_core/data/congress.py` (geplant)
- ⚠️ Shipping-Daten → `src/assembled_core/data/shipping.py` (geplant)
- ⚠️ News-Feeds → `src/assembled_core/data/news.py` (geplant)
- ⚠️ CoinGecko OHLC → `src/assembled_core/data/crypto.py` (geplant)
- ⚠️ ECB FX → `src/assembled_core/data/fx.py` (geplant)

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

### 2. Python-Skripte ersetzen PowerShell-Wrapper

**Vorher (Legacy)**:
```powershell
# PowerShell-Wrapper mit vielen Parametern
.\scripts\sprint9_backtest.ps1 -Symbols "AAPL" -StartCapital 10000
```

**Nachher (Core)**:
```bash
# Direkter Python-Call mit CLI-Args
python scripts/run_backtest_strategy.py --freq 1d --strategy trend_baseline --start-capital 10000
```

### 3. Core-Module ersetzen Legacy-Skripte

**Vorher (Legacy)**:
```python
# Legacy: scripts/features/build_daily_features.py
# Viele globale Variablen, unsaubere Struktur
```

**Nachher (Core)**:
```python
# Core: src/assembled_core/features/ta_features.py
from src.assembled_core.features.ta_features import add_all_features
features_df = add_all_features(prices_df)
```

### 4. Test-Infrastruktur ersetzen manuelle Checks

**Vorher (Legacy)**:
```powershell
# Manuelle Checks, keine Automatisierung
.\scripts\52_make_acceptance_intraday_sprint7.ps1
```

**Nachher (Core)**:
```bash
# Automatisierte Tests
pytest -m phase4 -q
# oder
.\scripts\run_phase4_tests.ps1
```

## Grafische Übersicht: Legacy → Core Migration

```
Legacy-Welt                          Core-Architektur
─────────────────────────────────────────────────────────────
┌─────────────────────┐            ┌─────────────────────┐
│ PowerShell-Jobs     │            │ CLI-Commands        │
│ (Task Scheduler)    │   ────→    │ (Python Scripts)    │
└─────────────────────┘            └─────────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────────┐            ┌─────────────────────┐
│ Legacy-Skripte      │            │ Core-Module          │
│ (sprint9_*.py)      │   ────→    │ (assembled_core.*)  │
│ (run_daily.py)      │            │                      │
└─────────────────────┘            └─────────────────────┘
         │                                   │
         ▼                                   ▼
┌─────────────────────┐            ┌─────────────────────┐
│ Manuelle Checks     │            │ Test-Infrastruktur  │
│ (Acceptance-Tests)  │   ────→    │ (pytest -m phase4)  │
└─────────────────────┘            └─────────────────────┘
```

## Empfohlene Migrations-Schritte

1. **Identifizieren**: Alle Legacy-Komponenten in `LEGACY_OVERVIEW.md` dokumentieren
2. **Mappen**: Legacy → Core Mapping in dieser Datei erstellen
3. **Ersetzen**: Schrittweise Legacy durch Core ersetzen
4. **Testen**: Nach jeder Ersetzung Phase-4-Tests laufen lassen
5. **Dokumentieren**: Status in dieser Datei aktualisieren

## Hinweise

1. **Work in Progress**: Diese Dokumentation wird nach und nach ergänzt, wenn weitere Legacy-Komponenten migriert werden.

2. **Platzhalter**: Alle Einträge mit "TODO:" sind Platzhalter und müssen vom Benutzer ausgefüllt werden.

3. **Status-Updates**: Bitte aktualisieren Sie den Status, wenn Komponenten migriert werden.

4. **Ergänzungen**: Bitte ergänzen Sie diese Tabelle mit weiteren bekannten Legacy-Komponenten.

