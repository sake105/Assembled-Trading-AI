# Known Issues & Open Topics

**Letzte Aktualisierung:** 2025-01-15  
**Status:** Phase 12.3 – Review-Vorbereitung

Dieses Dokument listet bekannte offene Punkte, technische Schulden und geplante Erweiterungen im Backend von Assembled Trading AI.

---

## 1. Funktionale Open Points

### 1.1 Labeling-Schemata (ML)

- [ ] **[enhancement]** `binary_outperformance` Labeling nicht vollständig implementiert  
  **Datei:** `src/assembled_core/qa/labeling.py` (Zeile ~277)  
  **Beschreibung:** Die Funktion `generate_trade_labels()` unterstützt aktuell nur `binary_absolute`. `binary_outperformance` (Vergleich mit Benchmark) und `multi_class` sind als TODO markiert.

- [ ] **[enhancement]** Multi-Class-Labeling für ML-Datasets  
  **Datei:** `src/assembled_core/qa/labeling.py` (Zeile ~359)  
  **Beschreibung:** Multi-Class-Labeling (z.B. 0=loss, 1=small_gain, 2=large_gain) ist geplant, aber noch nicht implementiert.

### 1.2 Trade-Level-Metriken

- [ ] **[enhancement]** Position-Tracking für präzise Trade-Level-Metriken  
  **Datei:** `src/assembled_core/qa/metrics.py` (Zeile ~369-375)  
  **Beschreibung:** Metriken wie `hit_rate`, `profit_factor`, `avg_win`, `avg_loss` sind als TODO markiert und benötigen Position-Tracking für genaue Berechnung.

### 1.3 Pre-Trade-Checks

- [ ] **[enhancement]** Weight-Checking in Pre-Trade-Checks  
  **Datei:** `src/assembled_core/execution/pre_trade_checks.py` (Zeile ~215)  
  **Beschreibung:** Weight-Checking (Position-Größe relativ zum Portfolio) ist als TODO markiert und wartet auf Portfolio + Capital-Info.

- [ ] **[enhancement]** Sector-Exposure-Checks  
  **Datei:** `src/assembled_core/execution/pre_trade_checks.py` (Zeile ~243)  
  **Beschreibung:** Sector-Exposure-Limits sind geplant, aber noch nicht implementiert (benötigt Sector-Daten).

- [ ] **[enhancement]** Region-Exposure-Checks  
  **Datei:** `src/assembled_core/execution/pre_trade_checks.py` (Zeile ~247)  
  **Beschreibung:** Region-Exposure-Limits sind geplant, aber noch nicht implementiert (benötigt Region-Daten).

### 1.4 Monitoring-API

- [ ] **[enhancement]** Persistierte Drift-Analyse-Ergebnisse  
  **Datei:** `src/assembled_core/api/routers/monitoring.py` (Zeile ~291)  
  **Beschreibung:** Monitoring-API liefert aktuell Dummy-Daten für Drift-Status. Persistierung von Drift-Analyse-Ergebnissen ist geplant.

### 1.5 Live-Trading-Mode

- [ ] **[enhancement]** Live-Trading-Mode (Environment.LIVE)  
  **Datei:** `src/assembled_core/config/settings.py` (Zeile ~28)  
  **Beschreibung:** Live-Trading-Mode ist als Kommentar markiert ("Future: Live trading mode (not yet implemented)").

---

## 2. Technische Schulden

### 2.1 Legacy-Migration

- [ ] **[tech-debt]** Legacy-Skripte migrieren/bereinigen  
  **Dateien:** `docs/LEGACY_OVERVIEW.md`, `docs/LEGACY_TO_CORE_MAPPING.md`  
  **Beschreibung:** Viele Legacy-Skripte (z.B. `sprint9_dashboard.ps1`, `sprint9_cost_grid.ps1`, `sprint10_param_sweep.ps1`) sind noch vorhanden, aber nicht in die neue Core-Architektur migriert. Status: "TODO: Phase 5/6".

- [ ] **[tech-debt]** Intraday-Resampling in Core-Architektur integrieren  
  **Datei:** `docs/LEGACY_TO_CORE_MAPPING.md` (Zeile ~23)  
  **Beschreibung:** Resampling 1m → 5m ist in Legacy-Skripten vorhanden (`scripts/50_resample_intraday.ps1`), aber noch nicht als Core-Modul (`src/assembled_core/data/resample.py`) implementiert.

### 2.2 Meta-Model-Training

- [ ] **[tech-debt]** Validation-Split für Meta-Model-Training  
  **Datei:** `src/assembled_core/signals/meta_model.py` (Zeile ~198)  
  **Beschreibung:** Aktuell wird auf allen Daten trainiert. Ein Validation-Split für Out-of-Sample-Validierung wäre wünschenswert.

### 2.3 API-Models

- [ ] **[tech-debt]** API-Models-Dokumentation aktualisieren  
  **Datei:** `src/assembled_core/api/models.py` (Zeile ~2)  
  **Beschreibung:** Docstring erwähnt "future implementation", aber Models sind bereits implementiert.

---

## 3. Performance & Skalierung

### 3.1 Backtest-Performance

- [ ] **[enhancement]** Parallelisierung von Backtests  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Für größere Datensätze oder Parameter-Sweeps wäre Parallelisierung (Multi-Processing) wünschenswert.

- [ ] **[enhancement]** Caching von Features  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Feature-Berechnungen könnten gecacht werden, um wiederholte Berechnungen zu vermeiden.

### 3.2 Daten-Ingest

- [ ] **[enhancement]** Incremental-Backtests  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Nur neue Daten verarbeiten, statt vollständigen Backtest neu zu starten.

---

## 4. Nice-to-Haves

### 4.1 Erweiterte Strategien

- [ ] **[enhancement]** Mean-Reversion-Strategien  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.1)  
  **Beschreibung:** RSI-basierte Mean-Reversion, Bollinger-Band-Mean-Reversion, Pairs-Trading.

- [ ] **[enhancement]** Breakout-Strategien  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.1)  
  **Beschreibung:** Bollinger-Band-Breakouts, Support/Resistance-Breakouts.

- [ ] **[enhancement]** Multi-Timeframe-Trend  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.1)  
  **Beschreibung:** Kombination von 1d- und 5min-Trend-Signalen.

### 4.2 Erweiterte Alt-Daten

- [ ] **[enhancement]** Congress-Trading-Daten  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.2)  
  **Beschreibung:** Congress-Member-Trades als Feature integrieren.

- [ ] **[enhancement]** News-Sentiment-Scoring  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.2)  
  **Beschreibung:** FinBERT oder ähnliches für News-Sentiment verwenden.

- [ ] **[enhancement]** Makro-Daten  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.2)  
  **Beschreibung:** Economic-Indicators (CPI, Unemployment), Fed-Announcements.

### 4.3 ML-Experimente

- [ ] **[enhancement]** Feature-Selection-Pipeline  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.3)  
  **Beschreibung:** Automatische Feature-Selection für Meta-Modelle (Univariate, RFE, L1-Regularization).

- [ ] **[enhancement]** SHAP-Explainability  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.3)  
  **Beschreibung:** SHAP-Values für Meta-Modelle berechnen und visualisieren.

- [ ] **[enhancement]** Walk-Forward-Analyse-Tool  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Walk-Forward-Analyse-Tool für robuste Validierung.

### 4.4 Visualisierung

- [ ] **[enhancement]** Erweiterte Reports  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Strategy-Comparison-Reports, Regime-Analysis-Reports, Feature-Importance-Reports.

- [ ] **[enhancement]** Bessere Visualisierung  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Equity-Curve-Plots mit Drawdowns, Trade-Distribution-Plots, Feature-Correlation-Matrix.

---

## 5. Dokumentation & Review

### 5.1 Research-Notebooks

- [ ] **[enhancement]** Research-Notebook-Templates ausfüllen  
  **Dateien:** `research/trend/trend_baseline_experiments.ipynb`, `research/meta/meta_model_calibration.ipynb`, etc.  
  **Beschreibung:** Notebook-Templates enthalten TODOs und müssen mit konkreten Experimenten gefüllt werden.

### 5.2 Legacy-Dokumentation

- [ ] **[tech-debt]** Legacy-Mapping vervollständigen  
  **Datei:** `docs/LEGACY_TO_CORE_MAPPING.md`  
  **Beschreibung:** Viele Einträge sind noch als "TODO" markiert und müssen ausgefüllt werden.

---

## Priorisierung

**Hoch (sollte bald angegangen werden):**
- Trade-Level-Metriken (Position-Tracking)
- `binary_outperformance` Labeling
- Legacy-Migration (wenn Legacy-Skripte noch aktiv genutzt werden)

**Mittel (wichtig, aber nicht kritisch):**
- Pre-Trade-Checks (Weight, Sector, Region)
- Persistierte Drift-Analyse
- Validation-Split für Meta-Modelle

**Niedrig (Nice-to-Have):**
- Erweiterte Strategien (Mean-Reversion, Breakout)
- Erweiterte Alt-Daten (Congress, News-Sentiment)
- Performance-Optimierungen (Parallelisierung, Caching)

---

**Hinweis:** Dieses Dokument wird regelmäßig aktualisiert. Neue Issues sollten hier eingetragen werden, bevor sie in GitHub Issues erstellt werden.

