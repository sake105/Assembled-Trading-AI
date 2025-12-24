# Batch Runner & Parallelisierung (P4) - Design & Task-Liste

**Last Updated:** 2025-01-XX  
**Status:** Design Phase - Task-Liste  
**Related:** [Performance Profiling P1](PERFORMANCE_PROFILING_P1_DESIGN.md), [Backtest Engine Optimization P3](BACKTEST_ENGINE_OPTIMIZATION_P3.md), [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md)

---

## 1. Overview & Current State

**Goal:** Systematische Ausführung vieler Backtests (Parameter-Sweeps, Bundle-Vergleiche, Experimente) mit optionaler Parallelisierung und sauberer Metriken-Aggregation.

**Current State (Stand 2025-01-XX):**

### Vorhandene Entry Points

1. **Batch-Runner (Basis):**
   - `scripts/batch_backtest.py`: Serieller Batch-Runner mit YAML/JSON-Config
   - CLI: `python scripts/cli.py batch_backtest --config-file <config.yaml>`
   - Status: Funktioniert seriell, `max_workers` nur validiert (nicht implementiert)

2. **Parameter-Sweeps (Ad-hoc):**
   - `scripts/sprint10_param_sweep.ps1`: PowerShell-basierter Sweep (exposure, commission)
   - `scripts/tools/param_sweep_report.py`: Report-Generator für Sweep-Ergebnisse
   - Status: Funktioniert, aber nicht systematisch integriert

3. **Experiment-Tracking:**
   - `src/assembled_core/qa/experiment_tracking.py`: ExperimentTracker (run.json, metrics.csv)
   - `scripts/summarize_backtest_experiments.py`: Experiment-Zusammenfassung
   - Status: Funktioniert, aber nicht in Batch-Runner integriert

4. **Walk-Forward-Analyse:**
   - `src/assembled_core/qa/walk_forward.py`: Walk-Forward-Splits und Backtest-Funktion
   - Status: Funktioniert, aber nicht in Batch-Runner integriert

5. **Profiling-Integration:**
   - `scripts/profile_jobs.py`: Job-Profiling mit cProfile/pyinstrument
   - `scripts/benchmark_backtest.py`: Benchmark-Harness für Speedup-Messung
   - Status: Funktioniert, aber nicht in Batch-Runner integriert

### Fehlende Bausteine

1. **Echte Parallelisierung:**
   - `batch_backtest.py` hat `max_workers` Parameter, aber nutzt `ProcessPoolExecutor` nicht
   - Keine Ressourcen-Limitierung (CPU/RAM)
   - Keine Retry-Logik bei fehlgeschlagenen Runs

2. **Systematische Parameter-Grid-Search:**
   - Kein Grid-Search-Generator (nur ad-hoc PowerShell-Sweeps)
   - Keine automatische Best-Parameter-Extraktion
   - Keine Sensitivitäts-Analyse

3. **Experiment-Runner mit Metriken-Vergleich:**
   - Keine automatische Metriken-Aggregation über Runs
   - Keine Ranking-Funktion (beste Sharpe, niedrigste Drawdown, etc.)
   - Keine Experiment-Vergleichs-Dashboard

4. **Integration von Experiment-Tracking:**
   - Batch-Runner nutzt ExperimentTracker nicht
   - Keine automatische run.json/metrics.csv für jeden Batch-Run
   - Keine Tag-basierte Filterung in Batch-Configs

5. **Hyperparameter-Optimierung:**
   - Keine Integration von Optuna/Bayesian-Optimierung
   - Keine automatische Suche nach optimalen Parametern

6. **Distributed Execution:**
   - Keine Ray/Dask-Integration (out of scope für P4, aber erwähnenswert)

7. **Experiment-Vergleichs-Dashboard:**
   - Keine visuelle Vergleichs-Ansicht für mehrere Experimente
   - Keine automatische Best-Parameter-Hervorhebung

---

## 2. P4 Task-Liste (10-15 Bullets)

### Phase 4.1: Parallelisierung aktivieren

- [ ] **Task 4.1.1:** Implementiere `ProcessPoolExecutor` in `batch_backtest.py` für echte Parallelisierung
  - Nutze `concurrent.futures.ProcessPoolExecutor` statt serieller Loop
  - Fehler-Handling: Einzelne Runs dürfen fehlschlagen ohne Batch-Abbruch
  - Ressourcen-Limitierung: `max_workers` default = `min(os.cpu_count() or 2, 4)`
  - Logging: Pro Worker separater Log-Stream (optional)

- [ ] **Task 4.1.2:** Retry-Logik für fehlgeschlagene Runs
  - Konfigurierbare Retry-Anzahl (default: 0, d.h. kein Retry)
  - Exponential Backoff zwischen Retries
  - Logging: Retry-Versuche klar dokumentieren

- [ ] **Task 4.1.3:** Ressourcen-Monitoring (optional)
  - CPU/RAM-Usage pro Worker tracken
  - Warnung wenn Ressourcen knapp werden
  - Optional: Auto-Scaling (Worker reduzieren bei hoher Last)

### Phase 4.2: Parameter-Grid-Search

- [ ] **Task 4.2.1:** Grid-Search-Generator in Batch-Config
  - YAML-Syntax für Parameter-Grids: `param_grid: {max_gross_exposure: [0.6, 0.8, 1.0, 1.2], commission_bps: [0.0, 0.5, 1.0]}`
  - Automatische Generierung aller Kombinationen
  - Run-IDs automatisch generieren: `run_{param1}_{param2}_...`

- [ ] **Task 4.2.2:** Best-Parameter-Extraktion
  - Automatische Identifikation bester Runs nach Metrik (Sharpe, Final PF, etc.)
  - Output: `best_params.json` mit Top-N Konfigurationen
  - Integration in Batch-Summary (Markdown-Tabelle mit Top-Runs)

- [ ] **Task 4.2.3:** Sensitivitäts-Analyse
  - Pro Parameter: Metriken-Aggregation (mean, std, min, max)
  - Output: `sensitivity_analysis.csv` mit Param-Sensitivität
  - Optional: Visualisierung (wenn matplotlib verfügbar)

### Phase 4.3: Experiment-Tracking-Integration

- [ ] **Task 4.3.1:** ExperimentTracker in Batch-Runner integrieren
  - Jeder Batch-Run erhält automatisch `run.json` und `metrics.csv`
  - Tags aus Batch-Config übernehmen (z.B. `["batch", "grid_search", "ai_tech"]`)
  - Run-ID Format: `batch_{batch_name}_{run_id}_{timestamp}`

- [ ] **Task 4.3.2:** Metriken-Aggregation über Runs
  - Automatische Extraktion von Metriken aus Backtest-Reports
  - Speicherung in `metrics.csv` (Sharpe, Final PF, Max Drawdown, Trades, etc.)
  - Integration in Batch-Summary (CSV + Markdown)

- [ ] **Task 4.3.3:** Tag-basierte Filterung in Batch-Configs
  - Optional: `filter_tags` in Batch-Config für Experiment-Vergleich
  - Nutzung von `ExperimentTracker.list_runs(tags=...)` für Vergleichs-Runs

### Phase 4.4: Experiment-Runner mit Metriken-Vergleich

- [ ] **Task 4.4.1:** Ranking-Funktion für Runs
  - Sortierung nach Metrik (Sharpe, Final PF, Max Drawdown, etc.)
  - Multi-Metriken-Ranking (gewichteter Score)
  - Output: `rankings.csv` mit Top-N Runs

- [ ] **Task 4.4.2:** Experiment-Vergleichs-Dashboard (Text-basiert)
  - Markdown-Report mit Vergleichstabelle aller Runs
  - Hervorhebung bester Runs (Top-3 pro Metrik)
  - Optional: HTML-Dashboard (wenn jinja2 verfügbar)

- [ ] **Task 4.4.3:** Automatische Best-Parameter-Hervorhebung
  - In Batch-Summary: Top-Runs visuell hervorheben (Markdown-Format)
  - Output: `best_params_summary.md` mit Top-N Konfigurationen und Metriken

### Phase 4.5: Walk-Forward-Integration (optional)

- [ ] **Task 4.5.1:** Walk-Forward-Batch-Config
  - YAML-Syntax für Walk-Forward-Splits in Batch-Config
  - Automatische Generierung von Runs pro Split
  - Integration mit `src/assembled_core/qa/walk_forward.py`

- [ ] **Task 4.5.2:** Walk-Forward-Metriken-Aggregation
  - Pro Split: Metriken extrahieren
  - Gesamt-Metriken: Mean/Std über alle Splits
  - Output: `walk_forward_summary.csv` mit Split-Ergebnissen

### Phase 4.6: Profiling-Integration

- [ ] **Task 4.6.1:** Optionales Profiling pro Batch-Run
  - Flag `--with-profiling` aktiviert cProfile pro Run
  - Output: `runs/{run_id}/profile/` mit `.prof` und Stats
  - Integration in Batch-Summary (Runtime pro Run)

- [ ] **Task 4.6.2:** Batch-Level-Profiling
  - Gesamt-Runtime des Batches
  - Pro-Step-Timing (Config-Load, Run-Execution, Summary-Generation)
  - Output: `batch_timings.json`

---

## 3. Priorisierung & Empfohlene Reihenfolge

**Phase 4.1 (Parallelisierung):** Hohe Priorität
- Ermöglicht Speedup auf Multi-Core-Systemen
- Basis für alle weiteren Optimierungen

**Phase 4.2 (Grid-Search):** Hohe Priorität
- Systematische Parameter-Optimierung
- Wichtig für Research-Workflows

**Phase 4.3 (Experiment-Tracking):** Mittlere Priorität
- Verbessert Reproduzierbarkeit und Vergleichbarkeit
- Basis für Dashboard (4.4)

**Phase 4.4 (Metriken-Vergleich):** Mittlere Priorität
- Wichtig für Research, aber kann manuell gemacht werden
- Automatisiert mühsame Vergleichs-Arbeit

**Phase 4.5 (Walk-Forward):** Niedrige Priorität
- Spezialisierter Use-Case
- Kann manuell mit bestehenden Tools gemacht werden

**Phase 4.6 (Profiling):** Niedrige Priorität
- Nice-to-have für Performance-Analyse
- P1/P3 decken bereits Profiling ab

---

## 4. Success Criteria

**Funktional:**
- Batch-Runner kann Backtests parallel ausführen (4.1)
- Grid-Search kann systematisch Parameter-Variationen generieren (4.2)
- Experiment-Tracking ist in Batch-Runner integriert (4.3)
- Metriken-Vergleich und Ranking funktionieren (4.4)

**Nicht-funktional:**
- Parallelisierung reduziert Gesamt-Runtime auf Multi-Core-Systemen
- Keine Regressionen in bestehenden Batch-Runner-Funktionen
- Kompatibel mit bestehenden CLI- und Profiling-Workflows

**Testing:**
- Unit-Tests für Grid-Search-Generator
- Integration-Tests für Parallelisierung (2-3 Runs parallel)
- Regression-Tests: Bestehende Batch-Tests bleiben grün

---

## 5. Out of Scope (für P4)

- **Distributed Execution (Ray/Dask):** Zu komplex für P4, kann später hinzugefügt werden
- **Hyperparameter-Optimierung (Optuna):** Spezialisierter Use-Case, kann als separate Phase implementiert werden
- **Live-Trading-Integration:** SAFE-Bridge bleibt unverändert
- **Änderung der Backtest-Logik:** P3 deckt Engine-Optimierung ab

---

## 6. References

- [Batch Backtest Design (P4)](BATCH_BACKTEST_P4_DESIGN.md) - Original P4 Design
- [Batch Backtests Workflow](WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md) - Workflow-Dokumentation
- [Performance Profiling P1](PERFORMANCE_PROFILING_P1_DESIGN.md) - Profiling-Infrastruktur
- [Backtest Engine Optimization P3](BACKTEST_ENGINE_OPTIMIZATION_P3.md) - Engine-Optimierungen

