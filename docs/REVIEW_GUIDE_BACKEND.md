# Backend Review Guide – Assembled Trading AI

**Letzte Aktualisierung:** 2025-01-15  
**Zielgruppe:** Externe Reviewer, neue Entwickler, Architektur-Interessierte

---

## 1. Ziel dieses Dokuments

Dieses Dokument soll externen Reviewern helfen, sich schnell im Backend von Assembled Trading AI zu orientieren und gezielt Feedback zu geben.

**Was soll ein externer Reviewer prüfen?**

- **Architektur & Modularität:** Ist die Code-Struktur klar und wartbar?
- **Tests & CI:** Sind Tests ausreichend und CI robust?
- **Logging & Observability:** Ist das System nachvollziehbar und debuggbar?
- **ML-Meta-Layer & Ensembling:** Ist der ML-Workflow verständlich und robust?
- **Research- und Experiment-Workflow:** Ist der Research-Prozess strukturiert und reproduzierbar?

**Was ist NICHT das Ziel?**

- Vollständige Code-Review aller Module (zu umfangreich)
- Performance-Benchmarks (dafür gibt es separate Tests)
- Security-Audit (dafür gibt es `docs/SECURITY_SECRETS.md`)

---

## 2. Empfohlene Reihenfolge

### Schritt 1: Übersicht & Kontext

1. **README.md** (Repository-Root)
   - Schnellstart, CLI-Übersicht, Status der Phasen
   - Verständnis: Was macht das System? Welche Phasen sind fertig?

2. **docs/BACKEND_ROADMAP.md**
   - Gesamt-Roadmap, abgeschlossene Phasen, geplante Sprints
   - Verständnis: Wo steht das Projekt? Was ist geplant?

### Schritt 2: Architektur

3. **docs/ARCHITECTURE_BACKEND.md**
   - High-Level-Komponenten, Datenfluss, Module-Struktur
   - Verständnis: Wie ist das System aufgebaut? Wie fließen Daten?

4. **docs/BACKEND_MODULES.md** (optional, detaillierter)
   - Detaillierte Modul-Beschreibungen
   - Verständnis: Was macht jedes Modul genau?

### Schritt 3: Phase-spezifische Dokumentation

5. **docs/PHASE7_META_LAYER.md**
   - ML-Meta-Layer: Labeling, Dataset-Builder, Meta-Modelle, Ensemble
   - Verständnis: Wie funktioniert der ML-Workflow?

6. **docs/PHASE8_RISK_ENGINE.md**
   - Risk-Engine: Portfolio Risk Metrics, Scenario Engine, Shipping Risk
   - Verständnis: Wie wird Risiko gemessen und gesteuert?

7. **docs/PHASE9_MODEL_GOVERNANCE.md**
   - Model Governance: Validation, Drift Detection, Model Inventory
   - Verständnis: Wie werden Modelle validiert und überwacht?

8. **docs/PHASE10_PAPER_OMS.md**
   - Paper-Trading & OMS-Light: Pre-Trade-Checks, Kill-Switch, Blotter
   - Verständnis: Wie funktioniert die Order-Verwaltung?

### Schritt 4: Research & Experimente

9. **docs/RESEARCH_ROADMAP.md**
   - Research-Prozess, Backlog, Fokus 3–6 Monate
   - Verständnis: Wie wird Research strukturiert?

10. **research/README.md**
    - Research-Workflow, Experiment-Struktur, Best Practices
    - Verständnis: Wie führt man ein Research-Experiment durch?

### Schritt 5: Known Issues

11. **KNOWN_ISSUES.md** (Repository-Root)
    - Offene Punkte, technische Schulden, Nice-to-Haves
    - Verständnis: Was ist noch offen? Was sollte verbessert werden?

---

## 3. Praktische Einstiegspunkte

### 3.1 Wichtige CLI-Kommandos

**Backtest ausführen:**
```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --strategy trend_baseline \
  --universe watchlist.txt \
  --generate-report
```

**EOD-Pipeline ausführen:**
```bash
python scripts/cli.py run_daily \
  --freq 1d \
  --start-capital 10000
```

**ML-Dataset bauen:**
```bash
python scripts/cli.py build_ml_dataset \
  --strategy trend_baseline \
  --freq 1d \
  --label-horizon-days 10 \
  --success-threshold 0.05
```

**Meta-Model trainieren:**
```bash
python scripts/cli.py train_meta_model \
  --dataset-path output/ml_datasets/trend_baseline_1d.parquet \
  --model-type gradient_boosting
```

**Backtest mit Experiment-Tracking:**
```bash
python scripts/cli.py run_backtest \
  --freq 1d \
  --track-experiment \
  --experiment-name "trend_ma20_50" \
  --experiment-tags "trend,baseline"
```

### 3.2 Wichtige Module

**Core-Module:**
- `src/assembled_core/data/`: Daten-Ingest (Preise, Events)
- `src/assembled_core/features/`: Feature-Engineering (TA, Insider, Shipping)
- `src/assembled_core/signals/`: Signal-Generierung (Trend, Event, Meta-Model, Ensemble)
- `src/assembled_core/portfolio/`: Position-Sizing
- `src/assembled_core/execution/`: Order-Generierung, Pre-Trade-Checks, Kill-Switch
- `src/assembled_core/qa/`: QA-Metriken, Backtest-Engine, Validation, Drift-Detection
- `src/assembled_core/api/`: FastAPI-Endpoints (Orders, Performance, QA, Monitoring, Paper-Trading, OMS)

**Meta-Layer:**
- `src/assembled_core/qa/labeling.py`: Trade-Labeling für ML
- `src/assembled_core/qa/dataset_builder.py`: ML-Dataset-Builder
- `src/assembled_core/signals/meta_model.py`: Meta-Model-Training & Prediction
- `src/assembled_core/signals/ensemble.py`: Ensemble-Layer (Filter/Scaling)

**Risk & QA:**
- `src/assembled_core/qa/backtest_engine.py`: Portfolio-Level-Backtest-Engine
- `src/assembled_core/qa/risk_metrics.py`: Portfolio Risk Metrics (VaR, CVaR, Volatility)
- `src/assembled_core/qa/scenario_engine.py`: Scenario-Engine (Stress-Tests)
- `src/assembled_core/qa/validation.py`: Model-Validation
- `src/assembled_core/qa/drift_detection.py`: Drift-Detection (Feature, Label, Performance)

**Experiment-Tracking:**
- `src/assembled_core/qa/experiment_tracking.py`: Experiment-Tracking (Runs, Metriken, Artefakte)

### 3.3 Test-Suite

**Phase-Tests ausführen:**
```bash
# Phase 4 (Backend Core)
pytest -m phase4

# Phase 6 (Event Features)
pytest -m phase6

# Phase 7 (ML-Meta-Layer)
pytest -m phase7

# Phase 8 (Risk Engine)
pytest -m phase8

# Phase 9 (Model Governance)
pytest -m phase9

# Phase 10 (Pre-Trade & Paper-Trading)
pytest -m phase10

# Phase 12 (Experiment-Tracking)
pytest -m phase12

# Alle relevanten Phasen
pytest -m "phase4 or phase6 or phase7 or phase8 or phase9 or phase10 or phase12" -W error
```

---

## 4. Checkliste für Reviews

### 4.1 Architektur & Modularität

- [ ] **Modulare Struktur:** Sind Module klar getrennt? Gibt es zirkuläre Abhängigkeiten?
- [ ] **Datenfluss:** Ist der Datenfluss von Daten → Features → Signals → Portfolio → Orders klar?
- [ ] **Konfiguration:** Ist die Konfiguration zentral (`settings.py`) und nachvollziehbar?
- [ ] **API-Design:** Sind API-Endpoints konsistent und gut dokumentiert?

**Relevante Dateien:**
- `src/assembled_core/` (Gesamtstruktur)
- `src/assembled_core/config/settings.py`
- `src/assembled_core/api/app.py`

### 4.2 Tests & CI

- [ ] **Test-Abdeckung:** Gibt es Tests für alle wichtigen Module?
- [ ] **Test-Qualität:** Sind Tests aussagekräftig (nicht nur Smoke-Tests)?
- [ ] **CI-Robustheit:** Läuft CI stabil? Gibt es flaky Tests?
- [ ] **Test-Organisation:** Sind Tests nach Phasen organisiert und klar markiert?

**Relevante Dateien:**
- `tests/test_*.py`
- `pytest.ini`
- `.github/workflows/backend-ci.yml`

### 4.3 Logging & Observability

- [ ] **Zentrales Logging:** Ist Logging zentral konfiguriert (`logging_config.py`)?
- [ ] **Run-IDs:** Werden Run-IDs konsistent verwendet?
- [ ] **Log-Level:** Sind Log-Level angemessen (nicht zu viel, nicht zu wenig)?
- [ ] **Strukturierte Logs:** Sind Logs strukturiert und nachvollziehbar?

**Relevante Dateien:**
- `src/assembled_core/logging_config.py`
- `scripts/cli.py` (Logging-Integration)
- `logs/` (Log-Dateien)

### 4.4 ML-Meta-Layer & Ensembling

- [ ] **Labeling:** Ist das Labeling-Schema klar und robust?
- [ ] **Dataset-Builder:** Ist der Dataset-Builder flexibel und wiederverwendbar?
- [ ] **Meta-Modelle:** Ist das Meta-Model-Training reproduzierbar?
- [ ] **Ensemble-Layer:** Ist der Ensemble-Layer klar dokumentiert und testbar?
- [ ] **Feature-Handling:** Werden Features konsistent benannt und gehandhabt?

**Relevante Dateien:**
- `src/assembled_core/qa/labeling.py`
- `src/assembled_core/qa/dataset_builder.py`
- `src/assembled_core/signals/meta_model.py`
- `src/assembled_core/signals/ensemble.py`
- `src/assembled_core/qa/backtest_engine.py` (Meta-Model-Integration)

### 4.5 Research- und Experiment-Workflow

- [ ] **Experiment-Tracking:** Ist das Experiment-Tracking einfach zu nutzen?
- [ ] **Reproduzierbarkeit:** Sind Experimente reproduzierbar (Config, Metriken, Artefakte)?
- [ ] **Research-Struktur:** Ist die Research-Struktur (`research/`) klar organisiert?
- [ ] **Dokumentation:** Ist der Research-Workflow gut dokumentiert?

**Relevante Dateien:**
- `src/assembled_core/qa/experiment_tracking.py`
- `research/README.md`
- `docs/RESEARCH_ROADMAP.md`
- `experiments/` (Experiment-Runs)

---

## 5. Konkrete Review-Fragen

### 5.1 API-Oberfläche

**Frage:** Gibt es Stellen, wo die API-Oberfläche unklar ist?

**Zu prüfen:**
- FastAPI-Endpoints (`src/assembled_core/api/routers/`)
- Pydantic-Models (`src/assembled_core/api/models.py`)
- API-Dokumentation (Swagger/OpenAPI unter `/docs`)

**Beispiele:**
- Sind Request/Response-Models klar?
- Gibt es inkonsistente Namenskonventionen?
- Fehlen wichtige Endpoints?

### 5.2 Risk-Engine-Architektur

**Frage:** Welche Teile der Risk-Engine würdest du anders schneiden?

**Zu prüfen:**
- `src/assembled_core/qa/risk_metrics.py`
- `src/assembled_core/qa/scenario_engine.py`
- `src/assembled_core/qa/shipping_risk.py`
- `src/assembled_core/execution/pre_trade_checks.py`

**Beispiele:**
- Ist die Trennung zwischen Portfolio-Risk und Pre-Trade-Checks klar?
- Sollte Shipping-Risk ein separates Modul sein?
- Gibt es Redundanzen zwischen Risk-Metriken und QA-Metriken?

### 5.3 Logging & Experiments

**Frage:** Sind Logging & Experiments für dich verständlich nutzbar?

**Zu prüfen:**
- `src/assembled_core/logging_config.py`
- `src/assembled_core/qa/experiment_tracking.py`
- CLI-Integration (z.B. `scripts/cli.py`)

**Beispiele:**
- Ist das Log-Format hilfreich?
- Ist Experiment-Tracking intuitiv zu nutzen?
- Fehlen wichtige Metadaten in Runs?

### 5.4 ML-Workflow

**Frage:** Ist der ML-Workflow (Labeling → Dataset → Training → Ensemble) klar und robust?

**Zu prüfen:**
- `src/assembled_core/qa/labeling.py`
- `src/assembled_core/qa/dataset_builder.py`
- `src/assembled_core/signals/meta_model.py`
- `src/assembled_core/signals/ensemble.py`

**Beispiele:**
- Ist das Labeling-Schema robust gegen Edge-Cases?
- Ist der Dataset-Builder flexibel genug?
- Gibt es Risiken für Data Leakage?

### 5.5 Code-Organisation

**Frage:** Gibt es Module, die zu groß oder zu klein sind? Gibt es klare Verantwortlichkeiten?

**Zu prüfen:**
- Gesamtstruktur unter `src/assembled_core/`
- Größe und Komplexität einzelner Module
- Abhängigkeiten zwischen Modulen

**Beispiele:**
- Sollte `backtest_engine.py` aufgeteilt werden?
- Gibt es Module mit zu vielen Verantwortlichkeiten?
- Fehlen Abstraktionsebenen?

---

## 6. Wo Feedback hin soll

### 6.1 Feedback-Struktur

Feedback sollte strukturiert abgegeben werden, z.B.:

1. **Bereich:** Architektur, Daten/Features, Signals/Portfolio, Meta-Layer, Risk/QA, Research/Experiments, Sonstiges
2. **Schweregrad:** Info, Nice-to-have, Wichtig, Kritisch
3. **Beschreibung:** Was ist dir aufgefallen? Was sollte verbessert werden?
4. **Referenzen:** Dateien/Module, Relevante Runs/Experimente (experiment_run_id, falls vorhanden)

### 6.2 Feedback-Kanäle

**Option 1: GitHub Issues**
- Verwende das Issue-Template: `.github/ISSUE_TEMPLATE/review_feedback.md`
- Erstelle ein Issue pro Feedback-Bereich (nicht alles in ein Issue)

**Option 2: Direkte Kommunikation**
- Für vertrauliche oder sensitive Themen
- Kontakt: [Bitte ergänzen]

**Option 3: Pull-Request-Review**
- Wenn du konkrete Code-Änderungen vorschlägst
- Erstelle einen Draft-PR mit deinen Vorschlägen

### 6.3 Priorisierung

Bitte priorisiere dein Feedback:
- **Kritisch:** Blockiert Produktivnutzung oder verursacht Datenfehler
- **Wichtig:** Sollte vor nächstem Release behoben werden
- **Nice-to-have:** Verbesserung, aber kein Blocker
- **Info:** Anmerkung, keine Aktion erforderlich

---

## 7. Nützliche Ressourcen

### 7.1 Dokumentation

- **Architektur:** `docs/ARCHITECTURE_BACKEND.md`
- **Module:** `docs/BACKEND_MODULES.md`
- **CLI-Referenz:** `docs/CLI_REFERENCE.md`
- **Testing:** `docs/TESTING_COMMANDS.md`
- **Known Issues:** `KNOWN_ISSUES.md`

### 7.2 Code-Beispiele

- **Backtest:** `scripts/run_backtest_strategy.py`
- **EOD-Pipeline:** `scripts/run_eod_pipeline.py`
- **Meta-Model-Training:** `scripts/cli.py` (Subcommand `train_meta_model`)
- **Experiment-Tracking:** `src/assembled_core/qa/experiment_tracking.py`

### 7.3 Tests als Beispiele

- **QA-Tests:** `tests/test_qa_*.py`
- **Signal-Tests:** `tests/test_signals_*.py`
- **API-Tests:** `tests/test_api_*.py`
- **Experiment-Tracking:** `tests/test_experiment_tracking.py`

---

## 8. Häufige Fragen (FAQ)

### 8.1 Wie starte ich einen Backtest?

```bash
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt
```

Siehe auch: `docs/CLI_REFERENCE.md`

### 8.2 Wo werden Experiment-Runs gespeichert?

In `experiments/` (konfigurierbar via `settings.experiments_dir`). Jeder Run hat einen eigenen Ordner mit `run.json`, `metrics.csv` und `artifacts/`.

### 8.3 Wie trainiere ich ein Meta-Model?

```bash
python scripts/cli.py train_meta_model \
  --dataset-path output/ml_datasets/trend_baseline_1d.parquet
```

Siehe auch: `docs/PHASE7_META_LAYER.md`

### 8.4 Wie nutze ich Experiment-Tracking?

Füge `--track-experiment --experiment-name "..." --experiment-tags "tag1,tag2"` zu Backtest- oder Meta-Model-Commands hinzu.

Siehe auch: `docs/RESEARCH_ROADMAP.md` (Sektion 5.3)

### 8.5 Wo finde ich bekannte Issues?

In `KNOWN_ISSUES.md` (Repository-Root).

---

**Vielen Dank für dein Feedback!** Jede Anmerkung hilft, das System zu verbessern.

