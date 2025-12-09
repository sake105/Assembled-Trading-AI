# Backend Roadmap - Assembled Trading AI

## Übersicht

Dieses Dokument beschreibt die geplante Entwicklungs-Roadmap für das Backend von Assembled Trading AI. Die Roadmap ist in Phasen und Sprints unterteilt.

**Status:** Phase 1 (Basis-Infrastruktur) - In Bearbeitung

---

## Phase 1: Basis-Infrastruktur

**Ziel:** Solide Basis für Trading-Pipeline und Backend-API schaffen.

### Sprint 1: Repo-Bereinigung & Packaging

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] Große Artefakte aus Git entfernen (ZIPs, Logs)
- [x] `.gitignore` aktualisieren
- [x] `pyproject.toml` erstellen
- [x] Package-Skelett (`src/assembled_core/`) anlegen
- [x] Architektur-Dokumentation erstellen

**Ergebnisse:**
- Sauberes, code-zentriertes Repo
- Installierbares Python-Package
- Vollständige Dokumentation

---

### Sprint 2: Package-Struktur & Module-Skelett

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] Package-Verzeichnisse anlegen (data/, features/, signals/, portfolio/, execution/, reports/)
- [x] Platzhalter-Module mit Docstrings erstellen
- [x] Integration mit bestehenden `pipeline.*`-Modulen dokumentieren

**Ergebnisse:**
- Klare Package-Struktur
- Platzhalter für zukünftige Erweiterungen
- Bestehende Funktionalität bleibt erhalten

---

## Phase 2: Core-Funktionalität (Geplant)

**Ziel:** Erweiterte Trading-Funktionalität implementieren.

### Sprint 3: Data Ingestion (Geplant)

**Ziele:**
- Multi-Source-Support (Yahoo, Alpha Vantage, lokale Dateien)
- Datenvalidierung und Normalisierung
- Automatische Fallback-Logik

**Module:**
- `data/prices_ingest.py` - Implementierung
- `data/validation.py` - Datenqualitäts-Checks

---

### Sprint 4: Technical Analysis Features (Geplant)

**Ziele:**
- Erweiterte TA-Indikatoren (RSI, MACD, Bollinger Bands, etc.)
- Feature-Engineering-Pipeline
- Feature-Normalisierung

**Module:**
- `features/ta_features.py` - Implementierung
- `features/normalization.py` - Feature-Skalierung

---

### Sprint 5: Signal-Generation-Framework (Geplant)

**Ziele:**
- Erweiterte Signal-Regeln (Mean-Reversion, Breakout, etc.)
- Signal-Kombination (Multi-Strategy)
- Signal-Filterung

**Module:**
- `signals/rules_trend.py` - Implementierung
- `signals/rules_mean_reversion.py` - Neue Regel-Typen
- `signals/combiner.py` - Signal-Kombination

---

### Sprint 6: Portfolio-Management (Geplant)

**Ziele:**
- Position-Sizing-Strategien (Kelly, Risk Parity, etc.)
- Portfolio-Optimierung
- Rebalancing-Logik

**Module:**
- `portfolio/position_sizing.py` - Implementierung
- `portfolio/optimizer.py` - Portfolio-Optimierung

---

### Sprint 7: Order-Execution (Geplant)

**Ziele:**
- Erweiterte Order-Typen (Limit, Stop-Loss)
- Execution-Cost-Modellierung
- Order-Routing-Simulation

**Module:**
- `execution/order_generation.py` - Implementierung
- `execution/execution_engine.py` - Execution-Logik

---

## Phase 3: Erweiterte Features (Geplant)

**Ziel:** Zusätzliche Datenquellen und Features integrieren.

### Sprint 8: Fundamentals & Insider (Geplant)

**Ziele:**
- SEC-Filings-Integration
- Insider-Transaktions-Daten
- Fundamental-Analyse-Features

**Module:**
- `data/fundamentals.py` - Fundamentals-Ingestion
- `data/insider.py` - Insider-Daten

---

### Sprint 9: Congress Trading (Geplant)

**Ziele:**
- Congress-Trading-Daten-Integration
- Politiker-Portfolio-Tracking
- Signal-Generierung basierend auf Congress-Aktivität

**Module:**
- `data/congress.py` - Congress-Daten
- `signals/rules_congress.py` - Congress-basierte Signale

---

### Sprint 10: Shipping & News (Geplant)

**Ziele:**
- Shipping-Daten-Integration
- News-Feed-Integration
- Sentiment-Analyse

**Module:**
- `data/shipping.py` - Shipping-Daten
- `data/news.py` - News-Ingestion
- `features/sentiment.py` - Sentiment-Analyse

---

## Phase 4: Backtests & QA-Grundlagen

**Ziel:** Robuste Backtest-Infrastruktur mit umfassender Qualitätssicherung.

**Status:** ✅ Abgeschlossen

### Übersicht

Phase 4 implementiert eine integrierte QA- und Backtest-Infrastruktur, die es ermöglicht, Trading-Strategien systematisch zu testen, zu evaluieren und zu validieren. Die Module arbeiten zusammen, um:

1. **Flexible Backtests** durchzuführen (Backtest-Engine)
2. **Performance-Metriken** zu berechnen (QA-Metriken)
3. **Qualitäts-Gates** zu evaluieren (QA-Gates)
4. **Walk-Forward-Analysen** durchzuführen (Walk-Forward)
5. **QA-Reports** zu generieren (QA-Reports)

### Sprint 14: Backtest-Engine & QA-Metriken

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] Portfolio-Level-Backtest-Engine (`qa.backtest_engine`)
- [x] Zentrale Metriken-Berechnung (`qa.metrics`)
- [x] PerformanceMetrics-Dataclass
- [x] Integration in EOD-Pipeline

**Ergebnisse:**
- Flexible Backtest-Engine mit custom Signal- und Sizing-Funktionen
- Umfassende Metriken-Berechnung (Returns, Risk-Adjusted, Risk, Trade Metrics)
- Integration in `run_eod_pipeline.py` (Step 5b)

---

### Sprint 15: QA-Gates & Walk-Forward

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] QA-Gates-System (`qa.qa_gates`)
- [x] Walk-Forward-Analyse (`qa.walk_forward`)
- [x] Integration in EOD-Pipeline (QA-Gates)
- [x] Tests für alle QA-Module

**Ergebnisse:**
- Automatisierte Qualitäts-Gates (OK/WARNING/BLOCK)
- Walk-Forward-Analyse mit IS/OOS-Metriken
- Integration in `run_eod_pipeline.py` (Step 5c)
- Umfassende Test-Suite (`tests/test_qa_*.py`)

---

### Sprint 16: QA-Reports

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] QA-Report-Generierung (`reports.daily_qa_report`)
- [x] Markdown-Report-Format
- [x] Convenience-Funktion (`generate_qa_report_from_files`)
- [x] Tests für Report-Generierung

**Ergebnisse:**
- Automatische QA-Report-Generierung
- Markdown-Reports mit Metriken, QA-Gates, Equity-Curve-Link, Config-Info
- Integration in EOD-Pipeline (optional)

**Extended Roadmap:** For advanced factor evaluation and event study frameworks, see [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Phase A (TA/Price Factor Library) and Phase C (Factor Analysis & Event Study Engine).

---

## Phase 5: Production-Ready (Geplant)

**Ziel:** Backend für Produktion vorbereiten.

### Sprint 17: Performance & Skalierung (Geplant)

**Ziele:**
- Performance-Optimierung
- Caching-Strategien
- Parallele Verarbeitung

---

### Sprint 18: Monitoring & Alerting (Geplant)

**Ziele:**
- Health-Check-Dashboard
- Alerting-System
- Performance-Monitoring

---

### Sprint 19: Testing & QA (Geplant)

**Ziele:**
- Umfassende Test-Suite
- Integration-Tests
- End-to-End-Tests

---

## Phase 12: "God-Level" Research & Evolution

**Ziel:** Kontinuierlich lernendes System mit strukturiertem Research-Prozess.

**Status:** ✅ Sprint 12.1, 12.2 & 12.3 abgeschlossen (Review-fähig)

### Sprint 12.1: Research-Prozess & Roadmap

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] Bestandsaufnahme bestehender Research-Artefakte
- [x] Research-Roadmap erstellt (`docs/RESEARCH_ROADMAP.md`)
- [x] Research-Ordner-Struktur angelegt (`research/`)
- [x] Notebook-Templates für Experimente erstellt
- [x] Dokumentation in README integriert

**Ergebnisse:**
- Strukturierte Research-Roadmap mit Zielbild, aktuellem Stand, 3–6-Monats-Fokus und konkretem Backlog
- Research-Ordner mit Kategorien (trend/, meta/, altdata/, risk/)
- Klare Arbeitsweise für Research-Experimente (Hypothese → Setup → Run → Auswertung → Doku)

### Sprint 12.2: Experiment-Tracking

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] Experiment-Tracking-Modul erstellt (`qa/experiment_tracking.py`)
- [x] Settings erweitert (`experiments_dir`)
- [x] Integration in Backtest (`run_backtest_strategy.py`)
- [x] Integration in Meta-Model-Training (`cli.py train_meta_model`)
- [x] CLI-Argumente hinzugefügt (`--track-experiment`, `--experiment-name`, `--experiment-tags`)
- [x] Tests erstellt (`tests/test_experiment_tracking.py`)
- [x] Dokumentation aktualisiert

**Ergebnisse:**
- Leichtgewichtiges Experiment-Tracking ohne externe Services
- Runs werden in `experiments/{run_id}/` gespeichert
- Struktur: `run.json` (Metadaten), `metrics.csv` (Zeitreihen-Metriken), `artifacts/` (Dateien)
- Integration in Backtests und Meta-Model-Training
- Vollständige Test-Suite (10 Tests, alle grün)

### Sprint 12.3: Review-Vorbereitung

**Status:** ✅ Abgeschlossen

**Aufgaben:**
- [x] Known Issues & TODOs eingesammelt
- [x] KNOWN_ISSUES.md erstellt (funktionale Open Points, technische Schulden, Performance, Nice-to-Haves)
- [x] REVIEW_GUIDE_BACKEND.md erstellt (Review-Anleitung für externe Reviewer)
- [x] Issue-Template für Review-Feedback erstellt (`.github/ISSUE_TEMPLATE/review_feedback.md`)
- [x] README aktualisiert (Links zu Review Guide und Known Issues)

**Ergebnisse:**
- Strukturierte Übersicht aller bekannten offenen Punkte
- Klare Anleitung für externe Reviewer
- Issue-Template für strukturiertes Feedback
- Backend ist review-fähig

**Nächste Schritte:**
- Erste Research-Experimente durchführen (geplant)
- Externe Reviews einholen

---

## Aktueller Status

**Phase 1 - Sprint 1 & 2:** ✅ Abgeschlossen  
**Phase 4:** ✅ Abgeschlossen (Backend Core)  
**Phase 6:** ✅ Abgeschlossen (Event Features)  
**Phase 7:** ✅ Abgeschlossen (ML-Meta-Layer)  
**Phase 8:** ✅ Abgeschlossen (Risk Engine)  
**Phase 9:** ✅ Abgeschlossen (Model Governance)  
**Phase 10:** ✅ Abgeschlossen (Paper-Trading & OMS)  
**Phase 11:** ✅ Abgeschlossen (Packaging, CI, Logging)  
**Phase 12 - Sprint 12.1:** ✅ Abgeschlossen (Research-Roadmap)
**Phase 12 - Sprint 12.2:** ✅ Abgeschlossen (Experiment-Tracking)
**Phase 12 - Sprint 12.3:** ✅ Abgeschlossen (Review-Vorbereitung)

**Nächste Schritte:**
- Externe Reviews einholen
- Erste Research-Experimente durchführen (geplant)

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur
- [Backend Modules](BACKEND_MODULES.md) - Modulübersicht
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht
- [Research Roadmap](RESEARCH_ROADMAP.md) - Research-Prozess, Backlog, Fokus (Phase 12)
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Extended roadmap for advanced factor development (Phases A–E)
- [Review Guide](REVIEW_GUIDE_BACKEND.md) - Anleitung für externe Reviewer (Phase 12.3)

