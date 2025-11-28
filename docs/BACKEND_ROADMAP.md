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

## Phase 4: Production-Ready (Geplant)

**Ziel:** Backend für Produktion vorbereiten.

### Sprint 11: Performance & Skalierung (Geplant)

**Ziele:**
- Performance-Optimierung
- Caching-Strategien
- Parallele Verarbeitung

---

### Sprint 12: Monitoring & Alerting (Geplant)

**Ziele:**
- Health-Check-Dashboard
- Alerting-System
- Performance-Monitoring

---

### Sprint 13: Testing & QA (Geplant)

**Ziele:**
- Umfassende Test-Suite
- Integration-Tests
- End-to-End-Tests

---

## Aktueller Status

**Phase 1 - Sprint 1 & 2:** ✅ Abgeschlossen

**Nächste Schritte:**
- Phase 2 - Sprint 3: Data Ingestion
- Phase 2 - Sprint 4: Technical Analysis Features

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur
- [Backend Modules](BACKEND_MODULES.md) - Modulübersicht
- [Data Sources](DATA_SOURCES_BACKEND.md) - Datenquellen-Übersicht

