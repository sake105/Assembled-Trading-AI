# Backend Overview - Assembled Trading AI

## Projekt-Beschreibung

**Assembled Trading AI** ist ein file-based Trading-Pipeline-System mit einem read-only FastAPI-Backend. Das System verarbeitet Marktdaten, generiert Trading-Signale, simuliert Backtests und Portfolio-Performance, und stellt die Ergebnisse über eine REST-API bereit.

**Kernprinzipien:**
- **Single Source of Truth:** Produktionscode liegt unter `src/assembled_core/`
- **File-based:** Keine Datenbank, alle Daten in CSV/Parquet-Dateien
- **SAFE-Bridge:** Keine Live-Trading-Anbindung, nur Simulation via `orders_*.csv`
- **Offline-first:** Lokale Daten bevorzugt, Netz-Calls nur in Pull-Skripten

---

## Produktionscode: `src/assembled_core/`

**WICHTIG:** Alle produktiven Backend-Module befinden sich in `src/assembled_core/`.

**Hauptmodule:**
- `pipeline/` - Core Trading-Pipeline (I/O, Signale, Orders, Backtest, Portfolio)
- `api/` - FastAPI Backend (App, Models, Routers)
- `qa/` - QA/Health-Checks
- `config.py` - Zentrale Konfiguration (OUTPUT_DIR, SUPPORTED_FREQS)
- `costs.py` - Cost-Model-Konfiguration
- `ema_config.py` - EMA-Parameter-Konfiguration

**Zukünftige Module (Skelett vorhanden):**
- `data/` - Data Ingestion (Multi-Source)
- `features/` - Technical Analysis Features
- `signals/` - Signal-Generation-Framework
- `portfolio/` - Portfolio-Management
- `execution/` - Order-Execution-Logik
- `reports/` - Report-Generierung

**Bei Code-Änderungen:**
- **Immer** `src/assembled_core/` als Ziel verwenden
- Bestehende Architektur-Dokumentation respektieren
- Keine Breaking Changes ohne explizite Anweisung

---

## Architektur-Dokumentation

**Diese Dokumente sind verbindlich und müssen respektiert werden:**

1. **[ARCHITECTURE_BACKEND.md](../../docs/ARCHITECTURE_BACKEND.md)**
   - Gesamtarchitektur und Datenfluss
   - Module-Struktur
   - EOD-Pipeline-Übersicht
   - FastAPI-Backend
   - **→ Einstiegs-Dokument für Gesamtüberblick**

2. **[BACKEND_MODULES.md](../../docs/BACKEND_MODULES.md)**
   - Detaillierte Übersicht aller Module in `src/assembled_core/`
   - Funktionen, Abhängigkeiten, Verwendungszwecke
   - **→ Referenz für alle verfügbaren Module**

3. **[BACKEND_ROADMAP.md](../../docs/BACKEND_ROADMAP.md)**
   - Entwicklungs-Roadmap (Phasen & Sprints)
   - Aktueller Status und geplante Erweiterungen
   - **→ Kontext für zukünftige Entwicklungen**

4. **[DATA_SOURCES_BACKEND.md](../../docs/DATA_SOURCES_BACKEND.md)**
   - Aktuelle und geplante Datenquellen
   - Datenformate und Konfiguration
   - **→ Referenz für Daten-Ingestion**

5. **[backend_core.md](../../docs/backend_core.md)**
   - Konfiguration & Testing
   - Test-Suite-Übersicht

6. **[backend_api.md](../../docs/backend_api.md)**
   - FastAPI-Endpoints-Dokumentation
   - API-Verwendung und Beispiele

7. **[eod_pipeline.md](../../docs/eod_pipeline.md)**
   - EOD-Pipeline-Orchestrierung
   - Run-Manifest-Schema

**Regel:** Bevor Code-Änderungen vorgenommen werden, sollten die relevanten Architektur-Dokumente gelesen werden.

---

## Scripts vs. Core

**Scripts (`scripts/`):**
- CLI-Wrapper für Pipeline-Schritte
- Data-Ingestion-Skripte (Pull-Skripte)
- Orchestrierungs-Skripte
- **Dürfen** `src/assembled_core/` importieren, aber enthalten keine Kernlogik

**Core (`src/assembled_core/`):**
- Alle produktiven Backend-Module
- Pure Functions (möglichst ohne Seiteneffekte)
- Testbare Module
- **Single Source of Truth** für Backend-Logik

**Regel:** Neue Funktionalität gehört in `src/assembled_core/`, nicht in `scripts/`.

---

## Datenfluss

```
Data Ingestion (scripts/live/pull_*.py)
  → data/raw/1min/*.parquet

Resampling (scripts/run_all_sprint10.ps1)
  → output/aggregates/5min.parquet

Pipeline (src/assembled_core/pipeline/)
  → output/orders_{freq}.csv
  → output/equity_curve_{freq}.csv
  → output/portfolio_equity_{freq}.csv
  → output/performance_report_{freq}.md
  → output/portfolio_report_{freq}.md

FastAPI (src/assembled_core/api/)
  → Liest aus output/* Dateien
```

**Regel:** Pipeline-Module schreiben in `output/`, API liest aus `output/`.

---

## Verwendung in Cursor

**Diese Regel referenzieren:**
```
@01-backend-overview
```

**Wann verwenden:**
- Bei Fragen zur Projekt-Struktur
- Bei Unsicherheit, wo Code-Änderungen gemacht werden sollen
- Bei Bedarf nach Architektur-Überblick

**Weiterführende Regeln:**
- `@02-backend-guidelines` - Coding-Guidelines und Best Practices

