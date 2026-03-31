# Backend Overview - Assembled Trading AI

## Projektbeschreibung

**Assembled Trading AI** ist ein modulares Python-Trading-Backend.
Es ist kein Skeleton und keine frühe Rohfassung, sondern ein umfangreich implementiertes System.

**Kernprinzipien:**
- **Single Source of Truth:** Produktionscode liegt unter `src/assembled_core/`
- **File-based:** Keine Datenbank — alle persistenten Daten in CSV/Parquet-Dateien
- **Offline-first:** Lokale Daten bevorzugt, Netzwerk-Calls nur in expliziten Pull-Scripts
- **Read-only API:** FastAPI-Schicht liest aus `output/`-Dateien, schreibt nicht

---

## Modulstruktur: `src/assembled_core/`

Alle produktiven Backend-Module liegen in `src/assembled_core/`. Stand: 22 Kernmodule.

### Voll implementierte Module

| Modul | Inhalt (repräsentativ) |
|-------|------------------------|
| `data/` | `prices_ingest.py`, `factor_store.py`, `security_master.py`, `universe.py`, `altdata/`, `news/`, `shipping/` |
| `features/` | `ta_features.py`, `ta_factors_core.py`, `event_features.py`, `congress_features.py`, `insider_features.py`, `news_features.py`, `shipping_features.py`, `factor_store_integration.py` |
| `signals/` | `rules_trend.py`, `meta_model.py`, `ensemble.py`, `multifactor_signal.py`, `signal_api.py`, `rules_event_insider_shipping.py` |
| `execution/` | `order_generation.py`, `kill_switch.py`, `pre_trade_checks.py`, `transaction_costs.py`, `fill_model.py`, `paper_trading_engine.py`, `risk_controls.py` |
| `portfolio/` | `position_sizing.py` (lean, funktional) |
| `pipeline/` | `orchestrator.py`, `trading_cycle.py`, `backtest.py`, `backtest_legacy.py`, `orders.py`, `portfolio.py`, `io.py` |
| `qa/` | `backtest_engine.py`, `metrics.py`, `qa_gates.py`, `risk_metrics.py`, `walk_forward.py`, `drift_detection.py`, `leakage_tests/` |
| `accounting/` | `broker_snapshot_importer.py`, `evidence_pack.py`, `ledger.py`, `reconciliation.py`, `position_engine.py` |
| `risk/` | `state_machine.py`, `regime_analysis.py`, `exposure_engine.py`, `profit_lock.py`, `georisk_overlay.py`, `turnover_budget.py` |
| `ops/` | `intel_orchestrator.py`, `paper_runner.py`, `paper_ledger.py`, `health_check.py`, `reconcile.py`, `compare.py` |
| `events/disclosures/` | `fetch_edgar.py`, `fetch_house_ptr.py`, `pipeline.py`, `normalize.py`, `models.py` |
| `events/news/` | `fetch_rss.py`, `fetch_gdelt.py`, `pipeline.py`, `tfidf.py`, `clustering.py` |
| `api/` | `app.py`, `models.py`, Routers: orders, performance, risk, signals, portfolio, qa, monitoring, paper_trading, oms |
| `config/` | `settings.py`, `models.py`, `factor_bundles.py` (Pydantic-basiert) |
| `paper/` | `paper_track.py`, `georisk_gate.py`, `strategy_adapters.py`, `ranking_hysteresis.py`, `rebalance_filter.py`, `intel_runner.py` |
| `strategies/`, `experiments/`, `reports/`, `intel/`, `utils/` | Implementiert |

### Teilweise implementiert / bekannte Stubs

| Modul | Status |
|-------|--------|
| `ml/` | Vorhanden (`factor_models.py`, `explainability.py`), aber Trainings-Pfad in `__init__.py` wirft `NotImplementedError` |
| `api/routers/monitoring.py` | Drift-Analyse-Endpoints geben teilweise Dummy-Daten zurück |

---

## Primäre Entry Points

**Installierte CLI-Befehle** (nach `pip install -e ".[dev]"`):

| CLI-Befehl | Script | Funktion |
|------------|--------|----------|
| `assembled-cli` | `scripts/cli.py` | Unified CLI — Dispatcher für alle Hauptoperationen |
| `assembled-run-daily` | `scripts/run_eod_pipeline.py` | Täglicher EOD-Pipeline-Run |
| `assembled-run-backtest` | `scripts/run_backtest_strategy.py` | Strategie-Backtest |

**Weitere direkt aufrufbare Scripts:**
- `scripts/run_api.py` — FastAPI-Server (0.0.0.0:8000)
- `scripts/batch_backtest.py` — Batch-Backtest via YAML-Config

**Legacy-Scripts (nicht mehr primär):**
`scripts/sprint9_execute.py`, `scripts/sprint9_backtest.py`, `scripts/sprint10_portfolio.py`
sind im Repo vorhanden, aber in ihren Dateien als `# LEGACY` markiert und deprecated.
Für neue Entwicklung nicht verwenden.

---

## Scripts vs. Core

**Scripts (`scripts/`):**
- CLI-Wrapper für Pipeline-Schritte
- Data-Pull-Scripts (`download_*.py`, `fetch_*.py`)
- Diagnose- und Report-Scripts
- Dürfen `src/assembled_core/` importieren, enthalten aber keine Kernlogik

**Core (`src/assembled_core/`):**
- Alle produktiven Backend-Module
- Single Source of Truth für Backend-Logik
- Neue Funktionalität gehört hierher, nicht in `scripts/`

---

## Datenfluss (verifizierbarer Hauptpfad)

```
data/raw/ + data/sample/
  → data/prices_ingest.py        (OHLCV laden, normalisieren)
  → features/ta_features.py      (TA-Features berechnen)
  → data/factor_store.py         (PIT-sicherer Feature-Cache)
  → signals/rules_trend.py       (Trendsignale)
  → execution/order_generation.py (Zielpositionen → Orders)
  → execution/pre_trade_checks.py (Risk-Filter, Kill-Switch)
  → pipeline/portfolio.py        (Cost-aware Simulation)
  → qa/metrics.py + qa_gates.py  (Performance-Metriken, QA)
  → accounting/evidence_pack.py  (Artefakte, Reports)

output/ ← Pipeline-Outputs (Parquet, CSV, Markdown)
api/    ← liest aus output/, schreibt nicht
```

---

## Architektur-Dokumentation

Die folgenden Dokumente sind vorhanden und referenzierbar:

| Dokument | Inhalt |
|----------|--------|
| `docs/ARCHITECTURE_BACKEND.md` | Gesamtarchitektur, Datenfluss, Modul-Struktur |
| `docs/BACKEND_MODULES.md` | Detaillierte Übersicht aller Module |
| `docs/BACKEND_ROADMAP.md` | Entwicklungsstand und Roadmap |
| `docs/DATA_SOURCES_BACKEND.md` | Datenquellen, Formate, Konfiguration |
| `docs/backend_core.md` | Konfiguration & Testing |
| `docs/backend_api.md` | FastAPI-Endpoints |
| `docs/eod_pipeline.md` | EOD-Pipeline-Orchestrierung |
| `docs/CLI_REFERENCE.md` | CLI-Befehlsreferenz |
| `docs/cursor/CONTEXT_PACK.md` | Projektüberblick & Glossar (Cursor-spezifisch) |

---

## Verwendung in Cursor

**Diese Regel referenzieren:**
```
@01-backend-overview
```

**Wann verwenden:**
- Bei Fragen zur Projektstruktur
- Bei Unsicherheit, wo Codeänderungen hingehören
- Als Ausgangspunkt für Architektur-Überblick

**Weiterführende Regeln:**
- `@02-backend-guidelines` — Coding-Guidelines und Best Practices
