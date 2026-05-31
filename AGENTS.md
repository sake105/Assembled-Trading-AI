# AGENTS.md

## Cursor Cloud — Arbeitskontext für automatisierte Agents

Diese Datei beschreibt den aktuellen Repo-Stand für Cursor Cloud Agents.
Sie ist **keine README** und kein Marketingtext — sie ist eine operative Kontextdatei.

**Grundregel:** Kein Agent darf einen Implementierungsstand behaupten, ohne ihn im Repo verifiziert zu haben.
Plan, Spec, Roadmap und tatsächlich implementierter Code sind unterschiedliche Realitäten.

---

## Projektüberblick

**Assembled Trading AI** ist ein umfangreich implementiertes modulares Python-Backend für:

- Research und Backtests
- Paper-/Simulations-Trading
- Risk-Overlays und QA
- News-, Intel- und Disclosure-Integration (teilweise implementiert)
- API-Schicht (FastAPI, read-only)

Das Projekt ist **kein Skeleton und kein Proof-of-Concept**.
Es umfasst **31 Kernmodule** in `src/assembled_core/`, **91 Scripts**, **557 Testdateien** und **17 CI-Workflows**. (Stand 2026-04-27 — `scripts/regenerate_agents_stats.py` für aktuelle Zahlen.)

**Quellen der Wahrheit:**
- `src/assembled_core/` — Kernlogik (Single Source of Truth für Backend-Module)
- `scripts/` — Entry Points und Runner
- `tests/` — Testabdeckung (12 Phasen)
- `.github/workflows/` — CI-Konfiguration (Ubuntu + Windows)
- `docs/` — Architektur- und Betriebsdokumentation

---

## Umgebung

**Python-Anforderung:** `>=3.10` (pyproject.toml). CI läuft auf 3.10 und 3.11.

**Cursor Cloud (Linux-Container):**
```bash
source /workspace/.venv/bin/activate
pip install -e ".[dev]"
# Optional: scipy und ML-Extras
pip install -e ".[dev,scipy,ml]"
```

**Lokal (Windows):**
```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

---

## Entry Points

Die folgenden Scripts sind die aktuellen operativen Einstiege. Sie sind in `pyproject.toml` als installierte CLI-Befehle registriert oder direkt aufrufbar.

### Installierte CLI-Befehle (nach `pip install -e ".[dev]"`)

| Befehl | Script | Funktion |
|--------|--------|----------|
| `assembled-cli` | `scripts/cli.py` | Unified CLI — Dispatcher für alle Hauptoperationen |
| `assembled-run-daily` | `scripts/run_eod_pipeline.py` | Täglicher EOD-Pipeline-Run |
| `assembled-run-backtest` | `scripts/run_backtest_strategy.py` | Strategie-Backtest |

**Beispiele:**
```bash
python scripts/cli.py run_daily --freq 1d
python scripts/cli.py run_backtest --freq 1d --universe watchlist.txt
python scripts/cli.py run_phase4_tests
python scripts/cli.py info
python scripts/cli.py --version

python scripts/run_eod_pipeline.py --freq 1d --start-capital 10000
python scripts/run_backtest_strategy.py --freq 1d --start-capital 10000 --generate-report
python scripts/batch_backtest.py --config-file configs/batch_backtest_example_doc_schema.yaml
```

### Weitere direkt aufrufbare Scripts

| Script | Funktion |
|--------|----------|
| `scripts/run_api.py` | FastAPI-Server starten (0.0.0.0:8000) |
| `scripts/batch_backtest.py` | Batch-Backtest via YAML-Config |
| `scripts/run_paper_track.py` | Paper-Trading-Run |
| `scripts/run_ab_experiment.py` | A/B-Experiment |
| `scripts/run_walk_forward_analysis.py` | Walk-Forward-Analyse |
| `scripts/import_broker_snapshot.py` | Broker-Snapshot importieren |
| `scripts/export_evidence_pack.py` | Evidence-Pack exportieren |

### Legacy-Scripts

Frühere Sprint-Runner (`scripts/sprint9_execute.py`, `scripts/sprint9_backtest.py`,
`scripts/sprint10_portfolio.py`) existieren **nicht mehr** im Repo (verifiziert 2026-05-30:
`scripts/sprint*.py` → keine Treffer). `scripts/run_all_sprint10.ps1` existiert noch, referenziert
diese Scripts aber **nicht** (grep → 0 Treffer). Falls ältere Doku oder Chat-Verläufe diese Pfade
nennen: veraltet, nicht mehr gültig.

---

## Verzeichnisstruktur (Übersicht)

```
src/assembled_core/    — 22 Kernmodule (data, features, signals, execution,
                          portfolio, pipeline, qa, api, ops, accounting,
                          risk, reports, events, config, ml, ...)
scripts/               — ~50 Entry-Point- und Hilfs-Scripts
tests/                 — ~330 Testdateien, 12 Phasen (phase4–phase12)
.github/workflows/     — 9 CI-Workflows (Ubuntu + Windows)
configs/               — policy.yaml, Batch-Configs, News-/Disclosure-Configs
data/                  — data/raw/, data/sample/, data/factors/ (nicht in Git)
output/                — Pipeline-Outputs (nicht in Git)
docs/                  — Architektur- und Betriebsdokumentation
```

---

## Lint / Test / Build

```bash
# Lint (Ruff)
ruff check src tests scripts --exclude scripts/tools --exclude scripts/00_seed_demo_data.py
# Bekannte Baseline: 76 pre-existing findings (Stand Sprint 13), CI zeigt diese ebenfalls

# Format-Check (ruff-format)
ruff format --check src tests scripts --exclude scripts/tools --exclude scripts/00_seed_demo_data.py

# Tests (CI-äquivalent, ohne externe und advanced Tests)
pytest -m "not advanced" -q --maxfail=3 --tb=short

# Gezielt: bewährte Phase-4-Tests
pytest tests/test_cli.py tests/test_features_ta.py tests/test_qa_metrics.py \
       tests/test_qa_gates.py tests/test_execution_kill_switch.py \
       tests/test_qa_risk_metrics.py -v
```

**Bekannte Testrealität:**
- ~19 Testdateien schlagen bei der Collection fehl (unfertige Stubs in `src/assembled_core/data/`)
- Optionale Dependencies (`scipy`, `scikit-learn`) führen zu erwarteten Skips, nicht zu Fehlern
- Phase-4-Baseline (~117 Tests) läuft durch

---

## FastAPI-Server

```bash
python scripts/run_api.py   # binds to 0.0.0.0:8000
```

20+ REST-Endpoints unter `/api/v1/` (orders, performance, risk, signals, portfolio, qa, monitoring, paper-trading, oms).
Alle Endpoints sind read-only — der Server liest aus `output/`-Dateien.

Schnelltest: `curl http://localhost:8000/api/v1/orders/5min`

---

## Sensible Bereiche — besonders vorsichtig behandeln

### Hart geschützte Tabu-Pfade (technisch blockiert)

Diese 6 Pfade sind die **Single Source of Truth** für den Schutz-Layer und decken sich mit
`CLAUDE.md` (Abschnitt „Sensible Zonen"), `.claude/settings.json` (`permissions.deny`) und der
Hook-Liste `PROTECTED_ZONES` in `.claude/hooks/protected_paths_guard.py`. Edit/Write **und**
destruktive Bash-Schreibzugriffe werden hier **technisch geblockt** — auch unter
`bypassPermissions`. Änderungen nur mit explizitem Auftrag:

- `src/assembled_core/execution/` — Order-Generierung, Kill-Switch, Pre-Trade-Checks
- `src/assembled_core/risk/` — Risk-Controls, Limits, Exposure-/Kill-Switch-Logik
- `src/assembled_core/accounting/` — Broker-Snapshot, Evidence-Pack, Ledger
- `src/assembled_core/pipeline/` — Trading-Cycle, Orchestrator, Backtest
- `src/assembled_core/paper/` — Paper-/Simulations-Engine, State-Writes
- `.github/workflows/` — CI-Betriebslogik, keine Experimentierfläche

### Weitere advisory-sensible Bereiche (nicht hart geblockt)

Kernlogik, die ohne expliziten Auftrag nicht umstrukturiert werden sollte, aber nicht vom
Schutz-Layer erzwungen wird:

- `src/assembled_core/portfolio/` — Position-Sizing, Exposure-Steuerung
- `src/assembled_core/qa/` — Backtest-Engine, Metriken, QA-Gates
- `src/assembled_core/data/` — sofern PIT-Sicherheit, Timing oder Backtest-Realismus betroffen

Regel: In allen diesen Bereichen erst Scope und Invarianten klären, dann minimal und
nachvollziehbar ändern.

---

## .gitignore-Verhalten (wichtige Ausnahme)

Das `.gitignore`-Pattern `data/` matcht auch `src/assembled_core/data/`.
Neue Dateien dort müssen explizit geforced werden:

```bash
git add -f src/assembled_core/data/<dateiname>
```

---

## Bekannte Problemstellen (pre-existing)

- ~19 Testdateien schlagen bei der Collection fehl (unfertige Stub-Funktionen in `src/assembled_core/data/`)
- Ruff meldet 76 Lint-Findings (hauptsächlich unused imports) — CI zeigt diese ebenfalls, sie sind bekannt
- `src/assembled_core/ml/__init__.py` enthält `NotImplementedError` — ML-Trainings-Pfad nicht implementiert
- Monitoring-Endpoints unter `/api/v1/monitoring/` geben teilweise Dummy-Daten zurück
- `pyproject.toml` (ranges) und `requirements.txt` (pins) können bei `pip install` zu Versionsunterschieden führen
