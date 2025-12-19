## P4 - Batch Backtests & Parallelisierung (Design)

### 1. Overview & Scope

**Goal:** P4 erweitert die bestehende Backtest-Infrastruktur um einen Batch-Runner, der mehrere Backtests aus einer Konfigurationsdatei (YAML/JSON) ausführen kann, optional parallelisiert und sauber versionierte Outputs + Summary-Reports erzeugt.

**Bezug zu P1–P3:**
- **P1 (Performance Profiling):** Liefert die Mess-Infrastruktur (`scripts/profile_jobs.py`, `timed_block`) und Baseline-Metriken.
- **P2 (Factor Store):** Reduziert I/O- und Faktor-Berechnungs-Overhead für wiederholte Backtests (schnelleres Laden von Panels/Faktoren).
- **P3 (Backtest Optimization):** Optimiert den einzelnen Backtest-Run (vektorisierte Position-Updates, Numba, Regressionstests, Benchmarks).
- **P4 (Batch Runner):** Skaliert die Backtests horizontal (viele Runs) mit optionaler Parallelisierung und integriert P1/P3 für sauberes Profiling.

**Scope P4:**
- Batch-Definition via YAML/JSON-Konfigurationsdatei.
- Serieller Batch-Runner mit optionaler Parallelisierung (mehrere Prozesse).
- Standardisierte Output-Struktur je Backtest-Run + globale Batch-Summary (CSV/Markdown).
- Integration mit bestehender CLI (`scripts/cli.py`) und Profiling (`scripts/profile_jobs.py`, `timed_block`).

**Out of Scope:**
- Verteilte Cluster-Ausführung (Kubernetes, Dask, Ray).
- Live-Trading oder Paper-Trading-Steuerung (SAFE-Bridge bleibt unverändert).
- Änderung der Backtest-Logik selbst (P3 deckt Engine-Optimierung ab).

---

### 2. Data Contract – Batch-Config (YAML/JSON)

Der Batch-Runner liest eine Konfigurationsdatei (YAML oder JSON) mit folgender Struktur:

```yaml
batch_name: "ai_tech_core_vs_mlalpha_2025"
description: "Compare AI/Tech core bundle vs. ML-alpha bundles across multiple periods"
output_root: "output/batch_backtests"

defaults:
  freq: "1d"
  data_source: "local"
  strategy: "multifactor_long_short"
  rebalance_freq: "M"
  max_gross_exposure: 1.0
  start_capital: 100000
  generate_report: true
  generate_risk_report: true
  generate_tca_report: false
  symbols_file: "config/universe_ai_tech_tickers.txt"

runs:
  - id: "core_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"

  - id: "core_ml_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_ml_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"

  - id: "ml_alpha_only_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_ml_alpha_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    generate_tca_report: true
```

**Felder:**

- **batch_name**: Eindeutiger Name des Batch-Runs (Pfad-Komponente).
- **description**: Freitext-Beschreibung (wird in Summary geschrieben).
- **output_root**: Basisverzeichnis für Batch-Outputs (Standard: `output/batch_backtests`).
- **defaults**: Standardwerte für alle Runs (können pro Run überschrieben werden).
  - `freq`, `data_source`, `strategy`, `rebalance_freq`, `max_gross_exposure`, `start_capital`
  - `generate_report`: bool (Backtest-Report wie heute).
  - `generate_risk_report`: bool (CLI `risk_report` auf dem Backtest-Dir).
  - `generate_tca_report`: bool (CLI `tca_report` auf dem Backtest-Dir).
  - `symbols_file` oder `universe_file`: wie bei `run_backtest_strategy.py`.
- **runs**: Liste einzelner Backtest-Konfigurationen:
  - `id`: Kurzname des Runs (wird Teil des Output-Pfads).
  - `bundle_path`: Pfad zum Factor-Bundle (z. B. Core, Core+ML, ML-only).
  - `start_date`, `end_date`: Datumsbereich.
  - Optionale Overrides: Alle Felder aus `defaults` können auf Run-Ebene gesetzt werden.

**Validierung:**
- `batch_name` darf nur `[a-zA-Z0-9_-]` enthalten (für Pfade).
- Jeder Run benötigt mindestens `id`, `bundle_path`, `start_date`, `end_date`.
- Kein Netzwerkzugriff: `data_source` muss `"local"` sein (oder per Guardrail erzwungen).

---

### 3. Batch-Runner API – Geplante Funktionen

Geplantes Modul: `scripts/batch_backtest_runner.py` (oder `src/assembled_core/qa/batch_runner.py` mit dünnem Script-Wrapper).

**Kernfunktionen:**

```python
def load_batch_config(path: Path) -> BatchConfig:
    """Load and validate batch config from YAML/JSON."""

def run_single_backtest(run_cfg: SingleRunConfig, batch_ctx: BatchContext) -> SingleRunResult:
    """Execute a single backtest + optional post-processing (risk, TCA)."""

def run_batch(batch_cfg: BatchConfig, parallel: bool = False, max_workers: int | None = None) -> BatchResult:
    """Execute all runs in the batch, optionally in parallel."""

def write_batch_summary(batch_result: BatchResult, output_dir: Path) -> None:
    """Write CSV/Markdown summaries for the batch."""
```

**Dataklassen (Konzept):**

- `BatchConfig`:
  - `name: str`
  - `description: str | None`
  - `output_root: Path`
  - `defaults: dict[str, Any]`
  - `runs: list[SingleRunConfig]`

- `SingleRunConfig`:
  - `id: str`
  - `bundle_path: Path`
  - `start_date: str`
  - `end_date: str`
  - plus expandierte Defaults (freq, strategy, etc.)

- `SingleRunResult`:
  - `run_id: str`
  - `status: str` (`"success"`, `"failed"`, `"skipped"`)
  - `backtest_dir: Path | None`
  - `runtime_sec: float`
  - `metrics: dict[str, float]` (Sharpe, final_pf, trades, etc.)
  - `risk_report_path: Path | None`
  - `tca_report_path: Path | None`
  - `error: str | None`

- `BatchResult`:
  - `batch_name: str`
  - `started_at: datetime`
  - `finished_at: datetime`
  - `results: list[SingleRunResult]`

**Interne Aufrufe:**
- Backtest via Python-Funktionsaufruf:
  - Entweder direkter Import von `run_backtest_strategy.main()` (falls sinnvoll extrahiert),
  - oder direkter Aufruf der Engine `run_portfolio_backtest()` mit klaren Helpern für Price/Signal/Sizing-Setup.
- Optionales Nachprocessing:
  - `scripts/cli.py risk_report --backtest-dir ...`
  - `scripts/cli.py tca_report --backtest-dir ...`

Für P4-Start: Minimum-Variante kann zunächst `subprocess`-Aufrufe auf `scripts/cli.py run_backtest` nutzen (wie `profile_jobs.py`), solange keine Logik dupliziert wird.

---

### 4. Logging- und Output-Konzept

**Verzeichnisstruktur:**

```text
output/batch_backtests/
  {batch_name}/
    batch_config.yaml           # Kopie der Eingabekonfig
    batch_summary.csv           # Eine Zeile pro Run
    batch_summary.md            # Menschlich lesbarer Überblick
    logs/
      batch_{timestamp}.log     # Batch-Level-Logs
    runs/
      {run_id}/
        backtest/               # Backtest-Output (wie heute)
        risk_report/            # Optionaler Risk-Report
        tca/                    # Optionaler TCA-Report
        profile/                # Optional: Profiling-Artefakte (P1)
```

**Filenaming:**
- Backtests nutzen bestehende Struktur:
  - `output/backtests/{strategy}_{bundle}_{timestamp}/...` oder alternativ nested unter `runs/{run_id}/backtest`.
- Batch-Runner sollte:
  - Pfad zum Backtest-Output speichern (für Summary-Links).
  - Optional `run_id` in Backtest-Config/Run-ID einbetten (z. B. via `generate_run_id`-Prefix).

**Logging:**
- Batch-Level-Logger:
  - Start/Ende des Batches.
  - Pro Run: Status, Laufzeit, Pfade zu Artefakten.
  - Zusammenfassung: Anzahl erfolgreicher/fehlgeschlagener Runs.
- Run-spezifische Logs:
  - Backtest-Scripts und Risiko-/TCA-Scripts loggen bereits in eigenen Dateien.

**Batch-Summary (CSV/Markdown):**

CSV-Spalten (Beispiel):
- `run_id`, `bundle_path`, `start_date`, `end_date`, `status`, `runtime_sec`
- `final_pf`, `sharpe`, `trades` (aus Backtest-Metriken)
- `risk_report_path`, `tca_report_path`

Markdown-Summary:
- Kurze Beschreibung des Batches.
- Tabelle mit wichtigsten Kennzahlen pro Run.
- Optionale Interpretation (z. B. welcher Bundle/Zeitraum hat beste Sharpe).

---

### 5. Parallelisierungs-Strategie

**Ziel:** Mehrere Backtests parallel ausführen, ohne die Engine-Logik zu ändern, und dabei CPU-Kerne effizient nutzen.

**Vorschlag:**
- Verwendung von `concurrent.futures.ProcessPoolExecutor`:
  - Ein Prozess pro Backtest-Run (keine GIL-Problematik bei Python-Only-Code).
  - Gute Isolation: Fehler in einem Run crashen nicht den gesamten Batch.
- Konfiguration:
  - `max_workers`:
    - Default: `min(os.cpu_count() or 2, 4)` (konservativ).
    - Override via CLI-Argument oder Umgebungsvariable (z. B. `ASSEMBLED_BATCH_MAX_WORKERS`).
- Ressourcen:
  - Kein Shared-State zwischen Prozessen (Config/Params werden serialisiert).
  - I/O (Lokale Files) können parallel genutzt werden, aber Pfade dürfen nicht kollidieren.

**API-Skizze:**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_batch(batch_cfg: BatchConfig, parallel: bool = False, max_workers: int | None = None) -> BatchResult:
    if not parallel:
        # Seriell
        results = []
        for run_cfg in batch_cfg.runs:
            results.append(_run_single_with_timing(run_cfg, batch_cfg))
        return BatchResult(...)

    # Parallel
    workers = max_workers or _default_max_workers()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_single_with_timing, run_cfg, batch_cfg): run_cfg.id
            for run_cfg in batch_cfg.runs
        }
        results = []
        for fut in as_completed(futures):
            run_id = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                # Wrap as failed result
                results.append(SingleRunResult(run_id=run_id, status="failed", error=str(exc), ...))

    return BatchResult(...)
```

**Fehlerverhalten:**
- Einzelne Runs dürfen fehlschlagen, ohne den Batch abzubrechen.
- Batch-Summary markiert fehlgeschlagene Runs klar (Status + Error-Message).
- Globaler Return-Code:
  - `0`, wenn mindestens ein Run erfolgreich war.
  - `1`, wenn alle Runs fehlgeschlagen sind (oder Config nicht lesbar).

---

### 6. Testing-Strategie

**Unit-Tests (unter `tests/`):**

- `tests/test_batch_backtest_config.py`:
  - Laden unterschiedlicher Konfigurationsdateien (YAML/JSON).
  - Validierung von Defaults/Overrides, Fehler bei fehlenden Pflichtfeldern.
  - Pfad-Normalisierung (`output_root` als `Path`).

- `tests/test_batch_backtest_runner.py`:
  - Mock/Dummy-Backtest-Funktion (kein echter Backtest, nur schneller Platzhalter).
  - Tests für:
    - Seriellen Run: alle Runs werden ausgeführt, Summary stimmt.
    - Parallel-Run: gleiche Ergebnisse wie seriell (für kleine Anzahl Runs).
    - Fehler-Handling: ein Run wirft Exception → Status `"failed"`, Batch läuft weiter.

**End-to-End-Minibatch:**

- Kleine Konfig mit 2–3 Runs:
  - Z. B. AI/Tech Core- vs. Core+ML-Bundle auf kurzem Zeitraum (oder synthetischen Daten).
  - Nutzung lokaler Daten (kein Netzwerk).
- Test prüft:
  - Batch-Summary-CSV existiert und enthält alle Runs.
  - Pro Run existiert ein Backtest-Output-Verzeichnis.
  - Optional: Risk-Report/TCA-Report für Runs mit entsprechenden Flags.

**Performance-Tests (optional):**
- Nicht Teil der Unit-Tests (würden CI verlangsamen), aber als Script:
  - Multi-Batch-Runs via `profile_jobs.py` oder eigenem Benchmark-Script.
  - Messen von Skalierungsverhalten mit/ohne Parallelisierung.

---

### 7. CLI-Integration

Neuer Subcommand in `scripts/cli.py`:

```text
batch_backtest
- --config FILE (YAML/JSON)
- --parallel / --no-parallel
- --max-workers INT (optional)
```

**Beispiel-Aufrufe:**

```powershell
python scripts/cli.py batch_backtest `
  --config configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml

python scripts/cli.py batch_backtest `
  --config configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml `
  --parallel `
  --max-workers 4
```

**Verhalten:**
- Liest Config, führt `run_batch()` aus.
- Schreibt Batch-Summary-Dateien und loggt Pfade.
- Return-Code entsprechend der Batch-Ergebnisse (siehe oben).

**Integration mit P1 (Profiling):**
- Optionaler Flag `--with-profiling`:
  - Aktiviert zusätzliche `timed_block`-Summary (z. B. Aggregation der Backtest-Schritte).
  - Kann einfache Laufzeitprotokolle pro Run in `runs/{run_id}/profile/` ablegen.
- Integration mit `scripts/profile_jobs.py`:
  - Neuer Job-Typ `BATCH_BACKTEST_JOB`, der intern `cli.py batch_backtest` aufruft.

---

### 8. Integration mit P3 (Optimierte Engine)

**Wichtige Punkte:**
- Batch-Runner ruft ausschließlich die bereits optimierte Backtest-Engine auf:
  - Entweder indirekt via `scripts/cli.py run_backtest`.
  - Oder direkt via Python-Funktionen, die `run_portfolio_backtest()` nutzen.
- P3-Optimierungen (vektorisierte Position-Updates, Numba, Regressionstests, Benchmarks) gelten unverändert für jeden Run.
- Keine Engine-spezifische Sonderlogik im Batch-Runner:
  - Batch-Runner ist eine dünne Orchestrierungsschicht.

**Vorteile:**
- P4 skaliert die bereits optimierten P3-Runs horizontal.
- Kombination von P1 + P3 + P4 ermöglicht:
  - Messung von End-to-End-Performance über viele Strategien/Zeiträume.
  - Vergleich verschiedener Bundles/Parameter-Sets in vertretbarer Zeit.

---

### 9. Risiken & Mitigation

**9.1. Ressourcenverbrauch (CPU/IO)**
- Risiko: Zu viele parallele Backtests überlasten Maschine (CPU, RAM, IO).
- Mitigation:
  - Konservative Default-Werte für `max_workers`.
  - Optionales CLI-Flag/Env-Var zur Limitierung.
  - Dokumentation von Hardware-Annahmen in Benchmarks.

**9.2. Pfad-Kollisionen**
- Risiko: Mehrere Runs schreiben in dasselbe Output-Verzeichnis.
- Mitigation:
  - Strikte Namenskonvention (`{batch_name}/runs/{run_id}/...`).
  - Assert, dass Zielverzeichnis pro Run leer/nicht vorhanden ist, bevor geschrieben wird.

**9.3. Fehlerpropagation**
- Risiko: Einzelner Fehler bricht gesamten Batch ab.
- Mitigation:
  - Fehler pro Run kapseln und als `"failed"` markieren.
  - Batch läuft weiter, Summary listet alle Fehler auf.

**9.4. Testbarkeit**
- Risiko: Zu enge Kopplung an reale Daten/Umgebung macht Tests fragil.
- Mitigation:
  - Unit-Tests verwenden synthetische Konfigs + Dummy-Backtest-Funktion.
  - E2E-Tests nur mit kleinen, stabilen Szenarien.

---

### 10. Success Criteria

**Funktional:**
- Batch-Konfig (YAML/JSON) kann mehrere Backtests definieren.
- Batch-Runner führt alle Runs seriell oder parallel aus.
- Für jeden Run existiert ein klar definiertes Output-Verzeichnis.
- Batch-Summary-CSV/Markdown enthalten alle relevanten Kennzahlen.

**Nicht-funktional:**
- Optional parallelisierte Ausführung reduziert Gesamt-Laufzeit bei multi-Core-Systemen.
- Keine Änderungen der Backtest-Logik (P3 bleibt unverändert).
- Kompatibel mit bestehenden CLI- und Profiling-Workflows.

**Testing:**
- Unit- und E2E-Tests für Batch-Runner bestehen.
- Bestehende Backtest-Tests bleiben grün (keine Regressionen).

---


