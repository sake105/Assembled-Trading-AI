# Batch Runner & Parallelisierung (P4)

## Übersicht

Phase 4 (P4) implementiert einen systematischen Batch-Runner für die Ausführung vieler Backtests mit optionaler Parallelisierung. Dies ermöglicht Parameter-Sweeps, Bundle-Vergleiche und strukturierte Experimente mit automatischer Metriken-Erfassung.

**Status:** Completed

**Verwandte Dokumente:**
- [Batch Runner Design (P4)](BATCH_RUNNER_P4_DESIGN.md) - Detailliertes Design
- [Backtest Engine Optimization (P3)](BACKTEST_ENGINE_OPTIMIZATION_P3.md) - Engine-Optimierungen
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Übersicht aller Phasen

---

## Quickstart

### Basis-Batch-Run

**1. Konfiguration erstellen:**

Erstelle eine YAML-Config-Datei (z.B. `configs/batch_example.yaml`):

```yaml
batch_name: "example_batch"
description: "Example batch backtest"
output_root: "output/batch_backtests"
seed: 42
max_workers: 4
fail_fast: false

base_args:
  freq: "1d"
  strategy: "multifactor_long_short"
  data_source: "local"
  rebalance_freq: "M"
  max_gross_exposure: 1.0
  start_capital: 100000.0

runs:
  - id: "run1"
    bundle_path: "config/bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    tags: ["example", "baseline"]
```

**2. Batch ausführen:**

```bash
# Serial (einzeln, deterministisch)
python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml --serial

# Parallel (mehrere Runs gleichzeitig)
python scripts/cli.py batch_backtest --config-file configs/batch_example.yaml --max-workers 4
```

**3. Ergebnisse prüfen:**

- `output/batch_backtests/example_batch/batch_summary.csv` - Metriken-Tabelle
- `output/batch_backtests/example_batch/batch_manifest.json` - Batch-Metadaten
- `output/batch_backtests/example_batch/runs/{run_id}/run_manifest.json` - Run-Metadaten

---

## Konfigurations-Beispiele

### Beispiel 1: Grid Search (Parameter-Sweep)

Systematische Variation von Parametern:

```yaml
batch_name: "exposure_commission_grid"
description: "Grid search over exposure and commission"
output_root: "output/batch_backtests"
seed: 42
max_workers: 4

base_args:
  freq: "1d"
  strategy: "multifactor_long_short"
  bundle_path: "config/bundle.yaml"
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  rebalance_freq: "M"
  start_capital: 100000.0

grid:
  max_gross_exposure: [0.6, 0.8, 1.0, 1.2]
  commission_bps: [0.0, 0.5, 1.0]
```

**Ergebnis:** 4 x 3 = 12 Runs werden automatisch generiert.

**Run-IDs:** `max_gross_exposure_0_6_commission_bps_0_0`, `max_gross_exposure_0_6_commission_bps_0_5`, etc.

### Beispiel 2: Bundle-Vergleich

Verschiedene Factor-Bundles vergleichen:

```yaml
batch_name: "bundle_comparison"
description: "Compare AI/Tech core vs. ML-alpha bundles"
output_root: "output/batch_backtests"
seed: 42

base_args:
  freq: "1d"
  strategy: "multifactor_long_short"
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  max_gross_exposure: 1.0
  start_capital: 100000.0

runs:
  - id: "ai_tech_core_2015_2020"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    tags: ["core", "ai_tech"]

  - id: "ml_alpha_2015_2020"
    bundle_path: "config/factor_bundles/ml_alpha_bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    tags: ["ml_alpha"]

  - id: "core_2021_2024"
    bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"
    start_date: "2021-01-01"
    end_date: "2024-12-31"
    tags: ["core", "recent"]
```

### Beispiel 3: Grid + Individual Runs kombiniert

Kombination aus Grid-Search und individuellen Runs:

```yaml
batch_name: "mixed_batch"
description: "Grid search + specific test cases"
output_root: "output/batch_backtests"
seed: 42

base_args:
  freq: "1d"
  strategy: "multifactor_long_short"
  bundle_path: "config/bundle.yaml"
  start_date: "2015-01-01"
  end_date: "2020-12-31"
  max_gross_exposure: 1.0
  start_capital: 100000.0

# Grid erzeugt Runs
grid:
  commission_bps: [0.0, 0.5]

# Zusätzliche individuelle Runs
runs:
  - id: "baseline_low_exposure"
    bundle_path: "config/bundle.yaml"
    start_date: "2015-01-01"
    end_date: "2020-12-31"
    overrides:
      max_gross_exposure: 0.5
    tags: ["baseline"]
```

**Wichtig:** `runs` und `grid` können kombiniert werden - beide werden ausgeführt.

---

## CLI-Optionen

### Basis-Kommando

```bash
python scripts/cli.py batch_backtest --config-file <config.yaml>
```

### Alle Optionen

- `--config-file` (erforderlich): Pfad zur Batch-Config YAML/JSON
- `--output-root`: Override für `output_root` aus Config
- `--max-workers`: Anzahl paralleler Worker (default: aus Config oder 4)
- `--serial`: Serielle Ausführung (keine Parallelisierung)
- `--fail-fast`: Batch abbrechen bei erstem fehlgeschlagenen Run
- `--dry-run`: Nur Plan anzeigen, keine Ausführung
- `--rerun`: Überschreibe existierende Batch-Outputs

### Beispiele

**Dry-Run (Plan anzeigen):**
```bash
python scripts/cli.py batch_backtest --config-file configs/batch.yaml --dry-run
```

**Serial mit Fail-Fast:**
```bash
python scripts/cli.py batch_backtest --config-file configs/batch.yaml --serial --fail-fast
```

**Parallel mit 8 Workern:**
```bash
python scripts/cli.py batch_backtest --config-file configs/batch.yaml --max-workers 8
```

**Rerun (überschreibt existierende Outputs):**
```bash
python scripts/cli.py batch_backtest --config-file configs/batch.yaml --rerun
```

---

## Output-Struktur

### Verzeichnisstruktur

```
output/batch_backtests/{batch_name}/
├── batch_manifest.json           # Batch-Metadaten (config hash, git commit, etc.)
├── batch_summary.json            # Batch-Zusammenfassung (JSON)
├── batch_summary.csv             # Metriken-Tabelle (CSV)
└── runs/
    ├── 0000_{run_id}/
    │   ├── run_manifest.json     # Run-Metadaten
    │   ├── run.log               # Run-spezifisches Log
    │   └── backtest/
    │       ├── portfolio_equity_1d.csv
    │       ├── orders_1d.csv
    │       └── ...
    ├── 0001_{run_id}/
    │   └── ...
    └── ...
```

### Batch Summary CSV

Die `batch_summary.csv` enthält vergleichbare Metriken für alle Runs:

| run_id | status | strategy | params_hash | start_date | end_date | sharpe | deflated_sharpe | max_dd | max_dd_pct | turnover | runtime_sec | error |
|--------|--------|----------|-------------|------------|----------|--------|-----------------|--------|------------|----------|-------------|-------|
| run1   | success | multifactor_long_short | abc123 | 2015-01-01 | 2020-12-31 | 1.2345 | 1.1000 | -150.25 | -15.02 | 2.50 | 45.2 | |
| run2   | failed | ... | def456 | ... | ... | | | | | | 30.1 | Backtest exited with code 1 |

**Spalten:**
- `run_id`: Run-Identifier
- `status`: `success` | `failed` | `timeout` | `skipped`
- `strategy`: Strategie-Name (aus Config)
- `params_hash`: Config-Hash für Reproduzierbarkeit
- `start_date`, `end_date`: Backtest-Zeitraum
- `sharpe`: Sharpe Ratio (annualisiert)
- `deflated_sharpe`: Deflated Sharpe Ratio (wenn berechenbar)
- `max_dd`, `max_dd_pct`: Maximum Drawdown (absolut, Prozent)
- `turnover`: Portfolio Turnover (annualisiert, wenn Trades verfügbar)
- `runtime_sec`: Ausführungszeit pro Run
- `error`: Fehlermeldung (falls failed)

**Hinweis:** Fehlende Metriken werden als leerer String geschrieben (kein NaN in CSV).

### Batch Manifest

`batch_manifest.json` enthält Metadaten für den gesamten Batch:

```json
{
  "batch_name": "example_batch",
  "description": "Example batch",
  "started_at": "2025-01-15T10:00:00",
  "finished_at": "2025-01-15T10:30:00",
  "total_runtime_sec": 1800.5,
  "config_hash": "abc123...",
  "git_commit_hash": "a1b2c3d",
  "seed": 42,
  "max_workers": 4,
  "fail_fast": false,
  "output_root": "output/batch_backtests",
  "base_args": {...},
  "expanded_runs": [...],
  "run_results_summary": {
    "total_runs": 12,
    "success_count": 10,
    "failed_count": 2,
    "run_ids": [...]
  },
  "versions": {
    "python": "3.11.9"
  }
}
```

### Run Manifest

Jeder Run erhält ein `run_manifest.json`:

```json
{
  "run_id": "run1",
  "run_index": 0,
  "status": "success",
  "started_at": "2025-01-15T10:00:00",
  "finished_at": "2025-01-15T10:02:30",
  "runtime_sec": 150.5,
  "config_hash": "def456...",
  "git_commit_hash": "a1b2c3d",
  "run_spec": {
    "bundle_path": "config/bundle.yaml",
    "start_date": "2015-01-01",
    "end_date": "2020-12-31",
    "tags": ["example"],
    "overrides": {}
  },
  "base_args": {...},
  "artifacts": [
    "backtest/portfolio_equity_1d.csv",
    "backtest/orders_1d.csv",
    "run.log"
  ],
  "error": null
}
```

---

## Parallelisierung

### Serial vs. Parallel

**Serial (`--serial`):**
- Runs werden sequenziell ausgeführt
- Deterministische Ausführung
- Einfacher zu debuggen
- Langsamer auf Multi-Core-Systemen

**Parallel (Standard):**
- Runs werden parallel mit `ProcessPoolExecutor` ausgeführt
- `max_workers` steuert Parallelitätsgrad
- Schneller auf Multi-Core-Systemen
- Ergebnisse bleiben deterministisch (Reihenfolge beibehalten)

### Best Practices

**Für kleine Batches (< 10 Runs):**
- Serial reicht oft aus
- Einfacher zu debuggen

**Für große Batches (> 20 Runs):**
- Parallel mit `max_workers = min(os.cpu_count(), 8)` empfohlen
- Verhindert Overhead bei zu vielen Workern

**Bei Ressourcen-Engpässen:**
- `max_workers = 2-4` für moderate Last
- Monitor CPU/RAM während Ausführung

---

## Error Handling

### Fail-Fast vs. Continue

**`fail_fast: false` (Standard):**
- Batch läuft weiter auch bei fehlgeschlagenen Runs
- Alle Runs werden ausgeführt
- Failures werden in `batch_summary.csv` dokumentiert

**`fail_fast: true`:**
- Batch wird beim ersten fehlgeschlagenen Run abgebrochen
- Verbleibende Runs werden gecancelt
- Nützlich für Debugging oder wenn Abhängigkeiten bestehen

### Timeout

Aktuell: Kein Timeout pro Run (läuft bis Completion oder Failure)

**Zukünftig (optional):**
- `timeout_per_run` Parameter in Config
- Runs die länger als Timeout dauern werden als `timeout` markiert

---

## Health Checks & Operations

Batch-Runs sind in Operations Health-Checks integriert:

```bash
# Batch-Checks aktivieren
python scripts/check_health.py --batch-root output/batch_backtests

# Mit Custom Failure-Rate Threshold
python scripts/check_health.py --batch-root output/batch_backtests --batch-max-failure-rate 0.1
```

**Checks:**
- `batch_latest_status`: Status des letzten Batches
- `batch_failure_rate`: Failure-Rate (OK/WARN/CRITICAL)
- `batch_missing_manifests`: Fehlende Run-Manifeste

**Siehe auch:** [Health Checks Documentation](../scripts/check_health.py)

---

## Best Practices

### 1. Config-Validierung

**Vor Ausführung:**
```bash
# Dry-Run zeigt Plan an
python scripts/cli.py batch_backtest --config-file configs/batch.yaml --dry-run
```

**Prüfe:**
- Anzahl generierter Runs (Grid-Expansion)
- Run-IDs sind eindeutig
- Bundle-Pfade existieren

### 2. Seed-Management

**Deterministische Reproduzierbarkeit:**
```yaml
seed: 42  # Fester Seed für Reproduzierbarkeit
```

**Für Experimente:**
- Gleicher Seed → gleiche Ergebnisse
- Verschiedene Seeds für Robustheits-Tests

### 3. Output-Organisation

**Strukturierte Batch-Namen:**
```yaml
batch_name: "experiment_2025_01_exposure_sweep"  # Datum + Zweck
```

**Output-Root:**
```yaml
output_root: "output/batch_backtests"  # Zentraler Ort für alle Batches
```

### 4. Metriken-Vergleich

**Batch Summary CSV analysieren:**
```python
import pandas as pd

df = pd.read_csv("output/batch_backtests/example_batch/batch_summary.csv")
# Filtere erfolgreiche Runs
successful = df[df["status"] == "success"]
# Sortiere nach Sharpe
top_runs = successful.nlargest(5, "sharpe")
```

**Beste Parameter identifizieren:**
- Filter nach `status == "success"`
- Sortiere nach Metrik (z.B. `sharpe`, `-max_dd_pct`)
- Prüfe `params_hash` für Reproduzierbarkeit

### 5. Grid-Search-Größe

**Kleine Grids (< 50 Runs):**
- Direkt ausführbar
- Vollständige Coverage

**Große Grids (> 100 Runs):**
- Betrachte Random Sampling (außerhalb P4)
- Oder: Mehrere kleinere Batches

---

## Troubleshooting

### Problem: "No batch directories found"

**Lösung:**
- Prüfe `output_root` in Config
- Stelle sicher, dass Batch bereits ausgeführt wurde
- Verwende `--skip-batch-if-missing` in Health-Checks

### Problem: "Run ID already exists"

**Lösung:**
- Grid-Expansion erzeugt eindeutige IDs automatisch
- Für individuelle Runs: Stelle sicher, dass IDs eindeutig sind
- Verwende `--rerun` um existierende Outputs zu überschreiben

### Problem: "Failed to load batch manifest"

**Lösung:**
- Prüfe ob `batch_manifest.json` existiert
- Validiere JSON-Format
- Prüfe Berechtigungen

### Problem: Parallele Runs scheitern

**Lösung:**
- Reduziere `max_workers` (z.B. `--max-workers 2`)
- Prüfe Ressourcen (CPU/RAM)
- Nutze `--serial` für Debugging

### Problem: Metriken fehlen in CSV

**Lösung:**
- Prüfe ob Equity-Curve-Dateien existieren
- Validiere Run-Outputs
- Metriken werden als leerer String geschrieben wenn nicht verfügbar

---

## Integration mit anderen Tools

### Health Checks

```bash
# Batch-Status in Health-Checks prüfen
python scripts/check_health.py --batch-root output/batch_backtests
```

### Benchmarking

Batch-Runs können mit P3-optimiertem Engine ausgeführt werden:

```bash
# Batch nutzt automatisch optimierten Engine (P3)
python scripts/cli.py batch_backtest --config-file configs/batch.yaml
```

### Experiment Tracking

Run-Manifeste enthalten Tags für Experiment-Tracking:

```yaml
runs:
  - id: "run1"
    tags: ["experiment_2025", "baseline", "ai_tech"]
```

**Zukünftig:** Integration mit `src/assembled_core/qa/experiment_tracking.py`

---

## Limits & Constraints

### Aktuelle Limits

- **Max Workers:** Kein Hard-Limit, aber `max_workers` wird validiert
- **Timeout:** Kein Timeout pro Run (läuft bis Completion)
- **Retry:** Keine automatische Retry-Logik (nur manuell via `--rerun`)

### Empfohlene Limits

- **Grid-Size:** < 100 Runs pro Batch (für Übersichtlichkeit)
- **Max Workers:** `min(os.cpu_count(), 8)` (verhindert Overhead)
- **Concurrent Runs:** Respektiere System-Ressourcen

---

## Weiterführende Dokumentation

- [Batch Runner Design (P4)](BATCH_RUNNER_P4_DESIGN.md) - Detailliertes Design und Task-Liste
- [Backtest Engine Optimization (P3)](BACKTEST_ENGINE_OPTIMIZATION_P3.md) - Engine-Optimierungen
- [Health Checks](../scripts/check_health.py) - Operations Health-Checks
- [CLI Documentation](../scripts/cli.py) - Zentrale CLI-Referenz

