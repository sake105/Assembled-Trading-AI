# Batch Runner MVP (P4)

## Übersicht

Reproduzierbare Batch-Backtests mit konfig-basierter Ausführung und Manifest pro Run.

**Status:** MVP

**Zweck:** Einfache Ausführung mehrerer Backtests mit reproduzierbarer Konfiguration und Metadaten-Tracking.

---

## YAML Schema

### Basis-Konfiguration

```yaml
batch_name: "example_batch"
output_root: "output/batch_backtests"
seed: 42

runs:
  - id: "run1"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
    use_factor_store: false
    factor_store_root: null
    factor_group: null
```

### Erweiterte Konfiguration

```yaml
batch_name: "factor_store_batch"
output_root: "output/batch_backtests"
seed: 42

runs:
  - id: "run1_with_factors"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
    use_factor_store: true
    factor_store_root: "data/factors"
    factor_group: "default"
  - id: "run2_no_factors"
    strategy: "trend_baseline"
    freq: "1d"
    start_date: "2020-01-01"
    end_date: "2023-12-31"
    universe: "watchlist.txt"
    start_capital: 100000.0
    use_factor_store: false
```

### Schema-Definition

**Top-Level:**
- `batch_name` (string, required): Name des Batches
- `output_root` (string, optional): Output-Verzeichnis (default: "output/batch_backtests")
- `seed` (int, optional): Seed für Reproduzierbarkeit (default: 42)
- `runs` (list, required): Liste von Run-Konfigurationen

**Run-Konfiguration:**
- `id` (string, optional): Eindeutige Run-ID (wird auto-generiert wenn nicht gesetzt)
- `strategy` (string, required): Strategie-Name (z.B. "trend_baseline", min. 1 Zeichen)
- `freq` (string, required): Trading-Frequenz (nur "1d" oder "5min" erlaubt)
- `start_date` (string, required): Start-Datum im Format YYYY-MM-DD (muss gültiges Datum sein)
- `end_date` (string, required): End-Datum im Format YYYY-MM-DD (muss >= start_date sein)
- `universe` (string, optional): Pfad zur Universe-Datei (wenn gesetzt, darf nicht leer sein)
- `start_capital` (float, optional): Startkapital (default: 100000.0, muss > 0 sein)
- `use_factor_store` (bool, optional): Factor Store nutzen (default: false)
- `factor_store_root` (string, optional): Factor Store Root-Pfad (wenn gesetzt, darf nicht leer sein)
- `factor_group` (string, optional): Factor Group (wenn gesetzt, darf nicht leer sein)

## Validierungsregeln

Die Batch-Config wird strikt validiert, um Fehler früh zu erkennen (besonders wichtig im Parallel-Mode):

### Batch-Level Validierung:
- `batch_name`: Muss gesetzt sein (nicht leer)
- `seed`: Muss eine nicht-negative Ganzzahl sein (>= 0)
- `runs`: Muss eine nicht-leere Liste sein
- `run_id` Uniqueness: Alle Run-IDs müssen eindeutig sein (nach Auto-Generierung)

### Run-Level Validierung:
- **Required Fields:** `strategy`, `freq`, `start_date`, `end_date` müssen gesetzt sein
- **Date Format:** `start_date` und `end_date` müssen im Format `YYYY-MM-DD` sein und gültige Datumsangaben sein
- **Date Logic:** `end_date` muss >= `start_date` sein
- **Freq Enum:** `freq` muss exakt "1d" oder "5min" sein (case-sensitive)
- **Start Capital:** `start_capital` muss > 0 sein (falls gesetzt)
- **String Fields:** `universe`, `factor_store_root`, `factor_group` dürfen nicht leer sein (wenn gesetzt)

### Fehlerbehandlung:
- Validierungsfehler werden früh geworfen (vor Ausführung)
- Fehlermeldungen enthalten Feld-Namen und konkrete Werte
- Bei Parallel-Mode werden Config-Fehler vor dem Start aller Worker erkannt

---

## Manifest Format

### Run Manifest (`run_manifest.json`)

Jeder Run erhält ein Manifest im Run-Output-Verzeichnis:

```json
{
  "run_id": "run1",
  "status": "success",
  "started_at": "2025-01-15T10:00:00Z",
  "finished_at": "2025-01-15T10:02:30Z",
  "runtime_sec": 150.5,
  "params": {
    "strategy": "trend_baseline",
    "freq": "1d",
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "universe": "watchlist.txt",
    "start_capital": 100000.0,
    "use_factor_store": false,
    "factor_store_root": null,
    "factor_group": null
  },
  "git_commit_hash": "a1b2c3d",
  "timings_path": "run_timings.json",
  "output_dir": "output/batch_backtests/example_batch/run1"
}
```

**Felder:**
- `run_id`: Run-Identifier
- `status`: "success" | "failed" | "skipped"
- `started_at`: Start-Zeitstempel (ISO 8601 UTC)
- `finished_at`: End-Zeitstempel (ISO 8601 UTC)
- `runtime_sec`: Ausführungszeit in Sekunden
- `params`: Run-Parameter (vollständige Konfiguration)
- `git_commit_hash`: Git Commit Hash (optional, null wenn nicht verfügbar)
- `timings_path`: Relativer Pfad zu Timings-JSON (optional, null wenn nicht verfügbar)
- `output_dir`: Absoluter Pfad zum Run-Output-Verzeichnis

---

## Verwendung

### Basis-Befehl

```bash
python scripts/batch_runner.py --config-file configs/batch_example.yaml
```

### CLI-Optionen

- `--config-file` (required): Pfad zur YAML-Config-Datei
- `--output-root` (optional): Override für `output_root` aus Config
  - Hinweis: `batch_name` wird immer als Unterordner verwendet (z.B. `{output_root}/{batch_name}/`)
- `--dry-run` (optional): Nur Plan anzeigen, keine Ausführung
- `--max-workers` (optional): Anzahl paralleler Worker (default: 1 = seriell)
- `--resume` (optional): Skip erfolgreiche Runs (basiert auf Manifest)
- `--rerun-failed` (optional): Rerun fehlgeschlagene Runs (nur mit `--resume`)

### Beispiel-Ausführung

```bash
# Dry-Run (Plan anzeigen)
python scripts/batch_runner.py --config-file configs/batch_example.yaml --dry-run

# Serielle Ausführung
python scripts/batch_runner.py --config-file configs/batch_example.yaml

# Parallele Ausführung (4 Worker)
python scripts/batch_runner.py --config-file configs/batch_example.yaml --max-workers 4

# Resume (Skip erfolgreiche Runs)
python scripts/batch_runner.py --config-file configs/batch_example.yaml --resume

# Resume + Rerun Failed
python scripts/batch_runner.py --config-file configs/batch_example.yaml --resume --rerun-failed

# Custom Output-Root
python scripts/batch_runner.py --config-file configs/batch_example.yaml --output-root /tmp/batch_outputs
# → Output: /tmp/batch_outputs/{batch_name}/
```

---

## Output-Struktur

Jeder Batch wird isoliert in einem eigenen Unterordner organisiert, um Kollisionen zwischen mehreren gleichzeitigen Batches zu vermeiden:

```
output/batch_backtests/{batch_name}/
├── summary.csv                     # Batch-Summary (CSV)
├── summary.json                    # Batch-Summary (JSON)
├── run1/                           # Run 1 Output
│   ├── run_manifest.json           # Run-Manifest
│   ├── run_timings.json            # Timings (optional)
│   └── backtest/                   # Backtest-Outputs
│       ├── portfolio_equity_1d.csv
│       ├── orders_1d.csv
│       └── ...
├── run2/                           # Run 2 Output
│   ├── run_manifest.json
│   └── ...
└── ...
```

**Pfad-Struktur:**
- Batch-Root: `{output_root}/{batch_name}/`
- Run-Verzeichnis: `{batch_root}/{run_id}/`
- Summary-Dateien: `{batch_root}/summary.csv` und `{batch_root}/summary.json`

**Beispiel:**
- `output_root`: `"output/batch_backtests"`
- `batch_name`: `"example_batch"`
- Batch-Root: `output/batch_backtests/example_batch/`
- Run 1: `output/batch_backtests/example_batch/run1/`
- Summary: `output/batch_backtests/example_batch/summary.csv`

---

## Implementierung

### Interner Ablauf

1. **Config laden:** YAML-Datei wird geladen und validiert
2. **Runs iterieren:** Für jeden Run in `runs[]`:
   - Run-Output-Verzeichnis erstellen
   - `argparse.Namespace` aus Run-Config bauen
   - `run_backtest_from_args(args)` direkt aufrufen (Python-Funktion, kein Subprocess)
   - Run-Manifest schreiben
   - Timings optional schreiben (wenn verfügbar)

### Backtest-Aufruf

Der Batch-Runner ruft `run_backtest_from_args` direkt auf:

```python
from scripts.run_backtest_strategy import run_backtest_from_args, parse_args

# Build args from run config
args = argparse.Namespace(
    freq=run_cfg["freq"],
    strategy=run_cfg["strategy"],
    start_date=run_cfg["start_date"],
    end_date=run_cfg["end_date"],
    universe=run_cfg.get("universe"),
    start_capital=run_cfg.get("start_capital", 100000.0),
    use_factor_store=run_cfg.get("use_factor_store", False),
    factor_store_root=run_cfg.get("factor_store_root"),
    factor_group=run_cfg.get("factor_group"),
    out=run_output_dir,
    # ... weitere Args
)

# Direct function call (no subprocess)
exit_code = run_backtest_from_args(args)
```

---

## Reproduzierbarkeit

### Seed-Management

Der `seed` wird in der Batch-Config gesetzt und kann für deterministische Ausführung genutzt werden.

### Git Hash

Das Manifest enthält optional den Git Commit Hash für Versions-Tracking:
- Wenn Git verfügbar: Hash wird erfasst
- Wenn nicht verfügbar: `null` im Manifest

### Parameter-Tracking

Alle Run-Parameter werden im Manifest gespeichert für vollständige Reproduzierbarkeit.

---

## Einschränkungen (MVP)

- **Serial Execution:** Runs werden seriell ausgeführt (keine Parallelisierung)
- **Einfaches Schema:** Kein Grid-Search, keine erweiterten Features
- **Basis-Manifest:** Keine Metriken-Aggregation oder Batch-Summary

**Zukünftige Erweiterungen:**
- Parallelisierung mit `ProcessPoolExecutor`
- Grid-Search-Syntax
- Batch-Summary mit Metriken-Vergleich
- Integration mit Experiment-Tracking

---

## Siehe auch

- [Batch Runner Design (P4)](BATCH_RUNNER_P4_DESIGN.md) - Detailliertes Design
- [Backtest Strategy CLI](../scripts/run_backtest_strategy.py) - Backtest-Entry-Point
