# Broker Snapshot Importer/Writer Workflow - Implementierungsplan

## Status: Plan (noch nicht implementiert)

## Ziel

Ermoeglicht es, externe Broker-Snapshots (von Broker API, CSV, JSON) in das standardisierte
Snapshot-Format zu importieren und im gleichen Layout wie Paper-Snapshots zu speichern.

## Aktuelle Situation

### Wo Snapshots geschrieben werden:

1. **Paper Snapshots** (automatisch):
   - `src/assembled_core/accounting/ledger_integration.py` (Zeile 250-286)
   - Wird ausgeloest durch `write_paper_broker_snapshot=True`
   - Verwendet: `store_broker_snapshot_json()` + `store_broker_snapshot_parquet()`
   - Input: Paper View (positions_df + cash_balance aus Ledger)

2. **CLI Flags**:
   - `scripts/run_backtest_strategy.py`: `--write-broker-snapshot`
   - `scripts/run_eod_pipeline.py`: `--write-broker-snapshot`

### Wo Snapshots geladen werden:

1. **Ledger Integration**:
   - `src/assembled_core/accounting/ledger_integration.py` (Zeile 156-157)
   - Verwendet: `load_broker_snapshot_json()` + `load_broker_snapshot_parquet()`
   - Policy: `ignore` / `prefer` / `require`
   - Normalisierung: `normalize_broker_snapshot()` (Zeile 176-182)

### Output-Layout (bereits standardisiert):

```
output/broker_snapshot_<run_id>/
  snapshot_<YYYY-MM-DD>.json
  positions_<YYYY-MM-DD>.parquet  (optional)
```

## Implementierungsplan

### 1) Neue Funktionen in `broker_snapshot_store.py`

**Funktion 1: `import_broker_snapshot_from_dict()`**
- Input: dict mit `cash` (float) und `positions` (list of dicts mit `symbol`, `qty`)
- Optional: `as_of_date` (wenn nicht vorhanden, wird aktuelles Datum verwendet)
- Validierung: required fields, Typ-Checks
- Normalisierung: via `normalize_broker_snapshot()` (trim, filter tiny residuals, sort)
- Storage: ruft `store_broker_snapshot_json()` + optional `store_broker_snapshot_parquet()` auf
- Return: Path zu geschriebenem Snapshot

**Funktion 2: `import_broker_snapshot_from_csv()`**
- Input: CSV-Pfad mit Spalten: `symbol`, `qty` (optional: `cash` als einzelne Zeile oder separater Wert)
- Parsing: pandas `read_csv()`
- Validierung: required columns
- Normalisierung: via `normalize_broker_snapshot()`
- Storage: wie oben
- Return: Path zu geschriebenem Snapshot

**Funktion 3: `import_broker_snapshot_from_json()`**
- Input: JSON-Pfad mit Schema: `{"cash": float, "positions": [{"symbol": str, "qty": float}]}`
- Parsing: `json.load()`
- Validierung: Schema-Check
- Normalisierung: via `normalize_broker_snapshot()`
- Storage: wie oben
- Return: Path zu geschriebenem Snapshot

**Gemeinsame Semantik:**
- Alle Funktionen normalisieren via `normalize_broker_snapshot()` (deterministisch)
- Atomic writes (Windows-safe)
- Deterministisches JSON (sort_keys, indent, newline)
- Fehler: ValueError bei Schema-Verletzungen

### 2) Neues CLI-Script: `scripts/import_broker_snapshot.py`

**Zweck:**
- Standalone-Tool zum Importieren von Broker-Snapshots
- Kann von Ops-Prozessen aufgerufen werden (z.B. nach Broker API Pull)

**CLI-Interface:**
```bash
python scripts/import_broker_snapshot.py \
  --source <csv|json|dict> \
  --input <path> \
  --run-id <run_id> \
  --as-of-date <YYYY-MM-DD> \
  --output-dir <path> \
  [--cash <float>]  # nur bei CSV ohne cash-Spalte
```

**Beispiele:**
```bash
# Import from CSV
python scripts/import_broker_snapshot.py \
  --source csv \
  --input broker_positions_2025-01-15.csv \
  --run-id ops_snapshot_20250115 \
  --as-of-date 2025-01-15 \
  --cash 10000.0

# Import from JSON
python scripts/import_broker_snapshot.py \
  --source json \
  --input broker_snapshot_2025-01-15.json \
  --run-id ops_snapshot_20250115 \
  --as-of-date 2025-01-15
```

**Implementierung:**
- Argument-Parsing
- Source-Detection (CSV/JSON)
- Aufruf entsprechender Import-Funktion
- Logging: Erfolg/Fehler
- Exit-Code: 0 = Erfolg, !=0 = Fehler

### 3) Tests

**Datei: `tests/test_broker_snapshot_importer.py`**

**Tests:**
1. `test_import_from_dict_basic()`: dict -> normalized snapshot
2. `test_import_from_csv_basic()`: CSV -> normalized snapshot
3. `test_import_from_json_basic()`: JSON -> normalized snapshot
4. `test_import_normalizes_positions()`: Verifiziert, dass Normalisierung angewendet wird (trim, sort, filter)
5. `test_import_validates_schema()`: ValueError bei fehlenden required columns
6. `test_import_deterministic_output()`: Gleiche Inputs -> byte-identische Outputs
7. `test_import_atomic_write()`: Keine tmp-Files nach Write (Windows-safe)

**Invariants:**
- Normalisierung wird immer angewendet (trim, sort, filter tiny residuals)
- Output ist deterministisch (sort_keys, mergesort)
- Atomic writes (keine tmp-Files nach Write)
- Schema-Validierung (required columns)

### 4) Dokumentation

**`docs/LEDGER_RECONCILIATION.md` (Ergaenzung):**

Neuer Abschnitt "Broker Snapshot Import":

```markdown
## Broker Snapshot Import

### Overview

External broker snapshots (from broker API, CSV, JSON) can be imported into the
standardized snapshot format using the import tool.

### Import Tool

**CLI:**
```bash
python scripts/import_broker_snapshot.py \
  --source csv \
  --input broker_positions.csv \
  --run-id ops_snapshot_20250115 \
  --as-of-date 2025-01-15 \
  --cash 10000.0
```

**Supported Sources:**
- CSV: Columns `symbol`, `qty` (optional: `cash` as separate value)
- JSON: Schema `{"cash": float, "positions": [{"symbol": str, "qty": float}]}`

**Normalization:**
- All imported snapshots are normalized via `normalize_broker_snapshot()`
- Symbols are trimmed, tiny residuals filtered, positions sorted deterministically
- Output is stored in standard layout: `output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json`

**Use Cases:**
- Import snapshots from broker API (after pulling via Ops script)
- Import historical snapshots from CSV exports
- Create test snapshots for replay/reproducibility
```

**`docs/PROJECT_STRUCTURE.md` (Ergaenzung):**

CLI-Beispiele ergaenzen:

```markdown
**Broker Snapshot Import:**
```bash
# Import from CSV (Ops workflow)
python scripts/import_broker_snapshot.py --source csv --input broker_positions.csv --run-id ops_20250115 --as-of-date 2025-01-15 --cash 10000.0

# Import from JSON
python scripts/import_broker_snapshot.py --source json --input broker_snapshot.json --run-id ops_20250115 --as-of-date 2025-01-15
```
```

## Dateien-Liste

### Neue Dateien:

1. `src/assembled_core/accounting/broker_snapshot_store.py` (Erweiterung)
   - `import_broker_snapshot_from_dict()`
   - `import_broker_snapshot_from_csv()`
   - `import_broker_snapshot_from_json()`

2. `scripts/import_broker_snapshot.py` (neu)
   - CLI-Entry-Point
   - Argument-Parsing
   - Source-Detection
   - Aufruf Import-Funktionen

3. `tests/test_broker_snapshot_importer.py` (neu)
   - 7 Tests (siehe oben)

### Geaenderte Dateien:

1. `docs/LEDGER_RECONCILIATION.md`
   - Abschnitt "Broker Snapshot Import" hinzufuegen

2. `docs/PROJECT_STRUCTURE.md`
   - CLI-Beispiele ergaenzen

## Call-Sites (aktuell)

### Snapshot-Schreiben:
- `src/assembled_core/accounting/ledger_integration.py:264-279` (Paper Snapshots)

### Snapshot-Laden:
- `src/assembled_core/accounting/ledger_integration.py:156-157` (Policy: prefer/require)

### Normalisierung:
- `src/assembled_core/accounting/ledger_integration.py:176-182` (vor Reconciliation)

## Constraints

1. **Determinismus:**
   - `normalize_broker_snapshot()` verwendet `mergesort`
   - JSON: `sort_keys=True`, `indent=2`, trailing newline
   - Atomic writes (temp -> rename/move)

2. **Windows-Safety:**
   - Atomic writes via `tempfile.NamedTemporaryFile` + `Path.replace()`
   - Keine hardcodierten Pfade

3. **Keine neuen Dependencies:**
   - Nur pandas, standard library
   - Keine Layering-Brueche (bleibt in `accounting/`)

4. **Backward Compatibility:**
   - Bestehende `store_broker_snapshot_*()` Funktionen bleiben unveraendert
   - Import-Funktionen sind additive Erweiterungen

## Integrationspunkte

### Ops-Workflow (typisch):

1. **Broker API Pull** (extern):
   - Ops-Script ruft Broker API auf
   - Erzeugt CSV/JSON mit Cash + Positions

2. **Import** (neues Tool):
   ```bash
   python scripts/import_broker_snapshot.py \
     --source csv \
     --input broker_api_output_2025-01-15.csv \
     --run-id ops_snapshot_20250115 \
     --as-of-date 2025-01-15 \
     --cash <from_api>
   ```

3. **Verwendung** (bestehend):
   - Backtest/EOD mit `--broker-snapshot-policy require`
   - Snapshot wird geladen und fuer Reconciliation verwendet

### Test-Workflow:

1. **Erzeuge Test-Snapshot**:
   ```bash
   python scripts/import_broker_snapshot.py \
     --source json \
     --input test_snapshot.json \
     --run-id test_replay \
     --as-of-date 2025-01-15
   ```

2. **Replay mit Snapshot**:
   ```bash
   python scripts/run_backtest_strategy.py \
     --strategy ema \
     --broker-snapshot-policy require \
     --broker-snapshot-run-id test_replay
   ```

## Zusammenfassung

**Neue Funktionen:** 3 Import-Funktionen in `broker_snapshot_store.py`
**Neues Script:** `scripts/import_broker_snapshot.py`
**Tests:** 7 Tests in `test_broker_snapshot_importer.py`
**Doku:** 2 Updates (LEDGER_RECONCILIATION.md, PROJECT_STRUCTURE.md)

**Keine Breaking Changes:** Alle Aenderungen sind additiv, bestehende Funktionen bleiben unveraendert.
