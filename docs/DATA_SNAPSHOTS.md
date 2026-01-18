# Data Snapshots - Assembled Trading AI

**Status:** Verbindlich (muss bei Code-Aenderungen befolgt werden)
**Letzte Aktualisierung:** 2025-01-04

## Zweck

Dieses Dokument definiert die **Data Snapshot Identity Policy** fuer das Assembled Trading AI Projekt. Ein Snapshot-ID identifiziert eindeutig ein spezifisches Preis-Panel und seine Interpretation (Frequenz, Kalender-Normalisierung, etc.) fuer Reproduzierbarkeit.

**Wichtig:** Snapshot-IDs sind deterministisch, order-invariant und robust gegenueber Dtype-Unterschieden.

---

## Snapshot-Modul: src/assembled_core/data/snapshot.py

### Zweck

Das Snapshot-Modul bietet deterministische, reproduzierbare Snapshot-IDs fuer Preis-Panels.

**Key Function:**
- `compute_price_panel_snapshot_id()`: Berechnet stabile Snapshot-ID fuer ein Preis-Panel

---

## Snapshot-ID Semantik

### Was identifiziert eine Snapshot-ID?

Eine Snapshot-ID identifiziert eindeutig:

1. **Die tatsaechlichen Preis-Daten:**
   - Timestamps (normalisiert zu UTC)
   - Symbols (normalisiert: uppercase, trimmed)
   - Close-Preise (normalisiert: float64)
   - Optionale Spalten (volume, open, high, low) - werden NICHT in den Hash einbezogen (nur timestamp, symbol, close)

2. **Die Frequenz-Interpretation:**
   - `freq="1d"` vs `freq="5min"` produzieren unterschiedliche IDs fuer dasselbe Panel

3. **Optionale Source-Metadaten:**
   - Datei-Pfad, Data-Source (yahoo, local, etc.)
   - Diese werden in den Hash einbezogen, wenn `source_meta` bereitgestellt wird

### Was wird NICHT in den Hash einbezogen?

- **Spalten-Reihenfolge:** Die Reihenfolge der Spalten im DataFrame ist irrelevant
- **Zeilen-Reihenfolge:** Die Reihenfolge der Zeilen ist irrelevant (sortiert nach symbol, timestamp)
- **Dtype-Unterschiede:** int64 vs float64 mit denselben Werten produzieren denselben Hash
- **Timezone-Unterschiede:** Alle Timestamps werden zu UTC normalisiert
- **Optionale Spalten:** Nur `timestamp`, `symbol`, `close` werden verwendet

---

## Determinismus-Regeln

### 1. Dedupe-Regel (Duplikate)

**Regel:** Duplikate auf `(symbol, timestamp)` werden deterministisch entfernt.

**Implementierung:**
- `drop_duplicates(subset=["symbol", "timestamp"], keep="last")`
- "last" bedeutet: Bei mehreren Zeilen mit gleichem (symbol, timestamp) wird die letzte Zeile behalten
- Dies ist deterministisch, da die Sortierung vor dem Dedupe stabil ist

**Beispiel:**
```python
# Panel mit Duplikat:
prices = pd.DataFrame({
    "timestamp": [ts1, ts1, ts2],  # ts1 erscheint zweimal
    "symbol": ["AAPL", "AAPL", "MSFT"],
    "close": [150.0, 151.0, 200.0],  # Erste ts1-Zeile wird entfernt
})

# Nach Dedupe: nur (AAPL, ts1, 151.0) und (MSFT, ts2, 200.0) bleiben
```

### 2. Order-Invariance

**Regel:** Dieselben Daten in unterschiedlicher Reihenfolge produzieren dieselbe Snapshot-ID.

**Implementierung:**
- DataFrame wird nach `(symbol, timestamp)` sortiert vor dem Hashing
- Spalten werden in festgelegter Reihenfolge (`timestamp`, `symbol`, `close`) verwendet

**Beispiel:**
```python
# Beide produzieren dieselbe Snapshot-ID:
prices1 = pd.DataFrame({
    "timestamp": [ts1, ts2, ts3],
    "symbol": ["AAPL", "MSFT", "GOOGL"],
    "close": [150.0, 200.0, 100.0],
})

prices2 = pd.DataFrame({
    "timestamp": [ts2, ts1, ts3],
    "symbol": ["MSFT", "AAPL", "GOOGL"],
    "close": [200.0, 150.0, 100.0],
})

# prices1 und prices2 haben dieselbe Snapshot-ID (nach Sortierung)
```

### 3. Dtype-Normalisierung

**Regel:** Unterschiedliche Dtypes mit denselben Werten produzieren dieselbe Snapshot-ID.

**Implementierung:**
- Numerische Spalten werden zu `float64` konvertiert
- `int64` und `float64` mit denselben Werten produzieren denselben Hash

**Beispiel:**
```python
# Beide produzieren dieselbe Snapshot-ID:
prices1 = pd.DataFrame({"close": [150, 151, 152]})  # int64
prices2 = pd.DataFrame({"close": [150.0, 151.0, 152.0]})  # float64
```

### 4. Timezone-Normalisierung

**Regel:** Alle Timestamps werden zu UTC normalisiert vor dem Hashing.

**Implementierung:**
- Naive Timestamps werden als UTC interpretiert
- Timezone-aware Timestamps werden zu UTC konvertiert

**Beispiel:**
```python
# Beide produzieren dieselbe Snapshot-ID:
prices1 = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, tz="UTC")
})
prices2 = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, tz="America/New_York").tz_convert("UTC")
})
```

### 5. NaN/Inf Policy

**Regel:** NaN und Inf werden konsistent behandelt.

**Implementierung:**
- NaN wird als `<NA>` in CSV-Repraesentation geschrieben
- Inf wird zu einem sentinel-Wert (1e308 / -1e308) normalisiert, da CSV Inf nicht darstellen kann
- NaN/Inf-Positionen sind Teil der Daten und werden stabil abgebildet

### 6. Robustheit

**Regel:** Fehlende Spalten und leere DataFrames werden konsistent behandelt.

**Implementierung:**
- Fehlende required columns fuehren zu `ValueError`
- Leere DataFrames produzieren einen definierten Hash (SHA256 von leerem String)

---

## API-Referenz

### compute_price_panel_snapshot_id()

```python
def compute_price_panel_snapshot_id(
    prices: pd.DataFrame,
    *,
    freq: str,
    source_meta: dict[str, Any] | None = None,
) -> str:
    """Compute stable, deterministic snapshot ID for a price panel.

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, close, ... (volume optional)
        freq: Frequency string ("1d" or "5min")
        source_meta: Optional metadata dict (e.g., {"source": "yahoo", "file": "path/to/file.parquet"})

    Returns:
        SHA256 hash as hex string (64 characters)
    """
```

**Beispiel:**
```python
from src.assembled_core.data.snapshot import compute_price_panel_snapshot_id

prices = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
    "symbol": ["AAPL"] * 10,
    "close": [150.0] * 10,
})

# Snapshot-ID ohne Source-Meta
snapshot_id = compute_price_panel_snapshot_id(prices, freq="1d")
print(f"Snapshot ID: {snapshot_id}")

# Snapshot-ID mit Source-Meta
snapshot_id_with_meta = compute_price_panel_snapshot_id(
    prices,
    freq="1d",
    source_meta={"source": "yahoo", "file": "data/raw/eod_2024.parquet"},
)
print(f"Snapshot ID (with meta): {snapshot_id_with_meta}")
```

---

## Panel Store Integration (D3)

### Append/Update Logik

**Modul:** `src/assembled_core/data/panel_store.py`

**Funktion:** `store_price_panel_parquet(df, mode="append")`

**Semantik:**
- `mode="replace"`: Full recompute (ersetzt gesamtes Panel)
- `mode="append"`: Append new data to existing panel (falls vorhanden)

**Dedupe-Regel (konsistent mit Snapshot-ID):**
- Gleiche Regel wie in Snapshot-ID: `drop_duplicates(subset=["symbol", "timestamp"], keep="last")`
- Garantiert konsistente Semantik zwischen Panel-Store und Snapshot-ID

**Workflow:**
1. Load existing panel (falls vorhanden)
2. Concat: `pd.concat([existing, new])`
3. Dedupe: `drop_duplicates(keep="last")`
4. Sort: Nach (symbol, timestamp)
5. Atomic rewrite: Temp-Datei -> rename

**Beispiel:**
```python
from src.assembled_core.data.panel_store import store_price_panel_parquet

# Initial panel (D1..D10)
store_price_panel_parquet(df_initial, freq="1d", mode="replace")

# Append new day (D11)
store_price_panel_parquet(df_new_day, freq="1d", mode="append")

# Result: Panel enthaelt D1..D11 (nach Dedupe und Sortierung)
```

---

## Integration (D4)

### Run Manifests

**Datei:** `output/run_manifest_{freq}.json`

**Feld:** `data_snapshot_id`

**Beispiel:**
```json
{
  "run_id": "abc123",
  "freq": "1d",
  "data_snapshot_id": "a1b2c3d4e5f6...",
  "started_at": "2024-01-04T10:00:00Z",
  ...
}
```

### Batch Summary

**Datei:** `output/batch_summary.csv`

**Spalte:** `data_snapshot_id`

**Beispiel:**
```csv
run_id,strategy,freq,data_snapshot_id,final_pf,sharpe,...
run_001,trend_baseline,1d,a1b2c3d4e5f6...,1.25,0.85,...
run_002,trend_baseline,1d,a1b2c3d4e5f6...,1.30,0.90,...
```

### QC Reports

**Datei:** `output/qc_report.json`

**Feld:** `data_snapshot_id` (optional, fuer Audit-Trail)

---

## Referenzen

- `src/assembled_core/data/snapshot.py` - Snapshot-Modul
- `docs/CONTRACTS.md` - Data Contracts (insbesondere Preis-Panel Contract)
- `docs/TIME_AND_CALENDAR.md` - TZ-Policy (UTC-normalisierung)

---

## Aenderungen an der Policy

**Wichtig:** Die Snapshot-ID-Policy ist **verbindlich** und sollte nicht ohne triftigen Grund geaendert werden.

**Prozess fuer Aenderungen:**
1. Aenderung in `docs/DATA_SNAPSHOTS.md` dokumentieren
2. Snapshot-Modul anpassen
3. Tests aktualisieren
4. Integrationstests ausfuehren (Manifests, Batch-Summary)
