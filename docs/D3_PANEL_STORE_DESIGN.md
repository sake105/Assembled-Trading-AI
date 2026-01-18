# D3 Panel Store Design - Assembled Trading AI

**Status:** Verbindlich (muss bei Code-Aenderungen befolgt werden)
**Letzte Aktualisierung:** 2025-01-04

## Zweck

Dieses Dokument definiert das Design fuer die "Local-first Snapshot Pipeline" (Sprint 3 / D3). Ziel ist es, eine einzige "blessed" Panel-Quelle zu definieren, die deterministisch und reproduzierbar ist.

---

## Analyse: Aktuelle Preis-Lade-Stellen

### Daily Entry Points

1. **`scripts/run_daily.py`** - `run_daily_eod()`
   - Laedt Preise via `price_file` Parameter oder Fallback
   - Funktion: `load_eod_prices()` aus `src/assembled_core/data/prices_ingest.py`

2. **`src/assembled_core/pipeline/orchestrator.py`** - `run_eod_pipeline()`
   - Laedt Preise via `get_price_data_source()` (Zeile 323-335)
   - Provider fetch (yahoo) oder local file load
   - Schreibt Cache: `output/aggregates/{freq}_live_cache.parquet`

### Backtest Entry Points

1. **`scripts/run_backtest_strategy.py`** - `load_price_data()`
   - Laedt Preise via `data_source` oder `price_file`
   - Funktion: `get_price_data_source()` aus `src/assembled_core/data/data_source.py`

2. **`src/assembled_core/pipeline/orchestrator.py`** - `run_backtest_step()`
   - Laedt Preise via `load_prices()` oder `load_prices_with_fallback()`
   - Funktionen aus `src/assembled_core/pipeline/io.py`

### Aktuelle Pfade

- **Rohdaten:** `data/raw/1min/*.parquet`
- **Aggregierte:** `output/aggregates/daily.parquet`, `output/aggregates/5min.parquet`
- **Cache:** `output/aggregates/{freq}_live_cache.parquet`

---

## Design: "Blessed" Panel-Quelle

### Pfad-Konvention

**Minimal (ein Panel pro freq):**
```
data/panels/{freq}/panel.parquet
```

**Erweitert (ein Panel pro freq+universe):**
```
data/panels/{freq}/{universe}/panel.parquet
```

**Beispiele:**
- `data/panels/1d/panel.parquet` (alle Symbole)
- `data/panels/1d/ai_tech/panel.parquet` (nur AI-Tech Universe)
- `data/panels/5min/panel.parquet` (alle Symbole, 5min)

### Semantik

- **"Cleaned" Panel:** Bereinigtes Panel (nach QC, Dedupe, TZ-Normalisierung)
- **Deterministisch:** Gleiche Input-Daten => gleiche Panel-Datei
- **Atomic Write:** Panel wird atomar geschrieben (keine teilweise geschriebene Dateien)
- **Contract-konform:** Erfuellt Preise/Panel Contract (docs/CONTRACTS.md)

### Workflow

1. **Daily Ingest:**
   - Provider fetch (yahoo) oder local raw load
   - Cleaning: QC, Dedupe, TZ-Normalisierung
   - Store: `store_price_panel_parquet(df, freq=freq, universe=universe)`
   - Panel wird in `data/panels/{freq}/{universe}/panel.parquet` geschrieben

2. **Backtest:**
   - Load: `load_price_panel_parquet(freq=freq, universe=universe)`
   - Backtests lesen NUR aus cleaned panels
   - Kein Provider fetch in Backtests (Hard Gate)

---

## Implementierung: panel_store.py

### Modul: `src/assembled_core/data/panel_store.py`

**Layering:**
- `data/` Layer (Bottom Layer)
- Importiert: Nichts (nur pandas, pathlib, tempfile)
- Exportiert: Funktionen zum Laden/Speichern von Panels

### Funktionen

#### `panel_path(freq: str, universe: str | None = None, root: Path | None = None) -> Path`

Berechnet deterministischen Pfad zum Panel.

**Args:**
- `freq`: Frequency string ("1d" or "5min")
- `universe`: Optional universe name (z.B. "ai_tech"). Wenn None, verwendet "default"
- `root`: Root-Verzeichnis (default: `data/` relativ zum Repo-Root)

**Returns:**
- `Path` zum Panel-File

**Beispiel:**
```python
path = panel_path(freq="1d", universe="ai_tech")
# -> Path("data/panels/1d/ai_tech/panel.parquet")
```

#### `load_price_panel_parquet(freq: str, universe: str | None = None, root: Path | None = None) -> pd.DataFrame`

Laedt cleaned Panel aus Parquet-Datei.

**Args:**
- `freq`: Frequency string ("1d" or "5min")
- `universe`: Optional universe name. Wenn None, verwendet "default"
- `root`: Root-Verzeichnis (default: `data/` relativ zum Repo-Root)

**Returns:**
- DataFrame mit Preise/Panel Contract (timestamp, symbol, close, ...)
- Sortiert nach (symbol, timestamp)
- UTC-normalisiert

**Raises:**
- `FileNotFoundError`: Wenn Panel nicht existiert

#### `store_price_panel_parquet(df: pd.DataFrame, freq: str, universe: str | None = None, root: Path | None = None, mode: str = "replace") -> Path`

Speichert cleaned Panel als Parquet (atomic write).

**Args:**
- `df`: DataFrame mit Preise/Panel Contract
- `freq`: Frequency string ("1d" or "5min")
- `universe`: Optional universe name. Wenn None, verwendet "default"
- `root`: Root-Verzeichnis (default: `data/` relativ zum Repo-Root)
- `mode`: Storage mode ("replace" oder "append"). Default: "replace"

**Modes:**
- `"replace"`: Ersetzt gesamtes Panel (full recompute, default)
- `"append"`: Fuegt neue Daten zu existierendem Panel hinzu (falls vorhanden)

**Append-Mode Semantik:**
- Laedt existierendes Panel (falls vorhanden)
- Concat: `pd.concat([existing, new], ignore_index=True)`
- Dedupe: `drop_duplicates(subset=["symbol", "timestamp"], keep="last")` (konsistent mit Snapshot-ID)
- Sort: Nach (symbol, timestamp) sortieren
- Atomic rewrite: Temp-Datei -> rename

**Dedupe-Regel (konsistent mit Snapshot-ID):**
- Gleiche Regel wie in `compute_price_panel_snapshot_id()`: `keep="last"`
- Garantiert konsistente Semantik zwischen Panel-Store und Snapshot-ID

**Returns:**
- `Path` zur geschriebenen Panel-Datei

**Atomic Write:**
- Schreibt zuerst in temp-Datei
- Dann atomic rename (Windows: move, Unix: rename)
- Verhindert teilweise geschriebene Dateien

**Validierung:**
- Prueft required columns (timestamp, symbol, close)
- Prueft TZ-Policy (UTC)
- Prueft Sortierung (symbol, timestamp)

#### `panel_exists(freq: str, universe: str | None = None, root: Path | None = None) -> bool`

Prueft ob Panel existiert.

**Args:**
- `freq`: Frequency string ("1d" or "5min")
- `universe`: Optional universe name. Wenn None, verwendet "default"
- `root`: Root-Verzeichnis (default: `data/` relativ zum Repo-Root)

**Returns:**
- `True` wenn Panel existiert, sonst `False`

---

## Integration (spaeter, nicht in D3)

### Daily Ingest

```python
# In scripts/run_daily.py oder orchestrator.py
from src.assembled_core.data.panel_store import store_price_panel_parquet

# Nach Cleaning:
store_price_panel_parquet(
    df=cleaned_prices,
    freq="1d",
    universe="ai_tech",  # oder None fuer "default"
)
```

### Backtest

```python
# In scripts/run_backtest_strategy.py
from src.assembled_core.data.panel_store import load_price_panel_parquet

# Laedt cleaned Panel:
prices = load_price_panel_parquet(
    freq="1d",
    universe="ai_tech",  # oder None fuer "default"
)
```

---

## Referenzen

- `docs/ARCHITECTURE_LAYERING.md` - Layering-Regeln (data/ Layer)
- `docs/CONTRACTS.md` - Preise/Panel Contract
- `src/assembled_core/data/snapshot.py` - Snapshot-ID (D3)
