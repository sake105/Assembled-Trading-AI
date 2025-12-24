# Factor Store

**Last Updated:** 2025-12-23  
**Status:** ✅ Implemented (P2)

---

## Overview

Der **Factor Store** ist ein Caching-System für berechnete Faktoren und Features. Er speichert technische Indikatoren und andere Features in einer strukturierten Form, um teure Neuberechnungen zu vermeiden und Performance zu verbessern.

### Vorteile

- **Performance**: Vermeidet wiederholte Berechnungen teurer Features (z.B. 200-Tage Moving Averages)
- **Konsistenz**: Gleiche Features werden in EOD-Pipelines, Backtests und ML-Workflows verwendet
- **Reproduzierbarkeit**: Gespeicherte Faktoren sind deterministisch und können versioniert werden
- **PIT-Safety**: Point-in-Time-Filterung verhindert Look-Ahead-Bias

---

## Directory Structure

Der Factor Store verwendet eine hierarchische Ordnerstruktur:

```
data/factors/
├── core_ta/                    # Factor Group: Core Technical Analysis
│   ├── 1d/                     # Frequency: Daily
│   │   ├── universe_watchlist/ # Universe Key (aus Symbol-Liste generiert)
│   │   │   ├── year=2023.parquet
│   │   │   ├── year=2024.parquet
│   │   │   └── _metadata.json  # Metadaten (Schema, Parameter, etc.)
│   │   └── universe_sp500/
│   │       └── year=2024.parquet
│   └── 5min/                   # Frequency: 5-Minuten
│       └── universe_watchlist/
│           └── year=2024.parquet
├── vol_liquidity/              # Factor Group: Volatility & Liquidity
│   └── 1d/
│       └── universe_watchlist/
│           └── year=2024.parquet
└── alt_insider/                # Factor Group: Alternative Data (z.B. Insider)
    └── 1d/
        └── universe_watchlist/
            └── year=2024.parquet
```

### Komponenten

1. **Factor Group**: Kategorie der Faktoren (`core_ta`, `vol_liquidity`, `alt_insider`, etc.)
2. **Frequency**: Trading-Frequenz (`1d`, `5min`)
3. **Universe Key**: Deterministischer Hash/Name aus der Symbol-Liste (z.B. `universe_watchlist`, `universe_sp500`)
4. **Year Partitioning**: Jahres-Partitionierung für große Datensätze (`year=2023.parquet`, `year=2024.parquet`)

### Default Location

Der Standard-Pfad ist `data/factors/` im Repository-Root. Dieser kann über `--factor-store-root` überschrieben werden.

---

## Usage

### EOD Pipeline (`run_daily.py`)

Verwende den Factor Store in der EOD-Pipeline, um Features zu cachen:

```bash
# Mit Factor Store (aktiviert Caching)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store

# Mit custom Factor Store Root
python scripts/run_daily.py --date 2024-12-31 --use-factor-store \
  --factor-store-root /custom/path/to/factors

# Mit custom Factor Group
python scripts/run_daily.py --date 2024-12-31 --use-factor-store \
  --factor-group core_ta
```

**CLI Flags:**
- `--use-factor-store`: Aktiviert Factor Store Caching
- `--factor-store-root`: Optional: Custom Root-Verzeichnis (default: `data/factors/`)
- `--factor-group`: Optional: Factor Group Name (default: `core_ta`)

### Backtest (`run_backtest_strategy.py`)

Verwende den Factor Store in Backtests:

```bash
# Mit Factor Store
python scripts/run_backtest_strategy.py \
  --freq 1d \
  --use-factor-store \
  --start-date 2020-01-01 \
  --end-date 2024-12-31

# Mit custom Factor Group
python scripts/run_backtest_strategy.py \
  --freq 1d \
  --use-factor-store \
  --factor-group core_ta \
  --start-date 2020-01-01 \
  --end-date 2024-12-31
```

**CLI Flags:**
- `--use-factor-store`: Aktiviert Factor Store Caching
- `--factor-store-root`: Optional: Custom Root-Verzeichnis
- `--factor-group`: Optional: Factor Group Name (default: `core_ta`)

### Profile Jobs (`profile_jobs.py`)

Verwende den Factor Store für Performance-Messungen (warm vs. cold cache):

```bash
# EOD_SMALL mit warm cache (cold build, dann warm load)
python scripts/profile_jobs.py \
  --job EOD_SMALL \
  --warm-cache \
  --use-factor-store

# BACKTEST_MEDIUM mit Factor Store
python scripts/profile_jobs.py \
  --job BACKTEST_MEDIUM \
  --use-factor-store
```

**CLI Flags:**
- `--warm-cache`: Für EOD_SMALL: 2 Läufe (cold build, dann warm load)
- `--use-factor-store`: Aktiviert Factor Store Caching

---

## Programmierschnittstelle (API)

### High-Level API: `build_or_load_factors()`

Die einfachste Art, den Factor Store zu nutzen:

```python
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.features.ta_features import add_all_features
import pandas as pd

# Lade Preise
prices = pd.read_parquet("data/raw/1d/prices.parquet")

# Build or Load Factors (prüft Cache, baut falls nötig)
factors = build_or_load_factors(
    prices=prices,
    factor_group="core_ta",
    freq="1d",
    builder_fn=add_all_features,
    builder_kwargs={
        "ma_windows": (20, 50, 200),
        "atr_window": 14,
        "rsi_window": 14,
        "include_rsi": True,
    },
)

# Faktoren enthalten: timestamp, symbol, date, log_return, ma_20, ma_50, ma_200, atr_14, rsi_14
```

### Low-Level API

Direkter Zugriff auf Load/Store-Funktionen:

```python
from src.assembled_core.data.factor_store import (
    load_factors,
    store_factors,
    compute_universe_key,
)
from pathlib import Path

# Compute Universe Key
symbols = ["AAPL", "MSFT", "GOOGL"]
universe_key = compute_universe_key(symbols)
# Result: "universe_AAPL_GOOGL_MSFT_<hash>"

# Load Factors (mit PIT-Filterung)
factors = load_factors(
    factor_group="core_ta",
    freq="1d",
    universe_key=universe_key,
    start_date="2024-01-01",
    end_date="2024-12-31",
    as_of="2024-06-30",  # PIT-safe: nur Daten bis 2024-06-30
    factors_root=Path("data/factors"),
)

# Store Factors
store_factors(
    df=factors_df,
    factor_group="core_ta",
    freq="1d",
    universe_key=universe_key,
    mode="append",  # oder "overwrite"
    factors_root=Path("data/factors"),
    metadata={
        "builder_fn": "add_all_features",
        "builder_kwargs": {"ma_windows": [20, 50, 200]},
    },
)
```

---

## Point-in-Time (PIT) Safety

Der Factor Store unterstützt Point-in-Time-Filterung, um Look-Ahead-Bias zu vermeiden:

```python
# Backtest für Datum 2024-06-30
# Sollte nur Features verwenden, die bis 2024-06-30 verfügbar waren

factors = build_or_load_factors(
    prices=prices,
    factor_group="core_ta",
    freq="1d",
    as_of="2024-06-30",  # PIT-Cutoff
    builder_fn=add_all_features,
)

# Alle Timestamps in `factors` sind <= 2024-06-30
```

**Wichtig:**
- `as_of`: Alle geladenen Faktoren haben `timestamp <= as_of`
- `start_date` / `end_date`: Definiert den gewünschten Datumsbereich
- Kombiniert: `factors` enthält nur Daten im Bereich `[start_date, min(end_date, as_of)]`

---

## Cache Management

### Cache löschen

Um einen spezifischen Factor-Cache zu löschen:

```bash
# Lösche alle Faktoren für core_ta/1d/universe_watchlist
rm -rf data/factors/core_ta/1d/universe_watchlist
```

Oder für eine komplette Factor Group:

```bash
# Lösche alle core_ta Faktoren
rm -rf data/factors/core_ta
```

### Cache neu aufbauen

Um den Cache zu überschreiben (Force Rebuild):

```python
factors = build_or_load_factors(
    prices=prices,
    factor_group="core_ta",
    freq="1d",
    force_rebuild=True,  # Überschreibt Cache, auch wenn vorhanden
    builder_fn=add_all_features,
)
```

Oder über CLI (indirekt, durch Löschen des Cache-Verzeichnisses):

```bash
# 1. Lösche Cache
rm -rf data/factors/core_ta/1d/universe_watchlist

# 2. Führe Pipeline/Backtest aus (baut Cache neu auf)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store
```

---

## Troubleshooting

### Cache Hit/Miss Logs

Der Factor Store loggt Cache-Hits und Cache-Misses:

```
[cache_hit] Factors loaded from store: core_ta/1d/universe_watchlist, date_range=[2024-01-01, 2024-12-31], rows=50000
[cache_miss] Computing factors: core_ta/1d/universe_watchlist, date_range=[2024-01-01, 2024-12-31]
[cache_stored] Factors stored to cache: 50000 rows
```

**Cache Hit**: Faktoren wurden erfolgreich aus dem Store geladen  
**Cache Miss**: Faktoren müssen neu berechnet werden  
**Cache Partial**: Cache existiert, deckt aber nicht den vollen Datumsbereich ab

### Häufige Probleme

#### 1. "Cache Miss" obwohl Cache existiert

**Ursache:** Cache deckt nicht den vollen Datumsbereich ab  
**Lösung:** Prüfe die Dateien im Cache-Verzeichnis (`data/factors/<group>/<freq>/<universe>/`). Möglicherweise fehlen Jahre oder der Datumsbereich ist unvollständig.

#### 2. "Universe Key mismatch"

**Ursache:** Symbol-Liste hat sich geändert (neue/entfernte Symbole)  
**Lösung:** Neue Universe erstellt automatisch neuen Cache-Pfad. Alte Cache-Daten bleiben erhalten, werden aber nicht verwendet.

#### 3. "Factors stored, but load returns empty"

**Ursache:** PIT-Filterung schneidet alle Daten ab (`as_of` zu früh)  
**Lösung:** Prüfe `as_of`, `start_date`, `end_date` Parameter. Stelle sicher, dass `as_of >= start_date`.

#### 4. Performance: Cache ist langsamer als Neuberechnung

**Ursache:** I/O-Overhead für große Parquet-Dateien oder Netzwerk-Speicher  
**Lösung:** Normalerweise sollte Cache-Hit deutlich schneller sein. Falls nicht, prüfe:
- Parquet-Dateien sind nicht fragmentiert (z.B. durch `mode="append"` in vielen kleinen Schritten)
- Storage ist lokal (nicht Netzwerk)
- Dateien sind nicht zu groß (Jahres-Partitionierung hilft)

---

## Metadaten

Jedes Universe-Verzeichnis enthält `_metadata.json` mit:

```json
{
  "factor_group": "core_ta",
  "freq": "1d",
  "universe_key": "universe_watchlist",
  "symbols": ["AAPL", "MSFT", "GOOGL", ...],
  "computed_at": "2024-12-23T10:00:00Z",
  "computation_params": {
    "ma_windows": [20, 50, 200],
    "atr_window": 14,
    "rsi_window": 14
  },
  "factor_columns": ["log_return", "ma_20", "ma_50", "ma_200", "atr_14", "rsi_14"],
  "date_range": {
    "start": "2023-01-01",
    "end": "2024-12-31"
  }
}
```

**Verwendung:**
- Cache-Invalidierung (wenn Parameter oder Daten-Hash sich ändern)
- Discovery (welche Faktoren sind verfügbar?)
- Debugging (wann wurden Faktoren berechnet?)

---

## Best Practices

1. **Jahres-Partitionierung nutzen**: Große Datensätze werden automatisch nach Jahren partitioniert (z.B. `year=2023.parquet`, `year=2024.parquet`)

2. **PIT-Safety beachten**: Verwende immer `as_of` in Backtests, um Look-Ahead-Bias zu vermeiden

3. **Factor Groups konsistent verwenden**: Verwende die gleiche Factor Group für ähnliche Feature-Sets (z.B. `core_ta` für Standard-TA-Features)

4. **Cache aufräumen**: Lösche alte/unvollständige Caches regelmäßig, um Platz zu sparen

5. **Metadata prüfen**: Prüfe `_metadata.json` bei Problemen, um zu sehen, welche Parameter verwendet wurden

---

## Related Documentation

- **Design Document**: [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md) – Detailliertes Design und API-Spezifikation
- **Performance Profiling**: [Performance Profiling P1 Design](PERFORMANCE_PROFILING_P1_DESIGN.md) – Performance-Messungen mit warm/cold Cache
- **Advanced Analytics**: [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) – P2: Factor Store & Data Layout ✅

---

## Examples

### Beispiel 1: EOD Pipeline mit Factor Store

```bash
# Erster Lauf: Cache Miss (baut Cache auf)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store
# Output: [cache_miss] Computing factors: core_ta/1d/universe_watchlist, ...

# Zweiter Lauf: Cache Hit (lädt aus Cache)
python scripts/run_daily.py --date 2024-12-31 --use-factor-store
# Output: [cache_hit] Factors loaded from store: core_ta/1d/universe_watchlist, ...
```

### Beispiel 2: Backtest mit Factor Store

```bash
# Backtest mit Factor Store (10 Jahre Daten)
python scripts/run_backtest_strategy.py \
  --freq 1d \
  --use-factor-store \
  --start-date 2015-01-01 \
  --end-date 2024-12-31 \
  --strategy trend_baseline

# Erster Lauf: Cache Miss für 2015-2024
# Zweiter Lauf: Cache Hit (wenn gleicher Datumsbereich)
```

### Beispiel 3: Python API

```python
from src.assembled_core.features.factor_store_integration import build_or_load_factors
from src.assembled_core.features.ta_features import add_all_features
import pandas as pd

# Lade Preise
prices = pd.read_parquet("data/raw/1d/prices.parquet")

# Build or Load (automatisches Caching)
factors = build_or_load_factors(
    prices=prices,
    factor_group="core_ta",
    freq="1d",
    builder_fn=add_all_features,
    builder_kwargs={"ma_windows": (20, 50, 200), "atr_window": 14},
)

# Verwende Faktoren für Signale/Backtest
# factors enthält: timestamp, symbol, log_return, ma_20, ma_50, ma_200, atr_14, ...
```

---

**Status:** ✅ Implemented (P2)  
**Related:** [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) – P2: Factor Store & Data Layout ✅

