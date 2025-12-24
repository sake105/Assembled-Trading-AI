# Backtest Engine Optimization (P3)

## Übersicht

Phase 3 (P3) optimiert den Backtest-Engine durch Vektorisierung und optionale Numba-Beschleunigung. Die Optimierungen verbessern die Performance ohne die Logik zu ändern.

**Status:** ✅ Abgeschlossen

## Optimierungen

### 1. Vektorisierung von Loops

**Per-Order Loops:**
- `_simulate_fills_per_order`: Loop über Orders → NumPy-Vektorisierung
- Fill-Preise, Fees, Notional werden vektoriell berechnet
- Edge Cases: buy/sell sign, NaNs, qty=0

**Per-Symbol Loops:**
- `_update_equity_mark_to_market`: Loop über Symbole → NumPy-Vektorisierung
- Equity-Berechnung: `equity = cash + sum(position_shares * price)` vektoriell

**Positions-Updates:**
- `generate_orders_from_targets`: `.apply()` → `np.where()` (vektoriell)
- Alignment/Sortierung sichergestellt für stabile Symbol-Reihenfolge

### 2. Optionale Numba-Beschleunigung

**Numba-Kernels** (`src/assembled_core/qa/numba_kernels.py`):
- `compute_mark_to_market_numba`: Mark-to-Market mit `@njit`
- `aggregate_position_deltas_numba`: Positions-Deltas aggregieren
- Automatischer Fallback auf pure NumPy wenn Numba nicht verfügbar

**Aktivierung:**
- Standard: `use_numba=True` (automatisch aktiviert wenn Numba installiert)
- Deaktivierung: `use_numba=False` (für Debugging oder Vergleich)

## Speedup-Messung

### Benchmark-Harness

Der Benchmark-Harness (`scripts/benchmark_backtest.py`) misst die Performance:

```bash
# BACKTEST_MEDIUM 3x ausführen, Median-Runtime loggen
python scripts/benchmark_backtest.py --job BACKTEST_MEDIUM --runs 3
```

**Output:**
- `output/profiles/benchmark/benchmark.json` mit:
  - `median_runtime`: Median-Runtime in Sekunden
  - `mean_runtime`: Durchschnittliche Runtime
  - `min_runtime`, `max_runtime`: Min/Max Runtime
  - `runtimes`: Liste aller Runtimes
  - `timestamp`: Zeitstempel des Benchmarks

### Baseline vs. Optimized Profiling

**Baseline (vor Optimierung):**
```bash
# Legacy-Implementierung verwenden (nur für Vergleich)
# In Tests: test_backtest_regression.py vergleicht alte vs. neue Implementierung
```

**Optimized (nach Optimierung):**
```bash
# Standard: Vektorisierung + optional Numba
python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler cprofile
```

**Mit Numba (wenn installiert):**
```bash
# Numba wird automatisch verwendet wenn verfügbar
python scripts/benchmark_backtest.py --job BACKTEST_MEDIUM --runs 3
```

**Ohne Numba (Fallback):**
```bash
# Pure NumPy-Fallback (automatisch wenn Numba nicht installiert)
# Gleiche API, nur langsamer
```

### Report Generation

**Performance-Profile:**
```bash
# cProfile Report generieren
python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler cprofile --top-n 50

# Output: output/profiles/BACKTEST_MEDIUM/<timestamp>/
#   - profile_BACKTEST_MEDIUM.prof
#   - profile_BACKTEST_MEDIUM_stats.txt
#   - profile_summary.log
```

**Benchmark-Report:**
```bash
# Benchmark ausführen
python scripts/benchmark_backtest.py --job BACKTEST_MEDIUM --runs 3

# Output: output/profiles/benchmark/benchmark.json
```

**Timings-Report (Step-Breakdown):**
```bash
# Backtest mit Timings
python scripts/run_backtest_strategy.py --freq 1d --enable-timings

# Output: output/timings.json (Step-Breakdown: load_prices, build_features, signals, etc.)
```

## Numba aktivieren

### Installation

```bash
pip install numba
```

### Verifizierung

```python
from src.assembled_core.qa.numba_kernels import NUMBA_AVAILABLE
print(f"Numba available: {NUMBA_AVAILABLE}")
```

### Automatische Aktivierung

Numba wird automatisch verwendet wenn:
1. `numba` installiert ist
2. `use_numba=True` (Standard)
3. Funktion unterstützt Numba (z.B. `_update_equity_mark_to_market`)

### Manuelle Deaktivierung

```python
from src.assembled_core.pipeline.backtest import _update_equity_mark_to_market

# Deaktivieren für Debugging
equity = _update_equity_mark_to_market(
    timestamp=ts,
    cash=cash,
    positions=positions,
    price_pivot=px,
    symbols=symbols,
    use_numba=False,  # Force pure NumPy
)
```

## Messartefakte

### Benchmark-Output

**Datei:** `output/profiles/benchmark/benchmark.json`

```json
{
  "job_name": "BACKTEST_MEDIUM",
  "num_runs": 3,
  "runtimes": [45.2, 44.8, 45.5],
  "median_runtime": 45.2,
  "mean_runtime": 45.17,
  "min_runtime": 44.8,
  "max_runtime": 45.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Profiling-Output

**Dateien:**
- `output/profiles/BACKTEST_MEDIUM/<timestamp>/profile_BACKTEST_MEDIUM.prof`
- `output/profiles/BACKTEST_MEDIUM/<timestamp>/profile_BACKTEST_MEDIUM_stats.txt`
- `output/profiles/BACKTEST_MEDIUM/<timestamp>/profile_summary.log`

### Timings-Output

**Datei:** `output/timings.json`

```json
{
  "load_prices": 2.5,
  "build_features": 15.3,
  "signals": 1.2,
  "positions": 0.8,
  "fills": 0.5,
  "equity": 0.3,
  "metrics": 0.1,
  "total": 20.7
}
```

## Regression-Tests

**Test-Suite:** `tests/test_backtest_regression.py`

Vergleicht optimierte vs. Legacy-Implementierung:
- Equity-Kurven: `rtol=1e-9, atol=1e-9`
- Metriken: `rtol=1e-6, atol=1e-6`
- Order-Counts: exakt

**Golden Mini-Backtest:** `tests/conftest.py::golden_mini_backtest_data`
- 3 Symbole, 10 Tage, deterministische Signale
- Reproduzierbare Tests

## Verwandte Dokumentation

- [Performance Profiling](docs/PERFORMANCE_PROFILING.md) (falls vorhanden)
- [Factor Store P2](docs/FACTOR_STORE_P2_DESIGN.md)
- [Backtest Engine](src/assembled_core/qa/backtest_engine.py)

## Commands Quick Reference

```bash
# Benchmark ausführen
python scripts/benchmark_backtest.py --job BACKTEST_MEDIUM --runs 3

# Profiling mit cProfile
python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler cprofile --top-n 50

# Profiling mit pyinstrument
python scripts/profile_jobs.py --job BACKTEST_MEDIUM --profiler pyinstrument

# Backtest mit Timings
python scripts/run_backtest_strategy.py --freq 1d --enable-timings

# Regression-Tests
pytest tests/test_backtest_regression.py -v
```

