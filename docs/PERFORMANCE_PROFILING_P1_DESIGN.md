# Performance Profiling & Baseline (P1)

## 1. Overview & Ziele

Dieses Modul definiert, **wie** wir Performance in `assembled-trading-ai` systematisch messen:

- Wo verbringt das System Zeit?
- Welche Jobs sind die größten Hotspots (Backtests, Factor-Building, ML, Risk)?
- Wie vergleichen wir Verbesserungen (P2–P4, Backtester-Refactor, Factor-Store) objektiv?

**Scope P1:**

- Definiert ein einheitliches Profiling-Setup (Scripts + Logging).
- Liefert reproduzierbare Messungen für typische Workloads.
- Keine Optimierungen – nur Mess-Infrastruktur und Baseline.

P1 ist die Grundlage für alle späteren Performance-Tasks (Factor-Store, Batch-Runner, Backtester-Optimierung).

---

## 2. Use Cases & Ziel-Jobs

Wir konzentrieren uns auf drei Haupt-Workloads:

1. **Backtest-Job (BASIC_BACKTEST)**
   - Typisch: `scripts/cli.py run_backtest ...`
   - Beispiel: Multi-Factor-Strategie auf mittlerem Universe & Zeitraum.

2. **Factor-/ML-Job (FACTOR_ML_JOB)**
   - Typisch: `export_factor_panel_for_ml.py` + `ml_validate_factors` oder Model-Zoo.
   - Ziel: Faktoren bauen + ML-Validation auf einem Panel.

3. **EOD-/Research-Playbook-Job (PLAYBOOK_JOB)**
   - Typisch: `ai_tech_multifactor_mlalpha_regime_playbook.py main()`
   - Ziel: End-to-End: Factor → ML → Backtests → Risk Reports.

Jeder dieser Jobs soll über das Profiling-Script (P1) gemessen werden können.

---

## 3. Tools & Methoden

### 3.1. High-Level Profiling (End-to-End)

- **Werkzeug:** `time.perf_counter()` und Logging.
- Verwendung:
  - Messen der Gesamt-Laufzeit eines Jobs.
  - Messen einzelner Hauptschritte (z.B. "load_data", "build_factors", "run_backtest", "generate_reports").

### 3.2. Low-Level Profiling (cProfile)

- **Werkzeug:** Python `cProfile` + `pstats`.
- Verwendung:
  - Detaillierte Funktions-Statistiken (ncalls, tottime, cumtime).
  - Ausgabe als:
    - Text-Report (Top-N Functions nach cumtime).
    - Optional: `.prof`-File für externe Tools (Snakeviz, pyprof2calltree, etc.).

**Optionale Erweiterung (später, nicht in P1 zwingend):**

- `pyinstrument` oder `line_profiler` für noch feinere Analyse.

---

## 4. Daten & Outputs

### 4.1. Log-Ausgaben

Alle Profiling-Runs schreiben Logs in:

- `output/perf_logs/`
  - `perf_<job_name>_<timestamp>.log`

Inhalte:

- Gesamt-Laufzeit
- Laufzeit pro Step (falls Steps konfiguriert)
- Pfad zum cProfile-Report (falls aktiviert)

### 4.2. cProfile-Reports

- `output/perf_profiles/`
  - `profile_<job_name>_<timestamp>.prof` (rohe Profildatei)
  - `profile_<job_name>_<timestamp>_stats.txt` (Top-N Funktionen)

### 4.3. Optional: CSV-Summary

Optional (P1.2/P1.3):

- `output/perf_profiles/perf_summary.csv`
  - Spalten: `timestamp, job_name, total_sec, top1_func, top1_cumtime, ...`

---

## 5. API & Implementierung

### 5.1. Profiling-Script: `scripts/profile_jobs.py`

CLI:

```bash
python scripts/profile_jobs.py --job BASIC_BACKTEST --with-cprofile
python scripts/profile_jobs.py --job FACTOR_ML_JOB
python scripts/profile_jobs.py --job PLAYBOOK_JOB --with-cprofile --top-n 50
```

**Argumente:**

- `--job {BASIC_BACKTEST, FACTOR_ML_JOB, PLAYBOOK_JOB, CUSTOM}` (required)
- `--with-cprofile` (bool)
- `--top-n`: Anzahl der Top-Funktionen im Stats-Report
- `--extra-args`: optionale argumente für den Subjob (Future-Work)

**Interne Steps:**

1. Mappt job auf eine Python-Funktion:
   - `run_basic_backtest_job()`
   - `run_factor_ml_job()`
   - `run_playbook_job()`
   - `run_custom_job()` (Future)

2. Misst Zeit (`perf_counter`) um den Funktionsaufruf.

3. Optional: Wrap mit `cProfile.Profile()`:
   - `prof.enable()` → Job → `prof.disable()`
   - `pstats.Stats(prof)` → Text-Report + `.prof` speichern.

### 5.2. Timing-Helper für Pipeline & Backtester (optional in P1)

Neues Utility (z.B. `src/assembled_core/utils/timing.py`):

```python
from contextlib import contextmanager
import time
import logging

logger = logging.getLogger(__name__)

@contextmanager
def timed_block(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info("TIMING | %s | %.3f sec", name, duration)
```

**Verwendung:**

```python
from assembled_core.utils.timing import timed_block

with timed_block("build_factors"):
    build_factors(...)

with timed_block("run_backtest"):
    run_backtest(...)
```

---

## 6. Implementation Steps (P1.x)

- **P1.1**: Design-Dokument + `scripts/profile_jobs.py` Skeleton
- **P1.2**: Implementierung der 3 Beispiel-Jobs (Backtest, Factor+ML, Playbook)
- **P1.3**: Integration `timed_block` in zentrale Pipeline/Backtest-Schritte
- **P1.4**: Erste Baseline-Runs dokumentieren (`docs/PERFORMANCE_PROFILE.md`)

---

## 7. Risiken & Limitierungen

- cProfile verlangsamt Jobs → nur selektiv einsetzen.
- Ergebnisse sind hardware-abhängig → Dokumentation der Testumgebung wichtig.
- Profiling selbst darf keine Logik ändern (Read-Only).

---

## 8. Next Steps nach P1

- **P2**: Factor-Store & Data Layout ✅ (Completed) - [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md)
- **P3**: Backtest-Engine-Refactor (Vektorisierung/Numba) ✅ (Completed) - [Backtest Optimization P3 Design](BACKTEST_OPTIMIZATION_P3_DESIGN.md)
- **P4**: Batch-Runner & Parallelisierung (Planned)

---

## 9. References

- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md): Gesamte Roadmap
- Python cProfile: https://docs.python.org/3/library/profile.html
- pstats: https://docs.python.org/3/library/profile.html#module-pstats

