# 01b Worktree-Diff — Ungesicherte Arbeit in 4 aktiven Worktrees
Stand: 2026-05-27 | Basis: main @ a7e01689

Zweck: Reine Analyse. Keine remove-, branch-delete- oder checkout-Befehle ausgeführt.

Methodik:
- Modified Dateien: `git -C <worktree> diff main -- <pfad>`
- Untracked Dateien: Existenz in main geprüft, dann `diff <worktree-datei> <main-datei>`

---

## Worktree 1: agent-a7b606cb13c522e1d

Pfad: `.claude/worktrees/agent-a7b606cb13c522e1d` | HEAD: `bb1ba02e` (4 hinter main)

### 1a. MODIFIED: `src/assembled_core/data/universe.py`

```
(leere Ausgabe)
```

**Befund:** Identisch mit main. Keine Änderungen.

---

### 1b. MODIFIED: `src/assembled_core/pipeline/orchestrator.py`

`git diff main` zeigt: Der Worktree hat eine **ältere Version** ohne die Benchmark-Attribution, die in main vorhanden ist.

```diff
@@ -1224,125 +1224,6 @@
-    try:
-        from src.assembled_core.qa.benchmark_metrics import compute_benchmark_metrics
-        # [125 Zeilen Benchmark-Attribution-Code entfernt]:
-        # - Equity-Curve-Laden (portfolio_equity_1d.csv / equity_curve_1d.csv)
-        # - SPY-Laden aus prices_1d.parquet
-        # - PIT-Guard: _spy = _spy[_spy.index <= _eq_max_date]
-        # - _bm_safe() für inf/nan-sichere JSON-Serialisierung
-        # - purge_old_dated_reports() für Retention
-        # - json.dump zu ops/benchmark_attr_{date}.json
-        # - logger.info mit alpha/beta/IR
-    except Exception as _bm_exc:
-        logger.warning("[EOD][BenchmarkAttr] Non-blocking Fehler: %s", _bm_exc)
```

Zusätzlich: Der Factor-Decay-Abschnitt ist vereinfacht (fehlt `_fd_result`-Logging, Import-Style abweichend).

**Befund:** Worktree hat Version **vor** Commit `72236ebb` (benchmark attribution). Main enthält den vollständigen, gehärteten Code. Der Worktree hat keine einzige Zeile, die nicht bereits in main wäre.

---

### 1c. UNTRACKED: `src/assembled_core/qa/factor_decay_reporter.py`

Datei existiert in main. `diff` zeigt ausschließlich Formatierungsunterschiede (schwarze/ruff Zeilenumbrüche):

```diff
84c84,89
<         return {"status": "skip", ...}  [1 Zeile]
---
>         return {            [5 Zeilen, aufgesplittet]
>             "status": "skip",
            ...
```

Logik identisch. Keine inhaltlichen Unterschiede.

**Befund:** Worktree hat die **ältere, unformatierte Version**. Main hat die ruff-formatierte Version. Kein einzigartiger Inhalt.

---

### 1d. UNTRACKED: `tests/test_factor_decay_reporter.py`

Datei existiert in main. `diff` zeigt ausschließlich Formatierungsunterschiede:
- Zeile 182–183: `run_factor_decay_monitoring(...)` war auf 1 Zeile, main hat multi-line
- Zeile 188–190: Methoden-Signatur aufgespaltet
- Zeile 195: `caplog.at_level(...)` aufgespaltet

**Befund:** Funktional identisch. Main hat die ruff-formatierte Version.

---

### 1e. UNTRACKED: `tests/test_universe_survivorship.py`

Datei existiert in main. `diff` zeigt ausschließlich Formatierungsunterschiede:
- `caplog.at_level(...)` auf mehrere Zeilen verteilt
- `assert any(...)` mit Klammern umgeformt
- `warning_messages = [...]` aufgespaltet

**Befund:** Funktional identisch. Main hat die ruff-formatierte Version.

---

### Fazit Worktree 1

**SICHER LÖSCHBAR**

Alle 5 Dateien enthalten ausschließlich ältere oder identische Stände. Die `orchestrator.py` des Worktrees ist eine Vorstufe ohne Benchmark-Attribution — main enthält den vollständig überarbeiteten und gehärteten Code. Die 3 untracked Dateien sind funktional identisch mit main, nur nicht ruff-formatiert.

---

## Worktree 2: agent-a8530ae9d7d5595f6

Pfad: `.claude/worktrees/agent-a8530ae9d7d5595f6` | HEAD: `9467b0ae` (2 hinter main)

### 2a. MODIFIED: `src/assembled_core/pipeline/orchestrator.py`

`git diff main` zeigt: Worktree hat eine **frühere Entwurfsversion** der Benchmark-Attribution mit mehreren Lücken gegenüber main:

```diff
# FEHLT in Worktree (vs. main):

# 1. Import numpy als lokaler _np_bm für inf-sichere Returns:
-    import numpy as _np_bm
-    _port_ret = _eq_df[_eq_col].pct_change()
-        .replace([_np_bm.inf, -_np_bm.inf], _np_bm.nan).dropna()
# Worktree hat stattdessen:
+    _port_ret = _eq_df[_eq_col].pct_change().dropna()
# → inf-Werte in Returns werden NICHT bereinigt

# 2. timestamp-Column-Guard für SPY-Preisdaten fehlt im Worktree:
-    if "timestamp" not in _spy.columns:
-        logger.debug("[EOD][BenchmarkAttr] prices_1d.parquet missing 'timestamp' column — skipping SPY")
-        _spy = _spy.iloc[0:0]
-    else:
-        _spy["date"] = pd.to_datetime(_spy["timestamp"], errors="coerce")
# Worktree hat stattdessen (unsicherer):
+    _spy["date"] = pd.to_datetime(_spy.get("timestamp", _spy.index), errors="coerce")

# 3. PIT-Guard fehlt im Worktree:
-    _eq_max_date = _eq_df.index.max()
-    _spy = _spy[_spy.index <= _eq_max_date]  # PIT-Guard
# Worktree kennt _eq_max_date nicht

# 4. Dateiname nicht PIT-sicher im Worktree:
-    _bm_date_str = _eq_max_date.strftime("%Y%m%d") if pd.notna(_eq_max_date) else ...
# Worktree hat:
+    _bm_date_str = pd.Timestamp.now("UTC").strftime("%Y%m%d")
# → Replay-Naming nicht deterministisch (Wallclock statt Equity-Curve-Datum)

# 5. NaN/inf-sichere JSON-Serialisierung fehlt:
-    def _bm_safe(v):
-        if isinstance(v, float) and not _np_bm.isfinite(v): return None
-        return v
-    json.dump({k: _bm_safe(v) for k, v in vars(_bm).items()}, ...)
# Worktree:
+    import json as _json2
+    _json2.dump({k: v for k, v in vars(_bm).items()}, ...)
# → kann inf/nan in JSON schreiben → ungültiges JSON

# 6. Retention-Purge fehlt im Worktree (purge_old_dated_reports):
-    purge_old_dated_reports(_bm_out.parent, "benchmark_attr_", ".json", keep_last_n=60)
```

**Befund:** Worktree enthält einen **Entwurfsstand** der Benchmark-Attribution mit 4 bekannten Mängeln (inf-Handling, timestamp-Guard, PIT-Guard, JSON-Serialisierung) und fehlendem Retention-Purge. Main hat die korrigierte Endversion.

---

### 2b. UNTRACKED: `tests/test_benchmark_attribution_wiring.py`

Datei existiert in main. `diff` zeigt: Worktree hat den **älteren, schmaleren Testsatz**.

**Fehlend im Worktree (nur in main):**
- `_make_prices_parquet` hat keine Parameter `include_timestamp`, `spy_only`, `extra_days`
- `_route` hat keinen `qty`-Parameter
- `_route_multi`-Hilfsfunktion fehlt vollständig

**3 Testfunktionen fehlen komplett im Worktree:**
```python
def test_benchmark_attr_skips_when_spy_not_in_prices(tmp_path):
    # Parquet mit nur AAPL, kein SPY → kein JSON geschrieben

def test_benchmark_attr_pit_filter_truncates_spy(tmp_path):
    # SPY-Daten über Equity-Curve-Ende hinaus müssen abgeschnitten werden

def test_benchmark_attr_skips_when_prices_missing_timestamp_col(tmp_path):
    # Preise ohne 'timestamp'-Spalte → kein JSON (F-senior-2 Fix)
```

**Befund:** Main hat die vollständige Testsuite. Worktree hat den Draft ohne Randfall-Tests.

---

### Fazit Worktree 2

**SICHER LÖSCHBAR**

`orchestrator.py` ist ein Entwurf mit 5 bekannten Lücken gegenüber main (fehlende Härtegrad-Funktionen). `test_benchmark_attribution_wiring.py` ist ein älterer Draft, dem 3 Testfunktionen fehlen, die in main vorhanden sind. Kein einzigartiger Inhalt.

---

## Worktree 3: agent-ac275289d3bf5b9ed

Pfad: `.claude/worktrees/agent-ac275289d3bf5b9ed` | HEAD: `bb1ba02e` (4 hinter main)

### 3a. MODIFIED: `configs/policy.yaml`

`git diff main` zeigt: Worktree fehlt der `algo_execution`-Abschnitt, der in main vorhanden ist:

```diff
# Vorhanden in main, NICHT in Worktree:
-  algo_execution:
-    enabled: true
-    algo: "TWAP"
-    n_slices: 10
-    participation_rate: 0.10
```

**Befund:** Worktree hat die ältere `policy.yaml` **vor** dem TWAP-Wiring-Commit `72236ebb`. Main enthält den vollständigen Stand.

---

### 3b. UNTRACKED: `scripts/dms_daemon.py`

Datei existiert in main. `diff` zeigt ausschließlich Formatierungsunterschiede (2 lange `print`- bzw. `logger.info`-Zeilen auf mehrere Zeilen umgebrochen).

**Befund:** Funktional identisch. Main hat die ruff-formatierte Version.

---

### 3c. UNTRACKED: `src/assembled_core/ops/dead_man_switch.py`

Datei existiert in main. `diff` zeigt ausschließlich Formatierungsunterschiede:
- `"timeout_seconds": 900.0,      # 15 minutes` → `900.0,  # 15 minutes` (Whitespace)
- `logger.info(...)` auf mehrere Zeilen umgebrochen
- Lange `if`-Bedingung auf 4 Zeilen umgebrochen

**Befund:** Funktional identisch. Main hat die ruff-formatierte Version.

---

### 3d. UNTRACKED: `tests/test_dead_man_switch.py`

Datei existiert in main. `diff` zeigt ausschließlich Formatierungsunterschiede — Worktree benutzt die alte `with patch(...) as x, patch(...) as y:` Syntax, main benutzt parenthesized context managers (Python 3.10+):

```diff
# Worktree (alt):
with patch("...check_liveness", ...) as _mock_cl, patch(
    "...auto_flatten_on_stale"
) as mock_flatten:

# Main (Python 3.10+ parenthesized):
with (
    patch("...check_liveness", ...) as _mock_cl,
    patch("...auto_flatten_on_stale") as mock_flatten,
):
```

Logik identisch. Keine inhaltlichen Unterschiede.

**Befund:** Funktional identisch. Main hat die modernere Syntax.

---

### Fazit Worktree 3

**SICHER LÖSCHBAR**

`policy.yaml` ist eine ältere Version ohne `algo_execution`-Block (main hat die vollständige Version). Die 3 untracked Dateien sind funktional identisch mit main — nur Formatierungsunterschiede. Kein einzigartiger Inhalt.

---

## Worktree 4: agent-aff57adf12c4aadd1

Pfad: `.claude/worktrees/agent-aff57adf12c4aadd1` | HEAD: `9467b0ae` (2 hinter main)

### 4a. MODIFIED: `configs/policy.yaml`

```
(leere Ausgabe)
```

**Befund:** Identisch mit main. Keine Änderungen.

---

### 4b. MODIFIED: `src/assembled_core/pipeline/_tc_execution.py`

`git diff main` zeigt: Worktree hat eine **ältere Implementierung** der Algo-Annotation mit zwei funktionalen Regressionen gegenüber main:

**Regression 1 — Einheitlicher statt per-Order Slice Count:**
```diff
# Worktree: berechnet _effective_slices einmal anhand der ERSTEN Order-Zeile:
+    _rep_qty = float(orders["qty"].iloc[0]) if not orders.empty else 1.0
+    _effective_slices = max(1, min(_algo_n_slices, int(_rep_qty)))
+    orders["algo_n_slices"] = _effective_slices   # ALLE Orders bekommen denselben Wert

# Main: per-Order, NaN/inf-sicher:
-    orders["algo_n_slices"] = (
-        orders["qty"].abs()
-        .replace([_np_exec.inf, -_np_exec.inf], _np_exec.nan)
-        .fillna(1.0)
-        .apply(lambda q: max(1, min(_algo_n_slices, int(max(float(q), 1.0)))))
-    )
```

**Regression 2 — algo_type/algo_n_slices fehlen im Trade-Journal (E-024-Reversion):**
```diff
# Worktree (book_fills): dict comprehension OHNE algo-Felder:
+    _tj_fills = [
+        {"symbol": str(r["symbol"]), "side": str(r["side"]),
+         "qty": float(...), "price": float(...)}   # kein algo_type, kein algo_n_slices
+        for r in _of[["symbol", "side", _qty_col, _price_col]].itertuples(index=False)
+    ]

# Main: itertuples-Schleife MIT conditionalem algo_type/algo_n_slices-Handling:
-    for _row in _of[["symbol", "side", _qty_col, _price_col] + _algo_avail].itertuples(...):
-        _e = {"symbol": ..., "qty": ..., ...}
-        if "algo_type" in _algo_avail:
-            _e["algo_type"] = str(getattr(_row, "algo_type", ""))
-        if "algo_n_slices" in _algo_avail:
-            _ns = getattr(_row, "algo_n_slices", None)
-            _e["algo_n_slices"] = int(_ns) if pd.notna(_ns) else 0
```

Zusätzlich: Worktree importiert `TWAPScheduler`/`VWAPScheduler` aus `algo_execution`, initiiert Scheduler-Instanzen und loggt nur 2 statt 3 Argumente. Diese Import-Struktur wurde in main vereinfacht.

**Befund:** Worktree-Version hat 2 echte Bugs (flat statt per-row Slicing, fehlende E-024 Trade-Journal-Felder). Main enthält die korrekte Endversion.

---

### 4c. UNTRACKED: `tests/test_twap_vwap_annotation.py`

Datei existiert in main. `diff` zeigt: Worktree hat deutlich weniger Tests.

**Im Worktree fehlend (nur in main):**

`_route_multi`-Helper fehlt komplett:
```python
def _route_multi(policy, qtys, mode="paper"):
    """Route with a multi-row orders DataFrame."""
    ...
```

**5 Testfunktionen fehlen komplett:**
```python
def test_algo_n_slices_clamped_to_qty():
    # qty=3 < n_slices=10 → effective=3, nicht 10

def test_nan_qty_falls_back_to_one_slice():
    # qty=NaN → 1 Slice

def test_inf_qty_does_not_crash():
    # qty=inf → 1 Slice (safe default)

def test_per_row_slice_count_with_mixed_qtys():
    # [100, 3, NaN, -50, inf] → [10, 3, 1, 10, 1] per-Order

# Trade-Journal E-024 Tests (3 Stück):
def test_trade_journal_entry_includes_algo_metadata(tmp_path):
def test_trade_journal_entry_omits_algo_metadata_when_absent(tmp_path):
def test_trade_journal_entry_omits_algo_metadata_when_empty_string(tmp_path):
```

**Befund:** Worktree hat den Draft **vor** dem E-024-Fix. Main hat 5 zusätzliche Testfunktionen, darunter die Randfälle (NaN, inf, per-row clamping) und 3 E-024-Trade-Journal-Tests, die die korrekte Endversion absichern.

---

### Fazit Worktree 4

**SICHER LÖSCHBAR**

`policy.yaml`: identisch mit main. `_tc_execution.py`: Entwurf mit 2 echten Bugs (flat slicing, E-024-Regression) — main hat die korrigierte Endversion. `test_twap_vwap_annotation.py`: Draft ohne 5 kritische Testfunktionen — main hat die vollständige Testsuite.

---

## Gesamttabelle

| Worktree | Kategorie | Unikale Arbeit |
|---|---|---|
| `agent-a7b606cb13c522e1d` | **SICHER LÖSCHBAR** | Keine. `orchestrator.py` ist Vorversion ohne Benchmark-Attribution; 3 untracked Dateien nur formatting-verschieden von main. |
| `agent-a8530ae9d7d5595f6` | **SICHER LÖSCHBAR** | Keine. `orchestrator.py` ist Entwurf mit 5 fehlenden Härtungen (PIT, inf, timestamp, Retention, JSON-NaN); Tests fehlen 3 Randfälle. |
| `agent-ac275289d3bf5b9ed` | **SICHER LÖSCHBAR** | Keine. `policy.yaml` fehlt `algo_execution`-Block; 3 untracked Dateien nur formatting-verschieden von main. |
| `agent-aff57adf12c4aadd1` | **SICHER LÖSCHBAR** | Keine. `_tc_execution.py` hat 2 echte Bugs (flat slicing, E-024 Trade-Journal fehlt); Tests fehlen 5 Funktionen incl. E-024-Abdeckung. |

---

## Schlussfolgerung

Alle 4 Worktrees enthalten **ausschließlich ältere Entwicklungsstände**. In keinem Worktree existiert eine einzige Zeile Code oder ein einziger Test, der **nicht bereits in main vorhanden und verbessert** wäre.

Die Reihenfolge der Entstandenheit ist in allen Fällen klar:
1. Draft entstand im Worktree als Subagent-Arbeit
2. Wurde committed und in main weitergepflegt
3. main erhielt in Folge-Commits Korrekturen (PIT-Guard, inf-Handling, per-row Slicing, E-024)
4. Worktree-Dateien wurden nie aktualisiert und spiegeln den unfertigen Ursprungsstand

**Empfehlung:** Alle 4 Worktrees können ohne Datenverlust entfernt werden. Reihenfolge: erst `git worktree remove --force <pfad>`, dann `git branch -d <name>`.
