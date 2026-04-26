# Repo-weiter Audit & Behebungs-Anleitung (v2)

**Datum:** 2026-04-26 (v2 — Nachbesserung mit zusätzlichen Befunden)
**Scope:** Alle Befunde aus 7 Diagnose-Schichten plus Web-Recherche zu pandas 3.0/numpy 2.0 Migration, **ohne** die bereits abgeschlossene `trading_cycle.py → trading_cycle_v2.py`-Migration (siehe `MIGRATION_TRADING_CYCLE_V2.md`).
**Zweck:** Diese Datei ist deine Arbeitsgrundlage. Jeder Befund hat Datei + Zeile, Begründung, konkreten Fix, und Akzeptanzkriterien.

**Was sich gegenüber v1 geändert hat:**
- Doppeltes Schema-Beispiel "A1 — Kurztitel" entfernt
- 6 neue Befunde: A12 (compliance broken), A13 (dual pytest config), A14 (orchestrator slippage), D17 (datetime.now Vollliste), D18 (UTC-Lüge), D19 (asyncio deprecated), D20 (numpy 2.0 readiness)
- Status-Tracking-Tabelle am Anfang
- Akzeptanzkriterien überall vereinheitlicht
- Aufwandsschätzungen als Spannen wo unsicher
- Empfehlungs-Reihenfolge konsistent: Quick Wins zuerst, dann Schwere-zuerst-bei-gleichem-Quick-Win-Status

---

## Inhaltsverzeichnis

- [Wie du diese Datei nutzt](#wie-du-diese-datei-nutzt)
- [Status-Tracking](#status-tracking)
- [Cluster A — Akute Korrektheits-Bugs](#cluster-a--akute-korrektheits-bugs) (14 Punkte)
- [Cluster B — Architektur-Konsolidierung](#cluster-b--architektur-konsolidierung) (8 Punkte)
- [Cluster C — Architektur-Hygiene](#cluster-c--architektur-hygiene) (6 Punkte)
- [Cluster D — Selbst-Audits & Strukturen](#cluster-d--selbst-audits--strukturen) (20 Punkte)
- [Empfohlene Reihenfolge](#empfohlene-reihenfolge)
- [Anhang: Pattern-Erkenntnis](#anhang-pattern-erkenntnis)

**Gesamt: 48 Befunde** (A1–A14, B1–B8, C1–C6, D1–D20). Die Sektionen unter "Empfohlene Reihenfolge" sind keine zusätzlichen Befunde, sondern Phasen-Gruppierungen.

---

## Wie du diese Datei nutzt

Jeder Befund ist nach diesem Schema strukturiert:

```
### X-Nummer — Kurztitel
**Datei:** path/to/file.py:LINE
**Schwere:** akut / mittel / niedrig
**Schicht:** in welchem Diagnose-Durchlauf gefunden
**Aufwand:** Stunden-Schätzung (Spanne wenn unsicher)

**Was ist das Problem?**  Beschreibung mit Code-Snippet.
**Warum gefährlich?**     Konsequenzen.
**Behebung:**             Schritt-für-Schritt.
**Akzeptanzkriterien:**   Prüfbare Bedingungen.
```

Arbeite chronologisch **innerhalb eines Clusters**, nicht über Cluster hinweg. Die akuten Punkte (A) sind echte Bugs mit Geld-Konsequenz — die zuerst.

---

## Status-Tracking

Trage hier ein, wenn du einen Befund bearbeitest oder erledigt hast. Status-Werte: `TODO`, `IN PROGRESS`, `DONE`, `WONTFIX`, `BLOCKED`.

| ID | Titel | Schwere | Aufwand | Status | Notiz |
|---|---|---|---|---|---|
| A1 | `triple_barrier.py` bfill ML-Leakage | akut | 2h | TODO | |
| A2 | `idempotency.py` nicht verkabelt | akut | 4h | TODO | |
| A3 | `symbol_kill_switch.py` non-atomic | akut | 1h | TODO | |
| A4 | `scenario_engine.py` Cholesky+Seed | akut | 2h | TODO | |
| A5 | `characterization`-Marker undeklariert | akut | 15min | TODO | Quick Win |
| A6 | `--disable-warnings` schluckt Warnings | akut | 30min | TODO | Quick Win |
| A7 | `enable_corporate_actions=False` Default | akut | 6h | TODO | |
| A8 | Slippage in v2 backtest fehlt | akut | 4h | TODO | |
| A9 | Walk-Forward ohne Embargo/Purge | akut | 6-8h | TODO | |
| A10 | Survivorship-Bias: PIT-API nicht verkabelt | akut | 8-40h | TODO | Daten-Beschaffung kann lang dauern |
| A11 | `validate_price_data` nicht aufgerufen | akut | 2h | TODO | |
| A12 | `compliance/` Modul broken | akut | 1h | TODO | NEU |
| A13 | Dual pytest config Konflikt | akut | 1h | TODO | NEU |
| A14 | `orchestrator.py` ohne Slippage | akut | 2h | TODO | NEU |
| B1 | 3 Quellen Wahrheit Dependencies | mittel | 3h | TODO | |
| B2 | 30 weitere non-atomic JSON-Writes | mittel | 6h | TODO | |
| B3 | NewsEvent Schema-Drift | mittel | 6-10h | TODO | |
| B4 | `predict_regime` doppelt | mittel | 3h | TODO | |
| B5 | `orchestrator.py` parallele Pipeline | mittel-hoch | 12-20h | TODO | |
| B6 | `nation_profiles.py` Migrations-Waise | niedrig | 30min | TODO | Quick Win |
| B7 | 34+ Tests testen unverkabelten Code | mittel | 4h | TODO | |
| B8 | Zwei Kill-Switch-Systeme | mittel | 2h | TODO | nach A3 |
| C1 | `size_positions()` CC=236 | mittel | 16-24h | TODO | |
| C2 | 122 `.iterrows()` | niedrig | 12h | TODO | |
| C3 | Top 7 Funktionen >500 LOC | mittel | quartal | TODO | |
| C4 | `scripts/cli.py` 4007 LOC | niedrig | 6-10h | TODO | |
| C5 | 199 Doku-Dateien, AGENTS.md falsch | mittel | 4h | TODO | |
| C6 | 204 Heartbeat-Commits | niedrig | 1h | TODO | |
| D1 | KNOWN_ISSUES Header von Januar 2025 | niedrig | 10min | TODO | Quick Win |
| D2 | 47 silent except: pass | niedrig | 3h | TODO | |
| D3 | Property-Tests 0.4% | niedrig | kontinuierlich | TODO | |
| D4 | 191 weak assert not df.empty | niedrig | kontinuierlich | TODO | |
| D5 | `_filter_prices_for_as_of` Bar-Konvention | niedrig | 1h | TODO | |
| D6 | CI-Python-Inkonsistenz | niedrig | 30min | TODO | Quick Win |
| D7 | 5 inplace=True (pandas 3.0) | niedrig | 30min | TODO | Quick Win |
| D8 | 3 tote Konfig-YAMLs | niedrig | 10min | TODO | Quick Win |
| D9 | Verzeichnisse mit Leerzeichen | niedrig | 30min | TODO | |
| D10 | 200KB-Datei mit `:`-im-Namen | niedrig | 5min | TODO | Quick Win |
| D11 | 45 Parquets im Repo | niedrig | 1h | TODO | |
| D12 | `external` Marker nicht benutzt | niedrig | 5min | TODO | Quick Win |
| D13 | `run_id` ohne Timezone | niedrig | 10min | TODO | Quick Win |
| D14 | 39/40 pd.read_csv ohne dtype | niedrig | 3h | TODO | |
| D15 | `mean_reversion_factors.py` rolling | niedrig | 30min | TODO | Quick Win |
| D16 | Keine LICENSE/SECURITY/CONTRIBUTING | niedrig | 30min | TODO | Quick Win |
| D17 | 9 datetime.now() Stellen ohne TZ (Vollliste) | niedrig | 1h | TODO | NEU |
| D18 | `daily_qa_report.py:162` UTC-Lüge | niedrig | 5min | TODO | NEU, Quick Win |
| D19 | asyncio.get_event_loop() deprecated | niedrig | 30min | TODO | NEU |
| D20 | numpy 2.0 readiness (NPY201 ruff rule) | niedrig | 1-2h | TODO | NEU |

---

## Cluster A — Akute Korrektheits-Bugs

14 Punkte mit Korrektheits-Risiko in Backtests oder Live-Trading. Höchste Priorität.

---

### A1 — `triple_barrier.py` `bfill` erzeugt ML-Label-Leakage

**Datei:** `src/assembled_core/features/triple_barrier.py:151`
**Schwere:** akut
**Schicht:** 4
**Aufwand:** 2h

**Was ist das Problem?**

```python
vol = log_ret.rolling(20).std().fillna(method="bfill")
```

`bfill` füllt fehlende Frühwerte mit dem **nächsten zukünftigen** Wert. Triple-Barrier ist eine ML-Label-Methode (López de Prado): "Wann erreicht der Preis Profit-Take, Stop-Loss oder Time-Barrier?" Die Volatilitätsschätzung steuert die Barriere-Höhe.

Doppeltes Problem: `fillna(method="bfill")` ist außerdem in pandas 3.0 (released 2026-01-21) **entfernt** — das Pattern wirft FutureWarning ab pandas 2.1, ValueError ab pandas 3.0.

**Warum gefährlich?**

Die ersten 20 Trade-Labels bekommen eine Volatilität, die erst Tage später bekannt ist. Wenn diese Labels für Meta-Model-Training genutzt werden, **lernt dein Modell aus dem Trainings-Datenleck**. Backtest-Performance sieht systematisch zu gut aus, Live-Performance enttäuscht. Plus: Code wird beim pandas-3.0-Upgrade brechen.

**Behebung:**

1. `bfill` durch `dropna()` ersetzen:
   ```python
   vol = log_ret.rolling(20).std()
   # Don't bfill — first 20 rows stay NaN, downstream code must handle
   ```

2. In `_numpy_triple_barrier` nach Zeile 152 Events-Filter:
   ```python
   for t0 in events:
       if t0 not in prices.index:
           continue
       v_at_t0 = vol.loc[t0] if t0 in vol.index else None
       if v_at_t0 is None or pd.isna(v_at_t0):
           continue  # Skip events before vol is known
       ...
   ```

3. Test in `tests/test_triple_barrier_no_leakage.py`:
   - Synthetische Returns
   - Compute Labels
   - Assert: Erste 20 Labels sind NaN (oder Events darin werden gefiltert)
   - Assert: Vol für Bar `t` benutzt nur Daten `<= t-1`

**Akzeptanzkriterien (alle müssen erfüllt sein):**

- [ ] `grep -rn "bfill\|backfill" src/assembled_core/features/triple_barrier.py` liefert exakt 0 Treffer
- [ ] `grep -rn 'fillna(method=' src/assembled_core/` liefert exakt 0 Treffer
- [ ] Neue Test-Datei `tests/test_triple_barrier_no_leakage.py` existiert mit ≥3 Test-Cases
- [ ] `pytest tests/test_triple_barrier_no_leakage.py` ist grün
- [ ] Bestehende Triple-Barrier-Tests sind grün oder bewusst angepasst (Commit-Message dokumentiert die Anpassung)

---

### A2 — `idempotency.py` ist nicht in der Pipeline verkabelt

**Datei:** `src/assembled_core/execution/idempotency.py` (existiert), `src/assembled_core/execution/paper_trading_engine.py:339` (nutzt UUIDs)
**Schwere:** akut
**Schicht:** 4
**Aufwand:** 4h

**Was ist das Problem?**

Modul `execution/idempotency.py` mit drei Funktionen:
- `compute_intent_hash(symbol, side, qty, order_type)`
- `build_client_order_id(signal_id, intent_hash)` — deterministisch, ≤48 Zeichen für Alpaca
- `is_duplicate_error(error_message)` — erkennt Broker-Fehler

5 Tests dafür. **Kein einziger Aufrufer im Pipeline-Code.** Statt `build_client_order_id` zu nutzen, generiert `paper_trading_engine.py:339` neue UUIDs:

```python
import uuid
order_id=str(uuid.uuid4())
```

**Warum gefährlich?**

Bei transientem Network-Hiccup zwischen Pipeline und Broker:
1. Pipeline submitted Order mit UUID-X
2. Broker erhält, antwortet aber nicht innerhalb Timeout
3. Pipeline retry → submitted gleiche logische Order mit **neuer UUID-Y**
4. Broker hat zwei verschiedene Order-IDs → zwei Orders gefüllt → **Doppel-Position**

`build_client_order_id` würde aus `(signal_id, intent_hash)` immer die gleiche ID liefern. Der Broker erkennt den Duplikat.

**Behebung:**

1. **In `execution/paper_trading_engine.py:339`** UUID durch deterministische ID ersetzen:
   ```python
   from src.assembled_core.execution.idempotency import (
       compute_intent_hash,
       build_client_order_id,
   )
   intent_hash = compute_intent_hash(symbol, side, qty, order_type)
   client_order_id = build_client_order_id(signal_id, intent_hash)
   order_id = client_order_id
   ```

2. **In `execution/broker_adapter.py`** (Alpaca-Adapter): bei Submit-Fehler `is_duplicate_error` checken:
   ```python
   from src.assembled_core.execution.idempotency import is_duplicate_error

   try:
       resp = self.client.submit_order(...)
   except Exception as e:
       if is_duplicate_error(str(e)):
           existing = self.client.get_order_by_client_order_id(client_order_id)
           return existing
       raise
   ```

3. **In `execution/algo_execution.py:154/271`** parent_order_id-Generation analog umstellen.

4. **Test:** `tests/test_idempotency_pipeline_integration.py`
   - Submit, simulierter Network-Error, Retry
   - Assert: zweiter Aufruf gibt keine neue Order-ID
   - Assert: Broker-Mock zählt nur 1 Submit

**Akzeptanzkriterien:**

- [ ] `grep -rn "build_client_order_id" src/assembled_core/` liefert ≥3 Treffer (1 Definition + ≥2 Aufrufer)
- [ ] `grep -rn "uuid.uuid4()" src/assembled_core/execution/` liefert 0 Treffer (oder nur in Test-Helpern)
- [ ] `tests/test_idempotency_pipeline_integration.py` existiert
- [ ] `pytest tests/test_idempotency_pipeline_integration.py` ist grün
- [ ] Mindestens 1 Test simuliert Network-Hiccup und prüft, dass nur 1 Order beim Broker landet

---

### A3 — `symbol_kill_switch.py` schreibt Sicherheits-State nicht atomic

**Datei:** `src/assembled_core/execution/symbol_kill_switch.py:72`
**Schwere:** akut
**Schicht:** 4
**Aufwand:** 1h

**Was ist das Problem?**

```python
def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
```

Nacktes `write_text` ohne tmp+rename. Beim Crash mitten im Schreiben → JSON-Datei korrupt.

`execution/kill_switch.py::_write_state` macht es korrekt mit `tmp.replace(p)`. Beide existieren parallel.

**Warum gefährlich?**

Beim nächsten Start kommt entweder Hard-Crash beim `json.loads` der korrupten Datei, oder schlimmer: Fallback auf "leerer State" → **zuvor blockierte Symbole sind plötzlich frei**. Kill-Switches sind Sicherheits-Mechanismen, müssen deterministisch sein.

**Behebung:**

`atomic_write_json_with_retry` aus `src/assembled_core/risk/state_machine.py` nutzen:

```python
from src.assembled_core.risk.state_machine import atomic_write_json_with_retry

def _write_state(path: Path, state: dict[str, Any]) -> None:
    """Write state atomically with retry."""
    atomic_write_json_with_retry(path, state, retries=3, backoff_ms=50)
```

Falls Cross-Modul-Dependency Execution → Risk unangenehm: Helper nach `utils/atomic_io.py` verschieben, beide Module davon abhängig machen.

**Akzeptanzkriterien:**

- [ ] `grep -A 3 "def _write_state" src/assembled_core/execution/symbol_kill_switch.py` zeigt atomic-write-Aufruf
- [ ] Test in `tests/test_symbol_kill_switch_atomic.py`: simuliere `OSError` während write → State-Datei bleibt unverändert oder leer (NICHT korrupt-truncated)
- [ ] `pytest tests/test_symbol_kill_switch_atomic.py` ist grün

---

### A4 — `qa/scenario_engine.py` Cholesky-Fallback mathematisch falsch

**Datei:** `src/assembled_core/qa/scenario_engine.py:993-998`
**Schwere:** akut
**Schicht:** 4
**Aufwand:** 2h

**Was ist das Problem?**

```python
try:
    L = np.linalg.cholesky(cov + np.eye(n) * 1e-8)
except np.linalg.LinAlgError:
    L = np.eye(n) * vols.mean()  # ← FALSCH

z = np.random.standard_normal((n_sims, n))  # ← UNGESEEDET
```

Drei Probleme in 6 Zeilen:

1. **Falscher Fallback:** `np.eye(n) * vols.mean()` weist allen Assets die Mittelwert-Volatilität zu. Korrekt: `np.diag(vols)`.
2. **Kein Seed:** `np.random.standard_normal` nutzt globalen NumPy-Random-State.
3. **Zum Vergleich:** `risk/risk_metrics.py:880-887` macht es korrekt mit Eigenvalue-Clipping.

**Warum gefährlich?**

VaR/CVaR-Outputs aus dieser Funktion sind nicht reproduzierbar (kein Seed) und im Fallback systematisch falsch (Asset-Risiko gemittelt → Tail-Risk-Konzentrationen unterschätzt).

**Behebung:**

```python
# Vor Cholesky: Eigenvalue-Clipping
eigvals, eigvecs = np.linalg.eigh(cov)
eigvals = np.maximum(eigvals, 1e-10)
cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

try:
    L = np.linalg.cholesky(cov)
except np.linalg.LinAlgError:
    logger.warning(
        "[scenario_engine] Cholesky failed — falling back to diagonal vols"
    )
    L = np.diag(np.sqrt(np.diag(cov)))

# Seed-Parameter
def compute_stressed_var_cvar(..., seed: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, n))
```

**Akzeptanzkriterien:**

- [ ] Funktion akzeptiert `seed`-Parameter
- [ ] Zwei Aufrufe mit `seed=42` liefern bitwise-identische Outputs (`assert np.array_equal`)
- [ ] Fallback nutzt `np.diag(np.sqrt(np.diag(cov)))`
- [ ] Test prüft Fallback-Pfad mit singulärer Cov

---

### A5 — `characterization`-Marker undeklariert in pytest.ini

**Datei:** `pytest.ini` (Markers-Sektion), 17 Test-Files in `tests/characterization/`
**Schwere:** akut
**Schicht:** 5
**Aufwand:** 15min

**Was ist das Problem?**

`@pytest.mark.characterization` wird in 17 Test-Files genutzt. **Nicht in `pytest.ini` deklariert** (aber in `pyproject.toml` deklariert — siehe A13 für die Doppel-Config).

Mit `--strict-markers` failt die Test-Collection live verifiziert:
```
ERROR collecting tests/characterization/test_invariants.py
'characterization' not found in `markers` configuration option
```

**Warum gefährlich?**

17 Snapshot-Tests, deren Aufgabe Verhaltens-Schutz ist, laufen unter keiner CI-Konfiguration. Verhaltensänderungen werden nicht erkannt.

**Behebung:**

In `pytest.ini` unter `markers =` ergänzen:
```ini
markers =
    fast: ...
    integration: ...
    ...
    characterization: Golden-master / snapshot tests (deterministic, seeded)
```

Plus: in CI-Workflow `-m "characterization"` aufnehmen oder eigenen Job anlegen.

**Akzeptanzkriterien:**

- [ ] `python -m pytest tests/characterization/test_invariants.py --collect-only --strict-markers` läuft fehlerfrei
- [ ] `pytest -m characterization --collect-only` zeigt > 15 Tests
- [ ] Mindestens ein CI-Workflow ruft characterization-Tests auf
- [ ] CI bleibt grün

---

### A6 — `--disable-warnings` schluckt eigene DeprecationWarnings

**Datei:** `pytest.ini` (`addopts`-Sektion)
**Schwere:** akut
**Schicht:** 1
**Aufwand:** 30min

**Was ist das Problem?**

`pytest.ini`:
```ini
addopts =
    -q
    --strict-markers
    -m "not external"
    --tb=short
    --disable-warnings   ← problematisch
```

`--disable-warnings` unterdrückt **alle** Warnings, auch eigene. Migration-Shim-Warnings (DeprecationWarning aus `trading_cycle.py`-Migration), pandas FutureWarnings (siehe A1 — `fillna(method=)` deprecation seit pandas 2.1) — alles unsichtbar.

**Warum gefährlich?**

Eingebaute Diagnostik ist abgeschaltet. pandas 3.0 (Januar 2026) entfernt viele deprecated APIs — du erkennst nicht, welche Stellen brechen werden.

**Behebung:**

```ini
addopts =
    -q
    --strict-markers
    -m "not external"
    --tb=short
    # --disable-warnings entfernt

filterwarnings =
    error::DeprecationWarning:src.assembled_core.pipeline.*
    error::FutureWarning:src.assembled_core.*
    ignore::PendingDeprecationWarning
    ignore::ResourceWarning
    # 3rd-party warnings als warnings, nicht als errors:
    default::DeprecationWarning
    default::FutureWarning
```

**Akzeptanzkriterien:**

- [ ] `--disable-warnings` ist nicht mehr in `addopts`
- [ ] `filterwarnings`-Block existiert mit mindestens 4 Regeln
- [ ] CI bleibt grün — falls nicht, sind echte Issues sichtbar geworden, die einzeln gefixt werden müssen

---

### A7 — `enable_corporate_actions: bool = False` als Default

**Datei:** `src/assembled_core/execution/unified_paper_engine.py` (Config-Default), `src/assembled_core/qa/backtest_engine.py` (kennt das Konzept gar nicht)
**Schwere:** akut
**Schicht:** 6
**Aufwand:** 6h

**Was ist das Problem?**

Im Paper-Engine ist Korporative-Aktionen-Adjustierung opt-in mit `enable_corporate_actions: bool = False`. `qa/backtest_engine.py` kennt das Konzept gar nicht.

Modul `data/corporate_actions.py` hat 7 ausgereifte Funktionen, davon ist nur 1 (`adjust_prices_for_splits`) verkabelt — opt-in.

**Warum gefährlich?**

- TSLA 3-zu-1-Split 2022-08-25 → ein Backtest, der das ignoriert, sieht $891 → $297 als 67%-Crash
- AAPL Dividenden ~$0.96/Jahr → Total-Return ohne Dividenden unterschätzt um ~0.6% p.a.
- Delistings → "verschwundene" Symbole

**Behebung:**

Empfehlung: Option 1 + 2 kombiniert.

1. Default umkehren in `unified_paper_engine.py`:
   ```python
   enable_corporate_actions: bool = True
   ```

2. Bei nicht-vorhandenem `corporate_actions_path` Fallback auf False mit Warning.

3. In `qa/backtest_engine.py` Step 0 ergänzen:
   ```python
   if config.enable_corporate_actions:
       from src.assembled_core.data.corporate_actions import (
           adjust_prices_for_splits,
       )
       splits = load_corporate_actions(config.corporate_actions_path)
       prices = adjust_prices_for_splits(prices, splits)
   ```

4. Beim Start loggen, wenn deaktiviert:
   ```python
   if not config.enable_corporate_actions:
       logger.warning(
           "[BACKTEST] Corporate actions DISABLED — splits/dividends ignored. "
           "Results may be misleading."
       )
   ```

**Akzeptanzkriterien:**

- [ ] `unified_paper_engine.py` Default ist `enable_corporate_actions: bool = True`
- [ ] `grep "adjust_prices_for_splits" src/assembled_core/qa/backtest_engine.py` liefert ≥1 Treffer
- [ ] Test mit synthetischen Split-Events: ohne Adjustierung gibt 50%-"Crash" am Split-Datum, mit Adjustierung nicht
- [ ] KNOWN_ISSUES.md hat einen Eintrag, der das Verhalten dokumentiert

---

### A8 — Slippage in `trading_cycle_v2 mode=backtest` nicht angewendet

**Datei:** `src/assembled_core/pipeline/trading_cycle_v2.py` (kein `add_cost_columns_to_trades`-Aufruf)
**Schwere:** akut
**Schicht:** 6
**Aufwand:** 4h

**Was ist das Problem?**

`trading_cycle_v2.py` hat `mode == "backtest"` Pfad. **`grep "add_cost_columns_to_trades\|slippage" src/assembled_core/pipeline/trading_cycle_v2.py` liefert 0 Treffer.**

Slippage-Anwendung passiert nur in:
- `src/assembled_core/pipeline/portfolio.py:131`
- `src/assembled_core/qa/backtest_engine.py:1229`

**Warum gefährlich?**

Wer `trading_cycle_v2` mit `mode="backtest"` direkt aufruft (Tests, Notebooks, eigene Skripte), bekommt Orders ohne Slippage. Fragiler Vertrag.

**Behebung:**

Innerhalb `trading_cycle_v2.book_fills` (Zeile 1882) bei `mode in ("backtest", "paper")` `add_cost_columns_to_trades` aufrufen:

```python
def book_fills(orders, ctx, ...):
    ...
    if ctx.mode in ("backtest", "paper"):
        from src.assembled_core.execution.transaction_costs import (
            add_cost_columns_to_trades,
            CommissionModel,
            SpreadModel,
            SlippageModel,
        )
        from src.assembled_core.costs import get_default_cost_model

        cost_model = ctx.cost_model or get_default_cost_model()
        commission_model = CommissionModel(commission_bps=cost_model.commission_bps)
        spread_model = SpreadModel(...)
        slippage_model = SlippageModel(k=cost_model.impact_w, ...)

        orders = add_cost_columns_to_trades(
            orders,
            commission_model=commission_model,
            spread_model=spread_model,
            slippage_model=slippage_model,
            prices=ctx.prices,
        )
    ...
```

**Akzeptanzkriterien:**

- [ ] `grep "add_cost_columns_to_trades" src/assembled_core/pipeline/trading_cycle_v2.py` liefert ≥1 Treffer
- [ ] Charakterisierungstest in `tests/characterization/test_v2_backtest_with_costs.py` läuft grün
- [ ] Bei `mode="live"` wird Slippage **nicht** angewendet (verifiziert per Test)

---

### A9 — Walk-Forward ohne Embargo/Purge zwischen Train und Test

**Datei:** `src/assembled_core/qa/walk_forward.py`, `src/assembled_core/qa/labeling.py`
**Schwere:** akut
**Schicht:** 7
**Aufwand:** 6-8h

**Was ist das Problem?**

`generate_walk_forward_splits` und `make_walk_forward_splits` haben **keinen Gap-Parameter** zwischen Train- und Test-Window. Plus `qa/labeling.py` mit `horizon_days=10`-Labels.

Train endet 2024-06-30, Test beginnt 2024-07-01, Labels haben 10-Tage-Horizont → die letzten 10 Train-Labels nutzen Daten aus den ersten 10 Test-Tagen.

**Warum gefährlich?**

Klassischer ML-Bias in Time-Series. López de Prado hat genau dafür "Purged K-Fold" eingeführt. Out-of-sample Sharpe lügt systematisch.

**Behebung:**

1. **`embargo_days` und `purge_days` Parameter** zu `WalkForwardConfig`:
   ```python
   @dataclass
   class WalkForwardConfig:
       train_days: int
       test_days: int
       embargo_days: int = 5
       purge_days: int = 10  # >= max_label_horizon
       ...
   ```

2. **In `generate_walk_forward_splits`:**
   ```python
   for window_start in ...:
       train_start = window_start
       train_end = window_start + train_days
       test_start = train_end + purge_days  # PURGE
       test_end = test_start + test_days
       windows.append(WalkForwardWindow(
           train_start, train_end, test_start, test_end
       ))
       window_start = test_end + embargo_days  # EMBARGO
   ```

3. **Validierungs-Assert:**
   ```python
   if purge_days < max_label_horizon:
       raise ValueError(
           f"purge_days ({purge_days}) must be >= max_label_horizon "
           f"({max_label_horizon})"
       )
   ```

4. **Test in `tests/test_walk_forward_no_leakage.py`:**
   - Daten mit klarem zeitlichen Marker (Returns = day_of_month)
   - Train auf Window N, predict auf Window N+1
   - Assert: Train-Label aus Run K endet ≥ purge_days vor Test-Start
   - Assert: Embargo-Gap zwischen Test-Window und nächstem Train-Window

5. **Bestehende Tests anpassen** — manche brechen, weil Daten kürzer als `purge_days`. "13 skips wegen 'No windows generated'" werden mehr.

**Akzeptanzkriterien:**

- [ ] `WalkForwardConfig` hat `embargo_days` und `purge_days`
- [ ] `generate_walk_forward_splits` erzwingt Gaps (auch wenn alte Tests dadurch brechen)
- [ ] Neuer Leakage-Test grün
- [ ] Default-Werte in `KNOWN_ISSUES.md` und Modul-Docstring dokumentiert

---

### A10 — Survivorship-Bias: PIT-Universe-API nicht verkabelt

**Datei:** `src/assembled_core/data/universe.py` (PIT-API existiert), `scripts/run_*.py` (nutzt statische Watchlist)
**Schwere:** akut
**Schicht:** 7
**Aufwand:** 8-40h (datenintensiv, je nach Beschaffungs-Aufwand)

**Was ist das Problem?**

`data/universe.py` definiert `get_universe_members_pit(as_of)`. Mit Doku zu Survivorship-Bias und Exception in `errors.py:87`. **Kein produktiver Aufrufer.** Universum kommt aus statischer `watchlist.txt`.

**Warum gefährlich?**

TSLA war vor 2010 ein Penny-Stock. Backtest 2010-2025 mit heutiger Watchlist "kennt" eine Aktie, die damals nicht trade-bar war. Bei US-Large-Caps moderate Wirkung (+1-2% p.a.), bei Mid-Caps oder kleineren Universen schnell **+5-10% p.a.**

**Behebung:**

1. **Historische Universum-Daten beschaffen** (das ist der unsichere Aufwand):
   - Alt-1: `intel/historical_index_membership/` Daten falls verfügbar
   - Alt-2: Anbieter (Sharadar, Norgate, FactSet) Symbol-Active-Status
   - Alt-3: kostenlos via `pandas-datareader` mit `get_iex_symbols` als Approximation

2. **`get_universe_members_pit` mit echten Daten:**
   ```python
   def get_universe_members_pit(as_of: pd.Timestamp) -> list[str]:
       df = pd.read_parquet("configs/universes/historical_membership.parquet")
       active = df[(df["start_date"] <= as_of) & (df["end_date"] >= as_of)]
       return active["symbol"].tolist()
   ```

3. **In `pipeline/trading_cycle_v2.py::ingest_data`:**
   ```python
   from src.assembled_core.data.universe import get_universe_members_pit
   universe = get_universe_members_pit(ctx.as_of)
   ```

4. **Übergangs-Backtest:** Vergleichs-Run mit alter Watchlist vs PIT, Performance-Differenz dokumentieren.

5. **Test:**
   ```python
   # TSLA in S&P500 ab 2020-12-21
   assert "TSLA" not in get_universe_members_pit("2020-12-20")
   assert "TSLA" in get_universe_members_pit("2020-12-22")
   ```

**Akzeptanzkriterien:**

- [ ] `get_universe_members_pit` hat ≥1 produktiven Aufrufer in `pipeline/` oder `scripts/`
- [ ] `configs/universes/historical_membership.parquet` (oder analog) existiert
- [ ] Test mit bekanntem historischem Membership-Event grün
- [ ] Backtest-Vergleich Watchlist vs PIT als Markdown-Dokument in `docs/audit/`

**Wenn Schritt 1 nicht sofort möglich:** Mache mindestens Schritt 5 (Test mit synthetischen historischen Memberships) und dokumentiere in `KNOWN_ISSUES.md`, dass Survivorship-Bias bekannt und nicht behoben ist.

---

### A11 — `validate_price_data` definiert, aber nirgends aufgerufen

**Datei:** `src/assembled_core/data/prices_ingest.py:203` (Definition), `:141` (`print()` statt logger)
**Schwere:** mittel-akut
**Schicht:** 7
**Aufwand:** 2h

**Was ist das Problem?**

Validierungs-Funktion mit NaN-Check, Negative-Preise-Check, Zero-Volume-Quote-Check. **Null Aufrufer.** Plus: `load_eod_prices` Zeile 141 macht eine Mini-Validation und meldet ungültige OHLC mit `print()` statt `logger.warning()`, **ohne dagegen zu handeln**.

**Warum gefährlich?**

Korrupte Daten gehen ungehindert durch. Bug in Datenquelle (Negativpreise, NaN, vertauschte high/low) infiziert den ganzen Backtest unbemerkt.

**Behebung:**

1. **In `load_eod_prices`:**
   ```python
   def load_eod_prices(...) -> pd.DataFrame:
       df = ...
       validation = validate_price_data(df)
       if not validation["valid"]:
           issues = validation.get("issues", [])
           logger.warning("[load_eod_prices] Issues: %s", issues)
           if config.strict_data_validation:
               raise DataQualityError(f"Validation failed: {issues}")
       return df
   ```

2. **`print()` in Zeile 141 ersetzen:**
   ```python
   if invalid.any():
       invalid_count = invalid.sum()
       logger.warning(
           "[PRICES] %d rows with invalid OHLC relationships",
           invalid_count,
       )
       if config.drop_invalid_ohlc:
           df = df[~invalid].copy()
   ```

3. **In `pipeline/trading_cycle_v2.ingest_data`:** Validation-Step ausdrücklich aufrufen.

4. **Test:** `tests/test_data_validation_wired.py`
   - Synthetische Preise mit NaN/Negativen
   - Assert: `validate_price_data` wird aufgerufen
   - Assert: Bei `strict_qc=True` wird `DataQualityError` geworfen

**Akzeptanzkriterien:**

- [ ] `grep -rn "validate_price_data" src/` liefert ≥3 Aufrufer (außer Definition)
- [ ] `grep -n 'print(' src/assembled_core/data/prices_ingest.py` liefert 0 Treffer
- [ ] `tests/test_data_validation_wired.py` existiert und ist grün

---

### A12 — `compliance/`-Modul broken: `__all__`-Export ohne Implementierung

**Datei:** `src/assembled_core/compliance/__init__.py`, archivierte Module in `archive/observability_graveyard_2026q2/compliance/`
**Schwere:** akut
**Schicht:** 7 (NEU in v2)
**Aufwand:** 1h

**Was ist das Problem?**

Verifiziert mit Live-Test:
```python
>>> from src.assembled_core.compliance import AuditLog
ImportError: cannot import name 'AuditLog' from 'src.assembled_core.compliance'
```

`compliance/__init__.py` exportiert 14 Namen via `__all__`:
```python
__all__ = [
    "AuditEntry", "AuditEventType", "AuditLog",
    "OTRAlertLevel", "OTRMonitor", "OTRSnapshot",
    "BestExecutionReport", "ModelInventoryEntry", "ModelInventoryReport",
    "RiskReport", "TransactionCostReport",
    "generate_best_execution_report", "generate_model_inventory",
    "generate_risk_report", "generate_transaction_cost_report",
]
```

Aber das Verzeichnis enthält **nur `__init__.py`**. Die echten Module (`audit_log.py`, `otr_monitor.py`, `regulatory_reports.py`) wurden archiviert nach `archive/observability_graveyard_2026q2/compliance/`.

Plus: `tests/test_compliance.py:17,21,25` importiert von Pfaden, die nicht existieren:
```python
from src.assembled_core.compliance.audit_log import (...)
from src.assembled_core.compliance.otr_monitor import (...)
from src.assembled_core.compliance.regulatory_reports import (...)
```

Der Test wird daher silently mit `0 items / 1 skipped` collected, nicht als Fehler markiert.

**Warum gefährlich?**

Wenn ein neuer Aufrufer (oder Cursor Cloud, oder externer Agent) versucht, `from src.assembled_core.compliance import AuditLog`, gibt es ImportError zur Laufzeit. Plus: Test-Coverage ist eine Lüge — `tests/test_compliance.py` beansprucht 600+ Zeilen Tests, die nicht laufen.

**Behebung:**

Drei Optionen, eine wählen:

**Option 1: Compliance-Modul wiederherstellen** (wenn benötigt)
```bash
git mv archive/observability_graveyard_2026q2/compliance/* src/assembled_core/compliance/
```
Dann verkabeln in `pipeline/` oder `ops/`.

**Option 2: Compliance ganz archivieren** (wenn nicht benötigt)
```bash
git mv src/assembled_core/compliance archive/observability_graveyard_2026q2/compliance_init
git mv tests/test_compliance.py archive/observability_graveyard_2026q2/test_compliance.py
# Optional: in deferred-imports von api/ checken, ob compliance noch referenziert wird
```

**Option 3: Stub-Implementierung** (wenn unsicher, später entscheiden)

`compliance/__init__.py` leeren oder mit klarer Doku versehen:
```python
"""Compliance module — currently archived, see archive/observability_graveyard_2026q2/."""
__all__ = []
```

**Akzeptanzkriterien:**

- [ ] `python -c "from src.assembled_core.compliance import AuditLog"` läuft ohne ImportError ODER `compliance/` ist archiviert
- [ ] `tests/test_compliance.py` läuft mit `> 0 tests collected` ODER ist archiviert
- [ ] `__all__` in `compliance/__init__.py` matched mit tatsächlich verfügbaren Namen
- [ ] AGENTS.md/KNOWN_ISSUES dokumentiert die Entscheidung

---

### A13 — Dual pytest-Konfiguration zwischen `pytest.ini` und `pyproject.toml`

**Datei:** `pytest.ini` und `pyproject.toml` `[tool.pytest.ini_options]`
**Schwere:** akut
**Schicht:** 7 (NEU in v2)
**Aufwand:** 1h

**Was ist das Problem?**

Verifiziert per `pytest --collect-only`-Run:
```
WARNING: ignoring pytest config in pyproject.toml!
```

Zwei pytest-Konfigurationen konfligieren:

| | pytest.ini | pyproject.toml |
|---|---|---|
| Marker-Anzahl | 27 | 11 |
| `--strict-markers` | ja (in addopts) | nein |
| `characterization` Marker | **NICHT deklariert** | **deklariert** |
| `phase4-13` Marker | deklariert | deklariert |
| `fast`, `regression`, `integration` etc. | deklariert | NICHT deklariert |

pytest priorisiert `pytest.ini` und ignoriert `pyproject.toml`. Daher wirkt die `pyproject.toml`-Deklaration von `characterization` nicht — was A5 verursacht.

**Warum gefährlich?**

Zwei verschiedene Konfig-Dateien geben zwei Wahrheiten. Je nach pytest-Version (oder zukünftigem Pre-Commit-Hook, der pyproject.toml direkt liest), kann sich das Verhalten ändern. Außerdem:
- Wer pyproject.toml editiert, denkt es wirkt — wirkt aber nicht
- Wer pytest.ini editiert, vergisst, dass pyproject.toml dasselbe duplikiert
- Beide divergieren über die Zeit weiter (siehe Marker-Drift)

**Behebung:**

Eine Quelle der Wahrheit. **Empfehlung: pyproject.toml als kanonisch**, da das der moderne Python-Standard ist (PEP 621).

1. **Inhalt von pytest.ini in pyproject.toml mergen:**
   ```toml
   [tool.pytest.ini_options]
   addopts = "-q --strict-markers -m 'not external' --tb=short"
   markers = [
       # ALLE Marker aus pytest.ini hier
       "fast: ...",
       "integration: ...",
       "regression: ...",
       "characterization: Golden-master snapshot tests",
       "phase4: ...",
       # ... alle 27
   ]
   filterwarnings = [
       "error::DeprecationWarning:src.assembled_core.pipeline.*",
       # siehe A6
   ]
   ```

2. **`pytest.ini` löschen:**
   ```bash
   git rm pytest.ini
   ```

3. **Verifizieren:** `pytest --collect-only` zeigt **keine** Warning mehr, alle Marker sind erkannt.

**Akzeptanzkriterien:**

- [ ] `pytest.ini` ist gelöscht
- [ ] Alle 27 Marker sind in `pyproject.toml` `[tool.pytest.ini_options].markers`
- [ ] `pytest --collect-only 2>&1 | grep -i "ignoring pytest config"` liefert 0 Treffer
- [ ] `pytest -m characterization --collect-only --strict-markers` zeigt > 15 Tests
- [ ] CI bleibt grün

**Hinweis:** A5 wird durch A13 mitgelöst, wenn man hier alle Marker korrekt in pyproject.toml zusammenführt. A5 separat lassen, falls A13 zu groß ist und du erstmal nur in `pytest.ini` den `characterization`-Marker ergänzen willst.

---

### A14 — `orchestrator.py` (parallele EOD-Pipeline) hat ebenfalls keine Slippage

**Datei:** `src/assembled_core/pipeline/orchestrator.py`
**Schwere:** akut
**Schicht:** 7 (NEU in v2)
**Aufwand:** 2h (zusätzlich zu A8)

**Was ist das Problem?**

Verifiziert: `grep "slippage\|add_cost_columns" src/assembled_core/pipeline/orchestrator.py` liefert **0 Treffer**.

`orchestrator.py` ist die zweite EOD-Pipeline (siehe B5) und wird vom CLI-Befehl `assembled-run-daily` über `scripts/run_eod_pipeline.py` aufgerufen. Damit ist diese Pipeline **die offiziell gestartete** — und sie ignoriert Slippage komplett.

A8 betrifft `trading_cycle_v2`. A14 ist das Schwester-Problem in der parallelen Pipeline.

**Warum gefährlich?**

Bei jedem `assembled-run-daily`-Aufruf werden Order-Listen ohne Slippage erzeugt. In Live-Mode geht es zum Broker, der reale Slippage liefert — OK. In Backtest-Mode (falls vorhanden in orchestrator) sind Returns systematisch zu optimistisch.

**Behebung:**

Analog zu A8: in `orchestrator.py` an der Order-Generierung `add_cost_columns_to_trades` aufrufen, wenn nicht-Live.

```python
# In orchestrator.run_eod_pipeline, nach Order-Generierung:
if not is_live_mode:
    from src.assembled_core.execution.transaction_costs import (
        add_cost_columns_to_trades,
    )
    orders = add_cost_columns_to_trades(orders, ...)
```

Falls B5 (Pipeline-Konsolidierung) bald passiert: A14 dort mit lösen, statt zwei Stellen separat zu fixen.

**Akzeptanzkriterien:**

- [ ] `grep "add_cost_columns_to_trades\|slippage" src/assembled_core/pipeline/orchestrator.py` liefert ≥1 Treffer
- [ ] ODER: B5 wurde umgesetzt und es gibt nur noch eine EOD-Pipeline (siehe A8)

---

## Cluster B — Architektur-Konsolidierung

8 Punkte zu Pipeline-Duplikation, Schema-Drift und unverkabeltem Code.

---

### B1 — Drei Quellen der Wahrheit für Dependencies, numpy Major-Konflikt

**Datei:** `pyproject.toml`, `requirements.txt`, `requirements.lock`
**Schwere:** mittel
**Schicht:** 1
**Aufwand:** 3h

**Was ist das Problem?**

| Package | pyproject.toml | requirements.txt | requirements.lock |
|---|---|---|---|
| numpy | `>=1.24.0` | `==2.3.3` | `==1.26.4` |
| pandas | `>=2.0.0` | `==2.3.3` | `==2.2.3` |
| pyarrow | `>=10.0.0` | `==21.0.0` | `==22.0.0` |
| alpaca-py | `>=0.30.0` | `==0.38.0` | `==0.43.2` |
| polygon-api-client | `>=1.12.0` | `==1.14.4` | `==1.16.3` |

numpy ist Major-Sprung (1.x ↔ 2.x). Wer welche Datei nutzt, bekommt unterschiedliche numerische Welten.

**Behebung:**

1. **`requirements.txt` durch pip-compile generiert:**
   ```bash
   pip install pip-tools
   pip-compile pyproject.toml -o requirements.txt
   ```

2. **`requirements.lock` regeneriert:**
   ```bash
   python -m venv .venv-fresh
   source .venv-fresh/bin/activate
   pip install -e ".[dev,ml,scipy]"
   pip freeze | grep -v "^-e " > requirements.lock
   ```

3. **CI:** Nutze `pyproject.toml` + `requirements.lock` als Constraint:
   ```bash
   pip install -e ".[dev]" -c requirements.lock
   ```

4. **README:** dokumentiere `pyproject.toml` als einzige Quelle.

**Akzeptanzkriterien:**

- [ ] `requirements.txt` ist gelöscht ODER explizit als auto-generiert markiert (Header-Kommentar)
- [ ] Diff zwischen `requirements.txt` und `requirements.lock` zeigt 0 Versions-Mismatches
- [ ] CI nutzt nur eine Constraint-Datei
- [ ] README dokumentiert die Convention

---

### B2 — 30 weitere non-atomic JSON-Writes

**Datei:** Liste unten
**Schwere:** mittel
**Schicht:** 4
**Aufwand:** 6h

**Was ist das Problem?**

Zwei saubere atomic-write-Helper existieren:
- `risk/state_machine.py::atomic_write_json_with_retry`
- `events/crisis_alpha/state_machine.py::_atomic_write_json`

30 andere Stellen schreiben JSON ohne tmp+rename. Kritische:

| Datei | Zeile | Kritikalität |
|---|---|---|
| `intel/pit_store.py` | 65 | **hoch** (PIT-Reproduzierbarkeit) |
| `execution/unified_paper_engine.py` | 376, 1760, 1882 | **hoch** (Paper-State) |
| `intel/news_dedupe.py` | 212 | mittel |
| `data/factor_store.py` | 229 | mittel |
| `accounting/accounting_report.py` | 393 | mittel |
| `accounting/reconciliation_report.py` | 335 | mittel |
| `qa/walk_forward.py` | 1011, 1032 | niedrig |
| `qa/trade_tca.py` | 209 | niedrig |
| `qa/experiment_tracking.py` | 310 | niedrig |
| `qa/data_qc.py` | 719 | niedrig |
| `utils/timing.py` | 127 | niedrig |

**Behebung:**

1. **Konsolidieren:** Verschiebe `atomic_write_json_with_retry` von `risk/state_machine.py` nach `utils/atomic_io.py`. Beide bestehenden Caller passen Imports an.

2. **Refactor priorisiert:** Beginne mit **hoch**:
   - `intel/pit_store.py`
   - `execution/unified_paper_engine.py`

3. **Test-Pattern pro Datei:**
   ```python
   def test_atomic_write_handles_crash(tmp_path, monkeypatch):
       def fail_replace(*args, **kwargs):
           raise OSError("Simulated crash")
       monkeypatch.setattr("os.replace", fail_replace)
       
       with pytest.raises(OSError):
           write_state(path, {"key": "value"})
       
       assert not path.exists() or path.read_text() != "{...incomplete..."
   ```

**Akzeptanzkriterien:**

- [ ] `utils/atomic_io.py` mit kanonischem Helper existiert
- [ ] Alle 4 **hoch**-Stellen migriert (pit_store + 3× unified_paper_engine)
- [ ] Mindestens 3 **mittel**-Stellen migriert
- [ ] Mindestens 3 Crash-Simulations-Tests existieren

---

### B3 — `NewsEvent` Schema-Drift zwischen `intel/` und `events/news/`

**Datei:** `src/assembled_core/intel/models.py`, `src/assembled_core/events/news/models.py`
**Schwere:** mittel
**Schicht:** 4
**Aufwand:** 6-10h

**Was ist das Problem?**

Zwei `NewsEvent`-Klassen mit gleichem Namen, unterschiedlichen Feldern. 26 Felder Unterschied. Beide werden aktiv benutzt:
- `events/news/run_news_pipeline()` (von `ops/intel_orchestrator`, `paper/intel_runner`, `cli.py`)
- `intel/news_*.py` (von `pipeline/orchestrator.py`, `scripts/run_intel_cycle.py`)

**Behebung:**

**Phase 1: Mapping-Layer** (kurzfristig, 2h) — neue Datei `events/news/compat.py` mit `intel_to_new()` und `new_to_intel()`.

**Phase 2: Architektur-Entscheid** — welches Schema kanonisch? Empfehlung: vereinheitlichen zu Pydantic-Modell mit `datetime`-TZ-aware statt Strings, alle Felder optional.

**Phase 3: Migration eine Richtung** (4-8h) — analog `trading_cycle_v2`-Migration. Schrittweise alte Module umstellen.

**Phase 4: Schema-Validierung** in CI:
```python
def test_newsevent_unified():
    from src.assembled_core.intel.models import NewsEvent as A
    from src.assembled_core.events.news.models import NewsEvent as B
    assert A is B
```

**Akzeptanzkriterien:**

- [ ] `python -c "from src.assembled_core.intel.models import NewsEvent as A; from src.assembled_core.events.news.models import NewsEvent as B; assert A is B"` läuft fehlerfrei (langfristig)
- [ ] ODER: `compat.py` mit Round-Trip-Tests existiert (Übergangs-Lösung)
- [ ] Mindestens 5 Round-Trip-Tests ohne Datenverlust

---

### B4 — `predict_regime` doppelt implementiert

**Datei:** `src/assembled_core/ml/regime_hmm.py` (416 LOC), `src/assembled_core/risk/regime_hmm.py` (163 LOC)
**Schwere:** mittel
**Schicht:** 3
**Aufwand:** 3h

**Was ist das Problem?**

Zwei HMM-Implementierungen mit gleicher Methode `predict_regime`. Beide nutzen `hmmlearn`.

**Behebung:**

1. **Diff:** `diff src/assembled_core/ml/regime_hmm.py src/assembled_core/risk/regime_hmm.py`

2. **Konsolidieren:**
   - Variante A: `risk/regime_hmm.py` löschen, alle Importer auf `ml/regime_hmm.py`
   - Variante B: `risk/regime_hmm.py` zur Re-Export-Shim:
     ```python
     """DEPRECATED: use ml/regime_hmm.py."""
     import warnings
     warnings.warn("risk.regime_hmm is deprecated", DeprecationWarning)
     from src.assembled_core.ml.regime_hmm import RegimeHMM, predict_regime
     ```

3. **Importer auditieren:** `grep -rn "from src.assembled_core.risk.regime_hmm\|from src.assembled_core.ml.regime_hmm"` und entscheiden.

**Akzeptanzkriterien:**

- [ ] Nur eine kanonische `predict_regime`-Implementierung
- [ ] Andere Datei ist gelöscht ODER zur Re-Export-Shim mit `DeprecationWarning`
- [ ] Alle Importer ziehen vom kanonischen Pfad

---

### B5 — `orchestrator.py` als parallele Pipeline zu `trading_cycle_v2`

**Datei:** `src/assembled_core/pipeline/orchestrator.py` (1.351 LOC), `src/assembled_core/pipeline/trading_cycle_v2.py` (2.114 LOC)
**Schwere:** mittel-hoch
**Schicht:** 2
**Aufwand:** 12-20h

**Was ist das Problem?**

Zwei parallele EOD-Pipelines:

| | trading_cycle_v2 | orchestrator |
|---|---|---|
| LOC | 2.114 (modular) | 1.351 (monolithisch) |
| Genutzt von | `scripts/run_daily.py`, `scripts/run_paper_track.py` | `scripts/run_eod_pipeline.py` |
| CLI-Befehl | (kein direkter) | **`assembled-run-daily`** |
| News-Pipeline | `events/news/` | `intel/news_*` |
| Slippage | nein (siehe A8) | nein (siehe A14) |

`orchestrator.py:523` Kommentar: `# Phase 9: Post-signal enrichment (same as trading_cycle)`.

**Behebung:**

**Phase 0: Diagnose** (1h)
```bash
python -X importtime scripts/run_daily.py 2>&1 | head -100 > /tmp/v2_imports.log
python -X importtime scripts/run_eod_pipeline.py 2>&1 | head -100 > /tmp/orch_imports.log
diff /tmp/v2_imports.log /tmp/orch_imports.log
```

**Phase 1: Architektur-Entscheidung**
- **Option A:** orchestrator ist Legacy → migriere identisch zur `trading_cycle.py`-Migration
- **Option B:** beide bleiben für unterschiedliche Use Cases → Doku klarstellen, gemeinsame Logik nach `pipeline/_shared_eod.py`

**Empfehlung: Option A.**

**Phase 2-4** analog `MIGRATION_TRADING_CYCLE_V2.md`.

**Akzeptanzkriterien:**

- [ ] Eine `run_eod_pipeline`-Implementierung ODER klare Dokumentation der zwei Use Cases
- [ ] `assembled-run-daily` und `scripts/run_daily.py` zeigen auf gleichen Code-Pfad ODER Dokumentation der Differenz
- [ ] AGENTS.md, README.md aktualisiert

---

### B6 — `intel/nation_profiles.py` als Migrations-Waise

**Datei:** `src/assembled_core/intel/nation_profiles.py`
**Schwere:** niedrig
**Schicht:** 4
**Aufwand:** 30min

**Was ist das Problem?**

Während `trading_cycle.py`-Migration wurde der Aufrufer korrekterweise gestrichen (Observability-Code). 209 LOC, 0 Aufrufer.

**Behebung:**

1. Verifizieren:
   ```bash
   grep -rn "nation_profiles\|load_nation_profiles\|compute_vulnerability_score" src/ scripts/ tests/ 2>/dev/null
   ```
2. Archivieren:
   ```bash
   git mv src/assembled_core/intel/nation_profiles.py archive/intel_research_2026q2/intel/nation_profiles.py
   git mv configs/nation_profiles.yaml archive/intel_research_2026q2/configs/nation_profiles.yaml
   ```

**Akzeptanzkriterien:**

- [ ] `nation_profiles.py` nicht mehr in `src/`
- [ ] Tests grün

---

### B7 — 34+ Tests testen unverkabelten Code (Pseudo-Coverage)

**Datei:** mehrere
**Schwere:** mittel
**Schicht:** 6
**Aufwand:** 4h

**Was ist das Problem?**

Spotcheck:
| Funktion | Tests | Aufrufer in `src/` |
|---|---|---|
| `build_client_order_id` | 5 | 0 |
| `is_duplicate_error` | 2 | 0 |
| `apply_splits_for_research_prices` | 7 | 0 |
| `compute_dividend_cashflows` | 8 | 0 |
| `apply_delisting_exits` | 6 | 0 |
| `apply_spinoff` | 6 | 0 |
| `load_nation_profiles` | 0 | 0 |

**Behebung:**

Drei Optionen pro Funktion:
- **Option 1:** Verkabeln (siehe A2, A7)
- **Option 2:** Mit `@pytest.mark.unwired_code` markieren
  ```python
  # In pyproject.toml markers:
  "unwired_code: tests for code defined but not yet wired into pipeline"
  
  @pytest.mark.unwired_code
  def test_compute_dividend_cashflows():
      ...
  ```
- **Option 3:** Archivieren falls niemals verkabelt

**Akzeptanzkriterien:**

- [ ] `unwired_code`-Marker registriert
- [ ] Mindestens 5 Tests damit markiert
- [ ] CI-Output zeigt: `pytest -m unwired_code --collect-only` listet Pseudo-Coverage transparent

---

### B8 — Zwei Kill-Switch-Systeme mit unterschiedlicher Atomicity

**Datei:** `src/assembled_core/execution/kill_switch.py`, `symbol_kill_switch.py`
**Schwere:** mittel
**Schicht:** 7
**Aufwand:** 2h (nach A3)

**Was ist das Problem?**

| | kill_switch.py | symbol_kill_switch.py |
|---|---|---|
| Scope | Global | Pro Symbol |
| Atomic write | ja ✓ | nein ✗ (siehe A3) |
| Audit-Log | ja | nein |

**Behebung (nach A3):**

1. **Konsistente API-Naming:**
   - `is_kill_switch_engaged()` ↔ `is_symbol_blocked(symbol)`
   - `activate_kill_switch()` ↔ `block_symbol(symbol, reason)`

2. **Gemeinsamer Audit-Log** in `execution/_kill_switch_audit.py`.

3. **Ein gemeinsamer Filter-Punkt:**
   ```python
   def filter_orders_with_kill_switches(orders):
       if is_kill_switch_engaged():
           return orders.iloc[0:0]
       blocked = get_blocked_symbols()
       return orders[~orders["symbol"].isin(blocked)]
   ```

**Akzeptanzkriterien:**

- [ ] Beide Switches haben atomic writes (durch A3)
- [ ] Konsistente API-Naming
- [ ] Ein gemeinsamer Filter-Punkt in Pipeline

---

## Cluster C — Architektur-Hygiene

6 Punkte. Quartalsplanung.

---

### C1 — `size_positions()` mit Cyclomatic Complexity 236

**Datei:** `src/assembled_core/pipeline/trading_cycle_v2.py:1004`
**Schwere:** mittel
**Schicht:** 3
**Aufwand:** 16-24h

**Was ist das Problem?**

577 Zeilen Funktion, CC=236. 14+ Sub-Schritte hintereinander. Direkt aus alter `trading_cycle.py` mitgenommen ohne Zerlegung.

**Behebung:**

Inkrementell, eine Sub-Funktion pro Sprint. 14 Phasen:

```python
# Phase 1: _apply_sizing_method
# Phase 2: _apply_liquidity_scaling
# Phase 3: _apply_exposure_overlay
# Phase 4: _apply_factor_risk_model
# Phase 5: _apply_trailing_stops
# Phase 6: _apply_turnover_budget
# Phase 7: _apply_correlation_guard
# Phase 8: _apply_crash_prediction_cap
# Phase 9: _apply_inverse_etf_hedge
# Phase 10: _apply_quantile_asymmetry
# Phase 11: _apply_crowding_detector
# Phase 12: _apply_crisis_alpha_cap
# Phase 13: _apply_rebalance_trigger
# Phase 14: _apply_cost_aware_shrinkage
```

Pro Phase: Charakterisierungstest vorher snapshotten, dann extrahieren, gegen Snapshot validieren.

**Akzeptanzkriterien:**

- [ ] `size_positions()` selbst hat CC < 30
- [ ] ≥14 extrahierte Sub-Funktionen, jeweils CC < 20
- [ ] Charakterisierungstest pro Sub-Funktion
- [ ] `pipeline/trading_cycle_v2.py` LOC reduziert (aktuell 2114 → Ziel ~1500)

---

### C2 — 122 `.iterrows()`-Aufrufe

**Datei:** Hotspots: `pipeline/trading_cycle_v2.py` (19), `execution/unified_paper_engine.py` (14), `data/corporate_actions.py` (7), `portfolio/position_sizing.py` (5)
**Schwere:** niedrig
**Schicht:** 3
**Aufwand:** 12h verteilt

**Behebung:**

Profile zuerst:
```bash
python -m cProfile -o profile.out scripts/run_backtest_strategy.py --freq 1d --start-capital 10000 --universe configs/security_master.csv
python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(30)" | grep -i iterrows
```

Top-5 nach Cumulative-Time vektorisieren:
```python
# OLD:
for idx, row in df.iterrows():
    if row["weight"] > 0.1:
        df.at[idx, "scaled"] = row["weight"] * 0.5

# NEW:
df["scaled"] = np.where(df["weight"] > 0.1, df["weight"] * 0.5, df["weight"])
```

**Akzeptanzkriterien:**

- [ ] Top-10-iterrows-Hotspots vektorisiert
- [ ] Backtest-Performance gemessen vor/nach (3 Runs mit `time`)
- [ ] ≥30% Beschleunigung

---

### C3 — Top 7 Funktionen alle >500 LOC

**Datei:** verschiedene
**Schwere:** mittel
**Schicht:** 2
**Aufwand:** quartalsweise

| Funktion | LOC |
|---|---|
| `pre_trade_checks.run_pre_trade_checks` | 959 |
| `qa.run_portfolio_backtest` | 955 |
| `ops.run_paper_daily_one` | 728 |
| `pipeline.orchestrator.run_eod_pipeline` | 702 |
| `events/news/pipeline.run_news_pipeline` | 588 |
| `pipeline/trading_cycle_v2.size_positions` | 577 |
| `paper/paper_track.run_paper_day` | 508 |

**Behebung:**

Wie C1, aber pro Funktion. Ziel: <200 LOC pro Funktion. Einigen davon synergiert mit B5.

**Akzeptanzkriterien:**

- [ ] Alle 7 Funktionen <200 LOC
- [ ] CC pro Funktion <30

---

### C4 — `scripts/cli.py` mit 4007 LOC

**Datei:** `scripts/cli.py`
**Schwere:** niedrig
**Schicht:** 1
**Aufwand:** 6-10h

**Behebung:**

```
scripts/
├── cli.py                 # Argparse-Setup + Dispatcher (~300 LOC)
└── commands/
    ├── __init__.py
    ├── run_daily.py
    ├── run_backtest.py
    ├── run_news.py
    └── ...
```

**Akzeptanzkriterien:**

- [ ] `scripts/cli.py` <500 LOC
- [ ] ≥5 Subcommand-Module
- [ ] Alle CLI-Tests grün

---

### C5 — 199 Doku-Dateien, AGENTS.md mit 60% falschen Zahlen

**Datei:** `AGENTS.md`, `KNOWN_ISSUES.md`, `PROJEKT_STATUS.md`, `docs/`
**Schwere:** mittel
**Schicht:** 1
**Aufwand:** 4h

**Was ist das Problem?**

`AGENTS.md`:
- 22 Kernmodule → real **25**
- ~50 Scripts → real **90**
- ~330 Testdateien → real **547**
- 9 CI-Workflows → real **17**

`KNOWN_ISSUES.md`, `PROJEKT_STATUS.md` Header-Datum **2025-01-15**.

**Behebung:**

1. **Auto-Generator** `scripts/regenerate_agents_stats.py`:
   ```python
   stats = {
       "core_modules": len(list(Path("src/assembled_core").glob("*/"))),
       "scripts": len(list(Path("scripts").glob("*.py"))),
       "test_files": len(list(Path("tests").rglob("test_*.py"))),
       "ci_workflows": len(list(Path(".github/workflows").glob("*.yml"))),
   }
   ```

2. **Header-Daten** entfernen oder als "Letzte Strukturänderung" markieren.

3. **`docs/index.md`** als single entry point.

**Akzeptanzkriterien:**

- [ ] AGENTS.md-Zahlen aktuell oder auto-generiert
- [ ] Header-Daten aktualisiert oder entfernt
- [ ] `docs/index.md` existiert

---

### C6 — 204 Heartbeat-Commits

**Datei:** `.github/workflows/nightly-sync.yml`
**Schwere:** niedrig
**Schicht:** 5
**Aufwand:** 1h

**Behebung:**

Empfehlung: einfach abschalten.
```yaml
# .github/workflows/nightly-sync.yml
# Beim "Write heartbeat" Step entfernen oder ganzen Workflow disablen.
```

**Akzeptanzkriterien:**

- [ ] `nightly-sync.yml` schreibt nicht mehr in `.github/heartbeat.txt`
- [ ] Keine neuen "CI: nightly heartbeat"-Commits ab dem Fix-Datum

---

## Cluster D — Selbst-Audits & Strukturen

20 kleinere Hygiene-Punkte. Kontinuierlich.

---

### D1 — `KNOWN_ISSUES.md` und `PROJEKT_STATUS.md` Header von Januar 2025

**Aufwand:** 10min
**Fix:** Header-Datum entfernen oder als "Letzte Strukturänderung" markieren.

**Akzeptanzkriterien:**
- [ ] Beide Dateien haben aktuelles Datum oder kein Datum

---

### D2 — 47 silent `except: pass`-Stellen

**Aufwand:** 3h

**Fix:** Jede Stelle bekommt mindestens `logger.debug()` davor:
```python
# OLD:
try:
    ...
except OSError:
    pass

# NEW:
try:
    ...
except OSError as e:
    logger.debug("[module_name] silent skip on %s: %s", op, e)
```

**Akzeptanzkriterien:**
- [ ] `grep -B 1 "    pass$" src/assembled_core/ -r --include="*.py" | grep -B 1 "except"` zeigt vor jedem `pass` ein `logger.debug` (oder begründet anderes)

---

### D3 — Property-Tests in 0.4% der Files

**Aufwand:** kontinuierlich, 4h pro neuem

**Fix:** Mindestens für `qa/`, `risk/`, `portfolio/` Property-Tests einführen.

**Akzeptanzkriterien:**
- [ ] ≥10 Test-Files nutzen Hypothesis (aktuell 2)
- [ ] `qa/`, `risk/`, `portfolio/` haben jeweils mindestens 1 Property-Test

---

### D4 — 191 weak `assert not df.empty`-Stellen

**Aufwand:** kontinuierlich

**Fix:** Pre-commit-Hook oder CI-Lint, der neue Tests mit nur `assert not df.empty` blockt.

**Akzeptanzkriterien:**
- [ ] Pre-commit-Hook oder CI-Check existiert

---

### D5 — `_filter_prices_for_as_of` `<=` ohne Bar-Konvention

**Datei:** `src/assembled_core/pipeline/trading_cycle_shared.py:393`
**Aufwand:** 1h

**Fix:** Docstring + Test mit Bar-Open vs Bar-Close.

**Akzeptanzkriterien:**
- [ ] Docstring dokumentiert Bar-Konvention explizit
- [ ] Test in `tests/test_pit_filter_bar_convention.py`

---

### D6 — CI-Python-Inkonsistenz

**Datei:** `.github/workflows/*.yml`
**Aufwand:** 30min

**Fix:**
- 14 Workflows auf 3.11 (oder Matrix [3.10, 3.11, 3.12])
- `repo-health.yml`: 3.13 → 3.11
- `nightly-sync.yml`: explizite Version

**Akzeptanzkriterien:**
- [ ] `grep -E "python-version" .github/workflows/*.yml | grep -v "3.11\|matrix"` liefert nur dokumentierte Ausnahmen
- [ ] `nightly-sync.yml` hat explizite Python-Version

---

### D7 — 5 `inplace=True` (deprecated pandas 3.0)

**Datei:** `features/ta_liquidity_vol_factors.py:351,386,419,461`, `qa/factor_analysis.py:2176`
**Aufwand:** 30min

**Fix:** Reassignment statt `inplace=True` (siehe pandas 3.0 Release Notes — `inplace=True` bleibt zwar erhalten, aber CoW macht chained inplace zur Falle).

**Akzeptanzkriterien:**
- [ ] `grep -rn "inplace=True" src/assembled_core/` liefert 0 Treffer
- [ ] Tests grün

---

### D8 — 3 tote Konfig-YAMLs

**Aufwand:** 10min

**Fix:** `git mv configs/.../X.yaml configs/_backlog/X.yaml` mit README, der erklärt warum.

**Akzeptanzkriterien:**
- [ ] `configs/_backlog/README.md` existiert
- [ ] Drei Dateien dort

---

### D9 — Verzeichnisse mit Leerzeichen

**Aufwand:** 30min

**Fix:**
```bash
git mv "autonome weiterarbeit" autonome_weiterarbeit
git mv "datensammlungen/altdaten/stand 3-12-2025" datensammlungen/altdaten/2025-12-03
grep -rn "autonome weiterarbeit\|stand 3-12-2025" .  # check refs
```

**Akzeptanzkriterien:**
- [ ] `find . -type d -name "* *" -not -path "./.git/*"` liefert 0 Treffer
- [ ] Tests grün

---

### D10 — 200KB-Datei mit `:`-im-Namen

**Aufwand:** 5min

**Fix:**
```bash
git rm "F:Python_ProjektAktiengerüst__profile_out.txt"
echo "*profile_out*.txt" >> .gitignore
```

**Akzeptanzkriterien:**
- [ ] Datei gelöscht
- [ ] Gitignore-Regel existiert

---

### D11 — 45 Parquets im Repo

**Aufwand:** 1h

**Fix:** Smoke-Daten nach `tests/fixtures/`, Rest in `.gitignore`, Daten via `scripts/download_historical_snapshot.py` laden.

**Akzeptanzkriterien:**
- [ ] `find . -name "*.parquet" -not -path "./tests/fixtures/*" -not -path "./.git/*"` liefert 0 Treffer
- [ ] `scripts/download_historical_snapshot.py` dokumentiert in README

---

### D12 — `external` Marker existiert, wird nicht benutzt

**Aufwand:** 5min

**Fix:** Marker aus `pytest.ini` (oder `pyproject.toml` nach A13) entfernen.

**Akzeptanzkriterien:**
- [ ] `external` ist nicht mehr in markers
- [ ] `-m "not external"` ist nicht mehr in addopts

---

### D13 — `run_id` ohne Timezone

**Datei:** `src/assembled_core/logging_config.py::generate_run_id`
**Aufwand:** 10min

**Fix:**
```python
from datetime import datetime, timezone

def generate_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    uuid_short = str(uuid4()).replace("-", "")[:8]
    return f"{prefix}_{timestamp}_{uuid_short}"
```

**Akzeptanzkriterien:**
- [ ] `generate_run_id` nutzt `datetime.now(timezone.utc)`
- [ ] Run-ID enthält 'Z' als TZ-Marker

---

### D14 — 39/40 `pd.read_csv` ohne `dtype=`

**Aufwand:** 3h

**Fix:** Bei Symbol-Spalten und potenziell mehrdeutigen IDs explizite dtypes:
```python
df = pd.read_csv(path, dtype={
    "symbol": "string",
    "timestamp": "string",
    "qty": "float64",
})
```

**Akzeptanzkriterien:**
- [ ] Mindestens 30 von 39 `pd.read_csv` haben jetzt `dtype=`
- [ ] Pre-commit-Hook prüft neue `read_csv`-Aufrufe

---

### D15 — `mean_reversion_factors.py` rolling ohne `min_periods`

**Datei:** `src/assembled_core/features/mean_reversion_factors.py:75,76,93,94,100`
**Aufwand:** 30min

**Fix:**
```python
rolling_mean = ret_3d.rolling(60, min_periods=60).mean()  # explizit
```

Plus: `altdata_news_macro_factors.py` — `min_periods=lookback_days // 2` statt `1`.

**Akzeptanzkriterien:**
- [ ] Alle 5 `.rolling(N).mean()`/`.std()` haben explizites `min_periods`
- [ ] `altdata_news_macro_factors.py` `min_periods=1` ersetzt durch sinnvollen Wert

---

### D16 — Keine LICENSE, SECURITY.md, CONTRIBUTING.md

**Aufwand:** 30min

**Fix:**

`LICENSE` (proprietär):
```
Copyright (c) 2026 Hans
All rights reserved.
```

`SECURITY.md`:
```markdown
# Security Policy

This is a personal research project, not a production system.
For security issues, contact: <your-email>
```

**Akzeptanzkriterien:**
- [ ] LICENSE existiert
- [ ] SECURITY.md existiert
- [ ] Optional: CONTRIBUTING.md

---

### D17 — 9 `datetime.now()`-Stellen ohne Timezone (Vollliste)

**Schicht:** 3 (NEU als eigener Punkt in v2)
**Aufwand:** 1h

**Was ist das Problem?**

Vollständige Liste verifiziert per `grep`:

| Datei | Zeile |
|---|---|
| `logging_config.py` | 64 |
| `logging_config.py` | 208 |
| `ops/alert_manager.py` | 73 |
| `qa/experiment_tracking.py` | 132 |
| `qa/experiment_tracking.py` | 146 |
| `qa/experiment_tracking.py` | 198 |
| `qa/experiment_tracking.py` | 305 |
| `reports/daily_qa_report.py` | 84 |
| `reports/daily_qa_report.py` | 162 |

D13 fixt nur Stelle 64 (`generate_run_id`). Alle anderen verwenden `datetime.now()` ohne Timezone, was zu falsch sortierten Logs/Reports/IDs führt, wenn der Server in einer anderen TZ läuft als die Daten erwarten.

**Behebung:**

Globaler Replace mit grep+sed:
```bash
# Manuell pro Datei: datetime.now() → datetime.now(timezone.utc)
# Plus: import anpassen falls nicht vorhanden
```

**Akzeptanzkriterien:**

- [ ] `grep -rn "datetime\.now()" src/assembled_core/` liefert nur dokumentierte Ausnahmen
- [ ] Alle anderen Stellen nutzen `datetime.now(timezone.utc)` oder `datetime.now(tz=ZoneInfo(...))`

---

### D18 — `daily_qa_report.py:162` UTC-Lüge

**Datei:** `src/assembled_core/reports/daily_qa_report.py:162`
**Schicht:** 7 (NEU in v2)
**Aufwand:** 5min

**Was ist das Problem?**

```python
lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
```

Der Report behauptet "UTC", aber `datetime.now()` ohne `tz` ist **lokale Zeit** des Servers. Wenn der Server in Europe/Berlin läuft, sagt der Report "UTC" mit Berliner Zeit.

**Warum gefährlich?**

Audit-Tauglichkeit. Berichte, die behaupten UTC zu sein, aber Lokalzeit enthalten, sind in Compliance-Zusammenhängen problematisch.

**Behebung:**

```python
from datetime import datetime, timezone

lines.append(
    f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
)
```

**Akzeptanzkriterien:**

- [ ] Zeile 162 nutzt `datetime.now(timezone.utc)`
- [ ] Test prüft, dass der String-Output mit `UTC` endet UND der Wert TZ-aware war

---

### D19 — `asyncio.get_event_loop()` deprecated in Python 3.10+

**Datei:** `src/assembled_core/data/tier_processor.py:108,184,200`
**Schicht:** 7 (NEU in v2)
**Aufwand:** 30min

**Was ist das Problem?**

```python
data = await asyncio.get_event_loop().run_in_executor(...)
```

`asyncio.get_event_loop()` ist deprecated in Python 3.10+ und gibt deutlich andere Semantik in Python 3.12+. Im async-Kontext sollte stattdessen `asyncio.get_running_loop()` oder direkt `asyncio.to_thread()` verwendet werden.

**Behebung:**

Variante A (eleganter):
```python
# OLD:
data = await asyncio.get_event_loop().run_in_executor(None, fn, *args)

# NEW:
data = await asyncio.to_thread(fn, *args)
```

Variante B (wenn Executor-Pool gewünscht):
```python
loop = asyncio.get_running_loop()
data = await loop.run_in_executor(None, fn, *args)
```

**Akzeptanzkriterien:**

- [ ] `grep "asyncio\.get_event_loop()" src/` liefert 0 Treffer
- [ ] Alle 3 Stellen nutzen `asyncio.to_thread()` oder `get_running_loop()`
- [ ] Tests grün

---

### D20 — numpy 2.0 Readiness (NPY201 Ruff-Regel)

**Datei:** `pyproject.toml` `[tool.ruff.lint]`
**Schicht:** 7 (NEU in v2, Web-Recherche)
**Aufwand:** 1-2h

**Was ist das Problem?**

`requirements.lock` pinnt `numpy==1.26.4`, `requirements.txt` aber `numpy==2.3.3` (Major-Sprung — siehe B1). NumPy 2.0 (Juni 2024) hat ~100 entfernte/verschobene Namespace-Mitglieder. Beispiele:
- `np.float_` → `np.float64`
- `np.Inf` → `np.inf`
- `np.core` ist private (`np._core`)
- Type-Promotion-Regeln geändert (`np.float32(3) + 3.0` → `float32` statt `float64`)

Ruff hat eine dedizierte Regel `NPY201`, die Numpy-2.0-Inkompatibilitäten automatisch erkennt und teilweise fixt. Aktuell `pyproject.toml`:
```toml
[tool.ruff.lint]
select = ["E", "F"]
ignore = ["E501", "E203", "E402"]
```

**NPY201 ist nicht aktiviert.**

**Behebung:**

1. **NPY201 aktivieren:**
   ```toml
   [tool.ruff.lint]
   select = ["E", "F", "NPY201"]
   ignore = ["E501", "E203", "E402"]
   ```

2. **Auto-Fix laufen lassen:**
   ```bash
   ruff check --fix --select NPY201 src/ scripts/
   ```

3. **Übrige Verstöße manuell fixen** (Ruff zeigt nicht-auto-fixbare Stellen).

4. **Pandas 3.0 Readiness parallel:** pandas 3.0 (Januar 2026) hat `fillna(method=)` entfernt. Aktuelle Treffer:
   - Nur `triple_barrier.py:151` (siehe A1)
   - Wird durch A1 mitgelöst.

5. **CI-Check:** Ruff in CI-Pipeline sicherstellen, dass NPY201 fail-on-violation ist.

**Akzeptanzkriterien:**

- [ ] `pyproject.toml` aktiviert `NPY201` in `select`
- [ ] `ruff check --select NPY201 src/` liefert 0 Verstöße
- [ ] CI-Workflow ruft Ruff mit dieser Konfiguration auf
- [ ] `requirements.lock` und `requirements.txt` haben konsistente numpy-Version (siehe B1)

---

## Empfohlene Reihenfolge

**Sortiert nach: Quick-Wins zuerst (Aufwand < 30min), dann Schwere absteigend bei gleicher Quick-Win-Klasse.**

#### Phase 1: Quick Wins (1 Nachmittag, ~3h Gesamt)

Diese Punkte sind alle <30min und du kannst sie an einem einzigen Nachmittag erledigen:

1. **A5** (15min) — `characterization` Marker ergänzen
2. **A6** (30min) — `--disable-warnings` raus, `filterwarnings` rein
3. **D1** (10min) — Header-Daten von KNOWN_ISSUES/PROJEKT_STATUS aktualisieren
4. **D6** (30min) — CI-Python-Versionen vereinheitlichen
5. **D7** (30min) — 5× `inplace=True` ersetzen
6. **D8** (10min) — 3 tote Konfig-YAMLs nach `_backlog/`
7. **D10** (5min) — 200KB-Datei mit `:`-im-Namen löschen
8. **D12** (5min) — `external` Marker entfernen
9. **D13** (10min) — `run_id` mit TZ
10. **D15** (30min) — `mean_reversion_factors.py` `min_periods` explizit
11. **D16** (30min) — LICENSE, SECURITY.md anlegen
12. **D18** (5min) — UTC-Lüge in `daily_qa_report.py`

#### Phase 2: Akute Bugs (Wochen 1-2, ~12h)

Sortiert nach: höchstes Korrektheits-Risiko zuerst.

13. **A1** (2h) — Triple-Barrier ML-Leakage (auch pandas 3.0 prep)
14. **A11** (2h) — `validate_price_data` verkabeln
15. **A3** (1h) — `symbol_kill_switch` atomic
16. **A4** (2h) — `scenario_engine` Cholesky+Seed
17. **A12** (1h) — `compliance/` Modul fixen oder archivieren
18. **A13** (1h) — Dual pytest config konsolidieren (löst auch A5 teilweise)
19. **A14** (2h) — `orchestrator.py` Slippage (zusammen mit A8)
20. **B6** (30min) — `nation_profiles.py` archivieren

#### Phase 3: Akute Bugs Fortsetzung (Wochen 3-4, ~14h)

21. **A2** (4h) — `idempotency.py` verkabeln
22. **A8** (4h) — Slippage in v2-Backtest
23. **A7** (6h) — Corporate Actions Default

#### Phase 4: ML-Korrektheit (Wochen 5-6, ~10h)

24. **A9** (6-8h) — Walk-Forward Embargo

#### Phase 5: Architektur-Konsolidierung (Quartal 2, ~30-50h)

25. **A10** (8-40h) — PIT-Universe (datenintensiv)
26. **B1** (3h) — Dependencies konsolidieren
27. **B2** (6h) — 30 non-atomic JSON-Writes
28. **B7** (4h) — Pseudo-Coverage markieren
29. **B8** (2h) — Kill-Switch-Konsistenz
30. **B4** (3h) — `predict_regime` deduplizieren
31. **B3** (6-10h) — NewsEvent Schema-Drift
32. **D14** (3h) — `pd.read_csv` dtypes
33. **D17** (1h) — `datetime.now()` Vollliste
34. **D19** (30min) — asyncio deprecated
35. **D20** (1-2h) — numpy 2.0 Readiness

#### Phase 6: Architektur-Hygiene (Quartal 3+, ~50-100h)

36. **B5** (12-20h) — `orchestrator.py` Konsolidierung
37. **C1** (16-24h) — `size_positions()` zerlegen
38. **C2** (12h) — `.iterrows()` Hotspots
39. **C3** (quartalsweise) — 7× >500-LOC-Funktionen
40. **C4** (6-10h) — `cli.py` Subcommands
41. **C5** (4h) — Doku-Updates
42. **C6** (1h) — Heartbeat-Abschalten
43. **D2** (3h) — silent except: pass
44. **D3** (kontinuierlich) — Property-Tests ausbauen
45. **D4** (kontinuierlich) — weak asserts
46. **D5** (1h) — Bar-Konvention dokumentieren
47. **D9** (30min) — Verzeichnisse mit Leerzeichen
48. **D11** (1h) — Parquets aus Repo

---

## Anhang: Pattern-Erkenntnis

Das wiederkehrende Muster über alle 7 Diagnose-Schichten:

**Du hast Disziplin punktuell, nicht durchgängig.**

Konkrete Beispiele:
- `risk/state_machine.py` macht atomic writes — `symbol_kill_switch.py` nicht
- `idempotency.py` ist sauber implementiert und getestet — aber nicht verkabelt
- `corporate_actions.py` hat 7 Funktionen — nur 1 verkabelt, opt-in
- `data/universe.py` hat PIT-API mit Survivorship-Doku — kein Aufrufer
- Leakage-Helper für altdata existiert — nicht für TA-Features oder Walk-Forward
- characterization-Tests existieren — laufen unter keinem CI-Marker
- Property-Tests sind eingerichtet — 0.4% der Files
- compliance-Modul wurde archiviert — `__init__.py` und Tests sind stehengeblieben (broken)
- pyproject.toml hat pytest-Config — wird ignoriert wegen pytest.ini

**Hauptherausforderung dieses Projekts: nicht Code-Qualität, sondern Wiring-Disziplin.** Du baust Features, dann das nächste, und die ersten gelangen nie in den Hauptdurchstich. Tests werden trotzdem geschrieben, also sieht CI grün aus, und Funktionen werden „abgehakt".

**Empfehlung:** Vor jedem „fertig"-Status drei Fragen:

1. **Wer ruft sie produktiv?** (Nicht: wer hat einen Test dafür)
2. **Was passiert, wenn ich diese Datei lösche** und Tests laufen lasse? (Wenn alles grün → nicht verkabelt)
3. **Steht das Verkabeln in einem konkreten Sprint?** Wenn nein: `# UNWIRED: not in production pipeline` als Banner.

Das schließt die Lücke zwischen „implementiert" und „wirksam".

---

**Ende.** 48 Befunde, 7 Schichten + Web-Recherche, alle mit Datei + Zeile + Fix + Akzeptanzkriterien dokumentiert.
