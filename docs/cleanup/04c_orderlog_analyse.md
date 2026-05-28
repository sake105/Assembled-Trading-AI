# Paket 4c — Order-Lifecycle-Audit-Log (GO_LIVE C1)

## Schritt 1 — Ist-Analyse

### Existierende Order-Zustände / Events

`src/assembled_core/execution/order_lifecycle.py` definiert ein vollständiges FSM:

```
CREATED → VALIDATED → SUBMITTED → PARTIAL_FILL → FILLED (terminal)
                    ↘ REJECTED (terminal)
                    ↘ CANCELLED (terminal)
CREATED → VALIDATED → REJECTED (terminal)
PARTIAL_FILL → CANCELLED (terminal)
```

`_VALID_TRANSITIONS` erzwingt das Schema; illegale Übergänge werfen `ValueError`.

### Wo durchläuft eine Order ihre Phasen?

**Pfad A — EOD/Backtest-Zyklus (`_tc_risk.py` + `_tc_execution.py`)**

- `_tc_risk.py` Step 6.9 (Zeile 315–339): Erstellt einen `OrderLifecycleTracker` für alle `orders_filtered`. Führt jeden Order durch CREATED → VALIDATED → SUBMITTED. Schreibt Ergebnis in `result.meta["order_lifecycle"]`. **Kein Disk-Write, nur in-memory.**
- `_tc_execution.py` Step 7.66 (Zeile 441–488): Schreibt `orders_filtered` als Fills in `trade_journal.jsonl`. Kein Lifecycle-Bezug — alle Zeilen werden als FILLED behandelt, ohne explizites Event.

**Pfad B — Paper Trading (`unified_paper_engine.py`)**

- `_lifecycle_attach()` (Zeile 2650): CREATED
- `_lifecycle_mark_validation()` (Zeile 2685): VALIDATED oder REJECTED (wenn risk check fehlschlägt)
- `_lifecycle_mark_submitted()` (Zeile 2715): SUBMITTED
- `_lifecycle_mark_fills()` (Zeile 2728): PARTIAL_FILL / FILLED / CANCELLED
- `_lifecycle_dump()` (Zeile 2826): Schreibt **pro terminal Order** eine Zeile nach `output/paper_lifecycle/lifecycle_{run_id}_{date}.jsonl` — aber **als Snapshot (order.to_dict())**, nicht als Event-Stream.

### Was schreibt wann pro Phase?

| Pfad | SUBMITTED | ROUTED | FILLED | REJECTED | CANCELLED | PARTIAL_FILL |
|------|-----------|--------|--------|----------|-----------|--------------|
| Backtest _tc_risk.py | nur in-memory | — | — | — | — | — |
| Backtest _tc_execution.py trade_journal | — | — | per Fill (implizit) | — | — | — |
| Paper unified_paper_engine _lifecycle_dump | snapshot | — | snapshot | snapshot | snapshot | — |

**Fazit**: Es gibt keinen persistierten, per-Event-Eintrag im Lifecycle-Log. `_lifecycle_dump` schreibt nur terminal-Order-Snapshots (1 Zeile pro abgeschlossener Order, nicht 1 Zeile pro Transition).

### Append-only Schema-Vorbild (Kill-Switch-Audit aus 4b)

`_append_audit()` in `kill_switch.py` schreibt JSONL mit Hash-Chain (SHA-256, `prev_hash` + `hash` pro Eintrag) für Tamper-Detection. Diese Komplexität ist für das Order-Lifecycle-Log **nicht nötig** — die Kill-Switch-Audit schützt sicherheitskritische Operator-Aktionen. Das Order-Log ist ein operatives Audit-Trail, keine Security-Kontrolle.

**Entscheidung**: Kein Hash-Chain. Stattdessen einfaches JSONL konsistent mit `trade_journal.jsonl`. Vorbild: `append_trade_journal_entry()` Konvention.

---

## Schritt 2 — Design

### Neue Datei: `src/assembled_core/ops/order_lifecycle_log.py`

**Zweck**: Append-only, dateibasiertes Lifecycle-Event-Log. Eine Zeile pro Zustandsübergang.

**Schema** (JSONL, eine Zeile pro Event):

```json
{
  "order_id": "abc-123",
  "timestamp_utc": "2026-05-28T14:00:00.000000+00:00",
  "event_type": "SUBMITTED",
  "symbol": "AAPL",
  "side": "BUY",
  "qty": 100.0,
  "price": 150.25,
  "reason": null,
  "strategy": "trend_baseline",
  "actor": "pipeline",
  "run_id": "2026-05-28"
}
```

| Feld | Typ | Pflicht | Beschreibung |
|------|-----|---------|--------------|
| `order_id` | str | ja | Eindeutige Order-ID |
| `timestamp_utc` | str (ISO-8601) | ja | UTC, monoton im JSONL |
| `event_type` | str | ja | SUBMITTED / ROUTED / PARTIAL_FILL / FILLED / REJECTED / CANCELLED |
| `symbol` | str | ja | Ticker |
| `side` | str | ja | BUY / SELL |
| `qty` | float | ja | Anzahl Anteile |
| `price` | float | nullable | Preis (bekannt bei FILLED/PARTIAL_FILL) |
| `reason` | str | nullable | Grund bei REJECTED / CANCELLED |
| `strategy` | str | "" | Strategie-Name |
| `actor` | str | "pipeline" | Wer hat den Übergang ausgelöst |
| `run_id` | str | "" | Lauf-Identifier |

**Öffentliche API**:

```python
DEFAULT_LIFECYCLE_LOG_PATH: Path

def append_lifecycle_event(
    event_type: str,
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    *,
    price: float | None = None,
    reason: str | None = None,
    strategy: str = "",
    actor: str = "pipeline",
    run_id: str = "",
    log_path: Path | str | None = None,
) -> None: ...

TERMINAL_EVENTS: frozenset[str]  # {"FILLED", "REJECTED", "CANCELLED"}

def find_open_orders(
    log_path: Path | str | None = None,
) -> list[str]: ...
```

`find_open_orders` liest das JSONL und liefert alle `order_id`s, die noch kein Terminal-Event haben. Verwendet für C1-Validator und Tests.

### Hook-Punkte (keine Trading-Logik-Änderung)

| Datei | Stelle | Event |
|-------|--------|-------|
| `_tc_risk.py:333` | nach `_olt.transition(SUBMITTED)` | SUBMITTED pro Order in orders_filtered |
| `_tc_execution.py:478` | nach `append_trade_journal_entries()` | FILLED pro Fill |
| `unified_paper_engine.py:_lifecycle_mark_validation` | bei REJECTED-Pfad | REJECTED |
| `unified_paper_engine.py:_lifecycle_mark_fills` | bei FILLED/PARTIAL_FILL/CANCELLED | FILLED / PARTIAL_FILL / CANCELLED |

Alle Hooks sind in `try/except Exception` gewrapped — kein Impact auf Trading-Pfad.

### Kein Ersatz bestehender Logs

- `trade_journal.jsonl` bleibt vollständig erhalten (rückwärtskompatibel)
- `paper_lifecycle/*.jsonl` (`_lifecycle_dump`) bleibt vollständig erhalten
- Das neue `order_lifecycle.jsonl` ergänzt diese um per-Event-Granularität

---

## Schritt 3 — Implementierung

### Neue Datei
- `src/assembled_core/ops/order_lifecycle_log.py` — Lifecycle-Logger

### Geänderte Dateien
- `src/assembled_core/pipeline/_tc_risk.py` — SUBMITTED-Hook in Step 6.9
- `src/assembled_core/pipeline/_tc_execution.py` — FILLED-Hook in Step 7.66
- `src/assembled_core/execution/unified_paper_engine.py` — REJECTED/FILLED/PARTIAL_FILL/CANCELLED-Hooks

---

## Schritt 4 — Testergebnis

**Testdatei**: `tests/test_order_lifecycle_log.py` — 7 Tests

| Test | Beschreibung | Ergebnis |
|------|-------------|---------|
| `test_normal_lifecycle_three_entries_correct_order` | SUBMITTED→ROUTED→FILLED: 3 Einträge, richtige Reihenfolge, kein offener Eintrag | PASS |
| `test_rejected_order_reason_set` | REJECTED: reason-Feld gesetzt, kein offener Eintrag | PASS |
| `test_cancelled_order_with_reason` | CANCELLED: reason="eod_no_fill", kein offener Eintrag | PASS |
| `test_partial_fill_multiple_entries_then_filled` | 2× PARTIAL_FILL → FILLED: 4 Einträge, nur letzter terminal | PASS |
| `test_validator_finds_open_order_without_terminal` | Validator findet künstlich offene Order, aber nicht die abgeschlossene | PASS |
| `test_validator_empty_log_returns_empty_list` | Leeres Log → leere Liste | PASS |
| `test_entry_schema_contains_required_fields` | Alle Pflichtfelder vorhanden | PASS |
| `test_submitted_and_filled_order_ids_align` | Integration: risk-seitiges SUBMITTED-id und execution-seitiges FILLED-id identisch → find_open_orders=[] | PASS |
| `test_find_open_orders_returns_only_non_terminal` | EOD-Validator: offene Order erkannt, abgeschlossene ausgeschlossen | PASS |

**Ausgeführt**: 9 passed, 0 failed (Python 3.13.7, pytest 9.0.2, Windows)

**Ruff**: All checks passed auf allen 5 geänderten Dateien.

---

## GO_LIVE C1 Evidenz

- Neues Modul: `src/assembled_core/ops/order_lifecycle_log.py`
  - `append_lifecycle_event()` — append-only JSONL, eine Zeile pro Transition
  - `find_open_orders()` — Validator für Orders ohne Terminal-Event
- Hook-Punkte (nur Logging, keine Trading-Logik-Änderung):
  - `_tc_risk.py` Step 6.9: SUBMITTED pro Order in orders_filtered
  - `_tc_execution.py` Step 7.66: FILLED pro Fill
  - `unified_paper_engine.py`: REJECTED/SUBMITTED/PARTIAL_FILL/FILLED/CANCELLED via `_log_lifecycle_event()`
- Trade journal rückwärtskompatibel erhalten
- CI: unverified (lokal 7/7 PASS, ruff clean)
