# 06d — Deep Audit: ACCOUNTING / LEDGER / PAPER ENGINE / RECONCILIATION

- **Datum:** 2026-05-30
- **Agent:** DEEP-AUDIT AGENT D (Round 2)
- **Cluster:** Accounting / Ledger / Paper Engine / Reconciliation
- **Scope:** read-only. Nichts geändert außer dieser Datei.

## Module dissected
- `src/assembled_core/execution/unified_paper_engine.py` (3000+ Zeilen, voll gelesen)
- `src/assembled_core/ops/paper_ledger.py` (REAL live-paper ledger mutation — `apply_fills_to_ledger`)
- `src/assembled_core/accounting/`: `ledger.py`, `ledger_store.py`, `ledger_integration.py`,
  `position_engine.py`, `reconciliation.py`, `attribution.py`, `broker_snapshot.py`, `__init__.py`
- `src/assembled_core/paper/paper_track.py` (state persistence)
- `scripts/run_live_paper.py` (entry point — wires `ops/paper_ledger`)

---

## Zwei parallele Ledger-Wahrheiten (Architektur-Kontext)

Es gibt **zwei vollständig getrennte Ledger-Implementierungen**, die nie konsolidiert wurden
(Verstoß gegen Rule 50 „keine zweite Wahrheit", aber historisch gewachsen):

1. **`accounting/` (Sprint 13)** — event-sourced, parquet-basiert. `ledger.py` (events_from_*) →
   `ledger_store.py` (atomic parquet) → `position_engine.build_positions_from_ledger` (avg-cost +
   realized PnL) → `reconciliation.py` + `ledger_integration.build_ledger_from_trades`.
   Konsumiert vom Pipeline/Backtest-Pfad.
2. **`ops/paper_ledger.py`** — JSON-state, in-place mutation. `load_ledger_state` /
   `apply_fills_to_ledger` / `save_ledger_state` / `mark_to_market_equity`.
   **Das ist der reale Live-/Paper-Pfad** (`scripts/run_live_paper.py`).
3. **`UnifiedPaperEngine`** — dritter, in sich geschlossener Engine mit eigenem JSON-state
   (`_load_state`/`_update_positions`/`_save_state`). Behauptet laut Docstring die drei Pfade zu
   „unifizieren", tut das aber NICHT — die anderen zwei existieren weiter und werden weiter benutzt.

### Realer PAPER-Ledger-Mutationspfad (Live/Paper, `scripts/run_live_paper.py`)
```
cmd_run (mode=paper)
  → run_paper_daily_one(...)                       # scripts/run_live_paper.py:566
      → ops/paper_ledger.load_ledger_state()       # paper_ledger.py:29  (LOAD)
      → ops/paper_ledger.simulate_fills()          # paper_ledger.py:175 (fill@close + bps)
      → ops/paper_ledger.apply_fills_to_ledger()   # paper_ledger.py:241 (CASH+POS MUTATION, Decimal)
      → ops/paper_ledger.mark_to_market_equity()   # paper_ledger.py:381 (equity = cash + Σ qty*px)
      → ops/paper_ledger.save_ledger_state()       # paper_ledger.py:138 (SAVE, filelock+backup)
  → [broker-mode only] sync_positions_from_broker() # run_live_paper.py:600 (REAL recon vs Alpaca)
```
### UnifiedPaperEngine-Mutationspfad (separater Engine)
```
run_paper_day                                       # unified_paper_engine.py:438
  → _load_state()                                   # :826  (LOAD, raise-on-corrupt)
  → _simulate_fills_with_cost()                     # :1354 (fill_price folds ALL costs in)
  → _write_ledger_events()  (artifact only)         # :1653 (parquet sidecar, NOT the cash book)
  → _update_positions(fills)                         # :1544 (CASH+POS MUTATION)
  → _apply_borrow_costs()                            # :1741 (cash -= borrow)
  → _run_reconciliation()                            # :1886 (self-compare no-op by default)
  → _maybe_save_state()                              # :880  (atomic+fsync JSON)
```
Round-1-Notiz bestätigt: `_write_ledger_events` schreibt **nur Artefakte** (parquet sidecar pro Tag),
die reale Cash-/Position-Mutation passiert in `_update_positions` (Position-Buch ist `self._state`,
nicht der parquet-Ledger). Der parquet-Ledger wird nie zurück in `_state` gelesen.

---

## Findings

| ID | Modul:Zeile | Fund | Snippet | Schwere | betrifft |
|----|-------------|------|---------|---------|----------|
| D-01 | `ops/paper_ledger.py:59-60` | **Corrupt/missing ledger fail-OPEN (E-025).** Wenn Hauptdatei UND alle 3 Backups unlesbar sind, wird stillschweigend ein frischer 10k-Seed zurückgegeben — sieht aus wie echtes Kapital, kein Sentinel, kein lautes Scheitern. Nächster `save_ledger_state` überschreibt dann ggf. das (recoverbare) korrupte File. Das ist die GENAUE Anti-Pattern, die `unified_paper_engine._load_state:840-863` bewusst vermeidet (raise). | `if data is None: return _fresh_state(start_capital)` | **HOCH** | Live/Paper |
| D-02 | `unified_paper_engine.py:1886-1937` | **Reconciliation ist per Default Self-Compare-No-Op.** Ohne `shadow_broker` (Default `None`) ist der „broker"-Snapshot eine Kopie des Engine-State: `broker_positions_df = ledger_positions_df.copy(); broker_cash = cash`. → `cash_diff` und `max_qty_diff` IMMER 0 → severity IMMER "ok". Die SLO-Schwellen (25 bps fail) werden nie verletzt, weil verglichen wird state-gegen-state. | `broker_positions_df = ledger_positions_df.copy()` `broker_cash = cash` | **HOCH** | Paper/OOS |
| D-03 | `unified_paper_engine.py:1792-1884` | **Corporate Actions NICHT idempotent.** Kein „CA bereits verbucht"-Marker. Bei Re-Run/Retry desselben `as_of_date` (State wurde zwischendurch gespeichert) werden Splits erneut angewandt (`positions[sym] *= ratio` doppelt) und Dividenden erneut gutgeschrieben (`cash += qty*per_share`). | `positions[sym] = positions[sym] * ratio` ... `cash += positions[sym] * per_share` | **MITTEL** | Live/Paper (re-run) |
| D-04 | `unified_paper_engine.py:106-113` | **Import-Bug → `_HAS_LEDGER` permanent False.** `from ...accounting.ledger import store_ledger_events_parquet` — die Funktion ist in `ledger_store.py`, NICHT in `ledger.py`. Import wirft ImportError → `except Exception: _HAS_LEDGER=False`. Folge: der dedup-/atomic-Parquet-Store wird im UnifiedPaperEngine NIE benutzt; `_write_ledger_events:1727` fällt auf rohes `df_events.to_parquet(...)` (non-atomic, kein event_id-dedup) zurück. | `from src.assembled_core.accounting.ledger import (store_ledger_events_parquet,)` | **MITTEL** | Paper/OOS (Artefakt-Integrität) |
| D-05 | `unified_paper_engine.py:1724` | **Falsche Signatur, immer im except-Fallback.** Selbst wenn `_HAS_LEDGER` True wäre: `store_ledger_events_parquet(df_events, ledger_path)` ruft 2 positionale Args, die echte Signatur ist `(events_df, output_dir, run_id, *, mode=)`. → TypeError → bare `except Exception:` → roher Parquet-Write. Der atomare/dedup-Pfad ist toter Code für diesen Caller. | `store_ledger_events_parquet(df_events, ledger_path)` | **MITTEL** | Paper/OOS |
| D-06 | `ops/paper_ledger.py:161-166` | **`save_ledger_state` ohne fsync.** `tmp.write_text(...)` → `tmp.replace(p)`. `os.replace` ist atomar, aber ohne vorheriges `os.fsync` kann ein Crash nach replace, bevor das OS den tmp-Inode flusht, ein partielles/0-Byte-File hinterlassen. Hat filelock + Backup-Rotation, aber unified_engine fsync't (`:956`), dieser Pfad nicht. | `tmp.write_text(...); tmp.replace(p); return p` | **MITTEL** | Live/Paper |
| D-07 | `paper/paper_track.py:942-949` | **`save_paper_track_state` ohne fsync.** Gleiche Schwäche: atomic rename, kein fsync. Hat `.backup`-Copy als Mitigation. | `with open(temp_path,"w")...; temp_path.replace(state_path)` | **MITTEL** | Paper |
| D-08 | `unified_paper_engine.py:1266-1267, 1198` | **Flat-Commission wird nie modelliert.** Commission gibt es NUR bei `enable_cost_tiers=True` (Default False). Ohne Tiers ist `commission_bps=0.0` → Cash-Delta enthält Spread+Impact, aber 0 Kommission. Optimistischer Kostenpfad. (Spread/Impact/Adversarial/SOR sind korrekt in `fill_price` gefaltet → kein Double-Count, keine Omission DIESER Komponenten.) | `tier_commission_bps = 0.0` (only set when `enable_tiers`) | **MITTEL** | Paper/OOS (Cost-Realismus) |
| D-09 | `unified_paper_engine.py:1702-1704` | **event_id-Kollision BUY+SELL desselben Symbols/Tags.** `event_id = f"{run_id}_{date}_{sym}_{side}_{event_type}"` — pro (run,date,sym,side,type) eindeutig, aber zwei FILLs gleicher Seite am selben Tag/Symbol (z. B. SOR-Child-Splits oder zwei Strategie-Legs) kollidieren → dedup im (hier toten) Store würde einen verschlucken. Im rohen `to_parquet` bleiben beide als Doppelzeile. | `event_id = f"{self.config.run_id}_{as_of_date}_{sym}_{side}_{event_type}"` | **MITTEL** | Paper/OOS |
| D-10 | `unified_paper_engine.py:2522` | **MTM-Fallback ohne Warnung.** Fehlt der Preis eines gehaltenen Symbols, wird still `cost_basis` (oder 0.0) benutzt — KEIN WARNING. `paper_ledger.mark_to_market_equity:425-435` warnt korrekt; der Unified-Engine-Pfad schweigt → Equity kann unbemerkt von Broker-Wahrheit driften. | `price = price_map.get(sym, cost_basis.get(sym, 0.0))` | **NIEDRIG** | Paper/OOS |
| D-11 | `unified_paper_engine.py:1322` (+ paper_ledger:226) | **`min_fill_qty`-Default 0.0 + flat-tier mischen** — kein Bug, aber: bei `enable_partial_fills=False` ist `notional = fill_qty*fill_price` mit `fill_qty==qty` (voller Fill, kein Partial). Cash-Gate (`:1393`) ist sequentiell-deterministisch korrekt. Notiert als verifiziert-OK. | `notional = fill_qty * fill_price` | OK | — |

### Verifiziert KORREKT (keine Bugs)
- **Sign/Quantity-Buchung `_update_positions` (:1590-1644)** und **`apply_fills_to_ledger` (:280-357)**:
  BUY debitiert immer `qty*price`, SELL kreditiert immer `qty*price`; Long-Add / Short-Cover /
  Cover-and-flip / Oversell-flip explizit behandelt. Cash-Invariante sauber. `paper_ledger` nutzt
  `Decimal` gegen Float-Drift (gut). Beide sind Schwester-Implementierungen mit identischer Logik
  (Comment-Refs F-A-1 / F-A2-4 bestätigen bewusste Parität).
- **`position_engine.build_positions_from_ledger`**: realized PnL bei Reduce/Flip korrekt;
  NaN-`cash_delta` wird NICHT still genullt sondern wirft (`:140-143`, gut, fail-loud).
- **`attribution.py`**: echte notional-gewichtete Berechnung, kein Dummy/Hardcode. Warnt bei
  Schema-Drift in regime_history (`:168-180`).
- **`reconcile_ledger_vs_broker`** + **`evaluate_reconcile_slo`**: echte Diff-Logik, Schwellen
  (25 bps fail / 100 bps p99 slippage) WERDEN angewandt, Alert-Dispatch + fsync'd audit log
  (`:206`, `:40-48`). Die Funktion selbst ist substanziell — das Problem (D-02) ist, WAS reingefüttert
  wird, nicht die Funktion.
- **`ledger_integration.build_ledger_from_trades`**: 3-Wege broker_snapshot_policy
  (ignore/prefer/require) und WARNT laut bei paper-vs-paper-Fallback (`:248-252`). Substanziell.
- **`run_live_paper` broker-mode**: post-execution `sync_positions_from_broker` vs ECHTEM Alpaca
  mit Threshold + Halt-Flag (`:600-636`). Das ist die einzige echte sim-vs-reality-Reconciliation
  im Live-Pfad, aber sie greift nur bei `execution_mode == "broker"`, NICHT im reinen Sim-`paper`-Modus.

---

## Verdicts

### Reconciliation: real-or-stub?
**GEMISCHT / „real aber leerlaufend per Default".** Die Reconciliation-Engine
(`reconciliation.py`, `evaluate_reconcile_slo`) ist substanziell — echte bps-Diffs, echte Schwellen,
Alerts, fsync'd Audit-Log. ABER:
- Im **UnifiedPaperEngine** (Default, kein shadow_broker) ist es ein **Self-Compare-No-Op** (D-02):
  state-gegen-state → immer "ok". Findet per Konstruktion nie einen echten Diff.
- In **`ledger_integration`** ist Default `policy="prefer"` → ohne gespeicherten Broker-Snapshot
  ebenfalls paper-vs-paper, ABER laut geloggt (Mitigation vorhanden).
- Nur **`run_live_paper` broker-mode** vergleicht gegen ECHTEN Broker (Alpaca) mit Halt-Flag.
  → Echte sim-vs-reality-Kontrolle existiert NUR im broker-mode, nicht im Sim-paper-mode.

### State-write safety: atomic? fsync? race?
**TEILS.** Drei Pfade, uneinheitlich:
- `unified_paper_engine._atomic_write_json` (:942-962): **atomic (os.replace) + fsync** — bester Pfad.
  Keine prozessübergreifende Sperre, aber Single-Writer per Engine-Instanz angenommen.
- `ops/paper_ledger.save_ledger_state` (:138-172): **atomic + filelock + Backup-Rotation, ABER KEIN
  fsync** (D-06). filelock mindert Concurrent-Write-Race; fehlende fsync lässt Crash-Window für
  partielles File offen.
- `paper/paper_track.save_paper_track_state` (:942-949): **atomic + `.backup`-Copy, KEIN fsync, KEINE
  Sperre** (D-07).
→ Concurrent-Write-Race ist nur in `ops/paper_ledger` (filelock) wirklich adressiert; die anderen
zwei verlassen sich auf Single-Writer-Annahme. fsync fehlt in 2 von 3 Pfaden.

### Corrupt-state handling: fail-loud or fail-open?
**WIDERSPRÜCHLICH zwischen den Pfaden.**
- `unified_paper_engine._load_state` (:840-863): **fail-LOUD korrekt** — renamed das korrupte File
  (`.corrupt.<ts>`), `logger.critical`, `raise RuntimeError`. Kein stiller Seed-Reset. Vorbildlich.
- `ops/paper_ledger.load_ledger_state` (:59-60): **fail-OPEN (E-025)** — totale Korruption (Haupt +
  3 Backups) → stiller `_fresh_state(10k)`. KEIN Sentinel, kein raise. Das ist der reale Live-Pfad.
- Der von Round-1 erwähnte `start_capital=-1.0`-Sentinel sitzt im **API-Router** (`api/routers/ledger.py`,
  Paket 5), NICHT in diesen Accounting-Modulen. Im hier auditierten Hot-Path existiert KEIN
  Sentinel-Schutz auf der `ops/paper_ledger`-Seite.

---

## Blast-Radius-Zusammenfassung
- **Live/Paper real-money-relevant:** D-01 (fail-open seed reset), D-03 (CA double-apply auf re-run),
  D-06 (kein fsync live-state).
- **Paper/OOS metric-integrity:** D-02 (recon no-op → falsches „grünes" Reconcile-Signal),
  D-08 (fehlende Commission → zu optimistische Netto-Returns), D-04/D-05 (Ledger-Artefakt nicht
  atomar/dedup), D-09 (Doppelzeilen-Events).
- **NIEDRIG/kosmetisch:** D-07, D-10.

Nichts geändert. Alle Funde read-only verifiziert; wo Laufzeit-Beweis (Import-Resolution D-04)
nötig wäre, ist es als statisch-abgeleitet markiert.
