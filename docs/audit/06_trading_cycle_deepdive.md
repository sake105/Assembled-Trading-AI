# 06 — Trading-Cycle Deep-Dive (Runde 2) — Konsolidiert

**Datum:** 2026-05-30
**Art:** Reine Analyse + Verifikation. NICHTS gelöscht/geändert an Produktivcode.
**Methode:** 5 spezialisierte Deep-Dive-Agenten, je ein Stage-Cluster der Handelsmaschine, line-by-line. 3 der schwersten NEU-Funde vom Orchestrator **direkt am Quellcode nachverifiziert** (markiert „**(verifiziert)**").

**Cluster-Berichte (Detail):**
- `06a_orchestrator_data.md` — Cycle-Orchestrierung + Daten/Feature-Ingress
- `06b_signals_sizing.md` — Signal-Erzeugung + Position-Sizing
- `06c_risk_execution.md` — Risk-Controls + Order-Execution + Order-Lifecycle
- `06d_accounting_paper.md` — Accounting/Ledger + Paper-Engine + Reconciliation
- `06e_costs_reporting_config.md` — Kostenmodelle + Reporting/QA/Evidence + Config

> **Abgrenzung zu Runde 1 (`00_SUMMARY.md`):** Runde 1 prüfte OOS-Korrektheit/Look-Ahead/Metrik-Mathematik. Runde 2 zerlegt die **operative Live/Paper-Maschine**. Die Funde hier betreffen überwiegend die **Live/Paper-Sicherheit**, nicht die OOS-Backtests. Die OOS-„kein Edge"-Schlussfolgerung aus Runde 1 **bleibt unberührt** (s. „Gesamteinordnung" unten).

---

## Headline-Einordnung (ehrlich)

**Runde 2 stürzt KEINE OOS-Ergebnisse um — aber sie zeigt, dass die Live/Paper-Risk-Maschinerie deutlich fragiler / fail-open ist, als Doku und GO_LIVE-Stand suggerieren.**

Drei systemische Muster ziehen sich durch die Maschine:
1. **Fail-OPEN statt fail-closed:** Schutz-Gates lassen Orders durch, wenn sie nicht auswerten können (Exception → „gate no-op"). Ein Gate, das wegen kaputter Inputs nicht rechnen kann, blockt genau dann nicht.
2. **Definiert-aber-unverdrahtet:** Mehrere Schutz-Guards (DMS, OrderGate, stale-order, PDT, fat-finger qty-cap) existieren als Code, werden aber im Live-Pfad **nie aufgerufen**. Die reichere Guard-Kette von `UnifiedPaperEngine` ist nicht der Live-Pfad.
3. **Stille Degradation:** ~13 Enrichment-/Overlay-Schichten + Reconcile + Corrupt-State fallen auf `log.debug`/Default zurück — ein kaputter Feed ist nicht von „kein Signal" unterscheidbar (E-025-Familie).

---

## NEU-Funde nach Schweregrad (Runde 2)

### HOCH

| ID | Fund | Ort (Beleg) | betrifft |
|---|---|---|---|
| **R2-1** | **Alle Risk-Gates fail-OPEN.** VaR-Gate, Auto-DD-Kill-Switch, Circuit-Breaker, Fat-Finger sind je in `try/except → „gate no-op"` gewickelt; bei Exception passieren die Orders **ungefiltert**. Geloggt auf WARNING (nicht still), aber ein Gate, das nicht auswerten kann, sollte **fail-closed** blocken. | `pipeline/_tc_risk.py:213-222 / 225-243 / 246-254 / 296-314` **(verifiziert)** | Live/Paper |
| **R2-2** | **Ein Flag entwaffnet die ganze Schicht.** `if not ctx.enable_risk_controls: result.orders = orders; return` — ein einziges Flag überspringt Kill-Switch, VaR, Auto-DD, Circuit-Breaker, Fat-Finger und Pre-Trade-Gates **in einem Rutsch**. | `pipeline/_tc_risk.py:64-67` **(verifiziert)** + `trading_cycle_shared.py:1534-1535` | Live/Paper |
| **R2-3** | **H-1 ist MATERIAL (Upgrade aus Runde 1).** Der fehlerhafte `max_drawdown_pct` (`qa/metrics.py:215-217`) speist den **harten QA-BLOCK-Gate** (−20%) — `qa_gates.py:153-155` **(verifiziert)** — sowie EOD-Pipeline-Gate (`orchestrator.py:911`), Daily-QA-Report (`reports/daily_qa_report.py:68,608`), Paper-Pilot-Panel (`paper/paper_track.py:602,657`), Run-JSON (`orchestrator.py:296`) und Operator-API (`api/routers/qa.py:237,381`). Untertriebener DD → Gate blockt **zu selten**. | `qa/metrics.py:215-217` → Consumer s. links | Live/Paper + Gate + Reports |
| **R2-4** | **Schwächere Guard-Kette im Live-Pfad + 5 tote Schutz-Guards.** Live = `trading_cycle_v2` (route_orders→check_risk→book_fills); `UnifiedPaperEngine` mit reicherer Guard-Kette ist **nicht** im Live-Pfad. DEFINED-but-UNWIRED: DMS (`ops/dead_man_switch.py`, Daemon nicht deployed), OrderGate PDT+RoundTrip (`execution/order_gate.py`, 0 Prod-Caller), `cancel_stale_orders` (`stale_order_guard.py`, 0 Caller), PDTCounter (Doppelstruktur `risk/` + `execution/`, beide tot), Fat-Finger qty-Multiple-Cap (de-facto tot ohne `history_qty_by_symbol`). | s. `06c` | Live/Paper-Sicherheit |
| **R2-5** | **Corrupt-Ledger fail-OPEN.** Bei korrupter Haupt- UND Backup-State gibt `load_ledger_state` eine **frische `start_capital`-State** zurück (Default 10k) — kein Sentinel/Raise am Loader (pro-Kandidat WARN geloggt). Maskiert Korruption (E-025). Die API-Schicht (Paket 5/F2) nutzt zwar `start_capital=-1.0`-Sentinel zur Erkennung, der Loader selbst fail-opened aber. | `ops/paper_ledger.py:55-60` **(verifiziert)** | Live/Paper-Accounting |
| **R2-6** | **EOD-Reconciliation ist ein No-Op.** `UnifiedPaperEngine._run_reconciliation` (`:1886-1937`) vergleicht die Sim **mit sich selbst** → passt immer. Echte Sim-vs-Broker-Reconciliation feuert nur in `run_live_paper.py` im Broker-Modus gegen Live-Alpaca. EOD-Paper-Runs reconcilen nichts (falsche Sicherheit). | `execution/unified_paper_engine.py:1886-1937` | Paper |
| **R2-7** | **Defensive/Event-Positionen umgehen die globale De-Risk-Skalierung.** crisis_alpha/news_alpha-Einträge werden **nach** dem globalen Multiplier (`geo×profit_lock×vol×stress×crisis×pm×hmm`, Clamp [0.05,3.0]) angehängt — `_tc_sizing.py:2080` vs `2124-2125`. Genau die Event-/Krisen-Positionen entkommen geo/stress/HMM-Scaling. | `pipeline/_tc_sizing.py:2080, 2124-2125` | Live/Paper |

### MITTEL

| ID | Fund | Ort (Beleg) | betrifft |
|---|---|---|---|
| **R2-8** | `pl_update`-Latest-only-Snapshot überschreibt `prices_filtered`, das `size_positions` UND den EVT/Copula-Pivot in `check_risk` speist → vol/cov/tail-Schätzung kollabiert auf 1-Zeile-pro-Symbol. | `trading_cycle_v2.py:606-608 → _tc_sizing.py:2067 → _tc_risk.py:81` | Backtest-Snapshot; Live falls Snapshot aktiv |
| **R2-9** | `groupby.last()` ohne Timestamp-Sort für Fill-Preis + EOD-Filter — Korrektheit hängt am dokumentierten, aber **unenforced** `TradingContext.prices`-Sort-Vertrag. | `trading_cycle_shared.py:429, 879-881` | Live/Paper + OOS |
| **R2-10** | Per-Bar `load_policy()` umgeht `ctx._policy_cache` (4 andere Stages nutzen den Cache) → redundante I/O + Mid-Run-Inkonsistenz-Fenster. | `_tc_sizing.py:2063`, `_tc_signals.py:623` | Live/Paper |
| **R2-11** | ~13 Enrichment-Schichten degradieren still auf `log.debug("…skipped")` — kaputter Feed == leeres Signal (E-025). | `_tc_signals.py` (diverse) | Live/Paper |
| **R2-12** | Meta-Model-Confidence-Scaling ist toter Code: `score_col="mf_score"`-Key-Mismatch zwischen `multifactor_signal.py:908/1010` und Caller `_tc_signals.py:591-596`. Drop-Filter aktiv, Scaling no-op. | `multifactor_signal.py:908,1010` ↔ `_tc_signals.py:591-596` | Live/Paper |
| **R2-13** | Kein Aggregat-Gross-Recheck nach den letzten gewichts-mutierenden Overlays (cost_aware/conformal/pre-earnings/M&A) → Gross-Cap nach Overlay überschreitbar. | `_tc_sizing.py` (nach :2125) | Live/Paper |
| **R2-14** | Vol-Targeting: realized-vol≈0 → Scale auf Max gepinnt (nur Div-by-Zero-Guard, kein Near-Zero-Floor) → Max-Leverage. | `risk/vol_targeting.py:48-76` | Live/Paper |
| **R2-15** | `_sp_apply_factor_risk` skaliert `target_weight`, aber nicht `target_qty` → Weight/Qty-Desync, Execution ist qty-basiert. | `_tc_sizing.py` (`_sp_apply_factor_risk`) | Live/Paper |
| **R2-16** | Corporate-Action-Apply nicht idempotent → Re-Run doppelt-adjustiert. | `unified_paper_engine.py:1792-1884` | Paper |
| **R2-17** | Kaputter Import von `store_ledger_events_parquet` (falsches Modul). | `unified_paper_engine.py:106-113` | Paper (event-store) |
| **R2-18** | Echter Ledger-Save ohne fsync (Durability-Lücke auf dem Live-Pfad); `paper_track.save_*` ebenfalls ohne fsync. Writes sind atomar (tmp→os.replace), aber nicht fsync'd. | `ops/paper_ledger.py:161-166`, `paper/paper_track.py:942-949` | Live/Paper |
| **R2-19** | Keine `order_id`-Dedupe beim Fill-Booking → ein wiederholter Run re-bookt Fills. | `ops/paper_ledger`/`unified_paper_engine` Fill-Booking | Live/Paper |
| **R2-20** | Stiller 0-bps-Live-Pfad falls `cost_model` leer (`paper_ledger.py:189` Default 0) — nur durch `policy.yaml` abgewendet, kein Guard. | `ops/paper_ledger.py:189` | Live/Paper-Kosten |
| **R2-21** | Live- ≠ OOS-Kostenmodell: live flach ~10.75 bps (`policy.yaml:828-833`) vs OOS ADV/vol-Engine ~1-3 bps (`portfolio.py:164`). Inkonsistenz Sim↔Live. | `ops/paper_ledger.py:180-238` vs `pipeline/portfolio.py:164` | Live↔OOS-Konsistenz |
| **R2-22** | Broad-`except` Kosten-Degradation bei tz/dtype-Merge-Fehler. | `execution/transaction_costs.py:192, 242` | Live/Reporting-Kosten |

### NIEDRIG / Architektur / Info

- `pipeline/trading_cycle.py` = dünner Re-Export-Shim (kein zweite-Wahrheit-Problem). `orchestrator.py` = dokumentiert-deferred zweite EOD-Batch-Pipeline, teilt nur Signal-Gen via `_shared_eod` — bewusste Divergenz, keine Schleichstruktur.
- PDTCounter-Doppelstruktur (`risk/pdt_counter.py` + `execution/pdt_counter.py`), beide tot — Rule-50-Drift.
- Policy-Load-Fail in `_tc_risk.py:54-61` ist fail-open (`policy={}` → policy-gated Guards aus), **aber** mit `log.warning` (anders als Runde-1 M-1 im Sizing, das still ist).

---

## Positiv bestätigt (Ehrlichkeit in beide Richtungen)

- **Bookkeeping-Mathematik korrekt:** Sign/Qty/Cover/Flip, Decimal-Cash, Cost-Folding — keine Dummy/hardcoded-Returns im Accounting (Agent D).
- **Sign-Handling in `signals_to_weights` korrekt** — keine Long/Short-Inversion (Agent B).
- **PIT `as_of`-Slicing konsistent** in pairs/crisis-GPR/news-Lookups; kein neuer Look-Ahead-past-`as_of` (Agent A+B).
- **State-Writes atomar** (tmp→os.replace) in allen drei Writern (Agent D) — nur fsync inkonsistent.
- **`evidence_pack` real** (sha256 + Manifest-Validierung), keine Stubs (Agent E).
- **Config-Override (M-7) ist geloggt + dokumentiert** (`paper_runner.py:1115-1133`, `log.info` :1127) — kein stiller Footgun, nur Observability auf INFO.
- **Promotion-Gate-Skript H-1-immun:** `scripts/ops/check_promotion_gate.py:97-116` rechnet MDD selbst peak-to-date (korrekt) — der formale Promotion-Gate ist NICHT vom H-1-Bug betroffen (nur die operativen Reports + der QA-BLOCK-Gate sind es).
- **Kill-Switch WIRED+ACTIVE** an zwei Punkten (`run_live_paper.py:356` Preflight + Cycle via `_apply_risk_controls_default`) — funktioniert, aber an `enable_risk_controls` gekoppelt (R2-2).

---

## Gesamteinordnung — was bedeutet das?

**(a) OOS-Ergebnis-Glaubwürdigkeit (Frage aus Runde 1):** unverändert. Kein Round-2-Fund erzeugt einen falschen OOS-Edge. Die fail-open Gates, toten Guards, Reconcile-No-Ops und Cost-Inkonsistenzen betreffen die **Live/Paper-Maschine**, nicht den OOS-Backtest-Pfad (`portfolio.py`, Runde-1-verifiziert sauber). Einzige Brücke: **H-1** kann den **ausgewiesenen MaxDD%** in OOS-Result-Docs untertreiben, falls diese `max_drawdown_pct` ziehen — optimistisch, erzeugt aber keinen Return-Edge.

**(b) Live/Paper-Go-Live-Reife:** deutlich schwächer als der GO_LIVE-Stand (12/16) suggeriert. Die Schutzschicht ist überwiegend fail-open, mehrere Guards sind tot, EOD-Reconcile prüft nichts, Corrupt-Ledger reseeded still auf 10k. Für echtes Geld (Account T) sind R2-1, R2-2, R2-4, R2-5, R2-6 die kritischen Blocker.

**(c) Ballast/Dummy:** PDTCounter-Doppelstruktur, unverdrahtete Guards, toter Confidence-Scaling-Pfad — Aufräum-/Verdrahtungs-Kandidaten, kein akuter Schaden.

---

## Verifikationsstatus

- **Direkt am Quellcode verifiziert:** R2-1 (`_tc_risk.py:213-314`), R2-2 (`_tc_risk.py:64-67`), R2-3/H-1-Consumer (`qa_gates.py:153-155`), R2-5 (`ops/paper_ledger.py:55-60`).
- **Agent-berichtet mit file:line, nicht einzeln nachverifiziert:** alle übrigen R2-Funde — Belege in `06a`–`06e`.
- **Nur statisch / lokal — NICHT CI-bestätigt.** Items, die Ausführung bräuchten, sind in den Cluster-Berichten als UNSURE markiert.

## Offene Folge-Hinweise (NICHT in diesem Durchgang ausgeführt)

- Fix-Kandidaten als separate, gezielte Tasks (Review-Chain-pflichtig, `src/` betroffen): R2-1 (Gates fail-closed), R2-3/H-1 (`max_drawdown_pct` peak-to-date), R2-5 (Corrupt-Ledger fail-loud), R2-6 (echte EOD-Reconcile), R2-7 (Overlay-Reihenfolge), R2-2 (Gate-Entkopplung).
- Scratch-Dateien aus Runde 1 liegen noch: `docs/audit/_scratch_numeric_verification.py`, `docs/audit/_scratch_dd_edgecase.py`.
