# 00 — SYSTEM-VOLLPRÜFUNG — Konsolidierter Gesamtbericht

**Datum:** 2026-05-30
**Art:** Reine Analyse + Verifikation. NICHTS gelöscht, NICHTS geändert (außer den 5 Einzelberichten + diesem Bericht unter `docs/audit/`).
**Methode:** 5 spezialisierte Sub-Agenten (statische Analyse, Daten-Tracing, Look-Ahead-Prüfung, unabhängige numerische Re-Implementierung, Test-Substanz-Prüfung). Die zwei höchst-relevanten Funde wurden vom Orchestrator **direkt am Quellcode nachverifiziert** (siehe „Verifikationsstatus").

**Einzelberichte:**
- `docs/audit/01_dead_code.md` — Tote Pfade & Ballast
- `docs/audit/02_wiring_dataflow.md` — Verdrahtung & Daten-Integrität
- `docs/audit/03_lookahead_correctness.md` — Look-Ahead & Backtest-Korrektheit
- `docs/audit/04_numeric_verification.md` — Numerische Verifikation gegen Referenz
- `docs/audit/05_test_quality.md` — Tests & CI-Substanz

---

## Kernfrage zuerst (ehrliches Fazit)

> **Gibt es einen Defekt, der erklären könnte, warum die Strategien keinen Edge zeigten — oder bestätigt die Prüfung, dass die Ergebnisse echt sind?**

**Die negativen OOS-Ergebnisse STEHEN. Es wurde KEIN Defekt gefunden, der einen echten Edge unterdrückt oder einen falschen Edge erzeugt hätte.**

Belege:
- **Look-Ahead (Agent 3):** 6 von 7 OOS-/Walk-Forward-Skripten sind **PIT-sauber** (stateless/parameterfreie Transforms, `shift(1)` vor dem Slicing, ffill-only, per-Rebalance-Slicing). Der einzige reale Leak (`mfv_long_short`, Winsorize über das volle Fenster) wirkt **optimistisch** — und betrifft genau die Strategie, die **trotzdem komplett versagt** hat. Ein Hindsight-Vorteil, der immer noch verliert, **bestärkt** das negative Ergebnis.
- **Kostenmodell (Agent 3):** Kein „0 bps statt 10 bps"-Defekt im OOS-Pfad. Alle 7 Skripte übergeben `commission_bps=10.0, include_costs=True`; `simulate_with_costs` (`portfolio.py:164-190`) zieht Commission+Spread+Slippage tatsächlich vom PnL ab. Verifiziert.
- **Benchmark (Agent 3):** SPY läuft **kostenfrei**, die Strategien zahlen volle Kosten. Das ist eine Verzerrung **gegen** die Strategien — „keine schlägt SPY" ist damit **konservativ**, nicht geschönt.
- **Drawdown-Bug (Agent 4, verifiziert):** Der eine echte Metrik-Bug verkleinert den ausgewiesenen MaxDD **optimistisch** — flattert die Strategien, erklärt also kein fehlendes Edge.

**Einschränkung der Aussagekraft (wichtig):** `multifactor_v2` lief im OOS auf nur **~9 von 34 Faktoren** — ~8-10 Faktoren sind strukturell ZERO (fehlende Datendateien: insider, congress, sector_rotation, news_sentiment, options/VIX, earnings_surprise_z). Das mfv2-Ergebnis ist also „kein Edge **mit den real verfügbaren Daten**", nicht ein Test der vollen 34-Faktor-These.

---

## Defekte nach Schweregrad

### HOCH

| ID | Fund | Ort (Beleg) | Verfälscht Strategie-Ergebnisse? |
|---|---|---|---|
| **H-1** | `max_drawdown_pct` teilt den Trough-Drop durch den **globalen** Peak (`rolling_max.max()`) statt durch den Peak-to-date am Tiefpunkt. Für jede Equity-Kurve, die später ein neues Hoch macht, wird der MaxDD% **zu klein** ausgewiesen. Probe `[100,90,80,120,200]`: korrekt −20%, Produktion −10%. Propagiert in **Calmar**. | `src/assembled_core/qa/metrics.py:215-217` **(direkt verifiziert)** | **JA** — der ausgewiesene **MaxDD% und Calmar sind optimistisch verzerrt** (zu gut). Absoluter $-MaxDD und der regime-segmentierte MaxDD (`:1333-1335`, korrekte Peak-to-date-Form) sind NICHT betroffen. Relevanz: Promotion-Gate-Kriterium „MaxDD < 20%" (TWO_ACCOUNT_SETUP.md) wird gegen eine **geschönte** Zahl geprüft. |
| **H-2** | `mfv_long_short` füttert das volle Warmup+Test-Fenster in einem Rutsch in `generate_multifactor_long_short_signals`; `_winsorize_series` (default `winsorize=True`) berechnet die Clip-Grenzen via `quantile()` über **alle** Fensterdaten gepoolt → frühe Testtage „sehen" Future-Tails. Optimistischer Look-Ahead. | `signals/.../multifactor_signal.py:80-81`, aufgerufen aus dem OOS-Skript `:166-177` | **JA, aber konservativ-folgenlos** — Bias ist optimistisch, klein (nur 1%/99%-Tail-Clip; z-score + Top/Bottom-Selektion sind pro-Datum kausal), und die Strategie versagt trotzdem. Auffällig: der fast identische Bruder `mfv2` ist sauber, weil er **pro Rebalance vorher** sliced. |

### MITTEL

| ID | Fund | Ort (Beleg) | Ergebnis-Relevanz |
|---|---|---|---|
| **M-1** | `except Exception: policy = {}` **ohne WARN** — ein Config-Load-Fehler deaktiviert still **alle** Sizing-Risk-Limits (`policy.get("position_sizing") or {}` → `{}`). | `pipeline/_tc_sizing.py:2062-2065` **(verifiziert)** | Live-Risiko, nicht OOS. Maskiert Config-Korruption (Anti-Pattern-Familie E-025). |
| **M-2** | `except Exception: _shared_rets = None` **ohne WARN** — EVT-Tail-VaR + Copula-Tail-Gates laufen nur `if _shared_rets is not None`, werden also bei jedem Pivot/dtype/tz-Fehler **still übersprungen**. | `pipeline/_tc_risk.py:101-102` **(verifiziert)** | Live-Risiko. Schutz-Overlay fällt geräuschlos aus. |
| **M-3** | Meta-Model-Threshold/Filter/Ensemble-Fehler werden auf DEBUG geschluckt bzw. `return pd.DataFrame()`. | `pipeline/_tc_signals.py:572 / 600 / 653` | Signal-Degradation kann unbemerkt bleiben. |
| **M-4** | Kosten-Merge `trades↔ADV` auf `["timestamp","symbol"]` — tz/dtype-Mismatch füllt Kosten-Inputs still mit NaN, maskiert vom `except` (`:192`). | `costs/transaction_costs.py:179-229` | **Separater** Live-/Reporting-Kostenpfad — **NICHT** der OOS-Pfad (der läuft über `portfolio.py:164-190`, verifiziert sauber). Betrifft OOS-Ergebnisse **nicht**. |
| **M-5** | mfv2 Daten-Ehrlichkeits-Lücke: ~8-10 von 34 Faktoren strukturell ZERO (fehlende Datendateien), via Dead-Factor-Filter (`multifactor_v2.py:1490-1496`) entfernt. Der Spalten-Strip in `prices_ingest.py:134-146` existiert weiter, hungert die Faktoren aber **nicht** aus (mfv2 re-computet inline). | `multifactor_v2.py:1179, 1490-1496` | **JA, indirekt** — schränkt die Aussagekraft des mfv2-OOS ein (s. o.). Die ursprüngliche „19 stille Faktoren werden gedroppt"-These ist als **Live-Failure-Mode widerlegt**, aber die Daten-Coverage-Lücke ist real. |
| **M-6** | Verworfene Rückgabewerte: `pl_update`-Frame überschreibt `prices_filtered`+`prices_latest`; Pre-Trade-Impact/Group-Cap-Meta-Dicts in `_`-Locals verworfen. | `trading_cycle_v2.py:603-608`, `_tc_execution.py:99,122` | Diagnostik-Verlust, keine Entscheidungsänderung nachgewiesen. |
| **M-7** | **Realer roter Test** (KEIN yfinance-Problem): `test_paper_run_ema_produces_trades` erwartet `ema_trend_v0`-Orders, bekommt aber `trend_baseline`, weil `policy.yaml active_strategy` Vorrang vor dem Test-`app_cfg` hat und der Test den Policy-Loader nicht monkeypatcht. Tagged `fast`+`unit` → auf der CI-Oberfläche. | `tests/test_ema_trend_v0.py` | Test-/Wiring-Defekt. **Korrigiert die Prämisse:** der „EMA-yfinance-Fehler" existiert so nicht — es gibt keinen yfinance-abhängigen EMA-Test. |

### NIEDRIG / Ballast / Dummy

**(c) Dummy-/Stub-Dateien** (Funktionen mit reinem `raise NotImplementedError` / `return 0.0`-Platzhalter; kein produktiver Aufrufer):
- `src/assembled_core/ml/logic_tensor_network.py` — `.fit()` (108-110) + `.predict()` (120-122) `raise NotImplementedError`. TOT+DUMMY.
- `src/assembled_core/ml/temporal_fusion_transformer.py` — `.fit()` (117-121) + `.predict()` (138-139) `raise NotImplementedError`. TOT+DUMMY.
- `src/assembled_core/features/chart_pattern_matrix.py` — kein produktiver Import; `composite_score.py:251-256 chart_pattern_score()` `return 0.0` Platzhalter. TOT/DUMMY.
- `src/assembled_core/domain/{trading,risk,accounting,research,operations}/__init__.py` — leere Hexagonal-Skeleton-Platzhalter (Month-1, Audit C-001..C-007). TOT (by design).

**(b) Ballast / nur Test-referenziert** (UNSURE — behalten, aber nicht im Live-Pfad):
- `signals/cross_asset_carry_v2.py`, `signals/lppls_crash.py` — nur von Tests referenziert, kein produktiver Import.
- Diverse research-tier Module (Agent-1-Bericht, 17 UNSURE) — überwiegend Research-Toolbox / ERWEITERUNG, bewusst nicht im Live-Pfad.

**No-op-/leere Tests** (prüfen nichts):
- `tests/test_session_2026_05_07_new_items.py:12031` — `assert True  # just document the check`.
- `tests/test_backtest_numba_fallback.py:172` — `test_settings_use_numba_env_var` setzt Env-Vars, **keine** Assertion.
- `tests/test_integration_run_daily.py:315` — `assert len(long_signals) >= 0` (immer wahr).

---

## Test- & CI-Substanz (Agent 5)

- **Collection:** 8059 Tests, **0 Collection-Errors** (Baseline-Memory war 5417 — kein Regress, Suite gewachsen). Lokal Windows, 2026-05-30.
- **Fast-Suite (2730 Tests):** **1 FAIL** (M-7 oben), 186 Skips (alle mit Grund), ~2543 Pass.
- **yfinance:** Kein yfinance-EMA-Test existiert. Einziger yfinance-Skip: `test_data_source_live.py:191` (intentional `@pytest.mark.slow`).
- **Kritischer-Pfad-Coverage-Lücke:** `pipeline/_tc_risk.py` EVT-/Copula-Branches haben **keine** Unit-Tests; `except Exception: log.debug(...)` macht einen stillen Import-Fehler von einem korrekten Daten-Skip ununterscheidbar (deckt sich mit M-2). `book_fills` in `_tc_execution.py` ohne isolierten Unit-Test.
- **Ehrlichkeit:** Nur **lokal** ausgeführt, Teilmenge — **NICHT** „CI-bestätigt".

---

## Numerische Verifikation (Agent 4) — unabhängige Re-Implementierung

Auf einem gemeinsamen 252-Punkt-Datensatz, raw numpy/pandas gegen Produktion:

| Metrik | Ort | Verdikt |
|---|---|---|
| Realisierte Vol | `qa/metrics.py:540-541` (`std(ddof=1)*√252`) | **MATCH** (abs diff 0.0) |
| Sharpe (rf=0 und rf=0.02) | `qa/metrics.py:141-149` | **MATCH** |
| `compute_realized_vol` + Scale | `risk/vol_targeting.py` | **MATCH** |
| max_drawdown (absolut $) | `qa/metrics.py:214` | **MATCH** |
| vol_target_overlay-Gewichte | `strategies/vol_target_overlay.py:97-116` | **MATCH** |
| **max_drawdown_pct** | `qa/metrics.py:215-217` | **MISMATCH → H-1** |

Fazit: Die Kernmetrik-Mathematik ist korrekt; der **einzige** numerische Fehler ist der MaxDD%-Bug (H-1).

---

## Was OOS-Ergebnisse verfälscht haben KÖNNTE vs. was nicht

**(a) Könnte OOS-Zahlen beeinflusst haben:**
- **H-1** MaxDD%/Calmar optimistisch verzerrt — **falls** die OOS-Reports `max_drawdown_pct` aus dieser Funktion ziehen (zu prüfen: einige OOS-Skripte nutzen eigene Metrik-Pfade). Richtung: **zu gut**.
- **H-2** `mfv_long_short` Winsorize-Leak — optimistisch, klein, Strategie versagt trotzdem.

**(b) Beeinflusst OOS-Ergebnisse NICHT:**
- M-4 (Kosten-Merge-Trap) — separater Live-Kostenpfad, nicht der OOS-`portfolio.py`-Pfad.
- M-1/M-2/M-3 — Live-Pipeline-Degradation, keine OOS-Skript-Wirkung.
- Benchmark-Free-Pass — wirkt gegen die Strategien (konservativ).

**(c) Ballast/Dummy** — siehe NIEDRIG-Sektion; keine Ergebnis-Relevanz.

---

## Verifikationsstatus (Ehrlichkeit)

- **Direkt am Quellcode nachverifiziert vom Orchestrator:** H-1 (`qa/metrics.py:211-218`), M-1 (`_tc_sizing.py:2062-2065`), M-2 (`_tc_risk.py:101-102`).
- **Agent-berichtet mit file:line-Beleg, nicht einzeln nachverifiziert:** alle übrigen Funde. Belege stehen in den Einzelberichten.
- **Nur lokal, nicht CI-bestätigt.** Keine „alles grün"-Aussage.

## Offene Aufräum-/Folge-Hinweise (NICHT in diesem Durchgang ausgeführt)

- Audit-Scratch-Dateien von Agent 4 liegen noch: `docs/audit/_scratch_numeric_verification.py`, `docs/audit/_scratch_dd_edgecase.py` (Wegwerf-Verifikationscode; können gelöscht werden).
- H-1, H-2, M-1, M-2, M-7 sind **Fix-Kandidaten** für separate, gezielte Tasks (je eigener Scope, Review-Chain-pflichtig, da `src/` betroffen). In diesem Durchgang **bewusst nicht** angefasst.

---

# Runde 2 (Deep-Dive) — Trading-Cycle / Handelsmaschine

**Datum:** 2026-05-30 · **Vollbericht:** `docs/audit/06_trading_cycle_deepdive.md` (+ Cluster `06a`–`06e`).
**Scope:** Runde 1 = OOS-Korrektheit/Metrik-Mathematik. Runde 2 = die **operative Live/Paper-Maschine** (`trading_cycle_v2` + `_tc_*` + Accounting/Paper + Kosten/Reporting), line-by-line.

## Kernaussage Runde 2 (ehrlich)

**Runde 2 stürzt KEINE OOS-Ergebnisse um — das negative „kein Edge" aus Runde 1 bleibt unberührt.** Aber sie zeigt: die **Live/Paper-Risk-Maschinerie ist deutlich fragiler / fail-open**, als der GO_LIVE-Stand (12/16) suggeriert. Drei systemische Muster: (1) Schutz-Gates **fail-OPEN** (Exception → Orders passieren); (2) mehrere Guards **definiert-aber-unverdrahtet** (toter Code im Live-Pfad); (3) **stille Degradation** (~13 Enrichment-Schichten + Reconcile + Corrupt-State fallen geräuschlos auf Default/`debug` zurück).

## H-1 Upgrade: HOCH → **MATERIAL**

Runde 1 hatte H-1 (`max_drawdown_pct`-Bug, `qa/metrics.py:215-217`) als „optimistisch verzerrt, ggf. nur Reporting" gewertet. Runde 2 hat die **Consumer-Kette verifiziert** — der Bug speist einen **harten Entscheidungs-Gate**:
- **QA-BLOCK-Gate** `qa/qa_gates.py:153-155` **(verifiziert)** — der −20%-DD-Gate prüft gegen den **untertriebenen** Wert → blockt **zu selten**.
- Weiter: EOD-Pipeline-Gate (`orchestrator.py:911`), Daily-QA-Report (`reports/daily_qa_report.py:68,608`), Paper-Pilot-Panel (`paper/paper_track.py:602,657`), Run-JSON (`orchestrator.py:296`), Operator-API (`api/routers/qa.py:237,381`).
- **Immun:** der formale Promotion-Gate `scripts/ops/check_promotion_gate.py:97-116` rechnet MDD selbst korrekt (peak-to-date) — das „MaxDD < 20%"-Promotion-Kriterium ist NICHT betroffen, wohl aber der QA-BLOCK-Gate + alle operativen Reports.

## Runde-2 HOCH-Funde (Live/Paper, NICHT OOS)

| ID | Fund | Ort (Beleg) |
|---|---|---|
| **R2-1** | Alle Risk-Gates **fail-OPEN** (VaR, Auto-DD-Kill, Circuit-Breaker, Fat-Finger): Exception → „gate no-op" → Orders ungefiltert. Auf WARNING geloggt, sollte aber fail-closed blocken. | `_tc_risk.py:213-222/225-243/246-254/296-314` **(verifiziert)** |
| **R2-2** | Ein Flag (`enable_risk_controls=False`) **entwaffnet die ganze Schicht** in einem Rutsch (Kill-Switch+VaR+DD+CB+FatFinger+Pre-Trade). | `_tc_risk.py:64-67` **(verifiziert)** |
| **R2-3** | H-1 ist **MATERIAL** (s. o.) — speist den harten QA-BLOCK-Gate. | `qa/metrics.py:215-217` → `qa_gates.py:153-155` **(verifiziert)** |
| **R2-4** | Schwächere Guard-Kette im Live-Pfad + **5 tote Schutz-Guards** (DMS-Daemon nicht deployed, OrderGate 0 Caller, `cancel_stale_orders` 0 Caller, PDTCounter-Doppelstruktur tot, Fat-Finger qty-Cap de-facto tot). | `06c` |
| **R2-5** | Corrupt-Ledger **fail-OPEN**: Loader gibt frische 10k-State zurück statt Sentinel/Raise (maskiert Korruption, E-025). | `ops/paper_ledger.py:55-60` **(verifiziert)** |
| **R2-6** | EOD-**Reconciliation ist No-Op**: vergleicht Sim mit sich selbst → passt immer (echte Broker-Reconcile nur im Live-Alpaca-Modus). | `unified_paper_engine.py:1886-1937` |
| **R2-7** | Defensive/Event-Positionen (crisis_alpha/news_alpha) **umgehen die globale De-Risk-Skalierung** (nach dem Multiplier angehängt). | `_tc_sizing.py:2080 vs 2124-2125` |

**MITTEL (R2-8 … R2-22):** u.a. Snapshot-Overwrite kollabiert Vol/Cov-Schätzung (R2-8); unenforced Sort-Vertrag bei `groupby.last()` (R2-9); toter Meta-Confidence-Scaling-Pfad durch Key-Mismatch (R2-12); Vol-Targeting near-zero → Max-Leverage (R2-14); Weight/Qty-Desync (R2-15); CA-Apply nicht idempotent (R2-16); kaputter event-store-Import (R2-17); fehlendes fsync (R2-18); keine Fill-Dedupe → Re-Run re-bookt (R2-19); **Live- ≠ OOS-Kostenmodell** (~10.75 bps flach vs ADV/vol-Engine, R2-21). Voll-Tabelle in `06_trading_cycle_deepdive.md`.

## Was Runde 2 für die Kernfrage bedeutet

- **(a) OOS-Glaubwürdigkeit:** **unverändert.** Kein R2-Fund erzeugt einen falschen OOS-Edge — alle betreffen die Live/Paper-Maschine, nicht den OOS-`portfolio.py`-Pfad (Runde-1-verifiziert sauber). Einzige Brücke bleibt H-1 (optimistischer ausgewiesener MaxDD%, kein Return-Edge).
- **(b) Live/Paper-Go-Live-Reife:** **deutlich schwächer** als 12/16 suggeriert. Für echtes Geld (Account T) sind R2-1, R2-2, R2-4, R2-5, R2-6 die kritischen Blocker.
- **(c) Ballast/Dummy:** PDTCounter-Doppelstruktur, unverdrahtete Guards, toter Confidence-Scaling — Aufräum-/Verdrahtungs-Kandidaten, kein akuter Schaden.

## Verifikationsstatus Runde 2

- **Direkt am Quellcode verifiziert:** R2-1, R2-2, R2-3/H-1-Consumer, R2-5. Übrige R2-Funde agent-berichtet mit file:line (Belege in `06a`–`06e`), nicht einzeln nachverifiziert. **Nur statisch/lokal — NICHT CI-bestätigt.**
- **Fix-Kandidaten (separate Review-Chain-pflichtige Tasks, `src/` betroffen):** R2-1, R2-2, R2-3/H-1, R2-5, R2-6, R2-7. In diesem Durchgang **bewusst nicht** angefasst.

---

# Runde 3 (Extended) — Security / Feeds / Ops / CI / Strategy-Breite / Silent-Except-Zensus

**Datum:** 2026-05-30 · **Vollbericht:** `docs/audit/07_extended_audit.md` (+ Cluster `07a`–`07f`).
**Scope:** 6 Agenten auf bisher **ungeprüften Feldern**: Security/Secrets/Auth/MNPI (`07a`, SEC-), Data-Sources/Feeds (`07b`, DAT-), Concurrency/Scheduler/DMS/Alerting (`07c`, OPS-), CI/CD/Dependencies (`07d`, CI-), Strategy/Feature/ML (`07e`, STR-), systematischer Silent-Except/Contracts/Determinismus-Zensus (`07f`, QUAL-).

## Kernaussage Runde 3 (ehrlich)

**Runde 3 ändert die OOS-„kein Edge"-Schlussfolgerung NICHT — verstärkt aber das Runde-2-Fazit: die operative Reife (Security, Feeds, Ops, CI-Schutz) ist deutlich schwächer als GO_LIVE 12/16 suggeriert.** Vier neue systemische Muster: (1) **Das Sicherheitsnetz ist gebaut, aber nicht angeschlossen** (DMS, Heartbeat, Alerting, 2 Data-Quality-Gates, Freshness-Monitor — alle 0 Prod-Caller). (2) **Silent-Degradation ist strukturell** (Zensus: 997 `except`, 44 % auf sensiblen Pfaden, nur 7 % re-raisen, 130 still auf DEBUG). (3) **CI-Schutz teils Theater** (governance-benannte Gates blocken nicht). (4) **Auth fail-OPEN per Default** (Kill-Switch-Activate unauthentifiziert).

## Runde-3 KRITISCH (Live-Sicherheit für unattended Betrieb)

| ID | Fund | Ort (Beleg) |
|---|---|---|
| **OPS-01** | **DMS-Daemon ist toter Code** — 0 Deployment-Referenzen → Auto-Flat-on-stale nicht erreichbar. | `scripts/dms_daemon.py` |
| **OPS-02** | **Heartbeat-Pfad-Mismatch (3-fach)** — Schreiber (`_tc_execution.py:526`, `paper_trading_scheduler.py:36`) und DMS-Leser (`dead_man_switch.py` default) uneinig → Staleness läuft ins Leere. | s. links |
| **OPS-03** | **Staleness-Detektor sieht nie echte Daten** — feuert nur gegen synthetischen Drill-Heartbeat. | `check_scheduler_health.py:41` |
| **OPS-04** | **Kill-Switch ohne Locking → TOCTOU** auf der Audit-Hash-Chain (Read-Modify-Write). | `execution/kill_switch.py:142,168` |

## Runde-3 HOCH

| ID | Fund | Ort (Beleg) |
|---|---|---|
| **SEC-1** | **API-Auth fail-OPEN per Default**; Kill-Switch-**Activate** ungesichert (DoS), `/ready` prüft `auth_is_configured()` nie. | `api/auth.py:43-46` **(verifiziert)** |
| **OPS-07** | **CI-Reconcile neutralisiert** (`--dry-run \|\| true`) — Halt-on-Mismatch nie rot. | `daily-paper-reconcile.yml:41` **(verifiziert)** |
| **CI-001** | **Release-Gate nicht-erzwingend auf Synthetik** (`--enforce` fehlt, Random-Walk seed=42) → nie rot. | `release-gate-ci.yml:94-95` **(verifiziert)** |
| **CI-002** | „Release-Gate"-Preset = 12 Doc/CLI-Smoke-Files, null Trading-Numerik. | `run_checks.py:416-432` |
| **CI-008** | **scipy/sklearn ungepinnt** über py3.10/3.11-Matrix → numerische Divergenz; Rule-40-Zahlen veraltet (real 2.2.3/2.2.6). | pyproject vs requirements |
| **DAT-001/003** | Voll-Quality-Gate + Freshness-Monitor **unverdrahtet** → corrupt/stale Feed fließt ungeprüft (Docstring „every batch validated" falsch). | `data/quality_gate.py:170`, `freshness_monitor.py` |
| **DAT-005** | Feeds **fail-OPEN auf leer** (E-025) — Outage nicht von leerem Fenster unterscheidbar. | fred/yfinance/newsapi/… |
| **DAT-006** | Delisting aus Coverage abgeleitet → Feed-Gap fehlklassifiziert lebendes Symbol als delisted. | `data/universe.py:221-250` |
| **QUAL/Zensus-1** | **Silent-Degradation strukturell** — 130 DEBUG-stille Swallows auf sensiblen Pfaden; schlimmste sind Schutz-Reduktionen, die fail-open **mehr** Risiko durchlassen. | `_tc_sizing.py:2330` **(verifiziert)**, `:2374/:2422/:996/:1120/:572/:1170`, `_tc_risk.py:129`, `_tc_signals.py:653`, `_tc_execution.py:519` |
| **STR-001** | **Forward-Label/Feature-Kollision** (latenter OOS-Leak, **eingedämmt**: precomputed-Panels überspringen Enrichment, kein `*_xrank`-Konsument). | `ta_factors_core.py:194-227` → `_tc_features.py:290` → `cross_sectional.py:48` |

**MITTEL (Auswahl):** SEC-2 ledger_path ohne Traversal-Guard; SEC-3 joblib `strict=False` entpickelt bei Hash-Mismatch; SEC-5 Insider/Congress MNPI-Foot-Gun (`as_of=None`); OPS-05 kritische Alerts nur Console/JSON; OPS-06 ungenutzter Run-Lock; CI-004/006 accounting+mypy `continue-on-error`; CI-007 16 open-ended CVE-Ignores; DAT-012 Split-Adjust nicht idempotent; STR-003 Meta-Confidence-Scaling No-Op (bestätigt R2-12); Contract-Drift `status`-Casing `oms.py:129`. Voll-Tabelle in `07_extended_audit.md`.

## Was Runde 3 für die Kernfrage bedeutet

- **(a) OOS-Glaubwürdigkeit:** **unverändert.** Einziger OOS-berührender Fund ist STR-001 — latent + eingedämmt, kein aktiver Leak. **Vorbehalt (UNSURE):** ob je ein OOS-Run Live-Enrichment statt precomputed-Panels nutzte, ist statisch nicht abschließend belegbar → Execution-Check empfohlen.
- **(b) Live/Paper/Ops-Reife:** **schwächer** als selbst Runde 2 nahelegte — das Sicherheitsnetz ist gebaut, aber nicht angeschlossen. Harte Blocker für echtes Geld: OPS-01/02/03/04, SEC-1, OPS-07.
- **(c) CI-Schutz:** **teils real** (2 echte Test-Gates + Security-Scans), **teils Theater** (governance-Gates blocken nicht). „grün" ≠ „geschützt"; merge-required ist off-repo nicht verifizierbar.
- **(d) Systemische Erkenntnis (Runde 1→3):** das wiederkehrendste Muster ist **fail-open Silent-Degradation**; Zensus 1 quantifiziert es erstmals (Design-Default der `_tc_*`-Pipeline, nicht Einzelbug).

## Positiv (Runde 3)

Secret-Hygiene solide (kein getrackter Live-Key, nur last-4 geloggt, kein `eval/exec/shell=True`); Kill-Switch-Deactivate fail-closed + State-Write voll-fsync; `alert_failover.py` loud-failt bei All-Channel-Fail; Umlaut-Pfad via powershell.exe; Data-Layer an geprüften Stellen PIT-sauber (keine E-030/31/32/33, int64); Strategy/ML mehrheitlich kausal (time-sorted Split+Embargo, dead-factor-renorm); Determinismus gesund (durchgehend geseedet); Collection 8059/0-Errors.

## Verifikationsstatus Runde 3

- **Direkt verifiziert:** SEC-1, OPS-07, CI-001 (`--enforce` fehlt), QUAL-05. Übrige agent-berichtet mit file:line (`07a`–`07f`). **Nur statisch — NICHT CI-bestätigt;** Execution-abhängige Items als UNSURE markiert.
- **Fix-Kandidaten (separate Review-Chain-pflichtige Tasks):** OPS-01/02/03/04, SEC-1, OPS-07, CI-001/002, CI-008, DAT-001/003, QUAL/Zensus-1. In diesem Durchgang **bewusst nicht** angefasst.

---

# Fix-Log

| Datum | Fund | Fix | Vorher → Nachher | Belegt |
|---|---|---|---|---|
| 2026-05-30 | **H-1** (`compute_drawdown` `max_drawdown_pct` nutzte GLOBAL End-Peak `rolling_max.max()` als Nenner statt Peak-to-Date) | `qa/metrics.py` `compute_drawdown`: `max_drawdown_pct` jetzt aus `(drawdown_series / rolling_max).min()` (Peak-to-Date je Talsohle); inf/-inf→nan-Guard; absoluter `$`-MDD und `peak_equity<=0→0.0`-Guard unverändert. Keine Consumer-Änderung. | Edge-Case `[100,90,80,120,200]`: MDD% **−10.0 % → −20.0 %** (jetzt ehrlich/größer); absoluter `$`-MDD bleibt **−20.0**. Normalfall ohne neues Hoch (`[100,90,75]`) unverändert −25 %. | 3 neue Regressionstests in `test_qa_metrics.py` (33 passed); Consumer-Selektion 209 passed / 68 skipped / 0 failed. `scripts/ops/check_promotion_gate.py` **NICHT betroffen** (rechnet MDD selbst peak-to-date). Nur lokal — **NICHT CI-bestätigt**. |
