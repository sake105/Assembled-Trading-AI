# 07 — Extended Audit (Runde 3) — Konsolidiert

**Datum:** 2026-05-30
**Art:** Reine Analyse + Verifikation. NICHTS gelöscht/geändert an Produktivcode.
**Methode:** 6 spezialisierte Recherche-Agenten auf bisher **ungeprüften Feldern**, je ein eigener Cluster-Bericht. Die schwersten NEU-Funde vom Orchestrator **direkt am Quellcode nachverifiziert** (markiert „**(verifiziert)**").

**Cluster-Berichte (Detail):**
- `07a_security_secrets_auth.md` — Security / Secrets / API-Auth / MNPI (SEC-)
- `07b_data_sources_ingestion.md` — Data-Sources / Feeds / Ingestion / Corporate-Actions (DAT-)
- `07c_concurrency_ops_scheduler.md` — Concurrency / Durability / Scheduler / DMS / Alerting (OPS-)
- `07d_ci_dependencies.md` — CI/CD-Workflows / Dependency-Drift / Supply-Chain (CI-)
- `07e_strategy_feature_ml.md` — Strategy / Feature-Engineering / ML-Model (STR-)
- `07f_error_handling_contracts.md` — Silent-Except-Zensus / Contract-Drift / Determinismus (QUAL-)

> **Abgrenzung:** Runde 1 (`00_SUMMARY.md`) = OOS-Korrektheit/Metrik-Mathematik. Runde 2 (`06_*`) = Live/Paper-Trading-Cycle. Runde 3 = **alles drumherum**: Security, Feeds, Ops/Scheduler, CI/CD, Strategy/ML-Breite, und ein systematischer Silent-Degradation-Zensus über das ganze `src/`.

---

## Headline-Einordnung (ehrlich)

**Runde 3 ändert die OOS-„kein Edge"-Schlussfolgerung NICHT — aber sie verstärkt das Runde-2-Fazit massiv: die operative Reife (Security, Feeds, Ops, CI-Schutz) ist deutlich schwächer, als der GO_LIVE-Stand (12/16) suggeriert.**

Vier neue systemische Muster, über alle Cluster hinweg:
1. **Das Sicherheitsnetz ist gebaut, aber nicht angeschlossen.** DMS, Heartbeat-Staleness, Multi-Channel-Alerting, zwei Data-Quality-Gates, Freshness-Monitor — alle existieren als sauberer Code und werden im Produktivpfad **nie aufgerufen** (Fortsetzung von R2-4, jetzt über Ops + Data bestätigt).
2. **Silent-Degradation ist nicht punktuell, sondern strukturell.** Zensus: **997 `except`-Handler**, davon **44 % auf sensiblen Pfaden**, **nur 7 % re-raisen**, **130 schlucken still auf DEBUG** — bei Prod-Log-Level unsichtbar. Die schlimmsten ~10 sind Schutz-Reduktionen in `_tc_sizing.py`, die bei Exception **fail-open Richtung MEHR Risiko** kippen.
3. **CI-Schutz ist teils real, teils Theater.** Zwei echte blockierende Test-Gates + echte Security-Scans, ABER die governance-benannten Gates (`release-gate`, `accounting-ci`) blocken nicht wirklich (non-enforcing / `continue-on-error` / `|| true`).
4. **Auth fail-OPEN per Default.** Ohne gesetzten Key sind alle Command-Endpoints offen — inkl. ungesicherter Kill-Switch-**Aktivierung** (DoS-Vektor).

---

## NEU-Funde nach Schweregrad (Runde 3)

### KRITISCH

| ID | Fund | Ort (Beleg) | betrifft |
|---|---|---|---|
| **OPS-01** | **DMS-Daemon ist toter Code.** `scripts/dms_daemon.py` hat **null** Deployment-Referenzen (kein `.ps1`/`.bat`/Task-Scheduler-Artefakt). Auto-Flat-on-stale-heartbeat ist im Live-Betrieb nicht erreichbar. | `scripts/dms_daemon.py` (0 Caller) | Live-Sicherheit (unattended) |
| **OPS-02** | **Heartbeat-Pfad-Mismatch (3-fach).** Produktion schreibt `_tc_execution.py:526` → `output_dir/state/heartbeat.json` UND `paper_trading_scheduler.py:36` → `output/ops/scheduler_heartbeat.json`; DMS liest Default `output/state/heartbeat.json`. Schreiber und Leser sind sich **uneinig** → Staleness-Detection läuft ins Leere. | `_tc_execution.py:526` / `paper_trading_scheduler.py:36` / `dead_man_switch.py` (default) | Live-Sicherheit |
| **OPS-03** | **Staleness-Detektor sieht nie echte Daten.** `check_scheduler_health.py:41` feuert nur gegen einen **synthetisch** veralteten Heartbeat im Drill (`run_alert_drill.py:60` / `fail-drill.yml`). Nie gegen die Live-Datei. | `check_scheduler_health.py:41` | Live-Sicherheit |
| **OPS-04** | **Kill-Switch ohne Locking → TOCTOU.** `execution/kill_switch.py` hat keinerlei File-Lock; die Audit-Hash-Chain ist ein Read-Modify-Write über `_last_audit_hash` (`:142`/`:168`) — unter gleichzeitiger Aktivierung kann die Kette brechen/racen. | `execution/kill_switch.py:142,168` | Concurrency / Audit-Integrität |

### HOCH

| ID | Fund | Ort (Beleg) | betrifft |
|---|---|---|---|
| **SEC-1** | **API-Auth fail-OPEN per Default.** `require_api_key` warnt + erlaubt, wenn `ASSEMBLED_API_KEY` ungesetzt → alle Command-Endpoints (Paper-Order, Reset, **Kill-Switch-ACTIVATE**) unauthentifiziert. Activate ist NICHT operator-token-gated (nur Deactivate ist es) → jeder kann das Trading per Kill-Switch lahmlegen. `/ready` ruft `auth_is_configured()` nie auf → Lücke unsichtbar. | `api/auth.py:43-46` **(verifiziert)** | Security (bei Nicht-Loopback-Exposure) |
| **OPS-07** | **CI-Reconcile neutralisiert.** `daily-paper-reconcile.yml:41` ruft `run_reconcile_worker.py --dry-run || true` → Halt-on-Mismatch kann den Job nie rot machen (deckt sich mit dem 2026-04-10 $412.54-Silent-Stall-Postmortem). | `daily-paper-reconcile.yml:41` **(verifiziert)** | Reconcile-Schutz |
| **CI-001** | **Release-Gate ist nicht-erzwingend auf Synthetik-Daten.** `release-gate-ci.yml:95` ruft `release_gate_walk_forward.py --verbose` **ohne `--enforce`**, Step-Name „grace period through 2026-07-01"; der WF + Deflated-Sharpe-Gate läuft auf synthetischem Random-Walk (`_synthetic_prices`, seed=42) → kann **nie** rot werden. | `release-gate-ci.yml:94-95` **(verifiziert)** + `release_gate_walk_forward.py:296-300` | Release-Gate (Theater) |
| **CI-002** | **„Release-Gate"-Preset prüft Docs, nicht Trading.** `run_checks.py:416-432` Release-Preset = 12 Doc-/CLI-Smoke-/Inventory-Testdateien. Null Risk/Execution/Portfolio/Accounting-Numerik. | `scripts/run_checks.py:416-432` | Release-Gate (Theater) |
| **CI-008** | **Dependency-Drift über Rule 40 hinaus.** `scipy`/`sklearn` **ungepinnt** über die py3.10/3.11-Matrix → Legs installieren verschiedene Versionen → numerische Divergenz. `arch` Major-Gap 6→8, numpy 1.24-Floor vs 2.2.6-Pin. Rule-40-Zahlen (pandas/numpy ==2.3.3) **veraltet** (real 2.2.3/2.2.6). `statsmodels`/`pandas-market-calendars` in requirements, aber NICHT in pyproject-deps. | `pyproject.toml` vs `requirements.txt`; `backend-ci.yml` Matrix | Local↔CI-Konsistenz |
| **DAT-001** | **Voll-OHLCV-Quality-Gate UNVERDRAHTET.** `data/quality_gate.py:170` (monotonic/null/high<low/pandera) hat **null** Prod-Caller; Docstring „every batch is validated" ist falsch. Zusatz-Bug: Gate prüft `Open/High/...` (Caps) gegen lowercase-Panel → würde selbst verdrahtet nie matchen. | `data/quality_gate.py:170` | Data-Integrität |
| **DAT-003** | **Kein Staleness-Gate auf Caches.** `freshness_monitor.py` ist rein in-memory, liest nie Cache-mtimes, wird von keinem Ingestion-Pfad importiert → ein veralteter Preis/Macro/News-Parquet fließt ungeprüft weiter. | `data/freshness_monitor.py` (0 Ingestion-Caller) | Data-Integrität |
| **DAT-005** | **Feeds fail-OPEN auf leer (E-025).** `fred_source.py:144-228`, `yfinance_source.py:128-159`, newsapi/worldbank/finnhub/cboe geben bei Fehler **empty** zurück → Total-Outage ist am Return-Typ nicht von „legitim leeres Fenster" unterscheidbar. | s. `07b` DAT-005 | Data-Integrität (systemisch) |
| **DAT-006** | **Delisting aus Coverage abgeleitet.** `universe.py:221-250` schließt Delisting aus Panel-Lücken; ein Feed-Gap **fehlklassifiziert ein lebendes Symbol als delisted** (Survivorship-Kopplung in die falsche Richtung). | `data/universe.py:221-250` | Universe/Survivorship |
| **QUAL/Zensus-1** | **Silent-Degradation strukturell.** 997 `except`-Handler, 435 (44 %) auf sensiblen Pfaden, nur 71 (7 %) re-raisen, **130 still auf DEBUG** (prod-unsichtbar). Schlimmste ~10 sind Schutz-Reduktionen, die fail-open **mehr** Risiko durchlassen: `_tc_sizing.py:2330` (Halt-Check) **(verifiziert)**, `:2374` (Buying-Power-Cap), `:2422` (Pre-Earnings-Cut), `:996` (Trailing-Stops), `:1120` (Korrelations-Guard), `:572` (Vol-Targeting), `:1170` (Crash-Cap), `_tc_risk.py:129` (EVT-VaR), `_tc_signals.py:653` (Ensemble-Drop, **null Log**), `_tc_execution.py:519` (Trade-Journal-Write → Audit-Lücke). | s. `07f` | Live/Paper-Risk (systemisch) |
| **STR-001** | **Forward-Label/Feature-Kollision (latenter OOS-Leak).** `ta_factors_core.py:194-227`: `returns_12m` + `momentum_12m_excl_1m` sind FORWARD-Labels (`shift(-N)`) im selben Frame; `_tc_features.py:290` Default-`rank_cols` ranked `momentum_12m_excl_1m` cross-sektional in eine **Live-Feature-Spalte** `*_xrank` (`cross_sectional.py:48`). **Heute eingedämmt** (precomputed-Backtests überspringen das Enrichment `_tc_features.py:254-261`; kein Konsument von `*_xrank`), aber latent. | `ta_factors_core.py:194-227` → `_tc_features.py:290` → `cross_sectional.py:48` | OOS (latent, eingedämmt) |

### MITTEL

| ID | Fund | Ort (Beleg) | betrifft |
|---|---|---|---|
| **SEC-2** | `ledger_path`-Query-Param ohne Path-Traversal-Guard (anders als `/health`) → File-Existence-Oracle, Pfad ins Log. | `api/routers/ledger.py:33-74` | Security |
| **SEC-3** | joblib-Hash-Check `strict=False` Default → bei SHA256-Mismatch wird **trotzdem entpickelt** (Modelle in user-writable `output/models/`). | `ml/model_registry.py:109-132`, `signals/meta_model.py:541` | Security/ML |
| **SEC-4** | Rohes `str(exc)` in HTTP-500-Detail über oms/paper_trading/risk/orders/diagnostics → Info-Leak. | `api/routers/*` | Security |
| **SEC-5** | Insider/Congress-PIT-Filter ist opt-in via `as_of`; `as_of=None` überspringt; `ensure_event_schema` leitet `disclosure_date` mit Null-Latenz aus Trade-Timestamp ab → MNPI-förmiger Look-Ahead (latent, Pipelines archiviert). | `features/insider_features.py:88`, `data/latency.py:53-57` | MNPI-Compliance (Foot-Gun) |
| **OPS-05** | `daily_scheduler.py:738` alarmiert via `AlertManager` (nur Console/JSON), **nicht** das Multi-Channel-`alerting.py` → kritische Events verlassen die Box nie. | `ops/daily_scheduler.py:738` | Alerting |
| **OPS-06** | `paper_trading_scheduler.py:38` `LOCK_PATH` definiert, **nie benutzt** → kein Run-Overlap-Guard (Doppellauf möglich). | `paper_trading_scheduler.py:38` | Concurrency |
| **OPS-08** | Soft-Timeout nur Checkpoint-basiert (`run_live_paper.py:458`); `register_paper_pilot_task.ps1:80` ExecutionTimeLimit 15 min vs Code-Annahme 25 min → OS-Hard-Kill vor Graceful-Bailout. | `run_live_paper.py:458`, `register_paper_pilot_task.ps1:80` | Scheduler |
| **CI-004** | `accounting-ci.yml:61-64` Accounting-Preset `continue-on-error` → kann nicht failen. | `accounting-ci.yml:61-64` | CI (Theater) |
| **CI-006** | `backend-ci.yml:138-142` mypy doppelt maskiert (`\|\| true` + `continue-on-error`) → Typ-Sicherheit unerzwungen. | `backend-ci.yml:138-142` | CI |
| **CI-007** | 16 pip-audit-CVE-Ignores open-ended (kein Ablauf/Issue-Link); `requests` CVE-2024-47081 überfällig (`==2.32.3`). | `backend-ci.yml:101-117` | Supply-Chain |
| **CI-009** | Windows-Governance-Jobs installieren ad-hoc **ungepinnte** pip-Listen statt requirements.txt. | `*-windows*.yml` | Local↔CI |
| **CI-010** | Kein `needs:` irgendwo → ein grüner Check impliziert nicht, dass Upstream-Jobs passten. | `.github/workflows/*` | CI-Topologie |
| **CI-011** | Producer-Crons soft-failen → Stale-File-Freshness-Gates können dunkel werden, während grün. | `.github/workflows/*` | CI |
| **DAT-002** | Zweites Voll-Gate `DataQualityGate` (`dataquality/gate.py:26`) ebenfalls test-only → zwei Gates, keines benutzt (Doppelstruktur, Rule-50-Drift). | `dataquality/gate.py:26` | Data/Architektur |
| **DAT-007** | Nur yfinance hat echtes Retry/Backoff; andere Feeds rotieren Keys, retrien aber transiente 5xx nicht. | s. `07b` | Data-Robustheit |
| **DAT-009** | `prices_ingest.py:148-161` invalide OHLC nur WARN, kein Block. | `data/prices_ingest.py:148-161` | Data |
| **DAT-010** | `incremental_update` non-atomar + restatement-blind. | s. `07b` DAT-010 | Data-Durability |
| **DAT-012** | Split-Adjust **nicht idempotent** (Doppel-Apply teilt doppelt), kein `already_adjusted`-Guard. | s. `07b` DAT-012 (UNSURE) | Data-Korrektheit |
| **STR-003** | Meta-Confidence-**Scaling** ist No-Op (`mf_score` vs Live-Col `score`) — bestätigt + lokalisiert R2-12. Filter wirkt, Scaling nicht. Doppelt aus (meta_model.enabled=false). | `multifactor_signal.py:1010` ↔ `_tc_signals.py:591` | Live (toter Pfad) |
| **STR-002** | `cross_asset_carry.py:47-61` zieht Live-yfinance-„now", kein as_of → nicht PIT-safe. Gedeckelt: nicht verdrahtet (0 Caller). | `signals/cross_asset_carry.py:47-61` | (research-only) |
| **QUAL/Zensus-2** | Contract-Drift: `status`-Casing `oms.py:129` „FILLED" vs contract lowercase „filled"; `side`-lowercase-Insel (`round_trip_detector`/`order_gate`/`order_management`/`limit_orders_v1`/`broker_adapter`) vs contract `BUY`/`SELL`. Empty-Orders-5-Spalten-Schema = wiederkehrende KeyError-Falle. | s. `07f` Census 2 | Contract-Konsistenz |

### NIEDRIG / Dummy / Info

- **Dummy/Stub (bestätigt):** `ml/logic_tensor_network.py` + `ml/temporal_fusion_transformer.py` (`NotImplementedError`-Stubs, ehrlich gelabelt). `orchestrator.py:1439` Factor-Decay mit `panel_df=None` → No-Op jeden Run. `news_alpha/exit_rules.py:91` Reversal-Exit dokumentiert, Param `new_trigger_items` ungenutzt. `data/download_all_market_data.py:33` `_fetched_at`-Marker ohne Konsument (DAT-004).
- **STR-009 (latent):** `meta_model.py:312-326` `exclude_cols` lässt Forward-Labels aus; Auto-Detect-Feature-Selection würde sie picken → katastrophaler Leak. Latent (Prod trainiert via `dataset_builder` Prefix-Allowlist).
- **STR-008:** `risk_metrics.py:601` Forward-`returns_12m` in einem Korrelations-Diagnostic (nur Diagnostik).
- **DAT-011:** `cboe`-Docstring sagt FRED, Code nutzt yfinance (Doku-Drift).
- **OPS-09…16:** Ledger ohne fsync (= R2-18), keine Fill-Dedupe (= R2-19), Shadow-Mode-False-Assurance-Log, SIGTERM unzuverlässig bei `taskkill /F`, JSON-vs-SQLite-Doppel-Ledger, Wall-Clock-`sleep` für Timeouts, `clock_drift.py` unverdrahtet, Subprocess-rc geschluckt.
- **Determinismus (Zensus 3): gesund.** Durchgehend geseedete RNG (`default_rng(seed)`, `random_state=42`); kein unseeded random, kein `datetime.now()`/`os.environ` in Entscheidungs-Pfaden. Einzige `set()`-Iteration (`trading_cycle_shared.py:1311`) ist kommutativ/kosmetisch. Ein TODO in sensiblen Dirs (`orchestrator.py:1431`).

---

## Positiv bestätigt (Ehrlichkeit in beide Richtungen)

- **Secret-Hygiene solide:** kein getrackter Live-Secret; Key-Rotator loggt nur last-4, nie Volltext; kein Secret in Logs; `.env.example` nur Platzhalter; `.gitignore` deckt `.env*`. Kein `eval`/`exec`/`os.system`/`shell=True`/unsafe-`yaml.load`/`pickle.load`-on-input.
- **Kill-Switch-Deactivate fail-closed** mit `hmac.compare_digest`; State-Write voll fsync+dir-fsync+atomic-replace (OPS-PC-02).
- **`alert_failover.py` loud-failt**, wenn ALLE Kanäle scheitern (kein stiller Alert-Verlust an der Failover-Schicht).
- **Umlaut-Pfad korrekt** via powershell.exe (CreateProcessW), nicht cmd.exe.
- **Data-Layer PIT-sauber an den geprüften Stellen:** Volume-Junk **raised** statt geschluckt; Corp-Actions raisen/WARNen bei Schema-Drift, PIT-korrekter Delisting-Preis; strikte PIT-Universe-API; Resample droppt Partial-Period; **keine E-030/E-031/E-032/E-033 im Data-Layer** (int64-Casts); Synthetik-Generator ehrlich gelabelt (kein Dummy-als-Real).
- **Strategy/ML mehrheitlich kausal + ehrlich:** Meta-Model time-sorted Split + Embargo; mfv2 Dead-Factor-Renormalisierung (Nullen verwässern nicht); `cross_sectional` per-Timestamp-PIT; `dual_momentum`/`low_max_lottery` vermeiden E-030/E-031 explizit; `vol_target` min_periods-Disziplin.
- **Collection gesund:** 8059 Tests, 0 Collection-Errors (collect-only verifiziert). Zwei **echte** blockierende Test-Gates (`ci.yml` ~7511 Tests, `backend-ci.yml` fast+regression ~2836, beide `--maxfail`) + echte blockierende pip-audit/bandit/gitleaks/detect-secrets-Scans.
- **Determinismus gesund** (Zensus 3, s. o.).

---

## Gesamteinordnung — was bedeutet Runde 3?

**(a) OOS-Ergebnis-Glaubwürdigkeit:** **unverändert** gegenüber Runde 1/2. Der EINZIGE Runde-3-Fund, der den OOS-Pfad berührt, ist **STR-001** (Forward-Label/Feature-Kollision) — und der ist **latent + eingedämmt**: precomputed-Backtests überspringen das Enrichment, und keine Downstream-Spalte konsumiert `*_xrank` namentlich. Es gibt damit **keinen** neuen aktiven OOS-Leak. **Offener Vorbehalt:** ob jemals ein OOS-Run die Live-Enrichment statt precomputed-Panels nutzte, ist statisch nicht abschließend belegbar → Execution-Check empfohlen (UNSURE).

**(b) Live/Paper/Ops-Reife:** **schwächer** als selbst Runde 2 schon nahelegte. Das Sicherheitsnetz (DMS, Heartbeat-Staleness, Multi-Channel-Alert, Data-Quality-Gates, Freshness-Monitor) ist **gebaut, aber im Produktivpfad nicht angeschlossen** (OPS-01/02/03, DAT-001/003). Für echtes Geld sind die KRITISCH-Items (OPS-01/02/03/04) + SEC-1 + OPS-07 die harten Blocker.

**(c) CI-Schutz:** **teils real, teils Theater.** „7 Workflows grün" ≠ „System CI-geschützt": die governance-benannten Gates (`release-gate`, `accounting-ci`) blocken nicht (CI-001/002/004). Ob die Workflows überhaupt merge-*required* sind, ist Branch-Protection (off-repo, nicht aus dem Repo verifizierbar).

**(d) Systemische Erkenntnis:** Das wiederkehrendste Muster des ganzen Audits (Runde 1→3) ist **fail-open Silent-Degradation**. Zensus 1 quantifiziert es erstmals: 130 DEBUG-stille Swallows auf sensiblen Pfaden, die schlimmsten in `_tc_sizing.py` kippen Schutz-Reduktionen bei Exception Richtung **mehr** Risiko. Das ist kein Einzelbug, sondern ein Design-Default der `_tc_*`-Pipeline.

---

## Verifikationsstatus

- **Direkt am Quellcode verifiziert:** SEC-1 (`auth.py:43-46`), OPS-07 (`daily-paper-reconcile.yml:41`), CI-001 (`release-gate-ci.yml:94-95` — `--enforce` fehlt), QUAL-05 (`_tc_sizing.py:2330`).
- **Agent-berichtet mit file:line, nicht einzeln nachverifiziert:** alle übrigen Funde — Belege in `07a`–`07f`.
- **Nur statisch / lokal — NICHT CI-bestätigt.** Execution-abhängige Items in den Cluster-Berichten als UNSURE markiert (u. a. DAT-012 Doppel-Split-Apply, STR-001 precomputed-vs-live-Enrichment).

## Offene Folge-Hinweise (NICHT in diesem Durchgang ausgeführt)

- Fix-Kandidaten als separate, gezielte Tasks (Review-Chain-pflichtig, `src/`/`.github/` betroffen): OPS-01/02/03 (Safety-Net verdrahten), OPS-04 (Kill-Switch-Lock), SEC-1 (Auth fail-closed + Activate-Gate), OPS-07 (Reconcile-`\|\| true` entfernen), CI-001/002 (Release-Gate erzwingen + echte Numerik), CI-008 (scipy/sklearn pinnen), DAT-001/003 (Quality-/Freshness-Gate verdrahten), QUAL/Zensus-1 (`_tc_*` Schutz-Swallows fail-closed).
- Nicht tief gelesen (Empfehlung bei Live-Schaltung): polygon/alphavantage/stooq/kalshi/polymarket/bls/finra/weather/wikipedia-Source-Bodies.
- Scratch-Dateien aus Runde 1 liegen noch: `docs/audit/_scratch_numeric_verification.py`, `docs/audit/_scratch_dd_edgecase.py`.
