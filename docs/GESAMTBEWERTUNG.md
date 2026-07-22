# GESAMTBEWERTUNG — Assembled-Trading-AI

**Datum:** 2026-07-19 · **Modus:** Read-only-Gesamtaudit mit frischen Augen
**Methode:** 9 parallele Bereichs-Audits (Datenlayer, Features/ML, Backtest/Validierung, Risk/Portfolio/Strategien, Execution/Paper/Ops/API, QA/CI, Tests, Governance/Doku, Research-Re-Review), jeweils mit Datei:Zeile-Belegpflicht. Die vier folgenreichsten Einzelbefunde habe ich zusätzlich selbst im Code nachgeprüft.
**Kennzeichnung:** **[V]** = verifizierter Befund (Beleg vorhanden) · **[E]** = Einschätzung/Verdacht · **[G]** = Geschmacksfrage.
**Nicht geprüft:** Es wurden keine Backtests reproduziert, keine CI-Runs angestoßen, kein Broker-API-Call gemacht. Testläufe: nur `pytest --collect-only` plus zwei Mini-Läufe (5 bzw. 11 Tests).

---

## 1. Gesamteindruck

Das ist ein ungewöhnliches Projekt: Die **Research-Ehrlichkeit ist außergewöhnlich gut** — besser als bei den meisten professionellen Quant-Shops, die ich als Maßstab nehmen würde. Pre-Registration, Verdict-Ledger, dokumentierte Selbstkorrekturen (E-051, E-052, mark-to-market→End-Liquidation), ein gelebtes Anti-Pattern-Register mit 53 Einträgen. Das negative Kernergebnis („kein Edge nach Steuern") wurde nicht wegdiskutiert, sondern als Ergebnis akzeptiert. Das ist die größte Stärke des Projekts, und sie ist selten.

Gleichzeitig gibt es ein systematisches Muster von **Selbstbetrug an anderer Stelle**, und zwar dort, wo es das Projekt am wenigsten erwartet: nicht in der Research-Schicht, sondern in der **Betriebs- und Absicherungsschicht**. Konkret:

- Die Test-Suite meldet „8884 Tests, 0 Collection-Errors" — real sind **~1000 Tests strukturell skip-tot** (importorskip auf gelöschte Module) und **~1000 weitere Keyword-Grep-Pseudotests**. Die Kopfzahl überzeichnet die Abdeckung massiv. **[V]**
- Die QA-Gate-Kette ist **im Live-Pfad nicht verdrahtet** (`qa_status=None` an beiden Call-Sites) — „QA sagt BLOCK" erreicht keine Order. **[V]**
- Der Order-Lifecycle-Log (GO_LIVE-Kriterium C1, als „erfüllt" abgehakt) ist **im echten Broker-Pfad nicht angeschlossen**. **[V]**
- Der DMS „flattet" nichts — `close_all_positions` existiert nirgends im Code; der Docstring behauptet es trotzdem. **[V]**
- Statusdokumente (GO_LIVE_CHECKLIST, AGENTS.md, KNOWN_ISSUES.md, PROJEKT_STATUS.md) enthalten **aktiv widersprüchliche „zweite Wahrheiten"** — genau das, was CLAUDE.md verbietet. **[V]**

Der Paper-Pilot selbst hat Stand heute ein **akutes operatives Problem** (offene, nie stornierte After-Hours-Orders vom 14.07. bei blindem Ledger — Details §2.5), das vor dem nächsten Scheduler-Lauf am Mo 20.07. 21:30 geklärt werden muss.

Kurz: **Research-Schicht: erwachsen. Betriebsschicht: Fassade an mehreren tragenden Stellen.** Das Projekt weiß viel über technische Ehrlichkeit — es wendet sie nur asymmetrisch an.

---

## 2. Bereich für Bereich

### 2.1 Datenlayer (inkl. EDGAR-Raw)

**Zustand: solide Basis mit drei konkreten Fallen.**

Gut **[V]**:
- Survivorship-Guards mit lautem `[SURVIVORSHIP-BIAS-RISK]`-Log (`data/universe.py:108,131,154,180-185`); Corporate-Actions fail-loud (`data/corporate_actions.py:364-370`); Macro-GPR mit PIT-Release-Lag 32 Tage (`data/macro/gpr.py:92-116`).
- EDGAR-Raw-Pulls (`data/raw/fundamentals/`, `data/raw/insider_congress/`) haben starke Manifeste mit expliziter PIT-Semantik (ACCEPTANCE-DATETIME als `available_at`, Restatement-Versionierung, benannte Failed-Symbole). Konsistenz stichprobenhaft plausibel (180 Dateien = 180 symbols_pulled).
- `scripts/ops/refresh_daily_cache_from_eodhd.py` (neu) **hat** den E-053-PIT-Cutoff (`:123-133`, `timestamp < today_utc`) und spiegelt das Schwester-Idiom korrekt; atomarer Write; 0 same-day-Rows im Live-Cache (heute geprüft).

Schlecht **[V]**:
- **Fail-open im einzigen verdrahteten Preisdaten-Gate:** `prices_ingest.py:164-170` — bei `valid=False` mit „blocking issues" wird nur gewarnt, das DataFrame trotzdem zurückgegeben. „Blocking" ist Etikett, nicht Verhalten.
- **Der dritte Cache-Writer hat KEINEN eigenen PIT-Cutoff:** `scripts/ops/refresh_daily_cache_from_panel.py` appendet ohne `timestamp < today`-Guard — Schutz nur transitiv über yfinance-Semantik. Direkter Verstoß gegen die E-053-Lehre „jeder Ingest-Pfad erzwingt den Cutoff selbst".
- **TOTAL-RETURN-Falle undokumentiert im Loader-Vertrag:** `daily.parquet.close` ist TR-adjusted, aber weder `pipeline/io.py` noch `prices_ingest.py` sagen das; `docs/CORPORATE_ACTIONS.md:50` behauptet das **Gegenteil** („Dividenden werden nicht adjustiert").
- **35 % der Rows haben `adj_close=NaN`** (97.135 von 275.308; NaN-Sentinel des Panel-Writers) — wachsende Falle für Direct-parquet-Konsumenten.
- Drei parallele Preis-Loader (`pipeline/io.py:17-47` vs. `:50-101` vs. `load_eod_prices`) mit unterschiedlicher Koercion/Sortierung — E-052-Muster im Kern selbst.
- Legacy-Writer `scripts/data/assemble_eod_daily.py` würde bei Aufruf `daily.parquet` mit einem 2-Symbol-Stooq-Bestand **überschreiben** — toter, gefährlicher Pfad.
- Cache-Stand 2026-07-14; 15.–17.07. fehlen (3 Handelstage stale).

Verdacht **[E]**: `unified_paper_engine.py:1898-1912` schreibt Dividenden als Cash gut und split-adjustiert Preise — auf einem bereits TR-adjustierten Store. Ob live eine CA-Quelle verdrahtet ist (→ Doppelzählung) oder `actions` leer bleibt (→ no-op), ist ungeklärt. Muss vor jeder Ergebnisdeutung des Piloten geklärt werden.

Offen: keine Hashes/Integritätsprüfung in den data/raw-Manifesten (Infrastruktur in `snapshot.py`/`data_versioning.py` vorhanden, ungenutzt); die 53+9 MB EDGAR-Pulls werden nur von research/ konsumiert — zwei getrennte EDGAR-Fetch-Stacks.

### 2.2 Feature-/Faktor-/Signal-Ebene

**Zustand: PIT-Disziplin weitgehend gut; ein verifizierter Rest-Look-Ahead; Faktor-Zoo größer als sein Nutzen.**

Gut **[V]**: `event_features.py:38-143` erzwingt `as_of` und filtert auf `disclosure_date`; der früher dokumentierte mfv2-Panel-Self-Anchoring-Leak ist gefixt (`multifactor_v2.py:1393-1469` slict Panel auf `timestamp <= as_of`); PEAD-XBRL-Pfad PIT-gated auf EDGAR-Acceptance (`:1229-1281`); genullte Faktoren tragen ehrliche Inline-Begründungen (`:244-298`).

Schlecht:
- **`_compute_options_factors` hat keinen `as_of`-Parameter** (`multifactor_v2.py:987-1026`, selbst nachgeprüft): fetcht CBOE live, nimmt `iloc[-1]` → Look-Ahead im Backtest/Replay für `options_put_call_extreme` (w=0.02) und `vix_regime_score` (w=0.01). Als Follow-up seit 2026-05-28 bekannt, **weiterhin offen**. **[V]**
- **21× `except Exception` in multifactor_v2.py** degradieren Faktoren still auf 0.0 — ein Feed-Ausfall reduziert unbemerkt die effektive Faktorzahl. **[V]**
- Faktoren, die in der OOS-Forensik (Paket 3c.2) ZERO-Beitrag zeigten (`sector_rotation_bias` 0.04, `news_sentiment_7d` 0.04, options/VIX 0.03), tragen weiter Default-Gewicht — Gewichtsbudget auf nachweislich beitragsfreien Faktoren. **[V]**
- Leichen: `src/erweiterung/` enthält nur noch `.pyc`-Dateien; `signals/pead_sue.py` (Finnhub-Duplikat) nur test-referenziert; Intel-Module `polymarket_loader`, `wild_card_detector`, `structural_cycles` orphan. **[V]**

ML-Ebene ehrlich eingeordnet **[V]**: produktiv sind HMM-Regime, Copula-Tail-Risk, EDCL (policy `enabled: true`); conformal sizing und meta_model sind implementiert, aber **policy-deaktiviert mit ehrlicher Begründung** (`policy.yaml:507-509`: „CAGR −3.99pp"; `:665-668`: „v2 AUC=0.649 had look-ahead bias; v6 AUC=0.51"). GNN/TFT/LTN sind deklarierte Stubs. Das ist sauber gehandhabt — nur der Alt-Kommentar `_tc_signals.py:566` bewirbt noch das kontaminierte AUC-0.649-Modell.

news_alpha ist real, kein Stub **[V]**: Pipeline verdrahtet (`_tc_sizing.py:1820-2030`, Cap-never-boost), `policy.yaml:159-167` scharf (`shadow_only: false`), Intraday-Runner mit echtem Alpaca-Pfad.

### 2.3 Backtest / CPCV / Walk-Forward / Validierung

**Zustand: methodisch reich, aber das kanonische Gate hat einen Formelfehler in die falsche Richtung.**

- **DSR-Standardfehler-Bug (anti-konservativ), selbst nachgeprüft [V]:** `qa/deflated_sharpe.py:130` setzt `(excess_kurtosis − 1)/4 · SR²` ein; BLP 2014 verlangt rohe Kurtosis (normal=3). Für normale Returns ergibt der Code `1 − SR²/4` statt korrekt `1 + SR²/2` → SE unterschätzt → DSR-Wahrscheinlichkeit **überhöht**, Gate zu freundlich. Bei Tages-Returns klein (<1 % relativ), wächst quadratisch mit SR. Da fast alle Verdicts FAIL waren, kippt rückwirkend vermutlich nichts **[E]** — aber jedes künftige knappe PASS wäre damit kontaminiert.
- **CPCV degradiert still:** `qa/cpcv_validation.py:104-108` — skfolio-Exception → `logger.debug` + Fallback auf `TimeSeriesSplit` **ohne Purge/Embargo**, für den Aufrufer unsichtbar. „Purged" kann stillschweigend „unpurged" bedeuten. **[V]**
- Walk-Forward-Purge existiert, Default 0, kein OOS-Runner setzt ihn (`qa/walk_forward.py:127-138`; `scripts/_oos_wf_*`) — für die MA-/Rotationsstrategien ohne Forward-Labels vertretbar und im Leakage-Check ehrlich dokumentiert. **[V]**
- `ml/purged_cv.py` purged in die korrekte Richtung, aber der Docstring behauptet das Gegenteil („embargo after test", `:22`). **[V]**
- **Turnover-Gate mit stillem Except-Skip:** `pipeline/_tc_sizing.py:1065-1066` — schlägt das Gate fehl, läuft der Zyklus ungecappt weiter, nur DEBUG-Log. **[V]**
- **Kosten-Defaults nicht konservativ:** effektiv ~1.75 bps/Trade im Default-Pfad (`costs.py:50`, `pipeline/backtest.py:205-207`); realistischere Tier-Tabelle nur im Paper-Engine-Pfad; `_TIER_YAML_PATH` CWD-relativ (`costs.py:58`). **[V]**
- Same-Bar-Close-Fills im Engine-Pfad (`qa/backtest_engine.py:873-899`): Signal aus Close[t], Fill zu Close[t] — nur mit MOC-Annahme fair. **[V]** Zusammen mit den Kosten-Defaults: Backtests sind strukturell 1–3 bps/Trade zu freundlich **[E]** — verdictsneutral bei FAILs, gefährlich bei knappen PASSes.
- E-051-Restmuster: `trading_cycle_shared.py:1388` (`for grp in set(groups)`) — kommutativ, nur Audit-Diff-Rauschen; `qa/factor_analysis.py:1553-1560` instabile Sortierung + `rank(method="first")` — nur Reporting-Pfad. **[V]**
- Frühere QA-Ehrlichkeitsbefunde (z-Score-DSR, Pseudo-PBO) sind sauber remediert (DeprecationWarnings, ehrliche Docstrings); echtes CSCV-PBO existiert weiterhin nicht (dokumentiert). **[V]**
- Top-level `experiments/20251203_*` ist Friedhof (Yahoo-Smoke-Runs Dez 2025). **[V]**

### 2.4 Risk / Portfolio / Strategien

**Zustand: die handgebauten Kern-Guards sind gut; die zweite Verteidigungslinie ist teils nicht armiert oder nicht existent.**

Gut **[V]**:
- Kill-Switch: Deaktivierung OPERATOR_KILL_TOKEN-gated mit `hmac.compare_digest` (`execution/kill_switch.py:417-462`), Engaged-State dreifach redundant, Lesefehler → fail-closed Block (`pre_trade_checks.py:1159-1163`).
- Drawdown-Soft-Halt exakt wie dokumentiert: Evaluations- und Persistenz-Scope sauber getrennt (`scripts/run_live_paper.py:380-447`), Halt-Write atomar, E-049-Lehre umgesetzt.
- `risk/state_machine.py:112-182` vorbildlich gegen Concurrent-Writes (unique-tmp + FileLock).
- Pre-Trade-Constraints fail-closed (`pre_trade_checks.py:272-281`, `:507-509`).

Schlecht:
- **Reconcile-Block-Gate: implementiert, aber nicht armiert.** `ops/_paper_runner_gates.py` (fail-closed `ReconcileDecision`) existiert — `configs/app.yaml:51-52` sagt `enabled: false`. Der bekannte „next-cycle seam" ist im Code geschlossen, in der Config offen. Bewusst oder vergessen? Nicht ablesbar. **[V]**
- **„Liquidation" liquidiert nichts:** `close_all_positions` existiert nirgends in src/ (Grep: 0 Treffer); `ops/dead_man_switch.py:145-158` aktiviert selbst bei `flatten_mode: market` nur den Kill-Switch (Order-Block). Docstring Z. 1 („auto-flatten all positions") ist **falsch**. Phase 2 unverändert nicht gebaut. **[V]**
- **Halt-Gate prüft nur cash_diff:** reine Positions-Mismatches (qty-Diffs, fehlende Symbole) bei kleinem Cash-Diff lösen nie einen Halt aus (`run_live_paper.py:66-87,673-705`). **[V]**
- Preflight fail-open bei Nebenchecks: gefundene **pending intents** (Crash-Reste) führen nur zu WARN, der Zyklus handelt trotzdem (`run_live_paper.py:508-516`). **[V]**
- CWD-Falle: `ack_halt.py:33` und `ops_watchdog.py:20-24` nutzen relative Pfade; unter Task-Scheduler mit CWD ≠ Repo-Root sieht der Watchdog keinen Halt-Flag bzw. ack_halt cleart ins Leere. **[E]** (Scheduler-Working-Dir read-only nicht verifizierbar.)
- target_qty-Konsolidierung unverändert offen: ~50 echte Emitterstellen in 17 Dateien, `target_notional`-stale-Falle nur per Docstring geschützt (`portfolio/position_sizing.py:1-28`). **[V]**
- Vol-Targeting-Overlay verdrahtet, aber `policy.yaml:338-339` `enabled: false`; Fehler im Overlay → fail-open auf volle Exposure (sichtbar degradiert). **[V]**

`strategies/` vs. `strategy/` ist **keine** Doppelwahrheit (Singular deklariert sich als RESEARCH-ONLY) **[V]** — aber `run_paper_live.py` vs. `run_live_paper.py` ist ein verwirrendes Legacy-Paar **[E]**. Live im Pilot ist nur `trend_baseline` (`app.yaml:18`); der Rest des Strategie-Zoos ist Research-Inventar.

### 2.5 Execution / Paper-Pilot / Ops / API

**Zustand: Idempotenz-Fundament gut; drei echte Bugs; und ein akutes operatives Problem.**

**Akut (vor Mo 20.07. 21:30 zu klären) [V]:**
- Der Pilot steht seit 14.07. (Rechner aus 15.–18.07.; Scheduler-Log heute 17:41 neu gestartet; keine Logs/Manifest-Einträge für 15.–17.07., alles Handelstage). Niemand hat alarmiert — der Watchdog läuft auf demselben Host (Single Point of Failure).
- Der letzte Run (14.07. 22:08 CEST = **16:08 ET, nach Börsenschluss**) submittete 5 Market-BUYs (AAL, BIIB 17, MRNA 52, TDG 3, V 10). Alle 5 liefen in den 120-s-Timeout („accepted — not converting to fill") und wurden **nicht storniert**. Das Ledger (cash 72.097,94, nur GLD/TLT) kennt sie nicht. Als DAY-Orders sind sie vermutlich am 15.07-Open gefüllt worden **[E]** → ~47k$ Positionsaufbau bei blindem Ledger → der nächste Reconcile-Halt ist programmiert. Es gibt **kein Market-Hours-Gate** vor Submit. **[V]**

Bugs **[V]:**
1. **Stale-Order-Cleanup storniert ALLE Orders:** `run_live_paper.py:481-492` berechnet `stale_ids`, ruft dann `adapter.cancel_all_orders()` — das Log („recent orders left untouched") beschreibt ein Verhalten, das der Code nicht hat. Selbst nachgeprüft.
2. **Teilfills gehen verloren:** nur `status=="filled"` wird gebucht (`broker_execution.py:328-357`); `partially_filled` kommt in execution/ nicht vor; `filled_qty` einer Timeout-Order wird nie gebucht → systematischer Ledger-Drift.
3. **Order-Lifecycle-Log im Broker-Pfad nicht verdrahtet:** `append_lifecycle_event` wird von `broker_execution.py` nie aufgerufen; `output/journal/order_lifecycle.jsonl` endet 2026-05-30 mit Pseudo-IDs. GO_LIVE-C1 gilt real nur für den Nicht-Broker-Pfad.

Solide **[V]**: deterministische `client_order_id` + Duplicate-Adoption (`broker_adapter.py:597-712`), Intent-Store vor Submit, Retry/Rate-Limit; Telegram-Alerting echt und deutsch (Feuerung am 14.07. im Log belegt); Reconcile-Halt-Schwellen ($100/10bps) funktional getrennt vom CLI-„FAILED" (cash_tol=$1) — aber die Log-Semantik („Reconciliation FAILED" + „reconcile=OK" im selben Log) ist verwirrend.

API **[V]**: bindet `0.0.0.0:8000` (`scripts/run_api.py:18`); Read-Endpunkte unauthentifiziert, Command-Endpunkte key-gated, Kill-Switch-Deactivate token-gated (403); Path-Traversal-Guards vorhanden; FastAPI 0.139/starlette 1.3.1 in requirements.txt. Solange nur lokal betrieben, akzeptabel **[E]** — der 0.0.0.0-Bind bleibt die bekannte E-050-Flanke.

DMS **[V]**: Task läuft laut Scheduler, bewusst shadow (26h) — aber `output/ops/dms_audit.jsonl` existiert nicht, obwohl die Policy Audit-Einträge beschreibt. Verdacht: er evaluiert faktisch nichts **[E]**. Als Sicherheitsnetz derzeit Papier.

`ops_adopt_external_positions.py` (neu): sauberes dry-run→apply-Muster, Cold-Start-Guard **[V]**; Schwächen: Adoption-Fills fehlen im trade_journal, Ledger-Pfad hartkodiert.

### 2.6 Accounting / Steuer

- FIFO-Lots + EUR/ECB in `accounting/tax_lots.py` (SQLite, Over-Close-Guard `:350-373`); `compliance/tax_report.py:26` mit 26,375 %-Konstante als einfache Schätzung. **[V]**
- Die **volle** deutsche Steuer-Engine (Verlusttopf, Sparerpauschbetrag) — Grundlage des autoritativen Mandats-Kernbefunds — liegt **nur in `research/mandat/verdict_engine.py`, untracked, und hat 0 Tests** (Grep `def test_` in research/ = 0). **[V]** Das wichtigste Einzelergebnis des Projekts hängt an ungetestetem, unversioniertem Code. Die dokumentierten Selbstkorrekturen (E-051/E-052) zeigen, dass er real Fehler hatte, die erst spät gefunden wurden.
- Ledger enthält Float-Staub (qty 7e-15), kein Epsilon-Cleanup. **[V]**
- Accounting-fail-closed-Tests (E-035-Guards) sind echte Verhaltenstests. **[V]**

### 2.7 QA-Schicht

**Zustand: reiche Analyse-Bibliothek, unterbrochener Enforcement-Faden.**

- QA-BLOCK im Orchestrator ist nur `logger.error`, kein Abbruch (`pipeline/orchestrator.py:1001-1005`). Der einzige harte Enforcement-Punkt (`pre_trade_checks.py:827-844`) wird von beiden echten Aufrufern mit `qa_status=None` gefüttert (`api/routers/paper_trading.py:143`, `pipeline/trading_cycle_shared.py:1635` — selbst nachgeprüft). **Die QA-BLOCK-Semantik existiert im laufenden Betrieb nicht.** **[V]**
- `check_leakage` — als „mandatory gate" dokumentiert (`qa_gates.py:516-522`) — hat **keinen einzigen Caller** und ist fail-open (feature_df None → OK). **[V]**
- Reproducibility-Certificate: nur test-referenziert; `verify_certificate` meldet PASS, wenn Artefakte in beiden Läufen fehlen („NOT_FOUND" == „NOT_FOUND", `certify/generator.py:46,246-250`) — tautologisch. **[V]**
- Die 7 Performance-Gates selbst sind ordentlich gebaut (worst-case-wins), aber „nicht berechenbar" degradiert immer nur zu WARNING. **[V]**

### 2.8 CI / Workflows / Dependencies

- 21 Workflows; blocking auf Push/PR: backend-ci (inkl. **mypy blocking**, aber nur auf `data features signals execution portfolio` gescoped — qa/pipeline/accounting/risk/paper/api ungeprüft), ci.yml, evidence-pack-ci, accounting-ci (broker_snapshot), ops-evidence-ci, secrets-scan; release-gate full nur auf push→main. **[V]**
- **pip-audit-Waiver vorbildlich:** 14 Ignores, jedes mit Rationale + Review-Datum 2026-09-01; starlette-Waiver nach echtem Fix entfernt. **[V]**
- Release-gate deklariert seinen walk-forward selbst ehrlich als „SYNTHETIC smoke, CANNOT certify" — aber die E3/E4-Grace-Deadline „through 2026-07-01" ist **abgelaufen**, Re-Evaluation steht aus (`release-gate-ci.yml:75-96,89`). **[V]**
- **requirements.lock massiv stale** (fastapi 0.122, starlette 0.50, …) und **das Dockerfile installiert aus dem Lock** → ein heutiger Docker-Build shippt genau die 5 starlette-CVEs, die Commit 409ddc58 auf CI-Ebene gefixt hat. **[V]**
- Secret-Scanning ist real, nicht Deko: gitleaks über volle History mit exit-code 1, detect-secrets-Baseline-Diff, pre-commit gespiegelt; `.gitleaks.toml` pinnt die 3 historischen Leak-Commits mit Incident-Referenz. **[V]**
- renovate.json durchdacht, aber keinerlei Renovate-Commits in der History — vermutlich nie aktiviert. **[V]**
- `version.manifest.json` beschreibt ein Repo, das es seit ~11 Monaten nicht mehr gibt (`src/etl/*`, `src/ui/app.py`). **[V]**
- Root-Hygiene: ~20 Junk-Dateien (Logs, tmp-MDs, JSON-Dumps, ältester Nov 2025), top-level `qa/` = 16 Bootstrap-JSON-Artefakte, die namentlich mit der QA-Schicht kollidieren. **[V]** **[G]** (kosmetisch, aber Rule 50 wird auf das eigene Repo nicht angewandt).

### 2.9 Governance (CLAUDE.md / Rules / Hooks / Deny)

**Zustand: real schützend — mit zwei seit Monaten bekannten, unmitigiert dokumentierten Löchern.**

- Das Schutzsystem ist **kein Theater**: Der PreToolUse-Guard hat während dieses Audits zwei harmlose Lesebefehle geblockt (fail-closed, über- statt unterblockend; Ursache: `protected_paths_guard.py:97` splittet an `|` auch innerhalb von Quotes). Deny-Regeln decken exakt die 6 Pfade. Review-Chain: 124 Marker vs. nur 16 Skips mit überwiegend substanziellen Begründungen; Destructive-Override nie benutzt. **[V]**
- **PowerShell-Lücke: dokumentiert, diagnostiziert, nie geschlossen.** `protected_paths_guard.py:19-20` benennt sie; `.claude/settings.json:26-36` matcht nur „Bash"; `.claude/.hook_diag.jsonl` beweist, dass das PowerShell-Tool benutzt und diagnostiziert wurde. Ein `Set-Content src/assembled_core/risk/x.py` via PowerShell umgeht alles. **[V]**
- **Die §20.8-Follow-ups aus der eigenen Bypass-Disclosure sind 2 Monate später nicht umgesetzt:** kein Hook-Heartbeat (der bereits einmal eingetretene silent-fail-open-Modus bleibt unentdeckbar), `.claude/hooks/` selbst nicht in PROTECTED_PREFIXES (`path_classifier.py:21-26`). **[V]**
- Doku-Widersprüche **[V]**: GO_LIVE_CHECKLIST sagt gleichzeitig „A2 OFFEN" (Body, Z. 31) und „16 von 16 ERFÜLLT" (Summary); AGENTS.md widerspricht sich selbst (Z. 24 vs. 105-110) und transportiert das von Rule 40 widerlegte „19 Collection-Failures"-Faktum; **ROADMAP_STATE.md existiert nicht**, wird aber von Rule 10/95 als Pflicht-Doku-Ort genannt; `docs/ARCHITECTURE_BACKEND.md` trägt den Header „2025-01-15"; `.cursorrules` beschreibt das Projekt von vor einem Jahr („keine Live-Trading-Anbindung implementieren").
- Das Anti-Pattern-Register ist gelebt (53 Einträge, aktiv, SessionStart-Hook lädt Top-10 real). **[V]**

Ehrliche Kosten-Nutzen-Einschätzung **[E]**: ~60 % des Apparats schützt real (deny+Guard+Review-Chain+E-Register), ~40 % ist Selbstberuhigung — die *Regeln* werden gepflegt, die *Statusdokumente* verrotten, und Dokumentation eines Risikos (PowerShell, Heartbeat) ist zum Ersatz für dessen Beseitigung geworden.

### 2.10 Doku

Asymmetrische Pflege **[V]**: Research-Doku (Ledger, Registry, FINAL_REPORT, Überprüfung, Diagnostik) exzellent; Statusdoku (PROJEKT_STATUS „zwei Zeitalter in einer Datei", KNOWN_ISSUES endet faktisch am 2026-06-05, OPERATING.md vor dem Relaunch geschrieben) verrottet. `docs/Diagnostik.md` ist ein ehrlicher Snapshot, aber nirgends steht, welche Blocker seither geschlossen wurden — G5 (Drawdown-Key-Mismatch: `policy.yaml:71-74` legt soft/hard/kill flach ab, Code liest `levels` mit Hardcode-Fallback `trading_cycle_shared.py:1188,1206`) ist **heute noch real**, V1/G7/V4 sind gefixt. **[V]**

---

## 3. Tests & Ergebnisse — Re-Review

### 3.1 Test-Suite: Die Kopfzahl lügt

- **8884 Tests collected, 0 Errors, exit 0** (heute lokal). Aber: **[V]**
  - **347 `importorskip`-Aufrufe auf First-Party-Module**, davon **99 Targets, die nicht mehr existieren** (git-belegt gelöscht in `65cb2bb5`, `b433189c`) → **77 Testdateien mit 1068 Testfunktionen strukturell skip-tot**. Laufzeit-verifiziert: `test_meta_labeling.py` → 1 passed, 10 skipped. Darunter sensibel klingende tote Suiten: `test_pit_guard_universe.py`, `test_candidate_gate_reconciliation.py`, `test_evt_tail_var.py`.
  - `test_session_2026_05_07_new_items.py`: **11.330 Zeilen, 1009 „Tests"**, überwiegend Keyword-Grep-Prüfungen im Quelltext (PASS wenn „merger" ≥3× vorkommt, Z. 6020-6034) — ~11 % der Suite sind Pseudotests.
  - Tautologien: `assert True` / `assert len(x) >= 0` in `test_integration_run_daily.py:325-330`, `test_run_daily_smoke.py:114`, `test_strategies_multifactor_regime_overlay.py:588`.
  - Marker `requires_scipy/sklearn/fastapi` und `external` deklariert, aber **0 Verwendungen** — tote Konfiguration.
  - **Keine Regressionstests für E-044…E-053** (Grep = 0 Treffer); E-049 ist unlabeled faktisch abgedeckt, E-053 nur für das alte Schwester-Script, nicht für den neuen EODHD-Writer.
  - **Steuer-Engine: 0 Tests** (siehe 2.6).
- Gegenbild **[V]**: Die handverlesenen Guards der sensiblen Pfade sind **echte Verhaltenstests** — Drawdown-Stop (5/5 passed heute, inkl. „Breach + Write-Fehler blockt trotzdem"), Kill-Switch (10 Dateien inkl. corrupt-fail-closed end-to-end), Reconcile-Schwellen, Accounting-E-035-Guards, Property-based PIT-FSM.

**Netto-Urteil:** Die real beweiskräftige Suite ist deutlich kleiner als 8884 — grob geschätzt um ein Viertel geschrumpft, wenn man Skip-Leichen und Grep-Tests abzieht **[E]**. „0 Collection-Errors" wurde durch die importorskip-Zweckentfremdung erkauft: Modul-Löschungen machen Tests unsichtbar statt rot. Das ist der größte einzelne Selbstbetrugs-Mechanismus im Repo.

### 3.2 Research-Ergebnisse: Was hält, was nicht

**Hält [V]:**
- Das Closure-Dokument (`docs/PROJEKT_ABSCHLUSS_2026_05.md` — 9 Strategien, nicht 10; Registry/ERGEBNIS.md zitieren „10", Zähl-Drift): Verdicts konservativ belastbar, weil alle bekannten Biases (Survivorship, Same-Bar-Fills, unterzeichneter SPY) die Strategien **begünstigten** — negative Verdicts überleben das. Der Juni-OOS-Rerun mit DSR+Leakage-Checks bestätigte REJECTED.
- **Fable H1 (Insider-BUYS) FAIL ist zwingender als das Memory suggeriert:** EW der 69 Survivor-Namen ohne jedes Signal = Sharpe 1.133 vs. Insider-Basket 1.134 (`research/fable_exploration/ERGEBNIS.md:120-130`) — das in-sample-Ergebnis war zu ~100 % Survivorship-Gift. Die deploybare Variante wurde getestet (Liquiditätsdrittel @20bps → 0.563 < SPY). Verdict korrekt.
- Mandats-Registry-Disziplin ist real (Pre-Registration, PBO über die Selektionsmenge, E-051/E-052-Korrekturen offen dokumentiert). H-081/082 (echte CBOE-Historie, 38J) schließen die Stillhalter-Tür auf **Index-Ebene** überzeugend.
- Keine Fehlinterpretation der TR-adjusted-Preise gefunden: Fable common-mode, Mandat mit explizitem Dividenden-Steuer-Modell (`verdict_engine.py:160-171`).

**Hält NICHT bzw. überschärft [V]:**
1. **Der Kernsatz „KEINE Strategie schlägt den ETF" hängt an einem Randfall in einer bekannten Modellierungslücke.** `research/registry.md:253-258` dokumentiert: Vorabpauschale weggelassen → ETF-Endwert **3–6 % zu hoch**; „Präzisierung lohnt erst, wenn ein Kandidat in dieser Kante landet." H-032 low-div landete nach der E-051-Korrektur bei **−1,3 %** (1.589.963 vs. 1.610.149, `ledger.md:929-943`) — mitten in der Kante. Die versprochene Präzisierung wurde nie nachgeholt. Ehrlich wäre: „low-div ≈ ETF innerhalb der Modellunsicherheit; alles andere klar darunter." (Der Gesamtbefund kippt vermutlich trotzdem nicht — low-div trägt eigene unmodellierte Kosten **[E]** — aber als Punktaussage ist das Verdict überschärft.)
2. **E-051-betroffene FAILs nie re-validiert:** Der Frozenset-Bug betraf laut Ledger H-029/031/032/047/035/036 mit ±10 %-Swings; deterministisch neu gerechnet wurden nur H-032 und H-024. H-029 scheiterte **nur** an DSR (4/5 Kriterien bestanden) — genau die Art Grenzfall, die durch denselben Bug kippen könnte. „Alle Verdicts deterministisch verifiziert" darf so nicht behauptet werden.
3. **Memory/Doku-Drift Covered-Call:** MEMORY.md führt Covered-Call noch als „ERSTES OFFENES Kandidatenfeld"; H-081 hat das auf Index-Ebene geschlossen. Umgekehrt ist die FINAL_REPORT-Formulierung „keine offene Tür" eine **Extrapolation** von Index auf Einzeltitel — vertretbar, aber als Extrapolation kennzeichnungspflichtig.
4. **Vorzeichenfehler in `docs/Überprüfung.md:210`:** „SPY ohne Dividenden" wird als konservativ (gegen die Strategie) gelabelt — es ist das Gegenteil: die Latte war zu niedrig, die Strategien verloren trotzdem. Der Fehler **stärkt** die negativen Verdicts, aber das Dokument trägt ein falsches Vorzeichen.
5. **Zwei Closure-Zeilen zu hart:** multifactor_long_short wurde als long-only-Fehlkonfiguration getestet („kein fairer Test", Überprüfung `:125-127`), etf_pairs_meanrev mit Ersatzpaaren („informational only") — korrekt wäre „nicht valide getestet", nicht „KEIN EDGE".
6. **KNOWN_ISSUES 0.1 „Survivorship — BEHOBEN"** ist als Überschrift falsch; die Fable-Exploration bewies +0.35 Sharpe Survivorship-Gift im selben Datenbestand. Der Fließtext ist ehrlich, die Überschrift nicht.
7. **Post-FINAL_REPORT-Befunde nicht integriert:** H-083 (Unified-Holdout Sharpe 1,02, ehrlich als pseudo-OOS gerahmt) und H-084/085 (Odd-Lot ~200–600 €/J — das erste reale, kapazitätsbeschränkte Alpha des Projekts) stehen nur im Ledger; FINAL_REPORT sagt N=1964, Ledger ist bei N=1971.

**Richtungsbilanz:** Beide echten Angriffsflächen (Vorabpauschale-Kante, E-051-Re-Runs) zeigen in Richtung „möglicherweise zu **pessimistisch**" — keine einzige in Richtung „zu optimistisch". Das ist das Beste, was man über ein negatives Forschungsergebnis sagen kann.

---

## 4. Was noch zu tun ist (priorisiert)

### KRITISCH (Sicherheit/Integrität, vor weiterem Pilot-Betrieb)

| # | Punkt | Beleg |
|---|-------|-------|
| K1 | **Broker-Zustand klären vor Mo 20.07. 21:30**: offene/gefüllte Orders AAL/BIIB/MRNA/TDG/V prüfen, ggf. adoptieren | §2.5 |
| K2 | Market-Hours-Gate vor Order-Submit + In-Run-Cancel von Timeout-Orders | broker_execution.py:328-334 |
| K3 | Stale-Cleanup-Bug: `cancel_all_orders()` → gezieltes Cancel der stale_ids | run_live_paper.py:488 |
| K4 | Teilfill-Buchung (`filled_qty` bei timed_out/partially_filled) | broker_execution.py:328-357 |
| K5 | Reconcile-Block-Gate armieren (oder Entscheid dokumentieren) | app.yaml:51-52 |
| K6 | PIT-Cutoff in `refresh_daily_cache_from_panel.py` + Invariant-Test „keine Rows ≥ today" | E-053-Restbestand |
| K7 | PowerShell-Matcher für den Hook-Guard + Hook-Heartbeat (§20.8-Follow-ups einlösen) | settings.json:26-36 |
| K8 | requirements.lock regenerieren (Docker shippt sonst starlette 0.50 mit 5 CVEs) | Dockerfile |
| K9 | Skip-Leichen-Triage: 77 Dateien/99 Missing-Module; danach importorskip-Verbot für First-Party + CI-Skip-Gate | §3.1 |

### WICHTIG (Ehrlichkeit/Methodik, nächste 2–4 Wochen)

| # | Punkt | Beleg |
|---|-------|-------|
| W1 | DSR-SE-Formel fixen (rohe Kurtosis) + Impact-Check auf Gate-Entscheidungen | deflated_sharpe.py:130 |
| W2 | Vorabpauschale in ETF-Pfad einbauen, H-032/H-024 neu vermessen; Kernsatz ggf. umformulieren | registry.md:253-258 |
| W3 | E-051-Re-Runs der nie re-validierten FAILs (mind. H-029, dann H-031/047/035/036) | ledger.md:915-922 |
| W4 | QA-Gate-Wiring: qa_status in den Zyklus einspeisen ODER Gate ehrlich als „nicht aktiv" deklarieren | paper_trading.py:143, trading_cycle_shared.py:1635 |
| W5 | `_compute_options_factors` as_of-PIT fixen (offen seit 2026-05-28) | multifactor_v2.py:987 |
| W6 | Steuer-Engine: aus research/ nach src/ heben ODER als research-only deklarieren; Mindesttests (FIFO/Verlusttopf/Pauschbetrag) | verdict_engine.py |
| W7 | Halt-Gate um Positions-Mismatches erweitern; „FAILED"-vs-Halt-Logging entwirren; Preflight bei pending intents blocken | run_live_paper.py:66-87, 508-516 |
| W8 | Order-Lifecycle-Log in broker_execution verdrahten (C1 ehrlich machen) | §2.5 |
| W9 | Doku-Wahrheitssweep: GO_LIVE-A2-Body, AGENTS.md-Zahlen, KNOWN_ISSUES auf Juli-Stand (E-051, G5 OPEN, Survivorship-Überschrift), ROADMAP_STATE-Referenz, Diagnostik-Closure-Status, MEMORY-Covered-Call-Eintrag | §2.9/2.10, §3.2 |
| W10 | G5-Drawdown-Key-Mismatch fixen (policy.yaml-Struktur vs. Code-Erwartung `levels`) | trading_cycle_shared.py:1188,1206 |
| W11 | CPCV-Fallback laut machen (WARNING + method-Feld); Turnover-Gate-Skip mind. WARNING+KPI | cpcv_validation.py:104, _tc_sizing.py:1065 |
| W12 | Externe Run-missed-Überwachung (Watchdog sieht ausgeschalteten Host nicht); DMS-Audit-Realität klären; DMS-Docstring korrigieren („blockt Orders, schließt nichts") | §2.5 |
| W13 | TR-Semantik in Loader-Docstrings + CORPORATE_ACTIONS.md korrigieren; Dividenden-Doppelzählungsfrage im Paper-Engine klären | §2.1 |
| W14 | Release-gate-Grace (abgelaufen 2026-07-01) re-evaluieren | release-gate-ci.yml:89 |

### NICE-TO-HAVE

- Leichen entfernen: `src/erweiterung/` (.pyc), `signals/pead_sue.py`-Duplikat, orphane Intel-Module, `experiments/20251203_*`, `assemble_eod_daily.py` guarden, `run_paper_live.py` deprecaten, Root-Junk (~20 Dateien), top-level `qa/`-JSONs, version.manifest.json.
- Tautologische Asserts fixen; `test_session_2026_05_07_new_items.py` aus der Default-Suite markern; tote pytest-Marker entfernen.
- Hashes in data/raw-Manifeste; adj_close-NaN-Bestand bereinigen; `_TIER_YAML_PATH` root-verankern; mypy-Scope erweitern (qa/pipeline/risk/paper); Renovate aktivieren oder entfernen; `.cursorrules` deprecaten; certify-NOT_FOUND-Tautologie; `check_leakage` verdrahten oder löschen; settings.local.json-Sediment; Guard-Separator quote-aware; target_qty-Konsolidierung; Kosten-Default konservativer; docstring-Fixes (purged_cv, costs.py, event_features, _tc_signals AUC-Kommentar); FINAL_REPORT um H-083/084/085 ergänzen; 9-vs-10-Zählung vereinheitlichen.

---

## 5. HANDLUNGSANWEISUNGEN (konkrete Reihenfolge)

> Reihenfolge nach: erst Betriebsrisiko stoppen, dann Wahrheitssysteme reparieren (Tests/Gates), dann Methodik, dann Hygiene. Sensible Zonen (execution/risk/pipeline/paper) brauchen expliziten Auftrag + Review-Kette — hier als solche markiert [SZ].

**Schritt 1 — Broker-Triage (SOFORT, vor Mo 20.07. 21:30, manuell/read-only zuerst).**
Was: Alpaca-Konto prüfen: offene Orders + Positionen AAL/BIIB/MRNA/TDG/V; falls gefüllt → `ops_adopt_external_positions.py` dry-run → `--apply` → ggf. ack. Danach entscheiden, ob der Montag-Run laufen darf.
Wo: Alpaca-Dashboard bzw. `scripts/ops_adopt_external_positions.py`.
Warum: 5 nie stornierte After-Hours-Market-Orders vom 14.07. bei blindem Ledger; sonst Reconcile-Halt bzw. unbemerkte ~47k$-Positionen.
Erledigt wenn: Broker-Ist == Ledger-Ist, Reconcile grün, dokumentierter Vermerk im Pilot-Manifest/Journal.

**Schritt 2 — Drei Execution-Fixes in einem Paket [SZ].**
Was: (a) Market-Hours-Gate vor Submit; (b) Timeout-Orders im selben Run canceln; (c) Stale-Cleanup auf gezieltes `cancel_order(stale_ids)` statt `cancel_all_orders()`.
Wo: `broker_execution.py` (~:328ff), `run_live_paper.py:481-492`, Broker-Adapter-API.
Warum: Genau diese drei Lücken haben zusammen das 14.07.-Problem erzeugt; sie sind klein, klar lokalisierbar, hochwirksam.
Erledigt wenn: Regressionstests (After-Hours-Submit wird geblockt; Timeout → Cancel-Aufruf; Cleanup cancelt nur stale_ids) grün + Review-Kette PASS.

**Schritt 3 — Reconcile-Gate armieren + Halt-Gate erweitern [SZ].**
Was: `app.yaml reconcile_block.enabled: true` (mind. `block_on: ["fail"]`); Halt-Schwelle zusätzlich auf Positions-Mismatches; Preflight blockt bei pending intents.
Wo: `configs/app.yaml:51-52`, `run_live_paper.py:66-87, 508-516`, `ops/_paper_runner_gates.py`.
Warum: Der Code für die zweite Verteidigungslinie existiert und ist fail-closed — er ist nur nicht eingeschaltet.
Erledigt wenn: Ein synthetischer Mismatch-Test zeigt: Zyklus handelt NICHT; Config-Diff dokumentiert.

**Schritt 4 — Teilfill-Buchung [SZ].**
Was: `filled_qty > 0` bei timed_out/partially_filled ins Ledger übernehmen.
Wo: `broker_execution.py:328-357`.
Warum: Systematischer, stiller Ledger-Drift — die Sorte Fehler, die Wochen später als „unerklärlicher" Reconcile-Halt wiederkommt.
Erledigt wenn: Unit-Test „Order timeout mit filled_qty=n → Ledger bucht n" grün.

**Schritt 5 — Test-Suite-Wahrheit wiederherstellen.**
Was: (a) Die 77 skip-toten Dateien triagieren: Test löschen (Modul bewusst archiviert) oder reaktivieren; (b) Conftest-/Lint-Guard: `importorskip("src.assembled_core.*")` verboten; (c) CI-Skip-Gate: „Skips wegen src.-Modulen = 0" blockend; (d) `test_session_2026_05_07_new_items.py` mit eigenem Marker aus der Default-Zählung nehmen.
Wo: tests/ (77 Dateien, Liste beim Test-Audit), `conftest.py`, `pyproject.toml`, backend-ci.
Warum: Solange die Suite Löschungen unsichtbar macht statt rot, ist jede „N Tests grün"-Aussage des Projekts wertlos — das ist der Selbstbetrugs-Hebel Nr. 1.
Erledigt wenn: Collected-Zahl ehrlich (erwartbar deutlich unter 8884), 0 First-Party-importorskips, CI-Gate aktiv.

**Schritt 6 — Governance-Löcher schließen.**
Was: PowerShell-Matcher in `.claude/settings.json` PreToolUse; Hook-Heartbeat-Log; `.claude/hooks/` in PROTECTED_PREFIXES; Guard-Separator quote-aware.
Wo: `.claude/settings.json:26-36`, `.claude/hooks/protected_paths_guard.py:97`, `path_classifier.py:21-26`.
Warum: Die zwei größten bekannten Löcher sind seit ~2 Monaten dokumentiert statt geschlossen; Dokumentation ersetzt keine Mitigation.
Erledigt wenn: PowerShell-Write in Schutzzone wird geblockt (manueller Probeversuch), Heartbeat-Einträge erscheinen, Audit-Notiz in review_chain_disclosure.

**Schritt 7 — PIT-Restbestände.**
Was: (a) Cutoff in `refresh_daily_cache_from_panel.py` + Invariant-Test auf daily.parquet; (b) `_compute_options_factors(as_of=…)` [SZ-nah]; (c) TR-Semantik in Loader-Docstrings + `CORPORATE_ACTIONS.md:50` korrigieren; (d) Dividenden-Doppelzählungsfrage im unified_paper_engine klären [SZ].
Warum: Das sind die letzten bekannten Look-Ahead-/Semantik-Fallen im aktiven Pfad.
Erledigt wenn: E-053-Regressionstest für beide Writer grün; options-Faktoren nehmen as_of; Doku widerspruchsfrei; Doppelzählungsfrage mit Ja/Nein + Beleg beantwortet.

**Schritt 8 — QA-Gate-Entscheidung [SZ].**
Was: Entweder QA-Summary real in den Trading-Zyklus einspeisen (Orchestrator→pre_trade), oder das Gate offiziell als „nicht aktiv" deklarieren (Doku + GO_LIVE-Checklist korrigieren). Dazu: `check_leakage` verdrahten oder löschen, certify-Tautologie fixen.
Warum: Ein Gate, das immer None sieht, ist gefährlicher als kein Gate — es erzeugt falsches Vertrauen.
Erledigt wenn: Ein synthetischer QA-BLOCK verhindert nachweislich eine Order — oder die Doku behauptet es nirgends mehr.

**Schritt 9 — Methodik-Nachschärfung Research.**
Was: (a) DSR-Formel fixen + einmaliger Re-Run der Gate-relevanten Reports; (b) Vorabpauschale in den ETF-Pfad, H-032/H-024 neu; (c) E-051-Re-Runs H-029/031/047/035/036; (d) Formulierungs-Fixes (low-div „≈ ETF", „nicht valide getestet"-Zeilen, Überprüfung.md:210-Vorzeichen, H-081-Scope, 9-vs-10).
Warum: Schließt die letzten zwei Angriffsflächen am Kernbefund — danach ist „nichts schlägt den ETF" wasserdicht oder ehrlich relativiert.
Erledigt wenn: Ledger-Einträge mit neuen Zahlen; FINAL_REPORT-Addendum; Verdicts ggf. umformuliert.

**Schritt 10 — Dependency-/Doku-/Repo-Hygiene (Sammelpaket).**
Was: requirements.lock regenerieren (+ CI-Check lock vs. txt); Doku-Wahrheitssweep (W9); Steuer-Engine-Entscheid (W6); release-gate-Grace; Nice-to-have-Liste abarbeiten, soweit billig.
Erledigt wenn: Docker-Build enthält starlette 1.3.1; kein Statusdokument widerspricht mehr einem anderen; KNOWN_ISSUES trägt Juli-Stand.

---

## 6. FAZIT

**Wo das Projekt steht:** Es hat seine Forschungsfrage beantwortet — ehrlicher und gründlicher als fast jedes Amateur- und viele Profi-Projekte (N≈1971 registrierte Backtests, dokumentierte Selbstkorrekturen, negatives Ergebnis akzeptiert). Die Antwort lautet: **Auf erreichbaren Daten und nach deutscher Steuer existiert kein deploybarer Aktien-Edge; der passive ETF gewinnt.** Dieses Ergebnis hält meinem Re-Review stand — mit zwei kleinen, beide Richtung „zu pessimistisch" zeigenden Restflanken (Vorabpauschale-Kante, E-051-Re-Runs), die man für ~einen Arbeitstag schließen kann.

**Was realistisch drin ist (Edge-Frage):** Nüchtern — nichts Skalierbares. Die einzigen realen Funde sind kapazitätsbeschränkte Kuriositäten (Odd-Lot ~200–600 €/J) und die pre-registrierte, daten-gated Insider-Retest-Option auf delisting-inklusiven Daten. Die DSR-Latte bei N≈2000 macht weiteres Mining auf demselben Datenbestand statistisch sinnlos; das hat das Projekt selbst korrekt erkannt. Wer hier weitersucht, sucht gegen die eigene Evidenz. **[E]**

**Der eigentliche Befund dieses Audits** ist nicht die Edge-Frage, sondern die Asymmetrie: Das Projekt hat Weltklasse-Ehrlichkeitsdisziplin in der Research-Schicht und gleichzeitig eine Betriebsschicht, in der mehrere Schutzmechanismen nur auf dem Papier existieren (QA-Gate unverdrahtet, DMS flattet nichts, Lifecycle-Log im Broker-Pfad tot, ~1000 skip-tote Tests, Reconcile-Gate gebaut aber ausgeschaltet, PowerShell-Loch dokumentiert statt geschlossen). Nichts davon ist böswillig — es ist das Sediment schnellen Vorankommens. Aber es ist exakt die Sorte „falsche Sicherheit", die CLAUDE.md verbietet, und sie konzentriert sich dort, wo echtes (Paper-)Geld bewegt wird.

**Wie ich von hier aus weiterarbeiten würde:**
1. **Pilot ehrlich machen oder einstellen.** Wenn der Paper-Pilot weiterläuft, dann als das, was er ist: ein Betriebs-Härtungstest für trend_baseline (die OOS klar unter SPY liegt — der Pilot beweist Infrastruktur, nicht Rendite). Dafür Schritte 1–4 zwingend. Wenn das Ziel nur noch „passiver Kern + Overlay" ist (Mandats-Endspezifikation 65-70/25/5-10), ist ein schlanker Rebalancing-Executor mit Reconcile die ehrlichere Architektur als der volle Zyklus-Apparat. Diese Entscheidung ist die wichtigste offene Projektentscheidung — wichtiger als jeder Einzelfix. **[G]**
2. **Die Wahrheitssysteme reparieren, bevor irgendein neues Feature entsteht** (Schritte 5–8). Ein Projekt, dessen Verfassung „technische Ehrlichkeit vor Tempo" heißt, kann sich keine Test-Suite leisten, die Löschungen unsichtbar macht.
3. **Research nur noch auf die zwei offenen, pre-registrierten Türen begrenzen** (Vorabpauschale-Präzisierung, ggf. Insider-Retest auf CRSP/Sharadar/Norgate) — und sonst zusperren. Das Projekt hat sich diese Disziplin selbst verordnet; sie gilt.

**Gesamturteil in einem Satz:** Ein forschungsseitig abgeschlossenes, methodisch beeindruckend ehrliches Projekt, dessen verbleibendes Risiko fast vollständig in der Lücke zwischen dokumentierter und tatsächlicher Betriebsabsicherung liegt — und diese Lücke ist mit ca. 2–3 Wochen fokussierter Arbeit schließbar.

---

## 7. NACHTRAG 2026-07-21 — Nachprüfung der fünf offenen Punkte

Die im Erstbericht als „dünn" markierten Stellen wurden nachgeprüft (Broker-Query read-only, voller Testlauf, Git-Historie, zwei weitere Tiefenaudits). Ergebnisse, inkl. **einer Korrektur des Erstberichts**:

### 7.1 Broker-Check: Prognose exakt eingetreten [V]

- Alle 5 Orders vom 14.07. wurden von Alpaca über Nacht gequeued (submitted 15.07. 08:00 UTC) und **am 15.07. ~13:30 UTC zur Markteröffnung gefüllt**: AAL 213 @15.62, BIIB 17 @191.21, MRNA 52 @67.90, TDG 3 @1203.77, V 10 @351.60 — **Notional 17.235 $** (Erstbericht schätzte ~47k$ — zu hoch).
- **Der prognostizierte Reconcile-Halt trat am 20.07. 19:31 UTC exakt ein**: `output/ops/halt_ack_required.json` — cash_diff $17.237,33 (1989 bps), „Missing in ledger: AAL, BIIB, MRNA, TDG, V". Die zwei Folge-Runs (19:41, 19:56) wurden vom Preflight-Gate korrekt geblockt — **die Halt-Mechanik funktioniert wie designed**.
- Pikantes Detail: Das Halt-JSON meldet `mismatches_count: 0`, obwohl 5 Symbole fehlen — bestätigt live den Befund „Halt-Gate zählt nur cash_diff" (§2.4/§2.5).
- Float-Dust-Befund live bestätigt: Ledger enthält CVX/KO/WMT mit 7e-15 und LLY mit 1e-9; der Broker hält das LLY-1e-9 sogar als reale Dust-Position.
- **Recovery vorbereitet:** `ops_adopt_external_positions.py` Dry-Run gelaufen — Adoption der 5 Positionen zu Broker-avg_entry ergäbe Rest-Cash-Diff **+1,83 $** (weit unter Schwelle). `--apply` + `ack_halt` nicht ausgeführt (Operator-Entscheid).

### 7.2 Dividenden-Doppelzählung: entkräftet — das reale Problem ist das Gegenteil [V]

- Der Pilot-Pfad (`run_live_paper.py` → `ops/paper_runner` + `ops/paper_ledger`) nutzt `unified_paper_engine` **nicht** (einziger Nutzer: `paper/paper_track.py`, Legacy). Dessen CA-Dividenden-Gutschrift ist zudem doppelt opt-in (`enable_corporate_actions` + expliziter CA-Datei-Pfad, `unified_paper_engine.py:1836-1851`). **Keine Doppelzählung im Live-Pfad.**
- Die TR-Adjustierung verzerrt die *aktuelle* MTM-Equity nicht (letzte Bar: adjusted = raw); sie betrifft historische Kurven/Backtests.
- **Real ist die umgekehrte Lücke** (Accounting-Tiefenaudit §7.5): Der Paper-Ledger **bucht Dividenden nie** (kein Dividenden-Code in ops/), der Broker erhält echte Cash-Ausschüttungen (TLT zahlt monatlich, grob 25–30 $/Monat **[E]**) → schleichender ledger<broker-Cash-Drift, vom $100-Gate still absorbiert, bis er irgendwann als „unerklärlicher" Reconcile-Beitrag auftaucht.

### 7.3 Adversarial-Nachprüfung der Einzelbefunde

- **CWD-Falle ENTKRÄFTET [V]:** Beide Scheduler-Tasks (PaperPilot, DMS) haben `WorkingDirectory=F:\Python_Projekt\Aktiengerüst` gesetzt (schtasks-XML). Die relativen Pfade in `ack_halt.py`/`ops_watchdog.py` funktionieren damit heute — bleiben aber fragil gegenüber Task-Neuanlage **[E]**.
- **Reconcile-Gate `enabled: false` war BEWUSST [V]:** Commit `54cc9026` „feat(ops): default-OFF next-cycle reconcile-blocking gate (Item 3 / B-acct-3)". Kein Vergessen — die Empfehlung, es jetzt zu armieren (K5), bleibt, aber die Formulierung „bewusst oder vergessen?" ist beantwortet.
- **Grep-Pseudotest-Charakterisierung bestätigt [V]:** `test_session_2026_05_07_new_items.py` enthält 1009 Testfunktionen und 699 Datei-Lese-Operationen (`read_text`/`open(`) — die Charakterisierung „überwiegend Quelltext-Grep-Tests" hält quantitativ.
- target_qty-Zählung (~50 Stellen/17 Dateien): aus dem Erstaudit übernommen, nicht erneut gezählt.

### 7.4 Voller Testlauf — KORREKTUR des Erstberichts [V]

**Ergebnis: 8614 passed, 0 failed, 0 errors, ~270–317 skipped (exit 0).** Zwei Konsequenzen:

1. **Positiv, im Erstbericht so nicht belegt:** Die Suite ist lokal **vollständig grün** (lokal ≠ CI-Bestätigung).
2. **Korrektur:** Die Erstbericht-Aussage „~1000 Tests strukturell skip-tot" (§3.1) war **überschätzt**. Real: **~278 src-Modul-Skips** (~3 % der Suite; z. B. `test_short_engine` 16×, `test_pit_guard_universe`, `ml.evt_models`, `portfolio.multi_period`). Die statische Zählung „1068 Testfunktionen in 77 Dateien" überzeichnete, weil in vielen Dateien nur ein Teil der Tests am fehlenden Import hängt. **Der strukturelle Befund bleibt** (importorskip auf First-Party-Module macht Modul-Löschungen unsichtbar statt rot — K9 gilt unverändert), aber die Größenordnung des toten Bestands ist ~280, nicht ~1000. Die Netto-Aussage „real beweiskräftige Suite um ein Viertel kleiner" reduziert sich entsprechend: skip-tot ~3 %, dazu ~11 % Grep-Pseudotests.
3. Nebenbefund: **matplotlib im .venv kaputt** (circular import, 8 Skips) — Umgebungsproblem, kein Repo-Problem.

### 7.5 Ausgelassene Ecken (zwei Tiefenaudits)

**`workflows/` (untracked):** 17 Claude-Workflow-Orchestrierungs-Prompts vom 2026-06-04 (Deferred-Sweep) — hochwertig, aber einmalig verwendet; Ergebnisse längst committed. Empfehlung: nach `docs/archive/` einfrieren oder löschen; der Top-Level-Name kollidiert gedanklich mit `.github/workflows/`. **[V]**

**`research/mandat/` ist KOMPLETT untracked — inkl. `FINAL_REPORT.md` und der Steuer-Engine.** Ein Plattencrash löscht das autoritative Abschluss-Dossier von N≈1964 Runs. Strukturbefund Steuer-Engine: `TaxedPortfolio` (FIFO+Verlusttopf+Pauschbetrag, ~80 Zeilen in `h011_kandidat_a.py:114-195`) ist solide — aber die geteilte Engine **importiert aus dem Hypothesen-Skript** (invertierte Abhängigkeit, `verdict_engine.py:24-30`), Steuersätze sind Modul-Globals mit Laufzeit-Monkeypatching als empfohlener API (`verdict_engine.py:33`), 0 Tests. Lift nach src/ mit Golden-Tests gegen `results/*.json`: ~1–2 Tage. **[V]**

**Accounting-Tiefe (Schutzzone, nur gelesen):**
- `tax_lots.py`: FIFO-Kern und §23-EUR-Umrechnung (Kauf- und Verkaufstag je eigener Kurs) **im Prinzip korrekt** — aber: **Exit-Fees fehlen komplett** in `match_fifo` (P&L steuerlich systematisch überhöht); **ECB-Fallback hart 0.93** bei jedem API-Fehler UND Wochenende (Docstring behauptet fälschlich Vortageskurs; Query setzt start=end=trade_date); read-then-write über zwei Connections ohne Transaktions-Span (latentes Race). **[V]**
- **`TaxLotStore` hat null Produktions-Caller** — kein Paper-/Live-Fill füttert ihn. Die Anlage-KAP-Fähigkeit ist spezifiziert, nicht integriert („Schaufenster-Code"). `position_engine.py` (die „kanonische P&L-Quelle") rechnet **Average-Cost, nicht FIFO** — als Steuerbasis wäre sie falsch. **[V]**
- Paper-Ledger: Schreiben atomar (filelock + Backups + replace), aber **kein Epsilon-Cleanup** (Dust-Quelle: float-Qty, pop nur bei exakt 0, `paper_ledger.py` Grep 1e-9/dust = 0 Treffer) und **keine unabhängige Equity-Invariante** (equity wird aus cash+positions konstruiert, nie gegen die Event-Historie rückgerechnet). **[V]**
- Evidence-Pack attestiert nur Datei-Integrität; bei Checksum-Berechnungsfehler wird die Datei **ohne Checksum** gepackt (kleines Attestierungs-Loch). **[V]**

**Doku-Nachsampling [V]:** Drei verschiedene Kill-Switch-State-Pfade über Spec (`output/state/`), Runbook (`output/runs/_kill_switch/`) und Code (`output/ops/`, `kill_switch.py:48`); `PILOT_OPERATIONS_PLAYBOOK.md` nennt noch den −8%-Stop (seit 02.07. −10% Soft-Halt) — im Ernstfall würde ein Operator falsch angeleitet. `docs/OPERATOR_RUNBOOK.md` dagegen aktuell und code-verifiziert (bestes Ops-Dokument).

### 7.6 Konsequenzen für die Priorisierung

- **K1 (Broker-Triage): Diagnose erledigt.** Offen ist nur noch die Operator-Entscheidung: Adoption `--apply` + `ack_halt` (Rest-Diff 1,83 $) — dann läuft der Pilot beim nächsten Scheduler-Lauf weiter; ohne Ack bleibt er sicher geblockt.
- **Neu in WICHTIG:** (W15) Dividenden-Buchung in den Paper-Pfad (7.2 — strukturelle Drift-Quelle im laufenden Pilot); (W16) `research/mandat/` committen (Datenverlustrisiko FINAL_REPORT + Steuer-Engine); (W17) Exit-Fees + ECB-Vortages-Fallback in `tax_lots.py`, falls die Steuer-Schiene je real genutzt wird — sonst explizit als „nicht integriert" deklarieren; (W18) Epsilon-Cleanup in `apply_fills_to_ledger` (Dust live am Broker nachgewiesen); (W19) Kill-Switch-Pfad-Angaben in Spec+Runbook auf `output/ops/` vereinheitlichen + Playbook −8%→−10%.
- **Abgeschwächt:** K9 (Skip-Leichen) bleibt richtig, aber Größenordnung ~280 statt ~1000; W7-Teil „CWD-Falle" entfällt als akutes Risiko (WorkingDirectory gesetzt), bleibt als Härtungs-Nice-to-have.
