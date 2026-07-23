# Diagnostik — Vollständiger Grundcheck Assembled-Trading-AI

**Erstellt:** 2026-06-02
**Branch / HEAD:** `main` @ `329a3240`
**Methode:** Read-only Multi-Agent-Diagnose (Ultracode / Dynamic Workflow). 22 modulweise Audit-Agents → unabhängige Verifikation jedes BLOCKER/MAJOR-Funds durch einen zweiten Agent (re-open am zitierten `path:line`, default-skeptisch) → Trading-Edge-Analyse + Web-Recherche (SPIVA/Methodik) → Completeness-Critic. 48 Agents, ~3,95 Mio. Tokens.
**Scope:** Gesamtes System (`src/assembled_core/` ~646 Module/172k LOC, `scripts/` 268/93k LOC, `tests/` 718/191k LOC, 21 CI-Workflows, Repo-Hygiene), nicht nur der Trading-Teil.

## Ehrlichkeits-Disclaimer (zwingend lesen)

- **Read-only.** Keine Datei verändert außer dieser Diagnostik. Keine Pipeline-/Backtest-/Paper-Läufe gestartet (E-035-Schutz).
- **CI nicht ausgeführt.** „7/7 green" ist aus statischem YAML **nicht** zertifizierbar; alle CI-Aussagen sind Konfigurations-Analysen, kein Live-Run-Status. Tests wurden via `pytest --collect-only` geprüft (8192 Tests, **0 Collection-Errors**), aber **nicht** ausgeführt.
- **Jeder MAJOR/BLOCKER wurde unabhängig gegengeprüft.** Wo die Verifikation einen Fund herabgestuft oder widerlegt hat, steht das **explizit** im Befund. Es werden **keine** widerlegten Funde als reale Bugs verkauft.
- **Belege.** Jeder Befund trägt `Datei:Zeile`. Wo die Verifikation eine Zeilen-/Pfad-Abweichung fand, ist das vermerkt.
- **Eigen-Korroboration des Hauptagenten:** `optimizers.py:61 from scipy.optimize import minimize` (top-level, bestätigt), scipy inzwischen in 3 Windows-Workflows, `oms.py:171` Broker-Routes Platzhalter, `insider_ingest.py` Dummy-Daten `allow_sample`-gated, committete Junk-Files via `git ls-files` (s. Modul *repo-hygiene*).

## Legende

**Schweregrad:** `BLOCKER` (blockiert sicheren Betrieb) · `MAJOR` (ernst, vor Live/Trust zu fixen) · `MINOR` (sollte gefixt werden) · `INFO` (Hinweis/Verbesserung).
**Konfidenz:** `hoch` / `mittel` / `niedrig` (Sicherheit, dass der Befund real ist).
**Befundtyp:** **(a) ✔Bug** = verifizierter Bug · **(b) ⚠Verdacht** = suspected (kontext-/datenabhängig) · **(c) ✎Verbesserung** = improvement.

## Gesamtüberblick — Modul-Gesundheit

| Modul | Health | Kernrisiko (Kurz) |
|---|---|---|
| execution | mostly_ok | Idempotenz/Duplicate-Recovery faktisch funktionslos; 1 Fail-open im Symbol-Kill-Switch-Wrapper |
| risk | mostly_ok | Mathe sauber; State-Machine fixed-tmp Concurrent-Write; CB-Wiring teil-fail-open |
| pipeline | mostly_ok | 2× VIX-`as_of`-Bypass (Backtest-Look-Ahead), `groupby().last()` ohne Sort-Vertrag, `book_fills` E-035 |
| portfolio | mostly_ok | scipy-Top-Level-Import bricht Paket-Import (A2); `target_qty` = Notional statt Shares |
| accounting | mostly_ok | Reconciliation **nicht** fail-closed; FIFO-Over-Close still verschluckt; Paper-vs-Paper „pass" |
| strategies | mostly_ok | `compute_signals` ohne `as_of` (Backtest-Look-Ahead bei Caller-Fehler); Faktor-Neutralisierung via `except→0.0` |
| signals | mostly_ok | viel Research-/Dead-Code; 2 gemeldete MAJOR → MINOR korrigiert (s. u.) |
| data+dataquality | mostly_ok | Macro-Loader ohne Release-Lag (Look-Ahead, latent); silent degradation |
| features | mostly_ok | Earnings/News T+0-Merge (Look-Ahead, latent da Faktoren=0) |
| qa | **fragile** | PBO ist KEIN BLP-PBO; zwei divergente DSR; `check_max_drawdown` crasht bei None; None-Sharpe fail-open |
| events | **fragile** | news_alpha EOD erzeugt **0 Signale** (topic/topic_id-Mismatch); Reversal-Exit nie implementiert |
| intel | mostly_ok | 3 gemeldete MAJOR → MINOR (nur aus Scripts erreichbar, nicht src) |
| api+ports+adapters | mostly_ok | Unauth Path-Traversal `/ledger` & `/live-curve`; Fail-open/closed-Inkonsistenz |
| ops | **fragile** | Alerts werden **nie an Menschen geliefert**; Reconcile-FAIL eskaliert nie; DMS nicht im Scheduler |
| paper | mostly_ok | Unlocked Read-Modify-Write (Lost-Update); Cost-Model fällt still auf Zero-Cost |
| ml | mostly_ok | Registry nicht thread-safe + non-atomic; HMM-Fehler still verschluckt; viele NotImplemented-Stubs |
| core-misc | mostly_ok | Canary-Hash PYTHONHASHSEED-instabil; Policy-Schema-Validierung still verschluckt |
| scripts | mostly_ok | fragiler E-035-Guard; geteilter Cache ohne Date-Guard; CI-Checks `exit 0` (keine Enforcement) |
| tests | mostly_ok | Anti-Patterns: Windows-`grep`-Count, 3 widersprüchliche Bounds, tautologisches `assert True` |
| ci+deps | **fragile** | 3-Wege-Dependency-Split; stale `requirements.lock`; backend-ci Ubuntu-only; Strategie-Gate synthetisch |
| repo-hygiene | **fragile** | 18 `autonome_weiterarbeit/`-Docs + weitere Artefakte trotz gitignore committet |
| cross-cutting | **fragile** | Policy-Load fail-open (`None`/`{}`); 2× E-030 ffill/bfill-Look-Ahead; 201 broad-except / 109 Dateien |

**Wiederkehrendes Signaturmuster:** *Silent Fail-Open am operativen Rand* — Exception → harmlos aussehender Default → kein WARNING. Das ist das dominante Systemrisiko (s. Querschnittsthemen).

---

# Modul-Befunde

## execution — `src/assembled_core/execution/`
Health: **mostly_ok**. Sicherheits-Primitive (`kill_switch.py` hash-chained + fsync + filelock + Token-Gate, `pre_trade_checks`, `fat_finger_guard`, `position_sync`) sind ungewöhnlich gut und fail-closed. Schwachstellen konzentriert im Idempotenz-/Duplicate-Pfad.

- **[MAJOR · hoch · (a)✔Bug]** Duplicate-Order-Recovery ist ein No-op — beide Zweige `raise`. `broker_adapter.py:644-656` (+`:738-749`): `except … if is_duplicate_error(): logger.warning(); raise` gefolgt von unbedingtem `raise`. → Das Idempotenz-Design (deterministische `client_order_id` → Broker lehnt Dup ab → vorhandene Order übernehmen) **erholt sich nie**; Crash-Retry propagiert nur die Exception. *Verifiziert: bestätigt.* **Fix:** bei Dup vorhandene Order per `client_order_id` holen & normalisiert zurückgeben.
- **[MAJOR · mittel · (b)⚠Verdacht]** `is_duplicate_error` zu strikt. `idempotency.py:67-70`: `return "duplicate" in msg and "client_order_id" in msg`. Reale Alpaca-Meldungen („already exists", „potential wash trade") enthalten oft nicht beide Tokens → Dup wird als generischer Fehler fehlklassifiziert; Worst Case zweite Live-Order. *Verifiziert: bestätigt.* **Fix:** auf reales Broker-Signal (HTTP 422 + „already exists") matchen + Test mit echtem String.
- **[MINOR · hoch · (a)✔Bug]** Fail-open im Unified-Symbol-Kill-Switch. `symbol_kill_switch.py:204-225`: globaler `is_kill_switch_engaged()` in `try/except Exception` nur `logger.debug`, fällt durch → bei I/O-/State-Fehler passieren alle Orders. *Verifiziert: bestätigt (Impact-Ceiling grenzt an MAJOR).* **Fix:** fail-closed (`orders.iloc[0:0]`), ERROR-Log.
- **[MINOR · mittel · (b)⚠Verdacht]** Float-`==` im Short-Flip. `unified_paper_engine.py:1641` `if current_qty == 0:` — Residual-Short `-1e-12` umgeht Zweig; Sibling-SELL nutzt `abs()<=1e-8` (`:1630`). **Fix:** Toleranzvergleich.
- **[INFO · mittel · (c)✎]** ADV-Cap `pre_trade_checks.py:1032` `.tail(adv_window)` ohne internen `as_of`-Slice (PIT-Vertrag liegt beim Caller). **[INFO · hoch · (c)✎]** Fraktionale Order-Qty ohne explizite Whole-Share-Policy (`order_generation.py:343-364`).

**Dead/Placeholder:** `unified_paper_engine.py:529/2593` Template-Method-Stub (Basis liefert leer); `order_management.py:474-486` `OrderStatusStream` synchroner Stub (nicht im EOD-Live-Pfad). Keine NotImplementedError in Safety-Files.
**Test-Lücken:** kein Test gegen reale Alpaca-Dup-Strings / Recovery-vs-reraise; kein Fail-closed-Test für `filter_orders_with_kill_switches` (raise von `is_kill_switch_engaged`).

## risk — `src/assembled_core/risk/`
Health: **mostly_ok**. VaR/CVaR/Cornish-Fisher-Vorzeichen konsistent korrekt (`var_methods.py:201,245,265`), GJR-GARCH & DCC-GARCH echte Implementierungen. Schwächen im operativen Safety-Wiring.

- **[MAJOR · mittel · (b)⚠Verdacht]** State-Machine Fixed-Tmp Concurrent-Write. `state_machine.py:142-152`: Non-Lock-Zweig schreibt fixen `*.tmp`, `finally` unlinkt unbedingt; lock-basiertes `atomic_write_json_with_retry` ist opt-in (`use_lock=False` default). Zwei Writer (EOD + Paper) → `os.replace` halbgeschriebener JSON; korrupter State → loader resetet still auf WATCH (`:90-94`). *Verifiziert: bestätigt (`.bak`-Backup mildert, eliminiert nicht).* **Fix:** unique tmp-Suffix bzw. Lock default.
- **[MINOR · hoch · (a)✔Bug] — gemeldet MAJOR, verifiziert MINOR:** Circuit-Breaker-Wiring fail-open. `trading_cycle_shared.py:1055` `except Exception: return None` (=„kein Breach"). *Verifikation korrigierte auf MINOR:* der einzige Consumer `_tc_risk.py:240-252` umschließt den Aufruf mit **fail-CLOSED** Handler („R2-1: unknown CB-state muss blocken"). Residualer Bug: der innere Swallow defeatet den äußeren Fail-closed nur für die In-Loop-Fehlerklasse. **Fix:** im inneren `except` konservativ blocken/raisen, ERROR-Log.
- **[MINOR · hoch · (a)✔Bug]** `CircuitBreaker.is_tripped` Cooldown nutzt `datetime.now()` gegen historische Observation-Timestamps (`circuit_breaker.py:99-106`); im verdrahteten Pfad (`:1040` liest nur `observe()`) tot, latente Falle für Backtest-Reuse. **[MINOR · mittel · (b)⚠]** Reflation-Regime liest eigenen in-progress-Output (`regime_analysis.py:181-185`, rückwärts, Analyse-only).
- **[INFO · hoch · (c)✎]** `transaction_costs.py:199-202` Cost-Komponenten (`commission/spread/slippage`) hartkodiert `0.0` Platzhalter — TCA/Attribution liest Nullen (`cost_total` korrekt).

**Dead/Placeholder:** `VolCircuitBreaker` definiert+getestet, aber nicht verdrahtet (nur `CircuitBreaker`); `is_tripped`-Cooldown im Live-Pfad tot.
**Test-Lücken:** kein Test für Fail-open-`except`-Pfad (`:1055`); kein Concurrent-Write-Test für `save_risk_state`; kein historischer-Timestamp-Cooldown-Test.

## pipeline — `src/assembled_core/pipeline/`
Health: **mostly_ok**. Zentrale PIT-Filterung (`_filter_prices_for_as_of`) und `_record_degraded_step` (fail-soft + `meta['degraded_steps']`) sind durchdacht. Reale Inkonsistenzen:

- **[MAJOR · hoch · (b)⚠Verdacht]** VIX-Read umgeht `as_of`-PIT (Raw-Panel-Tail). `_tc_signals.py:355` & `_tc_sizing.py:1196`: `float(ctx.prices["VIX"].iloc[-1])` auf der **rohen** `ctx.prices` (nie zu `prices_filtered` reassigned), nicht der as_of-Scheibe. In Backtest/Replay = Future-VIX, gleiche Klasse wie der Mai-2026 market_stress-Fix. *Verifiziert: bestätigt; in Live/EOD ist Tail=as_of, Bug beißt nur in Backtest/Replay.* **Fix:** `ctx.prices[ts<=as_of]` slicen (wie CB `:967-980`).
- **[MAJOR · mittel · (b)⚠Verdacht]** `groupby().last()` „latest-bar" ohne Sort-Vertrag. `trading_cycle_shared.py:460/470`: `groupby("symbol").last()` läuft **vor** dem `sort_values(["timestamp",…])` (`:465/:473`). `GroupBy.last()` = letzte Zeile in DataFrame-Reihenfolge, nicht max-Timestamp → falscher „latest"-Bar wenn Input nicht pro-Symbol sortiert. *Verifiziert: bestätigt; Korrektheit hängt an undokumentiertem Upstream-Sort.* **Fix:** vorher `sort_values(["symbol","timestamp"])` / idxmax.
- **[MAJOR · hoch · (a)✔Bug]** `book_fills` schreibt Operations-Artefakte ohne Backtest-Guard (E-035). `_tc_execution.py:491` (`trade_journal.jsonl`), `:501-517` (`order_lifecycle.jsonl`), `:529-538` (`heartbeat.json`) → `ctx.output_dir`, **kein** `mode=="backtest"`-Guard (während `_tc_risk.py:80` genau diesen Guard hat). *Verifiziert: bestätigt.* **Fix:** auf `mode != "backtest"` gaten.
- **[MINOR · hoch · (c)✎]** `_tc_sizing.py` God-File 2604 LOC / 38× `except Exception`. **[MINOR · hoch · (a)✔]** Backtest-Kill-Switch-Restore inaktiv (`kill_switch_persist=True` default nie False gesetzt; `backtest.py` ohne kill_switch-Referenz).
- **[INFO]** Wide-Format-`"VIX"`-Branch feuert im Long-Format-Pfad vermutlich nie (mildert die VIX-MAJORs latent — **vor** Priorisierung VIX-Lieferformat verifizieren). **[INFO]** `enable_risk_controls=False`-Backtest-Pfad ist bewusster Full-Bypass (nicht live-äquivalent lesen).

**Dead/Placeholder:** `trading_cycle_v2.py:539` Migrations-Hinweis; `orchestrator.py:1431` `TODO: wire factor-decay post-signal` (no-op stop-gap).
**Test-Lücken:** keine PIT-Regression für VIX-Overlays; kein Test für `.last()`-Ordering bei geshuffletem Panel; kein E-035-Guard-Test für `book_fills` im Backtest.

## portfolio — `src/assembled_core/portfolio/`
Health: **mostly_ok**. Kernmathematik (Leverage-Cap-Ordering, PSD-Checks) gehärtet.

- **[MAJOR · hoch · (a)✔Bug]** Ungeguardeter Top-Level-`scipy`-Import re-exportiert im Paket-`__init__`. `optimizers.py:61 from scipy.optimize import minimize` (kein try/except), gezogen via `__init__.py:37`. → `import src.assembled_core.portfolio` bricht hart ohne scipy (genau A2; `requirements.txt:49-51`: scipy>=1.16 braucht Py>=3.11 → Py3.10-Lanes betroffen). Sibling `black_litterman.py:34-40` guardet identisch. *Verifiziert: bestätigt; vom Hauptagenten eigen-korroboriert.* **Fix:** `try/except ImportError` + `SCIPY_AVAILABLE`-Flag / lazy re-export.
- **[MAJOR · hoch · (a)✔Bug]** `target_qty`-Platzhalter liefert Cash-Wert statt Shares. `position_sizing.py:104-106`: `target_qty = target_weight * total_capital` (keine Division durch Preis). Downstream, der die Spalte als Shares liest, sized ~Preis× zu groß; mit Default `total_capital=1.0` ~0 (dokumentierter B1-Failure-Mode). *Verifiziert: bestätigt.* **Fix:** Spalte `target_notional` nennen / durch Preis teilen.
- **[MINOR · hoch · (a)✔]** `dro_portfolio.py:52` scipy ungeguardet (nicht init-exportiert → kleinerer Radius). **[MINOR · hoch · (b)⚠]** `covariance.py:160-162` `estimate_covariance` schluckt alles, gibt leeren DF auf DEBUG (fail-open, E-025). **[MINOR · mittel · (b)⚠]** `hrp_sizing.py:104` `.tail(lookback+1)` ohne `as_of`-Param.
- **[INFO · hoch · (c)✎]** `quantum_portfolio.py` „research showcase stub" (nicht in `__init__`).

**Test-Lücken:** kein Import-ohne-scipy-Test (`test_portfolio_optimizers.py:17` `importorskip("scipy")` **skippt** das ganze Modul); kein `target_qty`-Semantik-Test.

## accounting — `src/assembled_core/accounting/`
Health: **mostly_ok**. Kernmathematik (Average-Cost-Engine, FIFO-Tax-Lots, atomare Writes) sorgfältig; gute Fail-closed-Guards (NaN-cash_delta raise, unknown-currency raise).

- **[MAJOR · hoch · (a)✔Bug]** Reconciliation **nicht** fail-closed. `ledger_integration.py:258-301`: `reconcile_ledger_vs_broker(…, fail_fast=False)`; bei `ok=False` nur `logger.warning`, Funktion läuft normal weiter; `evaluate_reconcile_slo` wird **nie** aufgerufen. → driftendes Book handelt weiter. *Verifiziert: bestätigt (MAJOR, nicht BLOCKER, da Ergebnis im Report sichtbar).* **Fix:** über `evaluate_reconcile_slo` routen, bei `fail` raisen/Kill-Switch.
- **[MAJOR · hoch · (a)✔Bug]** FIFO-Over-Close verschluckt `qty_remaining` still. `tax_lots.py:267-355` `close_lots()` iteriert nur `lots_closed`; `qty_remaining` (`match_fifo:149`) wird nie geprüft/geloggt → Schließen von mehr Shares als offene Lots = under-reported realized P&L (DE Anlage-KAP) ohne Audit-Trail. *Verifiziert: bestätigt.* **Fix:** WARNING/raise bei `qty_remaining>tol`.
- **[MAJOR · mittel · (b)⚠Verdacht]** No-Broker-Snapshot-Fallback reconciled Paper-vs-Paper und meldet „pass". `ledger_integration.py:247-255`: fehlender Snapshot → `broker_positions_df = positions_df`, `broker_cash = cash_balance` → `ok=True` trivial; `broker_view_source="paper_view"` wird getaggt, aber das konsumierte `reconciliation_ok` reflektiert es nicht. *Verifiziert: bestätigt.* **Fix:** `reconciliation_ok=None/"unverified"` bei `paper_view`.
- **[MINOR · hoch · (a)✔]** `reconcile_daily_pnl` Price-Feed-Gap unsichtbar (`reconciliation.py:471-475`: `skipped_symbols` zählt nicht in `ok`). **[MINOR · mittel · (a)✔]** `p_start == 0` Float-Sentinel (`:471`). **[MINOR · mittel · (a)✔]** FX-Fallback `DEFAULT_FX_RATES` still angewandt ohne As-of (`currency.py:14-24`, `position_engine.py:391`).
- **[INFO]** Ledger/Snapshot-Writes tmp+replace aber **nicht** fsync'd (`ledger_store.py:108-122`; Audit-Log `reconciliation.py:42-44` fsync't sehr wohl). **[INFO]** Orphan-Temp bei Write-Fehler (`broker_snapshot_store.py:133-143`). **[INFO]** unbekannter `side` → `qty=0.0` statt reject (`ledger.py:202-208`).

**Test-Lücken:** Over-Close-`qty_remaining`, echte Broker-Mismatch-Block, Feed-Gap-`ok=False`, Crash-Injection um atomic write.

## strategies — `src/assembled_core/strategies/`
Health: **mostly_ok**. Zeitreihen-Strategien (dual_momentum, low_max_lottery, etf_pairs, pairs, vol_target) explizit kausal (ffill-only, bfill-Vermeidung dokumentiert).

- **[MAJOR · hoch · (b)⚠Verdacht]** `mfv2.compute_signals` ohne `as_of`-Param, ankert PIT an Panel-Max. `multifactor_v2.py:1356` Signatur ohne `as_of`; `:1390 _bar_as_of = latest["timestamp"].max()`. → Alle Altdata-Faktoren ankern an letztem Panel-Bar; Backtest mit vollem Panel leakt Future-Altdata. *Verifiziert: bestätigt; per-Faktor-Guards greifen nur wenn Caller `as_of` explizit übergibt — `compute_signals` tut das nicht.* **Fix:** expliziter `as_of`-Param + Docstring-Vertrag.
- **[MAJOR · mittel · (c)✎]** Pervasives `except Exception → return 0.0/empty` neutralisiert Faktoren still (E-025-Familie). `multifactor_v2.py:519,539,559,804,950,992,1188,1192,1242,1292,1296` — kaputter Faktor ununterscheidbar von „keine Daten", DEBUG-Level. Trägt zum dokumentierten Dead-Faktor-Problem (19/34=0) bei. *Verifiziert: bestätigt.* **Fix:** Catch verengen / erste Occurrence WARNING.
- **[MINOR · hoch · (b)⚠]** `trend_baseline.compute_signals` ohne `as_of`, `.tail(1)` (`:79-84`) — naiver Backtest liest globalen letzten Bar (B1-Pitfall). **[MINOR · hoch · (a)✔]** `target_qty` hartkodiert `0.0` (`trend_baseline.py:132` u. a.) — Caller, der qty direkt liest, handelt nichts. **[MINOR · mittel · (b)⚠]** `multifactor_long_short` Regime-Build-Fehler degradiert still zu no-overlay (`:335-359`).
- **[INFO · hoch · (c)✎]** `sector_rotation_bias` dokumentierter Dead-Faktor (~0 Produktionsgewicht, `:578-584`).

**Dead/Placeholder:** mfv2-Faktoren sector_rotation_bias/insider_cluster/buyback_drift/pead_sue wired aber zero-fill in Backtest / ~0 Gewicht; `signal_decay_gate`/`ic_decay_weights` implementiert aber default `enabled=False`.
**Test-Lücken:** kein Test der `as_of`-Slicing-Precondition (höchster Wert); kaputter-Faktor-vs-absent nicht abgedeckt.

## signals — `src/assembled_core/signals/`
Health: **mostly_ok**. Mix aus PIT-disziplinierten verdrahteten Faktoren und großem Research-/Live-Fetch-Tail, der von **keiner** Live-Pipeline importiert wird.

- **[MINOR · — · (a)✔Bug] — gemeldet MAJOR, verifiziert WIDERLEGT-als-Mechanismus → MINOR:** options-IV-Faktor angeblich Live-Fetch-Look-Ahead. `multifactor_v2.py:1301-1340` `_compute_options_iv_factor` ruft `iv_skew(sym, None)`. *Verifikation:* `iv_skew` (`options_iv.py:137`) hat **keinen** yfinance/Live-Fetch-Code (reine Black-Scholes, 7 Pflicht-Args) → `iv_skew(sym, None)` wirft `TypeError`, der `except` zero-fillt. **Kein Look-Ahead — toter/kaputter Signatur-Branch.** **Fix:** Signatur reparieren oder Branch entfernen.
- **[MINOR · hoch · (a)✔] — gemeldet MAJOR, verifiziert MINOR:** sector-rotation stempelt `pd.Timestamp.now()` bei namenloser Row (`sector_rotation.py:164,166`). *Verifikation:* Downstream-Consumer nutzt nur `longs/shorts`, nicht `.date` → latente Timestamp-Hygiene, kein nachgewiesener PIT-Impact. **Fix:** expliziten `as_of` propagieren.
- **[MINOR · hoch · (a)✔]** Permanent-Null-Dimension in Composite. `composite_score.py:251-256` `chart_pattern_score → 0.0` als Dim 5; fixes 0.10-Gewicht in allen 4 Regimes → strukturell 10% totes Gewicht. *Verifiziert: bestätigt (Impact eher unterschätzt).*
- **[MINOR · hoch · (c)✎]** `cross_asset_carry.py:176-180` hartkodierte 1.2/0.8/1.0, Live-yfinance ohne `as_of`. **[MINOR · mittel · (c)✎]** breite `except→debug→0.0` über Faktor-Helpers (insider_cluster/buyback_drift/options_iv/etf_flows).
- **[INFO]** zwei `cross_asset_carry`-„Wahrheiten" (v1/v2), keine im Live-Pfad. **[INFO]** Macro-`ffill`-Reindexe sind PIT-safe (kein bfill) — explizit als Nicht-Bug verifiziert.

**Dead/Placeholder:** `recession_probability`, `lppls_crash`, `tail_risk_hedge`, `tail_risk_vvix`, `cross_asset_carry(_v2)`, `etf_flows` — keine Referenz aus {pipeline,strategies,paper,intel}; research-tier. `causal_ml.fit_causal_forest` graceful-degrade ohne sklearn.

## data+dataquality — `src/assembled_core/data/`, `dataquality/`
Health: **mostly_ok**. Preis-/Corporate-Action-/Universe-Kern stark gehärtet (fail-loud Schema, `allow_sample`-Guards, Feed-Status-Stamping, GPR Release-Lag 32d).

- **[MAJOR · hoch · (a)✔Bug]** Macro-Loader leakt Future-Daten — kein Publication-Release-Lag. `altdata_loader.py:222` filtert `timestamp <= as_of` auf rohen `macro.parquet`-Observation-Timestamps; Header behauptet „All functions are PIT-safe". CPI/Unemployment werden zum Monatsende „sichtbar", real aber Wochen später publiziert. Downstream real: `multifactor_v2.py:874 load_macro_indicators`. *Verifiziert: bestätigt; Live-Radius begrenzt (Macro-Faktoren aktuell in 19/34-Zeroed-Set).* **Fix:** `release_lag_days`-Shift wie GPR.
- **[MINOR · mittel · (b)⚠Verdacht] — gemeldet MAJOR, verifiziert MINOR:** Earnings-`event_date`-Fallback umgeht Disclosure-PIT (`altdata_loader.py:60`). *Verifikation:* Produktions-Parquet nutzt `disclosure_date`; Fallback feuert nur bei `event_date`-only-Feed → kontingent. **Fix:** konservativen Disclosure-Lag bei `event_date`-only.
- **[MINOR · hoch · (c)✎]** Modul-Level „silent degradation" widerspricht Datenrealismus-Regel (`altdata_loader.py:4-5`; fehlende Files nur DEBUG `:50,108,157,203`).
- **[INFO]** FRED-Feeder ohne Release-Lag-Spalte (`fred_source.py:135`). **[INFO]** Dummy-Generatoren wall-clock `now()` (`insider_ingest.py:89`, `shipping_routes_ingest.py:89`, `allow_sample`-gated). **[INFO]** `detect_unadjusted_splits` prüft nur Down-Side (`splits.py:23`, Reverse-Splits unentdeckt).

**Dead/Placeholder:** `insider_ingest`/`shipping_routes_ingest` „Phase-6-Skeleton" Dummy-Generatoren, jetzt `allow_sample=True`-gated (kein stiller Phantom-Daten-Pfad mehr). Insider-Daten 100% `unknown` → Faktor liest 0.

## features — `src/assembled_core/features/`
Health: **mostly_ok**. PIT-Infrastruktur (`event_features.py` braucht `as_of`, Forward-Returns hinter Flags) durchdacht. Schwäche: inkonsistente Release-Lag-Behandlung.

- **[MAJOR · mittel · (b)⚠Verdacht, latent]** Earnings-Surprise merged auf rohem Event-Timestamp ohne Release-Lag (T+0, E-002). `altdata_earnings_insider_factors.py:132-134` `disclosure_date=event_date`; `:282-290 merge_asof(on=timestamp_col, allow_exact_matches=True)`. → Event Tag T (nach Close/pre-Market T+1) fällt in Tag-T-Features. *Verifiziert: Code-Level wahr; Impact aktuell begrenzt (earnings_surprise_z = ZERO in mfv2-Full-Stack).*
- **[MAJOR · mittel · (b)⚠Verdacht, latent]** News/Macro-Sentiment default T+0 + Exact-Match-Merge. `altdata_news_macro_factors.py:134-135,307-321,340-353,661-673`. *Verifiziert: bestätigt; latent (news_sentiment ZERO in Produktion).* **Fix (beide):** auf `disclosure_date` mergen, `allow_exact_matches=False`, `apply_source_latency` routen — **bevor** diese Faktoren je >0 gewichtet werden.
- **[MINOR · hoch · (c)✎]** Inkonsistente Disclosure-Latency-Policy (insider T+2 vs earnings/news T+0). **[MINOR · hoch · (a)✔]** `compute_risk_on_off_indicator` expliziter Platzhalter (`market_breadth.py:301-310`). **[MINOR · mittel · (b)⚠]** `news_features._merge_feature` füllt fehlende Sentiment mit `0.0` (`:272`) = „keine News" ununterscheidbar von „neutral".
- **[INFO]** Forward-Return-Funktionen sind Look-Ahead **by design** aber korrekt gegated (Labels, keine Features).

## qa — `src/assembled_core/qa/`
Health: **fragile**. Kernmetriken + BLP-DSR/PSR/MinTRL-Formeln korrekt, aber zwei methodisch falsche/duplizierte Gates.

- **[MAJOR · hoch · (a)✔Bug]** PBO ist fehlbenannte Single-Partition-Rank-Fraction, **kein** BLP/CSCV-PBO. `metrics.py:1521-1544`: Docstring „Bailey/López de Prado 2014", Body `oos_rank/len(ranks_oos)` auf **einem** IS/OOS-Split (kein kombinatorisches Splitting, kein Logit, kein Anteil-negativer-Logits). → unterschätzt Overfitting; Wert nicht mit `<0.5/<0.3`-Schwellen vergleichbar. *Verifiziert: bestätigt.* **Fix:** `oos_rank_fraction` umbenennen oder echtes CSCV-PBO; **nicht** als BLP-PBO zur Deployment-Entscheidung zitieren.
- **[MAJOR · hoch · (a)✔Bug]** Zwei divergente, beide exportierte DSR. `deflated_sharpe.py` liefert Wahrscheinlichkeit `Φ(...)` (Euler-Mascheroni-Schwelle) vs `metrics.py:633 deflated_sharpe_ratio` liefert Z-Score (`sqrt(2logN)/sqrt(T)`); beide in `__init__.py`. → „DSR>0.95"-Gate semantisch mehrdeutig. *Verifiziert: bestätigt (verschiedene Namen, also keine Import-Kollision, aber Dual-Truth real).* **Fix:** eine autoritative DSR, andere deprecaten. *(Hinweis: das OOS-Gating nutzt `deflated_sharpe.py` = die korrekte Wahrscheinlichkeits-Variante — s. Trading-Verdict.)*
- **[MAJOR · hoch · (a)✔Bug]** `check_max_drawdown` ohne None-Guard. `qa_gates.py:153-155` `if max_dd < limit:` ohne `is None`-Check (alle Sibling-Gates haben ihn) → `TypeError`-Crash von `evaluate_all_gates`. **Fix:** `if max_dd is None: return WARNING`.
- **[MAJOR · mittel · (b)⚠]** None-Sharpe-Gate fail-OPEN (WARNING statt BLOCK). `qa_gates.py:86-96` — unberechenbarer Sharpe blockt Deployment nicht (risk-first sollte fail-closed). **Fix:** BLOCK / konfigurierbar default-closed.
- **[MINOR · hoch · (c)✎]** i.i.d.-Bootstrap überschätzt Signifikanz (`bootstrap_metrics.py:42,66,94` `rng.choice(replace=True)` zerstört Autokorrelation → zu enge CIs, zu optimistisches `sharpe_p_value` — falsche Richtung für ein Overfitting-Gate). **[MINOR · mittel · (b)⚠]** `_sharpe` gibt `0.0` bei Zero-Vol (`:11-13`). **[MINOR]** toter `freq="1d" # Dummy` (`:778,781`) latente Annualisierungs-Falle.
- **[INFO]** GO_LIVE-A1-„quality_gate FutureWarning" liegt **nicht** in `qa/` (grep leer) — Fix-Site ist Script/Pipeline.

**Dead/Placeholder:** `scenario_engine.py:345-352` „SHIP"-Substring-Heuristik-Placeholder; `factor_analysis.py:1904-1912` `n_tests=1 # Placeholder` (DSR un-deflated wenn Patch-Pfad übersprungen); `cpcv_validation.py:121` dummy-CV bei fehlendem sklearn.

## events — `src/assembled_core/events/` (news_alpha + crisis_alpha)
Health: **fragile**. crisis_alpha-State-Machine solide & fail-closed. news_alpha hat zwei reale Defekte, die **still** fehlschlagen (debug/no-op).

- **[MAJOR · hoch · (a)✔Bug]** news_alpha EOD-Pfad liest `topic`, Live-Trigger tragen `topic_id`. `signal_generator.py:83 item.get("topic","")` & `asset_router.py:161` vs kanonisch `trigger_scoring.py:196 "topic_id"`, `intel_context.py:93`. EOD-Wiring sourct dieselben Items (`_tc_sizing.py:1868-1873`). → jedes Item `raw_topic=""` → `get_route("")` None → **alle Signale via `logger.debug` geskippt**; news_alpha erzeugt **0 Einträge ohne Warnung** im EOD/Live-Pfad. Nur der separate Intraday-Runner (topic-keyed) funktioniert. *Verifiziert: bestätigt.* **Fix:** `item.get("topic") or item.get("topic_id")` / Keys am Wiring-Punkt normalisieren.
- **[MAJOR · hoch · (a)✔Bug]** Reversal-Exit (#4) dokumentiert + Parameter durchgereicht, **nie implementiert**. `exit_rules.py:7` Docstring; `check_exits` nimmt `new_trigger_items` (`:26`), Pipeline reicht durch (`pipeline.py:94`), Body (`:42-91`) referenziert es nie. → die beworbene Reverse-Alpha-Schutzlogik existiert nicht. *Verifiziert: bestätigt (aktuell durch MAJOR#1 ohnehin maskiert).* **Fix:** implementieren oder Param+Docstring entfernen.
- **[MINOR · hoch · (c)✎]** Direction-Conflict still „keep existing" (`signal_generator.py:236-245`, order-dependent). **[MINOR · mittel · (b)⚠]** `central_bank base_severity=1` erreicht Route-`min_severity=2` nur bei ≥5-Cluster (`trigger_scoring.py:102,179-180`).
- **[INFO]** news_alpha-Overlay entkommt globaler De-Risk-Kette (opt-in, `_tc_sizing.py:1339-1349`). **[INFO]** State-Korruption → still WATCH (`state_machine.py:128-138`, E-025; PAUSE-Auto-Clear ist der heikle Subfall).

**Dead/Placeholder:** `exit_rules.py:83-89` short-Branch tot (alle Inverse-ETFs `direction="long"`); news_alpha-EOD-Pfad zur Laufzeit faktisch tot (MAJOR#1).
**Test-Lücken:** kein Test mit `topic_id`-keyed Item (alle Tests nutzen `topic` → Bug für Suite unsichtbar); kein Reversal-Exit-Test.

## intel — `src/assembled_core/intel/`
Health: **mostly_ok**. Strukturell solide, gute Security-Hygiene (kein MNPI, parents[]-Bugs gefixt, GDELT-Merge gefixt). **Alle 3 gemeldeten MAJOR von der Verifikation auf MINOR herabgestuft** (Erreichbarkeit: nur aus Scripts, nicht src).

- **[MINOR · hoch · (b)⚠] — gemeldet MAJOR → MINOR:** `conviction_engine.py:141 embargo_minutes=0` defeatet PIT-Embargo. *Verifikation:* Call-Site backtest-gated (`trading_cycle_v2.py:397 allow_in_backtest=False`) → kein aktiver Look-Ahead im Backtest-Pfad; Code-Smell. **Fix:** `embargo_minutes=1`/dokumentieren.
- **[MINOR · hoch · (b)⚠] — gemeldet MAJOR → MINOR:** `market_confirmation.py:113,151` `yf.download` ohne `as_of`. *Verifikation:* **nicht** aus `src/` aufgerufen (nur `scripts/run_intel_cycle.py:632`) → latente Wiring-Falle, kein aktiver Bug. **Fix:** `as_of`-Param + historischer Cache.
- **[MINOR · mittel · (a)✔] — gemeldet MAJOR → MINOR (partiell):** `news_enricher.py` 13× `except Exception` (`:100,121,…,388`). *Verifikation:* `:121,128` loggen WARNING (beobachtbar); silent v. a. Klassifikations-/IC-Schritt (`:199,205,100`) → produziert Zero-Severity/Confidence-Events. Charakterisierung „alle 13 still" überzogen.
- **[MINOR · hoch · (a)✔]** `conviction_engine.py:89` `_iv_z == _iv_z` als NaN-Check (numpy-fragil). **[MINOR]** `news_archive.py:101` close()-Exception still; `rss_fetcher.py:248-262` Datum-Parse-Fallback → `datetime.now()` (PIT-Risiko in Replay); `pit_store.py:161` korrupte Manifest-Einträge still `continue`.
- **[INFO]** toter `_y_pred`-Assign (`conviction_engine.py:291`); `news_rag.py:63` `import anthropic` module-level trotz unwired.

**Dead/Placeholder:** `news_rag.py`, `polymarket_loader.py` (keine src-Call-Sites), `feedback_loops.py` (statische History → Stub-Computation).

## api+ports+adapters — `src/assembled_core/api/`, `ports/`, `adapters/`
Health: **mostly_ok**. Auth/Rate-Limit/Audit-Middleware + Kill-Switch-Auth fail-closed & gut (constant-time, fsync'd hash-chain, RLock um Paper-Engine). Schwächen in Path-Posture & Error-Leakage.

- **[MAJOR · hoch · (a)✔Bug]** Unauth GET Path-Traversal: beliebiger `ledger_path` auf `/ledger` & `/live-curve`. `ledger.py:33-55` & `performance.py:171-205`: `ledger_path: str = Query(...)` via `Path(...)` / `OUTPUT_DIR.parent / ledger_path` **ohne** Safe-Roots-Check (kontra `health.py:34-35,63`); kein `require_api_key`. *Verifiziert: bestätigt (Existenz-/JSON-Shape-Oracle, nicht beliebiger Content; GETs zudem unaudited da Middleware bei GET früh returned).* **Fix:** `_is_safe_output_dir` wiederverwenden.
- **[MAJOR · hoch · (a)✔Bug]** Fail-open/Fail-closed-Inkonsistenz bei identischem Loader-Fehler (E-025-adjazent). `performance.py:206-209` raised `HTTPException(500)` während `ledger.py:75-90` graceful `no_ledger` + `start_capital=-1.0`-Sentinel erkennt; `/live-curve` ruft `load_ledger_state(jpath)` **ohne** Sentinel → korrupte Ledger als valide leere/echte Kurve gerendert (Docstring verspricht sogar „never 404 or 500"). *Verifiziert: bestätigt.* **Fix:** `/live-curve` spiegelt `/ledger`.
- **[MINOR · hoch · (a)✔]** 500-Leakage interner Pfade/Exception-Text an unauth GETs (`oms.py:100,168`, `performance.py:86,162,208`, `monitoring.py`, `diagnostics.py`). **[MINOR · mittel · (a)✔]** OMS-Reads ohne den Paper-RLock (`oms.py:64,126 _engine.list_orders` ohne `_engine_lock`). **[MINOR · mittel · (b)⚠]** `/health?check_broker=true` unauth Outbound-Trigger (E-026; default-off, Rate-Limit default disabled).
- **[INFO]** Audit-Middleware Body-Hash verlässt sich auf undokumentiertes Starlette-Caching (>64KiB silent leer).

**Dead/Placeholder:** `oms.py:187-192` `/routes` nur `PAPER` (IBKR auskommentiert, KNOWN_ISSUES 6.6 — ehrlich); `ports/order_router.py:29-36` Skeleton-Protocol; `app.py:119-132` `/health/startup` immer `started:True`.

## ops — `src/assembled_core/ops/`
Health: **fragile**. Operativ breit, aber Safety-kritische Delivery-/Reconcile-Alert-Pfade haben reale Lücken, die in Produktion still fehlschlagen.

- **[MAJOR · hoch · (a)✔Bug]** Autonomer Alert-Worker liefert **nie** an einen menschlichen Kanal. `daily_scheduler.py:738-831` nutzt `alert_manager.AlertManager` (nur `logger` + `flush_to_json`); die echten Sinks `send_with_failover`/`post_discord`/`fire_alert` (alert_failover/alert_sinks/alerting) werden von **keinem** `src/`-Modul importiert (nur `scripts/`). → CRITICAL Kill-Switch-/Reconcile-/Stale-Model-Alerts landen in Datei+Konsole, **kein** Discord/E-Mail/Telegram. *Verifiziert: bestätigt.* **Fix:** `_alert_health_worker` nach `flush_to_json` über `send_with_failover` routen. *(Relativiert GO_LIVE E3.)*
- **[MAJOR · hoch · (a)✔Bug]** Reconcile-Invariant-FAIL triggert nie den CRITICAL-Alert (Scheduler-Pfad). `daily_scheduler.py:760-768` keyt auf `out_path.glob("*.error")`; `_reconcile_worker` schreibt Status in `reconcile_{date}.json` (`:316-319`)/`reconcile.py:121` — **nichts** in `src/` schreibt `*.error`. *Verifiziert: bestätigt (Nuance: `make_reconcile_fail_alert` wird in `paper_runner.py:887` aufgerufen, endet aber auch nur in `write_alerts_artifact` = JSON, keine Delivery → Eskalationslücke besteht end-to-end).* **Fix:** `reconcile_*.json` lesen, bei `status=="FAIL"` CRITICAL feuern.
- **[MAJOR · mittel · (b)⚠]** DMS-Daemon existiert, aber nicht im Task Scheduler verdrahtet. `dms_monitor_loop` nur aus `scripts/dms_daemon.py` aufgerufen; kein `ops/dms_daemon.py`. → Auto-Flatten-on-stale-Heartbeat läuft nur bei manuellem Start; eingefrorener Cycle wird nicht geflattet. *Verifiziert: Struktur bestätigt; „nicht im Scheduler" ist umgebungsabhängig (kein In-Repo-Manifest), via Memory 86468b0c gestützt.*
- **[MINOR · hoch · (a)✔]** `_news_fetch_worker` schluckt Per-Step-Fehler, Cycle läuft mit stale Artefakten weiter (`:113-128`). **[MINOR · hoch · (c)✎]** 5 Alert-Module + `AlertManager`-Namenskollision (alert_manager vs alerting) = Duplicate-Truth (Root-Cause von #1). **[MINOR · mittel · (b)⚠]** `_factor_curation_worker` nennt IC-t-Stat „DSR" (`:656-663`, advisory).
- **[MINOR · — · (a)✔] — gemeldet INFO → verifiziert MINOR/MAJOR-Grenze:** `order_lifecycle_log.py:32 DEFAULT…=Path("output/journal/order_lifecycle.jsonl")` schreibt in Live-`output/` bei jedem Caller inkl. Backtest (E-035, Obs 748). **Fix:** run-scoped `log_path` erzwingen.
- **[INFO]** `factor_decay_reporter` non-blocking no-op stop-gap (Memory 9467b0ae).

**Dead/Placeholder:** `alert_failover/alert_sinks/alerting` faktisch tot relativ zum autonomen Cycle (nur scripts); `scheduler.py` APScheduler-Wrapper registriert no-op-Lambdas (paralleler ungenutzter Scheduler); `mlflow_tracking.py` no-op ohne MLflow.

## paper — `src/assembled_core/paper/`
Health: **mostly_ok**. Single-Process-Persistenz sorgfältig (atomic tmp+replace, `.backup`, per-Date-Dedup, PIT-Guard auf sector-rotation). Schwächen in Concurrency & Cost-Defaulting.

- **[MAJOR · mittel · (b)⚠Verdacht]** Unlocked Read-Modify-Write auf Shared-Aggregaten → Lost-Update. `paper_track.py:1825-1845` (equity_curve), `:1859-1892` (trades_all), `:1908-1939` (positions_history): read→concat→`temp.replace`, **null** File-Locks. Zwei parallele Prozesse (EOD + Intraday, beide laut Memory aktiv) → späterer `.replace` überschreibt Zeile des früheren → still gedroppte Trades/Equity-Punkte. *Verifiziert: bestätigt (single-process dedup-safe).* **Fix:** File-Lock um read→concat→replace / Single-Writer.
- **[MAJOR · hoch · (b)⚠]** Cost-Model-Resolver fällt still auf Zero-Cost-Fills. `paper_runner.py:1136-1147 _resolve_cost_cfg` gibt `dict(app_cost)` (evtl. `{}`) ohne Log zurück; `ops/paper_ledger.py:231-239` → leeres cfg = `commission_bps=0, slippage_bps=0`. → fehlt/umbenannt `paper_pilot.cost_model` → Fills zu Exact-Close, **unrealistisch optimistisches Paper-P&L**, nur durch `policy.yaml:861-862`-Default maskiert (den der Resolver nicht erzwingt). *Verifiziert: bestätigt.* **Fix:** fail-closed bei fehlendem cost_model.
- **[MINOR · hoch · (c)✎]** Cost-Model-Duplicate-Truth: `policy.yaml:862 10.0` vs `paper_track/*.yaml 0.5` vs `*_live.yaml 5.0` (3 Werte, 20×-Spanne). **[MINOR · mittel · (b)⚠]** Tail-Read-Fallback ohne `as_of` wenn Scores keine Timestamp-Spalte (`intel_context.py:219-220`). **[MINOR · mittel · (b)⚠]** Non-atomic Truncate-Rewrite des historical_scores-Cache (`:480-485`).

**Test-Lücken:** kein Concurrency/Lost-Update-Test; kein Fail-closed-Test des Zero-Cost-Resolvers.

## ml — `src/assembled_core/ml/`
Health: **mostly_ok**. Solide Utilities (purged CV, BMA, HMM, copula) + ehrliche Research-Stubs.

- **[MAJOR · hoch · (a)✔Bug]** Module-Level-Registry-Cache nicht thread-safe. `model_registry.py:35-61`: `_registry_cache`/`_registry_mtime` Globals ohne Lock mutiert; `register_model():167` resetet nur `_registry_cache`, nicht `_registry_mtime` → stale mtime → alter Cache trotz neuer registry.json. *Verifiziert: bestätigt.*
- **[MAJOR · hoch · (a)✔Bug]** `_save_meta` schreibt registry.json non-atomar (`:261-264 write_text`); Crash mid-write → truncated → `except` (`:253`) gibt `[]` → `load_deployed` skippt Hash-Verify. *Verifiziert: bestätigt.* **Fix:** tmp+`os.replace`.
- **[MAJOR · hoch · (a)✔Bug]** `RegimeHMM.partial_update` behält bei Fitting-Exception still altes Modell (`regime_hmm.py:251-256`, `return self` unbedingt, kein Success-Flag). **[MAJOR · mittel · (a)✔]** `MultiFeatureRegimeHMM.predict_regime()` Fallback auf `predict_proba()` bei Exception, nur DEBUG (`:498-504`) → stille Regime-Substitution. *Beide verifiziert: bestätigt.*
- **[MINOR · hoch · (a)✔]** `register_model` resetet `_registry_mtime` nicht. **[MINOR · hoch · (c)✎]** `stability_filter` nutzt nicht-purged Folds (Train/Test-Overlap → upward-biased IC, `feature_selection.py:205-209`). **[MINOR]** `retraining_scheduler.py:93 .iloc[-20:]` ohne as_of; Scaler-Full-Fit (`regime_hmm.py:407-409`).
- **[INFO]** `verify_model_hash()` gibt `True` bei leerer Registry (fail-open, `:78-90`).

**Dead/Placeholder:** `gnn_signal.py`, `temporal_fusion_transformer.py`, `logic_tensor_network.py`, `differential_privacy.py:224-265` (DP-SGD), `retraining_scheduler.py:54-58` (Bandit no-op) — alle Stub/`NotImplementedError`.

## core-misc — config/compliance/certify/attribution/reports/application/domain/strategy/time/bootstrap/utils
Health: **mostly_ok**.

- **[MAJOR · hoch · (a)✔Bug]** Canary-Hash PYTHONHASHSEED-instabil. `config/feature_flags.py:49 hash(ticker) % 10 == 0` — Python-`hash()` von Strings per-Prozess randomisiert ohne fixen `PYTHONHASHSEED` → Canary-Bucket wechselt über Restarts, „stable 10%"-Vertrag gebrochen, A/B-Shadow ungültig. *Verifiziert: bestätigt.* **Fix:** `hashlib.md5(...)% 10`.
- **[MAJOR · hoch · (a)✔Bug]** Policy-Schema-Validierung still verschluckt. `config/policy_loader.py:57-69 except Exception: logger.debug("…skipped")`. → broken pydantic/Schema-Import → jeder Policy-Load umgeht Validierung inkl. Kill-Threshold-/Drawdown-Ordering-Checks, kein sichtbares Signal. *Verifiziert: bestätigt.* **Fix:** WARNING min., bei ImportError raisen.
- **[MINOR · — · (a)✔] — gemeldet MAJOR → MINOR:** `strategy/` (Singular) faktisch tot/Duplicate-Truth zu `strategies/` (Plural) + `config/models.py`. *Verifikation:* hat Test-Coverage, kein Live-Exec-Pfad. **Fix:** kanonischen Ort festlegen / als research-only markieren.
- **[MINOR]** `utils/market_calendar.py:143-153` O(N·days)-Busy-Loop; `policy_loader.py:90 except: pass` Conflict-Guard; `certify/generator.py:92,102,154` bare `pass`; `certify/mlflow_integration.py:80,159` bare `except: pass`; `attribution/storage.py:44-47` WAL `except: pass`; `feature_flags.py:45 getattr(...,"off")` Typo-still-off; `compliance/tax_report.py:1` Docstring „stubs" obwohl voll implementiert (irreführend).
- **[INFO]** `strategy/config.py` CompositeWeights divergieren von `configs/factor_weights_by_regime.json` (zwei Gewichtsquellen).

**Dead/Placeholder:** `domain/{accounting,operations,research,risk,trading}/__init__.py` leer (Hexagonal-Layer nur Verzeichnisstruktur); `application/use_cases/` Skeleton (nur `record_kill_switch_trip.py`).

## scripts — `scripts/`
Health: **mostly_ok**. Die 20+ OOS-Harnesses sind diszipliniert (1-Bar-Lag, `pos.shift(1)`, Warmup, DSR-Deflation, Honesty-Caveats inline).

- **[MAJOR · hoch · (b)⚠Verdacht]** Fragiler E-035-Guard. `_oos_wf_pipeline_realistic.py:83 os.environ.setdefault("ASSEMBLED_NO_CRISIS_OVERLAY","1")` — `setdefault` ist No-op wenn Var bereits auf `"0"` gesetzt → literale Pipeline schreibt time-traveled Records in `output/ops/crisis_alpha_state.json`. *Verifiziert: bestätigt; reale Exploitability niedrig (geo_score=0 → Overlay nie ACTIVE).* **Fix:** unbedingte Zuweisung + Startup-Assert.
- **[MAJOR · hoch · (a)✔Bug]** Geteilter Preis-Cache ohne Content/Date-Guard. `_oos_wf_mfv2.py:51`, `_oos_wf_mfv2_full.py:62`, `_oos_wf_mfv_long_short.py:52` → identischer `oos_alpaca_prices_cache.parquet`; Validity-Check (`:118-120`) prüft nur Symbol-Presence, nicht Date-Range/Fetch-Timestamp → OOS-Ergebnis eines Harness hängt still vom Vorlauf eines anderen ab. *Verifiziert: bestätigt.* **Fix:** Fetch-End-Date + Script-ID im Cache-Pfad.
- **[MINOR · — · (b)⚠] — gemeldet MAJOR → MINOR:** `adjustment="split"` in allen Alpaca-OOS-Harnesses (Dividenden-Auslassung). *Verifikation:* in jedem betroffenen Report **inline disclosed**, Bias-Richtung **konservativ** (Income-Assets unterschätzt). **Fix:** `adjustment="all"` / Header-Flag.
- **[MINOR · — · (b)⚠] — gemeldet MAJOR → MINOR (Mechanismus widerlegt):** `ffill()` auf Voll-History-Pivot vor Fold-Slicing. *Verifikation:* in den Stock-Universe-Harnesses wird ffill auf bereits **fold-gesliceten** `window_prices` angewandt → **intra-fold** Gap-Fill, **kein** Cross-Fold-Propagation; ETF-Universes benign.
- **[MINOR · hoch · (a)✔]** `batch_runner.py:657-658 except: pass` schluckt DD-Damper-Reset still. **[MINOR · hoch · (a)✔]** CI-Checks `ci/drift_check.py`, `walk_forward_check.py`, `retraining_check.py` fangen alles und `sys.exit(0)` → **können nie fehlschlagen** (null Enforcement-Wert). **[MINOR · mittel · (b)⚠]** `_oos_wf_mfv2.py:196-201` signal_fn skippt Rebalance-Dates still bei Exception (flacher Fold statt ehrlichem Fail).
- **[INFO]** `profile_jobs.py` `run_factor_ml_job` Placeholder (ML-Step skipped); `commands/paper.py:403,453 --run-news-pipeline` no-op/placeholder.

**Dead/Placeholder:** `dev/tmp_script.py`, `dev/tmp_check.py`, `dev/tmp_peek_ec.py` (Scratch); `ci/{drift,walk_forward,retraining}_check.py` (Stub, exit 0).

## tests — `tests/`
Health: **mostly_ok**. **8192 Tests sammeln fehlerfrei (0 Collection-Errors)** — die „19-Collection-Error"-Ära ist erledigt. Safety-Pfade (Kill-Switch-Auth, Pre-Trade-fail-closed, PIT-/bfill-Regression) gut abgedeckt. Aber Test-Anti-Patterns:

- **[MAJOR · hoch · (a)✔Bug]** Windows-inkompatibler `subprocess(["grep",…])`. `test_session_2026_05_07_new_items.py:8423` — auf Windows-CI nicht in PATH → `count=0` → `assert count<500` trivial grün, maskiert jedes Wachstum von bare-except. *Verifiziert: bestätigt.* **Fix:** portables `sys.executable -c`/`rglob`.
- **[MAJOR · hoch · (a)✔Bug]** Drei widersprüchliche `except Exception`-Bounds in derselben Datei: `:1703 <1200`, `:8426 <500`, `:11357 <=250` — messen nicht mal dasselbe (`:1703` zählt auch `except Exception as`). *Verifiziert: bestätigt; aktuelle Baseline ~891.* **Fix:** zwei lockerere löschen, eine mit gemessenem Count behalten.
- **[MAJOR · hoch · (a)✔Bug]** `TaxLotStore.close_lots` Over-Close untested — `qty_remaining>0` still ignoriert (`tax_lots.py:349-355`, nur DEBUG). *Verifiziert: bestätigt (Compliance-Lücke, Obs 741).*
- **[MAJOR · hoch · (a)✔Bug]** `test_integration_run_daily.py:122-148` `except Exception: pass` + `assert True` → einziger Integrationstest des `run_eod_pipeline`-Entrypoints **maskiert jeden Pipeline-Crash**. *Verifiziert: bestätigt (auch der `output_dir.exists()`-Check trivial via tmp_path erfüllt).*
- **[MINOR]** 7× `assert True`-No-op-Bodies (`:4231,4237,4794,8256,11777,12021,12224`); 3 non-strict `xfail` verstecken bekannte Gaps (Sunset 2026-07-01); `test_intel_to_signal.py` permanent geskippt (archiviertes Modul); schwacher Proxy-Check (`"logger" in content`).
- **[INFO]** `test_fifo_consistency.py` `except→None` maskiert Impl-Fehler; `test_disclosures.py:886 persistence.mode=live` ohne State-Path-Isolation (E-035-Risiko).

## ci+deps — `.github/workflows/`, `pyproject.toml`, `requirements*.txt`, `requirements.lock`
Health: **fragile**. Exit-127 & blanket `-W error` der A2-Historie sind **strukturell** gelöst (backend-ci pinnt via requirements.txt, gezieltes `filterwarnings`) → „7/7 green am 2026-05-28" plausibel, aber aus statischem YAML **nicht** zertifizierbar.

- **[MAJOR · hoch · (a)✔Bug]** Drei ungepinnte Windows-Installs = dritter Auflösungspfad. `accounting-ci.yml:40`, `ops-evidence-ci.yml:41`, `evidence-pack-ci.yml:40`: `pip install pandas pyarrow … scipy statsmodels scikit-learn` **ohne** Constraints. **[MAJOR · hoch · (a)✔]** Diese omitten inkonsistent Pakete (fastparquet/exchange_calendars/numpy/pyyaml) — `policy_loader` macht `import yaml` top-level (release-gate CI-002-Note). **[MAJOR · hoch · (a)✔]** `ci.yml:48 -e ".[dev]"` nutzt pyproject-Ranges während Rest pinnt → Drift. **[MAJOR · hoch · (a)✔]** `requirements.lock` stale (2026-04-08), divergiert (numpy 2.3.3 vs 2.2.6 etc.), von **0** Workflows konsumiert = Dead-Truth. **[MAJOR · mittel · (b)⚠]** `requirements.txt:51-52` scipy/sklearn als Ranges (3.10/3.11-Matrix-Drift). *Alle verifiziert: bestätigt.* **Fix:** alle auf `requirements.txt` vereinheitlichen; lock regenerieren/löschen.
- **[MINOR · hoch · (c)✎]** mypy non-blocking **und** doppelt suppressed (`backend-ci.yml:141 || true` + `:142 continue-on-error`). **[MINOR · mittel · (c)✎]** 16er CVE-Ignore-Liste ohne Expiry (inkl. `CVE-2024-47081` requests, `:84,109`).
- **[INFO]** backend-ci **Ubuntu-only**; Live-Host ist Windows (Umlaut-Pfad, Task Scheduler) → Windows-Regressionen (E-032/E-033) können CI passieren.

**Dead/Placeholder:** `requirements.lock`; `release-gate-ci.yml:73-119` Walk-Forward-Gate = **SYNTHETISCHER** Random-Walk-Smoke (seed=42) — kann die **echte** Strategie nicht zertifizieren; statistische Gates non-blocking via Grace-Date **2026-07-01** (~4 Wochen).

## repo-hygiene — archive/legacy/backup/autonome_weiterarbeit/experiments/…
Health: **fragile**. gitignore-Regeln existieren, wurden aber **nach** dem Tracking ergänzt → `git rm --cached` nie gelaufen.

- **[MAJOR · hoch · (a)✔Bug]** `autonome_weiterarbeit/` — 18 Planungs-/Audit-Docs trotz gitignore (`:95`) + Commit „remove from tracking" (5111d1f8 entfernte nur 2) noch getrackt; interne Strategie/COMPETITIVE_ANALYSIS/PAID_DATEN in voller Git-History. **[MAJOR · hoch · (a)✔]** `missing_symbols.txt` (eigen-korroboriert), `watchlist_29_backup.txt`, `watchlist_full.txt` committet ohne gitignore. **[MAJOR · hoch · (a)✔]** `system_check/runs/*/report.md` (3×) trotz Pattern committet. *Alle verifiziert.* **Fix:** `git rm --cached` (Patterns existieren).
- **[MINOR]** `experiments/*/run.json` (3×) ohne gitignore; `qa/bootstrap_multifactor_long_short_1d.json` degeneriert (`sharpe:0, sortino:Infinity, NaN` = RFC-8259-invalid) + Pattern matcht nicht; `archive/pipeline_legacy_2026q2/trading_cycle.py` importiert 13 protected Pipeline-Internals (Shim, kein Guard); `system_check/runner/tournament.py:536 except: heuristic` ohne Log.
- **[INFO]** `archive/` ~150 committete .py über 4 Graveyards (durch `testpaths=["tests"]` korrekt von pytest ausgeschlossen, aber Clone-Ballast); `.gitignore:87 F:*` Windows-Drive-Artefakt.
- *Hinweis:* `be_*.log`, `ci_*.json`, `jobs_*.json`, `*_pytest.log`, `test_run_debug.log` im Root sind **nicht** committet (nur Working-Dir-Clutter) — eigen-korroboriert.

## cross-cutting (src/ + scripts/)
Health: **fragile**. 201 broad `except Exception` über 109 src-Dateien; ffill auf Preis/Feature-Panels in 17 Dateien (meist korrekt forward-only annotiert); **keine** Hardcoded-Secrets (API-Keys env-var-gated). TODO/FIXME niedrig in src (2 Dateien), 47 Scripts.

- **[MAJOR · hoch · (a)✔Bug] — gemeldet BLOCKER → verifiziert MAJOR:** Policy-Load-Fehler disabled Risk-Controls still. `trading_cycle_shared.py:1565-1568 except: _cycle_policy=None`, dann in `filter_orders_with_risk_controls`. *Verifikation:* der äußere Handler (`:1587-1589`) ist bei Hard-Crash fail-closed; der reale Gap ist der **stille `None`-Pfad** (policy-getriebene Limits/Sektor-Caps werden still übersprungen, kein WARNING). **Fix:** fail-closed/Safe-Sentinel + Alert.
- **[MAJOR · hoch · (a)✔Bug] — gemeldet BLOCKER → verifiziert MAJOR:** Policy-Load liefert still leeres Dict in `_tc_signals.py:56-59` & `_tc_features.py:132-135` (`except: policy={}`, kein Log) → Threshold/Config-gated-Logik (Meta-Model-Threshold fällt auf hartkodiert 0.58) still disabled.
- **[MAJOR · hoch · (a)✔]** `_tc_signals.py:1065-1066, 1097-1098 except: pass` schluckt News-Dedup/Cluster-Fehler (korrupter Input in Signal-Scoring, kein Log). **[MAJOR · hoch · (a)✔]** `intermarket_factors.py:262 result.ffill(limit=5)` auf gesamtem Feature-Frame nach merge_asof → bis zu 5-Bar-Look-Ahead in abgeleitete Faktoren (E-030-Variante). **[MAJOR · hoch · (a)✔]** `qa/risk_metrics.py:97 equity_series.ffill().bfill()` — `bfill` propagiert ersten validen Equity-Wert rückwärts → inflationiert Early-Period-Returns, verzerrt MaxDD/Sharpe/CAGR **jedes** Backtests (E-030). **[MAJOR · mittel · (a)✔]** `unified_paper_engine.py:62-229` 18× Import-Level-`except: _HAS_X=False` → Safety-Subsysteme still optional (Safety-kritische loggen WARNING, aber **kein** Hard-Startup-Guard der Routing verweigert wenn `_HAS_KILL_SWITCH=False`). *Alle verifiziert: bestätigt.*
- **[MINOR · — · (b)⚠] — gemeldet MAJOR → MINOR:** `georisk_overlay.py:118-119 except: return target_positions` — *Verifikation:* das ist ein `import pandas`-Guard, kein Daten-Pfad-Swallow (unrealistisches Szenario).
- **[WIDERLEGT] — gemeldet MAJOR:** „event_bus `except: pass` killt alle Subscriber". *Verifikation:* `event_bus.py:174-178` ist `close()`-Cleanup (voluntary), **kein** Subscriber-Dispatch-Swallow. Befund existiert am zitierten Ort nicht → **kein Bug.**
- **[MINOR]** `trading_cycle_shared.py:1019,1055,1078,1121 except: return None` um Tail-Risk/VaR (Import-Guards + enabled-gated); `data/tick_store.py:123,169,230,253,272` silent pass; ML-Stubs callable trotz `NotImplementedError`.

---

# Querschnittsthemen (modulübergreifend, ≥3 Module)

1. **Silent Fail-Open am Safety-Rand via broad `except`** — das Signatur-Risiko des Projekts. Bestätigt in risk (CB-Wiring), execution (Symbol-Kill-Switch-Wrapper), accounting (Reconcile-kein-Halt), api (Ledger-Loader), events (State→WATCH), cross-cutting (Policy→None/`{}`), portfolio (Covariance→leer), intel (Enricher-Zeros). Immer dieselbe Form: Exception → harmloser Default → kein WARNING.
2. **Silent Degradation auf DEBUG statt WARN/ERROR** — weichere Variante von (1); verletzt die eigene Datenrealismus-Regel („nicht still verschlucken") systematisch (qa, signals, strategies, data, ops, paper, intel, core-misc). Höchster ROI: repo-weiter „Safety-Pfad DEBUG→WARNING"-Sweep.
3. **Latest-Bar/`as_of`-„Trust-the-Caller"-Vertrag** — `compute_signals` (trend_baseline/mfv2), `.tail()`-Reads (hrp_sizing, pre_trade-ADV, altdata, market_confirmation), VIX-`.iloc[-1]`, `groupby().last()`. PIT-Korrektheit wiederholt ohne internen Guard an den Caller delegiert — exakt der B1-OOS-Divergenz-Mechanismus.
4. **Disclosure-/Release-Lag-Asymmetrie über Altdata** — GPR macht `release_lag_days=32` korrekt; macro/earnings/news/insider default T+0 bzw. raw-Event-Merge `allow_exact_matches=True` (data, features, signals). Latent, weil Faktoren aktuell 0-gewichtet — Landmine bei Aktivierung.
5. **Duplicate-Truth / Doppelimplementierungen** (Rule-50-Verletzung breit) — DSR (Prob vs Z-Score), cross_asset_carry v1/v2, `AlertManager`×2 + 5 Alert-Module, `strategy/` vs `strategies/`, Cost-Model 10/5/0.5 bps, 3 except-count-Bounds, 3 Dependency-Auflösungspfade.
6. **Platzhalter als „echt" maskiert** — `target_qty`=Notional-als-Shares, PBO=Rank-Fraction-als-BLP, IC-t-Stat=„DSR", `chart_pattern_score`=0.0-Dim, Cost-Komponenten=0.0. Benannt wie das Echte, still inert/falsch.
7. **Tests, die nicht fehlschlagen können** — `assert True`-No-ops, Windows-`grep`-Count, Synthetik-Dup-String, `importorskip` skippt ganzes Modul, tautologischer Integrationstest (tests, execution, portfolio, ci).

# Coverage-Gaps / Re-Audit-Bedarf (Completeness-Critic)

- **`output/`-Writer-Concurrency als System** — in 6+ Modulen je eigene E-035/Lost-Update-Instanz gefunden (pipeline book_fills, paper RMW, ops order_lifecycle_log, scripts crisis_state, accounting non-fsync). **Niemand** besitzt die Gesamtfrage: vollständiges Writer-Set zu `output/`+`output/ops/` und welche Paare parallel laufen (EOD + Intraday + daily_scheduler + Backtest, der den Dir teilt). **Höchster Hebel — als ein Cross-Writer-Inventar re-auditieren.**
- **`feature_store.py`/`read_features_asof`-Embargo-Semantik** — PIT-Chokepoint, dem der ganze Faktor-Stack vertraut; nie primär auditiert.
- **`backtest.py`-Engine** — Harness hinter jedem „REJECTED"-Baseline; Isolation (output_dir, State-Restore, mode-Flag) nur als Negativ-Evidenz berührt. Load-bearing fürs Vertrauen in die OOS-Ergebnisse.
- **Data-Ingest-Sort-Vertrag** (`download_all_market_data.py`) — der `groupby().last()`-MAJOR hängt genau an „ist das Panel pro-Symbol timestamp-sortiert?"; der Upstream, der das (nicht) etabliert, wurde nie geprüft.
- **`policy_schema`** — zwei MAJOR hängen am stillen Policy-Load-Fehler, aber **was** das Schema validiert (werden Kill-Thresholds/Drawdown-Ordering überhaupt erzwungen, wenn es läuft?) wurde nicht geprüft.
- **Secrets/Key-Rotation** — Memory notiert chat-gepastete Finnhub/Alpaca-Keys; dedizierter Secrets-in-History-/Rotation-Status-Pass fehlt (nur „keine Hardcoded-Secrets gefunden").
- **monitoring/diagnostics-Router** — nur als 500-Leakage-Referenz berührt; nie geprüft, ob die Health/Monitoring-Zahlen real oder Platzhalter sind („Dummy-Monitoring"-Problemzone aus CLAUDE.md).

---

# TRADINGCENTER-URTEIL — Schlägt das System SPY?

## Verdict: **can_beat_spy = unlikely_as_is** (kein Wunschdenken)

**Kein einziges getestetes Konzept zeigt einen robusten, multiple-testing-deflationierten, statistisch signifikanten OOS-Edge über SPY netto Kosten.** Die negativen Ergebnisse sind **kein** Methodik-Artefakt — die Harnesses sind PIT-sauber, und die vorhandenen Biases (Survivorship) **flattern** die Kandidaten, können also keinen Edge in den abgelehnten Long-Books **verstecken**.

### Was die Evidenz zeigt (Belege)
- **trend_baseline** (die *tatsächlich gepilotete* Live-Strategie): **0/10 Folds** schlagen SPY, Ø CAGR **−6,1%** vs SPY +13,0% (`GO_LIVE_CHECKLIST.md:79-84`). Das ist der gravierendste Befund: die schwächste Strategie ist verdrahtet (`:116-125`).
- **multifactor_v2** erreicht nur SPY-**CAGR** (+12,9% vs +13,0%) bei **Sharpe 0,36 vs 0,95 (2,6× schlechter)** — und nur im degradierten TA-only-Test (19/34 Faktoren = 0). Full-Stack-Aktivierung: **Sharpe-Delta +0,00** (`mfv2_full_stack:78`).
- **multifactor_long_short**: Ø CAGR **−19,5%, 0/10** (`strategy_comparison.md:25`).
- **Stärkstes Einzelergebnis:** long-only Total-Return-Momentum (`mom_lo`): IR vs SPY **+1,09, t=+2,67** (signifikant), Sharpe +1,36 vs 0,91 — **aber DSR✗** (DSR-Prob 0,93<0,95 nach Deflation für n_trials=16), Beta **+1,09** (schlägt SPY v. a. durch *mehr* Marktrisiko), und auf survivorship-**inflationiertem** Universum. Vol-matched Excess +26,9% (etwas echter vol-adjustierter Überschuss), aber durch das Deflated-Sharpe-Gate gefiltert. Nur Konzept, keine verdrahtete Strategie.
- **sector_lo** (Full-History 1998–2026): besteht DSR (0,99) **aber** IR-t nur +0,85<1,96 → REJECTED.
- Alle L/S-, residual-momentum-, lowvol-, dual-momentum-, ETF-pairs-, new-factor-Sweeps: **REJECTED**.

### Sind die OOS-Methoden solide oder verstecken sie einen Edge?
**Solide; sie verstecken keinen.** Direkt aus den Harnesses verifiziert (`_oos_wf_residual_momentum.py`): Selektion `ref_idx=window_end_idx-1` (strikt pre-rebalance), Execution `pos.shift(1)` (1-Bar-Lag), Kosten auf Turnover-Delta, SPY auf gleicher Kostenbasis. Walk-Forward non-overlapping (252/252/252), CPCV-Leakage-Check 4/4 PASS, 6 PIT-Regressionstests. Deflation (DSR/PSR via `qa/deflated_sharpe.py`, n_trials=16) — genau das, was `mom_lo` killt. Survivorship-Richtung pro Book ehrlich begründet: für Long-Only **inflationierend** → saubere Universen wären *schwächer*; für Short-the-Junk-L/S **konservativ** → trotzdem REJECTED (starkes Negativ). Cross-validiert gegen die **echte** `run_trading_cycle` + Produktions-Cost/Fill-Model (`dual_momentum_literal_oos.md`, reproduziert REJECTED).

**Methodik-Caveats (ehrlich):** (a) Das OOS-Gating stützt sich auf die **korrekte** DSR-Implementierung (`qa/deflated_sharpe.py`, Wahrscheinlichkeitsvariante) — **nicht** auf das fehlbenannte „PBO" oder die divergente Z-Score-DSR (s. qa-Modul). Das Projekt sollte sein „PBO"-Gate **nicht** als BLP-PBO zitieren. (b) **MinTRL-Compliance vs Gesamtzahl getesteter Konfigurationen** ist nicht offengelegt (Methodik-Gap). (c) Die zwei E-030-ffill/bfill-Look-Aheads (`risk_metrics.py:97`, `intermarket_factors.py:262`) verzerren Metriken/Faktoren — sie betreffen v. a. die Multifaktor-/Reporting-Pfade, nicht die primären 1-Bar-Lag-OOS-Returns, sind aber vor jedem „Trust the number" zu fixen.

### Kostenmodell
**Realistisch bis leicht konservativ** für das getestete liquide US-Equity/ETF-Universum: 10–10,75 bps/Leg + spread_w 0,25 + impact_w 0,5, + 30–50 bps/yr Borrow auf L/S. Produktion ist reicher (per-Kategorie commission/spread/impact/slippage/SOR). SPY-Benchmark **ohne** Dividenden-Reinvest macht die Latte leicht *zu niedrig* (Kandidaten geflattert, nicht bestraft). Keine dieser Lücken würde ein abgelehntes Book retten. *(Methodik-Recherche merkt an: kein Square-Root-Market-Impact / Permanent-Impact-Komponente — für Low-Turnover-Daily-ETFs adäquat, für Small-Cap-L/S Unterschätzungsrisiko.)*

### Externe Realität (Web-Recherche, zitiert)
- **SPIVA YE-2025:** 79% der US-Large-Cap-Aktivfonds underperformen den S&P 500 (2025); 5J **89%**, 20J **93%** ([spglobal.com/spdji/spiva](https://www.spglobal.com/spdji/en/spiva/article/spiva-us/), [tker.co](https://www.tker.co/p/spiva-2025-active-manager-vs-benchmark)). Über 15J schlägt in **0 von 22** US-Equity-Kategorien eine Mehrheit aktiver Manager ([icfs.com](https://icfs.com/specialists-desk/spiva-scorecard-results)).
- **Faktor-Decay:** McLean & Pontiff (2016): Renditen 26% niedriger OOS, **58% niedriger post-publication** ([JoF 10.1111/jofi.12365](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)). Momentum ~10%/yr (1990er) → heute ~2%. Hou/Xue/Zhang: **65% von 452 Predictors** replizieren nicht ([JoF 10.1111/jofi.13249](https://onlinelibrary.wiley.com/doi/full/10.1111/jofi.13249)).
- **Retail-Realität:** Survivorship überschätzt Renditen 1–4%/yr; Backtest-Sharpe 3,0 → live 0,5 nach realistischen Fills; ab ~0,4% Kosten/Trade verschwindet EOD-Alpha statistisch ([elitetrader thread](https://www.elitetrader.com/et/threads/realistic-sharpe-ratios-in-2026-hft-vs-retail-algos-deep-dive.388680/), [quantifiedstrategies](https://www.quantifiedstrategies.com/survivorship-bias-in-backtesting/)). Free-Tier-Feeds ohne Point-in-Time-Fundamentals erzeugen nicht-korrigierbaren Look-Ahead ([quantstart](https://www.quantstart.com/articles/Successful-Backtesting-of-Algorithmic-Trading-Strategies-Part-I/)).

Das Projektergebnis (uniform REJECTED bei sauberer Methodik) ist also **exakt das wissenschaftlich erwartbare** — nicht Versagen der Umsetzung, sondern Bestätigung, wie schwer Konsistenz-Alpha über SPY ist.

### Unter welchen realistischen Bedingungen könnte ein Edge entstehen?
**Realistisch testenswert (evidenzgestützt):**
1. **Long-only-Momentum / 52-Wochen-Hoch als SPY-Tilt, beta-gesized, auf survivorship-sauberem Universum.** `mom_lo` t=+2,67 + `high52w` (MaxDD nur −10,7%) deuten einen realen, aber **kleinen** Tilt an. Ehrlicher Test: überlebt der vol-matched Excess auf delisting-vollständigem Universum **und** besteht DSR? Aktuelle Daten: „vielleicht klein", nicht „ja".
2. **Defensive-Momentum-Combo für Drawdown-Kontrolle, nicht Rendite** (`lowvol_momentum`: MaxDD −10,3% vs SPY −14,7% bei ~SPY-Sharpe) — ein plausibles **Risk-Reduction-Produkt** (besseres Calmar), kein Alpha-Produkt.
3. **Dual-Momentum/Vol-Target als Crash-Versicherung** (MaxDD 0,68–0,97× SPY) — schneidet Tails, hinkt im Bull-Sample. Edge nur in einem Sample mit echtem Bären (Testfenster 2018–2025 ist bull-dominiert → benachteiligt **jedes** defensive Book strukturell).

**Spekulativ (keine Evidenz):**
- **Voller mfv2-Altdata-Stack** — 19–25/34 Faktoren tot; die eine Aktivierung zeigte Sharpe-Delta +0,00. Hoffnung, kein Beleg, bis echte PIT-saubere Historie existiert.
- **News-/Event-getrieben (news_alpha/crisis_alpha)** — nicht OOS-falsifiziert (News-Daten erst ab 2025-12, kein historisches OOS möglich). **Echt unbekannt.** *(Achtung: news_alpha EOD-Pfad erzeugt derzeit 0 Signale — s. events-Modul.)*
- **Leverage/Short** — explizit out-of-scope; alle getesteten L/S-Books ohnehin REJECTED.

### Bottom Line (ehrlich)
Dies ist ein ungewöhnlich **ehrlicher** Research-Record: PIT-sauber, Walk-Forward-/CPCV-geprüft, kosten-bewusst (gegen reale Pipeline kreuzvalidiert), korrekt deflationiert. Über ~16 Konzepte liefert **keines** einen deflationierten, signifikanten, kostenüberlebenden Edge über SPY. Das einzige Signal mit signifikantem Roh-IR-t (long-only Momentum, t=+2,67) erreicht das überwiegend durch **höheres Marktbeta** und fällt am Deflated-Sharpe-Gate auf survivorship-*inflationiertem* Universum. **Der realistische Nahbereich-Wert ist nicht „SPY auf Alpha schlagen", sondern bestenfalls „SPY tracken mit deutlich kleineren Drawdowns"** (defensive-momentum/dual-momentum/vol-target: 0,7–0,97× SPY-MaxDD) — ein Risk-Management-Produkt, das nur in einem bull-dominierten Sample validiert wurde, das genau diese Books benachteiligt. Ein echter Return-Edge bräuchte, in Plausibilitätsreihenfolge: (a) survivorship-sauberen Re-Test von long-only-Momentum beta-gesized, (b) ein echtes Bären-Regime im OOS-Fenster, (c) echte PIT-saubere Historie für die toten Altdata-Faktoren — und selbst (a) zeigt aktuell „klein, DSR-grenzwertig", nicht „robust".

---

# Auf welchen Grundlagen weitermachen — priorisiert

## A. Blockt sicheren Paper-/Live-Betrieb (zuerst)
1. **Alert-Delivery reparieren** (ops #1) — autonomer Worker schreibt nur JSON+Konsole; CRITICAL-Alerts erreichen **keinen** Menschen. Alles andere setzt funktionierendes Alerting voraus. *(`daily_scheduler.py:738-831`)*
2. **Reconcile-FAIL eskalieren** (ops #2 + accounting #1+#3) — Scheduler keyt auf nie geschriebene `*.error`; Reconcile warnt nur & läuft weiter; No-Snapshot reconciled Paper-vs-Paper als „pass". Driftendes Book handelt still weiter.
3. **Policy-Load fail-closed** (cross-cutting) — `except→None`/`{}` betreibt Pre-Trade-Gate ohne Policy / disabled Signal-/Feature-Gating, kein WARNING. Live-Risk-Gate-Bypass bei transientem YAML-/Import-Fehler.
4. **Cost-Model fail-closed** (paper #2) — fehlend/umbenannt → Zero-Cost-Fills → still optimistisches Paper-P&L (genau das, worauf die Go/No-Go-Entscheidung beruht).
5. **Kill-Switch-Fail-Open-Varianten** (execution) — Symbol-Wrapper passt Orders bei Exception durch; Duplicate-Recovery erholt sich nie + Matcher verfehlt reale Alpaca-Strings → Risiko zweiter Live-Order bei Crash-Retry.
6. **DMS in Task Scheduler verdrahten/verifizieren** (ops #3) — Auto-Flatten-Watchdog läuft sonst nur bei manuellem Start.
7. **scipy-`__init__`-Import-Break** (portfolio) — `import …portfolio` bricht hart ohne scipy auf Py3.10 (A2-Klasse).

## B. Research-Qualität / Edge (den Zahlen erst nach Fix trauen)
8. **`output/`-Writer-Concurrency als System re-auditieren** — bis das vollständige Writer×Concurrency-Inventar steht, ist jede „Shared-Dir"-Mitigation per-Instanz & unbewiesen; Backtest, der den Live-Dir teilt, korrumpiert Ops-State (E-035) **und** vergiftet OOS-Inputs.
9. **Backtest-Engine-Isolation auditieren + `groupby().last()`-Sort-Invariante** — die „ALL REJECTED"-Baselines sind nur so vertrauenswürdig wie State-Isolation + Upstream-Sort.
10. **PIT-Release-Lag vereinheitlichen** — earnings/news/macro/insider über ein `apply_source_latency`, Merge auf `disclosure_date`, `allow_exact_matches=False`. **Bevor** einer dieser Faktoren je >0 gewichtet wird (aktuell nur latent, weil tot).
11. **E-030 fixen:** `risk_metrics.py:97 ffill().bfill()` (verzerrt MaxDD/Sharpe/CAGR jedes Backtests) + `intermarket_factors.py:262 ffill(limit=5)`.
12. **`as_of`-Guard/lauter Docstring-Vertrag** auf `compute_signals`/Strategie-Entrypoints (B1-Mechanismus, billig mechanisch absicherbar).
13. **QA-Gate-Integrität** — PBO ist kein BLP-PBO; zwei DSR-Wahrheiten; `check_max_drawdown` crasht bei None; None-Sharpe fail-open. Das sind die Gates, die den Edge *zertifizieren*.

## C. Hygiene / Wartbarkeit
14. **Safety-Pfad DEBUG→WARNING repo-weit** + broad `except` auf erwartete Typen verengen — höchster ROI, macht Themen 1+2 sichtbar.
15. **Duplicate-Truths kollabieren** — eine DSR, eine Alert-Fassade, eine Cost-Model-Quelle, ein Dependency-Pfad (Windows-Installs + `ci.yml` auf `requirements.txt`; `requirements.lock` regenerieren/löschen), `strategy/`-Singular retiren.
16. **Platzhalter-als-real umbenennen** — `target_qty`→`target_notional`, PBO→`oos_rank_fraction`, ops-„DSR"→`ic_tstat`, `chart_pattern_score`/Cost-Split als inert markieren.
17. **Nicht-fehlschlagbare Tests reparieren** — 3 except-count-Bounds, Windows-`grep`-Count, tautologischer Integrationstest, `assert True`-No-ops; fehlende Fail-closed-Tests ergänzen (kill-switch-raises, dup-recovery, over-close `qty_remaining>0`, zero-cost-resolver, reconcile-FAIL-escalates).
18. **Repo-Hygiene** — `git rm --cached` der 18 `autonome_weiterarbeit/`-Docs + watchlist/experiments/system_check-Artefakte; `qa/bootstrap_*.json` (Infinity/NaN) entfernen; `.gitignore:87 F:*` fixen; Archive-Import-Shim guarden.
19. **Secrets-Pass** — chat-gepastete Finnhub/Alpaca-Keys rotieren, `.env`-History-Status prüfen (Memory-Hinweis, nie auditiert).

---

## Schlusswort

Die **Mathematik und die Kern-Safety-Primitive** (Kill-Switch, VaR/CVaR, GARCH, Pre-Trade-Checks, atomare Accounting-Math, Kill-Switch-Auth) sind **echt gut gebaut** und vielfach review-gehärtet. Das systemische Risiko ist **stilles Fail-Open am operativen Rand**: Alerts, die nicht feuern; Reconcile-/Policy-/Cost-Fehler, die zu harmlos aussehenden Defaults degradieren; eine geteilte `output/`-Schreiboberfläche, die niemand als Ganzes auditiert hat. **Vor jedem Vertrauen in Paper-/Live-Signale: Delivery + Fail-Closed-Semantik fixen; vor jedem Vertrauen in eine OOS-Zahl: ffill/PIT/Release-Lag fixen.**

Zur SPY-Frage: **Es gibt aktuell keinen belegten, robusten Weg, SPY auf risikoadjustierter Alpha-Basis zu schlagen** — und die saubere Methodik des Projekts macht dieses negative Ergebnis *vertrauenswürdig*. Der ehrliche Nahbereich-Pfad ist ein **Drawdown-Reduktions-/Risk-Management-Produkt** (SPY tracken, kleinere Drawdowns), plus die offen-unbekannten event-getriebenen Pfade (news_alpha/crisis_alpha), die mangels Historie schlicht noch nicht beurteilbar sind. Ein echter Return-Edge bleibt möglich nur unter den oben genannten, ehrlich als „klein/grenzwertig/unbewiesen" markierten Bedingungen — nicht als gegeben.

---

# NACHTRAG (2026-06-03) — Gap-Closure & Erstprüfung der Top-Blocker

**Anlass:** Der Completeness-Critic hatte sieben Coverage-Gaps benannt, die im Hauptteil nur *gelistet*, nicht *auditiert* waren; zusätzlich wurden vier load-bearing Betriebs-Blocker unabhängig erstgeprüft. Dieser Nachtrag schließt das (read-only, jeder Befund first-hand am `path:line` verifiziert). **Er korrigiert zwei Überzeichnungen des Hauptteils und fördert fünf neue ernste Befunde zutage.**

## Korrekturen am Hauptteil (Ehrlichkeit)

1. **pipeline `groupby().last()` (war MAJOR) → MINOR / latenter Code-Smell.** Aufgelöst: Das Preis-Panel ist an **jedem** Produktions-Call-Site timestamp-sortiert, bevor `groupby("symbol").last()` läuft — `prices_ingest.py:173`, `sources/yfinance_source.py:187`, `qa/backtest_engine.py:1339`, `ops/replay_snapshot.py:260`; `_filter_prices_for_as_of` macht nur Boolean-Masking/`.copy()` (keine Umsortierung). Kein aktiver Bug; nur impliziter Cross-Modul-Vertrag ohne Assertion am Call-Site.
2. **pipeline `book_fills` E-035 (war MAJOR, pauschal) → präzisiert.** Korrektur zu obs #748: der **FILLED**-Hook (`_tc_execution.py:493`) liegt **im** `write_outputs`-Block und ist korrekt gegated. Tatsächlich leaken nur (a) der **SUBMITTED-Lifecycle**-Hook (`_tc_risk.py:350-377`, nur durch `output_dir` gegated, nicht `write_outputs`/`mode`) und (b) der **Heartbeat** (`_tc_execution.py:525-541`, **gar kein** Guard). Siehe G2.
3. **paper Unlocked-RMW (war MAJOR „EOD+Intraday") → Framing korrigiert.** Der Intraday-news_alpha-Runner schreibt **nur** `output/news_alpha_state.json` und teilt **keine** Datei mit dem EOD-Cycle (G1). Die reale Lost-Update-Kollision ist **zwei EOD-Cycles** bzw. **EOD vs. Backtest**, nicht EOD-vs-Intraday. Zudem ist die **Ledger-State**-Persistenz per `filelock` **geschützt** (`paper_ledger.py:195-215`) — der ungeschützte Teil sind die Aggregat-Parquets + Heartbeat.
4. **cross-cutting Policy-Load (war BLOCKER→MAJOR) → bleibt MAJOR, aber Charakter korrigiert.** Es ist **kein** Risk-Gate-Bypass (s. V2): das Pre-Trade-Gate + Standard/DD-Kill-Switch laufen unabhängig von der geladenen Policy. Realer Defekt = stilles Defaulting von **Signal-/Feature-Config** ohne Log (Observability), plus die separat schwerwiegende drawdown_policy-Shape-Sache (G5).
5. **ops Alert-No-op (war MAJOR, pauschal) → präzisiert (s. V1).** Kill-Switch-Aktivierung liefert sehr wohl (über die *andere* `AlertManager` → Telegram/E-Mail), **wenn** Env-Creds gesetzt sind. Der Scheduler-Worker bleibt no-op. Neu/schlimmer: Reconcile- und Circuit-Breaker-Alerts werden **still gedroppt** (Regel fehlt in `alerting.yaml`).

## Teil A — Aufgelöste Coverage-Gaps

### G1 — `output/`-Writer-Concurrency als System
**Verdict: Keine System-Serialisierung. Nur 2 Pfade cross-process gelockt; `utils/file_lock.py` ist Dead-Code.**
- **[MAJOR · hoch · (a)✔]** Keine systemweite Schreib-Serialisierung. `filelock` existiert in genau 3 Runtime-Files (`kill_switch.py`, `paper_ledger.py`, `experience_log.py`); **`utils/file_lock.py:FileLock` wird von nichts importiert (toter Code).** Concurrency-Safety ist per-Datei & ad hoc, nicht architektonisch.
- **[INFO]** Korrekt gelockt (cross-process): Kill-Switch-State/Audit (`execution/kill_switch.py:79-131`, OPS-04-Lock gegen DMS-Daemon-vs-Runner-Race) und Paper-Ledger-State (`ops/paper_ledger.py:195-215`).
- **[MAJOR · mittel · (b)⚠]** **Kollisionspaar `output/state/heartbeat.json`** — geschrieben von Scheduler-Parent (`paper_trading_scheduler.py:130-134`) **und** EOD-Child (`_tc_execution.py:527-538`), gleicher Pfad, **kein Lock**; sequenziell nur durch das *blockierende* `subprocess.run` (Zufall der Topologie). Bei 600s-Timeout-Overrun schreibt der Parent „alive", während ein verwaister Child noch schreibt → truncated tmp. **Höchste Konsequenz, weil der DMS den Heartbeat liest, um Flatten-Entscheidungen zu treffen.**
- **[MINOR · hoch · (a)✔]** `trade_journal.jsonl` (`trade_journal.py:106`; `_next_trade_id` liest ganze Datei vor jedem Append → Duplicate-`TJ-`-IDs bei Parallel-Append) und `order_lifecycle.jsonl` (`order_lifecycle_log.py:86`) — plain `open(...,"a")`, kein Lock; Kollision nur bei Two-EOD / EOD-vs-Backtest (E-035). `unified_paper_engine.py:1733` schreibt Ledger-Parquet **bare** `to_parquet` (nicht mal tmp+replace) → Truncation-Risiko bei Crash.

### G2 — `backtest.py`-Engine-Isolation
**Verdict: Isolation partiell & caller-abhängig. Die load-bearing `sector_rotation`-OOS-Verdicts sind sicher; Pipeline-Pfad-OOS ist exponiert.**
- **[MAJOR · hoch · (a)✔]** Kill-Switch-Trips im Replay **persistieren in Live-State.** `kill_switch.py:46-48` hartkodiert `output/ops/...` (nur env-, nie `output_dir`-scoped); `_tc_risk.py:215` (auto-DD) und `trading_cycle_v2.py:327` (CB) rufen das echte `activate_kill_switch` im Backtest auf (Engine-Default `enable_risk_controls=True`, `backtest_engine.py:359`); der Restore-Guard greift nur bei `kill_switch_persist=False`, das die Backtest-Engine **nie setzt** → Default `True` → Restore inaktiv.
- **[MAJOR · hoch · (a)✔]** Read-Side-Leak: Backtest liest denselben globalen Live-Kill-Switch (`is_kill_switch_engaged`, `kill_switch.py:474`) → ein live engagierter Kill-Switch **nullt OOS-Orders** (biast Ergebnisse **nach unten**); in-Replay-Trip gatet Folge-Bars.
- **[MINOR/MAJOR]** `mode=="backtest"` gatet die Ops-Writes **nicht** — `write_outputs` tut es; aber SUBMITTED-Lifecycle (nur `output_dir`-gated) und Heartbeat (ungated) leaken trotzdem in Live-`output/`.
- **[INFO · hoch]** **Trustworthy:** `_oos_wf_sector_rotation*.py` (und die vektorisierten Harnesses) enthalten **kein** `run_trading_cycle`, keinen Kill-Switch, keine Ops-Writes → die „ALL REJECTED"-Verdicts sind state-isolations-sicher. **Nur** Pipeline-Pfad-OOS (`_oos_wf_pipeline_realistic.py`, `dual_momentum_literal`) erbt die Risiken — und der Read-Leak-Bias ginge nach **unten**, kann also keinen Edge verstecken. **→ Das SPY-Urteil bleibt belastbar.**

### G3 — Data-Ingest-Sort-Vertrag → **kein Live-Bug** (s. Korrektur 1 oben).

### G4 — `feature_store`/`read_features_asof`
**Verdict: ASOF-Vertrag PIT-safe by design; einziger Pfad inert → MINOR. Kein Multi-Faktor-Look-Ahead-Vektor.**
- **[INFO]** Default `embargo_minutes=1`, ASOF-Join filtert auf `available_at` (`feature_store.py:160,203`) — korrektes PIT-Muster.
- **[MINOR · hoch]** Einziger src-Consumer `conviction_engine.py:141` ruft mit `embargo_minutes=0` — aber backtest-gated (`allow_in_backtest=false`).
- **[MAJOR (toter Pfad) · hoch · (a)✔]** **`event_beta`-Producer ist tot:** `compute_event_betas.py` schreibt `inference_ts=event_date` (Roh-Event-Zeit), **nie** eine `available_at`-Spalte, **nie** via `write_features()`, und flachen Pfad `event_beta/<date>.parquet` während der Reader `view=event_beta/**` globt → `compute_event_beta()` gibt **immer None** zurück (bestätigt `docs/edcl/decisions.md:254`). Inert, aber bei Aktivierung zusammen mit embargo+available_at+Pfad zu fixen.

### G5 — `policy_schema` — tatsächliche Enforcement
**Verdict: Validierung rein advisory; prüft den FALSCHEN Drawdown-Key-Tree. MAJOR + neuer schwerer Config/Code-Mismatch.**
- **[MAJOR · hoch · (a)✔]** Beide Validatoren raisen nie (`policy_schema.py:29,120` „Does NOT raise"); nur `warnings.append`/`logger.warning`. **Kein** Risk-/Execution-/Pipeline-Modul ruft sie auf (nur `scripts/health_check.py` + Tests). Kombiniert mit dem stillen Swallow in `policy_loader.py:57-69` (DEBUG): eine malformte Policy (invertierte Ordering, Out-of-Range-Kill, falscher Typ, `leverage_allowed:true`) lädt sauber und treibt Live-Trading **unentdeckt**.
- **[MAJOR · hoch · (a)✔ — NEU]** **Validatoren prüfen einen anderen Drawdown-Baum als der Live-Kill-Switch liest.** Schema prüft `risk_limits.max_drawdown.{soft,hard,kill}` (positiv, `soft<hard<kill`); der Live-Auto-DD-Kill-Switch `_evaluate_auto_dd_kill_switch` liest `policy["drawdown_policy"]["levels"]` (negativ, `trading_cycle_shared.py:1152,1170-1173`). **Zusätzlich** legt `configs/policy.yaml:71-74` die Thresholds direkt unter `drawdown_policy:` (positiv 0.10/0.15/0.20) ab, während der Code `drawdown_policy.levels` erwartet → **die operator-gesetzte Kill-Schwelle wird zur Laufzeit ignoriert; es greift der hartkodierte Default `{soft:-0.08, hard:-0.12, kill:-0.18}`.** Kein Validator fängt das. (Nicht BLOCKER nur, weil der Kill-Switch hartkodierte Fail-Safe-Defaults + `enable_kill_switch=True` hat — aber die operator-konfigurierbare Sicherheitsschwelle ist effektiv unvalidiert und teils ungelesen.)

### G6 — Secrets / `.env`-History (Security, Rule 20 — keine Werte ausgegeben)
**Verdict: BLOCKER — `.env`-Secrets sind in der Git-History exponiert; Keys müssen als kompromittiert behandelt und rotiert werden.**
- **[BLOCKER · hoch · (a)✔]** `.env` wurde bei `0ca19ef0` („Sprint 10: project init", 2025-10-05) committet und bei `e64fa215` („security: remove .env from git index after key rotation", 2026-04-19) aus dem Index entfernt — **der Blob bleibt aber in der History** und ist auf `main`, `origin/main` **und** `origin/ERWEITERUNG` extrahierbar (`git cat-file -t 0ca19ef0:.env` → blob, 87 bytes). Wer Klon-/Fetch-Zugriff auf origin hat, kann das historische `.env` lesen.
- **Pflicht (Rule 20):** (1) **Rotation ist zwingend** — jeder im Blob `0ca19ef0` enthaltene Key/Token ist beim Provider zu rotieren, unabhängig von aktueller Nutzung; die in `e64fa215` erwähnte Rotation ist auf Vollständigkeit zu **re-verifizieren**. (2) `.gitignore` (`:38-47`, korrekt) schützt **nur künftige** Commits, nicht die History. (3) **History-Bereinigung** (`git filter-repo`/BFG) ist destruktiv (Force-Push auf `origin/main`+`origin/ERWEITERUNG`, invalidiert alle Clones/Forks) → **offene Projektentscheidung**, hier benannt, nicht auto-empfohlen, nicht ausgeführt. Bis Rotation **und** Purge ist das Risiko nicht beseitigt.
- **[INFO]** Keine weiteren Credential-Files in History/Index (`*.pem` nur `.venv/` certifi-CA; `*token*`-Treffer sind Code/CSS-Design-Tokens/CI). `.secrets.baseline` + `secrets-scan.yml` (detect-secrets/gitleaks) vorhanden — Detektionslayer existiert, behebt den Alt-Leak aber nicht rückwirkend. `.env.example` = Platzhalter (by design getrackt).

### G7 — monitoring/diagnostics-Router (Dummy-Monitoring-Problemzone)
**Verdict: GEMISCHT — Kern-Endpoints real, aber 1 Dead-Import-BLOCKER + mehrere Placeholder, die als „live" auftreten.**
- **[BLOCKER · hoch · (a)✔ — NEU]** `/monitoring/alerts` Kill-Switch-Branch ist toter Code: `monitoring.py:534 from src.assembled_core.risk.kill_switch import KillSwitch` — **dieses Modul/diese Klasse existiert nicht** (real ist `execution/kill_switch.py` funktionsbasiert: `is_kill_switch_engaged()`/`get_kill_switch_state()`). Import raised immer → `except` (`:545`) swallowed → **ein aktiver Kill-Switch erzeugt am Dashboard NIE einen Alert.** Fix: auf `execution.kill_switch.get_kill_switch_state()` umstellen.
- **[MAJOR · hoch · (a)✔ — NEU]** `/monitoring/regime` (`:473`), `/monitoring/signals` (`:594,598`), zombie+correlation-Teil von `/monitoring/alerts` (`:516,554`) globen Datei-Patterns (`regime_state_*.json`, `zombie_report_*.json`, `correlation_guard_*.json`, `signal_scores_*`), die **kein** Code irgendwo schreibt → permanent `stale`/`unavailable`/leer, unabhängig vom echten Zustand. Placeholder, die als live auftreten (CLAUDE.md „Dummy-Monitoring").
- **[INFO · hoch]** **Real & vertrauenswürdig** (lesen echte Artefakte, 404/503 ehrlich): `/monitoring/qa_status`, `/risk_status`, `/drift_status` (raised 503 statt Fake-„NONE"), `/portfolio` (echter SQLite-Ledger), `/diagnostics/feature-drift` (echtes PSI via `qa.drift_detection.compute_psi`). `/diagnostics/modules` ist ein **statischer** Snapshot (`_MODULE_REGISTRY`, „ground truth … 2026-05-05"); Caller-Counts driften still.

## Teil B — Erstprüfung der 4 Top-Blocker (Ergebnis)

- **V1 alert-delivery → PARTIELL WIDERLEGT/präzisiert.** Zwei `AlertManager`: Scheduler-Worker (`ops/alert_manager.py`) log+JSON (no-op, bestätigt); Kill-Switch-Pfad (`kill_switch.py:354` → `ops/alerting.py` → `alerting.yaml` critical→telegram+email) **liefert, wenn Env-Creds gesetzt** (sonst `logger.warning`-Skip). **[BLOCKER · hoch · NEU]** `reconciliation_fail/warn` (`reconciliation.py:188-191`) + `circuit_breaker_tripped` (`circuit_breaker.py:219`) feuern Regeln, die in `configs/alerting.yaml` **fehlen** → `alerting.py:56-58` `fire()`→False (nur DEBUG) → **diese CRITICAL-Alerts erreichen niemanden, landen nirgends.** Failover/Discord-Sinks nur drill-only (`scripts/`), nie aus src-Pipeline.
- **V2 policy-failopen → REFUTED als BLOCKER → MAJOR.** Die 3 `except`-Sites (`_tc_signals.py:56-59`, `_tc_features.py:132-135`, `trading_cycle_shared.py:1565-1568`) existieren verbatim & ohne Log. Aber: `filter_orders_with_risk_controls` nutzt `policy` **nur** für den Crisis-Alpha-Kill-Switch (`if policy is not None`); Pre-Trade-Checks + Standard/DD-Kill-Switch laufen über `pre_trade_config`/Env. Risk-Limit-Defaults haben **eigenes** WARNING (`:1493-1496`), äußerer Handler ist fail-closed (`:1587-1598`). → **Kein Risk-Gate-Bypass**; realer Defekt = stilles Defaulting von Signal-/Feature-Config. Minimal-Fix: WARNING in die 3 except-Blöcke.
- **V3 cost-zero → BESTÄTIGT (MAJOR).** `_resolve_cost_cfg` (`paper_runner.py:1136-1147`) injiziert keinen Default, loggt nicht; leeres cfg → `commission_bps=0`/`slippage_bps=0` (`src/.../ops/paper_ledger.py:231-239`) → Fills zu Exact-Close. Maskiert **nur** durch `policy.yaml:861-866` (Config, nicht Code); eine soft policy-load-Fehlschlag (`_load_pilot_policy_fail_fast`→`{}`) erzeugt still Zero-Cost-Fills. (Nuance: Policy setzt kein `slippage_bps`; Slippage kommt aus dem spread/impact-Fallback.)
- **V4 newsalpha-zero → BESTÄTIGT & verstärkt zu BLOCKER.** Nicht nur der `topic`↔`topic_id`-Mismatch (`signal_generator.py:83` vs `trigger_scoring.py:196`), sondern **der EOD-Wiring erhält gar nie korrekt-geformte Trigger-Items:** `compute_news_geo` (`intel_runner.py:152-157`) liefert nur `geo_score/geo_confidence/state_hint/top_triggers` — **keinen** der drei in `_tc_sizing.py:1868-1872` gesuchten Keys; der einzige Setter von `ctx.news_geo["active_triggers"]` (`trading_cycle_v2.py:273`) liefert Trigger-ID-**Strings**, keine `{severity,topic,source}`-Dicts. → `_trigger_items=[]` → `generate_signals` iteriert nichts → **0 news_alpha-Signale im EOD-Pfad, still (kein Warning).** Intraday funktioniert, weil es eigene `"topic"`-Triggers baut und `news_geo` umgeht. **Der news_alpha-EOD-Live-Pfad ist faktisch komplett funktionslos.**

## Aktualisierte Top-Priorität (Nachtrag-Ergänzungen)

Zusätzlich zu Abschnitt A–C des Hauptteils, **neu/aufgewertet**:

- **NEU BLOCKER (Security): `.env` in Git-History → Keys rotieren** (G6) — sofort, providerseitig; e64fa215-Rotation auf Vollständigkeit prüfen.
- **NEU BLOCKER: Monitoring-Kill-Switch-Dead-Import** (G7) — `/monitoring/alerts` kann einen aktiven Kill-Switch nie melden (falscher Import-Pfad).
- **NEU BLOCKER: Reconcile- & Circuit-Breaker-Alerts fehlen in `alerting.yaml`** (V1) — CRITICAL-Reconcile/CB-Alerts werden still gedroppt; ergänzt Hauptteil-Blocker A1/A2.
- **NEU BLOCKER: news_alpha-EOD-Pfad komplett funktionslos** (V4) — Wiring liefert nie geformte Trigger; Feature gilt als „aktiv", ist inert.
- **NEU MAJOR: drawdown_policy-Shape-Mismatch** (G5) — operator-gesetzte DD-Kill-Schwelle wird ignoriert; hartkodierter Default greift. (Direkt sicherheitsrelevant: die wichtigste konfigurierbare Risk-Grenze ist nicht wirksam.)
- **MAJOR aufgewertet/präzisiert: backtest persistiert Kill-Switch in Live-State + Read-Leak** (G2) — Pipeline-Pfad-OOS isolieren (Engine `kill_switch_persist=False` erzwingen, Kill-Switch-Pfad output_dir-scopen), SUBMITTED-Lifecycle+Heartbeat auf `write_outputs` gaten.
- **MAJOR: keine System-Schreib-Serialisierung; Heartbeat-Kollisionspaar** (G1) — Heartbeat (vom DMS gelesen) ist die exponierteste ungeschützte Datei.

**Entwarnt:** `groupby().last()` (G3, upstream sortiert) und der feature_store-Embargo (G4, inert) — beide kein aktiver Bug. Das **SPY-Urteil bleibt unverändert belastbar**: die load-bearing OOS-Harnesses sind state-isoliert (G2), und kein Befund dieses Nachtrags ändert die Edge-Analyse.

*Ende der Diagnostik (inkl. Nachtrag 2026-06-03). Read-only erstellt; keine Code-Änderungen. Alle Befunde unabhängig zweitgeprüft; 11 Gap-/Blocker-Punkte first-hand verifiziert; Secrets gemäß Rule 20 ohne Wertausgabe behandelt; CI nicht ausgeführt.*

---

## Nachtrag Status 2026-07 (datiert 2026-07-23) — Remediation-Stand der Blocker/Majors

Umsetzungsstand nach GESAMTBEWERTUNG-Paketen P1–P4 (Commits `6a4fd712`, `f7777caf`, 2026-07-22):

- **G5 (drawdown_policy-Shape-Mismatch): CLOSED.** `configs/policy.yaml` hat jetzt
  `drawdown_policy.levels` (soft −10 % / hard −15 % / kill −20 %) im vom Code erwarteten
  Shape; operator-gesetzte DD-Schwellen greifen (Commit `6a4fd712`).
- **V1 (Reconcile-/Circuit-Breaker-Alerts still gedroppt): CLOSED** — fehlende Regeln
  ergänzt/verdrahtet (GESAMTBEWERTUNG P1–P3).
- **G7 (Monitoring-Kill-Switch-Dead-Import): CLOSED** (GESAMTBEWERTUNG P1–P3).
- **V4 (news_alpha-EOD-Pfad funktionslos): TEILREMEDIERT.** Teile des Wirings adressiert;
  der EOD-Pfad ist noch nicht vollständig als funktionsfähig verifiziert — Reststatus offen.

Alle übrigen Befunde dieses Dokuments: Stand 2026-06-03, dort nicht neu bewertet.
Der ursprüngliche Text bleibt unverändert (Audit-Artefakt).
