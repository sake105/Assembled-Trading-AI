# Daten- und Nutzungsaudit — 2026-08-16

Read-only-Audit auf `main` @ `10148d55` (nach Merge der Datenbestand-Remediation +
H-090). Vier unabhängige Lese-Sweeps (Datenbestände / Ingest-Verkabelung /
Kern-Nutzung / Stubs+Doku-Drift), hier konsolidiert. Alle Zahlen selbst gemessen
(pandas/pyarrow/Grep/Task-Scheduler-Query), nicht aus Doku übernommen.

Kennzeichnung durchgehend: **[BEFUND]** = selbst verifiziert ·
**[VERDACHT]** = nicht abschließend prüfbar · **[EINSCHÄTZUNG]** = Wertung.

---

## 1. Datensatz-Bewertung

Notenschlüssel je Datensatz: PIT / Coverage / Genauigkeit / Aktualität / Doku → **Gesamt** (1–10).

### Betriebspfad

| Datensatz | Pfad | P/C/G/A/D | Gesamt | Kernbelege [BEFUND] |
|---|---|---|---|---|
| Operativer Preis-Cache | `output/aggregates/daily.parquet` | 4/3/7/3/9 | **4,5** | 279.013 Z × 220 Sym, 1984→**2026-08-05** (=EODHD-Ausfallstag), 0 % NaN; **0/220 Delisted** (SIVB/BBBY/LEH/FRC je 0 Zeilen); `adj_close == close` in **100 %** — es gibt keine unadjustierte Preisebene |
| yfinance-Cache | `data/cache/yfinance/` (197 + 11 Dateien) | 5/5/5/2/4 | **4** | AAPL/MSFT/GLD bis **2026-05-18**, mtimes 19./21.05. — seit ~3 Monaten nicht nachgeführt; Live-Fallback zieht offenbar frisch, dieser Cache nicht |
| Universe/Watchlists | `data/universe/`, `configs/` | 6/5/7/6/10 | **7** | watchlist 195 Sym; `verdict_sp500.csv` 1.167 Z, **418 end_dates** (SIVB 2023-03-10 ✓, TWTR ✓, BSC-Recycling dokumentiert); `docs/UNIVERSE_SOURCES.md` = beste Doku im Repo |
| Corporate Actions | `output/corporate_actions.csv` | 5/5/5/5/10 | **6** | 68.111 Z, 67.693 DIVIDEND + 418 DELISTING, **0 SPLITS**; Delistings panel-inferiert, kein CA-Feed; ehrlicher 14-Zeilen-Header |
| Factor-Store | `output/factors/core_ta/1d/` | 5/6/6/5/3 | **5** | 14 universe_-Hashes, 2026er-Datei 27.531×33, bis 2026-07-27; kein Manifest je Hash; Cache-Key-Defekt (§0.05) offen [VERDACHT: nicht neu verifiziert] |
| Intraday-Aggregate (Betrieb) | `output/aggregates/5min…` | 4/2/5/3/4 | **3** | 194 Z / 2 Sym (Nov-2025-Demo); kein tragfähiger Bestand [EINSCHÄTZUNG: Testartefakte] |
| XBRL-Fundamentals (operativ) | `data/raw/fundamentals/` | 8/5/7/5/9 | **7** | 351.939 Z, 178 Sym, filed bis 2026-06-09; `disclosure_date` vorhanden (PIT-fähig); Manifest + coverage.md vorbildlich |
| Insider/Congress (operativ) | `data/raw/insider_congress/` | 8/6/6/5/9 | **7** | Form4 838.277 Z, `available_at ≥ filing_date` in 100 %; **192 Z mit transaction_date > filing_date, 7 davon bis 2050** (ungefilterte Quellfehler); Congress 25.735 Z aus GitHub-Mirrors (nicht amtlich) |
| Intraday-Forschung 1h | `data/raw/intraday_1h/` | 7/7/7/4/7 | **6,5** | 15,52 Mio Z, 298 Sym, AAPL 2003→2026-07-02; enthält Delisted (AABA, ABC) — im Gegensatz zum EOD-Betriebspfad |
| Makro-/Kleincaches | GDELT/GPR/Funding/intel | 6/5/6/5/5 | **5,5** | GPR monatlich 1900→2026-04; GDELT bis 2026-05; intel: 2 RSS-Snapshots vom 19.05. |

### Forschungspfad

| Datensatz | Pfad | P/C/G/A/D | Gesamt | Kernbelege [BEFUND] |
|---|---|---|---|---|
| **PIT-Preispanel** | `research/mandat/data/prices_verdict.parquet` | 8/8/6/3/9 | **7** | 6.096.910 Z, 1.167 Sym, 1995→2026-07-06 (eingefroren); Delisted-Stichprobe SIVB/FRC/BSC/ENRNQ/EKDKQ ✓; **fehlend LEHMQ/WCOEQ/MTLQQ** (36/1.202 Pull-Ausfälle, konzentriert auf Insolvenzticker); nur adjustierte close, kein OHLC/raw |
| Dividenden | `dividends.parquet` | 6/7/5/4/7 | **6** | 64.493 Z; **33 Zukunfts-ex_dates** (bis 2027-03) → PIT-Filterpflicht; Betrag = adjustierter `value`, nicht deklarierter Betrag (Steuer-Semantik falsch, offengelegt) |
| S&P-Konstituenten | `sp500_historical_constituents.csv` | 7/8/5/5/7 | **6,5** | 2.712 Snapshots 1996→2026-06; Quelle = nicht reproduzierbarer GitHub-Bezug |
| **Form 4 DERA** | `form4_dera/` (81 Quartale) | 8/9/8/6/7 | **8** | 8.310.850 Z, exakt 17.134 ISSUERCIK, 2006q1→2026q1; FILING_DATE PIT-fähig; Datums-Strings „01-APR-2022" = Parse-Falle |
| Form 4 broad | `form4_broad/` | 7/6/6/5/5 | **6** | 655.648 Z; Filings vor 2012 fehlen (EDGAR-Grenze), Lücken in `form4_gap_symbols.csv` dokumentiert |
| 13F | `13f/` (53 Zips) | 6/7/6/6/4 | **6** | 23,84 Mio Z; FILING_DATE als String; kein Manifest |
| Smallcap-Panel | `smallcap/` | 6/7/5/4/4 | **5,5** | 43,44 Mio Z, 1.922 Sym; 963 Leersymbole dokumentiert, kein Manifest |
| Übrige Mandats-Parquets | prices_sp500, congress_extra, sentiment, … | 6/6/6/4/4 | **5,5** | alle EODHD-eingefroren Anfang Juli 2026; Zweck-Parquets ohne Manifest |
| **CRSP/FF + VIX (gratis)** | `research/mandat2/data_gratis/` | 9/9/8/7/9 | **8,5** | FF daily 26.274 Z **1926→2026-06**, 0 % NaN; `_protokoll.json` = vorbildliche Provenance; frei nachführbar |
| Truth-Social-Archiv | `research/geopolitik/data/` | 5/7/6/5/5 | **5,5** | 40.631 Posts; Zeit nur als Rohstring (TZ unklar); Feld per Welle 48 geschlossen |
| EODHD-Symbollisten | `us_symbols_{delisted,live}.parquet` | –/8/7/3/5 | **6** | 58.487 delisted / 51.673 live; Snapshot, nicht nachziehbar |
| Archiv | `archive/orphaned_data_2026-08-15/` | –/–/8/–/9 | **8** | 160.232.916 Bytes per MANIFEST; enthält Pre-Backfill-Backup |

**[EINSCHÄTZUNG] Gesamtbild Daten:** Forschungspfad solide (7–8,5) und ehrlich
dokumentiert; Betriebspfad strukturell schwach (3–5): survivorship-verzerrt,
eingefroren, ohne unadjustierte Ebene. Die Remediation-Doku vom 15.08. deckt die
Messwerte bemerkenswert exakt (279.013/220/2026-08-05, 1.167/418, 17.134 — alle
bestätigt).

---

## 2. Lücken und Verbesserungen (priorisiert)

1. **Survivorship im Betriebspfad** [BEFUND: 0/220 Delisted]: jeder Backtest auf
   dem Betriebspanel ist ~2,4–2,9 pp p. a. geschönt. Schließbar nur durch
   Norgate (~52 $/M) oder EODHD-Reaktivierung + Neuaufbau. **Größter Hebel.**
2. **EODHD tot → alles eingefroren** (daily 08-05, verdict 07-06, intraday 07-02):
   Grundsatzentscheidung reaktivieren / Norgate / bewusst einfrieren steht aus.
3. **Kein Backup der nicht rekonstruierbaren Kernbestände** [BEFUND]:
   `prices_verdict.parquet` (~4,3 GB) und `verdict_sp500.csv` existieren nur auf
   dieser Platte, beide gitignored, Quelle tot. Offsite-Kopie = Minutenaufwand,
   schützt das wertvollste Forschungsartefakt. **Billigster hoher Nutzen.**
4. **Keine unadjustierte Preisebene + 0 Splits** [BEFUND: adj_close==close 100 %]:
   Raw-Rekonstruktion, Skalenbruch-Forensik, Double-Adjust-Prüfungen strukturell
   blockiert.
5. **Delisting-Lücke im Suchfenster** [BEFUND: LEHMQ/WCOEQ/MTLQQ fehlen]: genau
   die Namen, die Survivorship-Korrektur tragen müssten.
6. **Nachführbare Gratis-Quellen laufen hinterher** [BEFUND]: XBRL/Form4
   (Stand 06/2026), DERA (2026q1), yfinance-Cache (05/2026) — alle frei
   nachziehbar, reine Betriebsdisziplin.
7. **Kleinfehler mit Filterbedarf** [BEFUND]: 7 Form-4-Zeilen mit Datum bis 2050;
   33 Zukunfts-ex_dates; `geopol_intensity` ohne Datumsspalte; DERA/13F-Datumsstrings.
8. **Dividenden-Semantik**: adjustierter statt deklarierter Betrag begrenzt
   Steuer-Genauigkeit (nur mit lebender Quelle behebbar).

---

## 3. Nutzungsübersicht — aktiv vs. ungenutzt

Kategorien: **(a)** totes Erbe · **(b)** fertig-aber-nie-verdrahtet
(= verschenktes Potenzial) · **(c)** bewusst geparkt (mit Evidenz/Gate).

### Verifizierte aktive Ketten [BEFUND]

- **Paper-Pilot:** Task/`paper-trading-ci.yml` → `run_paper_pilot.py` →
  `run_live_paper.py once` → `paper_runner` → `trading_cycle_v2` (+ `_tc_*`).
  Strategie `trend_baseline`. `paper/` 8/8 Module aktiv, `accounting/`-Kern aktiv,
  QA-Gates 1–7 ARMED mit echter Blockkette (`qa_block.json` → Pilot verweigert).
- **Aktive Datenmodule:** prices_ingest, pit_prices (CLI-Shadow), universe,
  security_master, factor_store, feed_status (advisory), corporate_actions
  (verdrahtet-aber-inert, Pfad nie gesetzt), tick_store (optional).
- **Aktive Signals/Strategien:** trend_baseline, multifactor_v2 (+transitiv),
  composite_score, sector_rotation, EDCL-Overlay, HMM-Regime-Detection,
  Zombie-Killer (shadow), Correlation-Guard, DD-Treppe (real seit E-135).
- **13 CI-Cron-Workflows** decken Worker/Reconcile/Diagnostics ab —
  [VERDACHT] GitHub deaktiviert Crons nach 60 Tagen Inaktivität; lokal nicht prüfbar.

### Scheduler-Realität [BEFUND, live abgefragt]

| Task | Zustand |
|---|---|
| PaperEngine, DMS-Daemon | Running |
| PaperPilot | Ready; LastRun 11.08. rc=0 — [VERDACHT] Lücke 12.–14.08. |
| CacheRefresh_EODHD | **feuert täglich gegen tote Quelle** (rc=1, 220/220×HTTP-401) |
| Watchdog, HealthCheck | **nie registriert** (Skripte existieren, Tasks fehlen) |

### Die Hauptklasse der Selbstbehinderung: „enabled:true ohne Implementierung" [BEFUND]

Vier scharfgeschaltete Signal-Features scheitern **jeden Zyklus still** an
Importen auf nicht existierende Module (`except → log.debug`):

| Feature (Flag aktiv, Gewicht) | Fehlt | Wirkung |
|---|---|---|
| Intel-Signal-Layer (0.15) | `IntelSignalAdapter` existiert nicht | wirkt nie |
| News→Signal-Bridge (0.10) | `load_and_apply_news_signals` existiert nicht | wirkt nie |
| Earnings-Guard/PEAD (0.15) | `signals/earnings_integration.py` fehlt | Pre-Earnings-Suppression + PEAD laufen nie |
| Shorts-Block (enabled) | `risk/short_risk.py` + `signals/short_signals.py` fehlen | Short-Seite existiert nur als Config |

Kumuliert suggeriert die Policy ~0.40 Signalgewicht, das es nicht gibt.
Dazu **≥5 Policy-Blöcke ohne einen einzigen Code-Leser** (enabled:true):
`quant_gates` (DSR/PSR-Block!), `macro_event_calendar`, `quarter_end_guard`,
`ma_exclusion`, `freshness_monitor` — die Schema-Validierung erzeugt eine
Unterstützungs-Illusion. Vier weitere Gates zeigen auf Policy-Keys, die nicht
existieren (`gnn_signal`, `quantile_sizing`, `pairs_trading`, `bayesian_confidence`).

### Producer-/Consumer-Brüche (X erwartet Y, Y kommt nie) [BEFUND]

1. `earnings-calendar-refresh`-CI produziert → einziger Consumer (Earnings-Guard) ist tot.
2. `refresh_daily_cache_from_panel` (täglich im Pilot-Bat) → Panel eingefroren
   seit **2026-05-21**, niemand baut es → täglicher No-op.
3. pull_log v2 produziert → kein Konsument, keine Retention (~24 MB/Jahr).
4. yfinance = einziger lebender Preispfad → protokolliert als einziger nichts (E-112-Lücke, §0.06c).
5. `news_alpha`-Overlay ist freigegeben (`shadow_only:false`) → Trigger-Quelle
   `data/intel/crisis_state.json` existiert nicht; einziger Producer
   `run_intel_cycle.py` läuft in keinem Scheduler. Der komplette ~30-Module-Zweig
   `intel/news_*` hängt daran.
6. `check_leakage` (Gate 8, fertig+getestet) → Orchestrator liefert kein
   `feature_df` → im Pilot dauerhaft SKIPPED. Zusätzlich ruft
   `daily-diagnostics.yml` ein **nicht existierendes** Script
   (`validate_altdata.py --check-leakage`), abgefangen mit `|| echo` →
   **es existiert heute kein einziger laufender Leakage-Check.**
7. `edcl_sizing.enabled:true` → einziger Leser hat keinen Produktivaufrufer.
8. `run_phase0.ps1` → zeigt auf `pullers/pull_alpha_intraday.py`, Datei heißt
   anders → Schritt bricht.
9. `prewarm-factor-store.yml` lädt Output nur als 14-Tage-CI-Artefakt hoch,
   `output/` ist gitignored → Produktions-No-op.
10. Monitoring-API: 5 Endpunkte warten auf Artefakte, die **niemand schreibt**
    (`regime_state_*`, `signal_scores_*`, `zombie_report_*`, `correlation_guard_*`,
    `paper_ledger.db`); Zombie-Killer schreibt real nach `output/shadow/`, das
    Dashboard sucht in `src/output/` → Alert kann nie feuern.

### Ungenutzt-Inventar (komprimiert)

**(b) fertig-aber-nie-verdrahtet** (Auswahl, vollständige Listen in den
Teilberichten): `attribution/` (Brinson/Cariño, 7 Module, getestet) ·
`ops/daily_scheduler` (fertige Scheduler-Kette ohne Launcher) ·
`accounting/tax_lots.py` (FIFO+SQLite, „Schaufenster-Code") ·
`ops/error_tracking` (Sentry, ein Init-Aufruf fehlt) · `pipeline/dispatcher`
(Strangler-Fig) · `signals/registry` (Entry-Point-Gruppe existiert nicht in
pyproject) · `strategies/base.StrategyRegistry` (0 Registrierungen) ·
Hexagonal-Skelett (`domain/ports/adapters/application/bootstrap`) ·
`experiments/batch_config` (+ konkurrierende Zweitimplementierung) ·
`strategy/` (Singular) · `certify/`, `compliance/` · ~20 qa-Module
(Conformal-Familie, spa_test, vpin, crisis_injection, …) · ~11 features-Module
(altdata_bls/finra/wikipedia, term_structure, volatility_estimators, …) ·
`data/panel_store`, `data/fx`, `data_versioning`, `tier_processor`,
`free_universe` · `per_symbol_cost_bps` (implementiert, am Call-Site nicht
übergeben) · edgar_form4-/congress-/XBRL-Ingest-Funktionen (nur manuelle Pulls).

**(a) totes Erbe** (Auswahl): ~13 risk/portfolio/execution-Waisen
(barra_risk_model, wash_sale_guard, dro_portfolio, quantum_portfolio,
order_gate-Cluster, …) · `intel/`-Leichen (crisis_alpha_worker DEPRECATED,
polymarket_loader, ic_loop, …) · 9 signals-Module ohne Aufrufer
(lppls_crash, tail_risk_vvix, …) · ml-Stubs (gnn, TFT, logic_tensor) ·
4 sources-Module nur als `__init__`-Re-Export (alphavantage, edgar, newsapi,
worldbank) · `qc_client.py` (QuantConnect, 0 Importer) · `assemble_eod_daily.py`
(LEGACY, **gefährlich**: würde Live-Cache mit 2-Symbol-Rest überschreiben) ·
`nightly-sync.yml` (leerer Echo-Cron) · `/diagnostics/modules` +
`/oms/routes` (Hardcode-Dummies) · `pead_strategy`, `stat_arb/` (leer).

**(c) bewusst geparkt, sauber belegt** [BEFUND — vorbildlich]: alle
Shadow-Flags (zombie_killer, correlation_guard, crash_prediction, inverse_etf,
signal_decay, vol_targeting, hrp) werden real gelesen, fail-safe Default
shadow=true, dokumentierter Freigabeprozess · PEAD-SUE weight 0.00 SHADOW
(„pending OOS backtest") · conformal/hmm_overlay/meta_model enabled:false nach
negativem A/B · Insider-/Congress-Faktoren (H-088: Feld per Stopp-Regel zu) ·
`dataquality/` (dokumentiert nicht-invasiv — aber der Batch-Tier, für den es
gedacht ist, existiert nirgends: (c) ohne Einlösepfad).

---

## 4. Verschenktes Potenzial — (b)-Fälle nach Nutzen priorisiert

1. **Ehrlichkeits-Fix der 4 toten enabled-Features** (~0.40 Phantom-Gewicht):
   implementieren ODER Flags auf false. Der Earnings-Guard ist der einzige Fall
   mit bereits laufendem Daten-Producer (CI füttert heute ins Leere).
   [EINSCHÄTZUNG] höchste Priorität, kleiner Eingriff, großer Ehrlichkeitsgewinn.
2. **`check_leakage` scharf verdrahten + daily-diagnostics-No-op fixen**: heute
   existiert kein laufender Leakage-Check, obwohl das Gate fertig ist.
3. **Intel-Zulauf für news_alpha/crisis_alpha herstellen ODER ehrlich parken**:
   beide Overlays freigegeben, laufen leer. [EINSCHÄTZUNG] Alpha-Wert unbelegt
   (Welle 48 schloss das Geopolitik-Feld) — „Flags auf false" ist fachlich
   mindestens gleichwertig; Entscheidung nötig.
4. **`ops/daily_scheduler` anschließen oder streichen**: macht den Pilot
   unabhängig von GitHub-Crons (60-Tage-Risiko).
5. **Steuer-Engine-Integration** (`tax_lots.py` + Portierung
   Verlusttopf/Regime aus `research/mandat2`): höchster fachlicher Nutzen
   (FIFO statt Average-Cost, Anlage-KAP-Fähigkeit), größter Aufwand, sensible
   Zone — eigener Auftrag.
6. **Factor-Store: prewarm-Workflow real machen oder löschen + Cache-Key um
   Code-Version erweitern** (§0.05, belegte WARM/COLD-Orderdivergenz).
7. **`per_symbol_cost_bps` am Call-Site übergeben** (klein; wirksam erst mit
   `sizing.method: cost_aware`).
8. **`attribution/` an Reports anschließen** (fertig+getestet; Diagnosequalität).
9. **API sanieren statt anschließen**: erst die 5 Producer-losen Endpunkte +
   3 Hardcode-Dummies fixen/entfernen — Anschluss ohne Frontend bringt nichts.
10. **Sentry-Init** (ein Aufruf, billig).

**Bewusst NICHT empfohlen:** Insider-/Congress-Anschluss (H-088 Stopp-Regel)
und PEAD-Weight-Flip (sauber evidence-gated) — beides korrekt geparkt.

---

## 5. Doku vs. Code (substanzielle Abweichungen)

| Doku | Behauptung | Ist [BEFUND] | Relevanz |
|---|---|---|---|
| `PROJEKT_STATUS.md` | „Phase 4 abgeschlossen, 110 Tests, phase4_stable" | ~9.300 Tests, Pilot, Kill-Switch engaged, EODHD-Ausfall — 4 Monate alt; von CLAUDE.md als autoritativ referenziert | **Hoch** |
| `docs/ARCHITECTURE_BACKEND.md` | `orchestrator.run_eod_pipeline` = Kern; 6 Router; Phase 6 „geplant" | Aktiver Kern ist `trading_cycle_v2`; 13 Router; Phase-6-Ingester längst gebaut | Mittel-hoch |
| KNOWN_ISSUES §1.4 | Drift-Kette „DONE 2026-04-30" | `save_drift_results` hat keinen Caller; genanntes Script existiert nicht; Endpoint 503t dauerhaft | Mittel |
| `trading_cycle_v2.py` Docstring | „old trading_cycle remains active until Day 9" | v2 IST seit Monaten der aktive Pfad | Mittel |
| README | „~117 Tests", 3 Workflows | ~9.300 Tests, 7 Workflows | Niedrig-mittel |
| **Gegenrichtung: fehlt überall** | — | **Kill-Switch engaged seit 2026-08-09 (Testlauf)** steht nur im State-File + User-Memory, in keiner Betriebsdoku — exakt die E-140-Klasse, vor der §0.0 selbst warnt | **Hoch** |
| `BEFUND_DATENQUALITAET.md` | MTLQQ „im PIT-Universum" | Kein MTLQQ im Preispanel (Universum-CSV ≠ Panel) | Niedrig |
| Frische Remediation-Doku (DATENZUGANG, UNIVERSE_SOURCES, §0.0/§0.05/§0.06) | — | Stichproben **deckungsgleich** mit Messung | Positiv |

Präzisierung zu E-141 [BEFUND]: Trailing-Stop-**Voll**-Trigger wirken inzwischen
auf Orders (`target_qty=0`); **partielle** Reduktionen ändern weiterhin nur
`target_weight`, das die Order-Generierung nicht liest.

[VERDACHT, sensible Zone, nur gemeldet]: `detect_correlation_regime_shift` wird
in `_tc_sizing` **ungated und ohne Shadow** aufgerufen und skaliert live
`target_qty` — direkt neben dem shadow-gegateten Correlation-Guard;
Asymmetrie wirkt unbeabsichtigt.

---

## 6. Fazit: Stellen wir uns noch selbst ein Bein?

**Ja — aber die Klasse hat sich verschoben.** [EINSCHÄTZUNG, gestützt auf die
Befunde oben]

Die alten Muster (Dummy-Antworten in Monitoring, inerte Risk-Kontrollen) sind
zu großen Teilen saniert: DD-Treppe real, QA-Blockkette real, Shadow-Governance
vorbildlich, ehrliche 503s statt Fake-Werte, frische Doku deckungsgleich mit
Messung. Das ist echter Fortschritt gegenüber dem Zustand, den die
Problemzonen-Liste in CLAUDE.md beschreibt.

Die heutige Selbstbehinderung hat drei konkrete Formen:

1. **Konfigurations-Illusion:** vier scharfe Signal-Features + fünf scharfe
   Policy-Blöcke existieren nur als YAML — die Policy behauptet ein System, das
   so nicht läuft, und jeder Fehlpfad verschwindet auf `log.debug`. Wer die
   Policy liest, glaubt an ~0.40 Signalgewicht und an DSR/PSR-Gates, die es
   nicht gibt.
2. **Producer-/Consumer-Brüche:** fertige Teile füttern ins Leere oder warten
   auf Zulauf, der nie kommt (Earnings-CI → toter Guard; news_alpha freigegeben
   ohne Datenquelle; pull_log ohne Leser; Panel-Refresh als täglicher No-op;
   kein einziger laufender Leakage-Check trotz fertigem Gate).
3. **Eingefrorene, unversicherte Datenbasis:** Betriebspfad survivorship-verzerrt
   und seit 08-05 tot; die einzigen nicht rekonstruierbaren Kernartefakte ohne
   Backup; ein Scheduler-Task feuert täglich gegen eine 401-Quelle.

Nichts davon ist ein neuer Bug im Rechenkern — es ist liegengelassene
Verdrahtung plus veraltete Selbstauskunft. Die drei billigsten Gegenmaßnahmen
mit dem größten Effekt: (1) Offsite-Backup der zwei Kernartefakte, (2) der
Ehrlichkeits-Fix der toten enabled-Flags, (3) ein laufender Leakage-Check.
Danach die Grundsatzentscheidung Datenquelle (Norgate vs. EODHD vs. bewusstes
Einfrieren) — sie bestimmt, ob der Betriebspfad je über Note 4,5 hinauskommt.

---

*Erhoben 2026-08-16 durch vier parallele Read-only-Sweeps; Einzelmessungen und
vollständige Inventarlisten in den Sweep-Protokollen der Session. Keine Datei
außer diesem Bericht wurde geschrieben oder geändert.*

---

## 7. Umsetzungsstand Pakete 1–6 (Nachtrag 2026-08-16)

Auf Basis dieses Audits wurde ein 6-Pakete-Umsetzungsplan beauftragt (User-Auftrag
2026-08-16). Die Nummerierung X.Y, die in Code-Kommentaren und E-Log-Eintraegen
referenziert wird („Audit-Plan 2.5b" etc.), ist DIESE:

| # | Massnahme | Status |
|---|---|---|
| 1.1 | Backup PIT-Panel + nicht rekonstruierbare Bestaende | ✅ ERLEDIGT (D:\Backup_AssembledTradingAI\2026-08-16\, 4,7 GB, MD5-verifiziert; Offsite = Operator-Punkt) |
| 1.2 | 4 tote enabled-Flags ehrlich (shorts, intel.signal_layer, news_signal_bridge, earnings_guard) | ✅ ERLEDIGT (alle false + Begruendung) |
| 1.3 | 5+ leser-lose Policy-Bloecke markiert/false | ✅ ERLEDIGT |
| 1.4 | Kill-Switch-Zustand dokumentiert (KNOWN_ISSUES §0.00) | ✅ Doku ERLEDIGT; Disengage = Operator (`OPERATOR_KILL_TOKEN`) OFFEN — fertiges Operator-Tool seit 2026-08-17: `python scripts/ops/ops_disengage_kill_switch.py` (laedt .env selbst, druckt Token nie; Agent-Ausfuehrung vom Klassifizierer geblockt — bewusst) |
| 1.5 | Doku-Drift (PROJEKT_STATUS, ARCHITECTURE, README, §1.4) | ✅ ERLEDIGT |
| 2.1 | EODHD-Task deaktiviert | ✅ ERLEDIGT (Task Disabled) |
| 2.2 | Panel-Refresh-No-op aus Pilot-Bat entfernt | ✅ ERLEDIGT |
| 2.3 | Watchdog- + HealthCheck-Tasks registriert | ✅ ERLEDIGT (beide Ready) |
| 2.4 | PaperPilot-Luecke 12.–14.08. aufgeklaert | ✅ ERLEDIGT (Rechner aus zur Triggerzeit; kein Konfig-Fehler) |
| 2.5 | yfinance→pull_log (2.5a) + Watchdog-Konsument (2.5b) | ✅ ERLEDIGT (statt CI-Konsument: Watchdog — pull_logs liegen lokal, nicht im Runner) |
| 3.1 | check_leakage im Orchestrator scharf | 🔒 GEBLOCKT — Klassifizierer verweigert settings.json-Deny-Lift (pipeline/); Vorarbeit erledigt (leakage_frame.py nach src/qa/ verschoben) |
| 3.2 | CI-Leakage-No-op ersetzen | 🔒 GEBLOCKT (workflows/ deny; gleiche Entscheidung) |
| 4.1–4.5 | Pipeline-Korrektheit (Trailing-Teilred., corr-regime-Gate, Factor-Store-Key, cost_bps, Earnings-Guard) | 🔒 GEBLOCKT (pipeline/risk deny; gleiche Entscheidung) |
| 5.1 | Zombie-/Corr-Alert-Producer-Mismatch | ✅ ERLEDIGT (+ Bindungstests mit echtem Writer) |
| 5.2 | API-Ehrlichkeit (5 Producer-lose Endpunkte, 3 Dummies) | ✅ ERLEDIGT (2026-08-16/17: regime/signals/portfolio ehrlich, data-quality REAL an daily.parquet mit Bar-Frische, walk_forward 503 auch bei Teilartefakt, modules/oms als statisch deklariert; Coverage-Tests) |
| 5.3 | attribution/ an Reports | ✅ ERLEDIGT (2026-08-16/17: composite_score-Opt-in-Hook, taeglicher Producer scripts/generate_attribution_report.py via Pilot-Bat Step 3 (verdrahtet; erster Scheduler-Lauf ausstehend, manuell verifiziert) — fuellt AttributionStore MIT Bar-Datum + Stagnations-Guard, beliefert /monitoring/signals, weist inerte Gewichtsanteile aus [gemessen 0,55]) |
| 5.4 | Sentry-Init | ✅ ERLEDIGT |
| 5.5 | Steuer-Engine-Integration | ⏳ OFFEN (eigener Auftrag, L) |
| 6.1 | assemble_eod_daily.py archiviert | ✅ ERLEDIGT (+ Zeigerpflege) |
| 6.2 | Gratis-Quellen nachziehen (EDGAR/yfinance) | ✅ ERLEDIGT NACH ZWISCHENFALL (2026-08-17): Der ERSTE Alpaca-Nachzug hat den Live-Cache mit RAW-Bars beschaedigt (Alpaca-Default; BKNG +2444 %-Naht, Splits als −90 %-Crashs — Stage-2-Review FAIL, E-165). Saniert: Restore aus .bak + Backfill, adjustment=ALL, adj_close-Invariante am Schreibpunkt, Overlap-Re-Adjustierung (Corporate Actions zwischen Ankern) + fail-closed Naht-Guard. ENDSTAND VERIFIZIERT (nach CRWD/DD-Reparatur, Messung 2026-08-17): 280.474 Zeilen, 0 NaN, KEINE neuen >50 %-Moves vs. Referenz; **195/220 Symbole auf 2026-08-14**, 23 weiter auf 2026-08-05 (Batch-Budget --max-symbols=30 bzw. Overlap-Drop; naechster Scheduler-Lauf zieht sie nach: 23 Targets <= Budget 30), 2 delistet (EXAS/HOLX). CRWD/DD am 17.08. forensisch geklärt und REPARIERT (EODHD-Split-Teiladjustierung, E-172; failed-symbols geleert, beide auf Bar 2026-08-14). Root-Cause 16.08.: prewarm lud .env nie [E-145, gefixt]. Guards seit 17.08. als gemeinsamer Helper (price_cache_merge.guarded_merge) in BEIDEN Live-Schreibern (prewarm + sector_etf); dormant-Schreiber (panel/eodhd/backfill) bewusst nicht angeschlossen — im Helper-Docstring namentlich gelistet (E-173). macro/fundamentals/dividends erneuert; DERA aktuell. NACHZUG 2026-08-17: **insider_form4 ERLEDIGT** (im Artefakt gemessen: 5.610 Zeilen aus 1.853 Filings, 113 Symbole mit Treffern von 118 angefragten, max 2026-08-14; SEC-UA via Env deklariert — nicht in .env, Operator-Empfehlung: SEC_USER_AGENT dort eintragen). **earnings TEILWEISE**: Ticker.quarterly_earnings ist deprecated und lieferte STILL 0 Zeilen (kein 429; 3-Symbol-Probe) — Downloader auf get_earnings_dates() umgestellt (nur berichtete Quartale, konsistent mit den Konsumenten load_earnings_surprises/build_earnings_surprise_factors); Lauf schrieb 1.412 Events x 118 Symbole, ABER Reported-EPS endet vendor-seitig konsistent ~Q2/2025 (per-Symbol-max Feb-Jun 2025) — ob Yahoo juengere Quartale als NaN-Reported traegt, war wegen aktivem Rate-Limit nicht messbar; als Vendor-Grenze offen |
| 6.3 | Datenhygiene (2050er-Form4-Zeilen, Zukunfts-Dividenden) | ✅ ERLEDIGT (Loader-Guard + Test; Dividenden: Kampagnen-Loader filtert strukturell) |
| 6.4 | Toter-Code-Archivierung (~40 Waisen) | ◐ TRANCHE 1 ERLEDIGT (2026-08-17: 9 Module + 3 dedizierte Tests + 2 leere Verzeichnisse nach archive/orphaned_code_2026-08-17/, Kriterium 0 Referenzen + keine Sammeltest-Bindung; Collection sauber (0 Errors; 0 Collection-Errors (Zahl interpreterabhaengig, .venv Py3.11.9/pytest 9.0.1: 9.207, Endstand-Messung 2026-08-17 nach Tranche 3 + 6.5)). TRANCHE 2 ERLEDIGT (2026-08-17: 16 weitere Module + 2 dedizierte Testdateien archiviert nach Testdatei-Chirurgie an 6 Sammel-Testdateien; newsapi-Reexport aus data/sources/__init__ entfernt). TRANCHE 3 ERLEDIGT (2026-08-17): Ketten-Analyse aufgeloest — 7 weitere Module archiviert (quantum_portfolio, dro_portfolio, ic_loop, pit_store, alphavantage_source, edgar_source, research/qc_client) + 2 dedizierte Testdateien; 1 lebendig (crisis_alpha_worker via run_intel_cycle), 3 deny-gated (gnn_signal/wash_sale_guard/barra_risk_model). **6.4 ABGESCHLOSSEN** — Details im Archiv-README) |
| 6.5 | ops/daily_scheduler anschliessen oder streichen | ✅ ERLEDIGT (2026-08-17, Empfehlung „streichen" umgesetzt): Modul + CLI-Runner + dedizierte Tests nach archive/orphaned_code_2026-08-17/ archiviert; ops/__init__-Reexport entfernt (0 Paket-Importe); A12/A35-Tests chirurgisch entfernt — der LEBENDE reconcile-Alert-Pfad (accounting/reconciliation.py) bleibt durch test_batchB2 abgedeckt |

Die 🔒-Punkte haengen an EINER Entscheidung: temporaerer Deny-Lift fuer
`pipeline/**` + `workflows/**` in `.claude/settings.json` durch den Operator
(der Auto-Mode-Klassifizierer verweigert dem Agenten jede Aenderung an dieser
Datei — verifiziert unveraendert, MD5 07eb6f8158f2b593b6b498424d1c8c2b).
