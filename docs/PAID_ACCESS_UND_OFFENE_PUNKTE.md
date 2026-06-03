# Paid Access & Offene Punkte

**Erstellt:** 2026-06-03
**Quelle:** Triage von `docs/Diagnostik.md` (UMSETZUNG-Modus, gestaffelt).
**Zweck:** Ehrliche Dokumentation aller Befunde, die **NICHT** durch reinen Code-Fix lösbar sind — entweder weil sie eine **externe / kostenpflichtige Datenquelle / einen echten Broker** brauchen (Bucket D), oder weil sie eine **offene Projektentscheidung** sind, die nicht automatisch ausgeführt werden darf (Bucket E).
**Grundsatz:** Kein Fake-Implementieren, kein stilles Akzeptieren, kein Schönreden. Diese Punkte werden hier getrackt, damit wir bewusst entscheiden können.

---

## Bucket D — Braucht externe / paid Datenquelle, echten Broker oder echtes ML-Setup

> Diese Punkte sind **nicht** durch Code-Härtung „echt" lösbar. Ein Fake-Wiring würde nur Dummy-Daten als Produktionsbeweis tarnen (verboten laut CLAUDE.md Datenrealismus-Regel).

### D1 — multifactor_v2: tote Altdata-Faktoren (19–25 von 34 = 0.0)
- **Was:** earnings_surprise_z, insider_cluster_score, buyback_drift_score, sector_rotation_bias, news_sentiment, options_iv_skew, VIX-Faktoren, congress_activity, pead_sue — alle aktuell 0-gewichtet / liefern 0.
- **Wo:** `src/assembled_core/strategies/multifactor_v2.py:578-584` (Regime-Gewichte) und die jeweiligen Faktor-Helper.
- **Warum nicht code-fixbar:** Die Faktoren rechnen korrekt, aber es fehlt **PIT-saubere historische Datenbasis**. Alpaca Free Tier liefert keine Fundamentaldaten; ohne echte Disclosure-/Filing-Historie ist jeder OOS-Test entweder leer oder look-ahead-verseucht.
- **Nötiger paid access:** Sharadar SF1 (Fundamentals + Earnings), QuiverQuant (Congress, Insider), eine Options-IV-Quelle (ORATS/CBOE), eine News-/Sentiment-Historie mit Timestamps ab >2025-12.
- **Erwarteter Nutzen:** Die einzige *spekulative* Hoffnung auf einen Return-Edge im Diagnostik-Urteil. Bis echte Historie existiert, ist die Aktivierung „Hoffnung, kein Beleg" (die eine Aktivierung, die lief, ergab Sharpe-Delta +0.00).

### D2 — Insider- / Shipping-Daten-Feeds (Dummy-Generatoren)
- **Was:** `allow_sample=True`-gated Dummy-Generatoren statt echter Feeds.
- **Wo:** `src/assembled_core/data/insider_ingest.py:88`, `src/assembled_core/data/shipping_routes_ingest.py:88`.
- **Warum nicht code-fixbar:** Es existiert keine echte Datenquelle im Repo; Insider-Daten sind zudem zu 100% `unknown` transaction_type (Faktor liest 0).
- **Nötiger paid access:** EDGAR-Form-4-Scrape (frei, aber Aufbau nötig) / QuiverQuant (Insider), Lloyd's MIU oder MarineTraffic (Shipping).
- **Erwarteter Nutzen:** Alternative-Data-Faktoren mit potenziellem Edge; aktuell reiner Platzhalter. (Code-seitig sind die Dummies bereits korrekt fail-loud gated — kein Bug, nur Daten-Lücke.)

### D3 — Congress-Trading-Daten
- **Was:** `congress_trades_ingest.py` existiert nicht (nur stale `__pycache__`); `trading_cycle_shared.py:625-647` importiert es try/except → `include_congress=True` ist No-op.
- **Wo:** fehlendes Modul; Caller `pipeline/trading_cycle_shared.py:625` (Schutzpfad).
- **Warum nicht code-fixbar:** Keine Datenquelle.
- **Nötiger paid access:** QuiverQuant Congress-Trades API.
- **Erwarteter Nutzen:** Bekannter Alpha-Kandidat aus der Literatur; ohne Feed nicht testbar.

### D4 — Research-only Signal-Module (unwired)
- **Was:** `recession_probability`, `lppls_crash`, `tail_risk_hedge`, `tail_risk_vvix`, `cross_asset_carry(_v2)`, `etf_flows` — implementiert, aber von keiner Live-Pipeline importiert; teils Live-yfinance ohne `as_of`.
- **Wo:** `src/assembled_core/signals/`.
- **Warum nicht code-fixbar:** „Aktivieren" hieße neues Wiring + Datenquellen + OOS-Falsifikation — das ist Research, kein Bugfix. Ein Fake-Wiring wäre eine zweite Wahrheit.
- **Nötiger paid access:** je nach Modul (FRED frei für recession_prob; VVIX/Options-Daten für tail_risk; PIT-Feeds).
- **Erwarteter Nutzen:** Offen. Erst als Research-Item zu bewerten, nicht als Fix.

### D5 — Live-Broker-Routes / OMS-Outbound
- **Was:** `/routes` liefert nur `PAPER`; IBKR/Live auskommentierte Platzhalter; kein konkreter Outbound-Adapter unter `adapters/outbound/`.
- **Wo:** `src/assembled_core/api/routers/oms.py:176-192`, `src/assembled_core/ports/order_router.py:29-36`.
- **Warum nicht code-fixbar:** Braucht echte Broker-Integration (Alpaca-Live/IBKR) inkl. Pre-Trade-Gate-Verzahnung, Idempotency, Kill-Switch-Verzahnung, Reject-Handling.
- **Nötiger access:** Alpaca-Live-Account oder IBKR-Account + API-Keys; bewusster Live-Trading-Entscheid (CLAUDE.md: kein früher Live-Betrieb).
- **Erwarteter Nutzen:** Voraussetzung für echten Live-Betrieb; bis dahin korrekt als „Light/Paper" gekennzeichnet (ehrlich, kein Bug).

### D6 — ML-Stubs (NotImplementedError)
- **Was:** `gnn_signal`, `temporal_fusion_transformer`, `logic_tensor_network`, `differential_privacy` (DP-SGD) — `fit/predict` raisen NotImplementedError.
- **Wo:** `src/assembled_core/ml/`.
- **Warum nicht code-fixbar:** Braucht optionale ML-Libs (pytorch-forecasting, ltn, Opacus) **und** Trainingsdaten + Validierung. Eine Stub-Implementierung wäre Fake.
- **Nötiger access:** GPU/ML-Stack + gelabelte Daten; reine Research-Module.
- **Erwarteter Nutzen:** Forschungstier; aktuell kein Live-Wert. Wichtig: müssen gegen versehentliche Live-Wiring abgesichert bleiben (Startup-Guard) — *das* wäre ein optionaler A-Fix.

### D7 — Quantum-Portfolio (D-Wave QPU)
- **Was:** „research showcase stub / interface stub for D-Wave QPU execution".
- **Wo:** `src/assembled_core/portfolio/quantum_portfolio.py:1,15`.
- **Warum nicht code-fixbar:** Braucht Quantum-Hardware/-SDK.
- **Nötiger access:** D-Wave Leap (paid).
- **Erwarteter Nutzen:** Experimentell; nicht in `__init__` exportiert (kein Live-Risiko).

### D8 — `compute_risk_on_off_indicator` (Platzhalter)
- **Was:** naiver Advance/Decline-Ratio; Name impliziert sektor-bewusste Klassifikation, die nicht implementiert ist.
- **Wo:** `src/assembled_core/features/market_breadth.py:301-310`.
- **Warum teils code-fixbar:** Umbenennen/ehrlich kennzeichnen = **A** (Code). Echte sektor-bewusste Implementierung braucht Sektor-Klassifikationsdaten = **D**.
- **Nötiger access:** GICS/Sektor-Mapping-Quelle.
- **Erwarteter Nutzen:** Cyclical/Defensive-Rotation-Signal.

### D9 — `news_rag`, `polymarket_loader` (externe Services)
- **Was:** vollständig implementiert, aber unwired; `news_rag` importiert `anthropic`/`qdrant` auf Modulebene.
- **Wo:** `src/assembled_core/intel/news_rag.py`, `polymarket_loader.py`.
- **Warum nicht code-fixbar:** Brauchen externe Services/Keys (Anthropic, Qdrant-Vektor-DB, Polymarket-API).
- **Nötiger access:** API-Keys + laufende Vektor-DB.
- **Erwarteter Nutzen:** Research; *optionaler* A-Fix = Modul-Level-Import nach innen ziehen (Latenz/Versionskonflikt-Schutz).

### D10 — QA: CPCV „research-only", SHIP-Heuristik, n_tests=1-Placeholder
- **Was:** echtes CSCV-PBO nicht produktiv; `scenario_engine` identifiziert Shipping-Symbole per Substring „SHIP"; `factor_analysis` DSR mit `n_tests=1` Placeholder.
- **Wo:** `src/assembled_core/qa/cpcv_validation.py:121`, `qa/scenario_engine.py:345-352`, `qa/factor_analysis.py:1904-1912`.
- **Warum teils fixbar:** SHIP-Heuristik braucht echte Shipping-Exposure-Daten (D). Das `n_tests=1`-Update zu verdrahten ist Code (potenziell **A**, erst bestätigen ob Caller patcht). Echtes CSCV-PBO = größeres Feature.
- **Nötiger access:** Shipping-Exposure-Mapping; sonst nur Eng-Aufwand.
- **Erwarteter Nutzen:** Robustere Overfitting-Detektion (CSCV) — relevant fürs Edge-Urteil.

### D11 — feature_store `event_beta`-Producer (toter Pfad)
- **Was:** `compute_event_beta()` gibt immer None zurück — Producer schreibt kein `available_at`, falscher Pfad/Layout.
- **Wo:** `compute_event_betas.py:175-181` (Producer) vs `feature_store.py:184,199,203` (Reader).
- **Warum nicht jetzt fixbar:** Aktivierung braucht echte Event-Beta-Daten + Producer-Rebuild; ein Code-Fix allein erzeugt nur eine leere, aber „grüne" Pipeline.
- **Nötiger access:** Event-Daten mit echten Verfügbarkeitszeitstempeln.
- **Erwarteter Nutzen:** EDCL/Conviction-Beta-Boost; aktuell `beta_boost=0.0` (dokumentiert `docs/edcl/decisions.md:254`). *Optionaler A-Fix:* Producer+Reader-Vertrag angleichen (available_at, Pfad, embargo) — aber ohne Daten bleibt es inert.

### D12 — release-gate CI: synthetischer Walk-Forward-Smoke
- **Was:** `CI-001 SYNTHETIC` Random-Walk (seed=42); statistische Gates non-blocking bis Grace-Date 2026-07-01.
- **Wo:** `.github/workflows/release-gate-ci.yml:73-119` (Schutzpfad).
- **Warum nicht fixbar:** Ein „echtes" Gate setzt einen validierten Strategie-Edge voraus — den es laut Diagnostik nicht gibt. Das Gate kann die echte Strategie nicht zertifizieren, solange kein Edge existiert.
- **Nötiger access:** ein nachgewiesener OOS-Edge (Research), keine Datenquelle.
- **Erwarteter Nutzen:** Echte Release-Schranke statt Smoke; abhängig von D1/Research.

### D13 — Alert-Delivery: tatsächlicher Versand
- **Was:** Das *Wiring* der Alert-Sinks ist ein A-Fix (A10/A11). Der *tatsächliche Versand* (Telegram/E-Mail) braucht Credentials.
- **Wo:** `src/assembled_core/ops/alerting.py:125-147` (Early-Skip ohne Creds), `configs/alerting.yaml`.
- **Warum nicht code-fixbar:** Ohne `TELEGRAM_*` / `ASSEMBLED_SMTP_*` Env-Vars degradiert jeder Pfad zu log-only.
- **Nötiger access:** Telegram-Bot-Token + Chat-ID **oder** SMTP-Zugang.
- **Erwarteter Nutzen:** Operativ zwingend für Go-Live — ein Mensch muss CRITICAL-Alerts erhalten. (Code-Wiring zuerst per A10/A11, dann Creds setzen.)

### D14 — Monitoring-Producer (regime/signals/zombie/correlation)
- **Was:** Dashboard-Endpoints lesen Datei-Patterns, die kein Producer schreibt.
- **Wo:** `src/assembled_core/api/routers/monitoring.py:473,516,554,594`.
- **Warum teils fixbar:** Endpoints **ehrlich als unimplemented markieren** = **A9** (Code). Echte Live-Zahlen brauchen die fehlenden Producer (regime_state/zombie/correlation/signal_scores-Writer) — das ist neues Feature-Wiring, kein Bugfix.
- **Nötiger access:** kein paid; Engineering-Aufwand für Producer.
- **Erwarteter Nutzen:** Echtes Live-Monitoring statt Placeholder.

### D15 — news_alpha / crisis_alpha: Strategie-EDGE (nicht Wiring)
- **Was:** Der **Wiring-Fix** (EOD erzeugt 0 Signale) ist B19/A22 (Code). Der **Edge** (lohnt sich event-getriebenes News-Trading?) ist unbekannt.
- **Wo:** `src/assembled_core/events/news_alpha/`, `events/crisis_alpha/`.
- **Warum nicht fixbar:** News-Daten existieren erst ab ~2025-12 → kein historisches OOS möglich; der Edge ist nicht falsifizierbar, bis genug Historie da ist.
- **Nötiger access:** mehrjährige PIT-News-/Event-Historie.
- **Erwarteter Nutzen:** Der einzige *echt unbekannte* (nicht falsifizierte) Pfad zu Alpha — wert, nach dem Wiring-Fix mit echter Historie zu prüfen.

---

## Bucket E — Offene Projektentscheidung (NICHT automatisch ausführen)

### E1 — `.env`-Key-Rotation (SICHERHEIT, dringend)
- **Was:** `.env` war committet (`0ca19ef0`, 2025-10-05), aus Index entfernt (`e64fa215`, 2026-04-19); **Blob bleibt in der History** auf `main`/`origin/main`/`origin/ERWEITERUNG`, extrahierbar (87 bytes).
- **Warum nicht auto-ausführbar:** Rotation passiert **provider-seitig** (Finnhub/Alpaca/Telegram/…), nicht im Repo. Muss von dir ausgelöst werden.
- **Nötig:** Jeden im historischen `.env` enthaltenen Key beim Provider neu generieren; die in `e64fa215` erwähnte Rotation auf **Vollständigkeit** prüfen (deckte sie alle Keys ab?).
- **Erwarteter Nutzen:** Beseitigt das Kompromittierungs-Risiko. **Bis zur Rotation ist das Risiko aktiv** — `.gitignore` schützt nur künftige Commits. *(Auf Wunsch liste ich die betroffenen Key-NAMEN aus `.env.example` — ohne Werte, Rule 20.)*

### E2 — `.env`-History-Bereinigung
- **Was:** Den `.env`-Blob aus der Git-History entfernen (`git filter-repo` / BFG).
- **Warum nicht auto-ausführbar:** **Destruktiv** — Rewrite + Force-Push auf `origin/main` **und** `origin/ERWEITERUNG`, invalidiert alle Clones/Forks. History-Rewrite erfordert expliziten Auftrag (CLAUDE.md Rule 20).
- **Nötig:** Bewusste Projektentscheidung + Koordination mit allen Clone-Inhabern.
- **Erwarteter Nutzen:** Entfernt den Blob aus der History. **Ersetzt nicht die Rotation** (E1 hat Vorrang) — gemeinsam erst beseitigt das Risiko vollständig.

### E3 — DMS-Daemon im Task Scheduler registrieren
- **Was:** Der Dead-Man-Switch läuft nur, wenn der Daemon gestartet ist; aktuell nicht als Windows-Task registriert.
- **Wo:** Register-Skript existiert: `scripts/ops/register_dms_task.ps1`; Loop in `scripts/dms_daemon.py`.
- **Warum nicht auto-ausführbar:** Registrierung ist eine **Deployment-Aktion auf deinem Host** (Task Scheduler / Admin-Rechte), kein Repo-Code-Fix.
- **Nötig:** `register_dms_task.ps1` auf der Live-Maschine ausführen (von dir) + verifizieren, dass der Task feuert.
- **Erwarteter Nutzen:** Aktiviert das Auto-Flatten-on-stale-Heartbeat-Sicherheitsnetz im Live-Betrieb.

### E4 — `autonome_weiterarbeit/` aus Tracking entfernen
- **Was:** 18 interne Planungs-/Audit-Docs (inkl. COMPETITIVE_ANALYSIS_v4, PAID_DATEN) trotz `.gitignore:95` noch getrackt; in voller History.
- **Warum nicht auto-ausführbar:** Es ist eine **Inhaltsentscheidung**, welche dieser Docs untracked/gelöscht werden sollen (manche evtl. bewusst behalten). `git rm --cached` ist reversibel, aber die Auswahl ist deine.
- **Nötig:** Entscheidung welche Docs raus; dann `git rm --cached -r autonome_weiterarbeit/` (gitignore-Regel existiert bereits).
- **Erwarteter Nutzen:** Interne Strategie-/Wettbewerbs-Docs nicht mehr in künftigen Checkouts (History bleibt — separate Frage analog E2).

### E5 — `requirements.lock`: löschen vs. regenerieren
- **Was:** Stale Lock (2026-04-08), divergiert von `requirements.txt`, von 0 Workflows konsumiert.
- **Warum Entscheidung:** Beide Wege sind valide — entweder löschen (wenn nie genutzt) oder aus `requirements.txt` neu generieren (wenn als Reproduzierbarkeits-Referenz gewollt). Code-fixbar (A30), aber die Richtung ist deine Wahl.
- **Erwarteter Nutzen:** Beseitigt eine irreführende Dead-Truth-Datei.

### E6 — scipy / scikit-learn exakte Pins
- **Was:** `requirements.txt:51-52` lässt scipy/sklearn als Ranges (wegen Py3.10-Support in der CI-Matrix).
- **Warum Entscheidung:** Exakt-Pinnen ist erst sinnvoll, **wenn Python 3.10 fallengelassen wird** — das ist eine Versions-Policy-Entscheidung.
- **Nötig:** Entscheid, Py3.10 aus der `backend-ci.yml`-Matrix zu entfernen; dann exakt pinnen.
- **Erwarteter Nutzen:** Beseitigt die letzte ungepinnte Drift-Fläche in der autoritativen Pin-Datei.

### E7 — news_alpha EOD-Aktivierung
- **Was:** Der EOD-Pfad von news_alpha ist **dormant**: die EOD-Feed-Topic-Strings matchen keinen Routing-`topic_id` (`asset_router.ROUTING_TABLE`), daher droppt `generate_signals()` jedes Item → ZERO Output. Batch 9 hat das **nur sichtbar gemacht** (One-Shot-WARNING in `src/assembled_core/events/news_alpha/signal_generator.py`), nicht behoben.
- **Warum nicht auto-ausführbar:** Das Angleichen der EOD-Feed-Topic-Strings an die Routing-`topic_id`s würde den Pfad **scharfschalten** und **echte direktionale ETF-Trades (live paper, shadow_only=false)** auslösen. Das Aktivieren einer bislang stillen Entry-Logik ist eine **Nutzerentscheidung**, kein Auto-Fix.
- **Wo:** `src/assembled_core/events/news_alpha/signal_generator.py` (Topic→Route-Matching), `asset_router.ROUTING_TABLE` (Soll-`topic_id`s).
- **Nötig:** Entscheidung, ob EOD aktiviert werden soll; falls ja, Mapping der EOD-Feed-Topics auf die Routing-`topic_id`s definieren + separat reviewen (Entry-Pfad, Risk).
- **Erwarteter Nutzen:** Macht den EOD-News-Alpha-Pfad funktionsfähig — bewusst und kontrolliert, nicht als stiller Nebeneffekt.

---

## Zusammenfassung

- **Bucket D (Daten/Broker/ML-Setup nötig):** 15 Punkte — kein Fake-Implementieren. Mehrere haben einen *optionalen Code-Teilfix* (ehrliche Kennzeichnung / Guard / Wiring-Angleichung), der separat als A/B laufen kann; der *Wert* bleibt an externe Voraussetzungen gebunden.
- **Bucket E (Entscheidung/Deployment):** 7 Punkte — **E1 (.env-Key-Rotation) ist sicherheitskritisch und dringend** und sollte vor allem anderen entschieden werden.

*Read-only erstellt; keine Code-Änderungen. Secrets gemäß Rule 20 ohne Wertausgabe behandelt.*
