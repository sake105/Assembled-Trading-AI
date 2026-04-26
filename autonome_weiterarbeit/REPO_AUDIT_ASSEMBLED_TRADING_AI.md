# Repo-Audit — Assembled-Trading-AI (Stand 24.04.2026)

**Zweck:** Tiefen-Analyse des Repository-Stands im Vergleich zum Playbook-Paket (Rang 1-10). Für jeden Rang: was ist umgesetzt, was fehlt, was ist anders als geplant, und was ist die konkrete nächste Aktion.

**Methode:** Direktes Lesen von `README.md`, `pyproject.toml`, `PROJEKT_STATUS.md`, `KNOWN_ISSUES.md`, `CLAUDE.md`, `docs/PAPER_TRACK_PLAYBOOK.md`, `docs/STRATEGY_POLICY.md`. Strukturelle Analyse der Verzeichnis-Hierarchie (nicht jeder Code-Pfad).

**Baseline:** 737 Commits, Python 3.10-3.12, 13 Optional-Extras, Phasen 4-13 (Test-Marker), dokumentierter Status "Phase 12.3 abgeschlossen".

---

## Executive Summary — die drei Wahrheiten

### Wahrheit 1: Du bist technisch deutlich weiter als erwartet

Model Governance, Pre-Trade-Checks, Kill-Switch, OMS-Light, Factor-Store mit PIT-Safety, Experiment-Tracking (Sprint 12.2), Paper-Track-Runner, Deflated-Sharpe-Gate, Regime-Detection — das alles **existiert** und ist **getestet**. Die Strategy-Policy-Datei (`docs/STRATEGY_POLICY.md`) ist so konkret und nüchtern, wie sie in institutionellen Kontexten aussieht.

### Wahrheit 2: Die `CLAUDE.md` ist ein bemerkenswertes Dokument

"Plan ist nicht Implementierung", "Branch- und CI-Disziplin ist Teil der Architektur", "Keine falsche Sicherheit", "Backtest = Replay, nicht Parallelwelt". Das ist **richtig**. Nicht nur Policy-Sprech, sondern praktisch operationalisierte Disziplin. Wer auch immer diese Datei geschrieben hat (wohl iterativ du + Claude Code), hat verstanden, wie man ein System **nicht** kippen lässt.

### Wahrheit 3: Trotzdem gibt es strukturelle Brüche, die das System gefährden

- `config/` **und** `configs/` parallel
- `datensammlungen/altdaten/stand 3-12-2025/1d/` mit Leerzeichen und Datum im Pfad
- `autonome weiterarbeit/` mit Leerzeichen
- Root überladen (14+ Metadaten-Files)
- Phasen-Marker in `pytest.ini` nur `phase10-phase13` — `phase4`, `phase6`, `phase8`, `phase9` werden im README beschrieben aber **fehlen in der pytest-Config**
- `pyproject.toml` mit 13 Optional-Extras, davon `all` als Union, aber `system_check` nicht in `all` enthalten
- Keine Versions-Pins auf exakte Builds — nur Floors (`>=`)
- `PROJEKT_STATUS.md` ist Stand **2025-01-15** und nennt nur Phase 4 als abgeschlossen, während README Phase 4-12 als fertig markiert
- `KNOWN_ISSUES.md` ist Stand **2025-01-15** und beschreibt "Phase 12.3 – Review-Vorbereitung"

**Die Diskrepanz zwischen Status-Dateien und README-Status ist selbst ein Symptom.** Du dokumentierst Stand, verlierst aber den Überblick, was genau aktuell ist.

---

## Rang-für-Rang-Analyse

### Rang 1 — Migrations-Playbook (`60_MIGRATION_PLAYBOOK.md`)

**Status im Repo:** 🟡 Teilweise umgesetzt, aber Drift entstanden.

**Was funktioniert:**
- `src/assembled_core/` mit Domain-Modulen (data, features, signals, portfolio, execution, qa, pipeline, api)
- `scripts/` als Entry-Points
- `tests/` strukturiert
- `.github/workflows/` mit Ubuntu/Windows-CI
- `src`-Layout in `pyproject.toml` (`package-dir = {"" = "src"}`)
- `.pre-commit-config.yaml`, `.gitleaks.toml`, `.secrets.baseline` aktiv
- `CLAUDE.md` als Governance-Anker

**Was fehlt:**
- **Die Migration ist nicht abgeschlossen.** `KNOWN_ISSUES.md` sagt explizit: "Legacy-Skripte (z.B. `sprint9_dashboard.ps1`, `sprint9_cost_grid.ps1`, `sprint10_param_sweep.ps1`) sind noch vorhanden, aber nicht in die neue Core-Architektur migriert. Status: TODO: Phase 5/6." → Das ist **ein Jahr alt**. Es liegt nicht an der Migration-Komplexität, sondern daran, dass Feature-Work Vorrang hatte.
- **Zwei Config-Ordner:** `config/` und `configs/`. `STRATEGY_POLICY.md` referenziert `configs/policy.yaml`, `configs/paper_track/*.yaml`, aber `config/factor_bundles/*.yaml`. **Das ist ein aktiver Bug-Magnet.**
- **Phase-Naming in Tests:** `pytest.ini` hat `phase10, phase11, phase12, phase13`. README beschreibt `phase4, phase6, phase8, phase9`. Wenn du `pytest -m phase4` aufrufst, **wirft pytest warning "Unknown marker phase4"** — die Tests laufen zwar, aber als "unknown marker".
- **Root-Dateien:** `missing_symbols.txt`, `news_blacklist.yaml`, `news_whitelist.yaml`, `watchlist.txt`, `watchlist_full.txt`, `version.manifest.json`, `FPython_ProjektAktiengerüst__profile_out.txt` — das sind 7 Dateien, die in `configs/` oder `data/` gehören.
- **`FPython_ProjektAktiengerüst__profile_out.txt`:** Encoding-kaputt (das F ist ein Sonderzeichen, vermutlich ein Unicode-Artefakt). Das ist ein Profiling-Output, der versehentlich committed wurde. **Weg damit.**
- **`000_UpgradeToPS7.ps1`:** Ein PowerShell-Upgrade-Skript im Root-Verzeichnis. Gehört nach `scripts/ops/` oder in die README als Text-Anweisung.

**Konkrete Aktion:**
1. **Config-Vereinheitlichung:** Entscheide zwischen `config/` und `configs/`. Plural (`configs/`) ist gebräuchlicher. Dann Migration aller YAMLs. Grep auf alle Referenzen. Merge-Commit.
2. **pytest-Marker vollständig deklarieren:** `phase4, phase6, phase7, phase8, phase9` in `pyproject.toml` ergänzen unter `[tool.pytest.ini_options].markers`. Sonst werden deine Test-Runs laut — und irgendwann stumpf.
3. **Root-Cleanup:** Alle Data-Listen (Watchlists, Blacklists, Symbole) nach `configs/data/`. Version-Manifest nach `configs/`. Profiling-Output löschen. PowerShell-Upgrade nach `scripts/ops/`.
4. **Legacy-Skripte:** Entweder migrieren oder komplett löschen (aus Git-History ist nicht nötig, aus Working-Tree reicht). "Legacy noch drin, aber offiziell abgelegt" ist die schlechteste Variante, weil es Neueinsteiger verwirrt.
5. **Deutsche Ordner raus:** `datensammlungen/altdaten/stand 3-12-2025/1d/` → `data/archive/2025-12-03/1d/` und `autonome weiterarbeit/` → entweder `docs/autonomous_tasks/` oder `backlog/`.

**Aufwand:** 6-10 Stunden für den strukturellen Cleanup. Primär mechanische Arbeit, aber die Grep-Updates über Configs brauchen Sorgfalt.

---

### Rang 2 — News-Ground-Truth (`34_NEWS_GROUND_TRUTH.md`)

**Status im Repo:** 🟡 Infrastructure vorhanden, aber Ground-Truth-Set wahrscheinlich nicht gebaut.

**Was funktioniert:**
- `newsapi-python>=0.2.7` in Runtime-Deps
- `edgartools>=2.0.0` in Runtime-Deps
- `feedparser>=6.0.0` in Runtime-Deps
- `news_blacklist.yaml` + `news_whitelist.yaml` im Root (Regel-basierte Filter)
- README erwähnt `news_geo`, `geo_score`, `geo_confidence` als Signal-Quellen
- Phase-6-Feature: "Event-Strategie nutzt Insider-Trading + Shipping" (im README)
- `STRATEGY_POLICY.md` definiert News-Health-Gate (WATCH-only bei DEGRADED)

**Was fehlt:**
- **Ein Ground-Truth-Set von 100+ manuell gelabelten News-Events gibt es vermutlich nicht.** Das Playbook (Datei 34) fordert ein Labelling-Set, auf dem der Klassifikator trainiert/validiert wird. Ohne das fliegt die News-Pipeline blind.
- **FinBERT ist nicht in den Deps.** Nur `transformers + torch` als `ml-nlp`-Extra. README sagt "FinBERT oder ähnliches für News-Sentiment verwenden" (Nice-to-have in `KNOWN_ISSUES.md`, Abschnitt 4.2).
- **News-Dummy-Daten:** README sagt explizit zur Event-Strategie: "Verwendet Dummy-Daten für Insider- und Shipping-Events (echte Datenquellen geplant)". Du hast also einen Prototyp, keine produktive Pipeline.

**Konkrete Aktion:**
1. **100 News-Events manuell labeln.** 5-10 Ticker aus deiner Watchlist, pro Ticker 10-20 News aus den letzten 6 Monaten. Spalten: ticker, news_id, headline, ground_truth_sentiment (-1/0/+1), ground_truth_actionable (yes/no), notes. Ein Abend Arbeit.
2. **Baseline-Klassifikator:** VADER oder TextBlob gegen die 100 Events laufen lassen. Metrik: Accuracy, Precision, Recall für "actionable". **Das ist deine Baseline.**
3. **FinBERT nur wenn Baseline <60% Accuracy.** Wenn VADER schon 75% schafft, ist FinBERT 100 MB Model-Download wert, der dich langsam macht? Wahrscheinlich nicht für deinen Scale.
4. **Diese 100 Events committen** als `tests/ground_truth/news_labels_v1.csv` und regelmäßig re-evaluieren. Das ist dein Anker gegen stillen Drift.

**Aufwand:** 4-6 Stunden für Labeling, 2-3 Stunden für Baseline-Evaluation. Nicht groß.

---

### Rang 3 — Golden-Equity-Scenario-Tests (`35_GOLDEN_EQUITY_SCENARIO_TESTS.md`)

**Status im Repo:** 🟡 Konzept existiert unter anderem Namen, aber nicht als dedizierter Test-Typ.

**Was funktioniert:**
- `pytest.ini` hat `slow`-Marker — das entspricht der Idee "lange Equity-Tests laufen nur lokal"
- `tests/` hat phasen-spezifische Unter-Suiten
- `PAPER_TRACK_PLAYBOOK.md` fordert Regime-Coverage (mind. 3 Regime) als Gate-Kriterium — konzeptionell äquivalent zu Golden-Scenarios
- `STRATEGY_POLICY.md` definiert Soft/Hard/Kill-Drawdown-Schwellen (-15%, -25%, -35%) — konkrete Test-Checkpoints

**Was fehlt:**
- **Keine expliziten "Golden"-Test-Szenarien.** Die Struktur in `tests/` ist phasen-orientiert, nicht szenario-orientiert. Ein Test wie `test_2020_covid_crash_scenario.py` (der die Strategy durch März 2020 schickt und die Equity-Curve mit einer gespeicherten erwarteten Kurve vergleicht) existiert wahrscheinlich nicht.
- **Characterization-Tests für Strategy-Outputs sind nicht dokumentiert.** Wenn du die Strategy anpasst und der Sharpe geht von 1.4 auf 1.7, gibt es keinen Test, der dir sagt "Achtung, Trade 247 von 2023-03-15 verhält sich jetzt anders".
- **Deflated-Sharpe-Gate ist implementiert** (`docs/DEFLATED_SHARPE_B4_DESIGN.md` referenziert in Paper-Track-Playbook) — aber das ist ein Gate, kein Regression-Test.

**Konkrete Aktion:**
1. **3-5 historische Kern-Szenarien definieren:**
   - 2020-02-01 bis 2020-06-30 (COVID-Crash + Recovery)
   - 2022-01-01 bis 2022-12-31 (Bear-Market-Full-Year)
   - 2023-03-01 bis 2023-04-30 (SVB Bank Crisis)
   - 2024-04-01 bis 2024-08-31 (Normal-Bull)
   - 2025-08-01 bis 2025-10-31 (wenn du Daten hast — Korrektur-Phase)
2. **Für jedes Szenario Baseline-Outputs speichern:** `tests/golden/{scenario_name}/expected_equity.parquet`, `expected_trades.parquet`, `expected_metrics.json`.
3. **Test-Datei:** `tests/test_golden_scenarios.py` — für jedes Szenario ein Test, der den Backtest rennt und die Outputs mit Toleranz (z.B. ±1% auf Equity-Curve, exakt auf Trade-Count) vergleicht.
4. **CI-Integration:** diese Tests nur lokal oder auf `main`-Merge, nicht auf jeder PR (zu langsam). Markieren mit `@pytest.mark.golden`.

**Aufwand:** 8-12 Stunden für Setup, dann selbstlaufend.

---

### Rang 4 — Multi-Environment-Setup (`36_MULTI_ENVIRONMENT_SETUP.md`)

**Status im Repo:** 🔴 Kritischer Gap.

**Was funktioniert:**
- `pydantic-settings>=2.0.0` in Runtime-Deps — Infrastruktur für Environment-Config vorhanden
- `src/assembled_core/config/settings.py` existiert (referenziert in `KNOWN_ISSUES.md`)
- `Dockerfile` + `docker-compose.yml` im Root

**Was fehlt:**
- **Keine Hinweise auf `.env.dev`, `.env.staging`, `.env.prod` in README oder Doku.**
- **`KNOWN_ISSUES.md`:** "Live-Trading-Mode ist als Kommentar markiert (`Future: Live trading mode (not yet implemented)`)." → Environment-Trennung existiert nur konzeptionell.
- **`CLAUDE.md`** nennt Security/Secrets/.env explizit als **deferred** (Abschnitt 6): "Folgende Themen sind bewusst aufgeschoben."
- **Für Paper vs. Live:** Es gibt keinen klaren Mechanismus, der Alpaca-Paper-Keys vs. Live-Keys trennt und verhindert, dass ein Run aus Versehen auf Live geht.

**Das ist der gefährlichste Punkt im ganzen Repo.** Wenn du in 3 Monaten Live gehst und der erste Run nimmt aus Versehen `.env` aus dem Dev-Kontext — Katastrophe.

**Konkrete Aktion:**
1. **Drei `.env.*`-Dateien anlegen:**
   - `.env.dev` → `ALPACA_ENV=paper`, `ALPACA_API_KEY_ID=...`, `ALPACA_API_SECRET=...` (Paper-Keys)
   - `.env.staging` → `ALPACA_ENV=paper`, anderes Paper-Keyset (optional)
   - `.env.prod` → `ALPACA_ENV=live`, Live-Keys, **ZUSÄTZLICH:** `LIVE_TRADING_ARMED=false` als Dead-Man's-Switch
2. **`.env.*` in `.gitignore`** (falls nicht schon). Stattdessen `.env.dev.example`, `.env.staging.example`, `.env.prod.example` mit leeren Werten committen.
3. **`ENVIRONMENT` als Pflicht-Variable:** in `src/assembled_core/config/settings.py`, jede Pipeline checkt `ENVIRONMENT in ["dev", "staging", "prod"]` sofort beim Start.
4. **Explicit-Deny-Regel für Live:** `if ENVIRONMENT == "prod" and not LIVE_TRADING_ARMED: raise RuntimeError(...)`. Zwei Flags müssen gesetzt sein, bevor echte Orders rausgehen.
5. **Paper-Account-Differenzierung:** zweiter Alpaca-Paper-Account mit realistischem Budget (z.B. 2000 USD statt Default 100k) als `.env.staging` — damit du realistische Position-Sizing-Tests hast.

**Aufwand:** 4-6 Stunden. Nicht groß, aber hohe Wichtigkeit. **Priorität 1 nach Cleanup.**

---

### Rang 5 — Data-Quality-Gate (`37_DATA_QUALITY_GATE.md`)

**Status im Repo:** 🟡 Teilweise umgesetzt in eigener Form, aber nicht Pandera-basiert.

**Was funktioniert:**
- `src/assembled_core/qa/` Modul existiert
- `phase4`-Tests umfassen "QA-Gates: OK/WARNING/BLOCK-Logik" (aus README)
- `src/assembled_core/qa/point_in_time_checks.py` (referenziert in Paper-Track-Playbook)
- Health-Gates für News/Disclosures/MarketData in `STRATEGY_POLICY.md`
- QA-Reports werden in EOD-Pipeline generiert
- PIT-Checks sind implementiert (Factor-Store)

**Was fehlt:**
- **Pandera ist nicht in den Dependencies.** Du hast ein eigenes QA-Gate-System gebaut, statt das Standard-Tool zu nutzen. Das ist ok, aber du musst sicher sein, dass es abdeckt: Schema-Checks, Range-Checks, Null-Ratios, PIT-Violations, Stale-Data-Detection, Corporate-Action-Adjustments.
- **Die Check-Liste aus Playbook-37 sollte formal gemappt werden:** für jeden der ~25 Checks im Playbook eine konkrete Ja/Nein-Aussage, ob dein QA-Gate ihn enthält.

**Konkrete Aktion:**
1. **QA-Gate-Audit:** Geh das Playbook-37 durch und mach eine Spalten-Tabelle: "Check — Implementiert? — Datei — Test-Coverage". Dabei Lücken finden.
2. **Entscheidung:** Pandera einführen oder beim eigenen System bleiben? Bei deinem Scale und dem existierenden System: **bleibe beim eigenen**. Pandera-Migration wäre Nebenprojekt ohne klaren Nutzen.
3. **Stale-Data-Detection hinzufügen** (falls nicht da): Check, ob die neuesten Daten älter als N Tage sind. Kritisch für Paper-Runs.

**Aufwand:** 3-5 Stunden Audit + evtl. 4-8 Stunden für gefundene Lücken.

---

### Rang 6 — Feature-Attribution-Dashboard (`38_FEATURE_ATTRIBUTION_DASHBOARD.md`)

**Status im Repo:** 🟡 Basis vorhanden, Dashboard-Layer fehlt wahrscheinlich.

**Was funktioniert:**
- `shap>=0.44.0` als `ml-explain`-Extra verfügbar
- `src/assembled_core/api/routers/monitoring.py` existiert
- README: "Feature Importance & Explainability (E2) – Understand which factors drive ML predictions"
- Experiment-Tracking in `experiments/` (Sprint 12.2) — das ist die Backend-Infrastruktur
- README: "Compare model performance (Linear, Ridge, Lasso, Random Forest)"
- Factor-Exposure-Berechnung in `src/assembled_core/risk/factor_exposures.py`

**Was fehlt:**
- **Streamlit oder Dash nicht in den Dependencies.** Also kein interaktives Dashboard.
- **`KNOWN_ISSUES.md` Abschnitt 1.4:** "Monitoring-API liefert aktuell Dummy-Daten für Drift-Status." → Der Monitoring-Endpoint ist ein Platzhalter.
- **Keine Per-Trade-Attribution sichtbar:** Für jeden Trade die Feature-Contributions zeigen — das ist der Kern von Playbook-38.

**Konkrete Aktion:**
1. **Priorität zurückstellen.** Du brauchst das Dashboard erst, wenn du aktiv mehrere Strategien vergleichst. Aktuell ist dein Experiment-Tracking-System genug.
2. **Wenn doch:** Streamlit einführen (eine Line: `pip install streamlit`), minimal 3 Seiten: Equity-Curve, Factor-Contributions per Trade, Drift-Metrics.
3. **Dummy-Data-Problem fixen:** die Monitoring-API von Dummy auf real migrieren. Das ist ein konkreter TODO in deinem eigenen `KNOWN_ISSUES.md`.

**Aufwand:** 2-3 Tage wenn priorisiert, aber **nicht jetzt**.

---

### Rang 7 — Hyperparameter-Governance (`39_HYPERPARAMETER_GOVERNANCE.md`)

**Status im Repo:** 🟢 Gut umgesetzt, aber nicht mit MLflow.

**Was funktioniert:**
- Eigenes Experiment-Tracking in `experiments/` (Sprint 12.2)
- CLI-Integration: `--track-experiment --experiment-name "..." --experiment-tags "..."` für Backtests und Meta-Model-Training
- `run.json` (Metadaten) + `metrics.csv` (Zeitreihen) + `artifacts/` (Dateien) pro Run
- Deterministische Run-IDs (hash-based, aus README: "Deterministic run IDs (hash-based, reproducible)")
- `logs/{run_id}.log` mit strukturiertem Logging
- `configs/policy.yaml` als Single Source of Truth (STRATEGY_POLICY.md)

**Was fehlt:**
- **MLflow ist nicht im Stack.** Das ist bewusst (aus dem CLAUDE.md: "keine zusätzlichen externen Services"). Dein eigenes Tracking deckt 80% des MLflow-Nutzens ab.
- **Version-Manifest** existiert (`version.manifest.json`), aber unklar wie aktuell.
- **Optuna ist im `ml-tune`-Extra** — Hyperparameter-Search-Infrastructure ist also da, aber Integration mit Experiment-Tracking ist fraglich.

**Konkrete Aktion:**
1. **Keine Änderung nötig.** Dein Setup ist pragmatisch und funktioniert für deinen Scale.
2. **Optional:** Optuna-Studies in `experiments/`-Ordner integrieren. Jeder Optuna-Trial wird als Sub-Run gespeichert.
3. **`version.manifest.json`-Policy:** Entweder wird das bei jedem Release aktualisiert (in CI), oder es wird gelöscht. Stale Version-Manifests sind schlechter als keine.

**Aufwand:** 1-2 Stunden für Policy-Klärung. Ansonsten zufrieden sein.

---

### Rang 8 — PDT-Regel und Intraday-Margin (`41_PDT_REGEL_INTRADAY_MARGIN.md`)

**Status im Repo:** 🔴 Nicht umgesetzt.

**Was funktioniert:**
- `alpaca-py>=0.30.0` in Runtime-Deps
- Pre-Trade-Checks existieren (`src/assembled_core/execution/pre_trade_checks.py`)
- Kill-Switch existiert

**Was fehlt:**
- **Kein PDT-Tracker erwähnt in README oder Doku.**
- **Keine Round-Trip-Detection** für Day-Trades in `KNOWN_ISSUES.md` oder `docs/PHASE10_PAPER_OMS.md`.
- **Kein Cutover-Plan für 4. Juni 2026** im Repo.

**Konkrete Aktion:**
**Für Paper-Trading irrelevant, weil dein Account 100k USD hat** (über der 25k-Schwelle). Daher:
1. **Wenn du in den nächsten 3 Monaten live gehst mit <25k:** Rang-8-Playbook implementieren (2 Wochen Aufwand).
2. **Wenn du Paper-only bleibst bis Juni 2026:** Problem löst sich mit PDT-Abschaffung von alleine. **Stattdessen:** Paper-Account-Budget auf realistischen Live-Startbetrag setzen (siehe Rang 4).
3. **Trotzdem:** den 4. Juni 2026 als Task in deinen Kalender. Alpaca wird irgendwann danach migrieren, und deine API-Error-Handling muss auf neue 403-Patterns vorbereitet sein.

**Aufwand:** 0 Stunden jetzt. 2 Wochen wenn Live mit <25k.

---

### Rang 9 — Event-Replay-System (`42_EVENT_REPLAY_SYSTEM.md`)

**Status im Repo:** 🔴 Nicht umgesetzt.

**Was funktioniert:**
- Strukturiertes Logging pro Run (`logs/{run_id}.log`)
- Factor Store mit PIT-Safety (speichert Features-at-a-point-in-time)
- Deterministische Run-IDs (hash-based)
- `CLAUDE.md` Abschnitt 5.4: "Backtest = Replay, nicht Parallelwelt" — **du hast die Philosophie schon.**

**Was fehlt:**
- **Kein Event-Store.** Market-Ticks, News-Events, Order-Fills werden nicht als append-only-Stream persistiert.
- **Keine Clock-Abstraktion.** Grep nach `datetime.utcnow()` würde vermutlich viele Stellen finden.
- **Keine explizite Replay-Infrastruktur.**

**Konkrete Aktion:**
**Für jetzt zurückstellen.** Du brauchst Event-Replay erst, wenn:
- Du regelmäßig schwer reproduzierbare Bugs jagst (dann wäre es lebensrettend)
- Du mehrere Strategien parallel laufen lässt und deren Divergenz verstehen musst
- Du Forensik für Incidents brauchst

Aktuell ist keines davon gegeben. **Priorität:** niedrig bis mittel. Frühestens Q3 2026.

**Zwischenschritt, der sinnvoll ist:**
1. **Strukturierter Log-Audit:** eine Woche lang alle Strategie-Entscheidungen loggen als JSON-Lines. Wenn du nicht alle Inputs rekonstruieren kannst, weißt du, was in deinem zukünftigen Event-Store sein muss.

**Aufwand:** 0 Stunden jetzt. 5-6 Wochen wenn priorisiert.

---

### Rang 10 — Backtest-Reproducibility-Zertifikat (`43_BACKTEST_REPRODUCIBILITY_CERTIFICATE.md`)

**Status im Repo:** 🟡 Basis vorhanden, aber nicht systematisiert.

**Was funktioniert:**
- `Dockerfile` + `docker-compose.yml` existieren
- `requirements.lock` existiert als Pip-freeze-Output
- Deterministische Run-IDs (hash-based)
- Experiment-Tracking speichert Config + Artefakte
- `.github/workflows/` mit Evidence-Pack-CI — das ist in der Geistesrichtung

**Was fehlt:**
- **Kein SHA-256-Hash-basiertes Zertifikat pro Backtest-Run.**
- **Docker-Image ist nicht mit `@sha256:`-Digest gepinnt.**
- **uv nicht im Einsatz** (Pip + `requirements.lock`). Das ist okay, aber `uv`-basierte Lockfiles sind robuster als Pip-freeze.
- **Kein monatlicher Drift-Check.**

**Konkrete Aktion:**
1. **Zertifikats-System später.** Aktuell reichen dir Git-SHA + Config-Hash, das machst du implizit schon über Experiment-Tracking.
2. **Docker-Digest-Pinning jetzt:** `FROM python:3.11.9-slim-bookworm@sha256:<DIGEST>`. Einzeilige Änderung, sofort verfügbar.
3. **`requirements.lock` auf `uv.lock` migrieren** — aber nur wenn du Zeit hast. Nicht-kritisch.
4. **Monatlicher Drift-Check** (wie in Playbook-43 beschrieben): einen konservativen Backtest monatlich laufen lassen, Sharpe ± 0.01 ist der Fingerabdruck, signifikante Abweichung = Alert. Das ist in 30 Min Cron-Job erledigt.

**Aufwand:** 2-3 Stunden für Docker-Pinning + Drift-Check-Cron. Reproducibility-Zertifikat: 2 Wochen, aber später.

---

## Zusammenfassung — Priorisierte Umsetzungsreihenfolge

| # | Rang | Aufwand | Priorität | Risiko wenn nicht gemacht |
|---|------|---------|-----------|---------------------------|
| 1 | **Rang 4 (Multi-Env)** | 4-6h | **KRITISCH** | Live-Unfall |
| 2 | **Rang 1 (Migration/Cleanup)** | 6-10h | **HOCH** | Struktureller Drift eskaliert |
| 3 | **Rang 2 (News-Ground-Truth)** | 6-9h | HOCH | News-Pipeline fliegt blind |
| 4 | **Rang 3 (Golden-Scenarios)** | 8-12h | MITTEL | Regression unerkannt |
| 5 | **Rang 5 (QA-Audit)** | 4-8h | MITTEL | Unklare Coverage |
| 6 | **Rang 10 (Docker-Pinning + Drift)** | 2-3h | NIEDRIG-MITTEL | Environment-Drift |
| 7 | **Rang 7 (Version-Manifest-Policy)** | 1-2h | NIEDRIG | Kosmetik |
| 8 | **Rang 6 (Dashboard)** | 2-3 Tage | NIEDRIG | Nice-to-have |
| 9 | **Rang 8 (PDT)** | 0h jetzt | NULL | Nicht relevant bei Paper-100k |
| 10 | **Rang 9 (Event-Replay)** | 0h jetzt | NULL | Nicht jetzt nötig |

**Gesamt-Cleanup-Zeitrahmen:** 4-6 Wochen bei 10-15 Stunden/Woche, wenn in obiger Reihenfolge. Erstes Feedback nach 2 Wochen (Cleanup + Multi-Env).

---

## Die drei Sachen, die mir darüber hinaus auffallen

### 1. Die Philosophie ist nüchterner als der Code

`CLAUDE.md` und `STRATEGY_POLICY.md` klingen wie aus einer Hedgefund-Complianceabteilung. Aber der Code hat 13 optionale Extras, 2 parallele Config-Ordner, und ein kaputtes Profiling-File im Root. **Der Delta zwischen Governance-Texten und Code-Realität ist das eigentliche Risiko.** Du bist disziplinierter im Schreiben als im Aufräumen.

**Praktische Konsequenz:** Vor dem nächsten neuen Feature ein Aufräum-Sprint. Sonst wird die `CLAUDE.md` zur Fiktion.

### 2. Du hast den "`Assembled` heißt jetzt `assembled_core`"-Switch schon gemacht

Package-name ist `assembled-trading-core`, Hauptmodul ist `assembled_core/`. Repo-name ist `Assembled-Trading-AI`. Lokaler Pfad ist laut README `Aktiengerüst`. **Das sind vier verschiedene Namen für dasselbe Projekt.** 

Entscheidung treffen:
- **Wenn "Assembled" der Markenname werden soll:** lokaler Pfad umbenennen von `Aktiengerüst` zu `Assembled-Trading-AI`, README-Block `cd Aktiengerüst` entsprechend fixen.
- **Wenn "Aktiengerüst" dein Arbeits-Codename ist:** explizit in der README erwähnen als "Dies ist der lokale Arbeitsname, das Repo heißt extern Assembled-Trading-AI."

Kosmetisch, aber erleichtert Onboarding — und du **bist** dein eigenes "in 6 Monaten"-Onboarding.

### 3. Paper-Track-Playbook ist extrem gut, aber die Dokumentations-Ebenen widersprechen sich

`PROJEKT_STATUS.md` (Jan 2025): nur Phase 4 fertig, Rest "bereit für neue Features".
`README.md` (aktuell): Phase 4-12 komplett.
`KNOWN_ISSUES.md` (Jan 2025): "Phase 12.3 - Review-Vorbereitung".
`CLAUDE.md` (aktuell): "Phase 12.3 abgeschlossen" ist implizit, aber nicht explizit.

**Als Reader kann ich nicht entscheiden, was aktuell ist.** Die Status-Dokumente sind zu mehreren Zeitpunkten entstanden und wurden nicht konsolidiert.

**Aktion:** Eine einzige aktive Status-Quelle wählen (vorschlag: `PROJEKT_STATUS.md` aktualisieren auf heute, `KNOWN_ISSUES.md` behalten aber Datum refreshen). Die `README.md`-Status-Tabelle sollte aus diesen Dateien generiert werden, nicht manuell gepflegt.

---

## Abschluss-Einschätzung

**Was dir jetzt wirklich fehlt:**
1. **Ein Cleanup-Sprint** (Rang 1 + Rang 4 Kombination, ~15 Stunden)
2. **Konsistenz zwischen den Status-Dokumenten** (~2 Stunden)
3. **100 manuell gelabelte News-Events** als Ground-Truth-Anker (~4 Stunden)

**Was dir nicht fehlt:**
- Funktionalität. Du hast mehr Module als du in den nächsten 6 Monaten nutzen wirst.
- Governance-Dokumentation. `CLAUDE.md` und `STRATEGY_POLICY.md` sind Gold.
- CI-Infrastruktur. Multiple Workflows laufen.

**Was ich als dein Reviewer ehrlich sagen würde:**
Ein Stakeholder (Investor, späterer Partner, du selbst in 2 Jahren) der dieses Repo zum ersten Mal sieht, wird in den ersten 30 Minuten zwei Eindrücke haben: "wow, das ist sehr viel" und "was soll ich damit anfangen?" Die zweite Reaktion ist ein Reviewing-Problem, kein technisches. Aber sie kostet dich Glaubwürdigkeit. Ein ordentlicher Cleanup-Sprint vor dem nächsten Feature-Push ist das, was aus einem beeindruckenden Hobbyprojekt ein glaubwürdiges quantitatives System macht.

Das ist der Weg von hier.
