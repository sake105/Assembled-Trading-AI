# Gap-Analyse und Recherche-Prioritäten

**Datum:** 2026-04-24
**Basis:** Vergleich Audit (Stand 2026-04-23, Commit 55c0d06) × Plan-Pakets
**Hinweis:** Ich kann nicht ins aktuelle Repo schauen (privat). Diese Analyse basiert auf dem Audit-Stand plus den Plan-Dokumenten. Was du seitdem umgesetzt hast, ist mir nicht sichtbar — bitte beim Lesen jede Position selbst abgleichen.

---

## Teil A — Was der Plan adressiert vs. was noch offen ist

Ich habe jeden der 51 als **schwer** markierten Audit-Befunde gegen die Plan-Dateien geprüft.

### A.1 Vollständig durch den Plan adressiert (34 von 51)

| Audit-Befund | Plan-Datei | Wie adressiert |
|---|---|---|
| 1.1 `trading_cycle.py` Mega-Monolith | 33, Plugin-Arch | BaseSignal + Registry via entry_points ersetzt den Monolithen |
| 1.2 Wave-Wiring | 40 Roadmap P0 | Woche 2: "mlfinlab entfernen, mlfinpy", und generell: keine Scheinintegrationen |
| 1.9 Zwei Order-Pfade | 33 Execution | Ein Order-Pfad mit `client_order_id`, einheitlicher Position-Sizer |
| 2.2 ML-Module wired aber ungenutzt | 13, 40 | Priorisierte Module-Liste, keine Hype-Module |
| 2.3 Risk-Module wired aber ungenutzt | 33 | Risk-Module sind Pre-Trade-Gates, keine "observability-wired" |
| 2.5 Crisis-Alpha nie validiert | 32 | CPCV + Deflated Sharpe + Historical-Regime-Test |
| 3.1 Scheintests | 32, 33 | Shadow-Mode statt trivialer Importability-Tests |
| 3.2 Sharpe 98 durch | 32 | DSR, PBO, CPCV-Pipeline-Pflicht |
| 3.3 Keine E2E-Tests | 32 | Walk-Forward + Shadow-Mode-Pipeline |
| 4.1/4.2 Parquet-Daten | 14, 23 | Universum + Storage sauber spezifiziert |
| 4.12 Corporate Actions fehlen | 50 Compliance | Tax-Lot-Tabelle mit Split-Handling |
| 5.1 EMA-Crossover als Haupt-Strategie | 31 Composite | 9-Dimensionen ersetzen EMA als Baseline |
| 5.5 Qty=1.0 | 33 Sizing | Kelly + Vol-Target + Sektor-Cap-Position-Sizer |
| 5.6 Keine Exit-Logik | 33 §33.11 | ExitManager: Stop/PT/Vertical/Regime-Change |
| 6.4 Kein Slippage-Test | 33 §33.12 | ExecutionCostModel im Backtest |
| 7.1 Risk als Library | 33 | Risk-Gates im Submit-Pfad |
| 7.2 Position-Konzentration | 33 Sizer | Sector-Cap 20%, Position-Cap 5% hart |
| 7.3 Drawdown-Triggers | 33 Kill-Switch | Hard-Kill bei 3% daily / 7% weekly |
| 9.2 News-Pipeline nicht mit Signal verbunden | 30 Fusion | Drei Schichten explizit |
| 10.1 .env in Git | 40 Phase 0 | Gitleaks + Trufflehog + SOPS+age |
| 15.1 Survivorship-Bias | 14, 20, 23 | EODHD Delisted-Coverage in Phase 2 |
| 15.2 Look-Ahead-Bias | 31, 32 | `available_at` + `.shift(1)` + Purged-CV |
| 15.4 Cost-Multiplier | 33 | ExecutionCostModel |
| 16.1 Vision vs Realität | 40 | Phasen mit Go/No-Go-Gates |
| 17.1 yfinance Lizenz | 10, 50 | Als Privat-Tolerant markiert, Commercial würde EODHD brauchen |
| 21.3 Scenario Tests | 32 | Historische Regime-Tests (2008, 2020) in CPCV |

... und weitere.

### A.2 Nur teilweise adressiert (10 von 51) — **Lücken im Plan**

Diese Audit-Befunde sind im Plan nicht explizit genug. Das sind die **echten Lücken**:

#### Lücke 1 — Migration-Strategie vom aktuellen Repo zum neuen Plan

**Audit-Bezug:** 1.1, 1.2, 1.15, 2.2, 2.3 (Monolith + Wave-Wiring + 52 ungenutzte ML-Module + Rest)

**Problem:** Der Plan beschreibt, wie das System **sein soll**. Er beschreibt nicht, wie du vom **jetzigen** Zustand (10.544-Zeilen-Monolith, 309 Steps, 52 ungenutzte ML-Module, 147 Wave-Tests) dorthin kommst. Die Roadmap (40) startet auf der grünen Wiese — das stimmt nicht mit deiner Realität überein.

**Was fehlt:**
- Ein **Migration-Playbook**: welche Wave-Tests zuerst, welche ML-Module archivieren, wie Archive-Strategie, wie Parallelbetrieb alter/neuer Code
- Entscheidungsregel: "delete" vs "rewrite" vs "keep" pro Alt-Modul
- Schrittweise `trading_cycle.py`-Zerlegung mit Teststrategie

**Priorität: HOCH**. Du kannst den Plan nicht starten, ohne zu wissen, was mit dem Existierenden passiert.

---

#### Lücke 2 — Golden-Equity-Test + Scenario-Tests (2008, 2020, 2022)

**Audit-Bezug:** 3.7, 21.3

**Problem:** Der Plan (32) fordert CPCV + DSR + PBO, aber keinen **Golden-Equity-Regressionstest**. Wenn du `trading_cycle.py` zerlegst, musst du beweisen können: "das neue System produziert auf 2020-2025 dieselbe Equity-Kurve wie das alte (innerhalb Toleranz)".

**Was fehlt:**
- Golden-Equity-Baseline-Spec: welches Universum, welche Periode, welche Toleranzen?
- Scenario-Tests: 2008 (GFC), 2020 (COVID), 2022 (Rate-Shock) als Pflicht-Assertions mit erwartetem Drawdown-Range
- Replay-Harness für historische News-Events

**Priorität: HOCH**. Ohne Golden-Tests ist Refactoring riskant.

---

#### Lücke 3 — Feature-Versionierung und Model-Versioning

**Audit-Bezug:** 8.x allgemein, 21.x

**Problem:** Der Plan (31) spezifiziert 9 Dimensionen. Er sagt nicht: Was passiert, wenn ich Dim 2 von RSI-14 auf RSI-21 ändere? Wie wird alter Composite-Score mit neuem verglichen? Wie ist das Feature-Schema versioniert?

**Was fehlt:**
- Feature-Schema-Versionierung (z.B. `sentiment_vw_v2`)
- Model-Registry mit Aliases + expliziter Shadow-vs-Production-Lifecycle (in 12 erwähnt, aber nicht operationalisiert)
- A/B-Testing-Mechanik für Features (zwei Versionen parallel bewerten)

**Priorität: MITTEL**. Wird in Phase 2 relevant, wenn Features iteriert werden.

---

#### Lücke 4 — Data-Quality-Gate vor Feature-Berechnung

**Audit-Bezug:** 4.13 (Daily-Drift-Detection fehlt)

**Problem:** Der Plan (32) hat PSI-Drift-Monitoring für Features, aber keinen expliziten **Data-Quality-Gate vor dem Ingest**: Wenn Alpaca einen korrupten Bar liefert (Volume = 0, Close = 0.00001), muss das vor der Feature-Berechnung gestoppt werden.

**Was fehlt:**
- Ingest-Gate-Regeln (Spike-Detection, NaN-Rate, Schema-Validation)
- Quarantäne-Logik für verdächtige Bars
- Backfill-Strategie wenn Daten nachkommen

**Priorität: HOCH**. Ein korrupter AAPL-Bar während einer Session produziert Fake-Signale.

---

#### Lücke 5 — Live-Shadow-Parity-Check

**Audit-Bezug:** 6.3 (Alpaca nie operativ im Trial), 15.4 (Cost-Multiplier)

**Problem:** Der Plan (33) hat Execution-Drift-Monitor, aber der vergleicht nur **einzelne Fills** gegen Backtest-Erwartung. Was fehlt ist ein kontinuierlicher **Shadow-Parity-Check**: "läuft das Backtest-Modell **parallel** zum Live-System auf denselben Daten, und wenn ja, wie weit driften sie auseinander?"

**Was fehlt:**
- Shadow-Backtest-Replay auf Live-Daten (eine Bar vergeht real → Backtest macht denselben Schritt)
- Kumulative P&L-Divergenz als Prometheus-Metrik
- Alert bei >5% Divergenz über 30 Tage

**Priorität: MITTEL**. Kritisch für Paper→Live-Transition.

---

#### Lücke 6 — Feature-Attribution über Zeit

**Audit-Bezug:** 8.x (ML-Schicht)

**Problem:** Plan (33) hat SHAP-Attribution pro Trade. Was fehlt: **Attribution über Zeitfenster**. Frage: "Welche Features haben in Q1 2026 meinen P&L getrieben? Welche haben geschadet?"

**Was fehlt:**
- Aggregierte Feature-P&L-Attribution über Zeiträume
- Time-Series-Charts von Feature-IC
- Automatische Erkennung von "Dead-Features" (Feature mit IC≈0 über 90d)

**Priorität: MITTEL**. Relevanz steigt ab Phase 3.

---

#### Lücke 7 — Hyperparameter-Governance

**Audit-Bezug:** 8.x allgemein (ML-Schicht unklar)

**Problem:** Plan (30, 31) hat viele Parameter: θ_meta=0.55, τ_veto=1.5, KellyFraction=0.25, 5×5-Matrix-Kalibrierung. Keine **Governance**: wer darf wann wie ändern? Wie wird Change dokumentiert? Rollback?

**Was fehlt:**
- Parameter-Registry mit Audit-Trail
- Change-Control: "ändern nur nach CPCV-Validierung"
- Rollback-Prozedur

**Priorität: MITTEL**. Ohne das driften Parameter silent.

---

#### Lücke 8 — News-Pipeline Ground-Truth-Validation

**Audit-Bezug:** 9.10 (Keine Ground-Truth-Messung für News-Impact)

**Problem:** Plan (30) baut sechs News-Sub-Features. Keine Methodik, um zu **validieren**, dass sie echt Alpha bringen (und nicht Rauschen sind, das im Composite gewichtet wird).

**Was fehlt:**
- IC-Test pro News-Sub-Feature separat
- Event-Study-Framework (CAR-Messung nach News-Event pro Ticker)
- Vergleich FinBERT vs Haiku auf gelabelten Earnings-Headlines

**Priorität: HOCH**. Wenn News-Features kein echtes Signal sind, wird der 9. Composite-Dim zu Dead-Weight.

---

#### Lücke 9 — Ingest-Dedup über Quellen

**Audit-Bezug:** 9.x allgemein (News-Schicht)

**Problem:** Plan (30) hat Semantic-Dedup via hnswlib. Aber: GDELT, Finnhub, RSS können dieselbe Story haben. Der Plan sagt "Dedup über URL-Hash + Semantic-Similarity". Was fehlt: **Cross-Source-Dedup-Regel** mit Corroboration-Boost.

**Was fehlt:**
- Explizite Cross-Source-Dedup-Matrix (was passiert wenn GDELT und Finnhub dieselbe Story melden)
- Corroboration als positiver Score statt Dedup-Eliminierung
- Quellen-Priorität bei echtem Duplikat (Reuters wins vs. Seeking Alpha)

**Priorität: MITTEL**. Sub-optimale Dedup-Regel verzerrt News-Volume-Features.

---

#### Lücke 10 — PDT-relevante Strategie-Umgestaltung

**Audit-Bezug:** 5.x, 6.x (Paper-Pfad)

**Problem:** Plan (50) erwähnt PDT. Was fehlt: Falls dein Eigenkapital **live** unter 25k USD startet, müssen die Signal-Strategien **per-Ticker** PDT-aware sein, nicht nur der Order-Pfad. Der Plan hat Swing-orientierung (EOD), aber Composite-Score kann Intraday-Charakter haben (bei News-Events).

**Was fehlt:**
- Strategie-Level-PDT-Gate (nicht nur Order-Level)
- Swing-Forcing-Modus: minimale Halteperiode >1 Bar
- Explizite Dokumentation: welche Signale sind "day-trade-safe"

**Priorität: MITTEL-HOCH**. Kritisch, wenn Live-Start unter 25k Equity.

### A.3 Komplett neu in den Plan gehören (7 von 51)

Diese sind im Plan nirgendwo angesprochen:

#### Neu 1 — Event-Replay-Framework

**Begründung:** Wenn deine News-Pipeline einen Bug hat, willst du den einen kritischen Tag (z.B. FOMC-Meeting) isoliert rekonstruieren. Dafür brauchst du einen Replay-Mechanismus: Redis-Stream-Snapshot laden, Pipeline gegen Snapshot laufen lassen.

**Priorität: MITTEL**. Debuggen bei Incidents sonst unmöglich.

---

#### Neu 2 — Multi-Tenant-Free (aber isolierte Dev/Staging/Prod)

**Begründung:** Auch als Einzelperson brauchst du eine Dev/Staging/Prod-Trennung. Sonst testest du neue Signale gegen Live-Capital.

**Was fehlt:**
- `.env.dev`, `.env.staging`, `.env.prod` Policy
- Separate Alpaca-Accounts (Paper-Dev + Paper-Staging + Live-Prod)
- Merge-to-Main-Flow für Deployments

**Priorität: HOCH**. Ohne das sind Live-Deployments russisches Roulette.

---

#### Neu 3 — Decision-Log

**Begründung:** Du triffst kontinuierlich Entscheidungen ("Signal X deaktivieren", "θ_meta auf 0.6 heben"). Ohne Decision-Log vergisst du nach 3 Monaten, warum das so war.

**Was fehlt:**
- `docs/decisions/YYYY-MM-DD_<titel>.md` Template
- Standardfelder: Kontext, Optionen, Entscheidung, Konsequenzen

**Priorität: MITTEL**. Spart später Zeit.

---

#### Neu 4 — On-Call-Schedule für dich selbst

**Begründung:** Du bist Einzelperson. Wenn du im Urlaub bist, läuft das System ohne Aufsicht. Brauchst du geplante Auszeiten-Policy: "2 Wochen Urlaub → Kill-Switch soft, nur Liquidation-Orders erlaubt".

**Was fehlt:**
- Vacation-Mode im Kill-Switch
- Auto-Liquidation-Option bei längerem Outage
- Checkliste vor/nach Reisen

**Priorität: MITTEL**. Relevant ab Live.

---

#### Neu 5 — Cost-Tracking-Dashboard

**Begründung:** Plan hat Budget-Posten, aber keine automatische Verfolgung. Wenn Anthropic unerwartet auf 30 EUR/Monat springt, erfährst du es erst per Rechnung.

**Was fehlt:**
- `infrastructure.costs`-Tabelle mit täglichen Aggregaten
- Prometheus-Metrics für Anthropic, Hetzner, EODHD
- Alert bei Monats-Hochrechnung >100 EUR

**Priorität: NIEDRIG-MITTEL**. Finanzielle Kontrolle.

---

#### Neu 6 — Feature-Catalog mit Business-Description

**Begründung:** Du hast 52+ ML-Module gebaut, die meisten ungenutzt. Das ist auch ein Dokumentations-Problem: du wusstest nicht mehr, was was ist. Gegenmittel: Feature-Catalog, in dem jedes Feature dokumentiert ist mit "Zweck, Zeitrahmen, letzter IC, Status".

**Was fehlt:**
- `docs/features/` mit einem Markdown pro Feature
- Template: Purpose, Formula, Horizon, Sources, Last-IC, Status (active/deprecated/experimental)
- Auto-generiert aus Code-Docstrings?

**Priorität: MITTEL**. Ohne das wiederholst du den Audit-Befund in 2 Jahren.

---

#### Neu 7 — Backtesting-Reproduzierbarkeit (Seed-Management + Env-Capture)

**Begründung:** Ein Backtest heute und morgen müssen dasselbe Ergebnis liefern. Das erfordert: fester Seed + festgehaltene Library-Versionen + festgehaltene Daten-Snapshots.

**Was fehlt:**
- `git stash` + `data-snapshot-hash` pro Backtest-Run
- MLflow-Artefakt: `env.yaml` mit allen Library-Versionen
- Reproduzierbarkeits-Test in CI: "Backtest Run X sollte in 12 Monaten identisches Ergebnis liefern"

**Priorität: HOCH für wissenschaftliche Integrität, MITTEL für Praxis**.

---

## Teil B — Rangliste der Recherche-Prioritäten

Diese Reihenfolge ergibt sich aus Impact × verbleibender Unsicherheit. Top-Position = maximaler Rechercheeinsatz empfohlen.

### Rang 1 — Migration-Playbook (Monolith → Plugin-Architektur)

**Warum Nr. 1:** Ohne saubere Migration ist der gesamte Plan wertlos. Du hast 10.544-Zeilen-`trading_cycle.py` plus 52 ungenutzte ML-Module plus 147 Wave-Tests. Der Plan beschreibt die Ziel-Architektur, nicht den Weg dorthin.

**Was recherchieren:**
- Strangler-Fig-Pattern in Python-Monolith-Refactoring (2025/2026 Stand)
- Best-Practices für "Big-Ball-of-Mud"-Decomposition
- Konkrete Tooling-Empfehlungen: `vulture` für Dead-Code, `pydeps` für Dependency-Graphen, `radon` für Komplexitätsmessung
- Real-World-Cases: wie haben große Codebases (z.B. Shopify, Stripe) ähnliche Refactorings gemacht
- Test-Strategien während des Refactorings (characterization tests, approval tests)

**Output der Recherche:** `60_MIGRATION_PLAYBOOK.md` — schrittweise Anleitung mit Tool-Stack, Reihenfolge, Go/No-Go-Gates

---

### Rang 2 — Ground-Truth-Validation für News-Pipeline

**Warum Nr. 2:** Der komplette 30_NEWS_TA_FUSION-Ansatz steht und fällt mit der Frage, ob FinBERT + Haiku + HDBSCAN echt Alpha liefern oder ob es bloß Rauschen mit ML-Lipstick ist. Du hast Code gebaut, aber keine saubere Evaluation.

**Was recherchieren:**
- Event-Study-Methodologie für News in 2025/26 (CAR, AAR, BHAR-Methoden)
- Veröffentlichte Papers mit FinBERT-Validierung (realistische IC-Ranges)
- Benchmarks: wie performen OpenSource-NLP-Modelle vs. Proprietary (Ravenpack-Niveau)?
- Distant-Supervision-Methodik für Finanz-News-Labels
- SPADE, FPB, FiQA — welche Benchmarks sind 2026 relevant?
- ABSA (Aspect-Based Sentiment Analysis) Tools-Landschaft 2026

**Output:** `34_NEWS_GROUND_TRUTH.md` — Validierungs-Framework mit konkreten Benchmarks und Schwellwerten

---

### Rang 3 — Golden-Equity-Test und Scenario-Tests

**Warum Nr. 3:** Ohne Golden-Tests kannst du nicht sicher refactoren. Und ohne Scenario-Tests (2008, 2020, 2022) weißt du nicht, ob dein System Crisis-robust ist.

**Was recherchieren:**
- Regression-Testing-Frameworks für Zeitreihen-Systeme
- Hypothesis-basierte Property-Tests für Backtest-Invarianten
- Bekannte Scenario-Test-Frameworks (z.B. Bloomberg Portfolio & Risk Analytics, OpenRisk-Packages)
- Approval-Testing-Libraries in Python (z.B. `ApprovalTests.Python`, `snapshottest`)
- Historische Event-Kalender: wie simuliert man FOMC-Meeting, Flash-Crash, Brexit-Vote korrekt?
- Tolerance-Setting: welche Abweichungen akzeptabel bei deterministischen Backtests?

**Output:** Update in `32_VALIDIERUNG.md` + neue Section zu Scenario-Testing

---

### Rang 4 — Multi-Tenant-Setup (Dev/Staging/Prod)

**Warum Nr. 4:** Kritisch vor erstem Live-Deployment. Der Plan sagt nichts zur Trennung. Einzelperson heißt nicht "eine Environment".

**Was recherchieren:**
- Multi-Environment-Deployment-Patterns für Docker-Compose + Hetzner (2026)
- Alpaca-Account-Trennung (zwei Paper-Accounts, einen Live?)
- Git-Branch-Strategien: gitflow vs trunk-based für Solo-Projekte
- MLflow-Multi-Workspace-Setup
- Feature-Flags mit LaunchDarkly-Alternativen (open source: Flagsmith, Unleash)
- Blue-Green-Deployment auf einem Server (Kostenrahmen beachten)

**Output:** `35_MULTI_ENV_SETUP.md` oder Erweiterung in `22_PAID_INFRASTRUKTUR.md`

---

### Rang 5 — Data-Quality-Gate vor Feature-Berechnung

**Warum Nr. 5:** Plan hat PSI-Drift, aber kein Ingest-Gate. Ein einziger korrupter Bar kann alle Features vergiften.

**Was recherchieren:**
- Great Expectations für Market-Data-Quality-Checks (2025 Stand)
- Alternative: Pandera, Pydantic-Validation für DataFrame-Schemas
- Anomaly-Detection auf OHLCV-Streams: welche Algorithmen? (Isolation-Forest vs z-score)
- Alpaca-spezifische bekannte Daten-Artefakte (Dividenden-Gaps, Splits, Bad-Ticks)
- Out-of-Order-Bar-Handling bei WebSocket
- Canary-Bars: wie prüft man kontinuierlich, dass Feed vertrauenswürdig ist?

**Output:** `36_DATA_QUALITY_GATE.md` oder Erweiterung in `33_EXECUTION`

---

### Rang 6 — Feature-Attribution über Zeit

**Warum Nr. 6:** SHAP pro Trade ist gut, aber du brauchst Zeit-Aggregate, um Dead-Features zu erkennen.

**Was recherchieren:**
- SHAP-Aggregations-Patterns über Zeiträume (sum, mean, weighted)
- Shapley-Value-Interactions (Alpha über kombinierte Features)
- IC-Decay-Detection-Algorithmen (jüngste Literatur 2024/2025)
- Alphalens-Reloaded: aktueller Stand 2026
- Open-Source-Alpha-Dashboards (z.B. Qlib's Alpha360 UI)
- Auto-Feature-Deprecation: gibt es etablierte Kriterien?

**Output:** Erweiterung in `32_VALIDIERUNG.md` + Integration in MLflow-Dashboards

---

### Rang 7 — Hyperparameter-Governance

**Warum Nr. 7:** Nicht kritisch in Phase 1-2, aber vermeidet Silent-Drift ab Phase 3.

**Was recherchieren:**
- Configuration-as-Code-Patterns (Hydra von Meta, OmegaConf)
- Change-Control-Tools für ML-Parameter (Weights & Biases alternatives, DVC)
- Parameter-Registry-Designs in MLflow
- "Reproducibility-Bundles" — alles was für einen Run nötig war

**Output:** Erweiterung in `32_VALIDIERUNG.md` + evtl. `37_CONFIG_GOVERNANCE.md`

---

### Rang 8 — PDT-Strategie-Umgestaltung

**Warum Nr. 8:** Nur relevant, falls Live-Start mit <25k USD. Wenn Live-Start mit >25k, entfällt komplett.

**Was recherchieren:**
- PDT-Umgehung im Rahmen der Regel (legale Optionen: Overnight-Holds, Cash-Account)
- Alpaca-specific Day-Trade-Counter-API
- Swing-Trading-Signal-Kalibrierung (welche von den 9 Dimensionen sind day-trade-heavy?)
- Alternative: Offshore-Broker ohne PDT (zB IBKR Pro) — regulatorisch clean für EU-Resident?

**Output:** Erweiterung in `50_COMPLIANCE_RECHT.md`

---

### Rang 9 — Event-Replay-Framework

**Warum Nr. 9:** Debugging-Luxus, nicht kritisch.

**Was recherchieren:**
- Redis-Stream-Snapshot-Patterns
- Event-Sourcing-Replay in Python (EventStore, event-sourcery)
- Time-Travel-Debugging für Trading-Systeme
- Backtest-Engine-Integration als Replay-Engine

**Output:** `38_EVENT_REPLAY.md`

---

### Rang 10 — Backtesting-Reproduzierbarkeit

**Warum Nr. 10:** Wichtig für wissenschaftliche Integrität. Aber operativ nicht urgent.

**Was recherchieren:**
- Deterministische Zufalls-Seeds in sklearn/LightGBM/TensorFlow/PyTorch
- MLflow-Artifact-Patterns für Full-Environment-Capture
- `reprobench`, `reprozip` — Reproducibility-Tooling
- CUDA-Nondeterminismus-Handling (falls irgendwann GPU)

**Output:** Erweiterung in `99_STACK_LOCKFILE.md` + kurze Seed-Policy

---

## Teil C — Was ich explizit NICHT prüfen konnte

1. **Den aktuellen Stand deines Repos.** Ist privat, also sehe ich nicht, was du seit dem Audit gemacht hast. Welche Wave-Tests sind weg? Welche Plan-Module bereits umgesetzt? Musst du selbst abgleichen.
2. **Deine eigenen Entscheidungen seit dem Audit.** Vielleicht hast du bereits Lücken 1, 4, 5 angegangen.
3. **Laufzeit-Verhalten.** Der Plan ist Architektur-Review, nicht Performance-Review. Wenn dein System bei 585 Tickern pro Minute 2s braucht, muss das durch Lasttest gemessen werden.

---

## Teil D — Empfehlung für die nächste Session

**Ich würde vorschlagen:**

1. Du stellst das Repo kurzfristig privat-read oder teilst einzelne Dateien — dann kann ich sehen, was wirklich aktueller Stand ist, und die Gap-Liste wird präziser.

2. Du entscheidest, welcher Rang-1–3-Punkt der größte Schmerzpunkt ist. Wenn du schon mitten in der Migration bist, brauchst du **Rang 1** zuerst. Wenn du noch Alpha-Validation testest, **Rang 2**. Wenn du kurz vor Refactoring stehst, **Rang 3**.

3. Für den gewählten Rang starten wir eine Deep-Research wie bei den drei vorherigen Runs — breit und tief, mit konkreten Library-Empfehlungen, Code-Patterns, Timeline-Vorschlägen.

**Meine persönliche Empfehlung — in dieser Reihenfolge:**

- **Rang 1 zuerst** (Migration-Playbook) — ohne das baust du den Plan an deinem existierenden System vorbei
- **dann Rang 2** (News-Ground-Truth) — verhindert dass du Monate auf eine fragwürdige Innovation setzt
- **dann Rang 3** (Golden-Equity-Tests) — macht Refactoring sicher

Danach Rang 4-10 nach Bedarf.

---

## Gesamt-Bewertung: Bin ich jetzt zufrieden?

**Ehrlich: zu 85 %.** Die 15 % Unzufriedenheit kommen aus genau den 10 identifizierten Lücken plus den 7 neuen Punkten. Am dringendsten davon sind:

1. Migration-Playbook
2. News-Ground-Truth
3. Golden-Equity-Tests
4. Multi-Env-Setup
5. Data-Quality-Gate

Wenn diese fünf auch abgearbeitet sind, wäre ich zu ~95 % zufrieden. Das letzte 5 % sind Sachen, die man nur in der Umsetzung lernt — kein Plan überlebt den ersten Kontakt mit der Realität vollständig.

**Der Plan ist ein ausgezeichnetes Fundament.** Die Lücken sind nicht strukturell, sondern operativ — sie schließen sich durch die identifizierten Recherche-Runs.
