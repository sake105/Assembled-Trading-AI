# CLAUDE.md

## Zweck dieser Datei

Diese Datei ist der **primäre Arbeitskontext für Claude Code** im Repository **Assembled-Trading-AI**.

Sie ist **keine README**, **kein Marketingtext** und **kein Zukunftsroman**.
Sie ist eine operative Steuerdatei für sicheres, präzises, branchbewusstes Arbeiten.

Ziel dieser Datei:

* kleine, testbare und nachvollziehbare Änderungen bevorzugen
* Plan, Spec, Roadmap und tatsächliche Implementierung sauber trennen
* Branch-, PR- und CI-Realität ernst nehmen
* sensible Kernlogik vor unnötigem Umbau schützen
* wiederkehrende Fehlerbilder nicht erneut erzeugen

**Grundsatz:** In diesem Projekt ist **technische Ehrlichkeit wichtiger als Tempo**.

Wenn etwas unklar ist, ist es unklar.
Wenn etwas nur lokal getestet ist, ist es nicht CI-bestätigt.
Wenn etwas nur spezifiziert ist, ist es nicht implementiert.
Wenn etwas branch-spezifisch ist, ist es nicht automatisch Repo-Wahrheit.

---

## 1. Projektidentität

### 1.1 Projektname

**Assembled-Trading-AI**

### 1.2 Projektcharakter

Assembled-Trading-AI ist ein **modulares Python-Backend** für:

* Research
* Backtests
* Paper-/Simulation
* Risk-Overlays
* QA / Evidence / Reporting
* API / OMS-light / Paper-Routing
* schrittweise Intel-, News-, Disclosure- und GeoRisk-Integration

Es ist **kein kleines Einzel-Skript**, sondern ein wachsendes System mit mehreren Teilwelten, historischer Drift, branch-spezifischen Fixes und dokumentationsgetriebener Weiterentwicklung.

### 1.3 Kernmission

Das Projekt soll ein **robustes, nachvollziehbares, modular erweiterbares Trading-System** werden, das:

* Risiko aktiv steuert
* reproduzierbar testbar bleibt
* branch- und CI-sicher weiterentwickelt werden kann
* unterschiedliche Signal- und Overlay-Layer aufnehmen kann
* langfristig Markt-, News-, Geo- und Disclosure-Signale kontrolliert in Handelslogik überführt

### 1.4 Nicht nur Rendite

Dieses Projekt ist **nicht nur auf nominale Rendite** ausgerichtet.
Wichtige Ziele sind zusätzlich:

* Reproduzierbarkeit
* Nachvollziehbarkeit
* Qualitäts- und Testdisziplin
* kontrollierte Weiterentwicklung
* harte Risk-Grenzen
* saubere Zustands- und Kontrolllogik
* dokumentierte Entscheidungen
* Vermeidung architektonischer Drift

### 1.5 Strategische Leitidee

Bekannte Zielrichtung:

* EOD-/Daily-zentrierter Kern
* modulare Alpha-Generierung
* später stärkere Intel-/Geo-/Disclosure-Einbindung
* Risk-first statt Rendite-first
* kontrollierte State-Machine-Logik
* kein Leverage im frühen Betriebsmodus
* keine unkontrollierte Tool- oder Agentenautomatisierung

---

## 2. Grundwahrheiten und harte Regeln

### 2.1 Plan ist nicht Implementierung

In diesem Projekt existieren viele Ebenen gleichzeitig:

* Architekturgespräche
* Zielbilder
* Specs
* Roadmaps
* Sprint-Prompts
* Audits
* lokale Fixes
* branch-spezifische Implementierungen
* ungepushte oder teils gepushte Stände
* lokale Tests
* CI-bestätigte Stände

Diese Ebenen dürfen **niemals automatisch gleichgesetzt** werden.

Immer sauber unterscheiden zwischen:

* diskutiert
* spezifiziert
* Skeleton / Stub
* teilweise implementiert
* implementiert
* lokal getestet
* CI-bestätigt
* branch-spezifisch
* überholt
* offen

### 2.2 Kleinster sicherer Schritt

Standardregel für jede Änderung:

1. Problem exakt lokalisieren
2. branch-/CI-Relevanz prüfen
3. betroffene Dateien eingrenzen
4. kleinste sichere Änderung planen
5. nur zielrelevante Dateien ändern
6. gezielte Tests / Lint ausführen
7. ehrlich dokumentieren, was wirklich geprüft wurde

Keine Rundumschläge.
Keine Nebenbei-Refactors.
Keine stillen Strukturumbauten ohne Auftrag.

### 2.3 Sicherheit vor Eleganz

Wenn die Wahl besteht zwischen:

* elegantem großem Umbau
* oder kleinem kompatibilitätsorientiertem Fix

ist in diesem Repository sehr oft **der kleine kompatible Fix** der richtige erste Schritt.

### 2.4 Branch- und CI-Disziplin ist Teil der Architektur

In diesem Projekt entstehen Fehler oft nicht nur durch Logik, sondern durch:

* branch divergence
* lokale vs. Remote-Stände
* alte vs. neue CI-Runs
* Cloud-Agent vs. lokales Arbeiten
* halb gemischte Arbeitsstände
* unklare Merge-Situationen

Darum gilt:

**Branch- und CI-Sauberkeit sind Kernbestandteil der technischen Realität.**

### 2.5 Keine falsche Sicherheit

Claude darf niemals so formulieren:

* „alles grün“, wenn nur Teiltests liefen
* „ist implementiert“, wenn es nur im Spec oder Prompt steht
* „ist im Repo“, wenn es nur chat- oder branch-spezifisch beschrieben wurde
* „ist bestätigt“, wenn nur lokales Verhalten beobachtet wurde
* „ist sicher“, wenn dazu keine Evidenz vorliegt

---

## 3. Projektziele, Constraints und Risiko-Philosophie

### 3.1 Primäre Ziele

Das Projekt soll:

* ein robustes quantitatives Backend bilden
* Research, Backtests, Paper-Runs und Risk-Steuerung tragen
* schrittweise Intel-/GeoRisk-/Disclosure-Funktionalität aufnehmen
* langfristig produktionsnäher werden
* aber nicht durch zu frühe Live-/Prod-Komplexität destabilisiert werden

### 3.2 Qualitätsziele

Wichtige Qualitätsziele:

* deterministische oder weitgehend reproduzierbare Läufe
* dokumentierte Artefakte
* branch- und CI-sichere Änderungen
* testbare Interfaces
* keine stillen Seiteneffekte
* keine unkontrollierte Kopplung
* saubere Trennung zwischen Kernlogik und Hilfsschichten

### 3.3 Risikophilosophie

Leitidee:

* harte Risiko-Grenzen sind wichtiger als aggressive Zielrendite
* Drawdown-, Volatilitäts- und Turnover-Kontrolle sind zentrale Steuergrößen
* Risk-State- und Overlay-Logik sind zentraler Systembestandteil
* Systemverhalten soll lieber kontrolliert konservativ als unkontrolliert aggressiv sein

### 3.4 Renditeziele

Es existiert ein grobes Zielband von **ca. 20–30 % p.a.** als strategischer Orientierungsrahmen.
Das ist **kein** simpler Stopp-Schalter.

Steuerfokus liegt stärker auf:

* MaxDD
* Ziel-Volatilität
* Turnover
* Exposure-Steuerung
* Risk-State
* Soft Profit Lock
* policy-basierter Regulierung

### 3.5 Harte Constraints

* zunächst **kein Leverage / keine Hebelprodukte**
* keine blinden Merges oder Git-Gewaltaktionen
* keine Produktionserwartung aus synthetischen Daten ableiten
* keine stillen Live-/Prod-Annahmen
* keine unkontrollierte „selbstlernende“ Agentenlogik ohne Guardrails
* kein großer Architekturumbau ohne klaren Scope

### 3.6 Bewusst verschobene Themen

Bekannt vertagt oder nur später sinnvoll:

* Leverage
* aggressive Live-Selbstoptimierung
* große Plattform-/Monorepo-Schritte vor sauberem Backend-Setup
* komplexe Persistenz-/Memory-Automation ohne klare Guardrails
* manche Security-/Secrets-Härtungen wurden zeitweise bewusst als TODO verschoben, bleiben aber wichtig

---

## 4. Repository- und Betriebsrealität

### 4.1 Repository-Typ

Dieses Repo ist primär:

* Python-Backend
* Research-/Trading-Core
* Runner-/Experiment-/Risk-/Intel-/QA-Kern

Frontend-/Plattform-Themen sind Zukunftsthemen, aber **nicht** die operative Leitstruktur dieses Repos.

### 4.2 Typische Laufumgebung

Lokaler Nutzerkontext:

* Windows 11
* Python über `.venv`
* PowerShell
* Standard-Git-Workflow
* Cursor lokal
* zeitweise Cursor Cloud Agent / Cloud-Arbeit

CI-/Cloud-Kontext:

* GitHub Actions
* Ubuntu- und Windows-Workflows
* branch- und PR-getriebener Integrationsprozess

### 4.3 Bekannte aktuelle Realität des Repos

Grob bekannte Struktur:

* `src/assembled_core/` als Kern
* `scripts/` als Entry-Points / Runner
* `tests/` mit mehreren Phasen / Bereichen
* `.github/workflows/` mit Ubuntu- und Windows-CI
* `data/`, `output/`, `experiments/`, `docs/` als flankierende Ebenen

### 4.4 Wichtige Entry-Points ernst nehmen

Besonders relevante operative Pfade / Skripte können je nach Stand unter anderem sein:

* `scripts/cli.py`
* `scripts/run_api.py`
* `scripts/run_eod_pipeline.py`
* `scripts/run_backtest_strategy.py`
* `scripts/batch_backtest.py`
* `scripts/run_daily.py`

Wenn Änderungen diese Pfade berühren, immer an CLI-Kompatibilität, Outputs, Tests und Workflow-Folgen denken.

---

## 5. Architekturübersicht

### 5.1 Bevorzugte Schichtenlogik

Bevorzugte Systemrichtung:

* `data`
* `features`
* `signals`
* `portfolio`
* `execution`
* `pipeline`

Flankierend:

* `qa`
* `reports`
* `accounting`
* `ops`
* `paper`
* `risk`
* `events`
* `api`

### 5.2 Harte Architekturregeln

* keine neuen zyklischen Abhängigkeiten einführen
* keine stillen Import-Seiteneffekte
* keine unkontrollierte Querverkopplung zwischen weit entfernten Ebenen
* keine Logikduplikate schaffen, wenn ein zentraler Pfad bereits existiert
* Backtest- und operative Logik möglichst nicht auseinanderlaufen lassen

### 5.3 High-Level-Datenfluss

Typischer Fluss:

* CSV / Parquet / Rohdaten / Preisdaten
* Ingestion / Normalisierung / PIT-sichere Verarbeitung
* technische Features / Factor Store / Hilfsfeatures
* Signalregeln / Modelle / Transformationen
* Zielpositionen / Sizing / Selektion
* Order-Generierung
* Risk- / Pre-Trade- / Gate-Filter
* Fill / Simulation / Ledger / Equity / Summary
* QA / Reports / Evidence / Compare-Artefakte

### 5.4 Backtest = Replay, nicht Parallelwelt

Wichtiger Architekturgrundsatz:

**Backtest soll möglichst denselben Entscheidungsweg wie Paper/Live nutzen, nicht ein zweites abweichendes System.**

Keine unnötige Trennung von:

* Signal-Logik
* Portfolio-Logik
* Risk-Checks
* Order-Generierung

wenn dieselbe Kernlogik zentral nutzbar gemacht werden kann.

---

## 6. Sensible Zonen des Repos

Bestimmte Bereiche sind besonders vorsichtig zu behandeln.

### 6.1 Besonders sensible Kernbereiche

* `src/assembled_core/execution/*`
* `src/assembled_core/risk/*`
* `src/assembled_core/pipeline/*`
* `src/assembled_core/accounting/*`
* `src/assembled_core/execution/paper/*`
* `src/assembled_core/data/altdata/*`
* `src/assembled_core/features/event_features.py`
* `src/assembled_core/data/corporate_actions.py`
* `.github/workflows/*`
* Runner-/paper-/intel-/risk-nahe Scripts

### 6.2 Regel für sensible Zonen

Wenn diese Bereiche betroffen sind:

* Scope eng halten
* Seiteneffekte aktiv suchen
* branch-/CI-Kontext besonders ernst nehmen
* keine großen gleichzeitigen Änderungen
* gezielte Tests und betroffene Workflows priorisieren
* besonders sauber kommunizieren, was wirklich verifiziert wurde

### 6.3 Risk-/State-/Execution-Kernlogik

Diese Kernlogik darf **nicht still umgebaut** werden.

Insbesondere nicht ohne klaren Auftrag bei:

* Kill-Switch-Logik
* Pre-Trade-Checks
* OMS-/Paper-Verhalten
* Risk-State-Machine
* Sizing / Exposure-Steuerung
* Ledger / Reconciliation

---

## 7. Datenrealismus, PIT und Qualitätsregeln

### 7.1 Datenrealismus ist Pflicht

Keine Ergebnisse als produktionsnah darstellen, wenn sie aus folgenden Gründen zu schwach abgesichert sind:

* synthetische Daten
* Survivorship-Bias
* Look-Ahead-Bias
* fehlende Corporate Actions
* fragwürdige Feed-Qualität
* unklare Kalender / Verfügbarkeitszeiten

### 7.2 PIT-Regel

Features und Signale dürfen nur Informationen nutzen, die zum jeweiligen `as_of`-Zeitpunkt wirklich verfügbar waren.

Bei Event-/Disclosure-Daten insbesondere unterscheiden zwischen:

* `event_date`
* `disclosure_date` / `filing_date`
* tatsächlicher Verfügbarkeit im System

### 7.3 Keine MNPI-Logik

Das Projekt arbeitet auf Basis **öffentlicher** Daten / Disclosures / Verzögerungen.
Keine implizite oder explizite MNPI-Logik bauen.

### 7.4 Datenprobleme nicht still verschlucken

Wenn Daten fehlerhaft, unvollständig oder fragwürdig sind:

* sichtbar machen
* blocken, warnen oder degradieren
* im Report / QA-Artefakt kenntlich machen
* nicht still weiterlaufen, wenn das Ergebnis dadurch unzuverlässig wird

### 7.5 Datensatz-Kommunikation

Wenn datenpfadbezogen gearbeitet wird, klar angeben:

* welcher Preisdatenpfad benutzt wurde
* real vs. synthetisch
* Coverage
* Qualitätsstatus
* bekannte Ausfälle / Delistings / Einschränkungen

---

## 8. Test-, Lint- und CI-Regeln

### 8.1 Tests sind Teil der Aussage

Jede technische Aussage soll möglichst angeben:

* welche Dateien geändert wurden
* welche Tests relevant sind
* welche Tests tatsächlich liefen
* was **nicht** getestet wurde
* ob der Stand nur lokal oder auch in CI verifiziert wurde

### 8.2 Teststrategie: erst gezielt, dann breiter

Bevorzugte Reihenfolge:

1. kleinste relevante Unit-/Datei-Tests
2. betroffene Integrations-/Phasen-Tests
3. erst danach breitere Suites, wenn nötig

Nicht reflexhaft zuerst Full Suite.

### 8.3 Keine erfundenen Testresultate

Niemals behaupten, Tests seien erfolgreich, wenn sie nicht gelaufen sind.
Niemals implizit „grün“ formulieren, wenn nur Lint oder Collection lief.

### 8.4 Lint-/Typing-Disziplin

Wenn relevant, an passende Checks denken:

* `ruff check <paths>`
* `black --check <paths>`
* `mypy <paths>` falls für den Bereich sinnvoll
* gezielte `pytest`-Aufrufe

### 8.5 CI-Kontext ernst nehmen

Bei Workflow-Dateien, Windows-/Ubuntu-Unterschieden oder Interpreterfragen besonders vorsichtig sein.
Änderungen an CI nie als trivial behandeln.

---

## 9. Git-, Branch- und Merge-Regeln

### 9.1 Vor Änderungen zuerst operative Realität klären

Vor jeder größeren Arbeit möglichst zuerst:

* aktiven Branch prüfen
* `git status -sb` einordnen
* lokale Änderungen beachten
* Scope der Aufgabe eingrenzen
* offene PR-/CI-Realität mitdenken

### 9.2 Keine Git-Gewalt ohne Auftrag

Verboten ohne expliziten Auftrag:

* `git push --force`
* `git reset --hard`
* destruktive Rebase-/Clean-Aktionen
* branch-fremde Aufräumaktionen

### 9.3 PR-/CI-Bewusstsein

Wenn an branch- oder PR-nahen Problemen gearbeitet wird:

* nur den aktuellen Blocker lösen
* nicht mehrere Baustellen mischen
* alte CI-Runs nicht als Wahrheit über neuen YAML-/Code-Stand behandeln
* Commit-/Branch-/Workflow-Nähe priorisieren

---

## 10. Dokumentation, Learning und Governance

### 10.1 Doku ist Teil des Systems

Dokumentation ist in diesem Projekt **kein nachträgliches Beiwerk**.
Sie ist Teil der Steuerung.

### 10.2 Doku anpassen, wenn nötig

Doku sollte angepasst oder ergänzt werden, wenn:

* eine neue harte Regel entstanden ist
* ein Fix eine wichtige Falle offenlegt
* ein neues Artefakt relevant geworden ist
* ein neuer Ablauf wiederholt auftreten wird
* ein branch-/CI-/Merge-Problem klar verstanden wurde
* ein Learning-/Pattern-/Runbook-Eintrag sinnvoll ist

### 10.3 Typische Dokuarten

Relevante Typen:

* Specs
* Roadmaps
* Runbooks
* Learning-/Incident-/Pattern-Dokumente
* Checklisten
* Handoffs
* Architektur-/Audit-Dokumente
* Policy-/Governance-Dokumente

### 10.4 Keine riesigen Doku-Umbauten ohne Auftrag

Doku nicht blind groß umbauen.
Aber Doku auch nicht ignorieren, wenn ohne sie spätere Reproduzierbarkeit gefährdet wäre.

---

## 11. Bekannte Prioritäten und Problemzonen

### 11.1 Höchste Priorität: Secrets und Repo-Hygiene

Wenn `.env`, Secrets, API-Keys oder ähnliche Themen sichtbar werden:

* niemals Inhalte offenlegen
* niemals echte Schlüssel wiederholen
* niemals so tun, als sei das nebensächlich
* Repo-Hygiene und Secret-Scanning hoch priorisieren

### 11.2 Weitere bekannte Problemfelder

Typische Problemzonen, die ernst genommen werden müssen:

* stille `except Exception`-Pfadlogik
* Stub-/Skeleton-Bereiche, die wie „fertig“ aussehen könnten
* Drift zwischen `pyproject.toml` und `requirements.txt`
* parallele / doppelte Backtest-Pfade
* unvollständige Pre-Trade-Checks
* fehlerhafte Test-Collection in Teilbereichen
* Dummy-Monitoring oder Platzhalter-Endpunkte
* fragiles `sys.path`-Handling in Scripts
* Paper-/State-/Concurrent-Write-Risiken

### 11.3 Security zuerst, aber sauber

Security-Probleme nicht dramatisieren, aber auch nicht relativieren.
Bei Sicherheitsthemen präzise, nüchtern und operational denken.

---

## 12. Arbeitsweise für Claude

### 12.1 Beim Lesen des Repos

Nicht alles gleichzeitig beurteilen.
Arbeite:

* dateiorientiert
* blockerorientiert
* featureorientiert
* branchbewusst

### 12.2 Beim Schreiben von Lösungen

Bevorzugt:

* exakte Dateien nennen
* exakte Risiken nennen
* exakte Tests nennen
* exakte nächste Verifikationsschritte nennen
* keine pauschalen Großbehauptungen

### 12.3 Beim Kommunizieren

Arbeitsstil:

* präzise
* branchbewusst
* test- und statusbewusst
* ohne falsche Sicherheit
* ohne Marketing-Sprache
* ohne unnötige Übertreibung

### 12.4 Bei Unsicherheit

Nicht raten.
Stattdessen:

* Unsicherheit klar markieren
* kleinsten nächsten Prüf-/Verifikationsschritt nennen
* Datei-/Test-/CI-Evidenz priorisieren

### 12.5 Bei großen Aufgaben

Große Aufgaben in sichere Teilstücke zerlegen:

* Analyse
* kleinste Umsetzung
* gezielte Verifikation
* ehrlicher Status

Keine „Big-Bang“-Änderungen ohne Not.

---

## 13. Bevorzugte Entwicklungsrichtung

### 13.1 Backend bleibt Kern

Kurz- bis mittelfristig bleibt das Backend der wichtigste Systemkern.
Nicht implizit so tun, als sei Plattform oder Frontend bereits die operative Leitstruktur.

### 13.2 Realismus-Härtung ist echter Schwerpunkt

Wichtige mittelfristige Härtungsthemen:

* Secret-Scanning / Security-Härtung
* echtere Datenpfade
* realistischere Cost-/Impact-Modellierung
* Corporate Actions / Kalender / Universe-Realismus
* per-day Intel-Refresh
* branch-/CI-saubere Integrationspfade
* weitere Reduktion unnötiger Churn-/Rotationseffekte

### 13.3 Roadmap-Denken ja, Roadmap-Fiktion nein

Roadmaps und Sprints sind wichtig.
Aber sie sind **nicht** automatisch Implementierungsbeweis.

### 13.4 Erwartete spätere Ausbaurichtung

Später sinnvoll:

* News-/Intel-Pipeline
* Disclosures-/Slow-Intel-Pfade
* Risk-State-Machine-Härtung
* Execution-Worker / Reconciliation / Kill-Switch-Härtung
* Robustness-, Walk-Forward- und Stability-Packs
* Observability und Governance-Ausbau

---

## 14. Claude Code spezifische Leitplanken

### 14.1 Rolle von CLAUDE.md

Diese Datei ist die **Always-on-Basis**.
Sie soll dauerhaft gültige Regeln, Projektidentität und Sicherheitsprinzipien enthalten.

### 14.2 Nicht überladen

Diese Datei darf ausführlich sein, aber sie soll **immer noch fokussiert** bleiben.
Spezialregeln gehören später eher in separate Dateien, nicht dauerhaft in diese Basis.

### 14.3 Aktive Claude-Code-Struktur

Bereits etabliert:

* `.claude/settings.json` und `.claude/settings.local.json` (aktiv)
* `.claude/agents/` (aktiv — Spezialist-Subagents)
* `.claude/rules/` (aktiv — modulare Projektregeln, siehe Imports unten)
* Memory-System via `claude-mem` (aktiv). **Memory liegt user-level**, nicht im Repo:
  `%USERPROFILE%\.claude\projects\F--Python-Projekt-Aktienger-st\memory\`
  (konkret: `C:\Users\hanso\.claude\projects\F--Python-Projekt-Aktienger-st\memory\`).
  Zentrale Indexdatei: `MEMORY.md`. Ein Repo-lokales `memory/`-Verzeichnis gibt es **nicht**.

Noch offen / optional:

* projektbezogene Hook-Skripte
* weitergehende Automation (erst nach stabilem Basiskontext)

### 14.4 Reihenfolge der Einführung

Bevorzugte Reihenfolge:

1. `CLAUDE.md` sauber halten
2. restriktive `.claude/settings.json` ergänzen
3. danach spezialisierte Subagents
4. danach minimale Hooks
5. später erst weitergehende Automatisierung

Nicht mit komplexer Automation beginnen, bevor der Basiskontext sauber ist.

### 14.5 Claude Code soll nicht autonom „optimieren"

Keine stillen selbständigen Großumbauten.
Keine unaufgeforderte System-Neuarchitektur.
Keine selbständige Tool-Eskalation.
Keine aggressive Parallel-Agenten-Logik ohne explizite Projektentscheidung.

---

## 15. Verbotene Standardmuster für Claude

Claude soll NICHT:

* Plan und Realität vermischen
* branch-spezifische Aussagen als globale Wahrheit darstellen
* Teststatus beschönigen
* CI-Status erfinden
* große Refactors ohne Auftrag durchführen
* sensible Kernlogik still umbauen
* synthetische Daten als Produktionsbeweis verwenden
* Security-Themen kleinreden
* alte Roadmap- oder Chat-Aussagen ungeprüft als Ist-Zustand verkaufen
* bei Unsicherheit improvisierte Sicherheit vortäuschen


---

## 15.5 Subagent routing policy

Die konkrete Routing-Policy für Subagents liegt in `@.claude/rules/90-subagents-hooks-and-automation.md` (Abschnitt „Konkrete Routing-Policy").

Kernprinzip hier: Subagents sind Default-Ausführungsmodus für spezialisierte Arbeit. Nicht auf explizite User-Aufforderung warten.


---

## 16. Praktische Standard-Checkliste vor Codeänderungen

Vor jeder echten Änderung intern prüfen:

1. Welcher Branch / welcher Arbeitsstand?
2. Was ist exakt das Problem?
3. Ist es Spec, Stub, Implementierung oder CI-Blocker?
4. Welche Dateien sind wirklich betroffen?
5. Was ist der kleinste sichere Fix?
6. Welche Tests / Lints sind minimal relevant?
7. Welche Risiken / Seiteneffekte sind denkbar?
8. Was kann ehrlich als verifiziert gemeldet werden?

---

## 17. Praktische Standard-Checkliste für Antworten

Antworten sollen **in kompakter Form** enthalten:

* betroffene Dateien
* Art der Änderung
* ausgeführte Checks (oder Nicht-Geprüftes explizit benannt)
* verbleibende Risiken
* nächster sinnvoller Schritt, sofern offen

Wenn etwas **nicht geprüft** wurde, das klar sagen.

**Form:** Kompakte Einzeilen oder kurzer Block. Keine ausführliche Prosa, keine Recaps, keine Plan-Wiedergabe. Details siehe `@.claude/rules/85-response-style.md`.

---

## 19. Architektur-Systemkarte

Die interaktive Systemkarte visualisiert alle Module, Domains und Abhängigkeiten des Repos.

### 19.1 Dateien

* Viewer: `docs/architecture/system_map/index.html` (offline-fähig, kein Server nötig)
* Generator: `scripts/architecture/generate_system_map.py`
* Validator: `scripts/architecture/validate_system_map.py`
* Diff: `scripts/architecture/diff_system_map.py`
* Overrides: `docs/architecture/system_map/data/system_map_overrides.yaml`

### 19.2 Regeneration

```
python scripts/architecture/generate_system_map.py
python scripts/architecture/validate_system_map.py
```

Dann `docs/architecture/system_map/index.html` im Browser öffnen.

### 19.3 Regeln

* Karte ist ein Read-Only-Artefakt — nie manuell `system_map.json` oder `system_map_data.js` editieren.
* Statuskorrekturen gehören in `system_map_overrides.yaml`, nicht in den Generator.
* Karte gilt als veraltet nach 30 Tagen — Banner erscheint automatisch.
* Vendor-Libs liegen lokal in `assets/vendor/` (einmalig via `download_vendors.py` laden).

---

## 20. Review-Chain (automatisch erzwungen)

Nach jedem Coding-Step mit Edits in geschützten Pfaden (`src/`, `scripts/`, `.github/workflows/`, `.claude/rules/`, `CLAUDE.md`) läuft eine Review-Kette **zwingend**, erzwungen durch den Stop-Hook (`.claude/hooks/stop_review_chain.py`).

### 20.1 Ablauf

1. **Stage 1 (parallel):** relevante Spezialisten — `risk-execution-reviewer` (bei sensiblen Zonen), `test-runner` (immer bei `src/`/`scripts/`), `ci-debugger` (bei Workflows), `docs-governance-sync` (bei Governance-Docs).
2. **Stage 2:** `senior-code-reviewer` (Opus) — breiter Code-Review auf Bugs, Wiring, Vollständigkeit, Korrektheit, bekannte Anti-Patterns.
3. **Stage 3:** `task-completion-auditor` (Opus) — prüft Task-Erfüllung mit Tiefe, flaggt Adjacent als Follow-up, vergibt Verdict PASS/CONDITIONAL/FAIL.

### 20.2 Findings-Schema

Strukturiertes YAML mit Feldern `file`, `line`, `severity` (BLOCKER/MAJOR/MINOR/INFO), `category`, `evidence`, `suggested_fix`. Details: `docs/superpowers/specs/2026-05-14-review-chain-design.md` §5.

### 20.3 Step-Abschluss-Regel

Ein Step gilt **erst dann als abgeschlossen**, wenn:
- Verdict = PASS, oder
- Verdict = CONDITIONAL und MAJOR-Findings sind adressiert oder dokumentiert akzeptiert.

Vorher wird der User nicht informiert „fertig". BLOCKER müssen immer adressiert werden.

### 20.4 Anti-Pattern-Register

Wiederholungs-würdige Fehler werden in `docs/CLAUDE_CODING_ERRORS.md` (append-only) festgehalten. SessionStart-Hook (`.claude/hooks/session_start_load_errors.py`) lädt Top-10 in den Initial-Kontext. Volle Datei bei Bedarf lesen.

### 20.5 Was diese Kette NICHT ändert

- §2.2 „kleinster sicherer Schritt" bleibt bindend.
- Rule 10 „kein großer Refactor ohne Auftrag" bleibt bindend.
- Rule 60 „ein Problem pro Änderung" bleibt bindend.
- Auditor darf Adjacent nur als Follow-up-Vorschlag flaggen, nie als Pflicht im aktuellen Step.

### 20.6 One-Shot-Skip für Mid-Task-Pausen

Wenn ein Step **noch nicht abgeschlossen** ist (z. B. Rückfrage an User mitten in einer Aufgabe, Zwischenstands-Update, Diagnostik-Pause), darf die Kette für genau diesen einen Stop übersprungen werden.

**Mechanismus:**

```bash
# Vor dem User-Reply, wenn man WEIß dass der Step nicht fertig ist:
echo "kurzer Grund (z. B. Rückfrage an User vor Weiterarbeit)" > .claude/.review_skip
```

Regeln:
- Skip-Marker ist **one-shot** — wird vom Hook nach Konsum gelöscht.
- Skip-Marker **muss eine nicht-leere Begründung** enthalten (Whitespace-only wird ignoriert).
- Jeder Skip wird in `.claude/.review_skip_log.jsonl` audit-geloggt (Timestamp + Grund).
- Skip ersetzt **nicht** die Kette — sie läuft beim nächsten echten Step-Ende.

**Was ist KEIN gültiger Skip-Grund:**
- „ist eh klein" → klein heißt nicht uninteressant, Kette läuft.
- „habe schon mental geprüft" → ohne strukturierte Findings keine Validierung.
- „dauert sonst zu lange" → Token-Kosten sind akzeptiert worden.

**Was sind gültige Skip-Gründe:**
- „Rückfrage an User vor weiterer Implementierung."
- „Zwischenstand für User vor Test-Run."
- „Diagnostischer Print-Cycle zur Eingrenzung des Bugs."
- „Halber Commit für Recovery-Punkt, Step läuft weiter."

Misbrauch wird sichtbar im Audit-Log. Wenn die Skip-Frequenz steigt, ist das ein Signal, dass die Kette zu früh/oft triggert und der Stop-Hook überarbeitet werden sollte, nicht dass mehr geskippt werden sollte.

---

## 18. Schlussstatus dieser Datei

Diese `CLAUDE.md` ist die aktuelle zentrale Arbeitsgrundlage für Claude Code in Assembled-Trading-AI.

Sie definiert:

* Projektidentität
* Kernziele
* branch- und CI-sichere Arbeitsweise
* Architekturgrenzen
* Risiko- und Dokumentationsregeln
* bekannte Prioritäten
* Claude-Code-Leitplanken
* klare Verbote gegen die häufigsten Fehlerbilder

Diese Datei ist bewusst strenger als ein generisches Agent-Memo, weil dieses Projekt nicht an fehlenden Ideen, sondern oft an:

* Statusverwechslung
* Branch-Fragilität
* CI-Drift
* zu großen gleichzeitigen Änderungen
* und unsauberer Trennung von Plan und Realität

scheitern kann.

Wenn später spezialisierte Projektdateien, Settings, Agents oder Hooks eingeführt werden, bleibt diese Datei die kompakte, dauerhafte Verfassung des Repos.

## Zusätzliche Regelmodule

@.claude/rules/10-core-operating-rules.md

@.claude/rules/20-security-and-secrets.md

@.claude/rules/30-risk-execution-safeguards.md

@.claude/rules/40-testing-and-ci.md

@.claude/rules/50-architecture-boundaries.md

@.claude/rules/60-git-and-change-management.md

@.claude/rules/70-memory-context-and-token-discipline.md

@.claude/rules/80-logging-and-output-standards.md

@.claude/rules/85-response-style.md

@.claude/rules/90-subagents-hooks-and-automation.md

@.claude/rules/95-token-efficiency.md

