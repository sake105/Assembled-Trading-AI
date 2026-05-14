# 90 Subagents, Hooks and Automation

## Zweck

Diese Regeln definieren, wie Automatisierung in diesem Projekt eingesetzt werden soll.

## Grundprinzip

Automatisierung ist in diesem Repo nur dann gut, wenn sie:

- Sicherheit erhöht
- Drift reduziert
- Prüfungen konsistenter macht
- den Hauptkontext entlastet
- keine unkontrollierten Nebenwirkungen erzeugt

## Subagent-Regeln

Subagents sollen spezialisiert statt universell sein.
Empfohlene Spezialisierungen:

- CI / Test-Debugger
- Risk / Execution Reviewer
- Docs / Spec Sync
- Packaging / Imports / Tooling
- Research / Intel / später Frontend

## Subagent-Pflichten

- enger Scope
- klare Zuständigkeit
- keine unnötige Querarbeit
- Ergebnis immer komprimiert an Hauptagent zurückgeben
- keine zweite unabhängige Projektphilosophie entwickeln

## Konkrete Routing-Policy

Subagents sind Default-Ausführungsmodus für spezialisierte Arbeit.
Nicht auf explizite User-Aufforderung warten, wenn die Aufgabe klar zu einem Spezialbereich passt.

### Automatische Erzwingung via Stop-Hook (seit 2026-05-14)

Nach jedem Step mit Code-Änderungen in geschützten Pfaden (`src/`, `scripts/`, `.github/workflows/`, `.claude/rules/`, `CLAUDE.md`) erzwingt der Stop-Hook `.claude/hooks/stop_review_chain.py` die Review-Kette automatisch — siehe CLAUDE.md §20 und Spec `docs/superpowers/specs/2026-05-14-review-chain-design.md`.

### Routing-Regeln (für nicht-erzwungene Pfade und ad-hoc Reviews)

- **`ci-debugger`** proaktiv bei: CI-Failures, Workflow-Failures, plattformspezifischer Test-Divergenz (Windows vs. Ubuntu), Dependency-Drift, Collection-Failures, Artifact-Konflikten, Local-vs-CI-Mismatches.
- **`test-runner`** proaktiv bei: gezielter Testausführung, Marker-Handling, Regression-Validierung, Failing-Test-Triage, Minimal-Repro-Verifikation.
- **`risk-execution-reviewer`** proaktiv bei jeder Aufgabe, die `src/assembled_core/execution/`, `src/assembled_core/risk/`, `src/assembled_core/paper/`, `src/assembled_core/pipeline/`, Portfolio-Constraints, Order-Generierung, Pre-Trade-Checks, Kill-Switch-Logik oder cost-aware Execution betrifft.
- **`docs-governance-sync`** proaktiv bei Änderungen an `CLAUDE.md`, `.claude/rules/`, `AGENTS.md`, `.cursor/rules/`, `docs/cursor/` oder jedem Agent-Governance-/Repo-Instruction-Layer.
- **`memory-tracker`** proaktiv, wenn eine Session bedeutsame Entscheidungen, Statuswechsel, Debug-Conclusions, Governance-Änderungen oder neue Risk-Annahmen produziert hat, die über Sessions hinweg stabil bleiben sollen.
- **`senior-code-reviewer`** (NEU 2026-05-14): Stage 2 der Review-Kette. Automatisch vom Stop-Hook getriggert nach den Spezialisten. Auch manuell aufrufbar für ad-hoc Code-Review.
- **`task-completion-auditor`** (NEU 2026-05-14): Stage 3 der Review-Kette. Automatisch vom Stop-Hook getriggert nach `senior-code-reviewer`. Auch manuell aufrufbar bei „bin ich wirklich fertig?"-Zweifeln.

### Pflichtverhalten

- Stop-Hook-Erzwingung **kann nicht umgangen werden** durch „ich vergesse mal". Der Hook blockiert das Stop-Event bis Marker geschrieben ist.
- Außerhalb der Erzwingungs-Pfade: Spezialist-Delegation **bevorzugen**, nicht als Zusatzoption behandeln.
- Sensible Zonen nie ohne Spezialdelegation überspringen, außer mit explizitem Grund.

### Prioritätsreihenfolge bei Konflikt

1. `risk-execution-reviewer`
2. `senior-code-reviewer`
3. `task-completion-auditor`
4. `ci-debugger`
5. `test-runner`
6. `docs-governance-sync`
7. `memory-tracker`

## Hook-Regeln

Hooks sollen zuerst Guardrails und Ehrlichkeit verbessern, nicht blind Vollautomatik auslösen.

## Gute frühe Hook-Einsätze

- Schutz vor Lesen sensibler Dateien
- Schutz vor gefährlichen Git-Befehlen
- leichter Reminder nach Dateiänderungen für passende Tests
- Abschlusscheck für ehrliche Statuszusammenfassung

## Schlechte frühe Hook-Einsätze

- nach jeder kleinen Änderung die volle Test-Suite starten
- aggressive Auto-Fixes ohne Sichtbarkeit
- automatisches Umschreiben vieler Dateien
- komplexe Kettenreaktionen bei jedem Tool-Use

## Automationsprinzip

Immer zuerst:

1. klein
2. transparent
3. reversibel
4. pfadbegrenzt
5. projektkonform

## Priorisierung

- Must-Have: Guardrails
- Should-Have: gezielte Review-/Test-Hinweise
- Experimental: größere Auto-Fix- oder Multi-Agent-Loops
