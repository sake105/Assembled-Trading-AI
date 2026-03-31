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
