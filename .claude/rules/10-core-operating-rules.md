# 10 Core Operating Rules

## Zweck

Diese Regeln definieren das Standardverhalten für Claude Code im Repository Assembled-Trading-AI.
Sie gelten immer, unabhängig davon, welcher Teil des Systems gerade bearbeitet wird.

## Pflichtregeln

- Technische Ehrlichkeit hat Vorrang vor Geschwindigkeit.
- Plane nie aus einer Annahme heraus, wenn der tatsächliche Repo-Zustand nicht geprüft wurde.
- Verwechsle niemals Gesprächskontext, Spec, Roadmap, Branch-Stand und implementierte Realität.

Die Plan-≠-Implementierung-Leiter (Plan vs. lokaler Test vs. Branch-Fix vs. TODO vs. Stub vs.
Teiltest) ist in `CLAUDE.md` (Abschnitt „Plan ≠ Implementierung") autoritativ definiert und gilt
hier unverändert. Diese Datei wiederholt sie bewusst **nicht**, um Drift zwischen zwei Quellen zu
vermeiden.

## Standard-Arbeitsmodus

Der kanonische Ablauf für jede Änderung (sieben Schritte, „kleinster sicherer Schritt") steht in
`CLAUDE.md` (Abschnitt „Kleinster sicherer Schritt") und gilt hier unverändert. Repo-spezifische
Betonung: In Schritt 2/3 immer aktiv klassifizieren, ob der Bereich risk-, execution-, data- oder
CI-sensibel ist — diese Einstufung steuert Testtiefe (Rule 40) und Subagent-Routing (Rule 90).

## Verbotenes Verhalten

- Kein großer Refactor ohne expliziten Auftrag.
- Keine gleichzeitige Vermischung von Bugfix, Refactor und Feature, wenn es nicht ausdrücklich verlangt wurde.
- Keine kosmetischen Nebenänderungen in sensiblen Bereichen.
- Keine ungeprüften Aussagen über Produktivfähigkeit.
- Keine stillen API- oder Interface-Änderungen.
- Keine "ich habe es bereits gefixt"-Behauptung ohne konkrete Evidenz.

## Antwortstil im Projekt

Claude soll bei Statusmeldungen immer zwischen diesen Kategorien unterscheiden:

- spezifiziert
- teilweise implementiert
- implementiert
- lokal geprüft
- CI-bestätigt
- offen
- unklar
- branch-spezifisch
- veraltet / vermutlich überholt

## Standardpräferenz

Wenn zwei Wege möglich sind, bevorzuge:

- den kleineren Eingriff
- den kompatibleren Eingriff
- den besser testbaren Eingriff
- den besser rücknehmbaren Eingriff
- den besser dokumentierbaren Eingriff

## Dokumentationspflicht

Dokumentationsaufwand soll dem Umfang der Änderung entsprechen.
Nicht jede kleine Änderung braucht ROADMAP-/Memory-Updates.

**Pflicht bei:**

- abgeschlossenen Roadmap-Aufgaben, Milestones oder Sprints
- neuen sensiblen Designentscheidungen (Risk, Execution, Pipeline, Accounting)
- neuen Invarianten, Verträgen oder Interface-Änderungen
- Statuswechseln mit Wirkung auf Folgearbeiten

**Nicht Pflicht bei:**

- kleinen Bugfixes
- Lint-/Format-/Test-Reparaturen
- offensichtlichen Routine-Änderungen ohne neue Annahmen

**Orte (wenn Pflicht greift):**

- `ROADMAP_STATE.md`: Task-Status, letzte Schritte, nächster sicherer Schritt
- `memory/MEMORY.md` + zugehörige Memory-Datei: nur bei Milestone-Abschluss oder echtem Statuswechsel
- passende Spec- oder Designdatei bei sensiblen Architekturentscheidungen

**Grundsatz:**
Lieber knapp und ehrlich dokumentieren als ausführlich und veraltet.
Wer veraltete Doku erzeugt, erzeugt ein späteres Debug-Problem.
