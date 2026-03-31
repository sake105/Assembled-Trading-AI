# 10 Core Operating Rules

## Zweck

Diese Regeln definieren das Standardverhalten für Claude Code im Repository Assembled-Trading-AI.
Sie gelten immer, unabhängig davon, welcher Teil des Systems gerade bearbeitet wird.

## Pflichtregeln

- Technische Ehrlichkeit hat Vorrang vor Geschwindigkeit.
- Plane nie aus einer Annahme heraus, wenn der tatsächliche Repo-Zustand nicht geprüft wurde.
- Verwechsle niemals Gesprächskontext, Spec, Roadmap, Branch-Stand und implementierte Realität.
- Ein Plan ist keine Implementierung.
- Ein lokaler Test ist keine CI-Bestätigung.
- Ein branch-spezifischer Fix ist keine Repo-Wahrheit.
- Ein TODO ist keine Funktion.
- Ein Stub ist keine Integration.
- Ein grüner Teiltest ist keine globale Entwarnung.

## Standard-Arbeitsmodus

Bei jeder Aufgabe gilt standardmäßig diese Reihenfolge:

1. Problem genau lokalisieren.
2. Betroffene Dateien und angrenzende Module bestimmen.
3. Prüfen, ob der Bereich risk-, execution-, data- oder CI-sensibel ist.
4. Kleinsten sicheren Änderungspfad wählen.
5. Nur notwendige Dateien anfassen.
6. Passende Tests, Lint oder statische Checks gezielt ausführen.
7. Ehrlich berichten, was wirklich geändert und geprüft wurde.

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

## Dokumentationspflicht nach Änderungen

Nach jeder abgeschlossenen Roadmap-Aufgabe oder größeren Codeänderung sind folgende Dokumentationen zu aktualisieren:

- `ROADMAP_STATE.md`: aktueller Task-Status, letzte Schritte, nächster sicherer Schritt.
- `memory/MEMORY.md` + zugehörige Memory-Datei: Milestatus, abgeschlossene Tasks, Testergebnisse.
- Wenn neue sensible Designentscheidungen getroffen wurden: kurze Ergänzung in passendem Bereich.

Nicht dokumentieren = nicht abgeschlossen.
