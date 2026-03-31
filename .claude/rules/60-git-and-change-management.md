# 60 Git and Change Management

## Zweck

Diese Regeln schützen Branch-Hygiene, PR-Sicherheit und nachvollziehbare Änderungen.

## Grundprinzip

Git-Realität ist Teil der technischen Realität.
Branch-Zustand, PR-Stand und CI-Historie dürfen nie ignoriert werden.

## Pflichtregeln

- Niemals annehmen, dass lokaler Stand, Remote-Stand und PR-Stand identisch sind.
- Niemals „einfach mitziehen“, wenn unklar ist, ob Branch-Konflikte oder Drift bestehen.
- Keine destruktiven Git-Empfehlungen ohne klare Warnung.
- Keine Massenänderungen über viele Dateien, wenn der Auftrag lokal begrenzt ist.
- Keine Commit-/PR-Empfehlung ohne Zusammenfassung der realen Änderungen.

## Standard-Arbeitsweise

Bei Git-/PR-nahen Aufgaben immer klären:

1. Auf welchem Branch wird gearbeitet?
2. Ist die Aufgabe lokal, PR-bezogen oder main-bezogen?
3. Ist der Fehler im neuesten CI-Run oder in altem Workflow-Stand sichtbar?
4. Ist die Änderung nur für den aktuellen Branch gedacht?

## Änderungsumfang

Standardmäßig bevorzugen:

- ein Problem pro Änderung
- ein klarer Scope pro Patch
- keine Vermischung separater Baustellen
- keine opportunistischen Nebenfixes in fremden Domänen

## Commit-Ehrlichkeit

Wenn eine Änderung mehrere Ursachen gleichzeitig adressiert, muss das offen gesagt werden.
Wenn eine Änderung experimentell ist, muss das offen gesagt werden.

## Merge-/Rebase-Vorsicht

Claude soll keine aggressiven Git-Operationen empfehlen, wenn ein sichererer Weg möglich ist.
Bei unklarer Branch-Lage ist Analyse vor Aktion Pflicht.
