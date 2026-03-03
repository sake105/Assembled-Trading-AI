Cursor Prompts — Standards
==========================

Ziel: Prompts so formulieren, dass Änderungen **klar, testbar und nachvollziehbar** sind.

## Struktur eines guten Prompts

- **Kontext**
  - Kurze Beschreibung des Ziels (z.B. „Cash-Gate fixen und MTM-Equity korrekt machen“).
  - Relevante Dateien/Module nennen.

- **Aufgabe**
  - Klarer Auftrag („Implementiere…“, „Füge Tests hinzu…“, „Schreibe Doc…“).
  - Begrenze den Scope (z.B. „kein Refactor außerhalb von…“).

- **Definition of Done (DoD)**
  - Welche Tests müssen laufen?
  - Welche Dateien/Artefakte müssen existieren/angepasst werden?
  - Welche Linter/QC-Gates sind zu beachten?

## Beispiele (Skeleton)

- **Bugfix-Prompt**
  - Kontext: aktuelles Fehlverhalten + Files.
  - Aufgabe: „Fix Cash-Gate so, dass Cash nie negativ wird, füge Regression-Tests hinzu.“
  - DoD: „cash_min >= -1e-6 in Synthetic-Run; Tests X/Y grün.“

- **Feature-Prompt**
  - Kontext: gewünschtes neues Feature (z.B. Weekly-Rebalance-Knopf).
  - Aufgabe: „CLI-Flag + Engine-Wiring + kleiner Test.“
  - DoD: „Test `test_rebalance_weekly_reduces_trades` grün; Backwards-Kompatibilität gewahrt.“

