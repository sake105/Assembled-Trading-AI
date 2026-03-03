Cursor Rules — Zusammenarbeit mit dem AI-Agent
==============================================

Diese Datei legt fest, **wie** Cursor in diesem Repo arbeiten soll.

## Grundprinzipien

- **Kleine, fokussierte Änderungen**
  - Bevorzugt kleine, klar umrissene PRs/Commits statt großer „Rewrite“-Wellen.
  - Änderungen möglichst **file-by-file** und thematisch getrennt.

- **Tests sind Pflicht**
  - Keine funktionalen Änderungen ohne passende Tests (neu oder angepasst).
  - Vor dem Commit: relevante `pytest`-Suite lokal oder im CI ausführen.

- **Keine ungetesteten Refactors**
  - Refactors nur mit klarer Motivation (z.B. „Dead code removal“, „Performance“, „Lesbarkeit“).
  - Sicherstellen, dass bestehende Tests unverändert grün bleiben.

## Arbeitsweise von Cursor

- **Konservativ mit Side-Effects**
  - Keine geheimen Änderungen an Config/Security/Secrets.
  - Explizit markieren, wenn Migrations/Schema-Änderungen nötig sind.

- **Deterministische Backtests**
  - Änderungen an Backtest-/Simulation-Code müssen deterministisch bleiben (gleiche Inputs → gleiche Outputs).
  - Zeit-/Randomness-Quellen kapseln (Seed/Clock abstrahieren).

- **File-by-file**
  - Pro Änderung klar dokumentieren, welche Dateien betroffen sind und warum.
  - Keine „Drive-by“-Änderungen in nicht betroffenen Bereichen.

