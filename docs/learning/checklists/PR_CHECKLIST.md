PR Checklist
============

Diese Checkliste gilt für **jeden** PR (auch kleine).

- **Tests laufen**
  - `pytest` (oder gezielte Testauswahl) wurde lokal oder im CI ausgeführt.
  - Keine neuen, ungeklärten Testfehler.

- **QC Gates**
  - Linter/Format (`ruff check`, `ruff format --check`, `mypy` falls aktiv) ausgeführt.
  - Warnungen/Fehler adressiert oder begründet.

- **Repro-Steps dokumentiert**
  - Für Bugs: klare Reproduktionsschritte (inkl. Input/Command/Expected vs. Actual).
  - Für Features: kurze Beschreibung, wie man das Feature „happy path“ validiert.

- **Logging / Telemetry**
  - Neue Fehlerpfade oder kritische Logik haben sinnvolle Logs/Telemetry (falls relevant).
  - Keine übermäßig lauten Logs (kein Log-Spam).

- **Scope im Griff**
  - Keine großen Refactors „nebenbei“ ohne klaren Grund.
  - Große Refactors separat planen und PR-technisch trennen.

- **Bugfix-Regel**
  - Für Bugfixes existiert ein **Incident-Eintrag** (siehe `docs/learning/incidents`).
  - Mindestens **ein neuer oder erweiterter Test** deckt den Fix ab.

