Release Checklist
=================

Diese Checkliste gilt für Releases (Tag/Version) und größere Deployments.

- **Version / Tag**
  - Version/Tag ist definiert (z.B. `v1.2.3`).
  - Changelog/Release Notes aktualisiert.

- **Tests**
  - Smoke-Tests (schnelle End-to-End Pfade) erfolgreich.
  - Relevante Phase-/Regression-Tests ausgeführt (mindestens einmal pro Release).

- **Artifacts / Outputs**
  - Wichtige Artefakte geprüft (z.B. Backtest-Outputs, Reports, Parquet/CSV-Schemas).
  - Pfade/Dateinamen konsistent (kein versehentliches Überschreiben älterer Läufe).

- **Rollback-Plan**
  - Klare Anweisung, wie auf die vorherige Version zurückgerollt wird.
  - Vorherige Artefakte sind verfügbar (nicht gelöscht/überschrieben).

- **Known Issues**
  - Bekannte Einschränkungen/Issues sind im Release dokumentiert.
  - Falls kritische Risiken existieren: deutlich markiert (Workarounds beschrieben).

