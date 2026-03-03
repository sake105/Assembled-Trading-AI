INC-0001 — Incident Template
============================

> Kopiere diese Datei und passe `INC-0001` + Titel an (z.B. `INC-0002-cash-gate-mtm-equity.md`).

## Kontext

- Kurze Beschreibung des Systems/Features (welcher Teil der Pipeline? welche Strategie?).
- Relevante Versionen / Branch / Umgebung (dev, staging, prod/paper).

## Symptom

- Was wurde beobachtet?
- Konkrete Metriken/Artefakte (z.B. „equity_curve flach“, „cash wurde negativ“).

## Impact

- Welche Auswirkungen hatte der Incident?
- Betroffene Nutzer / Runs / Strategien / Metriken.

## Detection / Signal

- Wie wurde das Problem bemerkt?
- Alerts, Dashboards, QC-Gates, manuelle Reviews, Tests, Logs.

## Root Cause

- **Technische** Ursache (Code, Daten, Konfiguration).
- **Prozessuale** Ursache (fehlende Checks, unklare Verantwortung, etc.).

## Fix

- Kurze Beschreibung der Lösung.
- **Code-Pointer (Platzhalter):**
  - Datei/Funktion: `<FILE:LINE>` (z.B. `src/assembled_core/execution/fill_model.py`)
  - Wichtige Änderungen in 1–3 Stichpunkten.

## Tests

- Neue Tests:
  - `<TEST_NAME>` (z.B. `tests/test_cash_gate_prevents_negative_cash.py::test_cash_curve_min_non_negative_after_backtest`)
- Angepasste Tests:
  - `<TEST_NAME>` (falls vorhanden)

## Prevention (Guardrails / Checks)

- Welche zusätzlichen Checks/Guards verhindern Wiederholung?
  - z.B. QC-Gate, Lint-Regel, zusätzliche Assertion, Monitoring, Dashboard.

## Follow-ups (Backlog Items)

- Konkrete Tickets/Tasks (kurz, umsetzbar):
  - `[ ] <Follow-up 1>`
  - `[ ] <Follow-up 2>`

