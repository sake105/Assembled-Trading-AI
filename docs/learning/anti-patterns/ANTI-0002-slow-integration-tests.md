# ANTI-0002 — Langsame / fragile Integrationstests

## Symptom

- Tests laufen > 60–120 Sekunden pro Suite.
- Häufige Timeouts (`subprocess.TimeoutExpired`, Netzwerk- oder API-Delays).
- Flaky Tests, die gelegentlich wegen externer Abhängigkeiten fehlschlagen.

## Warum problematisch?

- Verlangsamt Feedback-Schleifen (Developer warten auf CI).
- Erhöht Risiko, dass NEWS-Pipeline-Änderungen aus Angst vor langen Runs seltener getestet werden.
- Versteckt Performance-/Robustness-Probleme hinter hohen Timeouts statt sie sichtbar zu machen.

## Besseres Muster

- **Boundaries mocken:**
  - Externe Dienste (HTTP, GDELT, RSS, Filesystem) im Test via Mocks/Fakes ersetzen.
  - Für NEWS v1: `fetch_rss`, `fetch_gdelt`, Files für Baseline/Dedupe werden in `tmp_path` simuliert.

- **Kleine, deterministische Tests:**
  - Fokus auf eine Pipeline-Stufe (z.B. Normalize, Dedupe, Clustering) mit künstlichen Inputs.
  - Klar definierte, deterministische Expectations (keine Zeit/Netzwerk-abhängigen Checks).

- **Timeouts niedrig halten:**
  - Zeitouts in Tests möglichst < 30s, ideal < 5s.
  - Falls nötig: einen einzelnen „Langlauf“-Test kennzeichnen und selten ausführen.

## Action Items

- Bestehende langsame Integrationstests identifizieren (Laufzeiten messen).
- Boundary-Mocks für NEWS v1 konsequent nutzen (siehe `test_news_pipeline_v1.py`).
- Neue Tests bevorzugt als Unit-/kleine Integrations-Tests designen.

