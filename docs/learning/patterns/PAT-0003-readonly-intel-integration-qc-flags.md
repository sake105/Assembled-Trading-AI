# PAT-0003 — Read-only Intel Integration + QC-Flags

## Kontext

Neue Intel-Quellen (z.B. NEWS-Trigger) sollen in den Trading-Kontext integriert werden, ohne das Trading-Verhalten sofort zu verändern. Fehlende oder fehlerhafte Intel-Artefakte dürfen **nicht** den Trading-Cycle crashen.

## Problem

- Direkte, „harte“ Integration (z.B. `open(...)` ohne Fehlerbehandlung) führt zu:
  - Crashes, wenn Dateien fehlen/kaputt sind.
  - Schwer interpretierbaren Fehlermeldungen für Operatoren.
  - Geringer Bereitschaft, neue Intel-Signale früh zu integrieren.

## Lösung (Pattern)

**Read-only Intel Integration** mit klaren Quality-Flags:

- **Toleranter Loader:**
  - Prüft `schema_version`, Struktur (z.B. `items` ist Liste) und JSON-Parsebarkeit.
  - Bei Fehlern: gibt einen **leeren Snapshot** zurück (statt Exception).

- **QC-Flags im Context:**
  - Statt Crash werden Flags gesetzt, z.B. `intel_health_flags["intel_news_triggers"] = "DEGRADED"`.
  - Der TradingContext bleibt lauffähig; Strategien können Flags auswerten, müssen aber nicht.

- **Keine Trading-Side-Effects in v1:**
  - Intel-Felder (`news_triggers`) sind read-only.
  - Keine direkten Änderungen an Orders/Positions in der ersten Integrationsphase.

## Checklist

- [ ] Loader fungiert als Boundary (tolerant, schema_version-Check, keine harten Exceptions).
- [ ] Intel-Daten werden im Context separat gehalten (`news_triggers`, `summary`, `intel_health_flags`).
- [ ] Fehlende/kaputte Dateien setzen QC-Flag statt Crash.
- [ ] Trading-/Risk-Logik liest Intel-Daten nur opt-in (spätere Sprints).

## Beispiel (NEWS-Trigger)

- `output/intel/news/triggers_latest.json` wird nur gelesen:
  - Snapshot: `NewsTriggerSnapshot` mit `generated_utc`, `triggers`, `summary`.
  - QC: fehlende/invalid Datei → `intel_news_triggers="DEGRADED"`.
- Weitere Details:
  - `docs/integrations/NEWS_TRIGGERS_TRADINGCONTEXT.md`
  - `docs/news/ARTIFACTS.md`

Dieses Pattern erlaubt es, Intel-Signale früh und sicher in den Context zu bringen und später kontrolliert in Trading-Regeln zu überführen.

