# PAT-0002 — NEWS-Artefakte als Contracts

## Kontext

Die NEWS v1 Pipeline produziert mehrere JSON-Artefakte (`events_latest.json`, `clusters_latest.json`, `triggers_latest.json`, ...). Diese Artefakte dienen nicht nur als Logs, sondern als **stabile Verträge** zwischen Komponenten (NEWS ↔ Backtest/Monitoring/Research).

## Problem

Ohne klar definierte Schemas, `schema_version` und atomare Writes kann es zu:
- stillen Schema-Brüchen
- inkonsistenten Reads (halb geschriebene Dateien)
- schwer reproduzierbaren Bugs kommen.

## Lösung (Pattern)

- **Schema-Versionierung:** Jedes Artefakt trägt eine `schema_version` (`news.v1`, `news.health.v1`, ...).
- **Deterministischer Output:** Felder/Sortierung sind stabil; Tests prüfen Struktur und semantische Invarianten.
- **Atomare Writes:** Artefakte werden in eine temporäre Datei geschrieben und via `os.replace` atomar ersetzt.
- **Artifact-Katalog:** `docs/news/ARTIFACTS.md` dokumentiert Pfade, Zwecke und Schlüssel-Felder.

## Beispiele

- `events_latest.json` enthält eine flache Liste normalisierter `NewsEvent`s mit Fingerprints/Entities/Ländern.
- `clusters_latest.json` kapselt Cluster inkl. `topics`, `evidence`, `top_entities`, `top_phrases`.
- `bursts_latest.json` bietet sowohl eine Primary-View (`items`) als auch Multi-Window-View (`windows`).

## Enforcement

- Unit-/Integrationstests prüfen:
  - `schema_version`-Felder und erwartete Keys.
  - Atomare Writes (kein Partial-Write bei Crash).
  - Backward-Kompatibilität bei Wrappern (z.B. Bursts-Wrapper).

Dieses Pattern stellt sicher, dass NEWS v1 auch bei späteren Erweiterungen ein stabiler Contract bleibt.

