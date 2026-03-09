# Disclosures v1 Artefakte

Übersicht über alle Disclosures v1 Outputs (Artefakte), deren Pfade, Schema-Versionen und Zwecke.

## Kern-Artefakte (`output/intel/disclosures/`)

- **Events**
  - **Pfad**: `output/intel/disclosures/events_latest.json`
  - **Schema**: `schema_version = "disclosures.v1"`
  - **Purpose**: Normalisierte Disclosure-Events (`DisclosureEvent`), z.B. Form-4-Filings; Input für spätere Trigger/Integration.
  - **Wann**: Jede Pipeline-Ausführung.
  - **Key-Felder**: `event_id`, `source_id`, `action_type`, `person_or_entity`, `fingerprint`, `accession_or_link`, `published_utc`.

- **Health**
  - **Pfad**: `output/intel/disclosures/health_latest.json`
  - **Schema**: `schema_version = "disclosures.health.v1"`
  - **Purpose**: Pipeline-Gesundheitszustand (OK/DEGRADED/ERROR), Quellen-Status, Item-Zahlen.
  - **Wann**: Jede Pipeline-Ausführung.
  - **Key-Felder**: `health.status`, `sources_total/ok/failed`, `items_raw/after_dedupe`, `notes`.

- **Fetch-Report**
  - **Pfad**: `output/intel/disclosures/fetch_report_latest.json`
  - **Schema**: `schema_version = "disclosures.fetch_report.v1"`
  - **Purpose**: Per-Source-Statistiken für Operatoren (HTTP-Status, Dauer, Items, **cached**-Flag).
  - **Wann**: Jede Pipeline-Ausführung.
  - **Key-Felder**: `totals`, `per_source[]` (z.B. `source_id`, `ok`, `http_status`, `duration_ms`, `items`, `cached`).

- **Triggers (Stub)**
  - **Pfad**: `output/intel/disclosures/triggers_latest.json`
  - **Schema**: `schema_version = "disclosures.triggers.v1"`
  - **Purpose**: Platzhalter für spätere Trigger-Logik; aktuell leeres Array `[]`.
  - **Wann**: Jede Pipeline-Ausführung.
  - **Key-Felder**: (stub)
  - **Konsument**: Im Paper Runner wird `triggers_latest.json` in **real Intel mode** (`paper_runner.intel.mode=real`) vor dem Trading Cycle geladen und für Confirm-Gate/State-Machine verwendet.

## Cache-/State-Artefakte (`output/intel/disclosures/cache/`)

- **Fetch-State**
  - **Pfad**: `output/intel/disclosures/cache/fetch_state.json`
  - **Schema/Shape**: Kein explizites `schema_version`; Struktur ist pro Quelle (keyed by `source_id`). Enthält z.B. `cached_entries`, `cached_utc` je Quelle für Cache- und stale-on-error-Logik.
  - **Purpose**: Persistenter Fetch-Cache-State (cache_minutes, stale_on_error_minutes); siehe `docs/learning/patterns/PAT-0004-fetch-cache-stale-on-error.md`.
  - **Wann**: Alle Runs, in denen EDGAR-Form-4-Fetch mit Cache verwendet wird.
  - **Key-Struktur**: Objekt mit Keys = `source_id`; je Eintrag u.a. `cached_entries`, `cached_utc` (und ggf. weitere Felder für ETag/Last-Modified, falls später ergänzt).
