# NEWS v1 Artefakte

Übersicht über alle NEWS v1 Outputs (Artefakte), deren Pfade, Schema-Versionen und Zwecke.

## Kern-Artefakte (`output/intel/news/`)

- **Events**
  - **Pfad**: `output/intel/news/events_latest.json`
  - **Schema**: `schema_version = "news.v1"`
  - **Purpose**: Normalisierte News-Events (`NewsEvent`), Input für Clustering/Bursts/Backtests.
  - **Wann**: Jede Pipeline-Ausführung (hourly/daily).
  - **Key-Felder**: `event_id`, `source_id`, `title`, `canonical_url`, `countries`, `entities`, `fingerprint64`.

- **Health**
  - **Pfad**: `output/intel/news/health_latest.json`
  - **Schema**: `schema_version = "news.health.v1"`
  - **Purpose**: Zusammenfassung des Pipeline-Gesundheitszustands (OK/DEGRADED/ERROR + Metriken).
  - **Wann**: Jede Pipeline-Ausführung.
  - **Key-Felder**: `health.status`, `sources_total/ok/failed`, `items_raw/after_dedupe`, `notes`, `metrics.cluster_quality`, `metrics.baseline`, `metrics.bursts`, `metrics.triggers`.

- **Fetch-Report**
  - **Pfad**: `output/intel/news/fetch_report_latest.json`
  - **Schema**: `schema_version = "news.fetch_report.v1"`
  - **Purpose**: Per-Source-Statistiken (HTTP-Status, Dauer, Items, Cache-Hits) für Operatoren.
  - **Wann**: Jede Pipeline-Ausführung.
  - **Key-Felder**: `totals`, `per_source[]` (Quelle, Typ, `ok`, `http_status`, `items`).

- **Clusters**
  - **Pfad**: `output/intel/news/clusters_latest.json`
  - **Schema**: `schema_version = "news.clusters.v1"`
  - **Purpose**: Gruppierung verwandter News-Events.
  - **Wann**: Jede Pipeline-Ausführung (wenn `clustering.enabled`).
  - **Key-Felder**: `cluster_id`, `event_ids`, `topics`, `candidate_triggers`, `evidence`, `top_entities`, `top_phrases`.

- **Triggers**
  - **Pfad**: `output/intel/news/triggers_latest.json`
  - **Schema**: `schema_version = "news.triggers.v1"`
  - **Purpose**: Abgeleitete Trigger (Severity/Confidence) für spätere Strategy-Integration.
  - **Wann**: Jede Pipeline-Ausführung (wenn `trigger_scoring.enabled`).
  - **Key-Felder**: `trigger_id`, `cluster_id`, `trigger_type`, `topic_id`, `severity`, `confidence`, `ttl_hours`, `decay`, `evidence_ok`.
  - **Konsument**: Wird read-only vom `TradingContext` geladen (siehe `docs/integrations/NEWS_TRIGGERS_TRADINGCONTEXT.md`).

- **Bursts**
  - **Pfad**: `output/intel/news/bursts_latest.json`
  - **Schema**: `schema_version = "news.bursts.v1"`
  - **Purpose**: Burst-Erkennung (Entities/Phrasen) relativ zur 30d-Baseline.
  - **Wann**: Jede Pipeline-Ausführung (wenn `burst.enabled`).
  - **Key-Felder**:
    - Legacy: `window_hours`, `count`, `items` (Primary Window).
    - Neu: `windows[]` mit `window_hours`, `top_entities_burst`, `top_phrases_burst`, `top_clusters_burst`.

- **Housekeeping**
  - **Pfad**: `output/intel/news/daily_housekeeping_latest.json`
  - **Schema**: `schema_version = "news.housekeeping.v1"`
  - **Purpose**: Protokoll der täglichen GDELT-Cache-Bereinigung.
  - **Wann**: Nur bei `cadence == "daily"`.
  - **Key-Felder**: `pruned_gdelt_cache_entries`, `notes`.

## Baseline-Artefakte (`output/intel/news/baseline/`)

- **Baseline (aggregiert)**
  - **Pfad**: `output/intel/news/baseline/baseline_latest.json`
  - **Schema**: `schema_version = "news.baseline.v1"`
  - **Purpose**: Aggregierte 30d-Baseline für Entities, Phrasen, Topics.
  - **Wann**: Nur bei `cadence == "daily"` und `burst.enabled`.
  - **Key-Felder**: `baseline_days`, `version_hash`, `entity_counts`, `phrase_counts`, `topic_counts`, `window.start_utc/end_utc`.

- **Baseline-State (per Day)**
  - **Pfad**: `output/intel/news/baseline/baseline_state.json`
  - **Schema**: `schema_version = "news.baseline_state.v1"`
  - **Purpose**: Interner State mit Tages-Buckets (für Rebuild von `baseline_latest`).
  - **Wann**: Nur bei Daily-Runs.
  - **Key-Felder**: `days[YYYY-MM-DD].entity_counts/phrase_counts/topic_counts`.

## Cache-/State-Artefakte (`output/intel/news/cache/`)

- **Fetch-State**
  - **Pfad**: `output/intel/news/cache/fetch_state.json`
  - **Schema**: `schema_version = "news.fetch_state.v1"` (implizit, Struktur ist stabil dokumentiert).
  - **Purpose**: ETag/Last-Modified für RSS + GDELT-Cache-Metadaten.
  - **Wann**: Alle Runs, wenn Fetch/Cache verwendet wird.
  - **Key-Felder**: `rss`, `gdelt`-Subtrees mit `etag`, `last_modified`, `cached_utc`.

- **Dedupe-Store (SQLite)**
  - **Pfad**: `output/intel/news/cache/dedupe_store.sqlite`
  - **Schema/Meta**: `meta.schema_version = "news.dedupe_store.v1"` in SQLite-Metatabelle.
  - **Purpose**: Persistenter Dedupe-Store für `canonical_url` + 64-bit-Fingerprint-Buckets.
  - **Wann**: Alle Runs, in denen Dedupe-Store aktiviert ist.
  - **Key-Struktur**: Tabelle `seen_events(canonical_url, fp64, fp_bucket, event_id, source_id, published_utc, ingested_utc)` + Index auf `fp_bucket`.

