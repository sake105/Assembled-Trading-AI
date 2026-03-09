# Disclosures Pipeline — Spec v1 (Contract & Skeleton)

## Goals

- Ingest disclosure events from **House PTR** (PDF) and **SEC EDGAR** (Form 4 / 13D / 13G).
- Normalize into a single **DisclosureEvent** schema; dedupe by fingerprint.
- Emit artifacts under `output/intel/disclosures/` with versioned schemas.
- **Health gates**: OK / DEGRADED / ERROR; operator behavior (alert, block, or continue) is config-driven.

## Non-goals (v1)

- No heavy parsing of PDF/HTML body content; minimal fields only.
- No real network calls in contract phase; stubs only.
- No downstream trading signals; outputs are for ops/audit and future integration.

## Schema versioning

- All emitted JSON has a top-level `schema_version` (e.g. `disclosures.v1`, `disclosures.health.v1`).
- Breaking changes increment the segment (e.g. `disclosures.v2`); additive changes can stay on v1.

## Phases (v0 .. vX)

- **v0**: Contract + skeleton (this phase). Stubs for fetch; real normalize/dedupe/health/emit structure.
- **v1**: House PTR fetch (mock or real PDF list); EDGAR Form 4/13D/13G stub or real.
- **vX**: Full parsing, entity resolution, notional extraction.

## Sources

| Source        | Type    | Description                    |
|---------------|---------|--------------------------------|
| House PTR     | PDF     | Congress periodic transaction reports |
| SEC EDGAR     | Form 4  | Insider transactions          |
| SEC EDGAR     | 13D/13G | Beneficial ownership           |

Config: `configs/disclosures/sources.yaml` (registry) and `configs/disclosures/disclosures.yaml` (params).

## Health gates

- **OK**: At least one source succeeded; items may be zero.
- **DEGRADED**: Some sources failed or below `min_sources_ok`.
- **ERROR**: No sources succeeded.

Operator behavior (alert / block / continue) is defined in config; pipeline returns exit code 1 on ERROR.

## Output artifacts

- `events_latest.json` — schema `disclosures.v1`; list of DisclosureEvent.
- `health_latest.json` — schema `disclosures.health.v1`; DisclosuresHealth.
- `triggers_latest.json` — schema `disclosures.triggers.v1`; stub `[]`.
- `fetch_report_latest.json` — schema `disclosures.fetch_report.v1`; stub per-source stats.

All writes are atomic (tmp + rename).

## Implemented (v1 current)

- **EDGAR Form 4 Atom feed fetch**  
  Source type: `edgar_form4`. Fetches SEC EDGAR Form 4 filings via Atom feed; config in `configs/disclosures/disclosures.yaml` (e.g. `edgar.form4.feed_url`, `user_agent`).

- **Caching**  
  Via `fetch_state.json` under `output/intel/disclosures/cache/`, keyed by `source_id`. Parameters:
  - `cache_minutes`: serve cached entries until this many minutes have passed.
  - `stale_on_error_minutes`: on fetch error (e.g. network/SEC outage), allow serving stale cache for up to this duration so the pipeline can stay DEGRADED instead of ERROR.

- **Normalization**
  - `action_type = "FORM4_FILED"` for Form 4 raw items.
  - `person_or_entity` taken from company (from feed).
  - `fingerprint = sha256("edgar_form4|" + accession_or_link)` for deduplication; `accession_or_link` is the accession number or feed link when accession is missing.

- **Fetch report**  
  Schema `disclosures.fetch_report.v1`: per-source stats (e.g. `source_id`, `ok`, `http_status`, `duration_ms`, `items`, `cached`). Totals and per_source array for operator inspection.
