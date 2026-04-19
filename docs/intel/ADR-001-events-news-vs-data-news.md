# ADR-001: events/news/ as Single Source of Truth for News Data

**Date:** 2026-04-19  
**Status:** Accepted  
**Context:** T3.7 from goofy-questing-crystal remediation plan

## Context

Two parallel directories handle news data:

- `src/assembled_core/events/news/` — active pipeline (fetch, normalize, dedupe, cluster, pipeline)
- `src/assembled_core/data/news/` — older news-related data utilities

The audit confirmed that `events/news/` is the active production path while `data/news/` is not wired into any active pipeline.

## Decision

**`src/assembled_core/events/news/` is the single source of truth for all news ingestion, normalization, deduplication, and clustering.**

`data/news/` should be audited in a follow-up task and either:
- Migrated into `events/news/` if useful utilities exist
- Archived alongside `archive/intel_research_2026q2/` if pure dead code

## Rationale

- `events/news/` has active tests, an active worker (`run_news_worker.py`), and active pipeline consumers
- Adding a second path creates confusion about which normalizer, schema, or dedupe index to use
- Layer violations risk: `data/` importing from `events/` or vice-versa creates upward coupling

## Consequences

- All new news-related code goes into `events/news/` (or `events/disclosures/` for disclosure data)
- Any `data/news/` utilities that have no equivalent in `events/news/` should be migrated before archiving
- The follow-up audit of `data/news/` is tracked as T3.7 follow-up work
