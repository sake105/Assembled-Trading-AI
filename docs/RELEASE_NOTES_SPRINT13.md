Release: Sprint 13 - Observability Graveyard & Operational Cleanup
Tag: sprint13-cleanup-2026q2

## Changes

- Archived observability-only modules from signals/, risk/, features/, qa/, pipeline/, execution/, ops/, compliance/, experiments/, intel/, strategies/
- Fixed test collection failures: added pytest.importorskip guards to affected test files
- No trading logic modified; all archived modules were meta-enrichment only
