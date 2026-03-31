---
id: PAT-0005
title: Wire before declaring done — stub pipelines hide integration gaps
type: pattern
severity: medium
discovered: 2026-03-29
milestone: M1
---

## Problem

A module (`trigger_scoring.py`) was fully implemented, tested in isolation, and documented — but never called from the main pipeline (`pipeline.py`). The output artifact (`triggers_latest.json`) was emitted with `count: 0, items: []` as a permanent empty scaffold. Tests for the module passed, but the integration test (`test_pipeline_produces_triggers_with_mock`) failed because the pipeline never invoked the scoring logic.

Similarly, `entity_linking.py` had a stub signature `(news, symbols=None)` but tests were written against a richer interface `(news, mapping_df, security_master_df, missing)`. The stub accepted calls but silently did nothing.

## Why it happened

- Skeleton-first development: modules were created with correct internal logic but never wired into the orchestrator.
- Tests for the module passed, masking the integration gap.
- The output artifact file existed with the right schema version, making it appear as if the system was producing triggers.

## Pattern: verify integration, not just isolation

When a new module is added to a pipeline:
1. Confirm the module is actually **called** from the orchestrator (grep for the function name in `pipeline.py`).
2. Confirm the output artifact is **populated** (count > 0 on known inputs), not just structurally valid.
3. Run the integration test that asserts end-to-end behavior, not just the unit test for the module.

## Applied fix

- Added `from .trigger_scoring import score_triggers` import to `pipeline.py`.
- Inserted `score_triggers()` call after health computation, before artifact emission.
- Added `health.metrics["triggers"]` with `trigger_count` and `max_severity`.
- Replaced `entity_linking.py` stub with full implementation matching the test contract.

## How to apply

Before marking a pipeline module as "done":
- [ ] Is the function imported in the orchestrator?
- [ ] Is the function called in the orchestrator at the right point?
- [ ] Does the output artifact have non-zero items when given relevant input?
- [ ] Does the integration test pass (not just the unit test)?
