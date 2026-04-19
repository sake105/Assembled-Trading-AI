# ADR-002: Evidence Engine Status Resolution

**Date:** 2026-04-19  
**Status:** Accepted  
**Context:** T7.3 — resolves conflict between ROADMAP_STATE (says COMPLETE) and module_activation_plan (says Deprecation)

## Problem

Two documents contradicted each other:

- `ROADMAP_STATE.md`: marks Evidence Engine as COMPLETE
- `module_activation_plan.md` (or similar): suggests Deprecation

## Resolution

**Evidence Engine status: ACTIVE and WIRED — not deprecated.**

The confusion arose because:
1. The early activation plan suggested deprecating the standalone evidence engine in favor of inline crisis_alpha logic
2. However, the engine is actively used in `events/crisis_alpha/gates.py:126` and drives the grade-gate logic

**Correct state:**
- `events/evidence_engine.py`: Active — provides `EvidenceGrade`, `check_evidence_grade_gate()`
- `events/crisis_alpha/gates.py`: Active consumer — `check_evidence_grade_gate_from_ctx()`
- The engine is NOT deprecated; it is the correct abstraction

## Decision

Retain `events/evidence_engine.py` as-is. Mark any "deprecation" references in old docs as superseded.

The only open evidence-engine work is:
- T7.9: Write A/B/C/D grades into evidence manifest per run
- T6.5: Write unit tests for `qa/event_study.py` and wire into daily QA report
