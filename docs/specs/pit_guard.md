# Spec: `data/pit_guard.py`

## Purpose

Point-in-time safety guard with two modes: feature-level leakage and
universe-level leakage. See ADR-004 for the reasoning behind the
two-mode split.

## Public API

- `PITGuard(policy)` — instantiated with a policy dict.
  - `validate(features_df, as_of) -> pd.DataFrame` — feature mode.
    Rows with disclosure timestamps after `as_of` are dropped or
    flagged depending on policy (`strict` vs `warn`).
  - `validate_universe(universe, as_of) -> list[str]` — universe
    mode. Filters the universe to members whose listing window
    covers `as_of`.
  - `truncate(df, as_of)` — convenience hard-cut.
- `PITViolationError` — raised in strict mode.

## State

Append-only JSONL audit at `output/pit_audit/pit_audit.jsonl`. Every
decision (drop, flag, pass) is recorded with timestamp, row count,
mode, and reason.

## Invariants

- Validation is monotone in `as_of`: more information can only
  become available, never disappear. A row allowed at an earlier
  `as_of` must still be allowed at a later one.
- Strict mode raises before any row is returned. There is no "half
  validated" DataFrame.
- The audit log is written before the caller sees the result. A
  crash mid-write is discoverable from the truncated JSONL.

## Error handling

- Missing disclosure timestamp column → raise in strict mode, warn
  and pass through in warn mode, but always write an audit event.
- Missing universe registry → raise. Universe mode cannot "best
  effort" its way through a missing registry.

## Test strategy

- Known delisting (e.g. LEH Sep 2008) must be absent from universe
  queries after the delist date.
- A feature row dated after `as_of` must be dropped or flagged.
- `validate` twice with the same inputs must be idempotent.
- Property: monotonicity in `as_of`.

## Known limits

- Universe mode depends on a universe registry with `start_date` /
  `end_date`. For watchlist-scale use this is a single CSV; for
  S&P 500 scale it requires a sourced history (see Sprint 1
  `scripts/build_universe_parquet.py`).
