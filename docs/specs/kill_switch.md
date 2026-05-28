# Spec: `execution/kill_switch.py`

## Purpose

Global three-tier kill switch that can throttle or halt order flow.
Single authority for "should an order be placed at all right now".

## Public API

- `activate_kill_switch(throttle_pct, reason, actor)` — engage at a
  specific throttle percentage (0-100). Tier 1 soft = 50, tier 2
  hard = 80, tier 3 full = 100.
- `deactivate_kill_switch(reason, actor, operator_token)` — clear state.
  Requires OPERATOR_KILL_TOKEN env var and a matching `operator_token` argument.
  Raises `PermissionError` if env var is absent or token mismatches.
  Both rejection types are written to the audit log as `REJECT_DEACTIVATE`
  before the exception is raised (non-repudiation preserved).
  `activate_kill_switch()` is intentionally not gated — emergency stop must
  always work with no barrier.
- `get_kill_switch_state() -> dict` — current tier, reason, timestamps.
- `is_kill_switch_engaged() -> bool` — convenience for callers.
- `get_throttle_pct() -> float` — current throttle percentage.
- `check_drawdown_kill_switch(...)` — evaluate drawdown against
  configured thresholds and auto-escalate.
- `guard_orders_with_kill_switch(orders) -> DataFrame` — filter an
  order DataFrame by the current throttle.

## State

Persistent JSON at `output/state/kill_switch_state.json`. Append-only
audit at `output/audit/kill_switch_audit.jsonl`. Every transition is
recorded with timestamp, old tier, new tier, reason, actor.

## Invariants

- Activation is one-way within a session. Deactivation requires an
  explicit call.
- Every state change is audited before the in-memory state flips.
- Throttle percentage is monotone in tier (higher tier → ≥ throttle).
- A read of the state file that fails to parse must not silently
  return "not engaged". It must surface an error to the caller.

## Error handling

- State file corruption → `RuntimeError` with path and parse error.
- Audit write failure → log + re-raise. Do not silently continue with
  an un-audited state change.

## Test strategy

- `tests/test_kill_switch_persistent_state.py` — round-trip state
  across activate → deactivate.
- `tests/test_chaos_kill_switch_race.py` — concurrent activation /
  deactivation sanity (see regression test map C21-1 / C21-2).
- Property: `throttle_pct` is always in [0, 100].

## Known limits

- Global only; per-symbol blocking lives in `symbol_kill_switch.py`.
- The state file write is not atomic under concurrent writers. Fine
  for single-process use; multi-process callers need file locking.
