# Spec: `risk/state_machine.py`

## Purpose

Persistent risk-state record that tracks drawdown level, geo-risk
tier, regime label, and the cooldown hours since the last transition.
Phase 2 of the trading cycle loads the state, `compute_next_state`
returns the new state given the current context and policy, and the
new state is saved back.

## Public API

- `RiskStateRecord` — dataclass with `drawdown_level`, `geo_level`,
  `regime`, `updated_at`, and cooldown fields.
- `load_risk_state(path) -> RiskStateRecord` — read JSON, fall back
  to a safe default if missing.
- `save_risk_state(record, path)` — atomic write with retry.
- `compute_next_state(ctx, current, policy) -> RiskStateRecord` —
  pure function from `(ctx, current, policy)` to the new record.
- `compute_drawdown_risk_level(drawdown, policy)` — map drawdown % to
  a discrete level.
- `compute_regime_risk_limits(regime, policy)` — map regime label to
  the applicable exposure limits.

## State

JSON at `output/state/risk_state.json`. Written via
`atomic_write_json_with_retry`: temp file → fsync → atomic rename.
Retries on Windows rename races.

## Invariants

- `compute_next_state` is pure. Same inputs → same output. No file
  I/O, no clock read except for the `now_utc` argument.
- Transitions are cooldown-gated. The state cannot escalate and
  de-escalate in the same cycle.
- Auto-recovery (de-escalation without human input) is forbidden
  above a configured threshold — see the safety invariant in the
  root `CLAUDE.md`.

## Error handling

- Corrupt state file → return default record and log a warning. The
  daily cycle should not fail just because a single state file is
  unreadable; the missing-data path is explicit.
- Save failure → raise. Losing a state transition is worse than
  failing loudly.

## Test strategy

- Unit tests per transition: bull → sideways, sideways → bear, bear
  → crisis, and the reverse where legal.
- Cooldown tests: a rapid sequence of conflicting signals must not
  cause oscillation.
- Property: `compute_next_state` is idempotent when the context is
  unchanged.

## Known limits

- State is global, not per-strategy. The multi-strategy-parallel
  roadmap item needs an addendum (see ADR-005 / ADR-006 discussion).
