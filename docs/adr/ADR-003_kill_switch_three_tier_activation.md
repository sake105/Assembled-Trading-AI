# ADR-003: Three-Tier Kill Switch Activation

**Status:** accepted
**Date:** 2026-04-11 (retrospective)
**Deciders:** trading-ops, risk

## Context

A binary on/off kill switch has a well-known failure mode: in a
deteriorating but not yet catastrophic situation, the operator faces
a false binary — either do nothing (and watch the position deteriorate
further) or kill everything (and lose optionality). The binary
design also encourages "only use it in an emergency" hesitation,
which means the switch is often activated too late.

We also needed a mechanism that automatic drawdown rules (see
ADR-007) could target without triggering a full halt on the first
soft threshold.

## Decision

`execution/kill_switch.py` uses a three-tier activation model:

- **Tier 1 — soft throttle** (e.g. `throttle_pct=50`): new orders are
  reduced to 50% of target size, existing positions are not touched.
- **Tier 2 — hard throttle** (e.g. `throttle_pct=80`): new orders are
  reduced to 20% of target size, existing positions can still be
  closed.
- **Tier 3 — full kill** (`throttle_pct=100`): no new orders at all,
  existing positions can still be closed by exit logic.

Every activation writes a persistent JSON state file and an JSONL
audit trail. Manual reset requires a human — the safety invariant in
`CLAUDE.md` forbids automatic recovery.

## Consequences

### Positive

- Graduated response matches graduated risk.
- Automatic drawdown rules (ADR-007) can target the right tier
  without triggering a full halt on a soft breach.
- Audit trail is append-only; every state change is traceable.
- The same throttle_pct primitive is reused by circuit breakers
  (flash crash, vol spike) and by manual operator commands.

### Negative

- Three tiers are harder to test than two. The test suite covers
  each tier explicitly and also the transitions between them.
- Documentation must be explicit about what "throttle" means;
  otherwise a reader might assume tier 2 closes positions, which it
  does not.
- Fractional throttle interacts non-trivially with the turnover
  gate; the order of phases in the trading cycle is designed so
  turnover gate runs before risk controls, not the other way.

## Alternatives Considered

- **Binary on/off**: rejected. See context.
- **Continuous scale `[0,1]`**: rejected. Too many edge cases in
  testing and in operator reasoning; discrete tiers are clearer.
- **Per-symbol only**: rejected for the global case, but a per-symbol
  block list is a separate sidecar (see `execution/symbol_kill_switch.py`
  added in C27).
