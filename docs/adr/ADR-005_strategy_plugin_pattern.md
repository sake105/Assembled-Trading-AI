# ADR-005: Strategy Plugin Pattern

**Status:** accepted
**Date:** 2026-04-11 (retrospective)
**Deciders:** research, architecture

## Context

The system needs to support multiple strategies in parallel. Early
options considered were:

1. Fork the entire pipeline per strategy (rejected in ADR-001).
2. Subclass a base strategy class and pass an instance around.
3. Inject strategy-specific callables into the single orchestrator.

Subclassing sounds clean but creates an implicit contract that is
hard to test in isolation — the base class accumulates hooks, each
subclass overrides a subset, and reviewers have to read multiple
files to understand a single execution. The inheritance chain also
makes it hard to run two strategies side-by-side in a shadow
configuration.

## Decision

A strategy is a triple of pure callables:

- `signal_fn(ctx) -> pd.DataFrame`
- `position_sizing_fn(signals, ctx) -> pd.DataFrame`
- `check_exit_signals(positions, prices, cfg) -> pd.DataFrame`

plus a small dict of configuration read from `configs/app.yaml`
under `paper_runner.strategy`. The runner looks up the triple by
`strategy.name` and passes them to the orchestrator.

`multifactor_v1.py`, `ema_trend_v0.py`, and the planned
`multifactor_v2.py` each implement this triple. Nothing else about
them is shared by inheritance.

## Consequences

### Positive

- A strategy is three functions + a config block, not a class
  hierarchy.
- Two strategies can run in parallel by dispatching two triples;
  nothing in the runner or orchestrator has to be aware of that.
- Tests can call any of the three functions in isolation without
  constructing a strategy instance.
- A new strategy can be added without touching existing strategy
  files.

### Negative

- There is no compiler-enforced interface. A typo in a function
  signature will only fail at call time. This is partially
  mitigated by tests that import every registered strategy.
- Sharing code between strategies has to go through a helper
  module, not a base class. This is by design but requires
  discipline.
- The config loader and the dispatcher are the two places where
  a new strategy must be registered — both are easy to forget.

## Alternatives Considered

- **ABC + subclass pattern**: rejected. See context.
- **Entry-point / plugin registry (setuptools)**: rejected as
  overengineering for the current size. Can be revisited if the
  strategy count grows into the dozens.
- **One-file-per-strategy with a decorator registry**: interesting,
  deferred until there is a clear second use case.
