# ADR-001: Unified Trading Cycle Orchestrator

**Status:** accepted
**Date:** 2026-04-11 (retrospective)
**Deciders:** trading-ops, architecture

## Context

Early in the project the paper-trading path and the backtest path had
drifted into two separate pipelines. Signal logic, sizing logic, risk
overlays, and order generation had started to diverge — the paper path
had overlays the backtest path did not run, and the backtest path had
some realism checks the paper path was missing.

This is the classic quant "two systems, two truths" failure mode: any
strategy number you produce in a backtest is then not directly
comparable to what the paper runner actually does, and any bug caught
in paper cannot be reproduced in the backtest because the code paths
are different.

## Decision

All decision-making runs through a single orchestrator:
`src/assembled_core/pipeline/trading_cycle.py:run_trading_cycle`. The
paper runner, the backtest replay, and any future live runner all call
the same function with the same phase ordering. Strategy-specific
behaviour is injected through callables (`signal_fn`,
`position_sizing_fn`, `check_exit_signals`) — not through parallel
pipelines.

The phases are numbered and fixed:

1. Validate prices
2. Risk state machine
3. Intel triggers
4. GeoRisk intel
5. Market stress
6. Price filter (as_of)
7. Feature building
8. Signal generation
9. Position sizing
10. QA block gate
11. Profit lock overlay
12. Correlation guard
13. GeoRisk overlay
14. Vol targeting
15. Turnover gate
16. Zombie killer
17. Order generation
18. Risk controls

New overlays are added as `N.x` sub-phases (e.g. `5.5` for circuit
breaker) rather than by splitting the pipeline.

## Consequences

### Positive

- Backtest numbers are directly comparable to paper numbers because
  they come from the same code path.
- A bug found in paper is reproducible in the backtest with the same
  inputs.
- New overlays only need to be wired once.
- Risk invariants are enforced in a fixed, auditable order.

### Negative

- The orchestrator is now a large file with many phases and is harder
  to refactor casually. This is intentional — it is a load-bearing
  contract, not a utility file.
- Callers cannot "skip phase 15" for a one-off experiment without
  adding a config flag.
- Feature-flag proliferation is a real risk; see ADR-007 on how
  flags are structured.

## Alternatives Considered

- **Keep two pipelines**: rejected. This was the starting point and
  was the cause of the problem we are now fixing.
- **One pipeline per strategy**: rejected. Strategies should be
  callables that the pipeline invokes, not separate pipelines.
- **Event-driven orchestration**: rejected for now. The daily cadence
  and the pipeline's sequential safety invariants make a numbered
  phase list clearer than an event graph.
