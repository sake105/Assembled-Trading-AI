# ADR-004: PIT Guard with Feature-Mode and Universe-Mode

**Status:** accepted
**Date:** 2026-04-11
**Deciders:** data, research

## Context

Point-in-time safety is the single biggest correctness trap in
quant backtesting. There are two different classes of PIT violation:

1. **Feature-level leakage**: a feature value at time `t` was
   computed using data that only became available after `t`. Example:
   using a revised fundamental number instead of the one originally
   published.
2. **Universe-level leakage**: the universe at time `t` includes
   symbols that were not in the index at `t`. This is the classic
   survivorship bias: a 2010 backtest that only considers today's
   S&P 500 is quietly assuming every firm that exists today also
   existed then.

These two leaks happen in different places in the code and need
different checks. Conflating them into one "PIT guard" gives you a
check that can only catch one of them.

## Decision

`data/pit_guard.py` runs in two distinct modes:

- **Feature mode** validates a feature DataFrame against a
  disclosure timestamp column and an `as_of` cutoff. Any row whose
  disclosure time is after `as_of` is either dropped or flagged in
  the audit log. This mode is used inside feature building.
- **Universe mode** validates a universe snapshot against a
  per-symbol `start_date` / `end_date` registry. Symbols whose
  listing window does not cover the requested date are rejected.
  This mode is used by `data/universe.py:get_universe_members_pit`.

Every PIT decision is appended to `output/pit_audit/pit_audit.jsonl`
in a structured form so a later reviewer can trace why a row or
symbol was included or dropped. The audit file is append-only.

## Consequences

### Positive

- Feature leakage and universe leakage are caught by different,
  explicit checks. Neither can hide behind the other.
- The audit log lets a reviewer answer "why was this symbol in the
  universe on this date?" without re-running the pipeline.
- The universe mode is what makes delisted symbols (Lehman 2008,
  Enron 2001) correctly absent from pre-delisting-date backtests.

### Negative

- Requires a universe-history parquet with `start_date` / `end_date`
  columns. Initial build is a one-off data task.
- The feature mode requires every disclosure source to carry a
  timestamp. Sources that only emit latest values must be wrapped
  with an explicit "published-at" timestamp before they can pass
  the guard.
- The audit file grows unbounded; retention / rotation is a
  separate concern.

## Alternatives Considered

- **Single mode guarding both levels**: rejected. The two levels
  live at different layers of the pipeline and have different
  natural keys.
- **No universe mode, rely on careful symbol curation**: rejected.
  "Be careful" is not a control.
- **Audit as a database**: deferred. A JSONL file is sufficient
  for current volume and is trivially inspectable.
