# ADR-002: Factor Store with Content-Hash Caching

**Status:** accepted
**Date:** 2026-04-11 (retrospective)
**Deciders:** research, trading-ops

## Context

Computing features from raw prices is expensive. A typical cycle
evaluates a 15+ factor stack over a ~60-day rolling window, and a
multi-year backtest recomputes the same factors every time the same
inputs appear. Early versions of the project recomputed every factor
every run, which made iterating on strategy logic painful: each
small code change that didn't touch features still paid the full
compute cost.

A naive "just cache by date" solved part of the problem but introduced
a different failure: if the feature config changes but the cache key
does not, the cycle silently reads stale features from a previous
config. That is a correctness bug, not a performance bug.

## Decision

The factor store keys its cache on a content hash of the **union** of:

- the universe key (symbol set)
- the feature config
- the upstream price data signature
- the feature module version

Functions: `features/factor_store_integration.py:build_or_load_factors`
and `data/factor_store.py:compute_universe_key`. A cache hit is only
accepted when the hash matches exactly; any mismatch forces a
recompute.

The cache lives under `data/factor_store/` and is written atomically.

## Consequences

### Positive

- Changing feature config invalidates the cache automatically.
- Changing the universe invalidates the cache automatically.
- Iteration on strategy logic is fast when only downstream code
  changes.
- The cache is a replay log that can be inspected for reproducibility.

### Negative

- The hash must cover every input that can affect the feature output,
  or stale reads will slip through. New inputs need conscious addition
  to the hash.
- Cache directory grows over time; a pruning policy is still a follow-up.
- Atomic write on Windows needs explicit handling; the code uses
  write-to-temp + replace.

## Alternatives Considered

- **No cache**: rejected. Too slow for daily iteration.
- **Mtime-based cache**: rejected. File mtime is not a correctness
  signal when config changes shift compute results.
- **Manual cache key**: rejected. Humans forget to update keys when
  they add new inputs; a content hash does not.
