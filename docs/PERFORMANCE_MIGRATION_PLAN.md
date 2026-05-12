# Performance Migration Plan

> Audit B-001 (Polars), B-002 (Numba), B-003 (Rust/PyO3) — concrete
> migration approach when the underlying dependencies are installed.
>
> **Status today:** none of `polars`, `numba`, or a Rust toolchain are
> in the project venv. Doing the migration autonomously would (a) need
> a `pip install` the audit-sweep explicitly avoided to keep CI surface
> stable, and (b) risk breaking the existing pandas-based pipelines.
> This document is the **plan-of-record** so the work can happen as a
> deliberate sprint instead of an ad-hoc experiment.

## 1. Polars (B-001) — Highest ROI

### Why
- Independent benchmarks: 5-15× faster than Pandas on typical group-by
  + join + scan workloads.
- 4-8× lower peak RAM.
- Lazy-engine + query optimiser handles much of what Pandas requires
  manual indexing for.

### Scope (1-day sprint estimate)
- One module, one PR: `src/assembled_core/features/ta_features.py`.
- This module is the single biggest CPU sink in the EOD pipeline
  (computed across 500+ symbols × 12+ features per run).

### Migration steps
1. `pip install polars==1.x` and pin in `requirements.lock`.
2. Add adapter layer: `_pd_to_pl(df)` + `_pl_to_pd(df)` at module
   boundary so callers stay pandas-typed.
3. Translate each feature from Pandas `groupby + rolling` to Polars
   `.over("symbol")` + `rolling_*` expressions. The Polars semantics
   are closer to SQL — group is a partition, not a separate DataFrame.
4. **Equivalence test**: run both side-by-side on a 100-symbol fixture,
   assert `pl.testing.assert_frame_equal` with tolerance 1e-9. Sit on
   that test for two weeks of paper-track output before retiring
   the pandas path.
5. Snapshot the Polars output via syrupy (W8 pattern) so future
   numeric drift is caught.

### Acceptance criteria
- 5y × 500 symbols feature computation: <10 s (pandas baseline ~45 s).
- Memory: <1 GB RSS during run (pandas: ~4 GB).
- Existing `pytest -m "not slow"` stays green.

### NOT in scope
- Migrating the entire repo to Polars. Pandas stays at the I/O edge
  (yfinance/Polygon/Alpaca return pandas), the report layer (matplotlib
  expects pandas), and the QA layer where snapshots are already pinned.

## 2. Numba (B-002) — Tier-2 ROI

### Why
- Hotspot of the *event-driven* backtest (per-bar simulate_trades loop)
  is pure-Python today.
- Audit observed Pure-Python 42 ms → Numba @njit warm 70 µs = ~600×.

### Scope (½-day sprint)
- One function only: `simulate_trades(prices, signals, fees, slippage)`
  in `src/assembled_core/qa/backtest_engine.py` (or wherever the
  current event loop lives).

### Migration steps
1. `pip install numba` (no version pin yet — minor versions are usually
   safe). Confirm `np` and `pd` co-versions are still compatible.
2. Refactor the loop so it takes pure numpy arrays in / out (Numba
   doesn't speak pandas).
3. `@njit(cache=True, fastmath=True, boundscheck=False)` decorator.
4. The function should be a *pure function*: no global state, no
   logging inside the hot loop, no exceptions.
5. Add a benchmark (use the W15 stdlib pattern, not pytest-benchmark
   which is also missing): 10 000 bars in <2 ms.

### Acceptance criteria
- Numerical parity with the pure-Python baseline to 1e-9.
- Cold-start (JIT compile) <2 s for the first call; warm calls <1 ms.

### Risks
- Numba @njit cannot call into pandas / sklearn / requests. The loop
  must be fully numerical before wrapping. Refactor first, jit second.
- AOT compile increases CI install time by 30-90 s. Mitigation: keep
  the JIT path optional via env (`ASSEMBLED_USE_NUMBA=1`).

## 3. Rust extension via PyO3 (B-003) — DEFER

### Why deferred
- Audit's own recommendation: "Erst MITTELFRISTIG (Monat 4–6)".
- Solo-Dev cost: a second build chain (Rust + maturin) for a 2-4×
  speedup over Numba.
- Polars+Numba together close enough to make Rust a luxury.

### When to revisit
- Optuna hyperparameter sweeps with ≥5 000 backtests per session.
- A single backtest iteration costing >50 ms even after Numba.
- Available time for maintaining a second codebase.

## 4. Async I/O (B-004) — **shipped** in Wave 16

`src/assembled_core/utils/async_fetch.py` is the new dep-free version
(httpx + stdlib retry). Callers can opt in incrementally; the existing
sync paths are not touched.

## 5. Caching (B-005) — **partly shipped**

FactorStore (`src/assembled_core/data/factor_store.py`) already has
sha256-anchored Parquet caching with the PIT-safe `as_of` filter.
W13 added `tests/test_property_fsm_pit.py` order-invariance property
test for the universe key.

What is still **not** wired: L1 `lru_cache`, L2 `cachetools.TTLCache`,
L3 `joblib.Memory`. Each is < 30 LoC to add when the call-site for it
materialises. Documented as on-demand.

## 6. Memory micro-wins (B-006) — **shipped** in Wave 15

`slots=True` on PaperOrder / PaperPosition / OrderEvent / TrackedOrder.
Memory per instance halved; attribute-typo protection as a bonus.

## 7. Vectorized vs event-driven backtest (B-008) — DEFER

The current backtest engine is mixed (some vectorised, some event-driven
per pipeline path). Splitting cleanly into two engines is a large
refactor with no obvious near-term ROI — the audit's own bench shows
Pandas vectorised is already 10-30× faster than the current event loop
for trend-style strategies. Numba (§2) closes the event-driven gap;
Polars (§1) speeds up the vectorised path. The architectural split is
a Q3 task at the earliest.

## 8. Hexagonal architecture (C-001..C-007) — DEFER

40-80 hours of structural refactor. Audit's own estimate. **Not** an
audit-sweep deliverable; tracked separately in
`autonome_weiterarbeit/AUDIT_SWEEP_2026-05-12.md` §3.2.

The minimum-viable architectural improvement the sweep DID ship: each
critical capability now has a **single source of truth** (latency
constants, alerting, kill-switch state). That gives a future hexagonal
refactor a clean perimeter to cut around.

## Owner-readable checklist (TL;DR)

| Item | State | Effort to ship |
|---|---|---|
| Polars migration of ta_features | Plan only | 1 day |
| Numba @njit on simulate_trades | Plan only | ½ day |
| Rust/PyO3 extension | Deferred | weeks |
| Async I/O helper | ✓ shipped (W16) | — |
| Caching tiers L1/L2/L3 | On demand | <½ day each |
| dataclass slots | ✓ shipped (W15) | — |
| Vectorised / event-driven split | Deferred | weeks |
| Hexagonal architecture | Deferred | weeks |
