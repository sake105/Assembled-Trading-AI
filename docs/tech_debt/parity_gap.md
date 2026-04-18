# Tech Debt — Backtest ↔ Paper Parity Gap

**Opened:** 2026-04-18
**Sunset target:** 2026-07-01
**Tracked P0:** A8 (Deep Run v2, 2026-04-18)
**Owner:** unassigned
**Status:** Open

---

## What this is

The Ultra-Plan (E0.1) requires that `run_portfolio_backtest` and
`run_paper_replay` emit **bit-identical** order streams given the same
fixture and seed. Today they do not.

Until 2026-04-18, that gap was masked by a **non-strict xfail** in
`tests/regression/test_backtest_paper_parity.py::test_bit_identical_order_stream_backtest_vs_paper`.
The test was removed as part of P0 finding A8 because a permanent
non-strict xfail is indistinguishable from a deleted test in CI — it
neither blocks a regression nor flags a remaining-work signal.

## Where the gap lives

Two independent loops today:

1. `src/assembled_core/qa/backtest_engine.py::run_portfolio_backtest`
   — uses the backtest-side position-evolution model
   (pure internal ledger, no Alpaca-shaped fill semantics, no TCA).
2. `src/assembled_core/ops/replay_snapshot.py::run_paper_replay`
   — uses the paper-engine's position-evolution model
   (fill model, TCA, intent store, cost calibrator — closer to live).

Until the fill-model and position-evolution semantics are unified behind a
single interface, the two loops cannot produce identical `orders_df`
content even on the same fixture.

## Known determinism issue (fixed 2026-04-18)

~~`run_paper_replay.orders_df` records `timestamp` as `time.time_ns()` —
two replays of the same fixture produce different timestamps.~~

**Fixed** by passing `order_timestamp=ts` into `TradingContext` in
`run_paper_replay` (was defaulting to `pd.Timestamp.now("UTC")` via the
dataclass `default_factory`). The `test_run_paper_replay_emits_
deterministic_orders` regression test is now green. The remaining gap
below is the cross-loop equality, not replay determinism.

## Definition of done (sunset criteria)

To close this tech-debt entry:

1. `run_paper_replay` uses a deterministic timestamp source
   (no `time.time_ns()` / `time.time()` / `datetime.now()` inside the
   order-emission path).
2. The position-evolution and fill model are unified — either both loops
   delegate to the same `PaperEngine` or both call a shared ledger
   primitive.
3. A new **strict** test replaces the removed xfail:
   ```python
   @pytest.mark.phase_zero  # or its successor marker after A9
   def test_bit_identical_order_stream_backtest_vs_paper() -> None:
       ...
       pd.testing.assert_frame_equal(bt_orders, paper_orders, check_dtype=False)
   ```
   no xfail decorator, no `strict=False` workaround.
4. E0.1 gate in release-gate-ci gets this test as a required check.

## What to do if 2026-07-01 arrives without closure

Two options, pick one before the sunset:

- **(a) Ship the real fix** per the DoD above. Preferred.
- **(b) Formally re-open this entry** with a new sunset date and
  documented reason (e.g., a scope decision to deprecate
  `run_portfolio_backtest` in favour of a single paper-engine-driven
  backtest). Update this file, do not silently extend.

What is **not** acceptable: re-introducing a non-strict xfail with a new
reason string. That is the exact antipattern A8 was opened to kill.

## Related

- P0 A1 `.env` history: separate incident, docs/incidents/2026-04-18_env_exposure.md
- P0 A3 release-gate-ci walk-forward: now blocking, not this gap
- P0 A4 paper-stall postmortem: orthogonal
- Memory: `followup_mtm_equity_for_gross_cap.md` (A6 follow-up, same review cycle)
