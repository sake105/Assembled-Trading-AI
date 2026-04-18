# Runbook 13 — Crisis-Replay Fixture Backfill

**Opened:** 2026-04-18
**Tracked P0:** A7 follow-up (Deep Run v2, 2026-04-18)
**Owner:** unassigned
**Status:** Week-1 scaffold done with synthetic fixtures; real data backfill deferred.

---

## What this is

A7's Week-1 scope was **scaffold + synthetic fixtures** (see
`tests/regression/test_crisis_replay.py`). The harness replays the reference
backtest over synthetic price paths shaped to the stylized facts of four
historical crisis windows and asserts engine-level invariants (no
bankruptcy, bounded drawdown, non-pathological trade count).

This runbook captures the **follow-up** that A7 explicitly deferred: replace
the synthetic fixtures with **real EOD bars** pulled from Polygon so the
harness detects real-world behavioural regressions, not just shape
regressions on toy data.

## Why this was deferred

Two reasons:

1. **Polygon data has a cost.** Backfilling bars for 4 windows × ~30 days ×
   10-15 tickers is roughly $50 of API calls on the monthly tier. That is
   cheap, but it is not free, and A7 Week-1 had no pre-authorised budget.
2. **Behavioural invariants should be pinned first.** A real-data harness
   is only useful after the invariants it checks are themselves stable. The
   synthetic scaffold forces us to agree on what "engine behaved correctly"
   means before we layer real prices on top.

## Event windows to fetch

| Scenario | Window | Symbols | Stylized fact |
|---|---|---|---|
| `flash_2010` | 2010-05-03 → 2010-05-10 | SPY + 10 mega-caps | single-session -9% drop, same-day recovery |
| `covid_2020` | 2020-02-14 → 2020-04-15 | SPY + 10 mega-caps | 20-session drawdown to -35%, vol ~5%/d |
| `gme_2021`   | 2021-01-20 → 2021-02-05 | GME + 10 unrelated large-caps | +1000% idiosyncratic squeeze |
| `svb_2023`   | 2023-03-06 → 2023-03-17 | SIVB, SBNY, FRC + 10 large-caps | banking-cluster crash |

Fetch via `scripts/fetch_polygon_bars.py` (already exists in the repo) or
the Polygon REST `v2/aggs/ticker/<sym>/range/1/day/<start>/<end>` endpoint.

## What to change when the real fixtures land

1. Replace `_build_<scenario>()` helpers in
   `tests/regression/test_crisis_replay.py` with:

   ```python
   def _load_<scenario>() -> pd.DataFrame:
       return pd.read_parquet(FIXTURE_ROOT / "<scenario>.parquet")
   ```

2. Commit the parquets under `tests/regression/crisis_fixtures/`
   (≤ 100 KB each — small enough for git, deterministic across clones).

3. Re-run `test_crisis_fixtures_actually_contain_shock` with the real
   drawdown numbers and update the thresholds. The test must still fail
   if someone replaces the parquets with blank baselines — that's the
   whole point of the guard.

4. Keep the synthetic builders in a separate module (e.g.
   `tests/regression/_synthetic_crisis_builders.py`) so they remain
   available for unit tests that do not want to depend on committed
   binary fixtures.

## Acceptance criteria

- [ ] All four `<scenario>.parquet` fixtures committed.
- [ ] `test_crisis_replay_engine_invariants` passes on real fixtures.
- [ ] `test_crisis_fixtures_actually_contain_shock` thresholds updated to
      reflect actual empirical drawdowns (and still fails on wiped fixtures).
- [ ] This runbook archived with status **Closed**.

## Related

- Scaffold: `tests/regression/test_crisis_replay.py` (A7 Week-1, 2026-04-18)
- Parity gate: `docs/tech_debt/parity_gap.md`
- Marker migration: `docs/tech_debt/markers_migration.md`
