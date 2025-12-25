UNIFIED TRADING CYCLE INTEGRATION IN BACKTEST (B1)
====================================================

This document describes the plan for integrating `run_trading_cycle()` into the
backtest engine to unify the trading cycle logic across EOD, backtest, and
paper-trading workflows.

1. Current Backtest Flow (Status Quo)
--------------------------------------

1.1 Entry Point: scripts/run_backtest_strategy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The CLI script `scripts/run_backtest_strategy.py` creates strategy-specific
callables and calls the backtest engine:

- `create_trend_baseline_signal_fn(ma_fast, ma_slow)`: Returns `signal_fn(prices_df) -> signals_df`
- `create_position_sizing_fn()`: Returns `position_sizing_fn(signals_df, capital) -> target_positions_df`
- Calls `run_portfolio_backtest(prices, signal_fn, position_sizing_fn, ...)`

1.2 Backtest Engine: src/assembled_core/qa/backtest_engine.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function `run_portfolio_backtest()` orchestrates:

**Step 1: Feature Computation (once, upfront)**
- If `compute_features=True`: Compute TA features for all prices
  - Uses `add_all_features()` or `build_or_load_factors()` (if `use_factor_store=True`)
  - Results in `prices_with_features` DataFrame

**Step 2: Signal Generation (once, upfront)**
- Call `signal_fn(prices_with_features)` to generate signals for all timestamps
- Results in `signals` DataFrame with columns: timestamp, symbol, direction, score
- Optional: Apply meta-model ensemble (if `use_meta_model=True`)

**Step 3: Per-Timestamp Loop (main rebalancing loop)**
- Group signals by `timestamp` to get rebalancing points
- For each timestamp `ts` in timeline:
  - Get `signal_group` (signals for this timestamp)
  - Call `_process_rebalancing_timestamp()`:
    1. Compute target positions: `targets = position_sizing_fn(signal_group, capital)`
    2. Generate orders: `orders = generate_orders_from_targets(targets, current_positions, ts, prices)`
    3. Update positions: `updated_positions = _update_positions_vectorized(orders, current_positions)`
  - Append orders to all_orders list
  - Update `current_positions` for next iteration

**Step 4: Equity Simulation**
- If `include_costs=True`: Call `simulate_with_costs(orders, prices, ...)`
- If `include_costs=False`: Call `simulate_equity(orders, prices, ...)`
- Returns equity curve DataFrame

**Step 5: Metrics Computation**
- Call `compute_all_metrics(equity)` to get performance metrics
- Returns BacktestResult with equity, metrics, optional trades/signals/targets

1.3 Key Hotspots (Current Structure - Legacy Path)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File: src/assembled_core/qa/backtest_engine.py**

- `run_portfolio_backtest()` (line ~423):
  - Feature computation (Step 1): Lines ~540-562 (if `compute_features=True` and `cycle_fn=None`)
  - Signal generation (Step 2): Line ~574 (if `cycle_fn=None`)
  - Per-timestamp loop (Step 3): Lines ~790-850 (legacy path with `_process_rebalancing_timestamp()`)
  - Equity simulation (Step 4): Lines ~850-920 (estimated)

- `_process_rebalancing_timestamp()` (line ~317):
  - Position sizing: Line ~353
  - Order generation: Lines ~356-374
  - Position update: Line ~377

**File: src/assembled_core/pipeline/trading_cycle.py (New Unified Path)**

- `run_trading_cycle()` (line ~460):
  - Feature building hook (line ~331-391): Calls `build_or_load_factors()` if `use_factor_store=True`
  - Signal generation hook (line ~394-430): Calls `signal_fn(prices_with_features)`
  - Position sizing hook (line ~432-465): Calls `position_sizing_fn(signals, capital)`
  - Order generation hook (line ~467-497): Calls `generate_orders_from_targets()`

- `make_cycle_fn()` (line ~248, in backtest_engine.py):
  - Adapter function that creates callable for each timestamp
  - Calls `run_trading_cycle()` per timestamp (line ~312)

**File: scripts/run_backtest_strategy.py**

- `run_backtest_from_args()` (line ~731):
  - Strategy-specific signal/sizing function creation: Lines ~884-1053
  - Call to `run_portfolio_backtest()`: Line ~1101
  - TradingContext template creation: Lines ~1070-1090 (if using TradingCycle path)

1.4 Inputs as Callables (Current Pattern)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Current approach uses callables passed as function parameters:

```python
signal_fn: Callable[[pd.DataFrame], pd.DataFrame]
  # Input: prices_with_features (all timestamps)
  # Output: signals DataFrame (all timestamps)

position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame]
  # Input: (signals_df for single timestamp, capital)
  # Output: target_positions DataFrame for that timestamp
```

2. Target Architecture: Unified Trading Cycle
---------------------------------------------

2.1 Goal
~~~~~~~~

Replace the separate steps (features -> signals -> sizing -> orders) in the
backtest engine with a unified `run_trading_cycle()` call per timestamp.

**Target Loop:**
```python
for as_of in timestamps:
    ctx.as_of = as_of
    ctx.prices = prices  # Full prices DataFrame
    ctx.current_positions = current_positions
    result = run_trading_cycle(ctx)
    orders = result.orders
    apply_fills(orders)
    update_equity(cash, positions)
    update_positions(current_positions, orders)
```

2.2 Benefits
~~~~~~~~~~~~

- **Code Reuse**: Same logic for EOD, backtest, paper-trading
- **Consistency**: Guaranteed same behavior across workflows
- **Maintainability**: Single place to fix bugs or add features
- **PIT Safety**: `as_of` is explicit and enforced per timestamp

2.3 Key Differences from Current Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Current:** Features computed once upfront, then signals for all timestamps
**Target:** Features computed per timestamp (with PIT filtering via `as_of`)

This difference is important for:
- PIT-safe alt-data features (B2): Features must be computed with `as_of=timestamp`
- Feature caching: Can still cache features, but filter by `as_of` per timestamp

3. Migration Strategy
---------------------

3.1 Step 1: Adapter Function (Minimal Changes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an adapter that bridges current callable-based API to `run_trading_cycle()`:

**Option A: Wrapper Function**
```python
def _run_trading_cycle_adapter(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    signal_fn: Callable,
    position_sizing_fn: Callable,
    capital: float,
    current_positions: pd.DataFrame,
    feature_config: dict | None,
    use_factor_store: bool,
) -> TradingCycleResult:
    """Adapter: Build TradingContext from callables and run trading cycle."""
    # Filter prices to as_of (PIT-safe)
    prices_filtered = prices[prices["timestamp"] <= as_of].copy()
    
    # Build context
    ctx = TradingContext(
        prices=prices_filtered,
        as_of=as_of,
        freq="1d",  # or from config
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=capital,
        current_positions=current_positions,
        feature_config=feature_config,
        use_factor_store=use_factor_store,
        write_outputs=False,
    )
    
    # Run trading cycle
    return run_trading_cycle(ctx)
```

**Option B: Direct Integration**
Replace `_process_rebalancing_timestamp()` with direct `run_trading_cycle()` call.

3.2 Step 2: Engine Refactoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refactor `run_portfolio_backtest()` to use `run_trading_cycle()`:

**Before (Current):**
```python
# Step 1: Features (once)
prices_with_features = add_all_features(prices, ...)

# Step 2: Signals (once)
signals = signal_fn(prices_with_features)

# Step 3: Per-timestamp loop
for ts in timeline:
    signal_group = signals[signals["timestamp"] == ts]
    orders, updated_positions, _ = _process_rebalancing_timestamp(
        ts, signal_group, current_positions, position_sizing_fn, ...
    )
```

**After (Target):**
```python
# Per-timestamp loop (features computed per timestamp with PIT)
for as_of in timeline:
    # Use adapter or direct run_trading_cycle()
    cycle_result = _run_trading_cycle_adapter(
        prices=prices,
        as_of=as_of,
        signal_fn=signal_fn,
        position_sizing_fn=position_sizing_fn,
        capital=current_capital,  # Updated each iteration
        current_positions=current_positions,
        ...
    )
    
    orders = cycle_result.orders
    # Continue with fills, equity update, etc.
```

3.3 Step 3: Feature Computation Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Challenge:** Features computed per timestamp might be slow.

**Solution:** Use factor store with PIT-safe loading:

```python
# In _run_trading_cycle_adapter or run_trading_cycle():
if use_factor_store:
    prices_with_features = build_or_load_factors(
        prices_filtered,
        factor_group=factor_group,
        as_of=as_of,  # PIT-safe: only factors <= as_of
        ...
    )
else:
    prices_with_features = add_all_features(prices_filtered, ...)
```

The factor store ensures that features are cached but loaded with PIT filtering.

3.4 Step 4: Backward Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Keep the current API working:

- `signal_fn` and `position_sizing_fn` callables still accepted
- Internally, wrap them in `TradingContext` adapters
- No breaking changes to existing backtest scripts

4. Risks and Mitigation
------------------------

4.1 Performance: Feature Computation Per Timestamp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Risk:** Computing features for each timestamp individually might be slower
than computing once upfront.

**Mitigation:**
- Use factor store with caching (already PIT-safe)
- Only compute features for timestamps that actually have signals/rebalancing
- Benchmark: Compare old vs. new performance (should be similar if factor store used)

4.2 PIT Safety: Features Must Respect as_of
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Risk:** Features computed upfront might leak future information.

**Current State:** Features are computed once upfront, which is safe for TA
features (they only use past prices), but might be unsafe for alt-data features
that should respect disclosure_date.

**Target State:** Features computed per timestamp with `as_of` ensures PIT safety
for all feature types (TA and alt-data).

**Mitigation:**
- Factor store already supports `as_of` parameter
- Alt-data feature builders already support `as_of` parameter
- Tests ensure PIT safety (existing B2 tests)

4.3 Determinism: Same Results as Before
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Risk:** Refactoring might change behavior/results.

**Mitigation:**
- Keep regression tests (e.g., `test_backtest_regression.py`)
- Compare equity curves (should be identical for TA-only strategies)
- For alt-data strategies, new results should be more PIT-safe (may differ)

4.4 Complexity: Adapter Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Risk:** Adding adapter layer adds complexity.

**Mitigation:**
- Start with simple adapter, can be removed later once all strategies use
  TradingContext directly
- Document adapter clearly (temporary bridge)

5. Testing Strategy
--------------------

5.1 Regression Tests
~~~~~~~~~~~~~~~~~~~~

- Run existing backtest regression tests
- Compare equity curves: old vs. new (should be identical for TA strategies)
- Verify metrics (Sharpe, trades, etc.) are unchanged

5.2 PIT Safety Tests
~~~~~~~~~~~~~~~~~~~~

- Use existing B2 PIT tests
- Verify that alt-data features respect `as_of` in backtest context
- Test with events that have disclosure_date > event_date

5.3 Performance Tests
~~~~~~~~~~~~~~~~~~~~~

- Benchmark old vs. new backtest runtime
- Target: < 10% slowdown (or speedup if factor store used efficiently)
- Profile hotspots: feature computation, signal generation

5.4 Integration Tests
~~~~~~~~~~~~~~~~~~~~~

- Test with all strategy types: trend_baseline, event_insider_shipping, multifactor_long_short
- Test with factor store enabled/disabled
- Test with meta-model ensemble enabled

6. Migration Steps (Detailed)
------------------------------

6.1 Phase 1: Adapter Function (No Breaking Changes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create `_run_trading_cycle_adapter()` function
- Keep existing `run_portfolio_backtest()` unchanged
- Add tests for adapter function
- **Deliverable:** Adapter function exists and tested

6.2 Phase 2: Optional Integration (Feature Flag)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Add parameter `use_trading_cycle: bool = False` to `run_portfolio_backtest()`
- If `use_trading_cycle=True`, use adapter internally
- Keep old path as fallback
- **Deliverable:** Both paths work, can toggle via flag

6.3 Phase 3: Default to Trading Cycle (Backward Compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Change default: `use_trading_cycle: bool = True`
- Keep old path available via `use_trading_cycle=False`
- Run regression tests, fix any issues
- **Deliverable:** Trading cycle is default, old path still available

6.4 Phase 4: Remove Old Path (Cleanup)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Remove old feature/signal computation logic
- Remove `use_trading_cycle` flag (always use trading cycle)
- Update documentation
- **Deliverable:** Single path, cleaner codebase

7. Key Design Decisions
------------------------

7.1 Per-Timestamp vs. Batch Feature Computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision:** Per-timestamp with factor store caching.

**Rationale:**
- Enables PIT-safe alt-data features
- Factor store provides caching (no performance penalty)
- Simpler logic: same pattern for all feature types

7.2 Adapter vs. Direct Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision:** Start with adapter, migrate to direct integration later.

**Rationale:**
- Minimal risk: existing code continues to work
- Can test adapter independently
- Can migrate strategies one by one

7.3 Capital Updates in Loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision:** Update capital per timestamp based on equity value.

**Rationale:**
- Position sizing should use current portfolio value, not start_capital
- Matches paper-trading behavior
- More realistic (compounding)

8. Open Questions
-----------------

8.1 Meta-Model Ensemble
~~~~~~~~~~~~~~~~~~~~~~~

**Current:** Meta-model applied to signals after generation (all timestamps).

**Question:** Should meta-model be applied per timestamp in trading cycle?

**Proposal:** Move meta-model to a hook in `run_trading_cycle()` (e.g., `risk_controls` hook).

8.2 Cost Model Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Current:** Costs applied in equity simulation step (after order generation).

**Question:** Should costs be applied during order generation in trading cycle?

**Proposal:** Keep costs in equity simulation (separation of concerns: trading cycle generates orders, backtest engine simulates execution).

8.3 Factor Store Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Current:** Factor store params passed to `run_portfolio_backtest()`.

**Question:** Should factor store be configured in TradingContext or passed separately?

**Proposal:** Configure in TradingContext (matches EOD/paper-trading pattern).

9. Performance Mitigation (P3)
------------------------------

9.1 Hotspots (Functions + Files)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File: src/assembled_core/qa/backtest_engine.py**

- `run_portfolio_backtest()` (line ~423):
  - Per-timestamp loop (line ~751-790): Calls `cycle_fn()` for each timestamp
  - Feature computation per timestamp (via TradingCycle, line ~331-391)
  - Signal generation per timestamp (via TradingCycle, line ~394-430)
  - Position sizing per timestamp (via TradingCycle, line ~432-465)
  - Order generation per timestamp (via TradingCycle, line ~467-497)

- `make_cycle_fn()` (line ~248):
  - Adapter function that creates callable for each timestamp
  - Calls `run_trading_cycle()` per timestamp (line ~312)

**File: src/assembled_core/pipeline/trading_cycle.py**

- `run_trading_cycle()` (line ~460):
  - Feature building hook (line ~331-391): Calls `build_or_load_factors()` if `use_factor_store=True`
  - Signal generation hook (line ~394-430): Calls `signal_fn(prices_with_features)`
  - Position sizing hook (line ~432-465): Calls `position_sizing_fn(signals, capital)`
  - Order generation hook (line ~467-497): Calls `generate_orders_from_targets()`

**File: src/assembled_core/features/factor_store_integration.py**

- `build_or_load_factors()` (line ~26):
  - Cache check (line ~89-117): Loads from factor store if available
  - Feature computation (line ~125-140): Computes features if cache miss
  - Store to cache (line ~149-160): Stores computed factors (append mode for incremental writes)

**File: src/assembled_core/portfolio/position_sizing.py**

- `compute_target_positions_from_trend_signals()` (if used):
  - Per-symbol loop for position sizing calculations

9.2 Performance Mitigations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Factor Store Caching (P2)**
- **Mitigation:** Use `use_factor_store=True` to cache computed features
- **Implementation:** `build_or_load_factors()` checks cache first, computes only if missing
- **Impact:** Avoids recomputing features for same timestamps in repeated backtests
- **Note:** In backtest mode with TradingCycle, features are computed per timestamp but cached by year partition

**2. Append Mode for Incremental Writes**
- **Mitigation:** Factor store uses `mode="append"` (not "overwrite") when storing factors
- **Implementation:** `store_factors(..., mode="append")` merges with existing year partitions
- **Impact:** Avoids rewriting entire factor files per timestamp, only missing partitions written
- **File:** `src/assembled_core/features/factor_store_integration.py`, line ~154

**3. Per-Timestamp vs. Batch Computation**
- **Trade-off:** Features computed per timestamp (PIT-safe) vs. batch computation (faster)
- **Mitigation:** Factor store caching ensures features only computed once per timestamp
- **Future Optimization:** Could pre-compute features for all timestamps upfront, then filter by `as_of` per timestamp (if PIT safety can be guaranteed)

**4. Vectorization (P3)**
- **Mitigation:** Order execution and position updates use vectorized NumPy operations
- **Implementation:** `_update_positions_vectorized()` uses NumPy arrays instead of per-order loops
- **File:** `src/assembled_core/qa/backtest_engine.py`, line ~125-245

**5. Optional Numba Acceleration (P3)**
- **Mitigation:** Numba JIT compilation for critical loops (if Numba installed)
- **Implementation:** `backtest_engine_numba.py` provides `@njit` functions for position deltas
- **Fallback:** Pure NumPy if Numba not available (no API changes)
- **File:** `src/assembled_core/qa/backtest_engine_numba.py`

9.3 Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Profiling:**
```bash
# Profile BACKTEST_MEDIUM with cProfile
python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler cprofile

# Profile with pyinstrument
python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler pyinstrument
```

**Benchmark:**
```bash
# Run BACKTEST_MEDIUM benchmark (3 iterations, median runtime)
python scripts/profile_job.py --job BACKTEST_MEDIUM --profiler cprofile
```

**Expected Hotspots (after P3 optimizations):**
1. Feature computation (if factor store cache miss)
2. Signal generation per timestamp
3. Position sizing per timestamp
4. Equity simulation (vectorized, should be fast)

**Target Performance:**
- < 10% slowdown vs. batch computation (with factor store enabled)
- Factor store cache hit: < 5% overhead vs. batch computation
- Factor store cache miss: Similar to batch computation (first run)

9.4 Future Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~

**Potential Optimizations:**
- Pre-compute features for all timestamps upfront (if PIT safety can be guaranteed)
- Batch signal generation for multiple timestamps (if signals don't depend on `as_of`)
- Parallelize per-timestamp loop (if order generation is independent)
- Incremental feature computation (only compute features for new timestamps)

**Constraints:**
- PIT safety must be maintained (no look-ahead bias)
- Deterministic results required (no race conditions)
- Backward compatibility (existing backtests must produce same results)

10. Related Documentation
-------------------------

- `docs/UNIFIED_TRADING_CYCLE_B1.md`: Original design document
- `docs/POINT_IN_TIME_AND_LATENCY.md`: PIT safety rules (B2)
- `docs/FACTOR_STORE_P2.md`: Factor store documentation (P2)
- `docs/BACKTEST_ENGINE_OPTIMIZATION_P3.md`: Performance optimizations (P3)
- `src/assembled_core/pipeline/trading_cycle.py`: Implementation

11. Success Criteria
--------------------

- Backtest results identical (for TA-only strategies) or more PIT-safe (for alt-data strategies)
- Performance: < 10% slowdown (or speedup) with factor store enabled
- All existing tests pass
- Code coverage maintained or improved
- Documentation updated

