# Robustness Playbook (Sprint 12)

## Purpose

This document defines the robustness suite for strategy validation. The robustness suite
ensures that strategies are not overfitted and perform consistently across different
market conditions, parameter settings, and time periods.

## Goals (RB1-RB5)

### RB1: Deterministic Walk-Forward Analysis

**Objective**: Ensure walk-forward splits are deterministic and reproducible.

**Requirements**:
- Walk-forward splits must be generated deterministically (same config -> same splits)
- Split dates must be stable (no floating-point date arithmetic issues)
- Output ordering must be deterministic (sorted by split_index or test_start)

**Artefacts**:
- `output/robustness/walk_forward_{strategy_id}_{freq}.json`: Walk-forward result (JSON)
- `output/robustness/walk_forward_{strategy_id}_{freq}_splits.csv`: Split details (CSV)
- `output/robustness/walk_forward_{strategy_id}_{freq}_metrics.csv`: Per-split metrics (CSV)

**Determinism Rules**:
- Sort splits by `split_index` (ascending) before writing
- Use UTC timestamps (no timezone ambiguity)
- Use integer day arithmetic (no floating-point days)
- JSON output: `sort_keys=True`, `indent=2` for stable formatting
- CSV output: Sort by `split_index` (ascending)

### RB2: Parameter Sweep (Small Grid)

**Objective**: Test strategy robustness across parameter variations.

**Requirements**:
- Small, deterministic parameter grid (e.g., 3x3 = 9 combinations)
- Each combination must be run deterministically
- Results must be aggregated and compared

**Artefacts**:
- `output/robustness/param_sweep_{strategy_id}_{freq}.json`: Sweep result (JSON)
- `output/robustness/param_sweep_{strategy_id}_{freq}_results.csv`: Per-combination results (CSV)
- `output/robustness/param_sweep_{strategy_id}_{freq}_summary.csv`: Summary statistics (CSV)

**Determinism Rules**:
- Parameter combinations must be generated deterministically (sorted parameter keys)
- Results CSV: Sort by parameter combination (lexicographic order)
- JSON output: `sort_keys=True`, `indent=2`
- No random seeds needed (deterministic backtest engine)

### RB3: Deflated Sharpe Ratio (Multiple Testing Adjustment)

**Objective**: Adjust Sharpe ratio for multiple testing (parameter sweeps, factor tests).

**Requirements**:
- Compute deflated Sharpe ratio for each strategy/parameter combination
- Use correct `n_tests` (number of parameter combinations or factors tested)
- Report deflated Sharpe in robustness reports

**Artefacts**:
- Included in `param_sweep_*_summary.csv`: `deflated_sharpe` column
- Included in `walk_forward_*_metrics.csv`: `deflated_sharpe` column (if applicable)

**Determinism Rules**:
- `n_tests` must be deterministic (count of parameter combinations)
- Deflated Sharpe calculation must be deterministic (no random components)
- Use existing `deflated_sharpe_ratio()` from `qa.metrics` (already deterministic)

### RB3: Sensitivity Suite (Cost, Slippage, Alt-Data Delay)

**Objective**: Test strategy robustness to cost variations, slippage, and alt-data disclosure delays.

**Requirements**:
- Run backtest with multiple sensitivity variants:
  - baseline: Original configuration
  - costs_x2: All costs doubled (commission_bps, spread_w, impact_w × 2)
  - slippage_x2: Only slippage/impact doubled (impact_w × 2)
  - alt_delay_{d}: Alt-data disclosure_date shifted by d days
- Variants must be generated deterministically (sorted order)
- PIT-safety must be preserved for delay_days > 0

**Artefacts**:
- `output/robustness/<run_id>/sensitivity_results.csv`: Per-variant results (CSV)

**Determinism Rules**:
- Variant names must be sorted deterministically
- CSV output: Sort by variant_name (ascending)
- delay_days < 0 must be clearly marked with WARNING (may introduce leakage)

**PIT-Safety Rules**:
- delay_days > 0: Events become visible LATER (stricter PIT, safe)
- delay_days < 0: Events become visible EARLIER (stress test, WARNING required)
- delay_days = 0: No change (covered by baseline)

### RB4: Crisis Windows Evaluation

**Objective**: Test strategy performance during historical crisis periods.

**Requirements**:
- Run backtest restricted to specific crisis date ranges
- Evaluate pass/fail flags per window (max_dd threshold, sharpe floor)
- Standard windows: GFC (2007-12 to 2009-06), COVID (2020-02 to 2020-04), 2022_RATES (2022)

**Artefacts**:
- `output/robustness/<run_id>/crisis_windows.csv`: Per-window results (CSV)

**Determinism Rules**:
- Windows must be sorted by start date, then by name (deterministic)
- CSV output: Sort by window_start, then window_name (ascending)
- Date ranges are [start, end) (start inclusive, end exclusive)

**Pass/Fail Criteria**:
- pass_max_dd: max_drawdown >= max_dd_threshold (default: -0.30 = -30%)
- pass_sharpe: sharpe >= sharpe_floor (default: -1.0)
- pass_overall: pass_max_dd AND pass_sharpe

**Semantics**:
- Date ranges are [start, end) (start inclusive, end exclusive)
- Backtest is restricted to window date range via config start_date/end_date
- All dates are in UTC timezone

### RB4b: Stress Testing (Crisis Scenarios)

**Objective**: Test strategy performance under stress conditions (crashes, volatility spikes).

**Requirements**:
- Apply predefined stress scenarios to price data
- Run backtest on stressed prices
- Compare baseline vs. stressed metrics

**Artefacts**:
- `output/robustness/stress_{strategy_id}_{freq}_{scenario_name}.json`: Stress result (JSON)
- `output/robustness/stress_{strategy_id}_{freq}_{scenario_name}_comparison.csv`: Baseline vs. stressed (CSV)

**Determinism Rules**:
- Scenario application must be deterministic (same prices + scenario -> same stressed prices)
- Use existing `scenario_engine` (already deterministic)
- JSON output: `sort_keys=True`, `indent=2`
- CSV output: Sort by `timestamp` (ascending)

### RB5: Multiple Testing / Deflated Sharpe

**Objective**: Correct for selection bias and multiple testing in parameter sweeps.

**Requirements**:
- Compute deflated Sharpe ratio adjusted for multiple testing (Bailey & López de Prado 2014)
- Build heuristic warnings for potential overfitting (large n_trials + inflated best metric)
- Integrate into RB2 (parameter sweep) to report deflated Sharpe for best run

**Artefacts**:
- `output/robustness/<run_id>/warnings.json`: Multiple testing warnings (JSON)
- Deflated Sharpe computed for best parameter combination in RB2

**Determinism Rules**:
- Deflated Sharpe computation must be deterministic (same inputs -> same output)
- Warnings JSON: `sort_keys=True`, `indent=2`
- NaN/Inf values converted to None for JSON compatibility

**Deflated Sharpe Formula**:
- Adjusts observed Sharpe ratio for:
  - Multiple testing (Bonferroni-like correction: alpha / n_trials)
  - Non-normality (skewness, kurtosis adjustments)
- Formula: DS = SR * sqrt((1 - gamma * SR) / (n_obs - 1)) - sqrt((1 - gamma * SR) / (n_obs - 1)) * Z(1 - alpha / n_trials)
  where gamma = (skew * SR) / 4 + ((kurt - 3) * SR^2) / 24

**Warning Heuristic**:
- warning_inflated = True if: n_trials >= 10 AND metric_spread > 2.0
  where metric_spread = best_metric - median_metric
- Provides human-readable warning message when triggered

### RB5: Robustness Pack (Combined Report)

**Objective**: Generate a combined robustness report with all RB1-RB4 results.

**Requirements**:
- Aggregate results from RB1-RB4
- Generate summary report (Markdown)
- Include pass/fail criteria for candidate status

**Artefacts**:
- `output/robustness/robustness_pack_{strategy_id}_{freq}.md`: Combined report (Markdown)
- `output/robustness/robustness_pack_{strategy_id}_{freq}.json`: Machine-readable summary (JSON)

**Determinism Rules**:
- Report generation must be deterministic (same inputs -> same report)
- Markdown sections must be in fixed order: RB1, RB2, RB3, RB4, Summary
- JSON output: `sort_keys=True`, `indent=2`
- Candidate status must be deterministic (clear pass/fail criteria)

## DoD: Candidate Status Only with Robustness Pack

**Rule**: A strategy can only achieve "candidate" status if it has passed the robustness pack.

**Pass Criteria** (all must be met):
1. **Walk-Forward (RB1)**: Mean OOS Sharpe >= 0.5, OOS Win Rate >= 0.5
2. **Parameter Sweep (RB2)**: At least 50% of parameter combinations have Sharpe >= 0.5
3. **Sensitivity Suite (RB3)**: All variants (costs_x2, slippage_x2, alt_delay) have Sharpe >= 0.0
4. **Crisis Windows (RB4)**: At least 50% of crisis windows pass (pass_overall = True)
5. **Deflated Sharpe (RB5)**: Deflated Sharpe >= 0.3 (adjusted for multiple testing)
6. **Stress Testing (RB4b)**: Stressed Sharpe >= 0.0 (strategy doesn't break under stress)

**Implementation**:
- `robustness_pack_*_summary.json` must contain `candidate_status: "pass" | "fail"`
- `robustness_pack_*_summary.json` must contain `pass_criteria` dict with per-rule status
- Report must clearly indicate candidate status in Markdown header

## Test Plan

### Deterministic Walk-Forward Splits

**Test**: `tests/test_robustness_walk_forward_deterministic.py`
- Generate splits twice with same config -> assert identical splits
- Assert split dates are UTC-aware and stable
- Assert output CSV is sorted by `split_index`

### Deterministic Parameter Sweeps

**Test**: `tests/test_robustness_param_sweep_deterministic.py`
- Generate parameter grid twice -> assert identical combinations
- Run backtest twice with same parameters -> assert identical results
- Assert output CSV is sorted deterministically

### Sensitivity Suite Determinism

**Test**: `tests/test_robustness_sensitivity_deterministic.py`
- Run sensitivity suite twice -> assert identical variant order
- Verify costs_x2 and slippage_x2 multipliers work correctly
- Verify alt_delay variants apply delay to events_df

### Alt-Data Delay PIT-Safety

**Test**: `tests/test_alt_delay_pit_safety.py`
- delay_days > 0: Events become visible LATER (stricter PIT, safe)
- delay_days < 0: Events become visible EARLIER (leakage risk, WARNING required)
- Verify PIT filtering respects shifted disclosure_date

### Crisis Windows Determinism

**Test**: `tests/test_robustness_crisis_windows.py`
- Run crisis windows twice -> assert identical window order
- Verify date range slicing (backtest restricted to window dates)
- Verify pass/fail flags computed correctly
- Verify deterministic ordering (sorted by start, then name)

### Deflated Sharpe / Multiple Testing

**Test**: `tests/test_deflated_sharpe_basic.py`
- Monotonicity: more trials -> lower deflated Sharpe (for same observed Sharpe)
- Deterministic outputs (same inputs -> same outputs)
- Edge cases: n_trials=1 (no adjustment), invalid inputs (return None)
- Floating point tolerance checks

**Test**: `tests/test_multiple_testing_warning.py`
- Heuristic detects inflated best metric (n_trials >= 10 AND spread > 2.0)
- Deterministic warnings (same input -> same warnings)
- Edge cases: missing columns, all-NaN, some-NaN
- Threshold edge cases (exactly at threshold)

### Deflated Sharpe Consistency

**Test**: `tests/test_robustness_deflated_sharpe.py`
- Compute deflated Sharpe twice -> assert identical values
- Verify `n_tests` is correctly counted from parameter grid
- Verify deflated Sharpe <= raw Sharpe (always)

### Stress Testing Determinism

**Test**: `tests/test_robustness_stress_deterministic.py`
- Apply scenario twice to same prices -> assert identical stressed prices
- Run backtest twice on stressed prices -> assert identical results

### Robustness Pack Integration

**Test**: `tests/test_robustness_pack_integration.py`
- Run full robustness pack -> assert all artefacts are generated
- Verify candidate status is computed correctly
- Verify report is deterministic (same inputs -> same report)

## Implementation Tasks

### RB1: Deterministic Walk-Forward

**Files**:
- `src/assembled_core/qa/robustness/walk_forward_robustness.py` (NEW)
- `scripts/run_robustness_walk_forward.py` (NEW)

**Tasks**:
1. Wrap existing `walk_forward.py` with deterministic output formatting
2. Ensure split generation is deterministic (already is, but verify)
3. Add CSV export for splits and metrics
4. Add JSON export for full result
5. Tests: `tests/test_robustness_walk_forward_deterministic.py`

### RB2: Parameter Sweep (Small Grid)

**Files**:
- `src/assembled_core/qa/robustness/param_sweep.py` (NEW)
- `scripts/run_robustness_param_sweep.py` (NEW)

**Tasks**:
1. Implement deterministic parameter grid generation
2. Run backtest for each combination (use existing `backtest_engine`)
3. Aggregate results (mean, std, min, max Sharpe across combinations)
4. Compute deflated Sharpe with correct `n_tests`
5. Export CSV and JSON
6. Tests: `tests/test_robustness_param_sweep_deterministic.py`

### RB3: Sensitivity Suite

**Files**:
- `src/assembled_core/qa/robustness.py` (extend with sensitivity functions)
- `tests/test_robustness_sensitivity_deterministic.py` (NEW)
- `tests/test_alt_delay_pit_safety.py` (NEW)

**Tasks**:
1. Implement `run_sensitivity_suite()` with variants:
   - baseline, costs_x2, slippage_x2, alt_delay_{d}
2. Implement `apply_disclosure_delay()` for alt-data delay
3. Ensure deterministic variant ordering
4. Add WARNING for delay_days < 0 (leakage risk)
5. Export results to CSV
6. Tests: Determinism, PIT-safety, cost multipliers

### RB3b: Deflated Sharpe Integration

**Files**:
- Extend `src/assembled_core/qa/metrics.py` (if needed)
- Integrate into RB1 and RB2

**Tasks**:
1. Ensure `deflated_sharpe_ratio()` is used correctly in RB1 and RB2
2. Verify `n_tests` is correctly computed (count of parameter combinations)
3. Add deflated Sharpe to CSV exports
4. Tests: `tests/test_robustness_deflated_sharpe.py`

### RB4: Crisis Windows Evaluation

**Files**:
- `src/assembled_core/qa/robustness.py` (extend with crisis windows functions)
- `tests/test_robustness_crisis_windows.py` (NEW)

**Tasks**:
1. Implement `get_standard_crisis_windows()` with GFC, COVID, 2022_RATES
2. Implement `run_crisis_windows()` to evaluate each window
3. Add pass/fail flags (max_dd threshold, sharpe floor)
4. Export results to CSV
5. Tests: Determinism, date range slicing, pass/fail logic

### RB5: Multiple Testing / Deflated Sharpe

**Files**:
- `src/assembled_core/qa/robustness.py` (extend with deflated Sharpe functions)
- `tests/test_deflated_sharpe_basic.py` (NEW)
- `tests/test_multiple_testing_warning.py` (NEW)

**Tasks**:
1. Implement `compute_deflated_sharpe()` (Bailey & López de Prado 2014 formula)
2. Implement `build_multiple_testing_warnings()` (heuristic detection)
3. Integrate into RB2: compute deflated Sharpe for best run after sweep
4. Export warnings to JSON
5. Tests: Monotonicity, determinism, edge cases

### RB4b: Stress Testing

**Files**:
- `src/assembled_core/qa/robustness/stress_testing.py` (NEW)
- `scripts/run_robustness_stress.py` (NEW)

**Tasks**:
1. Use existing `scenario_engine` for stress scenarios
2. Define standard stress scenarios (equity_crash, vol_spike)
3. Run backtest on baseline and stressed prices
4. Compare metrics (baseline vs. stressed)
5. Export CSV and JSON
6. Tests: `tests/test_robustness_stress_deterministic.py`

### RB5: Robustness Pack

**Files**:
- `src/assembled_core/qa/robustness/robustness_pack.py` (NEW)
- `scripts/run_robustness_pack.py` (NEW)

**Tasks**:
1. Orchestrate RB1-RB4 execution
2. Aggregate results into summary
3. Compute candidate status (pass/fail criteria)
4. Generate Markdown report
5. Export JSON summary
6. Tests: `tests/test_robustness_pack_integration.py`

## Existing Infrastructure

### Walk-Forward

**Location**: `src/assembled_core/qa/walk_forward.py`

**Status**: Fully implemented, deterministic split generation
- `WalkForwardConfig`: Configuration dataclass
- `generate_walk_forward_splits()`: Deterministic split generation (already uses UTC, integer day arithmetic)
- `run_walk_forward_backtest()`: Main execution function
- `WalkForwardResult`: Aggregated results with summary metrics
- Tests: `tests/test_qa_walk_forward.py` (comprehensive coverage)
- CLI: `scripts/run_walk_forward_analysis.py`, `scripts/cli.py walk_forward`

**Integration**: Wrap with deterministic output formatting (CSV/JSON exports)

**Note**: Split generation is already deterministic (UTC timestamps, integer days). Need to add CSV/JSON export with deterministic sorting.

### Parameter Sweep / Grid Search

**Location**: `src/assembled_core/experiments/batch_runner.py`, `src/assembled_core/experiments/batch_config.py`

**Status**: Grid search expansion exists, but not focused on robustness
- `BatchConfig.expand_runs()`: Generates parameter combinations deterministically (sorted keys)
- `run_batch_serial()`, `run_batch_parallel()`: Execute batch runs
- Grid expansion uses `itertools.product()` with sorted keys (deterministic)
- Tests: `tests/test_experiments_batch_config.py` (grid expansion tests)

**Integration**: Create focused robustness parameter sweep module that:
- Uses existing grid expansion logic
- Focuses on small grids (3x3 = 9 combinations) for robustness
- Computes deflated Sharpe with correct `n_tests`
- Exports CSV/JSON with deterministic sorting

**Note**: Grid expansion is already deterministic. Need to add robustness-specific aggregation and deflated Sharpe computation.

### Deflated Sharpe

**Location**: `src/assembled_core/qa/metrics.py` (`deflated_sharpe_ratio()`)

**Status**: Fully implemented, deterministic
- `deflated_sharpe_ratio(sharpe_annual, n_obs, n_tests, skew, kurtosis)`: Core function
- `deflated_sharpe_ratio_from_returns(returns, n_tests, scale)`: Convenience wrapper
- Already used in `batch_runner.py` and `factor_analysis.py`
- Tests: `tests/test_qa_deflated_sharpe.py` (if exists)

**Integration**: Use in RB1 and RB2 with correct `n_tests` (number of parameter combinations)

**Note**: Function is deterministic (no random components). Need to ensure `n_tests` is correctly counted from parameter grid.

### Stress Testing / Scenario Engine

**Location**: `src/assembled_core/qa/scenario_engine.py`

**Status**: Fully implemented, deterministic scenario application
- `Scenario`: Dataclass for scenario definition (equity_crash, vol_spike, shipping_blockade)
- `apply_scenario_to_prices(prices, scenario)`: Deterministic scenario application
- `run_scenario_on_equity(equity_series, scenario, freq)`: Run scenario on equity curve
- Scenarios are deterministic (same prices + scenario -> same stressed prices)

**Integration**: Use for RB4 stress testing
- Define standard stress scenarios (equity_crash: -20%, vol_spike: 2x)
- Run backtest on baseline and stressed prices
- Compare metrics (baseline vs. stressed)

**Note**: Scenario application is already deterministic. Need to integrate with backtest engine and add comparison metrics.

### Manifest Writing

**Location**: `src/assembled_core/pipeline/orchestrator.py` (`run_eod_pipeline()`)

**Status**: Manifest writing exists for EOD pipeline
- Writes `run_manifest_{freq}.json` with `json.dump(..., indent=2)`
- **Issue**: Does NOT use `sort_keys=True` (needs to be added for determinism)
- Manifest includes: timestamps, completed_steps, qa_metrics, qa_gate_result, data_snapshot_id

**Integration**: Similar manifest structure for robustness pack
- Use `json.dump(..., sort_keys=True, indent=2)` for deterministic JSON
- Include strategy_id, freq, robustness results (RB1-RB4), candidate_status

**Note**: Need to ensure `sort_keys=True` is used for deterministic JSON output.

## Determinism Rules (Summary)

1. **Sorting**: All CSV/JSON outputs must be sorted deterministically
   - CSV: Sort by primary key (split_index, parameter combination, timestamp)
   - JSON: `sort_keys=True`, `indent=2`

2. **Timestamps**: All timestamps must be UTC-aware (no naive timestamps)

3. **Date Arithmetic**: Use integer days (no floating-point days)

4. **Random Seeds**: Not needed (backtest engine is deterministic)

5. **File Paths**: Use deterministic paths (no timestamps in filenames, use strategy_id)

6. **JSON Serialization**: Use `json.dump(..., sort_keys=True, indent=2)` for stable formatting

7. **CSV Writing**: Use `csv.DictWriter` with sorted fieldnames

## File Locations

**Output Directory**: `output/robustness/`

**Naming Convention**:
- `{component}_{strategy_id}_{freq}.{ext}` (e.g., `walk_forward_ema_20_60_1d.json`)
- `robustness_pack_{strategy_id}_{freq}.{ext}` (e.g., `robustness_pack_ema_20_60_1d.md`)

**Strategy ID**: Short identifier (e.g., `ema_20_60`, `multifactor_ls`)

## Findings (Sprint 12 Initial State Capture)

### Baseline Status
- **pytest**: Not run (path issues in terminal, but test suite exists)
- **ruff check**: Not run (path issues, but linting config exists)
- **py_compile**: Not run (path issues, but compilation should work)

**Note**: Baseline tests should be run manually before starting implementation to establish current state.

### Existing Code Analysis

1. **Walk-Forward (`qa/walk_forward.py`)**:
   - ✅ Fully implemented with deterministic split generation
   - ✅ Uses UTC timestamps, integer day arithmetic
   - ✅ Comprehensive tests in `tests/test_qa_walk_forward.py`
   - ❌ Missing: CSV/JSON export with deterministic sorting
   - ❌ Missing: Deflated Sharpe integration

2. **Parameter Sweep (`experiments/batch_runner.py`, `batch_config.py`)**:
   - ✅ Grid expansion exists and is deterministic (sorted keys)
   - ✅ Uses `itertools.product()` with sorted parameter keys
   - ✅ Tests for grid expansion in `tests/test_experiments_batch_config.py`
   - ❌ Missing: Robustness-focused module (small grids, deflated Sharpe)
   - ❌ Missing: CSV/JSON export for robustness results

3. **Deflated Sharpe (`qa/metrics.py`)**:
   - ✅ Fully implemented: `deflated_sharpe_ratio()` and `deflated_sharpe_ratio_from_returns()`
   - ✅ Deterministic (no random components)
   - ✅ Already used in `batch_runner.py` and `factor_analysis.py`
   - ❌ Missing: Integration with walk-forward and parameter sweep for robustness

4. **Stress Testing (`qa/scenario_engine.py`)**:
   - ✅ Fully implemented: `apply_scenario_to_prices()`, `run_scenario_on_equity()`
   - ✅ Deterministic scenario application
   - ✅ Supports: equity_crash, vol_spike, shipping_blockade
   - ❌ Missing: Integration with backtest engine for robustness testing
   - ❌ Missing: Standard stress scenarios definition
   - ❌ Missing: Baseline vs. stressed comparison metrics

5. **Manifest Writing (`pipeline/orchestrator.py`)**:
   - ✅ Manifest writing exists for EOD pipeline
   - ❌ **Issue**: Does NOT use `sort_keys=True` (needs fix for determinism)
   - ✅ Uses `indent=2` for readable JSON

6. **Robustness Package**:
   - ❌ **Missing**: `src/assembled_core/qa/robustness/` directory does not exist
   - ❌ **Missing**: All RB1-RB5 modules need to be created

### Entry Points

- **Walk-Forward CLI**: `scripts/run_walk_forward_analysis.py`, `scripts/cli.py walk_forward`
- **Backtest CLI**: `scripts/run_backtest_strategy.py`
- **Batch Runner**: `scripts/cli.py batch_run` (for parameter sweeps)
- **EOD Pipeline**: `src/assembled_core/pipeline/orchestrator.py` (`run_eod_pipeline()`)

## Implementation Tasks Summary (RB1-RB5)

### RB1: Deterministic Walk-Forward Output

**Priority**: High (foundation for other robustness tests)

**Files to Create**:
- `src/assembled_core/qa/robustness/__init__.py` (NEW - package init)
- `src/assembled_core/qa/robustness/walk_forward_robustness.py` (NEW)
- `scripts/run_robustness_walk_forward.py` (NEW)
- `tests/test_robustness_walk_forward_deterministic.py` (NEW)

**Key Tasks**:
1. Create `robustness/` package directory
2. Wrap `walk_forward.run_walk_forward_backtest()` with CSV/JSON export
3. Export splits CSV: columns `split_index, train_start, train_end, test_start, test_end, n_train, n_test` (sorted by `split_index`)
4. Export metrics CSV: columns `split_index, test_sharpe, test_cagr, test_max_dd, ...` (sorted by `split_index`)
5. Export JSON: full `WalkForwardResult` with `sort_keys=True, indent=2`
6. Add deflated Sharpe column to metrics CSV (compute with `n_tests=len(splits)`)
7. Write deterministic tests (same config -> identical outputs)

**Dependencies**: `qa.walk_forward`, `qa.metrics.deflated_sharpe_ratio`

### RB2: Parameter Sweep (Small Grid)

**Priority**: High (core robustness test)

**Files to Create**:
- `src/assembled_core/qa/robustness/param_sweep.py` (NEW)
- `scripts/run_robustness_param_sweep.py` (NEW)
- `tests/test_robustness_param_sweep_deterministic.py` (NEW)

**Key Tasks**:
1. Implement `run_param_sweep_robustness()` function
2. Generate parameter grid deterministically (sorted keys, use `itertools.product()`)
3. For each combination: run backtest via `backtest_engine.run_portfolio_backtest()`
4. Aggregate results: mean, std, min, max Sharpe across combinations
5. Compute deflated Sharpe with `n_tests=len(combinations)`
6. Export results CSV: columns `param1, param2, ..., sharpe, cagr, max_dd, deflated_sharpe` (sorted by parameter combination)
7. Export summary CSV: columns `metric, mean, std, min, max` (e.g., `sharpe_mean, sharpe_std, ...`)
8. Export JSON: full sweep result with `sort_keys=True, indent=2`
9. Write deterministic tests (same grid -> identical combinations and results)

**Dependencies**: `experiments.batch_config` (for grid expansion logic), `qa.backtest_engine`, `qa.metrics.deflated_sharpe_ratio`

### RB3: Deflated Sharpe Integration

**Priority**: Medium (integration task)

**Files to Modify**:
- `src/assembled_core/qa/robustness/walk_forward_robustness.py` (add deflated Sharpe)
- `src/assembled_core/qa/robustness/param_sweep.py` (add deflated Sharpe)

**Files to Create**:
- `tests/test_robustness_deflated_sharpe.py` (NEW)

**Key Tasks**:
1. In RB1: Compute deflated Sharpe for each split (use `n_tests=len(splits)`)
2. In RB2: Compute deflated Sharpe for each combination (use `n_tests=len(combinations)`)
3. Add `deflated_sharpe` column to CSV exports
4. Verify `n_tests` is correctly counted (test with known grid sizes)
5. Write tests: deflated Sharpe <= raw Sharpe (always), deterministic computation

**Dependencies**: `qa.metrics.deflated_sharpe_ratio` (already exists)

### RB4: Stress Testing

**Priority**: Medium (important for robustness, but depends on RB1/RB2)

**Files to Create**:
- `src/assembled_core/qa/robustness/stress_testing.py` (NEW)
- `scripts/run_robustness_stress.py` (NEW)
- `tests/test_robustness_stress_deterministic.py` (NEW)

**Key Tasks**:
1. Define standard stress scenarios:
   - `equity_crash_20pct`: -20% crash starting at shock_start
   - `vol_spike_2x`: 2x volatility multiplier
2. Implement `run_stress_test()` function:
   - Load baseline prices
   - Apply scenario via `scenario_engine.apply_scenario_to_prices()`
   - Run backtest on baseline prices
   - Run backtest on stressed prices
   - Compare metrics (baseline vs. stressed)
3. Export comparison CSV: columns `metric, baseline_value, stressed_value, delta, delta_pct` (sorted by metric name)
4. Export JSON: full stress result with `sort_keys=True, indent=2`
5. Write deterministic tests (same prices + scenario -> identical stressed prices and results)

**Dependencies**: `qa.scenario_engine`, `qa.backtest_engine`

### RB5: Robustness Pack

**Priority**: High (final integration, candidate status gate)

**Files to Create**:
- `src/assembled_core/qa/robustness/robustness_pack.py` (NEW)
- `scripts/run_robustness_pack.py` (NEW)
- `tests/test_robustness_pack_integration.py` (NEW)

**Key Tasks**:
1. Implement `run_robustness_pack()` function:
   - Orchestrate RB1-RB4 execution (sequential or parallel)
   - Collect all results
   - Compute candidate status (pass/fail criteria)
2. Generate Markdown report:
   - Fixed section order: RB1, RB2, RB3, RB4, Summary
   - Include pass/fail status for each rule
   - Include candidate status in header
3. Export JSON summary:
   - `candidate_status: "pass" | "fail"`
   - `pass_criteria: {rb1: bool, rb2: bool, rb3: bool, rb4: bool}`
   - Aggregated metrics from RB1-RB4
   - Use `sort_keys=True, indent=2`
4. Write integration tests:
   - Run full pack -> assert all artifacts generated
   - Verify candidate status computation
   - Verify deterministic report generation

**Dependencies**: RB1, RB2, RB3, RB4 (all must be implemented first)

### Infrastructure Fixes

**Priority**: High (affects determinism)

**Files to Modify**:
- `src/assembled_core/pipeline/orchestrator.py` (line ~632)

**Key Tasks**:
1. Fix manifest writing: Change `json.dump(manifest, f, indent=2)` to `json.dump(manifest, f, sort_keys=True, indent=2)`
2. Verify all JSON outputs in robustness suite use `sort_keys=True`

## Next Steps (Execution Order)

1. **Baseline Tests**: Run full `pytest`, `ruff check`, `py_compile` to establish baseline
2. **Infrastructure Fix**: Fix manifest writing `sort_keys=True` (quick win)
3. **RB1**: Implement deterministic walk-forward output (foundation)
4. **RB2**: Implement parameter sweep (core robustness)
5. **RB3**: Integrate deflated Sharpe into RB1 and RB2
6. **RB4**: Implement stress testing
7. **RB5**: Implement robustness pack (final integration)
8. **Tests**: Write all deterministic tests
9. **Documentation**: Update any missing documentation
10. **Final Check**: Run full test suite, verify all artifacts are deterministic
