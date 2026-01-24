# Ledger and Reconciliation System (Sprint 13)

## Purpose

This document defines the accounting/ledger system for paper trading that ensures
cash, positions, and PnL are tracked accurately and can be reconciled against
broker snapshots. The system must maintain full auditability and determinism.

## Goals

1. **Accurate Cash Tracking**: Every cash movement (trades, costs, dividends) is recorded
2. **Position Engine**: Track positions with average cost basis, realized/unrealized PnL
3. **Ledger Events**: Immutable, timestamped events for all accounting changes
4. **Reconciliation**: Verify ledger state matches broker snapshots (paper or live)
5. **Determinism**: Same inputs -> same ledger state (UTC, stable sorting, atomic writes)

## Ledger Event Schema

### Required Fields

- `event_id`: str (deterministic, stable across runs for same event)
- `timestamp`: datetime (tz-aware UTC)
- `event_type`: str (e.g., "TRADE", "COST", "DIVIDEND", "CASH_ADJUSTMENT")
- `symbol`: str (optional for non-trade events)
- `cash_delta`: float (change in cash, positive = inflow, negative = outflow)
- `position_delta`: float (change in position quantity, positive = buy, negative = sell)

### Optional Fields

- `order_id`: str (reference to original order)
- `fill_id`: str (reference to fill record)
- `price`: float (execution price for trades)
- `cost_basis_delta`: float (change in total cost basis)
- `realized_pnl`: float (realized PnL from this event)
- `commission`: float (commission cost)
- `spread_cost`: float (spread cost)
- `slippage_cost`: float (slippage cost)
- `notes`: str (human-readable description)

### Event Types

1. **TRADE**: Execution of a trade (buy/sell)
   - `symbol`, `price`, `position_delta`, `cash_delta` (negative for buy, positive for sell)
   - `cost_basis_delta`, `realized_pnl` (if closing position)
   - `commission`, `spread_cost`, `slippage_cost`

2. **COST**: Standalone cost (e.g., subscription fees)
   - `cash_delta` (negative), `notes`

3. **DIVIDEND**: Dividend payment
   - `symbol`, `cash_delta` (positive), `notes`

4. **CASH_ADJUSTMENT**: Manual cash adjustment
   - `cash_delta`, `notes`

5. **POSITION_ADJUSTMENT**: Manual position adjustment (e.g., corporate actions)
   - `symbol`, `position_delta`, `notes`

## Storage Layout

```
output/ledger_<run_id>/
  ├── ledger_events.parquet          # All ledger events (sorted by timestamp, then event_id)
  ├── ledger_summary.json            # Summary: final cash, positions, total PnL
  ├── positions_snapshots.parquet    # Position snapshots at key timestamps
  ├── cash_history.parquet           # Cash balance over time
  └── reconciliation_report.json     # Reconciliation results (if broker snapshots available)
```

## Position Engine Semantik

### Average Cost Basis

- **Buy**: `new_avg_cost = (old_cost_basis + fill_price * fill_qty) / (old_qty + fill_qty)`
- **Sell**: Average cost basis unchanged (FIFO or average cost method)
- **Partial Fill**: Update average cost basis incrementally per fill

### Realized PnL

- Calculated on position reduction (sell):
  - `realized_pnl = (sell_price - avg_cost_basis) * sell_qty`
- For short positions: `realized_pnl = (avg_cost_basis - sell_price) * sell_qty`

### Unrealized PnL

- `unrealized_pnl = (current_price - avg_cost_basis) * current_qty`
- For short positions: `unrealized_pnl = (avg_cost_basis - current_price) * current_qty`

### Position States

- **Long**: `qty > 0`
- **Short**: `qty < 0`
- **Flat**: `qty == 0`

### Partial Fills

- Each partial fill creates a separate ledger event
- Average cost basis updated incrementally
- Cash and positions updated incrementally
- Event IDs must be stable (deterministic) for same fill sequence

## Reconciliation Semantik

### Reconciliation Sources

1. **Broker Snapshots**: Cash balance, positions, PnL from broker (paper or live)
2. **Ledger State**: Computed from ledger events

### Reconciliation Checks

1. **Cash Reconciliation**:
   - `ledger_cash == broker_cash` (within tolerance, e.g., 0.01)
   - Fail-fast if mismatch exceeds tolerance

2. **Position Reconciliation**:
   - For each symbol: `ledger_qty == broker_qty` (exact match)
   - Fail-fast if any symbol mismatch

3. **PnL Reconciliation**:
   - `ledger_realized_pnl == broker_realized_pnl` (within tolerance)
   - `ledger_unrealized_pnl == broker_unrealized_pnl` (within tolerance, if prices available)

### Tolerances

- **Cash**: Default 0.01 (1 cent) or configurable
- **Positions**: Exact match (no tolerance for integer quantities)
- **PnL**: Default 0.01 or configurable

### Reconciliation Report

```json
{
  "reconciliation_timestamp": "2024-01-15T10:00:00Z",
  "cash_match": true,
  "cash_ledger": 10000.50,
  "cash_broker": 10000.50,
  "cash_diff": 0.0,
  "positions_match": true,
  "position_diffs": [],
  "pnl_match": true,
  "realized_pnl_ledger": 150.25,
  "realized_pnl_broker": 150.25,
  "unrealized_pnl_ledger": 50.10,
  "unrealized_pnl_broker": 50.10,
  "overall_match": true,
  "warnings": []
}
```

## Determinism Rules

1. **UTC Policy**: All timestamps in UTC (tz-aware)
2. **Stable Event IDs**: Deterministic generation (e.g., hash of timestamp + symbol + order_id + fill_seq)
3. **Sorting**: Events sorted by `timestamp` (ascending), then `event_id` (ascending), using `mergesort`
4. **Atomic Writes**: Parquet files written atomically (write to temp, then rename, Windows-safe)
5. **JSON Output**: `sort_keys=True`, `indent=2`, NaN/Inf -> None
6. **Canonical Float Formatting**: Floats in event_id generation use `Decimal` quantization to avoid rounding issues
   - **Method**: `Decimal(str(value)).quantize(Decimal(10)**-precision, rounding=ROUND_HALF_UP)`
   - **Precision**: 10 decimal places (configurable)
   - **Purpose**: Ensures `0.1 + 0.2` and `0.3` produce same canonical string representation
   - **Implementation**: `_canonical_float_str()` in `ledger.py`
7. **String Normalization**: All strings trimmed (symbols, event_type)
8. **Append Mode Deduplication**: `drop_duplicates(subset=["event_id"], keep="first")` before sorting

## Integration Points

### run_daily / Paper Path

- Ledger events generated during order execution (fill_model_pipeline)
- Position engine updates during portfolio simulation
- Reconciliation run after each trading day (if broker snapshots available)

### Manifest Fields

Add to `run_manifest.json`:
- `ledger_path`: str (path to ledger directory)
- `ledger_final_cash`: float
- `ledger_final_positions`: dict[str, float] (symbol -> qty)
- `ledger_total_realized_pnl`: float
- `reconciliation_match`: bool | None (None if no broker snapshot)

### Orchestrator Integration

- After portfolio simulation: generate ledger events from fills
- After each day: run reconciliation (if enabled)
- Write ledger artifacts to `output/ledger_<run_id>/`

## Test Plan (L5)

### L5.1: Partial Fills Accounting

**Test**: `tests/test_ledger_partial_fills.py`
- Order: BUY 100 shares @ $50
- Partial fills: 30 @ $50.10, 40 @ $50.05, 30 @ $50.00
- Verify:
  - 3 ledger events (one per fill)
  - Average cost basis: (30*50.10 + 40*50.05 + 30*50.00) / 100 = 50.05
  - Cash: -5005.0 (sum of fills + costs)
  - Position: +100

### L5.2: Cash Invariants

**Test**: `tests/test_ledger_cash_invariants.py`
- Start cash: 10000.0
- Series of trades with costs
- Verify:
  - Final cash = start_cash + sum(cash_delta) for all events
  - Cash never negative (if short-selling not allowed)
  - Cash matches equity curve (if available)

### L5.3: Realized PnL Calculation

**Test**: `tests/test_ledger_realized_pnl.py`
- Buy 100 @ $50 (avg_cost = $50)
- Sell 50 @ $55
- Verify:
  - Realized PnL = (55 - 50) * 50 = 250.0
  - Remaining position: 50 @ $50 (avg_cost unchanged)
  - Cash increased by: 50 * 55 - costs = 2750 - costs

### L5.4: Reconciliation Match

**Test**: `tests/test_ledger_reconciliation.py`
- Generate ledger events
- Create broker snapshot (matching state)
- Run reconciliation
- Verify: `overall_match == True`

### L5.5: Reconciliation Mismatch Detection

**Test**: `tests/test_ledger_reconciliation_mismatch.py`
- Generate ledger events
- Create broker snapshot (cash differs by 0.02)
- Run reconciliation with tolerance 0.01
- Verify: `overall_match == False`, `cash_match == False`, error message clear

### L5.6: Replay-Day Determinism

**Test**: `tests/test_ledger_replay_determinism.py`
- Run same backtest twice
- Compare ledger events (event_id, timestamp, cash_delta, position_delta)
- Verify: identical events (deterministic)

### L5.7: Short Position Accounting

**Test**: `tests/test_ledger_short_positions.py`
- Short 100 @ $50 (avg_cost = $50)
- Cover 50 @ $45
- Verify:
  - Realized PnL = (50 - 45) * 50 = 250.0 (profit on short)
  - Remaining position: -50 @ $50 (avg_cost unchanged)
  - Cash increased by: 50 * 50 - 50 * 45 - costs = 250 - costs

## Implementation Tasks

### Task 1: Ledger Event Generation

**Files**:
- `src/assembled_core/accounting/ledger.py` (NEW)
- `src/assembled_core/accounting/position_engine.py` (NEW)

**Tasks**:
1. Implement `LedgerEvent` dataclass
2. Implement `generate_ledger_events_from_fills(fills_df, orders_df, costs_df) -> list[LedgerEvent]`
3. Implement `PositionEngine` class:
   - `update_position(symbol, qty, price) -> (new_avg_cost, realized_pnl)`
   - `get_position(symbol) -> (qty, avg_cost_basis, unrealized_pnl)`
4. Tests: `tests/test_ledger_event_generation.py`

### Task 2: Ledger Storage

**Files**:
- `src/assembled_core/accounting/ledger_store.py` (NEW)

**Tasks**:
1. Implement `store_ledger_events(events, output_dir, run_id) -> Path`
2. Implement `load_ledger_events(ledger_path) -> pd.DataFrame`
3. Implement `generate_ledger_summary(events) -> dict`
4. Atomic writes (temp file + rename)
5. Tests: `tests/test_ledger_store.py`

### Task 3: Reconciliation Engine

**Files**:
- `src/assembled_core/accounting/reconciliation.py` (NEW)

**Tasks**:
1. Implement `reconcile_ledger_vs_broker(ledger_state, broker_snapshot, tolerances) -> ReconciliationReport`
2. Implement cash, position, PnL checks
3. Fail-fast on mismatches
4. Tests: `tests/test_ledger_reconciliation.py`

### Task 4: Integration with Execution Pipeline

**Files**:
- `src/assembled_core/pipeline/portfolio.py` (UPDATE)
- `src/assembled_core/execution/fill_model_pipeline.py` (UPDATE)

**Tasks**:
1. Generate ledger events during fill processing
2. Update position engine during portfolio simulation
3. Write ledger artifacts after simulation
4. Integration tests: `tests/test_ledger_integration.py`

### Task 5: Manifest Integration

**Files**:
- `src/assembled_core/pipeline/orchestrator.py` (UPDATE)

**Tasks**:
1. Add ledger fields to manifest
2. Write ledger path to manifest
3. Run reconciliation if broker snapshots available
4. Tests: Verify manifest fields present

## DoD (Definition of Done)

- [ ] Ledger events generated for all trades/costs
- [ ] Position engine tracks average cost basis correctly
- [ ] Realized/unrealized PnL calculated correctly
- [ ] Partial fills handled correctly (incremental updates)
- [ ] Reconciliation detects mismatches (fail-fast)
- [ ] All tests (L5.1-L5.7) green
- [ ] Determinism: same inputs -> same ledger events
- [ ] Documentation: ASCII-only, complete
- [ ] Integration: ledger artifacts in manifest

## Findings from Codebase Analysis

### Existing Components

1. **Fill Model Pipeline** (`src/assembled_core/execution/fill_model_pipeline.py`):
   - Already handles partial fills (`fill_qty`, `fill_price`, `status`, `remaining_qty`)
   - Fill schema contract defined in `fill_model.py`
   - Costs computed based on `fill_qty` (not original `qty`)

2. **Transaction Costs** (`src/assembled_core/execution/transaction_costs.py`):
   - Already computes `commission_cash`, `spread_cash`, `slippage_cash`, `total_cost_cash`
   - Costs are based on `fill_qty * fill_price` (notional for filled portion)
   - Rejected fills have costs = 0.0

3. **Portfolio Simulation** (`src/assembled_core/pipeline/portfolio.py`):
   - `simulate_with_costs()` computes `cash_delta` per order
   - Cash tracking via cumulative sum: `equity[1:] = start_capital + cumsum(cash_deltas)`
   - Positions tracked as dict[str, float] (symbol -> qty) in `backtest.py`
   - **Gap**: No average cost basis tracking, no realized PnL calculation

4. **Paper Track** (`src/assembled_core/paper/paper_track.py`):
   - Already tracks `cash`, `equity`, `positions` in `PaperTrackState`
   - Computes position value via `_compute_position_value()`
   - **Gap**: No average cost basis, no realized/unrealized PnL, no ledger events

5. **Backtest Engine** (`src/assembled_core/qa/backtest_engine.py`):
   - `_update_positions_vectorized()` updates positions (qty only, no cost basis)
   - Returns `BacktestResult` with `trades` DataFrame (includes fill_qty, fill_price, costs)
   - **Gap**: No ledger event generation, no position engine with cost basis

6. **Manifest/Orchestrator** (`src/assembled_core/pipeline/orchestrator.py`):
   - Manifest writes JSON with `indent=2` (missing `sort_keys=True` - **FIXED**)
   - No ledger fields in manifest yet
   - JSON serialization uses standard `json.dump()` (no NaN handling)

### Integration Points

1. **Fill Processing**:
   - `apply_fill_model_pipeline()` in `portfolio.py` (line 80)
   - Output: orders DataFrame with `fill_qty`, `fill_price`, `status`, `remaining_qty`
   - **Hook**: Generate ledger events after fill processing, before cost calculation

2. **Cost Calculation**:
   - `add_cost_columns_to_trades()` in `portfolio.py` (line 106)
   - Output: orders DataFrame with `commission_cash`, `spread_cash`, `slippage_cash`, `total_cost_cash`
   - **Hook**: Use cost columns in ledger events

3. **Cash Delta Calculation**:
   - `cash_delta` computed in `portfolio.py` (line 147)
   - Formula: BUY: `-(qty * price * (1 + s + im) + total_cost_cash)`, SELL: `+(qty * price * (1 - s - im) - total_cost_cash)`
   - **Note**: Uses `qty` not `fill_qty` - may need adjustment for partial fills

4. **Position Updates**:
   - `_update_positions_vectorized()` in `backtest_engine.py` (line 139)
   - Updates positions dict (symbol -> qty)
   - **Gap**: No cost basis tracking, no realized PnL

5. **Paper Track State**:
   - `PaperTrackState` in `paper_track.py` (line 157)
   - Tracks `cash`, `equity`, `positions` (dict[str, float])
   - **Gap**: No cost basis, no PnL breakdown

### Missing Components

1. **Position Engine**: No average cost basis tracking
2. **Ledger Events**: No immutable event log
3. **Realized PnL**: No calculation on position reduction
4. **Reconciliation**: No broker snapshot comparison
5. **Ledger Storage**: No parquet-based event storage

### Implementation Tasks Summary

**L1: Ledger Event Generation**
- Create `src/assembled_core/accounting/ledger.py` with `LedgerEvent` dataclass
- Create `src/assembled_core/accounting/position_engine.py` with `PositionEngine` class
- Generate events from fills (one event per fill, including partial fills)
- Track average cost basis per symbol
- Calculate realized PnL on position reduction

**L2: Ledger Storage**
- Create `src/assembled_core/accounting/ledger_store.py`
- Store events in parquet (sorted by timestamp, event_id)
- Generate summary JSON (final cash, positions, total PnL)
- Atomic writes (temp file + rename)

**L3: Reconciliation Engine**
- Create `src/assembled_core/accounting/reconciliation.py`
- Compare ledger state vs broker snapshot
- Fail-fast on mismatches (cash, positions, PnL)
- Generate reconciliation report JSON

**L4: Integration**
- Update `portfolio.py`: Generate ledger events after fills
- Update `backtest_engine.py`: Use PositionEngine for position updates
- Update `orchestrator.py`: Write ledger path to manifest
- Update `paper_track.py`: Generate ledger events for paper trading

**L5: Tests**
- `tests/test_ledger_partial_fills.py`: Partial fill accounting
- `tests/test_ledger_cash_invariants.py`: Cash balance invariants
- `tests/test_ledger_realized_pnl.py`: Realized PnL calculation
- `tests/test_ledger_reconciliation.py`: Reconciliation match/mismatch
- `tests/test_ledger_replay_determinism.py`: Determinism checks
- `tests/test_ledger_short_positions.py`: Short position accounting

## Notes

- **No External Dependencies**: Use only pandas, numpy, standard library
- **Backward Compatible**: Existing backtest/portfolio code continues to work
- **Performance**: Ledger generation should not significantly slow down backtests
- **Auditability**: Every cash/position change must be traceable to a ledger event
- **Determinism**: Event IDs must be stable (same fill -> same event_id)

## Running Checks (Windows-compatible)

### Quick Start

Use the unified check script:
```bash
# All checks (py_compile → ruff → pytest)
python scripts/dev/run_checks.py

# Only ledger/reconciliation tests
python scripts/dev/run_checks.py --skip-compile --skip-ruff --pytest-args tests/test_ledger*.py tests/test_reconciliation*.py tests/test_orchestrator_manifest_writer.py -v

# Broker snapshot importer tests
python scripts/dev/run_checks.py --skip-compile --skip-ruff --pytest-args tests/test_broker_snapshot_importer*.py tests/test_broker_snapshot_policy_require_with_import.py -v
```

This runs all checks in sequence: `py_compile` → `ruff` → `pytest`.

### Manual Checks

If you prefer to run checks manually, use `python -m` to avoid PATH issues:

**With venv (recommended):**
```bash
# Windows
.venv\Scripts\python.exe -m py_compile src/assembled_core/accounting/ tests/test_ledger*.py
.venv\Scripts\python.exe -m ruff check src/assembled_core/accounting/ tests/test_ledger*.py
.venv\Scripts\python.exe -m pytest tests/test_ledger*.py -v

# Linux/Mac
.venv/bin/python -m py_compile src/assembled_core/accounting/ tests/test_ledger*.py
.venv/bin/python -m ruff check src/assembled_core/accounting/ tests/test_ledger*.py
.venv/bin/python -m pytest tests/test_ledger*.py -v
```

**Without venv:**
```bash
python -m py_compile src/assembled_core/accounting/ tests/test_ledger*.py
python -m ruff check src/assembled_core/accounting/ tests/test_ledger*.py
python -m pytest tests/test_ledger*.py -v
```

### Regression Commands (Windows)

**After changes to broker snapshot importer, daily run, or reconciliation reports:**

```bash
# Syntax check (Windows: no wildcards, list files explicitly)
py -3 -m py_compile \
  src/assembled_core/accounting/broker_snapshot_importer.py \
  src/assembled_core/accounting/ledger_integration.py \
  src/assembled_core/accounting/reconciliation_report.py \
  src/assembled_core/pipeline/orchestrator.py \
  src/assembled_core/qa/backtest_engine.py \
  src/assembled_core/qa/candidate_gate.py \
  scripts/run_eod_pipeline.py \
  scripts/run_backtest_strategy.py \
  scripts/run_daily.py \
  tests/test_broker_snapshot_importer_smoke.py \
  tests/test_broker_snapshot_policy_require_with_import.py \
  tests/test_cli_broker_snapshot_import.py \
  tests/test_daily_broker_snapshot_controls_smoke.py \
  tests/test_reconcile_report_includes_broker_meta.py \
  tests/test_run_daily_argparse_smoke.py \
  tests/test_candidate_gate_reconciliation.py

# Linting (ruff supports wildcards)
py -3 -m ruff check \
  src/assembled_core/accounting/broker_snapshot_importer.py \
  src/assembled_core/accounting/ledger_integration.py \
  src/assembled_core/accounting/reconciliation_report.py \
  src/assembled_core/pipeline/orchestrator.py \
  src/assembled_core/qa/backtest_engine.py \
  src/assembled_core/qa/candidate_gate.py \
  scripts/run_eod_pipeline.py \
  scripts/run_backtest_strategy.py \
  scripts/run_daily.py \
  tests/test_broker_snapshot_importer*.py \
  tests/test_broker_snapshot_policy_require_with_import.py \
  tests/test_cli_broker_snapshot_import.py \
  tests/test_daily_broker_snapshot_controls_smoke.py \
  tests/test_reconcile_report_includes_broker_meta.py \
  tests/test_run_daily_argparse_smoke.py \
  tests/test_candidate_gate_reconciliation.py

# Tests (pytest supports wildcards)
py -3 -m pytest -q \
  tests/test_broker_snapshot_importer*.py \
  tests/test_broker_snapshot_policy_require_with_import.py \
  tests/test_cli_broker_snapshot_import.py \
  tests/test_daily_broker_snapshot_controls_smoke.py \
  tests/test_reconcile_report_includes_broker_meta.py \
  tests/test_run_daily_argparse_smoke.py \
  tests/test_candidate_gate_reconciliation.py
```

**Alternative (using run_checks.py, supports wildcards):**
```bash
python scripts/dev/run_checks.py \
  --pytest-args "tests/test_broker_snapshot_importer*.py tests/test_broker_snapshot_policy_require_with_import.py tests/test_cli_broker_snapshot_import.py tests/test_daily_broker_snapshot_controls_smoke.py tests/test_reconcile_report_includes_broker_meta.py tests/test_run_daily_argparse_smoke.py tests/test_candidate_gate_reconciliation.py -v"
```

### How to Verify

**Verify broker snapshot import:**
```bash
# Check that snapshot was imported
ls output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json

# Verify snapshot content
cat output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json
```

**Verify reconciliation report includes broker_meta:**
```bash
# Check JSON report
cat output/reconcile_report_<run_id>/reconcile_<YYYY-MM-DD>.json | grep -A 5 broker_meta

# Check Markdown report
cat output/reconcile_report_<run_id>/reconcile_<YYYY-MM-DD>.md | grep -A 5 "Broker Source"
```

**Verify candidate gate behavior:**
```bash
# Check manifest for reconciliation_ok
cat output/run_manifest_<freq>.json | grep reconciliation_ok

# Test candidate gate (Python)
python -c "from src.assembled_core.qa.candidate_gate import read_reconciliation_ok_from_manifest, check_candidate_allowed; from pathlib import Path; ok = read_reconciliation_ok_from_manifest(Path('output/run_manifest_1d.json')); allowed, msg = check_candidate_allowed(robustness_ok=True, reconciliation_ok=ok); print(f'Candidate allowed: {allowed}, Message: {msg}')"
```

### Important Notes

- **Always use `python -m`**: This avoids PATH issues on Windows where `pytest`/`ruff` might not be in PATH
- **Venv detection**: The `run_checks.py` script automatically detects `.venv` and uses it if available
- **Exit codes**: The script returns `0` on success, `!=0` on failure (deterministic for CI/CD)

See `docs/PROJECT_STRUCTURE.md` for more details on the check strategy.

## Broker Snapshot Policy

### Overview

The ledger system supports three policies for broker snapshot usage:

1. **`ignore`**: Never use stored broker snapshots, always use paper broker view
2. **`prefer`** (default): Use stored broker snapshot if available, otherwise fall back to paper view
3. **`require`**: Broker snapshot must exist, raise `ValueError` if missing

### Configuration

**In `build_ledger_from_trades()`:**
```python
build_ledger_from_trades(
    ...,
    broker_snapshot_policy="prefer",  # or "ignore", "require"
    write_paper_broker_snapshot=False,  # Write paper view as snapshot
    broker_snapshot_run_id=None,  # Optional: different run_id for snapshot
)
```

**In `run_portfolio_backtest()`:**
```python
run_portfolio_backtest(
    ...,
    broker_snapshot_policy="prefer",
    write_broker_snapshot=False,
)
```

**CLI (backtest script):**
```bash
# Use default policy (prefer)
python scripts/run_backtest_strategy.py --strategy ema

# Ignore snapshots (always use paper view)
python scripts/run_backtest_strategy.py --strategy ema --broker-snapshot-policy ignore

# Require snapshot (fail if missing)
python scripts/run_backtest_strategy.py --strategy ema --broker-snapshot-policy require

# Write paper view as snapshot (for replay/reproducibility)
python scripts/run_backtest_strategy.py --strategy ema --write-broker-snapshot
```

### Snapshot Layout and Naming

Broker snapshots are stored in:
```
output/broker_snapshot_<run_id>/
  snapshot_<YYYY-MM-DD>.json
  positions_<YYYY-MM-DD>.parquet  (optional, only if positions exist)
```

**JSON Schema:**
```json
{
  "as_of_date": "2025-01-15T10:00:00+00:00",
  "cash": 10000.0,
  "positions": [
    {"symbol": "AAPL", "qty": 100.0},
    {"symbol": "MSFT", "qty": 50.0}
  ]
}
```

**Determinism:**
- Positions are normalized (trimmed symbols, filtered tiny residuals, sorted by symbol)
- JSON is written deterministically (`sort_keys=True`, `indent=2`, trailing newline)
- Atomic writes (Windows-safe via temp file → rename)

### Decision Logic

The reconciliation process follows this decision tree:

```
if broker_snapshot_policy == "ignore":
    use paper view
else:
    try load snapshot
    if found:
        use snapshot
    else:
        if broker_snapshot_policy == "require":
            raise ValueError("Broker snapshot required but not found ...")
        use paper view (fallback)

if write_paper_broker_snapshot:
    store snapshot (cash + normalized positions)
```

### Use Cases

**`prefer` (default):**
- Production runs: Use real broker snapshots when available, fall back to paper view for testing
- Backward compatible: Existing behavior (snapshot if present, paper view otherwise)

**`ignore`:**
- Testing: Always use paper view, ignore any stored snapshots
- Debugging: Isolate paper view behavior without snapshot interference

**`require`:**
- Production: Enforce that real broker snapshots are present
- Ops: Fail fast if snapshot is missing (prevents silent fallback to paper view)

**`write_paper_broker_snapshot=True`:**
- Replay/Reproducibility: Save paper view as snapshot for deterministic replay
- Testing: Create snapshots from backtests for integration testing

### Warning

**Snapshot precedence can change reconciliation results:**
- If a stored snapshot differs from paper view, reconciliation will use the snapshot
- This can cause reconciliation to pass/fail differently than expected
- Always check `broker_snapshot_path` in manifest to see which source was used
- Use `broker_snapshot_policy="ignore"` if you want to force paper view reconciliation

**Note on `--write-broker-snapshot`:**
- This flag writes the PAPER broker view as a snapshot for replay/reproducibility
- It does NOT magically turn paper view into a real broker snapshot
- Use this for debugging, deterministic replay, or creating test snapshots
- In production, real broker snapshots should come from your broker API or Ops process

## Broker Snapshot Import

### Overview

External broker snapshots (from broker API, CSV, JSON) can be imported into the
standardized snapshot format using the import functions.

### Import Functions

**Module:** `src/assembled_core/accounting/broker_snapshot_importer.py`

**Main Function:** `import_broker_snapshot()`

```python
from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot

result = import_broker_snapshot(
    snapshot_path="broker_positions_2025-01-15.json",
    run_id="ops_snapshot_20250115",
    snapshot_date="2025-01-15",
    output_dir=Path("output"),
    qty_tol=1e-8,
    store_parquet=True,
    cash_override=None,  # Optional: override cash from file
)
```

**Supported Formats:**
- **JSON** (required): Schema `{"cash": float, "positions": [{"symbol": str, "qty": float}]}`
- **CSV** (optional): Columns `symbol`, `qty` (optional: `cash` column)

**Normalization:**
- All imported snapshots are normalized via `normalize_broker_snapshot()`
- Symbols are trimmed, tiny residuals filtered (abs(qty) <= qty_tol), positions sorted deterministically
- Output is stored in standard layout: `output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json`

**Return Value:**
```python
{
    "broker_snapshot_path": "broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json",  # relative
    "broker_positions_path": "broker_snapshot_<run_id>/positions_<YYYY-MM-DD>.parquet",  # or None
    "cash": 10000.0,  # normalized cash value
}
```

**Use Cases:**
- Import snapshots from broker API (after pulling via Ops script)
- Import historical snapshots from CSV exports
- Create test snapshots for replay/reproducibility

**Error Handling:**
- `FileNotFoundError`: Snapshot file does not exist
- `ValueError`: Invalid file format, missing required fields, parse errors (with clear context)

### CLI Import Integration

**EOD Pipeline:**
```bash
# Import snapshot from file before reconciliation
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-file broker_positions_2025-01-15.json \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-policy require

# Import from CSV with cash override
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-file broker_positions.csv \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-run-id ops_snapshot_20250115
```

**Backtest:**
```bash
# Import snapshot before backtest reconciliation
python scripts/run_backtest_strategy.py \
  --strategy ema \
  --broker-snapshot-file broker_positions.json \
  --broker-snapshot-policy require
```

**Workflow:**
1. Snapshot file is imported before Ledger/Reconciliation step
2. Imported snapshot is stored in standard layout: `output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json`
3. Reconciliation uses imported snapshot (if policy allows)
4. If `policy=require` and import fails, pipeline fails fast with clear error

**Note:**
- `--broker-snapshot-file` + `policy=require` ensures snapshot is available for reconciliation
- Snapshot date defaults to last trade date (or today) if not provided
- Import happens before ledger step, so snapshot is available when reconciliation runs

## Candidate Gate Integration

### Overview

When accounting/reconciliation is active, reconciliation failures act as a gate that blocks candidate status (analogous to the robustness gate). This ensures that strategies with reconciliation failures cannot be marked as "candidate" for production use.

### Gate Logic

The candidate gate checks both robustness and reconciliation status:

**Reconciliation Gate Policy:**
- `reconciliation_ok=False`: Candidate is **blocked** (reconciliation failed)
- `reconciliation_ok=True`: Candidate is **allowed** (reconciliation passed)
- `reconciliation_ok=None`: Candidate is **allowed** with warning (backward compatible, reconciliation not run)

**Combined Gate Behavior:**
- Both robustness and reconciliation must pass for candidate status
- If either gate fails, candidate is blocked
- If reconciliation is not run (`None`), candidate is allowed (backward compatible) but a warning is logged

### Implementation

**Helper Function:**
```python
from src.assembled_core.qa.candidate_gate import read_reconciliation_ok_from_manifest

reconciliation_ok = read_reconciliation_ok_from_manifest(manifest_path)
```

**Gate Check:**
```python
from src.assembled_core.qa.candidate_gate import check_candidate_allowed

candidate_allowed, message = check_candidate_allowed(
    robustness_ok=True,
    reconciliation_ok=reconciliation_ok,
    reconcile_report_path="reconcile_report_run_id/reconcile_2024-01-15.json",
)
```

### Manifest Integration

The `reconciliation_ok` field is automatically written to the run manifest:
- `reconciliation_ok=True`: Reconciliation passed (or not performed)
- `reconciliation_ok=False`: Reconciliation failed
- `reconciliation_ok=None`: Reconciliation was not run (backward compatible)

### Use Cases

**Production:**
- Reconciliation must pass for candidate status
- Failures block candidate marking

**Development/Testing:**
- If reconciliation is not run (`None`), candidate is allowed (backward compatible)
- Warning is logged to indicate missing reconciliation

**Backward Compatibility:**
- Existing runs without reconciliation continue to work
- No breaking changes for runs that don't use accounting/reconciliation

## Broker Snapshot in Daily Runs

### Overview

The daily run (`scripts/run_daily.py`) supports broker snapshot controls and import, consistent with EOD and Backtest pipelines. This enables reconciliation workflows in daily order generation runs.

### CLI Examples

**Import snapshot before daily run:**
```bash
# Import external snapshot and require it for reconciliation
python scripts/run_daily.py \
  --date 2025-01-15 \
  --broker-snapshot-file broker_positions_2025-01-15.json \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-policy require
```

**Prefer snapshot with fallback:**
```bash
# Import snapshot, but allow fallback to paper view if import fails
python scripts/run_daily.py \
  --date 2025-01-15 \
  --broker-snapshot-file broker_positions_2025-01-15.json \
  --broker-snapshot-policy prefer
```

**Write paper snapshot:**
```bash
# Write paper view as snapshot for replay/reproducibility
python scripts/run_daily.py \
  --date 2025-01-15 \
  --write-broker-snapshot \
  --broker-snapshot-policy prefer
```

### Integration Points

- Broker snapshot import happens before ledger/reconciliation step (if `--broker-snapshot-file` provided)
- Snapshot date defaults to target date if not provided
- Snapshot run_id defaults to "daily_snapshot" if not provided
- Policy behavior: `require` fails fast if snapshot missing, `prefer` falls back to paper view

### Note

Currently, `run_daily.py` generates orders but does not perform portfolio simulation or ledger/reconciliation by default. The broker snapshot controls are prepared for future integration when ledger/reconciliation is added to the daily run workflow.

## Reconcile Report Broker Source Fields

### Overview

Every reconciliation report includes `broker_meta` fields that clearly indicate which broker source was used for reconciliation. This enables operators to verify the reconciliation source and troubleshoot discrepancies.

### Broker Meta Fields

**JSON Report (`reconcile_<YYYY-MM-DD>.json`):**
```json
{
  "reconciliation_date": "2025-01-15T00:00:00+00:00",
  "run_id": "test_run_001",
  "ok": true,
  "broker_meta": {
    "broker_view_source": "stored_snapshot",
    "broker_snapshot_run_id": "snapshot_run_001",
    "broker_snapshot_date": "2025-01-15T00:00:00+00:00",
    "broker_snapshot_path": "broker_snapshot_snapshot_run_001/snapshot_2025-01-15.json"
  },
  ...
}
```

**Markdown Report (`reconcile_<YYYY-MM-DD>.md`):**
```markdown
## Broker Source

- **Source:** stored_snapshot
- **Snapshot Run ID:** snapshot_run_001
- **Snapshot Date:** 2025-01-15T00:00:00+00:00
- **Snapshot Path:** broker_snapshot_snapshot_run_001/snapshot_2025-01-15.json
```

### Source Values

- `"stored_snapshot"`: Broker snapshot was loaded from store (imported or previously written)
- `"paper_view"`: Paper broker view was used (fallback when snapshot not available)

### Use Cases

**Verification:**
- Check `broker_view_source` to confirm which source was used
- Verify `broker_snapshot_path` points to expected snapshot file
- Use `broker_snapshot_run_id` to trace snapshot namespace

**Troubleshooting:**
- If reconciliation fails, check `broker_view_source` to see if paper view or snapshot was used
- Compare `broker_snapshot_path` with expected snapshot location
- Verify `broker_snapshot_date` matches reconciliation date

## Candidate Gate Behavior (reconciliation_ok)

### Overview

When accounting/reconciliation is active, reconciliation failures act as a gate that blocks candidate status. This ensures that strategies with reconciliation failures cannot be marked as "candidate" for production use.

### Gate Logic

The candidate gate checks both robustness and reconciliation status:

**Reconciliation Gate Policy:**
- `reconciliation_ok=False`: Candidate is **blocked** (reconciliation failed)
- `reconciliation_ok=True`: Candidate is **allowed** (reconciliation passed)
- `reconciliation_ok=None`: Candidate is **allowed** with warning (backward compatible, reconciliation not run)

**Combined Gate Behavior:**
- Both robustness and reconciliation must pass for candidate status
- If either gate fails, candidate is blocked
- If reconciliation is not run (`None`), candidate is allowed (backward compatible) but a warning is logged

### Implementation

**Helper Function:**
```python
from src.assembled_core.qa.candidate_gate import read_reconciliation_ok_from_manifest

reconciliation_ok = read_reconciliation_ok_from_manifest(manifest_path)
```

**Gate Check:**
```python
from src.assembled_core.qa.candidate_gate import check_candidate_allowed

candidate_allowed, message = check_candidate_allowed(
    robustness_ok=True,
    reconciliation_ok=reconciliation_ok,
    reconcile_report_path="reconcile_report_run_id/reconcile_2024-01-15.json",
)
```

### Manifest Integration

The `reconciliation_ok` field is automatically written to the run manifest:
- `reconciliation_ok=True`: Reconciliation passed (or not performed)
- `reconciliation_ok=False`: Reconciliation failed
- `reconciliation_ok=None`: Reconciliation was not run (backward compatible)

### Use Cases

**Production:**
- Reconciliation must pass for candidate status
- Failures block candidate marking

**Development/Testing:**
- If reconciliation is not run (`None`), candidate is allowed (backward compatible)
- Warning is logged to indicate missing reconciliation

**Backward Compatibility:**
- Existing runs without reconciliation continue to work
- No breaking changes for runs that don't use accounting/reconciliation

## Broker Snapshot Import Workflow

### Overview

The broker snapshot import workflow allows external broker snapshots (from broker API, CSV exports, or manual files) to be imported into the standardized snapshot format and used for reconciliation. This enables production workflows where broker snapshots are pulled from external sources and integrated into the ledger system.

### Workflow Steps

1. **Import Snapshot**: External file (JSON/CSV) is imported via `import_broker_snapshot()` or CLI flags
2. **Normalization**: Snapshot is normalized (trim symbols, filter tiny residuals, sort deterministically)
3. **Storage**: Normalized snapshot is stored in standard layout: `output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json`
4. **Reconciliation**: Ledger reconciliation uses imported snapshot (based on policy)

### Examples

#### Example 1: Import + Require Policy

**Scenario**: Import a broker snapshot and enforce that it is used for reconciliation (fail-fast if missing).

**EOD Pipeline:**
```bash
# Import snapshot and require it for reconciliation
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-file broker_positions_2025-01-15.json \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-policy require
```

**Backtest:**
```bash
# Import snapshot before backtest reconciliation
python scripts/run_backtest_strategy.py \
  --strategy ema \
  --broker-snapshot-file broker_positions.json \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-policy require
```

**Behavior:**
- Snapshot is imported before ledger/reconciliation step
- If import fails, pipeline fails fast with clear error
- Reconciliation uses imported snapshot (policy=require ensures it exists)
- If snapshot is missing after import, ValueError is raised

#### Example 2: Prefer Policy + Fallback

**Scenario**: Use imported snapshot if available, otherwise fall back to paper view.

**EOD Pipeline:**
```bash
# Import snapshot, but allow fallback to paper view if import fails
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-file broker_positions_2025-01-15.json \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-policy prefer
```

**Backtest:**
```bash
# Import snapshot with fallback
python scripts/run_backtest_strategy.py \
  --strategy ema \
  --broker-snapshot-file broker_positions.json \
  --broker-snapshot-policy prefer
```

**Behavior:**
- Snapshot is imported if file exists
- If import fails (file missing, parse error), pipeline continues with paper view
- Reconciliation uses imported snapshot if available, otherwise paper view
- No error if snapshot is missing (graceful fallback)

#### Example 3: Write Paper Snapshot

**Scenario**: Import external snapshot AND write paper view as snapshot for replay/reproducibility.

**EOD Pipeline:**
```bash
# Import external snapshot, write paper view snapshot
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-file broker_positions_2025-01-15.json \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-policy prefer \
  --write-broker-snapshot
```

**Backtest:**
```bash
# Import external snapshot, write paper view snapshot
python scripts/run_backtest_strategy.py \
  --strategy ema \
  --broker-snapshot-file broker_positions.json \
  --broker-snapshot-policy prefer \
  --write-broker-snapshot
```

**Behavior:**
- External snapshot is imported (if file exists)
- Paper view is written as snapshot: `output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json`
- Reconciliation uses imported snapshot if available, otherwise paper view
- Paper snapshot can be used for future replay runs

#### Example 4: CSV Import with Cash Override

**Scenario**: Import positions from CSV (no cash column) and override cash value.

**EOD Pipeline:**
```bash
# Import CSV snapshot with cash override
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-file broker_positions.csv \
  --broker-snapshot-date 2025-01-15 \
  --broker-snapshot-run-id ops_snapshot_20250115
```

**CSV Format:**
```csv
symbol,qty
AAPL,100.0
MSFT,50.0
```

**Behavior:**
- CSV is imported (cash must be provided via cash_override in Python API, or defaults to 0.0)
- Positions are normalized (trim, filter, sort)
- Snapshot is stored in standard layout
- Reconciliation uses imported snapshot

**Note**: CLI does not support cash override directly; use Python API for cash override:
```python
from src.assembled_core.accounting.broker_snapshot_importer import import_broker_snapshot

result = import_broker_snapshot(
    snapshot_path="broker_positions.csv",
    run_id="ops_snapshot_20250115",
    snapshot_date="2025-01-15",
    output_dir=Path("output"),
    cash_override=10000.0,  # Override cash
)
```

### Import Workflow Summary

| Step | Action | Result |
|------|--------|--------|
| 1 | Import external snapshot (JSON/CSV) | Normalized snapshot stored in `output/broker_snapshot_<run_id>/` |
| 2 | Build ledger from trades | Ledger events, positions, cash computed |
| 3 | Load broker snapshot (based on policy) | Snapshot loaded from store (or paper view fallback) |
| 4 | Reconcile ledger vs broker | Reconciliation report generated |
| 5 | Write reports | Reconciliation report, accounting report written |

### Error Handling

**Import Failures:**
- `FileNotFoundError`: Snapshot file does not exist (only fails if `policy=require`)
- `ValueError`: Invalid file format, missing required fields, parse errors (clear context provided)

**Policy Failures:**
- `policy=require` + missing snapshot: `ValueError` raised (fail-fast)
- `policy=prefer` + missing snapshot: Fallback to paper view (no error)
- `policy=ignore`: Snapshot import is skipped, always use paper view

### Best Practices

1. **Production**: Use `policy=require` + `--broker-snapshot-file` to ensure snapshots are present
2. **Testing**: Use `policy=prefer` + `--write-broker-snapshot` to create test snapshots
3. **Replay**: Import snapshots before replay runs to ensure deterministic reconciliation
4. **Ops**: Store imported snapshots in separate `run_id` namespace (e.g., `ops_snapshot_<date>`)
