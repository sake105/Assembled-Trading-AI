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
- **JSON** (required): Schema `{"cash": float | str, "positions": [{"symbol": str, "qty": float | str, ...}]}`  
  - Unknown keys in each `positions` entry are ignored
  - `cash` may be string (`"1000.00"`) or numeric
- **CSV** (optional): Columns `symbol`, `qty` (optional: `cash` column)

**Accepted Formats / Normalization Rules:**

- **Symbols (`symbol`):**
  - Leading/trailing whitespace is stripped
  - Internal whitespace is collapsed (e.g. `"  AAPL   US  "` → `"AAPL US"`)
  - Empty symbols after normalization are rejected with a clear error

- **Quantities (`qty`):**
  - Robust parsing via internal `_parse_float_like()` helper:
    - Numeric values (`1`, `1.5`) are accepted
    - Strings with whitespace (`" 2.5 "`) are accepted
    - Thousands separators (`"1,000"`, `"12,345.67"`) are accepted and parsed as `1000.0` / `12345.67`
    - Parentheses notation (`"(5)"`) is treated as negative (`-5.0`)
    - Empty strings / missing values are rejected for required `qty` fields (with file path and row/index in the error message)

- **Cash (`cash`):**
  - JSON: `cash` may be float or string (e.g. `"1000.00"`)
  - CSV: If a `cash` column exists, the first non-empty value is parsed using the same float-like rules
  - CLI / API `cash_override` (if provided) always has precedence over file content

- **Duplicate Symbols:**
  - `normalize_broker_snapshot()` aggregates duplicate symbols by summing `qty` per symbol
  - Aggregation happens *before* tiny residual filtering and deterministic sorting
  - After aggregation, any symbol with `abs(qty) <= qty_tol` is removed

**Normalization:**
- All imported snapshots are normalized via `normalize_broker_snapshot()`
- Symbols are trimmed and normalized (whitespace rules above)
- Duplicate symbols are aggregated deterministically
- Tiny residuals are filtered (abs(qty) <= qty_tol)
- Positions are sorted deterministically by symbol (stable `mergesort`)
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

The candidate gate combines both robustness and reconciliation gates deterministically, ensuring that both quality checks must pass for candidate status.

### Combined Gate Logic

The candidate gate checks both robustness and reconciliation status and combines them deterministically:

**Individual Gate Policies:**

**Robustness Gate:**
- `robustness_ok=True`: Robustness pack passed
- `robustness_ok=False`: Robustness pack failed → **blocks candidate**
- `robustness_ok=None`: Robustness pack not run → **allows with warning** (backward compatible)

**Reconciliation Gate:**
- `reconciliation_ok=True`: Reconciliation passed
- `reconciliation_ok=False`: Reconciliation failed → **blocks candidate**
- `reconciliation_ok=None`: Reconciliation not run → **allows with warning** (backward compatible)

**Combined Gate Behavior (Deterministic):**
- **If any gate is `False`**: Candidate is **blocked** (fail-fast)
- **If both gates are `True`**: Candidate is **allowed**
- **If one or both gates are `None`**: Candidate is **allowed with warning** (backward compatible)

**Message Format:**
- Messages include report links when `robustness_pack_path` or `reconcile_report_path` are provided
- Format: `"Robustness pack passed - Reconciliation failed (report: path/to/report.json) - candidate NOT allowed"`
- Links enable quick troubleshooting by pointing directly to relevant reports

### Implementation

**Helper Functions:**
```python
from src.assembled_core.qa.candidate_gate import (
    read_robustness_ok_from_manifest,
    read_reconciliation_ok_from_manifest,
)

robustness_ok = read_robustness_ok_from_manifest(manifest_path)
reconciliation_ok = read_reconciliation_ok_from_manifest(manifest_path)
```

**Gate Check:**
```python
from src.assembled_core.qa.candidate_gate import check_candidate_allowed

candidate_allowed, message = check_candidate_allowed(
    robustness_ok=robustness_ok,
    robustness_pack_path="robustness_pack_run_id",  # Optional: included in message if set
    reconciliation_ok=reconciliation_ok,
    reconcile_report_path="reconcile_report_run_id/reconcile_2024-01-15.json",  # Optional: included in message if set
)

if not candidate_allowed:
    logger.error(f"Candidate blocked: {message}")
```

**Example Messages:**

**Both gates passed:**
```
Robustness pack passed - Reconciliation passed - candidate allowed
```

**Reconciliation failed (with report link):**
```
Robustness pack passed - Reconciliation failed (report: reconcile_report_run1/reconcile_2025-01-15.json) - candidate NOT allowed
```

**Both gates failed (with report links):**
```
Robustness pack failed (report: robustness_pack_run1) | Reconciliation failed (report: reconcile_report_run1/reconcile_2025-01-15.json) - candidate NOT allowed
```

**Reconciliation not run (backward compatible):**
```
Robustness pack passed - Reconciliation not run (backward compatible) - candidate allowed
```

### Manifest Integration

Both `robustness_ok` and `reconciliation_ok` fields are automatically written to the run manifest:
- `robustness_ok=True`: Robustness pack passed
- `robustness_ok=False`: Robustness pack failed
- `robustness_ok=None`: Robustness pack was not run
- `reconciliation_ok=True`: Reconciliation passed
- `reconciliation_ok=False`: Reconciliation failed
- `reconciliation_ok=None`: Reconciliation was not run (backward compatible)

### Use Cases

**Production:**
- Both robustness and reconciliation must pass for candidate status
- Failures in either gate block candidate marking
- Report links in messages enable quick troubleshooting

**Development/Backtesting:**
- Gates can be `None` (not run) for backward compatibility
- Warnings are logged but candidate status is allowed
- Allows gradual adoption of new quality checks

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

### Daily Manifest

`run_daily.py` writes an optional daily manifest file (`manifest_daily_<run_id>.json`) that provides operational evidence and links to all generated outputs. The manifest structure is aligned with the orchestrator manifest for consistency.

**Manifest Location:**
- Path: `output/manifest_daily_<run_id>.json`
- Run ID format: `daily_YYYYMMDD` (e.g., `daily_20250115`)

**Manifest Schema (aligned with orchestrator):**

All fields are present in the manifest (even if `None`), ensuring consistent schema for downstream tools.

**Core Fields:**
- `run_id`: Run identifier (string)
- `target_date`: Target trading date (YYYY-MM-DD string)
- `safe_orders_path`: Relative path to generated SAFE orders CSV (POSIX slashes)

**Broker Snapshot Fields:**
- `broker_snapshot_policy`: Broker snapshot policy used (`"ignore"`, `"prefer"`, or `"require"`)
- `broker_snapshot_date`: Snapshot date (YYYY-MM-DD string, or `None`)
- `broker_snapshot_file`: Snapshot file path (relative or basename, or `None` if not imported)
- `broker_snapshot_import_ok`: Whether broker snapshot import succeeded (`True`, `False`, or `None` if no file provided)
- `broker_snapshot_path`: Relative path to snapshot directory (POSIX slashes, or `None` if not exists)
- `broker_snapshot_run_id`: Snapshot run ID (string, or `None`)
- `write_paper_broker_snapshot`: Whether paper snapshot was written (boolean, aligned with orchestrator field name)

**Ledger/Accounting Fields (optional, `None` if not active):**
- `ledger_pack_path`: Relative path to ledger pack (POSIX slashes, or `None`)
- `reconcile_report_path`: Relative path to reconciliation report (POSIX slashes, or `None`)
- `reconciliation_ok`: Reconciliation status (boolean, or `None`)
- `evidence_index_path`: Relative path to evidence index JSON (POSIX, or `None`)
- `evidence_pack_path`: Relative path to evidence pack ZIP (POSIX, or `None`)
- `evidence_pack_manifest_path`: Relative path to pack manifest JSON (POSIX, or `None`)
- `write_evidence_pack`: Whether evidence pack was requested (boolean)

**Broker Snapshot Import Fields:**

When `--broker-snapshot-file` is provided:
- `broker_snapshot_file`: Set to relative path or basename of the imported file
- `broker_snapshot_import_ok`: Set to `True` if import succeeded, `False` if import failed (and policy was not `"require"`), or `None` if no file was provided

**Determinism:**
- JSON keys are sorted (`sort_keys=True`, `indent=2`)
- Paths are relative to output directory and use POSIX slashes (`/`) for portability
- Trailing newline for byte stability
- Writing the manifest twice with the same inputs produces byte-identical files

**Use Cases:**
- Operational evidence: Links to all outputs from a daily run
- Audit trail: Track which broker snapshot was used and whether import succeeded
- Troubleshooting: Quick access to all relevant files
- Schema consistency: Fixed schema (all keys present) prevents BI/ETL schema drift

**Example Manifest:**

```json
{
  "broker_snapshot_date": "2025-01-15",
  "broker_snapshot_file": "external_snapshot.json",
  "broker_snapshot_import_ok": true,
  "broker_snapshot_path": "broker_snapshot_daily_snapshot/snapshot_2025-01-15.json",
  "broker_snapshot_policy": "prefer",
  "broker_snapshot_run_id": "daily_snapshot",
  "evidence_index_path": null,
  "evidence_pack_manifest_path": null,
  "evidence_pack_path": null,
  "ledger_pack_path": null,
  "reconcile_report_path": null,
  "reconciliation_ok": null,
  "run_id": "daily_20250115",
  "safe_orders_path": "orders_20250115.csv",
  "target_date": "2025-01-15",
  "write_evidence_pack": false,
  "write_paper_broker_snapshot": false
}
```

### Note

Currently, `run_daily.py` generates orders but does not perform portfolio simulation or ledger/reconciliation by default. The broker snapshot controls and manifest structure are prepared for future integration when ledger/reconciliation is added to the daily run workflow. The `--write-evidence-pack` flag is accepted; evidence pack paths (`evidence_index_path`, `evidence_pack_path`, `evidence_pack_manifest_path`) in the daily manifest remain `None` until ledger/accounting are integrated. Use EOD or backtest pipelines for actual evidence pack creation.

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

**CSV Report (`reconcile_<YYYY-MM-DD>.csv`):**
The CSV report uses a **fixed schema** that always includes `broker_meta` fields as additional columns (constant values per row):
- `broker_view_source`: "stored_snapshot" or "paper_view" (or empty string if broker_meta=None)
- `broker_snapshot_run_id`: Run ID for snapshot namespace (or empty string if broker_meta=None)
- `broker_snapshot_date`: Snapshot date (ISO format, or empty string if broker_meta=None)
- `broker_snapshot_path`: Relative path to snapshot file (or empty string if None or broker_meta=None)

**Fixed Schema (no schema drift):**
The broker_meta columns are **always present** in the CSV, even when `broker_meta=None`. This prevents schema drift in BI/ETL tools. Empty values are serialized as empty strings (`""`) for consistency.

Example CSV columns (fixed schema):
```
type,symbol,ledger_value,broker_value,diff,match,broker_view_source,broker_snapshot_run_id,broker_snapshot_date,broker_snapshot_path
cash,,10000.0,10000.0,0.0,True,stored_snapshot,snapshot_run_001,2025-01-15T00:00:00+00:00,broker_snapshot_snapshot_run_001/snapshot_2025-01-15.json
position,AAPL,100.0,100.0,0.0,True,stored_snapshot,snapshot_run_001,2025-01-15T00:00:00+00:00,broker_snapshot_snapshot_run_001/snapshot_2025-01-15.json
```

Example CSV with broker_meta=None (empty strings):
```
type,symbol,ledger_value,broker_value,diff,match,broker_view_source,broker_snapshot_run_id,broker_snapshot_date,broker_snapshot_path
cash,,10000.0,10000.0,0.0,True,,,,,
position,AAPL,100.0,100.0,0.0,True,,,,,
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

### Format-Specific Behavior

**CSV:**
- `broker_meta` fields are included as columns (constant values per row)
- **Fixed schema**: Columns are always present, even when `broker_meta=None` (prevents BI/ETL schema drift)
- All rows have the same `broker_view_source`, `broker_snapshot_run_id`, `broker_snapshot_date`, and `broker_snapshot_path` values
- If `broker_meta` is not provided, columns contain empty strings (`""`) instead of being omitted
- Empty values are serialized as empty strings (not `null` or `NaN`) for consistency
- Useful for Excel/BI tools where CSV is the primary format

**JSON:**
- `broker_meta` is included as a top-level key (dictionary)
- All fields are present (or `null` if not applicable)
- Deterministic key ordering (`sort_keys=True`)

**Markdown:**
- `broker_meta` is included as a "Broker Source" section
- Human-readable format with ASCII-only characters
- Fields are displayed as bullet points

### Use Cases

**Verification:**
- Check `broker_view_source` to confirm which source was used (in CSV, JSON, or Markdown)
- Verify `broker_snapshot_path` points to expected snapshot file
- Use `broker_snapshot_run_id` to trace snapshot namespace

**Troubleshooting:**
- If reconciliation fails, check `broker_view_source` to see if paper view or snapshot was used
- Compare `broker_snapshot_path` with expected snapshot location
- Verify `broker_snapshot_date` matches reconciliation date
- In CSV reports, broker source information is available in every row (useful for Excel filtering/analysis)

## Candidate Gate Behavior (reconciliation_ok)

### Overview

When accounting/reconciliation is active, reconciliation failures act as a gate that blocks candidate status. This ensures that strategies with reconciliation failures cannot be marked as "candidate" for production use.

The candidate gate combines both robustness and reconciliation gates deterministically, ensuring that both quality checks must pass for candidate status.

### Combined Gate Logic

The candidate gate checks both robustness and reconciliation status and combines them deterministically:

**Individual Gate Policies:**

**Robustness Gate:**
- `robustness_ok=True`: Robustness pack passed
- `robustness_ok=False`: Robustness pack failed → **blocks candidate**
- `robustness_ok=None`: Robustness pack not run → **allows with warning** (backward compatible)

**Reconciliation Gate:**
- `reconciliation_ok=True`: Reconciliation passed
- `reconciliation_ok=False`: Reconciliation failed → **blocks candidate**
- `reconciliation_ok=None`: Reconciliation not run → **allows with warning** (backward compatible)

**Combined Gate Behavior (Deterministic):**
- **If any gate is `False`**: Candidate is **blocked** (fail-fast)
- **If both gates are `True`**: Candidate is **allowed**
- **If one or both gates are `None`**: Candidate is **allowed with warning** (backward compatible)

**Message Format with Report Links:**
- Messages include report links when `robustness_pack_path` or `reconcile_report_path` are provided
- Format: `"Robustness pack passed - Reconciliation failed (report: path/to/report.json) - candidate NOT allowed"`
- Links enable quick troubleshooting by pointing directly to relevant reports

### Implementation

**Helper Functions:**
```python
from src.assembled_core.qa.candidate_gate import (
    read_robustness_ok_from_manifest,
    read_reconciliation_ok_from_manifest,
)

robustness_ok = read_robustness_ok_from_manifest(manifest_path)
reconciliation_ok = read_reconciliation_ok_from_manifest(manifest_path)
```

**Gate Check:**
```python
from src.assembled_core.qa.candidate_gate import check_candidate_allowed

candidate_allowed, message = check_candidate_allowed(
    robustness_ok=robustness_ok,
    robustness_pack_path="robustness_pack_run_id",  # Optional: included in message if set
    reconciliation_ok=reconciliation_ok,
    reconcile_report_path="reconcile_report_run_id/reconcile_2024-01-15.json",  # Optional: included in message if set
)

if not candidate_allowed:
    logger.error(f"Candidate blocked: {message}")
```

**Example Messages:**

**Both gates passed:**
```
Robustness pack passed - Reconciliation passed - candidate allowed
```

**Reconciliation failed (with report link):**
```
Robustness pack passed - Reconciliation failed (report: reconcile_report_run1/reconcile_2025-01-15.json) - candidate NOT allowed
```

**Both gates failed (with report links):**
```
Robustness pack failed (report: robustness_pack_run1) | Reconciliation failed (report: reconcile_report_run1/reconcile_2025-01-15.json) - candidate NOT allowed
```

**Reconciliation not run (backward compatible):**
```
Robustness pack passed - Reconciliation not run (backward compatible) - candidate allowed
```

### Manifest Integration

Both `robustness_ok` and `reconciliation_ok` fields are automatically written to the run manifest:
- `robustness_ok=True`: Robustness pack passed
- `robustness_ok=False`: Robustness pack failed
- `robustness_ok=None`: Robustness pack was not run
- `reconciliation_ok=True`: Reconciliation passed
- `reconciliation_ok=False`: Reconciliation failed
- `reconciliation_ok=None`: Reconciliation was not run (backward compatible)

### Use Cases

**Production:**
- Both robustness and reconciliation must pass for candidate status
- Failures in either gate block candidate marking
- Report links in messages enable quick troubleshooting

**Development/Backtesting:**
- Gates can be `None` (not run) for backward compatibility
- Warnings are logged but candidate status is allowed
- Allows gradual adoption of new quality checks

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

**Note**: CLI does not support cash override directly; use Python API or standalone CLI tool for cash override.

**Standalone CLI Tool:**
```bash
# Import JSON snapshot
python scripts/import_broker_snapshot.py \
  --input broker_positions_2025-01-15.json \
  --run-id ops_snapshot_20250115 \
  --as-of-date 2025-01-15 \
  --output-dir output \
  --store-parquet

# Import CSV snapshot with cash override
python scripts/import_broker_snapshot.py \
  --input broker_positions.csv \
  --run-id ops_snapshot_20250115 \
  --as-of-date 2025-01-15 \
  --cash 10000.0 \
  --store-parquet
```

**Golden Path (3-step workflow):**
```bash
# Step 1: Import external broker snapshot
python scripts/import_broker_snapshot.py \
  --input broker_positions_2025-01-15.json \
  --run-id ops_snapshot_20250115 \
  --as-of-date 2025-01-15 \
  --output-dir output \
  --store-parquet

# Step 2: Run EOD pipeline with require policy (uses imported snapshot)
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --broker-snapshot-policy require \
  --broker-snapshot-run-id ops_snapshot_20250115

# Step 3: Verify reconcile report path in manifest
# Check: output/run_manifest_1d.json -> reconcile_report_path
# Or: output/reconcile_report_<run_id>/reconcile_2025-01-15.json
```

**Python API (alternative):**
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

## Evidence Index

### Overview

The evidence index is a central JSON file that links all accounting-related artifacts for a given run and date. It provides a single entry point for Ops/Support workflows, making it easy to locate all relevant files without searching through multiple directories.

### Location

The evidence index is written to:
```
output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json
```

**Example:**
- Run ID: `ledger_eod_1d`
- Date: `2025-01-15`
- Path: `output/evidence_ledger_eod_1d/evidence_2025-01-15.json`

### Purpose

The evidence index serves as:
- **Single entry point**: One file to find all related artifacts
- **Audit trail**: Links snapshot, ledger, reconciliation, accounting reports, and manifest
- **Ops tooling**: Enables automated scripts to discover all files for a run
- **Support workflows**: Quick access to all relevant files for troubleshooting

### Schema

**JSON Structure (deterministic: no `created_utc`, `as_of_date` is date-only YYYY-MM-DD):**
```json
{
  "schema_version": 1,
  "run_id": "ledger_eod_1d",
  "as_of_date": "2025-01-15",
  "paths": {
    "broker_snapshot_path": "broker_snapshot_ops_20250115/snapshot_2025-01-15.json",
    "ledger_pack_path": "ledger_ledger_eod_1d/ledger_events.parquet",
    "reconcile_report_path": "reconcile_report_ledger_eod_1d/reconcile_2025-01-15.json",
    "accounting_report_path": "accounting_report_ledger_eod_1d/accounting_2025-01-15.json",
    "manifest_path": "run_manifest_1d.json"
  },
  "broker_meta": {
    "broker_view_source": "stored_snapshot",
    "broker_snapshot_run_id": "ops_20250115",
    "broker_snapshot_date": "2025-01-15",
    "broker_snapshot_path": "broker_snapshot_ops_20250115/snapshot_2025-01-15.json"
  },
  "reconciliation_ok": true,
  "tool_version": "0.1.0"
}
```

**Fields:**
- `schema_version`: Schema version (currently `1`)
- `run_id`: Run identifier
- `as_of_date`: Report date (date-only `YYYY-MM-DD` for byte-determinism; no time or timezone)
- `paths`: Dictionary of relative POSIX paths to artifacts (may be `None` if not available). All paths are:
  - Relative to `output_dir` (no absolute paths)
  - POSIX-only (`/`), no backslashes
  - Deterministically serialized (same order, sort_keys=True)
- `paths.manifest_path`: Optional path to the orchestrator manifest (`run_manifest_<freq>.json`). This field is:
  - Backfilled by the orchestrator **after** the manifest is written
  - Used by the Evidence Pack exporter to include the manifest directly from the Evidence Index (no fallback needed when present)
- `broker_meta`: Optional broker metadata (same structure as in reconciliation reports)
- `reconciliation_ok`: Optional reconciliation status (`true`, `false`, or `null`)
- `tool_version`: Version of the tool that generated the index

**Backfill behavior (orchestrator):**
- `ledger_integration.py` initially writes the Evidence Index with `paths.manifest_path` set to `null`.
- After the orchestrator writes `run_manifest_<freq>.json`, it performs a best-effort backfill:
  - Loads the Evidence Index JSON
  - Sets `paths.manifest_path` to the relative POSIX path of the manifest
  - Re-writes the Evidence Index deterministically (`sort_keys=True`, `indent=2`, trailing newline, atomic temp→replace)
  - On any error, logs a warning and continues (Evidence Index remains valid without manifest_path).

### How to Read

**Python:**
```python
import json
from pathlib import Path

# Load evidence index
evidence_path = Path("output/evidence_ledger_eod_1d/evidence_2025-01-15.json")
with open(evidence_path) as f:
    evidence = json.load(f)

# Access paths (relative to output_dir)
reconcile_path = evidence["paths"]["reconcile_report_path"]
accounting_path = evidence["paths"]["accounting_report_path"]

# Check reconciliation status
if evidence.get("reconciliation_ok") is False:
    print(f"Reconciliation failed. Report: {reconcile_path}")
```

**CLI (Windows):**
```powershell
# Read evidence index
Get-Content output\evidence_ledger_eod_1d\evidence_2025-01-15.json | ConvertFrom-Json

# Extract reconcile report path
$evidence = Get-Content output\evidence_ledger_eod_1d\evidence_2025-01-15.json | ConvertFrom-Json
$evidence.paths.reconcile_report_path
```

**Integration:**
- The evidence index is automatically written by `build_ledger_from_trades()` after all reports are generated
- The manifest (`run_manifest_<freq>.json`) includes `evidence_index_path` as a top-level field
- Paths in the evidence index are relative to `output_dir` and use POSIX slashes (`/`) for portability

## Schema Versioning

### Overview

All key accounting artifacts (broker snapshots, reconciliation reports, accounting reports, orchestrator manifest, evidence index) include a `schema_version` field to enable long-term stability and upgradeability.

### Current Schema Version

**Schema Version: `1`**

All artifacts currently use `schema_version: 1`. This version indicates:
- Stable JSON structure (top-level keys, field names)
- Stable CSV column names and order
- Deterministic serialization (sort_keys, indent, trailing newline)

### What Schema Version Means

**Purpose:**
- **Stability**: Consumers can rely on a stable schema for a given version
- **Evolution**: Future schema changes can be versioned without breaking existing tools
- **Upgradeability**: Tools can detect schema version and handle migration if needed

**Current Behavior:**
- All artifacts write `schema_version: 1`
- Loaders accept `schema_version` (default to `1` if missing for backward compatibility)
- Invalid schema versions (non-integer or < 1) raise `ValueError` with clear error message

### Artifacts with Schema Version

| Artifact | Location | Field |
|----------|----------|-------|
| Broker Snapshot JSON | `output/broker_snapshot_<run_id>/snapshot_<YYYY-MM-DD>.json` | Top-level `schema_version` |
| Reconciliation Report JSON | `output/reconcile_report_<run_id>/reconcile_<YYYY-MM-DD>.json` | Top-level `schema_version` |
| Reconciliation Report CSV | `output/reconcile_report_<run_id>/reconcile_<YYYY-MM-DD>.csv` | Column `schema_version` (constant value `1`) |
| Accounting Report JSON | `output/accounting_report_<run_id>/accounting_<YYYY-MM-DD>.json` | Top-level `schema_version` |
| Accounting Report CSV | `output/accounting_report_<run_id>/accounting_<YYYY-MM-DD>.csv` | Column `schema_version` (constant value `1`) |
| Orchestrator Manifest | `output/run_manifest_<freq>.json` | Top-level `schema_version` |
| Evidence Index | `output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json` | Top-level `schema_version` |

### How to Upgrade

**Future Schema Evolution:**

When schema changes are needed (e.g., new fields, renamed fields, structural changes):

1. **Increment schema version**: New artifacts write `schema_version: 2`
2. **Backward compatibility**: Loaders accept both `schema_version: 1` and `schema_version: 2`
3. **Migration helpers**: Provide migration functions if needed (e.g., `migrate_schema_v1_to_v2()`)
4. **Documentation**: Update this section with migration notes

**Example (Future):**
```python
# Loader handles multiple schema versions
schema_version = data.get("schema_version", 1)
if schema_version == 1:
    # Handle v1 structure
    cash = data.get("cash")
elif schema_version == 2:
    # Handle v2 structure (e.g., cash renamed to cash_balance)
    cash = data.get("cash_balance")
else:
    raise ValueError(f"Unsupported schema_version: {schema_version}")
```

**Current Status:**
- All artifacts use `schema_version: 1`
- No migration needed
- All loaders default to `1` if `schema_version` is missing (backward compatible)

## Troubleshooting

### Policy=Require Errors

**Error: `ValueError: Broker snapshot required but not found`**

This error occurs when `broker_snapshot_policy="require"` is set, but no snapshot exists for the specified run_id and date.

**Error Message Format:**
```
ValueError: Broker snapshot required but not found for run_id='ops_20250115', date='2025-01-15'.
Expected path: output/broker_snapshot_ops_20250115/snapshot_2025-01-15.json
```

**How to Fix:**

1. **Check expected path**: The error message includes the exact path that was searched
2. **Verify run_id**: Ensure `--broker-snapshot-run-id` matches the snapshot namespace
3. **Verify date**: Ensure the snapshot date matches the run date (format: `YYYY-MM-DD`)
4. **Import snapshot**: If snapshot is missing, import it first:
   ```bash
   python scripts/import_broker_snapshot.py \
     --input broker_positions_2025-01-15.json \
     --run-id ops_20250115 \
     --as-of-date 2025-01-15 \
     --output-dir output
   ```

**Common Causes:**
- Snapshot was imported to a different `run_id` namespace
- Date mismatch (snapshot date vs. run date)
- Snapshot file was deleted or moved
- Import step was skipped in pipeline

### Namespace Mismatch Patterns

**Problem: Snapshot exists but in wrong namespace**

**Symptoms:**
- `policy=require` fails with "not found" error
- Snapshot exists in `output/broker_snapshot_<other_run_id>/`
- Run is looking in `output/broker_snapshot_<expected_run_id>/`

**Root Cause:**
- `broker_snapshot_run_id` parameter doesn't match the namespace where snapshot was stored
- Default `run_id` was used during import, but different `run_id` specified during reconciliation

**How to Fix:**

1. **Check snapshot location:**
   ```bash
   # List all snapshot directories
   dir output\broker_snapshot_*
   
   # Check snapshot files
   dir output\broker_snapshot_*\snapshot_*.json
   ```

2. **Match run_id:**
   ```bash
   # If snapshot is in 'ops_20250115', use same run_id:
   python scripts/run_eod_pipeline.py \
     --freq 1d \
     --broker-snapshot-policy require \
     --broker-snapshot-run-id ops_20250115  # Must match import run_id
   ```

3. **Or re-import to correct namespace:**
   ```bash
   # Re-import snapshot to expected namespace
   python scripts/import_broker_snapshot.py \
     --input broker_positions_2025-01-15.json \
     --run-id ledger_eod_1d  # Use run_id that pipeline expects
     --as-of-date 2025-01-15
   ```

**Prevention:**
- Use consistent `run_id` naming: `ops_snapshot_<date>` for imported snapshots
- Document `run_id` conventions in Ops runbooks
- Use `--broker-snapshot-run-id` explicitly (don't rely on defaults)

### Reconciliation Failures

**Problem: Reconciliation fails (mismatches detected)**

**Symptoms:**
- `reconciliation_ok: false` in manifest
- Reconcile report shows `cash_diff` or `position_diffs`
- Candidate gate blocks strategy

**How to Debug:**

1. **Check reconcile report:**
   ```bash
   # Read reconcile report JSON
   python -c "import json; print(json.dumps(json.load(open('output/reconcile_report_<run_id>/reconcile_2025-01-15.json')), indent=2))"
   ```

2. **Check broker source:**
   - Verify `broker_meta.broker_view_source` in reconcile report
   - If `"paper_view"`: Reconciliation used paper broker view (expected for backtests)
   - If `"stored_snapshot"`: Reconciliation used imported/stored snapshot

3. **Check tolerances:**
   - `cash_tol`: Default `1e-2` (0.01)
   - `qty_tol`: Default `1e-8`
   - Small differences within tolerance are ignored

4. **Check evidence index:**
   ```bash
   # Evidence index links all artifacts
   cat output/evidence_<run_id>/evidence_2025-01-15.json
   ```

**Common Causes:**
- Broker snapshot is stale (positions changed after snapshot)
- Ledger events missing (trades not recorded)
- Cost calculation mismatch (commission/spread/slippage)
- Timing mismatch (snapshot time vs. ledger time)

### Import Failures

**Problem: Import fails with parse error**

**Error: `ValueError: Invalid file format`**

**How to Fix:**

1. **Check file format:**
   - JSON: Must have `{"cash": ..., "positions": [...]}`
   - CSV: Must have `symbol`, `qty` columns

2. **Check data types:**
   - `qty` can be numeric or string (e.g., `"1,000"` is parsed)
   - `cash` can be numeric or string (e.g., `"1000.00"`)

3. **Check required fields:**
   - JSON: `positions` list is required (can be empty `[]`)
   - CSV: `symbol` and `qty` columns are required

**Example Fix:**
```python
# Valid JSON snapshot
{
  "cash": 10000.0,
  "positions": [
    {"symbol": "AAPL", "qty": 100.0}
  ]
}

# Valid CSV snapshot
symbol,qty
AAPL,100.0
MSFT,50.0
```

### Missing Evidence Index

**Problem: Evidence index not found**

**Symptoms:**
- `output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json` doesn't exist
- Manifest has `evidence_index_path: null`

**Common Causes:**
- Ledger integration not run (accounting disabled)
- Evidence index write failed (logged as warning, non-fatal)
- Run ID mismatch

**How to Fix:**
- Ensure `build_ledger_from_trades()` is called
- Check logs for evidence index write warnings
- Verify `run_id` matches expected namespace

### Schema Version Errors

**Problem: `ValueError: Invalid schema_version`**

**Error: `ValueError: Invalid schema_version in broker snapshot JSON: <value>`**

**How to Fix:**
- Ensure `schema_version` is an integer >= 1
- If missing, loaders default to `1` (backward compatible)
- If invalid type, fix the JSON file manually or re-import

**Example:**
```json
{
  "schema_version": 1,  // Must be integer >= 1
  "cash": 10000.0,
  "positions": []
}
```

## Evidence Pack Export

### Overview

The Evidence Pack Export creates a deterministic, portable ZIP archive containing all accounting-related artifacts for a given run and date. This enables audit trails, compliance workflows, and easy artifact sharing without path dependencies.

### Automatic Export (EOD/Backtest)

Evidence packs can be automatically created during EOD or Backtest runs using the `--write-evidence-pack` flag.

**EOD Pipeline:**
```bash
python scripts/run_eod_pipeline.py \
  --freq 1d \
  --write-evidence-pack
```

**Backtest:**
```bash
python scripts/run_backtest_strategy.py \
  --strategy ema \
  --freq 1d \
  --write-evidence-pack
```

**Behavior:**
- Evidence pack is created after evidence index is written
- ZIP and pack manifest are written to `output/evidence_<run_id>/`
- Pack paths are included in run manifest (`evidence_pack_path`, `evidence_pack_manifest_path`)
- Pack creation is best-effort (warnings logged, but run continues on failure)

### Standalone Export (CLI)

Use the standalone CLI tool to export evidence packs from existing evidence indices.

**Basic Export:**
```bash
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --output-dir output
```

**Strict Mode (fail if optional files missing):**
```bash
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --strict
```

**Exclude Optional Files:**
```bash
python scripts/export_evidence_pack.py \
  --run-id ledger_eod_1d \
  --as-of-date 2025-01-15 \
  --no-optional
```

### Output Files

**ZIP Archive:**
- Location: `output/evidence_<run_id>/pack_<YYYY-MM-DD>.zip`
- Contents: All referenced files from evidence index (snapshot, ledger, reconcile, accounting, manifest)
- Internal structure: POSIX paths, sorted entries, fixed timestamps

**Pack Manifest:**
- Location: `output/evidence_<run_id>/pack_manifest_<YYYY-MM-DD>.json`
- Contents: File list with SHA256 checksums, sizes, source types
- Schema: `schema_version: 1`, deterministic JSON

### Golden Path (Import -> Require -> Pack -> Verify -> Archive)

**Single canonical workflow:** **docs/OPS_EVIDENCE_GOLDEN_PATH.md** – When to use, 5-step Windows block (py -3), Evidence Index vs manifest fallback, verify gate, artifact locations. Use that doc for copy-paste; no duplication here.

### Verify Evidence Pack (offline)

A standalone CLI validates an Evidence Pack ZIP **offline** (no repo or output_dir required). Use it to check that a ZIP has a valid manifest, supported schema, correct checksums, and no illegal paths.

**CLI:**
```bash
# Required: path to the ZIP
python scripts/verify_evidence_pack.py --zip path/to/pack_2025-01-15.zip

# Optional: output result as deterministic JSON
python scripts/verify_evidence_pack.py --zip path/to/pack_2025-01-15.zip --json
```

**Exit codes:**
- **0** – Validation passed (manifest present, schema ok, checksums ok, no illegal paths).
- **1** – Validation failed (e.g. missing manifest, bad paths, checksum mismatches) or error (e.g. unsupported schema, file not found).

**Output (without `--json`):**
- Success: one line `OK: ok=True n_files=... missing_manifest=... bad_paths_count=... checksum_mismatches_count=...`
- Failure: one line `FAIL: ok=False ...` with the same keys.
- Error (e.g. unsupported schema): `ERROR: ...` on stderr.

**Output with `--json`:**
- Single JSON object (stable schema): `schema_version`, `zip_path`, `ok`, `error_code`, `missing_manifest`, `n_files`, `bad_paths_count`, `checksum_mismatches_count`, `details`. See **docs/EVIDENCE_PACK.md** (Verify Evidence Pack --json output schema). Deterministic: `sort_keys=True`, `indent=2`, trailing newline.

**Interpretation:**
- `ok=True` – Pack is valid (manifest present, schema version 1, all checksums match, no illegal ZIP paths).
- `ok=False` – Check `missing_manifest`, `bad_paths_count`, `checksum_mismatches_count` to see what failed.
- `ERROR:` – Invalid manifest (e.g. unsupported `schema_version`) or missing ZIP file; fix the pack or path and re-run.

All status and error lines are ASCII-only.

### Determinism

Evidence packs are byte-deterministic when built with same inputs:
- **File Order**: Sorted lexicographically (POSIX, case-sensitive)
- **ZIP Timestamps**: Fixed timestamp (1980-01-01 00:00:00 or custom)
- **Pack Manifest**: Deterministic JSON (`sort_keys=True`, `indent=2`, trailing newline)
- **Checksums**: SHA256 hashes enable content verification

### Use Cases

**Audit Trails:**
- Complete snapshot of all accounting evidence for a run/date
- Portable ZIP can be archived or shared
- Checksums enable integrity verification

**Compliance:**
- Single file contains all related artifacts
- Deterministic packs enable reproducible verification
- Pack manifest provides file inventory

**Ops/Support:**
- Easy export for troubleshooting
- All files in one place (no path dependencies)
- Can be extracted on any OS (POSIX paths)

### Integration

**Manifest Fields:**
- `evidence_pack_path`: Relative path to ZIP file (POSIX)
- `evidence_pack_manifest_path`: Relative path to pack manifest JSON (POSIX)

**Python API:**
```python
from src.assembled_core.accounting.evidence_pack import build_evidence_pack

result = build_evidence_pack(
    output_dir=Path("output"),
    run_id="ledger_eod_1d",
    as_of_date="2025-01-15",
    include_optional=True,
)

# Result contains:
# - pack_path: Relative path to ZIP
# - pack_manifest_path: Relative path to manifest
# - n_files: Number of files included
# - missing_optional: List of missing optional files
# - checksums: Dict mapping paths to SHA256 hashes
```
