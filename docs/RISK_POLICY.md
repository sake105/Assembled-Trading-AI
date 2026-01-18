# Risk Policy: Pre-Trade Exposure Limits (Sprint 8 R1-R5)

## Overview

This document defines the risk policy and design for pre-trade exposure limits in Sprint 8.
Risk controls are applied **before fills** to block or reduce orders that would violate exposure limits.

**Scope (Sprint 8):**
- max_weight_per_symbol: Maximum position weight per symbol
- turnover_cap: Maximum daily turnover limit
- drawdown_de_risking: Reduce exposure when drawdown exceeds threshold

**Out of Scope (Sprint 9):**
- SecurityMaster integration (sector/region/FX limits)
- Complex heuristics (deterministic reduce only)

---

## Integration Point

### Location
- **File:** `src/assembled_core/pipeline/trading_cycle.py`
- **Function:** `_apply_risk_controls_default()` (line 512)
- **Step:** Step 6 (line 866-881) - Risk Controls
- **Hook:** `risk_controls` (line 868-872)

### Execution Order
1. Load prices (Step 1)
2. Build features (Step 2)
3. Generate signals (Step 3)
4. Size positions (Step 4)
5. Generate orders (Step 5)
6. **Apply risk controls (Step 6)** ← Integration point
7. Apply fill model (after risk controls, in backtest/portfolio)
8. Write outputs (Step 7)

### Available Data in TradingContext

At the risk controls step, the following data is available:

- **`current_positions`**: pd.DataFrame
  - Columns: `symbol`, `qty`
  - Current portfolio positions (before orders)
  - May be None (empty portfolio)

- **`orders`**: pd.DataFrame
  - Columns: `timestamp`, `symbol`, `side`, `qty`, `price`
  - Proposed orders (from Step 5)

- **`prices`**: pd.DataFrame
  - Columns: `timestamp`, `symbol`, `close`, ... (OHLCV)
  - Latest prices for exposure calculation
  - Can be filtered to `as_of` timestamp

- **`capital`**: float
  - Total capital for position sizing
  - Used for weight calculation

- **`equity`**: float (optional, not directly in TradingContext)
  - Current portfolio equity (cash + mark-to-market positions)
  - Can be computed from `current_positions` + `prices` if needed

---

## Post-Trade Exposure Definition

### Target Holdings Calculation

Post-trade exposure is computed as: **current positions + proposed orders → target holdings**

For each symbol:
```
target_qty = current_qty + order_delta

where:
  order_delta = sum(qty) for BUY orders - sum(qty) for SELL orders
  (if multiple orders for same symbol, aggregate by side)
```

### Exposure Metrics

From target holdings, compute:

1. **Gross Exposure:**
   ```
   gross_exposure = sum(abs(target_qty * price))
   ```

2. **Net Exposure:**
   ```
   net_exposure = sum(target_qty * price)
   ```

3. **Per-Symbol Weight:**
   ```
   weight_symbol = (target_qty * price) / equity
   ```
   where `equity = cash + sum(current_qty * price)` (or use `capital` if equity not available)

4. **Notional:**
   ```
   notional_symbol = abs(target_qty * price)
   ```

5. **Turnover:**
   ```
   turnover = sum(abs(order_delta * price)) / equity_prev
   ```
   where `order_delta` is the change in position from orders, and `equity_prev` is previous equity

---

## Rule Execution Order

Risk checks are applied in the following order (in `run_pre_trade_checks()`):

1. **QA Status Check** (Step 1): If QA gates block trading, all orders are blocked.
2. **Max Notional Per Symbol** (Step 3): Order-level check (legacy, not post-trade exposure based).
3. **Max Weight Per Symbol** (Step 4): Post-trade exposure based. Reduces orders if `abs(weight) > threshold`.
4. **Sector Exposure Limit** (Step 8): Group-level check. Reduces orders in sector if `gross_weight > limit`.
5. **Region Exposure Limit** (Step 8): Group-level check. Reduces orders in region if `gross_weight > limit`.
6. **FX Exposure Limit** (Step 8): Group-level check. Reduces orders in non-base currency if `gross_weight > limit` (FX rates not yet implemented, fail-fast if non-base currency present).
7. **Turnover Cap** (Step 5): Portfolio-wide check. Scales all orders proportionally if `turnover > cap`.
8. **Drawdown De-Risking** (Step 6): Portfolio-wide check. Scales all orders if `drawdown >= threshold`.
9. **Max Gross Exposure** (Step 9): Portfolio-wide check (legacy, order-level).

**Rationale:**
- **Drawdown first (Step 6):** Most conservative, applies to all orders regardless of symbol.
- **Max Weight second (Step 4):** Symbol-specific, requires post-trade exposure calculation.
- **Turnover third (Step 5):** Portfolio-wide, but less conservative than drawdown.

**Note:** The order matters because each check modifies the orders DataFrame, which is then passed to the next check.

---

## Limit Classes (Sprint 8)

### R1: max_weight_per_symbol

**Definition:** Maximum position weight (as fraction of equity) per symbol.

**Limit:** `max_weight_per_symbol: float` (e.g., 0.10 = 10%)

**Execution Order:** Step 4 (after QA, before turnover)

**Check:**
```
For each symbol in target holdings:
  if abs(weight_symbol) > max_weight_per_symbol:
    action = reduce
```

**Semantics:**
- **pass:** `abs(weight_symbol) <= max_weight_per_symbol` for all symbols
- **reduce:** Reduce order qty so that `abs(weight_symbol) <= max_weight_per_symbol`
- **SELL orders that reduce exposure are allowed** even if still over limit

**Reduce Logic (deterministic):**
```
max_target_notional = max_weight_per_symbol * equity
max_target_qty = max_target_notional / price
order_delta_needed = max_target_qty - current_qty
scale_factor = order_delta_needed / total_order_delta
new_order_qty = order_qty * scale_factor
```

**Example:**
```
Input:
  current_positions: AAPL=50 shares @ 150 = 7500 (75% of 10000 equity)
  orders: BUY 50 AAPL @ 150
  max_weight_per_symbol: 0.10 (10%)

Post-trade exposure:
  target_qty = 50 + 50 = 100 shares
  target_weight = 100 * 150 / 10000 = 1.5 (150%) > 0.10

Action:
  max_target_qty = 0.10 * 10000 / 150 = 6.67
  order_delta_needed = 6.67 - 50 = -43.33
  scale_factor = -43.33 / 50 = -0.867 (invalid, clamp to 0)
  Actually: reduce order so target_qty = 6.67
  new_order_qty = 1.67 (reduced from 50)

Output:
  filtered_orders: BUY 1.67 AAPL @ 150
  reduced_orders: [{"reason": "RISK_REDUCE_MAX_WEIGHT_PER_SYMBOL", ...}]
```

### R2: turnover_cap

**Definition:** Maximum daily turnover (as fraction of equity).

**Limit:** `turnover_cap: float` (e.g., 0.50 = 50% of equity)

**Execution Order:** Step 5 (after max_weight, before drawdown)

**Check:**
```
turnover = sum(abs(order_notional)) / equity
where order_notional = abs(qty * price)

if turnover > turnover_cap:
  action = reduce (scale all orders proportionally)
```

**Semantics:**
- **pass:** `turnover <= turnover_cap`
- **reduce:** Scale down all orders proportionally so that `turnover <= turnover_cap`

**Reduce Logic (deterministic):**
```
scale_factor = turnover_cap / turnover
for each order:
  new_order_qty = int(order_qty * scale_factor)  # Round to integer
  if abs(new_order_qty) < 1e-10:
    drop order
```

**Example:**
```
Input:
  orders: BUY 100 AAPL @ 150, BUY 50 MSFT @ 200
  equity: 10000
  turnover_cap: 0.5 (50%)

Turnover calculation:
  order_notionals = [100*150, 50*200] = [15000, 10000]
  total_turnover = 25000 / 10000 = 2.5 (250%) > 0.5

Action:
  scale_factor = 0.5 / 2.5 = 0.2
  AAPL: 100 * 0.2 = 20 (rounded to 20)
  MSFT: 50 * 0.2 = 10 (rounded to 10)

Output:
  filtered_orders: BUY 20 AAPL @ 150, BUY 10 MSFT @ 200
  reduced_orders: [
    {"reason": "RISK_REDUCE_TURNOVER_CAP", "symbol": "AAPL", ...},
    {"reason": "RISK_REDUCE_TURNOVER_CAP", "symbol": "MSFT", ...}
  ]
```

### R3: drawdown_de_risking

**Definition:** Reduce exposure when drawdown exceeds threshold.

**Limit:** `drawdown_threshold: float` (e.g., 0.20 = 20%), `de_risk_scale: float` (e.g., 0.25 = reduce to 25%)

**Execution Order:** Step 6 (after turnover, before max_gross_exposure)

**Check:**
```
drawdown = 1 - current_equity / peak_equity

if drawdown >= drawdown_threshold:
  action = reduce (scale all orders by de_risk_scale)
```

**Semantics:**
- **pass:** `drawdown < drawdown_threshold`
- **reduce:** Scale down all orders by `de_risk_scale` (deterministic)
- **block:** If `de_risk_scale = 0.0`, all orders are blocked

**Reduce Logic (deterministic):**
```
if drawdown >= drawdown_threshold:
  for each order:
    new_order_qty = int(order_qty * de_risk_scale)  # Round to integer
    if abs(new_order_qty) < 1e-10:
      drop order
```

**Example:**
```
Input:
  orders: BUY 100 AAPL @ 150
  current_equity: 7000
  peak_equity: 10000
  drawdown_threshold: 0.2 (20%)
  de_risk_scale: 0.25 (25%)

Drawdown calculation:
  drawdown = 1 - 7000/10000 = 0.3 (30%) >= 0.2

Action:
  AAPL: 100 * 0.25 = 25 (rounded to 25)

Output:
  filtered_orders: BUY 25 AAPL @ 150
  reduced_orders: [{"reason": "RISK_DERISK_DRAWDOWN", ...}]

Example (full block):
  de_risk_scale: 0.0
  Output:
    filtered_orders: (empty)
    reduced_orders: [{"reason": "RISK_DERISK_DRAWDOWN", "new_qty": 0.0, ...}]
```

**Note:** Requires `current_equity` and `peak_equity`. If not available, check is skipped with explicit reason in `summary`.

---

## Semantics: pass / block / reduce

### pass
Order passes risk controls unchanged. No action needed.

### block
Order is completely blocked (removed from orders DataFrame).

**When to block:**
- Reduction would violate other constraints (e.g., reducing one order would violate max_weight on another)
- Kill switch engaged (existing)
- Critical violation that cannot be reduced deterministically

### reduce
Order quantity is reduced deterministically (no heuristics).

**Requirements:**
- Reduction must be deterministic (same inputs → same outputs)
- Reduction must not violate other limits
- Reduction must be proportional or absolute (no complex logic)

**Examples:**
- Scale all orders by factor (turnover_cap, drawdown_de_risking)
- Cap per-symbol weight (max_weight_per_symbol)

---

## Implementation Design

### Module Structure

```
src/assembled_core/execution/pre_trade_exposure_limits.py
  - compute_post_trade_exposure()
  - check_max_weight_per_symbol()
  - check_turnover_cap()
  - check_drawdown_de_risking()
  - apply_exposure_limits()
```

### Integration with Existing Risk Controls

The new exposure limits will be integrated into the existing risk controls flow:

1. **Existing:** `filter_orders_with_risk_controls()` in `risk_controls.py`
2. **Existing:** `run_pre_trade_checks()` in `pre_trade_checks.py`
3. **New:** `apply_exposure_limits()` in `pre_trade_exposure_limits.py`

**Integration point:** Extend `PreTradeConfig` to include exposure limit parameters, and call `apply_exposure_limits()` within `run_pre_trade_checks()`.

### Data Flow

```
TradingContext (Step 6)
  ├─ current_positions (symbol, qty)
  ├─ orders (timestamp, symbol, side, qty, price)
  ├─ prices (timestamp, symbol, close, ...)
  └─ capital (float)
      │
      v
filter_orders_with_risk_controls()
  ├─ Prepare current_positions_df from current_positions
  ├─ Prepare prices_latest_df from prices (latest price per symbol)
  ├─ Compute equity from capital (or use provided equity)
  └─ Call run_pre_trade_checks()
      │
      v
run_pre_trade_checks()
  ├─ Step 1: QA status check
  ├─ Step 3: Max notional per symbol (legacy)
  ├─ Step 4: Max weight per symbol
  │   ├─ compute_target_positions(current_positions, orders)
  │   ├─ compute_exposures(target_positions, prices_latest, equity)
  │   └─ Reduce orders if abs(weight) > threshold
  ├─ Step 5: Turnover cap
  │   ├─ Calculate turnover = sum(abs(order_notional)) / equity
  │   └─ Scale all orders if turnover > cap
  ├─ Step 6: Drawdown de-risking
  │   ├─ Calculate drawdown = 1 - current_equity / peak_equity
  │   └─ Scale all orders if drawdown >= threshold
  └─ Step 7: Max gross exposure (legacy)
      │
      v
PreTradeCheckResult
  ├─ filtered_orders: Orders after all checks
  ├─ reduced_orders: List of reduction reasons
  └─ summary: Check status and metrics
```

---

## Configuration

### PreTradeConfig

`PreTradeConfig` (in `pre_trade_checks.py`) includes all risk limit parameters:

```python
@dataclass
class PreTradeConfig:
    max_notional_per_symbol: float | None = None
    max_weight_per_symbol: float | None = None  # e.g., 0.10 = 10%
    turnover_cap: float | None = None  # e.g., 0.50 = 50% of equity
    drawdown_threshold: float | None = None  # e.g., 0.20 = 20%
    de_risk_scale: float = 0.0  # e.g., 0.25 = reduce to 25% when drawdown exceeded
    max_gross_exposure: float | None = None
    max_sector_exposure: dict[str, float] | None = None
    max_region_exposure: dict[str, float] | None = None
```

**Usage:**
```python
config = PreTradeConfig(
    max_weight_per_symbol=0.10,  # 10% max weight per symbol
    turnover_cap=0.5,  # 50% turnover cap
    drawdown_threshold=0.2,  # 20% drawdown threshold
    de_risk_scale=0.25,  # Reduce to 25% when drawdown exceeded
)

result, filtered_orders = run_pre_trade_checks(
    orders,
    current_positions=current_positions,
    prices_latest=prices_latest,
    equity=equity,
    current_equity=current_equity,
    peak_equity=peak_equity,
    config=config,
)
```

---

## Testing Strategy

### Unit Tests

- `tests/test_risk_max_weight_per_symbol.py`: max_weight_per_symbol limit
- `tests/test_risk_turnover_cap.py`: turnover_cap limit
- `tests/test_risk_drawdown_derisk.py`: drawdown de-risking

### Integration Tests

- `tests/test_sprint8_risk_integration.py`: Full risk pipeline integration
  - `test_max_weight_reduces_buy_order()`: max_weight reduces BUY based on post-trade exposure
  - `test_turnover_reduces_portfolio_wide()`: turnover reduces orders portfolio-wide
  - `test_drawdown_derisk_scales_or_blocks()`: drawdown de-risking scales or blocks
  - `test_rule_order_drawdown_then_max_weight_then_turnover()`: Rule execution order
  - `test_deterministic_behavior_same_inputs()`: Deterministic behavior verification

---

## Acceptance Criteria (DoD)

- [x] `docs/RISK_POLICY.md` exists and documents all limit classes
- [x] Integration point clearly identified (trading_cycle.py, Step 6)
- [x] Post-trade exposure calculation implemented (via `exposure_engine`)
- [x] max_weight_per_symbol limit implemented (pass/reduce)
- [x] turnover_cap limit implemented (pass/reduce)
- [x] drawdown_de_risking implemented (pass/reduce/block)
- [x] All limits are deterministic (no heuristics)
- [x] Tests pass (unit + integration)
- [x] No QA layer imports in execution layer (layering maintained)

---

## Future Work (Sprint 9)

- SecurityMaster integration (sector/region/FX limits)
- More complex exposure aggregation (e.g., sector-level limits)
- Dynamic limits based on volatility/regime
- Position-level limits (not just symbol-level)
