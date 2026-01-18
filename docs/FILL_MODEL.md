# Fill Model: Schema + Contract

## Overview

This document defines the contract for trades/fills in the trading system. The fill model supports partial fills, rejected orders, and full fills while maintaining backward compatibility with existing code.

## Purpose

- Support partial fills without breaking strategies/orders
- Extend trade/fill schema while maintaining backward compatibility
- Provide deterministic, validated fill data for backtesting and analysis

## Order Schema

### Required Columns (Minimum)

All order DataFrames MUST have the following columns:

1. **timestamp** (pd.Timestamp, UTC-aware)
   - Order timestamp
   - Must be UTC-aware (tz-aware)
   - No NaNs allowed

2. **symbol** (str)
   - Trading symbol (e.g., "AAPL", "MSFT")
   - No NaNs allowed

3. **side** (str)
   - Order side: "BUY" or "SELL"
   - No NaNs allowed

4. **qty** (float64)
   - Order quantity (requested quantity)
   - Always positive
   - No NaNs allowed

5. **price** (float64)
   - Order price (reference price, e.g., close price at order time)
   - Must be > 0
   - No NaNs allowed

### Optional Columns

6. **order_type** (str) [OPTIONAL]
   - Order type: "market" (default) or "limit"
   - If missing, defaults to "market"
   - No NaNs allowed (if present)

7. **limit_price** (float64) [OPTIONAL]
   - Limit price (required if order_type="limit")
   - Must be > 0 if present
   - For BUY limit: limit_price is maximum price willing to pay
   - For SELL limit: limit_price is minimum price willing to accept
   - No NaNs allowed (if present)

## Fill/Trade Schema

### Required Columns (Minimum)

All fill/trade DataFrames MUST have the following columns:

1. **timestamp** (pd.Timestamp, UTC-aware)
   - Fill execution timestamp
   - Must be UTC-aware (tz-aware)
   - No NaNs allowed

2. **symbol** (str)
   - Trading symbol (e.g., "AAPL", "MSFT")
   - No NaNs allowed

3. **side** (str)
   - Order side: "BUY" or "SELL"
   - No NaNs allowed

4. **qty** (float64)
   - Original order quantity (requested quantity)
   - Always positive
   - No NaNs allowed

5. **price** (float64)
   - Original order price (requested price)
   - Must be > 0
   - No NaNs allowed

6. **fill_qty** (float64) [NEW]
   - Actually filled quantity
   - Must satisfy: 0 <= fill_qty <= qty
   - For full fills: fill_qty == qty
   - For partial fills: 0 < fill_qty < qty
   - For rejected orders: fill_qty == 0
   - No NaNs allowed

7. **fill_price** (float64) [NEW]
   - Actual fill price (may differ from order price due to slippage)
   - Must be > 0 if fill_qty > 0
   - For rejected orders (fill_qty == 0): fill_price == price (or NaN, but prefer price for consistency)
   - No NaNs allowed (use price as fallback)

8. **status** (str) [NEW]
   - Fill status: "filled" | "partial" | "rejected"
   - "filled": fill_qty == qty (full fill)
   - "partial": 0 < fill_qty < qty (partial fill)
   - "rejected": fill_qty == 0 (order rejected)
   - No NaNs allowed

9. **remaining_qty** (float64) [NEW]
   - Remaining unfilled quantity
   - Must satisfy: remaining_qty = qty - fill_qty
   - For full fills: remaining_qty == 0
   - For partial/rejected: remaining_qty > 0
   - No NaNs allowed

10. **commission_cash** (float64)
    - Commission cost in cash
    - Must be >= 0
    - For rejected orders: commission_cash == 0
    - No NaNs allowed

11. **spread_cash** (float64)
    - Spread cost in cash (half-spread)
    - Must be >= 0
    - For rejected orders: spread_cash == 0
    - No NaNs allowed

12. **slippage_cash** (float64)
    - Slippage cost in cash
    - Must be >= 0
    - For rejected orders: slippage_cash == 0
    - No NaNs allowed

13. **total_cost_cash** (float64)
    - Total transaction cost in cash
    - Must satisfy: total_cost_cash = commission_cash + spread_cash + slippage_cash
    - For rejected orders: total_cost_cash == 0
    - No NaNs allowed

## Rules and Constraints

### Fill Quantity Rules

1. **fill_qty <= qty** (always)
2. **fill_qty >= 0** (always)
3. **remaining_qty = qty - fill_qty** (always)

### Status Rules

1. **status == "filled"** if and only if fill_qty == qty
2. **status == "partial"** if and only if 0 < fill_qty < qty
3. **status == "rejected"** if and only if fill_qty == 0

### Rejected Order Rules

If status == "rejected":
- fill_qty == 0
- remaining_qty == qty
- commission_cash == 0
- spread_cash == 0
- slippage_cash == 0
- total_cost_cash == 0
- fill_price == price (preferred) or NaN (but prefer price for consistency)

### Cost Rules

1. **total_cost_cash = commission_cash + spread_cash + slippage_cash** (always)
2. All cost columns must be >= 0
3. For rejected orders: all costs == 0

### Price Rules

1. **price > 0** (always)
2. **fill_price > 0** if fill_qty > 0
3. **fill_price == price** for full fills without slippage
4. **fill_price != price** possible due to slippage

### Deterministic Ordering

All fill/trade DataFrames MUST be sorted by:
1. **timestamp** (ascending)
2. **symbol** (ascending, secondary sort)

This ensures reproducible results and consistent aggregation.

## UTC Policy

All timestamps MUST be UTC-aware:
- Use `pd.to_datetime(..., utc=True)` when creating timestamps
- Validate with `pd.api.types.is_datetime64_any_dtype()` and check `.tz` attribute
- No timezone-naive timestamps allowed

## Backward Compatibility

### Full Fill Default

For backward compatibility, existing code that creates trades without fill_qty/fill_price/status/remaining_qty should be treated as full fills:

- If fill_qty missing: assume fill_qty = qty
- If fill_price missing: assume fill_price = price
- If status missing: assume status = "filled"
- If remaining_qty missing: assume remaining_qty = 0

### Migration Path

1. Existing code continues to work (full fills assumed)
2. New code can explicitly set fill_qty, fill_price, status, remaining_qty
3. Helper functions can convert orders to fills with full fill assumption

## File Locations

Fill/trade DataFrames are used in:
- `output/orders_*.csv` - Order files (may contain fills if already executed)
- `BacktestResult.trades` - Backtest trade list
- TCA reports - Transaction cost analysis
- Portfolio simulation - Equity curve computation

## Session Gate

Orders must be executed only within valid trading sessions. The session gate enforces this:

### Rules

1. **Trading Day Check**: Orders on weekends or holidays are rejected (status="rejected", fill_qty=0).

2. **For freq="1d" (Daily)**:
   - Orders are only accepted at session close (16:00 ET, normalized to UTC).
   - If order timestamp is not at session close (within 1 minute tolerance), order is rejected.
   - Uses `session_close_utc()` from `data/calendar.py`.

3. **For freq="5min" (Intraday)**:
   - Orders are only accepted within trading session (9:30 ET - 16:00 ET, normalized to UTC).
   - If order timestamp is outside session open/close, order is rejected.
   - Uses `exchange_calendars` schedule to determine session boundaries.

### Fallback Behavior

If `exchange_calendars` is not available:
- If `strict=True`: Raises `ImportError` (requires calendar library).
- If `strict=False`: Warns and allows all orders (permissive fallback for testing).

This ensures deterministic behavior: either all orders are validated (strict) or all are allowed (permissive).

### Implementation

The session gate is applied via `apply_session_gate()` in `execution/fill_model.py`:

```python
from src.assembled_core.execution.fill_model import apply_session_gate

fills = apply_session_gate(orders, freq="1d", strict=True)
```

Rejected orders have:
- `status="rejected"`
- `fill_qty=0`
- `remaining_qty=qty`
- All costs set to 0 (commission_cash=0, spread_cash=0, slippage_cash=0, total_cost_cash=0)

## Validation

All fill/trade DataFrames should be validated before use:
- Required columns exist
- No NaNs in key columns (timestamp, symbol, side, qty, price, fill_qty, fill_price, status, remaining_qty)
- Constraints satisfied (fill_qty <= qty, remaining_qty = qty - fill_qty, etc.)
- UTC policy enforced
- Deterministic ordering
- Session gate applied (if using session-aware fills)

See `tests/test_fill_model_schema.py` for validation tests.
See `tests/test_fill_model_session_gate.py` for session gate tests.
See `tests/test_fill_model_limit_orders.py` for limit order tests.
See `tests/test_fill_model_costs_consistency.py` for cost consistency tests.
See `tests/test_fill_model_tca_integration.py` for TCA integration tests.

## Fill Model Pipeline Integration

### Combined Pipeline Order

The fill model pipeline is applied in the following order (via `apply_fill_model_pipeline`):

1. **Session Gate** (first):
   - Reject orders outside trading sessions (weekends, holidays)
   - For `freq="1d"`: only accept orders at session close
   - For `freq="5min"`: only accept orders within session hours

2. **Limit Order Eligibility** (second):
   - For limit orders: check if limit price is reachable
   - BUY limit: reject if `bar_low > limit_price`
   - SELL limit: reject if `bar_high < limit_price`
   - If limit not reachable: status="rejected", fill_qty=0

3. **Partial Fill Model** (third):
   - Apply ADV-based partial fill cap
   - If order exceeds participation cap: status="partial", fill_qty < qty
   - If order within cap: status="filled", fill_qty = qty

### Cost Calculation

After fill model pipeline, costs are computed based on **filled notional** (fill_qty * fill_price):

- **Commission:** Based on filled notional (not original qty)
- **Spread:** Based on filled notional
- **Slippage:** Based on filled notional
- **Rejected fills:** All costs = 0.0

This ensures that partial fills and rejected fills have correct cost attribution.

### TCA Reporting

TCA reports use filled notional (fill_qty * fill_price) for:
- Notional aggregation
- cost_bps calculation (total_cost_cash / filled_notional * 10000)

This ensures accurate cost analysis for partial fills.