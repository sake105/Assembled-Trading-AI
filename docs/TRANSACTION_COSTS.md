# Transaction Costs Policy & Schema

## Overview

This document defines the policy and schema for transaction cost analysis (TCA) in the Assembled Trading AI system.

**Purpose:** Provide deterministic, reproducible transaction cost modeling for backtest strategies.

**Scope:** Commission, Spread, and Slippage costs are computed per trade and aggregated for reporting.

---

## Cost Components

### Commission

**Definition:** Broker fees charged per trade.

**Models:**
- **bps-only:** Commission as basis points of notional (commission_bps / 10000 * notional)
- **fixed-only:** Fixed commission per trade (fixed_per_trade)
- **bps-plus-fixed:** Both bps and fixed commission

**Default:** 0.0 bps (no commission)

**Implementation:** `src/assembled_core/execution/transaction_costs.CommissionModel`

### Spread

**Definition:** Bid/ask spread cost (half-spread approximation).

**Model:** Based on ADV (Average Daily Volume) buckets.

**Formula:** `spread_cash = notional * (spread_bps / 10000) * 0.5`

The 0.5 factor represents half-spread (paid cost when crossing bid/ask spread).

**ADV Proxy:** Rolling mean of (close * volume) over adv_window days.

**Buckets:** ADV thresholds map to spread_bps:
- Example: [(1e6, 5.0), (1e7, 3.0), (1e8, 1.0)] means:
  - ADV < 1e6: 5.0 bps
  - 1e6 <= ADV < 1e7: 3.0 bps
  - 1e7 <= ADV < 1e8: 1.0 bps
  - ADV >= 1e8: fallback_spread_bps

**Fallback:** If volume is missing or ADV cannot be computed, use fallback_spread_bps.

**Default:** 5.0 bps (fallback)

**Note:** For partial fills, spread is computed based on filled notional (fill_qty * fill_price), not original order notional.

**Implementation:** `src/assembled_core/execution/transaction_costs.SpreadModel`

### Slippage

**Definition:** Market impact cost based on volatility and participation rate.

**Model:** Volatility and liquidity-based.

**Formula:** `slippage_bps = clip(k * sigma * sqrt(participation) * 10000, min_bps, max_bps)`

where:
- `sigma` = volatility (rolling std of log returns)
- `participation` = notional / adv_usd (clipped to participation_rate_cap)
- `k` = scaling factor
- `min_bps`, `max_bps` = clamps for stability

**Slippage Cash:** `slippage_cash = notional * (slippage_bps / 10000)`

**Fallback:** If volatility/ADV cannot be computed, use fallback_slippage_bps.

**Default:** k=1.0, min_bps=0.0, max_bps=50.0, fallback_slippage_bps=5.0

**Implementation:** `src/assembled_core/execution/transaction_costs.SlippageModel`

---

## Gross vs Net

**Gross Return:** Return before transaction costs.

**Net Return:** Return after transaction costs (gross return - total_cost_cash).

**Cost Impact:** Difference between gross and net performance metrics.

---

## TCA Report Schema

### Input Schema (Trades DataFrame)

**Required Columns:**
- `timestamp`: UTC timestamps (datetime64[ns, UTC])
- `symbol`: Symbol names (string)
- `qty`: Trade quantities (float64, always positive)
- `price`: Trade prices (float64, always positive)
- `commission_cash`: Commission costs per trade (float64, >= 0.0, never NaN)
- `spread_cash`: Spread costs per trade (float64, >= 0.0, never NaN)
- `slippage_cash`: Slippage costs per trade (float64, >= 0.0, never NaN)
- `total_cost_cash`: Total costs per trade (float64, >= 0.0, never NaN)

**No NaN Policy:** All cost columns must be non-NaN (default to 0.0 if not computable).

### Output Schema (TCA Report DataFrame)

**Columns:**
- `date`: Date (date object, UTC-normalized)
- `symbol`: Symbol name (string)
- `notional`: Sum of notional values (abs(qty) * price) for day+symbol (float64)
- `commission_cash`: Sum of commission costs (float64, >= 0.0, never NaN)
- `spread_cash`: Sum of spread costs (float64, >= 0.0, never NaN)
- `slippage_cash`: Sum of slippage costs (float64, >= 0.0, never NaN)
- `total_cost_cash`: Sum of total costs (float64, >= 0.0, never NaN)
- `cost_bps`: Total cost in basis points (total_cost_cash / notional * 10000) (float64, never NaN)
- `n_trades`: Number of trades for day+symbol (int64)

**Sorting:** Deterministic by (date, symbol) ascending.

**No NaN Policy:** All cost columns and cost_bps must be non-NaN (default to 0.0).

---

## File Locations

**TCA Reports:**
- CSV: `output/tca_report_{freq}.csv` (or `reports/tca_report_{freq}.csv` if repo convention)
- Markdown: `output/tca_report_{freq}.md` (optional)

**Integration:**
- TCA reports are generated from trades with cost columns (from `add_cost_columns_to_trades()`)
- Reports can be generated after backtest or daily run

---

## Usage

### Building TCA Report

```python
from src.assembled_core.qa.tca import build_tca_report

# Trades DataFrame with cost columns
trades = ...  # From backtest or daily run

# Build report
tca_report = build_tca_report(
    trades_df=trades,
    freq="1d",
    strategy_name="trend_baseline",
)
```

### Writing TCA Report

```python
from src.assembled_core.qa.tca import write_tca_report_csv, write_tca_report_md
from pathlib import Path

# Write CSV
csv_path = write_tca_report_csv(tca_report, Path("output/tca_report_1d.csv"))

# Write Markdown (optional)
md_path = write_tca_report_md(
    tca_report,
    Path("output/tca_report_1d.md"),
    strategy_name="trend_baseline",
)
```

---

## Determinism Rules

1. **UTC Normalization:** All timestamps must be UTC-aware.
2. **No NaNs:** All cost columns default to 0.0 if not computable.
3. **Deterministic Sorting:** Reports are sorted by (date, symbol) ascending.
4. **Stable Aggregation:** Sum operations are deterministic (no floating-point order dependence).

---

## References

- Commission Model: `src/assembled_core/execution/transaction_costs.CommissionModel`
- Spread Model: `src/assembled_core/execution/transaction_costs.SpreadModel`
- Slippage Model: `src/assembled_core/execution/transaction_costs.SlippageModel`
- TCA Reporting: `src/assembled_core/qa/tca`
