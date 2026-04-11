# Spec: `risk/turnover_budget.py`

## Purpose

Daily turnover gate. Estimates the one-way turnover implied by
moving from current positions to target positions and scales the
target towards the current book if the estimate exceeds the budget.

## Public API

- `estimate_turnover(current_positions, target_positions, prices, equity) -> dict`
  — returns `{ "gross_dollar", "gross_pct", "symbol_breakdown" }`.
- `apply_turnover_gate(current_positions, target_positions, prices, equity, max_turnover_pct) -> tuple[pd.DataFrame, dict]`
  — returns the possibly-scaled target positions and a report.

## Inputs

- `current_positions`: DataFrame with `symbol`, `qty`.
- `target_positions`: DataFrame with `symbol`, `target_weight`.
- `prices`: latest price panel, one row per symbol.
- `equity`: portfolio equity used to convert weights into dollars.
- `max_turnover_pct`: hard cap on one-way turnover as % of equity.

## Invariants

- Scale factor is in `[0, 1]`. The gate only reduces, never amplifies.
- Monotone: a larger `max_turnover_pct` yields a scale factor ≥ the
  scale factor at a smaller cap.
- If the estimate is already under the cap, the function is a no-op
  (returns target unchanged and `scale_factor=1.0`).

## Error handling

- Missing price for a target symbol → drop the target with a warning.
  Do not substitute a fake price.
- Zero equity → raise. A turnover budget on zero equity is
  meaningless.

## Test strategy

- Happy path: target within budget, scale factor is 1.0.
- Binding path: target above budget, scale factor is `budget /
  estimate`.
- Mixed path: some symbols unchanged, only the delta counts.
- Property: monotonicity in `max_turnover_pct`.

## Known limits

- Turnover is estimated on a snapshot. The actual fills may differ if
  the strategy chases over the day; the gate is a budget, not a
  guarantee.
