# Spec: `signals/crash_prediction.py`

## Purpose

Aggregate a set of crash-leading indicators into a single
`crash_probability ∈ [0, 1]` that the orchestrator can use as an
exposure multiplier. The goal is proportional de-risking as crash
probability rises, not a single binary kill.

## Public API

- `CrashSignal` — dataclass representing a single input signal
  (value, category weight, timestamp).
- `CrashPredictionEngine` — stateful engine that accepts signals,
  weights them per category (technical, regime, geopolitical,
  macro), and returns a probability.
- `compute_rolling_percentile_thresholds(series, window) -> tuple`
  — helper for calibrating per-indicator thresholds from history.

## Category weights (default)

- `technical`: 0.30
- `regime`: 0.25
- `geopolitical`: 0.25
- `macro`: 0.20

These sum to 1.0 by construction. Any category with no live signals
contributes zero; the remaining categories are NOT renormalised.
That is intentional: missing data reduces confidence, it does not
redistribute trust to the categories we happened to observe.

## Invariants

- Output is in `[0, 1]`.
- Output is monotone in every signal strength: a stronger crash
  signal cannot decrease `crash_probability`.
- Category weights are a policy input. The engine does not rewrite
  them based on live data.

## Error handling

- Signal timestamp after `as_of` → drop the signal (PIT safety).
- Unknown category → log warning, skip. Do not raise; crash
  prediction must degrade gracefully, not halt the cycle.

## Test strategy

- All-zero signals → probability 0.
- All-max signals → probability 1.
- One-category only → probability ≤ that category's weight.
- Monotonicity: adding a positive signal never decreases output.

## Known limits

- Geopolitical and macro categories currently feed from dormant
  modules (W6 wiring, Sprint 3). MVP uses technical + regime only.
- The engine is a weighted sum, not a trained classifier. A learned
  layer on top is a deliberate non-goal — interpretability matters
  more than fit.
