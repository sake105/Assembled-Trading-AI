"""Adaptive AC convergence validation: does eta_hat reach a sensible level?

Procedure:
  1. Simulate 20 trading days (1 month).
  2. Each day: 5 parent orders, each executed in 5 child slices.
  3. Market impact model: slippage ≈ eta_true * (qty / sigma) + noise.
  4. Track eta_hat (EWMA estimate) after every fill.
  5. Check convergence: does eta_hat approach eta_true within ~30%?

Market-impact literature benchmarks (Almgren-Chriss 2001, Barra):
  - eta (temporary impact) for large-cap US equities: 0.05 – 0.20
  - SPX-level liquidity (very high ADV): lower end, ~0.05 – 0.10
  - Mid-cap / lower ADV: 0.10 – 0.25
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import numpy as np

from assembled_core.execution.execution_router import (
    AdaptiveACState,
    ExecutionConfig,
    Order,
    adaptive_ac_split,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
rng = np.random.default_rng(2024)

ETA_TRUE      = 0.12     # true market-impact η we're trying to discover
SIGMA_DAILY   = 0.013    # daily vol ~ 1.3% (SPX-ish)
PRICE         = 100.0    # reference price
N_DAYS        = 20       # trading days in 1 month
ORDERS_PER_DAY = 5       # parent orders per day
QTY_PER_ORDER = 5000     # shares per parent order
N_CHILD       = 5        # AC slices per parent
NOISE_SCALE   = 0.3      # fractional noise on fills (realistic fill randomness)
EWMA_ALPHA    = 0.12     # learning rate for eta_hat

# Config: start with a prior that is deliberately off (tests adaptation)
ETA_PRIOR     = 0.08     # deliberately low prior

cfg = ExecutionConfig(
    twap_slices=N_CHILD,
    almgren_eta=ETA_PRIOR,
    almgren_gamma=0.05,
    almgren_lambda=1e-6,
)
state = AdaptiveACState.from_config(cfg, ewma_alpha=EWMA_ALPHA)

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------
eta_history: list[tuple[int, int, float]] = []   # (day, order, eta_hat)
fill_count = 0

print(f"Adaptive AC convergence simulation")
print(f"  eta_true={ETA_TRUE:.3f}  eta_prior={ETA_PRIOR:.3f}  alpha={EWMA_ALPHA}")
print(f"  {N_DAYS} days × {ORDERS_PER_DAY} orders × {N_CHILD} slices = "
      f"{N_DAYS * ORDERS_PER_DAY * N_CHILD} fills total")
print()

for day in range(1, N_DAYS + 1):
    for order_idx in range(ORDERS_PER_DAY):
        # Slight price drift each order
        price = PRICE * (1 + rng.normal(0, SIGMA_DAILY))

        parent = Order(
            symbol="SPY",
            side="BUY",
            quantity=QTY_PER_ORDER,
            price=price,
            order_id=f"d{day:02d}o{order_idx}",
        )

        slices = adaptive_ac_split(parent, cfg, state)

        for sl in slices:
            # Simulate fill: actual price = expected + impact + noise
            # Impact model consistent with update() formula:
            #   slippage_frac = eta * qty * sigma_daily^2 / price
            # => implied_eta = slippage * price / (qty * sigma^2) recovers eta_true
            impact_frac = ETA_TRUE * sl.quantity * SIGMA_DAILY ** 2 / price
            noise_frac  = rng.normal(0, NOISE_SCALE * impact_frac)
            actual_price = price * (1 + impact_frac + noise_frac)

            # Update adaptive state
            state.update(
                qty_filled=sl.quantity,
                expected_price=price,
                actual_price=actual_price,
                side="BUY",
                sigma_daily=SIGMA_DAILY,
            )
            fill_count += 1

        eta_history.append((day, order_idx, state.eta_hat))

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
eta_final    = state.eta_hat
eta_day_end  = [eta_history[(d * ORDERS_PER_DAY) - 1][2] for d in range(1, N_DAYS + 1)]
convergence  = abs(eta_final - ETA_TRUE) / ETA_TRUE   # fractional error

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("=" * 60)
print("ADAPTIVE ALMGREN-CHRISS eta CONVERGENCE REPORT")
print("=" * 60)

print(f"\nSimulation: {fill_count} fills over {N_DAYS} trading days")
print(f"  True eta:       {ETA_TRUE:.4f}")
print(f"  Prior eta:      {ETA_PRIOR:.4f}")
print(f"  EWMA alpha:     {EWMA_ALPHA:.2f}")

print(f"\nDay-end eta_hat trajectory:")
print(f"  {'Day':>4}  {'eta_hat':>8}  {'error%':>8}")
for d, eta_d in enumerate(eta_day_end, 1):
    err_pct = (eta_d - ETA_TRUE) / ETA_TRUE * 100
    print(f"  {d:>4}  {eta_d:>8.5f}  {err_pct:>+7.1f}%")

print(f"\nConvergence:")
print(f"  eta_hat (final):   {eta_final:.5f}")
print(f"  eta_true:          {ETA_TRUE:.5f}")  # noqa
print(f"  Fractional error:  {convergence*100:.1f}%")
print(f"  Within 30%:        {'YES' if convergence <= 0.30 else 'NO'}")
print(f"  Within 20%:        {'YES' if convergence <= 0.20 else 'NO'}")

# Convergence speed: how many days to get within 40%?
days_to_40pct = next(
    (d for d, eta_d in enumerate(eta_day_end, 1)
     if abs(eta_d - ETA_TRUE) / ETA_TRUE < 0.40),
    None
)
print(f"  Days to <40% err:  "
      f"{'day ' + str(days_to_40pct) if days_to_40pct else '>20 (no convergence)'}")

# Literature plausibility check
print(f"\nLiterature range for large-cap US equities: eta in [0.05, 0.20]")
print(f"  eta_true ({ETA_TRUE:.3f}) in range:  "
      f"{'YES' if 0.05 <= ETA_TRUE <= 0.20 else 'NO (out of range)'}")
print(f"  eta_hat ({eta_final:.3f}) in range:   "
      f"{'YES' if 0.05 <= eta_final <= 0.20 else 'NO (diverged)'}")

if convergence <= 0.30:
    print(f"\nVerdict: CONVERGES. eta_hat tracks eta_true within {convergence*100:.0f}% after {N_DAYS} days.")
else:
    print(f"\nVerdict: SLOW CONVERGENCE. Fractional error = {convergence*100:.0f}%.")
    print(f"         Consider increasing EWMA alpha or reducing fill noise.")
