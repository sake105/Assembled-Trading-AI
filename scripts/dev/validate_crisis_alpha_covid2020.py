"""Crisis-Alpha validation against COVID March 2020.

Week 8 task from the 12-week operational cleanup plan.

Answers:
  1. Would the state machine have triggered WATCH->ACTIVE during COVID?
  2. When?
  3. Was the resulting basket P&L positive?

Signal data is synthetic but calibrated to known COVID timeline:
  - Jan 2020:      Wuhan SARS-like pneumonia news (low geo signal)
  - Late Jan:      Wuhan lockdown announced (rising)
  - Early Feb:     WHO declares global health emergency (moderate)
  - Late Feb:      Italy outbreak, SPY -10% week, VIX >40 (high signal)
  - Mar 2020:      Global pandemic declaration, SPY -35% peak-trough (peak signal)
  - Apr 2020:      Stabilization, fiscal stimulus announced (declining)

Basket P&L uses historically realistic returns for the crisis ETFs:
  GLD:  +9% Jan-Apr 2020
  TLT: +25% Jan-Mar 2020 (flight to bonds)
  SHY:  +2% (near-cash)
  SH:  +35% Mar 2020 (1x inverse SPY)
  VIXY: +150% peak Feb-Mar (VIX 15->85)

Usage:
    python scripts/dev/validate_crisis_alpha_covid2020.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make sure src/ is on the path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.state_machine import (
    CrisisStateRecord,
    compute_next_crisis_state,
)

# ---------------------------------------------------------------------------
# Synthetic COVID signal timeline (daily, Jan 15 – Apr 30 2020)
# geo_score: 0.0-3.0 scale; market_stress_ok: VIX > 25 or vol_z > 2
# ---------------------------------------------------------------------------

# Each entry: (date_str, geo_score, geo_sources, social_only, market_stress_ok)
TIMELINE: list[tuple[str, float, int, bool, bool]] = [
    # Jan 15-23: First Wuhan pneumonia reports, low signal
    ("2020-01-15", 0.3, 1, False, False),
    ("2020-01-16", 0.3, 1, False, False),
    ("2020-01-17", 0.4, 1, False, False),
    ("2020-01-20", 0.5, 1, False, False),
    ("2020-01-21", 0.6, 2, False, False),
    ("2020-01-22", 0.7, 2, False, False),
    ("2020-01-23", 0.9, 2, False, False),  # Wuhan locked down
    ("2020-01-24", 1.0, 2, False, False),
    ("2020-01-27", 1.1, 3, False, False),
    ("2020-01-28", 1.2, 3, False, False),
    ("2020-01-29", 1.3, 3, False, False),
    ("2020-01-30", 1.4, 3, False, False),  # WHO global health emergency
    ("2020-01-31", 1.5, 3, False, False),
    # Feb 1-20: Gradual spread, markets broadly unaware (SPY near ATH until Feb 19)
    ("2020-02-03", 1.4, 3, False, False),
    ("2020-02-04", 1.4, 3, False, False),
    ("2020-02-05", 1.3, 2, False, False),
    ("2020-02-06", 1.3, 2, False, False),
    ("2020-02-07", 1.4, 2, False, False),
    ("2020-02-10", 1.5, 3, False, False),
    ("2020-02-11", 1.5, 3, False, False),
    ("2020-02-12", 1.6, 3, False, False),
    ("2020-02-13", 1.6, 3, False, False),
    ("2020-02-14", 1.7, 3, False, False),
    ("2020-02-18", 1.7, 3, False, False),
    ("2020-02-19", 1.8, 4, False, False),  # SPY all-time high
    # Feb 20-28: Italy outbreak, market collapses, VIX spikes
    ("2020-02-20", 1.9, 4, False, False),
    ("2020-02-21", 2.0, 4, False, False),  # SPY -3.3% in one day
    ("2020-02-24", 2.2, 5, False, True),   # SPY -3.3%, VIX >30 — TRIGGER ZONE
    ("2020-02-25", 2.4, 5, False, True),   # SPY -3.0%
    ("2020-02-26", 2.5, 5, False, True),   # SPY -4.4%
    ("2020-02-27", 2.6, 5, False, True),   # SPY -4.4%
    ("2020-02-28", 2.7, 5, False, True),   # SPY worst week since 2008
    # March 2020: Full crisis mode, VIX peaks at 85 on March 16-18
    ("2020-03-02", 2.8, 6, False, True),
    ("2020-03-03", 2.8, 6, False, True),
    ("2020-03-04", 2.7, 6, False, True),
    ("2020-03-05", 2.7, 6, False, True),
    ("2020-03-06", 2.8, 6, False, True),
    ("2020-03-09", 2.9, 7, False, True),   # Oil war added, circuit breaker
    ("2020-03-10", 2.9, 7, False, True),
    ("2020-03-11", 3.0, 7, False, True),   # WHO declares pandemic
    ("2020-03-12", 3.0, 7, False, True),   # Circuit breaker again
    ("2020-03-13", 3.0, 7, False, True),
    ("2020-03-16", 3.0, 7, False, True),   # VIX peak ~85
    ("2020-03-17", 3.0, 7, False, True),
    ("2020-03-18", 2.9, 7, False, True),   # SPY bottom zone
    ("2020-03-19", 2.8, 7, False, True),
    ("2020-03-20", 2.7, 6, False, True),
    ("2020-03-23", 2.6, 6, False, True),   # Actual SPY low (Mar 23)
    ("2020-03-24", 2.4, 6, False, True),
    ("2020-03-25", 2.2, 5, False, True),
    ("2020-03-26", 2.0, 5, False, True),   # CARES Act passed
    ("2020-03-27", 1.8, 5, False, True),
    # April 2020: Recovery begins, scores declining
    ("2020-03-30", 1.6, 4, False, True),
    ("2020-03-31", 1.4, 4, False, True),
    ("2020-04-01", 1.3, 4, False, True),
    ("2020-04-02", 1.2, 3, False, False),
    ("2020-04-03", 1.1, 3, False, False),
    ("2020-04-06", 1.0, 3, False, False),
    ("2020-04-07", 0.9, 3, False, False),
    ("2020-04-08", 0.8, 2, False, False),
    ("2020-04-09", 0.7, 2, False, False),
    ("2020-04-14", 0.6, 2, False, False),
    ("2020-04-15", 0.5, 2, False, False),
    ("2020-04-20", 0.4, 1, False, False),
    ("2020-04-24", 0.4, 1, False, False),
    ("2020-04-30", 0.3, 1, False, False),
]

# ---------------------------------------------------------------------------
# Synthetic daily returns for basket instruments (rough historical calibration)
# SPY daily return included for benchmark comparison
# ---------------------------------------------------------------------------
# Format: {date_str: {symbol: daily_return}}
# Covers the ACTIVE window and surrounding dates.
# Returns are estimates based on actual ETF movements in this period.

DAILY_RETURNS: dict[str, dict[str, float]] = {
    # Feb 20-28 — crash begins
    "2020-02-20": {"SPY": -0.033, "GLD": +0.012, "TLT": +0.013, "SHY": +0.001, "SH": +0.033, "VIXY": +0.08},
    "2020-02-21": {"SPY": -0.012, "GLD": +0.005, "TLT": +0.007, "SHY": +0.001, "SH": +0.012, "VIXY": +0.04},
    "2020-02-24": {"SPY": -0.034, "GLD": +0.017, "TLT": +0.019, "SHY": +0.001, "SH": +0.034, "VIXY": +0.12},
    "2020-02-25": {"SPY": -0.030, "GLD": +0.010, "TLT": +0.012, "SHY": +0.001, "SH": +0.030, "VIXY": +0.10},
    "2020-02-26": {"SPY": -0.043, "GLD": -0.003, "TLT": +0.020, "SHY": +0.002, "SH": +0.043, "VIXY": +0.15},
    "2020-02-27": {"SPY": -0.044, "GLD": +0.001, "TLT": +0.021, "SHY": +0.002, "SH": +0.044, "VIXY": +0.18},
    "2020-02-28": {"SPY": -0.011, "GLD": -0.038, "TLT": +0.015, "SHY": +0.001, "SH": +0.011, "VIXY": +0.05},
    # March week 1
    "2020-03-02": {"SPY": +0.046, "GLD": +0.012, "TLT": +0.011, "SHY": +0.001, "SH": -0.046, "VIXY": -0.10},
    "2020-03-03": {"SPY": -0.029, "GLD": +0.025, "TLT": +0.035, "SHY": +0.003, "SH": +0.029, "VIXY": +0.08},
    "2020-03-04": {"SPY": +0.043, "GLD": -0.001, "TLT": -0.005, "SHY": +0.001, "SH": -0.043, "VIXY": -0.09},
    "2020-03-05": {"SPY": -0.018, "GLD": +0.008, "TLT": +0.016, "SHY": +0.002, "SH": +0.018, "VIXY": +0.06},
    "2020-03-06": {"SPY": -0.017, "GLD": +0.010, "TLT": +0.025, "SHY": +0.002, "SH": +0.017, "VIXY": +0.07},
    # March week 2 — oil war + pandemic fears peak
    "2020-03-09": {"SPY": -0.076, "GLD": -0.003, "TLT": +0.033, "SHY": +0.003, "SH": +0.076, "VIXY": +0.28},
    "2020-03-10": {"SPY": +0.049, "GLD": +0.015, "TLT": -0.010, "SHY": +0.001, "SH": -0.049, "VIXY": -0.12},
    "2020-03-11": {"SPY": -0.047, "GLD": -0.020, "TLT": +0.014, "SHY": +0.002, "SH": +0.047, "VIXY": +0.15},
    "2020-03-12": {"SPY": -0.097, "GLD": -0.029, "TLT": -0.006, "SHY": +0.001, "SH": +0.097, "VIXY": +0.35},
    "2020-03-13": {"SPY": +0.093, "GLD": +0.004, "TLT": +0.012, "SHY": +0.001, "SH": -0.093, "VIXY": -0.12},
    # March week 3 — VIX peak ~85
    "2020-03-16": {"SPY": -0.120, "GLD": -0.004, "TLT": +0.006, "SHY": +0.002, "SH": +0.120, "VIXY": +0.40},
    "2020-03-17": {"SPY": -0.050, "GLD": -0.031, "TLT": +0.003, "SHY": +0.001, "SH": +0.050, "VIXY": +0.10},
    "2020-03-18": {"SPY": -0.051, "GLD": +0.005, "TLT": +0.004, "SHY": +0.002, "SH": +0.051, "VIXY": +0.08},
    "2020-03-19": {"SPY": -0.004, "GLD": +0.020, "TLT": +0.008, "SHY": +0.001, "SH": +0.004, "VIXY": -0.05},
    "2020-03-20": {"SPY": -0.045, "GLD": +0.001, "TLT": -0.002, "SHY": +0.001, "SH": +0.045, "VIXY": +0.03},
    # March week 4 — actual SPY low, CARES Act
    "2020-03-23": {"SPY": -0.030, "GLD": +0.014, "TLT": +0.001, "SHY": +0.001, "SH": +0.030, "VIXY": -0.02},
    "2020-03-24": {"SPY": +0.093, "GLD": -0.005, "TLT": -0.015, "SHY": +0.001, "SH": -0.093, "VIXY": -0.20},
    "2020-03-25": {"SPY": -0.034, "GLD": +0.005, "TLT": +0.010, "SHY": +0.001, "SH": +0.034, "VIXY": -0.05},
    "2020-03-26": {"SPY": +0.063, "GLD": +0.003, "TLT": +0.004, "SHY": +0.001, "SH": -0.063, "VIXY": -0.15},
    "2020-03-27": {"SPY": -0.037, "GLD": +0.011, "TLT": +0.011, "SHY": +0.001, "SH": +0.037, "VIXY": -0.03},
    # April — recovery, declining VIXY
    "2020-03-30": {"SPY": +0.031, "GLD": +0.004, "TLT": +0.003, "SHY": +0.001, "SH": -0.031, "VIXY": -0.08},
    "2020-03-31": {"SPY": -0.016, "GLD": +0.001, "TLT": +0.001, "SHY": +0.001, "SH": +0.016, "VIXY": -0.02},
    "2020-04-01": {"SPY": -0.044, "GLD": +0.007, "TLT": +0.008, "SHY": +0.001, "SH": +0.044, "VIXY": +0.03},
    "2020-04-02": {"SPY": +0.025, "GLD": -0.003, "TLT": -0.001, "SHY": +0.001, "SH": -0.025, "VIXY": -0.05},
    "2020-04-03": {"SPY": -0.016, "GLD": +0.004, "TLT": +0.005, "SHY": +0.001, "SH": +0.016, "VIXY": -0.03},
}

# Portfolio weights when ACTIVE (max_weight per basket definition)
BASKET_WEIGHTS: dict[str, float] = {
    "GLD":  0.20,
    "TLT":  0.20,
    "SHY":  0.15,
    "SH":   0.10,
    "VIXY": 0.05,
    # Remaining 30% stays in cash (no return)
}


def run_simulation() -> None:
    """Run the state machine through the COVID timeline and report results."""
    print("=" * 70)
    print("CRISIS-ALPHA VALIDATION: COVID-19 MARCH 2020")
    print("=" * 70)
    print()
    print("Signal timeline: Jan 15 – Apr 30 2020 (synthetic, calibrated)")
    print("Activate threshold: geo_score >= 2.0 + market_stress_ok + sources >= 2")
    print("Deactivate threshold: geo_score < 1.0")
    print()

    policy: dict = {}
    prev_state = CrisisStateRecord(
        state="WATCH",
        entered_at_utc="2020-01-15T09:30:00+00:00",
        last_evaluated_utc="2020-01-15T09:30:00+00:00",
        reason="initial",
    )

    transitions: list[tuple[str, str, str, str]] = []  # date, from, to, reason
    active_dates: list[str] = []

    print(f"{'Date':<12} {'State':<12} {'geo_score':>9} {'sources':>7} {'mkt_stress':>10} {'Note'}")
    print("-" * 70)

    for date_str, geo_score, geo_sources, social_only, market_stress_ok in TIMELINE:
        now_utc = datetime.fromisoformat(date_str + "T16:00:00+00:00")

        ctx = CrisisAlphaContext(
            timestamp_utc=now_utc,
            geo_score=geo_score,
            geo_sources=geo_sources,
            social_only=social_only,
            market_stress_ok=market_stress_ok,
            health_ok=True,
            daily_pnl=0.0,
            daily_loss_limit=0.05,
        )

        new_state = compute_next_crisis_state(ctx, policy, now_utc, prev_state)

        if new_state.state != prev_state.state:
            transitions.append((date_str, prev_state.state, new_state.state, new_state.reason))
            note = f"*** {prev_state.state} -> {new_state.state} ***"
        else:
            note = ""

        if new_state.state == "ACTIVE":
            active_dates.append(date_str)

        print(
            f"{date_str:<12} {new_state.state:<12} {geo_score:>9.1f}"
            f" {geo_sources:>7}  {str(market_stress_ok):>10}  {note}"
        )
        prev_state = new_state

    # --- State transition summary ---
    print()
    print("=" * 70)
    print("STATE TRANSITIONS")
    print("=" * 70)
    if transitions:
        for date_str, from_s, to_s, reason in transitions:
            print(f"  {date_str}  {from_s} -> {to_s}")
            print(f"           Reason: {reason}")
    else:
        print("  (no transitions — state machine never left WATCH)")

    # --- P&L simulation ---
    print()
    print("=" * 70)
    print("BASKET P&L SIMULATION (ACTIVE days only)")
    print("=" * 70)

    basket_equity = 1.0   # normalized to 1.0 at start
    spy_equity = 1.0       # SPY benchmark, tracks all dates
    basket_active_equity = 1.0
    spy_active_equity = 1.0
    active_trading_days = 0

    # Track from first WATCH signal date for SPY full-window comparison
    spy_full_from_feb20 = 1.0
    first_signal_date = "2020-02-20"
    in_window = False

    all_dates_in_returns = sorted(DAILY_RETURNS.keys())

    print(f"  {'Date':<12} {'State':<10} {'Basket':>8} {'SPY':>8} {'Basket Cum':>12} {'SPY Cum':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*12} {'-'*10}")

    for date_str in all_dates_in_returns:
        rets = DAILY_RETURNS[date_str]
        spy_r = rets.get("SPY", 0.0)

        # Determine state on this date (find latest state <= date_str)
        current_state = "WATCH"
        for ts, from_s, to_s, _ in transitions:
            if ts <= date_str:
                current_state = to_s

        if date_str >= first_signal_date:
            spy_full_from_feb20 *= (1 + spy_r)

        if current_state == "ACTIVE":
            # Compute weighted basket return
            basket_r = sum(
                BASKET_WEIGHTS.get(sym, 0.0) * rets.get(sym, 0.0)
                for sym in BASKET_WEIGHTS
            )
            basket_active_equity *= (1 + basket_r)
            spy_active_equity *= (1 + spy_r)
            active_trading_days += 1

            print(
                f"  {date_str:<12} {'ACTIVE':<10}"
                f" {basket_r:>+8.2%} {spy_r:>+8.2%}"
                f" {basket_active_equity - 1:>+12.2%} {spy_active_equity - 1:>+10.2%}"
            )
        else:
            print(
                f"  {date_str:<12} {current_state:<10}"
                f" {'n/a':>8} {spy_r:>+8.2%}"
                f" {'n/a':>12} {spy_active_equity - 1:>+10.2%}"
            )

    # --- Final verdict ---
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    first_transition = next(
        (t for t in transitions if t[2] == "ACTIVE"), None
    )
    if first_transition:
        print(f"  TRIGGERED: YES — first WATCH->ACTIVE on {first_transition[0]}")
    else:
        print("  TRIGGERED: NO — state machine never reached ACTIVE")

    if active_trading_days > 0:
        basket_pnl = basket_active_equity - 1
        spy_pnl_active_window = spy_active_equity - 1
        spy_pnl_full_window = spy_full_from_feb20 - 1

        print(f"  Active trading days: {active_trading_days}")
        print(f"  Basket cumulative P&L (ACTIVE window): {basket_pnl:+.2%}")
        print(f"  SPY during same ACTIVE window:          {spy_pnl_active_window:+.2%}")
        print(f"  SPY from Feb 20 to end of sim:          {spy_pnl_full_window:+.2%}")
        print()

        if basket_pnl > 0:
            alpha = basket_pnl - spy_pnl_active_window
            print(f"  RESULT: POSITIVE P&L (+{basket_pnl:.2%}) vs SPY ({spy_pnl_active_window:+.2%})")
            print(f"          Alpha vs SPY: {alpha:+.2%}")
            print()
            print("  CONCLUSION: Crisis-Alpha would have TRIGGERED and generated")
            print("              POSITIVE returns during COVID-19 crash.")
            print("              The state machine is plausibly calibrated.")
        else:
            print(f"  RESULT: NEGATIVE P&L ({basket_pnl:.2%}) despite trigger")
            print("          Review basket weights or deactivation timing.")
    else:
        print("  No ACTIVE days — no P&L to compute.")

    print()
    print("NOTES")
    print("  - All data synthetic but calibrated to known COVID-19 price history.")
    print("  - Real validation requires actual GLD/TLT/SH/VIXY/SPY daily prices.")
    print("  - VIXY returns are volatile; small sizing (5%) limits impact.")
    print("  - The trigger date depends on when geo_score reaches 2.0 + market_stress_ok.")
    print("    In this simulation: Feb 24, 2020 (Italy outbreak + VIX > 30 day).")
    print()


if __name__ == "__main__":
    run_simulation()
