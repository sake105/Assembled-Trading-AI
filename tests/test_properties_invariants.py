"""Property-style invariants (Sprint 4 / Plan C20).

These tests exercise invariants, not single examples. They use seeded
random inputs rather than `hypothesis` to avoid adding a new
dependency to the CI-critical lane. Each test runs N scenarios drawn
from a fixed seed, so failures are reproducible.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.reconciliation import (  # noqa: E402
    reconcile_ledger_vs_broker,
)
from src.assembled_core.risk.turnover_budget import apply_turnover_gate  # noqa: E402

SCENARIOS = 50


def _random_targets(rng: random.Random, n: int = 5) -> pd.DataFrame:
    symbols = [f"SYM{i}" for i in range(n)]
    weights = [rng.uniform(0.0, 0.2) for _ in range(n)]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "target_weight": weights,
            "target_qty": [w * 1000 for w in weights],
        }
    )


def _random_current(rng: random.Random, n: int = 5) -> pd.DataFrame:
    symbols = [f"SYM{i}" for i in range(n)]
    qtys = [rng.uniform(0.0, 100.0) for _ in range(n)]
    return pd.DataFrame({"symbol": symbols, "qty": qtys})


# ---------------------------------------------------------------------------
# Property 1 — apply_turnover_gate is monotone in the cap.
#
# Larger caps must produce a scale factor >= smaller caps. This is the
# core contract that lets callers tune the cap at runtime without
# fearing non-monotone behaviour.
# ---------------------------------------------------------------------------


def test_turnover_gate_monotone_in_cap() -> None:
    rng = random.Random(4711)
    for _ in range(SCENARIOS):
        targets = _random_targets(rng)
        current = _random_current(rng)
        est_turnover = rng.uniform(0.1, 1.0)

        cap_small = rng.uniform(0.05, 0.4)
        cap_large = cap_small + rng.uniform(0.05, 0.5)

        _, sf_small = apply_turnover_gate(
            target_positions=targets,
            current_positions=current,
            cap=cap_small,
            estimated_turnover=est_turnover,
        )
        _, sf_large = apply_turnover_gate(
            target_positions=targets,
            current_positions=current,
            cap=cap_large,
            estimated_turnover=est_turnover,
        )
        assert sf_large >= sf_small - 1e-12, (
            f"monotonicity violated: cap {cap_small}->{sf_small}, "
            f"cap {cap_large}->{sf_large}"
        )


# ---------------------------------------------------------------------------
# Property 2 — apply_turnover_gate is a no-op when already under budget.
# ---------------------------------------------------------------------------


def test_turnover_gate_noop_under_budget() -> None:
    rng = random.Random(1337)
    for _ in range(SCENARIOS):
        targets = _random_targets(rng)
        current = _random_current(rng)
        est_turnover = rng.uniform(0.01, 0.1)
        cap = est_turnover + rng.uniform(0.1, 0.5)  # cap > est

        out, sf = apply_turnover_gate(
            target_positions=targets,
            current_positions=current,
            cap=cap,
            estimated_turnover=est_turnover,
        )
        assert sf == 1.0
        pd.testing.assert_series_equal(
            out["target_weight"].reset_index(drop=True),
            targets["target_weight"].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Property 3 — reconcile_ledger_vs_broker is idempotent.
#
# Calling the reconciler twice on the same inputs must produce the
# same verdict both times. This rules out any hidden state in the
# reconciler.
# ---------------------------------------------------------------------------


def test_reconcile_is_idempotent() -> None:
    rng = random.Random(2024)
    for _ in range(SCENARIOS):
        n = rng.randint(1, 6)
        symbols = [f"SYM{i}" for i in range(n)]
        ledger = pd.DataFrame(
            {
                "symbol": symbols,
                "qty": [rng.uniform(0.0, 100.0) for _ in range(n)],
            }
        )
        # Broker is sometimes identical, sometimes drifted.
        broker = ledger.copy()
        if rng.random() < 0.5:
            drift_idx = rng.randint(0, n - 1)
            broker.iloc[drift_idx, broker.columns.get_loc("qty")] += rng.uniform(
                0.1, 5.0
            )

        ledger_cash = 10_000.0
        broker_cash = 10_000.0 + (
            rng.uniform(-10.0, 10.0) if rng.random() < 0.3 else 0.0
        )

        r1 = reconcile_ledger_vs_broker(
            ledger_positions_df=ledger,
            ledger_cash=ledger_cash,
            broker_positions_df=broker,
            broker_cash=broker_cash,
            fail_fast=False,
        )
        r2 = reconcile_ledger_vs_broker(
            ledger_positions_df=ledger,
            ledger_cash=ledger_cash,
            broker_positions_df=broker,
            broker_cash=broker_cash,
            fail_fast=False,
        )
        assert r1["ok"] == r2["ok"]
        assert r1["cash_match"] == r2["cash_match"]
        assert r1["cash_diff"] == pytest.approx(r2["cash_diff"])
        assert r1["missing_in_ledger"] == r2["missing_in_ledger"]
        assert r1["missing_in_broker"] == r2["missing_in_broker"]


# ---------------------------------------------------------------------------
# Property 4 — reconciler flags drift strictly outside tolerance.
# ---------------------------------------------------------------------------


def test_reconcile_flags_drift_outside_tolerance() -> None:
    rng = random.Random(99)
    base = pd.DataFrame([{"symbol": "AAPL", "qty": 10.0}])
    for _ in range(SCENARIOS):
        # Drift at least one unit above qty_tol=1e-6.
        drift = rng.uniform(1e-3, 2.0)
        broker = pd.DataFrame([{"symbol": "AAPL", "qty": 10.0 + drift}])
        report = reconcile_ledger_vs_broker(
            ledger_positions_df=base,
            ledger_cash=1_000.0,
            broker_positions_df=broker,
            broker_cash=1_000.0,
            fail_fast=False,
        )
        assert report["ok"] is False
        assert not report["position_diffs_df"].empty
