"""B0 — Golden-equity-baseline 1e-9 regression test.

The reference backtest (fixture = 25 symbols × 126 business-days, seed=42,
signal_fn + sizing_fn from ``scripts/profile_backtest.py``) must reproduce the
committed ``golden_equity_baseline.json`` bit-for-bit (``abs_tol=1e-9``,
``rel_tol=1e-12``). This is the gate for Part B speed fixes: every B1-B5
change must still pass this test.

Rebuild the baseline only when a deliberate semantic change is accepted:

    python scripts/build_golden_equity_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.phase_speed

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "regression" / "golden_equity_baseline.json"


def _run_reference():
    # Imported lazily so collection of this module doesn't drag in the full
    # backtest stack when it's not being executed.
    import sys

    sys.path.insert(0, str(REPO))
    from scripts.build_golden_equity_baseline import run_reference

    cfg = json.loads(GOLDEN.read_text(encoding="utf-8"))["config"]
    return run_reference(
        n_symbols=int(cfg["n_symbols"]),
        n_days=int(cfg["n_days"]),
        seed=int(cfg["seed"]),
    )


def test_golden_equity_baseline_exists() -> None:
    assert GOLDEN.exists(), (
        f"Golden baseline missing: {GOLDEN}. "
        "Run `python scripts/build_golden_equity_baseline.py` to regenerate."
    )
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert "equity" in data and data["equity"], "baseline has no equity rows"
    assert "config" in data, "baseline missing config block"


def test_reference_backtest_is_bit_identical_to_baseline() -> None:
    baseline = json.loads(GOLDEN.read_text(encoding="utf-8"))
    candidate = _run_reference()

    assert baseline["n_bars"] == candidate["n_bars"], (
        f"bar count drifted: baseline={baseline['n_bars']} "
        f"candidate={candidate['n_bars']}"
    )

    base_eq = np.asarray([r["equity"] for r in baseline["equity"]], dtype=np.float64)
    cand_eq = np.asarray([r["equity"] for r in candidate["equity"]], dtype=np.float64)
    assert base_eq.shape == cand_eq.shape

    diff = np.abs(cand_eq - base_eq)
    max_abs = float(diff.max()) if len(diff) else 0.0
    denom = np.where(np.abs(base_eq) > 0, np.abs(base_eq), 1.0)
    max_rel = float((diff / denom).max()) if len(diff) else 0.0

    assert max_abs <= 1e-9, (
        f"B0 REGRESSION: max_abs_diff={max_abs:.3e} > 1e-9 — "
        f"a B-phase change altered the reference equity curve. If this is "
        f"intentional, regenerate the baseline."
    )
    assert max_rel <= 1e-12, (
        f"B0 REGRESSION: max_rel_diff={max_rel:.3e} > 1e-12"
    )


def test_reference_timestamps_match_baseline() -> None:
    import pandas as pd

    baseline = json.loads(GOLDEN.read_text(encoding="utf-8"))
    candidate = _run_reference()
    base_ts = pd.to_datetime([r["timestamp"] for r in baseline["equity"]], utc=True)
    cand_ts = pd.to_datetime([r["timestamp"] for r in candidate["equity"]], utc=True)
    assert (base_ts == cand_ts).all(), "timestamp sequence drifted from baseline"
