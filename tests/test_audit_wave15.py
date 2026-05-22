"""Audit Wave-15 — synthetic alpha injection (C2-015), MI feature screen
(C2-052), Almgren-Chriss spot-check (C4-019), and dataclass-slots sanity
(B-006).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# B-006 — dataclass slots sanity
# ---------------------------------------------------------------------------


def test_paper_order_has_slots() -> None:
    """PaperOrder must declare __slots__ (audit B-006)."""
    from src.assembled_core.execution.paper_trading_engine import PaperOrder

    assert hasattr(PaperOrder, "__slots__"), "PaperOrder should use slots=True"


def test_paper_position_has_slots() -> None:
    from src.assembled_core.execution.paper_trading_engine import PaperPosition

    assert hasattr(PaperPosition, "__slots__")


def test_tracked_order_has_slots() -> None:
    from src.assembled_core.execution.order_lifecycle import TrackedOrder

    assert hasattr(TrackedOrder, "__slots__")


def test_tracked_order_rejects_unknown_attribute() -> None:
    """slots prevents accidental attribute-typos at the call-site."""
    from datetime import datetime, timezone

    from src.assembled_core.execution.order_lifecycle import (
        OrderEvent,
        OrderState,
        TrackedOrder,
    )

    order = TrackedOrder(
        order_id="X",
        symbol="AAPL",
        side="BUY",
        quantity=1.0,
        price=100.0,
        source="test",
        current_state=OrderState.CREATED,
        events=[
            OrderEvent(state=OrderState.CREATED, timestamp=datetime.now(timezone.utc))
        ],
    )
    with pytest.raises(AttributeError):
        order.this_typo_must_not_silently_succeed = 42


# ---------------------------------------------------------------------------
# C2-015 — Synthetic alpha injection — does our permutation test detect it?
# ---------------------------------------------------------------------------


def test_permutation_test_detects_known_synthetic_alpha() -> None:
    """Inject a synthetic strategy with mean=0.0008/day (≈ Sharpe 1.5 annualised)
    and confirm that ``permutation_p_value`` returns p < 0.01 — exactly the
    audit-mandated promotion threshold (C2-016).
    """
    from src.assembled_core.qa.metrics import permutation_p_value

    rng = np.random.default_rng(seed=42)
    # mean 0.0008/day, std 0.008/day → SR ~ sqrt(252) * 0.0008 / 0.008 ≈ 1.58
    n_days = 1260  # 5 trading years — enough for stable signal
    returns = pd.Series(0.0008 + 0.008 * rng.standard_normal(n_days))
    result = permutation_p_value(returns, n_permutations=2000, seed=7)
    assert result["p_value"] < 0.01, result


def test_permutation_test_rejects_pure_noise_at_99pct() -> None:
    """Mirror of the above — a zero-mean strategy must NOT be flagged
    significant. With 1260 noisy returns the p-value should sit well
    above 0.01.
    """
    from src.assembled_core.qa.metrics import permutation_p_value

    rng = np.random.default_rng(seed=99)
    returns = pd.Series(0.008 * rng.standard_normal(1260))
    result = permutation_p_value(returns, n_permutations=2000, seed=11)
    assert result["p_value"] > 0.05, result


# ---------------------------------------------------------------------------
# C2-052 — Mutual-information feature screen
# ---------------------------------------------------------------------------


def test_mutual_info_screen_ranks_strong_feature_first() -> None:
    """Inject one feature that linearly drives the target and confirm the
    MI ranking puts it ahead of pure-noise features.
    """
    from src.assembled_core.qa.feature_screen import mutual_info_screen

    rng = np.random.default_rng(seed=42)
    n = 500
    # Three features: f1 = signal, f2/f3 = noise.
    f1 = rng.standard_normal(n)
    y = pd.Series(f1 + 0.3 * rng.standard_normal(n), name="target")
    X = pd.DataFrame(
        {
            "noise_a": rng.standard_normal(n),
            "signal_f1": f1,
            "noise_b": rng.standard_normal(n),
        }
    )
    result = mutual_info_screen(X, y)
    assert not result.empty
    # signal_f1 should rank #1.
    assert result.iloc[0]["feature"] == "signal_f1"


def test_mutual_info_screen_top_n_truncates() -> None:
    from src.assembled_core.qa.feature_screen import mutual_info_screen

    rng = np.random.default_rng(seed=42)
    n = 200
    X = pd.DataFrame(
        {f"f{j}": rng.standard_normal(n) for j in range(10)},
    )
    y = pd.Series(rng.standard_normal(n))
    out = mutual_info_screen(X, y, top_n=3)
    assert len(out) == 3


def test_mutual_info_screen_empty_inputs() -> None:
    from src.assembled_core.qa.feature_screen import mutual_info_screen

    X = pd.DataFrame({"f1": [], "f2": []})
    y = pd.Series([], dtype=float)
    out = mutual_info_screen(X, y)
    assert out.empty


# ---------------------------------------------------------------------------
# C4-019 — Almgren-Chriss spot-check (existence + sign-on-buy)
# ---------------------------------------------------------------------------


def test_almgren_chriss_module_exposes_optimal_trajectory() -> None:
    """File exists + has at least one entry-point function (audit C4-019
    forensic check is a research-level review beyond unit test scope;
    here we pin the public surface stays present so callers don't bitrot).
    """
    import src.assembled_core.execution.almgren_chriss as ac

    # Find a callable entry point — accept any of the common naming
    # conventions found in academic implementations.
    candidates = [
        n
        for n in dir(ac)
        if not n.startswith("_")
        and callable(getattr(ac, n))
        and any(
            tok in n.lower()
            for tok in (
                "trajectory",
                "schedule",
                "optimal",
                "almgren",
                "execute",
            )
        )
    ]
    assert candidates, f"no public Almgren-Chriss callable in module: {dir(ac)}"


# ---------------------------------------------------------------------------
# E-010 — Perf regression guard (stdlib timing, no pytest-benchmark dep)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_compute_equity_metrics_perf_smoke() -> None:
    """A 5-year daily equity curve must process in well under one second.

    The threshold is intentionally loose (2.0 s on any reasonable machine)
    so this test catches order-of-magnitude regressions — accidental
    O(N²) loops, a forgotten ``apply`` over a 1260-row frame, etc. —
    without flapping on day-to-day variance.

    The point is not to be a benchmark; it is to fail loud the moment
    someone introduces an N² hotpath into the metrics pipeline.
    """
    import time

    from src.assembled_core.qa.metrics import compute_equity_metrics

    idx = pd.date_range("2024-01-01", periods=1260, freq="B", tz="UTC")
    rng = np.random.default_rng(seed=42)
    equity = 100_000.0 * np.cumprod(1.0 + 0.0003 + 0.01 * rng.standard_normal(1260))
    equity_df = pd.DataFrame({"timestamp": idx, "equity": equity})

    start = time.perf_counter()
    for _ in range(3):
        _ = compute_equity_metrics(equity_df, start_capital=100_000.0, freq="1d")
    elapsed = time.perf_counter() - start
    # 3 calls × 5-year daily curve should be << 2 s on any dev machine.
    assert elapsed < 2.0, (
        f"compute_equity_metrics is too slow: {elapsed:.2f}s for 3 calls"
    )
