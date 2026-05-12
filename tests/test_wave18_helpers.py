"""Wave 18 — autonomously-actionable KNOWN_ISSUES §8 items.

Covers:
    * utils.halt_cache.HaltCache (§8.6 _tc_sizing.py TODO)
    * risk.tilt_detection (audit C2-073)
    * portfolio.kelly_robust (audit C2-065)
    * qa.cagr_attribution (audit C2-068)
    * adapters.outbound.event_bus_inprocess (audit C2-053)
    * qa.conformal_adaptive — ACI (audit C2-031)
    * qa.conformal_quantile — CQR (audit C2-032)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# halt_cache
# ===========================================================================


class _FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now_monotonic(self) -> float:
        return self.t


def test_halt_cache_first_call_refreshes() -> None:
    from src.assembled_core.utils.halt_cache import HaltCache

    clock = _FakeClock()
    calls = {"n": 0}

    def supplier() -> list[str]:
        calls["n"] += 1
        return ["AAPL", "MSFT"]

    cache = HaltCache(supplier=supplier, ttl_seconds=60, clock=clock)
    snap = cache.snapshot()
    assert snap == frozenset({"AAPL", "MSFT"})
    assert calls["n"] == 1


def test_halt_cache_does_not_refresh_within_ttl() -> None:
    from src.assembled_core.utils.halt_cache import HaltCache

    clock = _FakeClock()
    calls = {"n": 0}

    def supplier() -> list[str]:
        calls["n"] += 1
        return ["AAPL"]

    cache = HaltCache(supplier=supplier, ttl_seconds=60, clock=clock)
    cache.snapshot()
    clock.t = 30.0  # well below TTL
    cache.snapshot()
    cache.snapshot()
    assert calls["n"] == 1


def test_halt_cache_refreshes_after_ttl() -> None:
    from src.assembled_core.utils.halt_cache import HaltCache

    clock = _FakeClock()
    seq = iter([["A"], ["A", "B"], ["B"]])

    def supplier() -> list[str]:
        return next(seq)

    cache = HaltCache(supplier=supplier, ttl_seconds=10, clock=clock)
    assert cache.snapshot() == frozenset({"A"})
    clock.t = 11
    assert cache.snapshot() == frozenset({"A", "B"})
    clock.t = 25
    assert cache.snapshot() == frozenset({"B"})


def test_halt_cache_failsoft_on_supplier_error() -> None:
    from src.assembled_core.utils.halt_cache import HaltCache

    clock = _FakeClock()
    state = {"raise": False}

    def supplier() -> list[str]:
        if state["raise"]:
            raise RuntimeError("upstream timeout")
        return ["AAPL"]

    cache = HaltCache(supplier=supplier, ttl_seconds=10, clock=clock)
    assert cache.snapshot() == frozenset({"AAPL"})
    state["raise"] = True
    clock.t = 11
    # Should keep the previous snapshot.
    assert cache.snapshot() == frozenset({"AAPL"})
    assert cache.consecutive_failures == 1


def test_halt_cache_force_refresh_bypasses_ttl() -> None:
    from src.assembled_core.utils.halt_cache import HaltCache

    clock = _FakeClock()
    seq = iter([["A"], ["B"]])

    def supplier() -> list[str]:
        return next(seq)

    cache = HaltCache(supplier=supplier, ttl_seconds=999, clock=clock)
    assert cache.snapshot() == frozenset({"A"})
    assert cache.force_refresh() == frozenset({"B"})


def test_halt_cache_rejects_bad_ttl() -> None:
    from src.assembled_core.utils.halt_cache import HaltCache

    with pytest.raises(ValueError):
        HaltCache(supplier=lambda: [], ttl_seconds=0)


# ===========================================================================
# tilt_detection
# ===========================================================================


def _make_history(values: list[tuple[int, float, float]]) -> list:
    """values: list of (days_ago, realized_pnl, equity)."""
    from src.assembled_core.risk.tilt_detection import DailyPnLPoint

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    return [
        DailyPnLPoint(ts=now - timedelta(days=d), realized_pnl=p, equity=e)
        for (d, p, e) in sorted(values, key=lambda t: -t[0])
    ]


def test_tilt_empty_history_returns_clean() -> None:
    from src.assembled_core.risk.tilt_detection import detect_tilt

    state = detect_tilt([])
    assert state.is_tilted is False
    assert state.triggered_rules == ()


def test_tilt_consecutive_losses_fires() -> None:
    from src.assembled_core.risk.tilt_detection import detect_tilt

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    history = _make_history([(5, -100, 100_000), (3, -120, 99_800), (1, -50, 99_700)])
    state = detect_tilt(history, now=now)
    assert state.is_tilted is True
    assert "consecutive_loss_days" in state.triggered_rules
    assert state.consecutive_losses_count == 3


def test_tilt_weekly_drawdown_fires() -> None:
    from src.assembled_core.risk.tilt_detection import detect_tilt

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    history = _make_history(
        [
            (6, +100, 100_000),
            (5, +0, 100_000),
            (1, -500, 90_000),  # 10% drop
        ]
    )
    state = detect_tilt(history, now=now)
    assert "weekly_drawdown" in state.triggered_rules
    assert state.weekly_dd_pct >= 0.08


def test_tilt_monthly_drawdown_fires_without_weekly() -> None:
    from src.assembled_core.risk.tilt_detection import detect_tilt

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    history = _make_history(
        [(28, +100, 100_000), (20, +0, 100_000), (15, -2_000, 80_000)]
    )
    state = detect_tilt(history, now=now)
    assert "monthly_drawdown" in state.triggered_rules
    # weekly window has 0 points, so should not fire weekly_drawdown
    assert "weekly_drawdown" not in state.triggered_rules


def test_tilt_no_signal_in_calm_market() -> None:
    from src.assembled_core.risk.tilt_detection import detect_tilt

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    history = _make_history([(5, +50, 100_050), (3, +30, 100_080), (1, +20, 100_100)])
    state = detect_tilt(history, now=now)
    assert state.is_tilted is False


# ===========================================================================
# kelly_robust
# ===========================================================================


def test_robust_kelly_zero_edge_returns_zero() -> None:
    from src.assembled_core.portfolio.kelly_robust import robust_kelly_fraction

    res = robust_kelly_fraction(mu=0.0, sigma2=0.04, n_samples=250)
    assert res.capped_fraction == 0.0
    assert res.binding_constraint == "zero_edge"


def test_robust_kelly_below_cap_shrinks_via_browne_whitt() -> None:
    from src.assembled_core.portfolio.kelly_robust import robust_kelly_fraction

    # raw kelly = 0.10 / 1.0 = 0.10
    # half-kelly = 0.05
    # browne-whitt (T=10, d=2) = 10/12 ≈ 0.833
    # -> ≈ 0.0417, below cap 0.25
    res = robust_kelly_fraction(
        mu=0.10, sigma2=1.0, n_samples=10, fractional_kelly=0.5, max_fraction=0.25
    )
    assert res.binding_constraint == "none"
    assert 0.04 < res.capped_fraction < 0.045


def test_robust_kelly_hits_max_fraction_cap() -> None:
    from src.assembled_core.portfolio.kelly_robust import robust_kelly_fraction

    res = robust_kelly_fraction(
        mu=1.0, sigma2=0.5, n_samples=10_000, fractional_kelly=1.0, max_fraction=0.25
    )
    assert res.capped_fraction == 0.25
    assert res.binding_constraint == "max_fraction"


def test_robust_kelly_browne_whitt_converges_to_one_with_n() -> None:
    from src.assembled_core.portfolio.kelly_robust import robust_kelly_fraction

    res_small = robust_kelly_fraction(mu=0.01, sigma2=0.04, n_samples=10)
    res_big = robust_kelly_fraction(mu=0.01, sigma2=0.04, n_samples=10_000)
    assert res_big.estimation_shrinkage > res_small.estimation_shrinkage


def test_robust_kelly_from_returns_smoke() -> None:
    from src.assembled_core.portfolio.kelly_robust import robust_kelly_from_returns

    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.001, scale=0.02, size=250)
    res = robust_kelly_from_returns(returns)
    assert res.capped_fraction >= 0.0


def test_robust_kelly_rejects_bad_inputs() -> None:
    from src.assembled_core.portfolio.kelly_robust import robust_kelly_fraction

    with pytest.raises(ValueError):
        robust_kelly_fraction(mu=0.1, sigma2=-1.0, n_samples=10)
    with pytest.raises(ValueError):
        robust_kelly_fraction(mu=0.1, sigma2=0.04, n_samples=0)
    with pytest.raises(ValueError):
        robust_kelly_fraction(mu=0.1, sigma2=0.04, n_samples=10, fractional_kelly=2.0)


# ===========================================================================
# cagr_attribution
# ===========================================================================


def _equity_curve(days: int, growth_rate_daily: float = 0.001) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=days, freq="D")
    return pd.Series(
        100_000.0 * np.cumprod(1.0 + np.full(days, growth_rate_daily)),
        index=idx,
    )


def test_cagr_attribution_quarterly_smoke() -> None:
    from src.assembled_core.qa.cagr_attribution import attribute_by_period

    eq = _equity_curve(365 * 2)
    res = attribute_by_period(eq, period="Q")
    assert not res.per_period.empty
    assert res.overall_cagr > 0
    assert res.overall_max_dd == 0.0  # monotone curve has no drawdown


def test_cagr_attribution_finds_worst_period() -> None:
    from src.assembled_core.qa.cagr_attribution import attribute_by_period

    idx = pd.date_range("2020-01-01", periods=400, freq="D")
    # Up then crash then recover.
    vals = np.linspace(100_000, 120_000, 200).tolist()
    vals += np.linspace(120_000, 80_000, 100).tolist()
    vals += np.linspace(80_000, 95_000, 100).tolist()
    eq = pd.Series(vals, index=idx)
    res = attribute_by_period(eq, period="Q")
    assert res.worst_period_label is not None
    # Max DD should be approximately (120k - 80k) / 120k ≈ 0.333
    assert 0.30 < res.overall_max_dd < 0.35


def test_cagr_attribution_rejects_non_datetime_index() -> None:
    from src.assembled_core.qa.cagr_attribution import attribute_by_period

    bad = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        attribute_by_period(bad)


# ===========================================================================
# InProcessEventBus
# ===========================================================================


def test_event_bus_pub_sub_basic() -> None:
    from src.assembled_core.adapters.outbound.event_bus_inprocess import (
        InProcessEventBus,
    )

    bus = InProcessEventBus()
    seen: list[dict] = []
    bus.subscribe("orders", lambda e: seen.append(dict(e)))
    bus.publish("orders", {"id": 1, "qty": 100})
    bus.publish("orders", {"id": 2, "qty": 200})
    assert len(seen) == 2
    assert seen[0]["id"] == 1
    assert bus.publish_count == 2
    assert "orders" in bus.topics()


def test_event_bus_isolates_failing_subscriber() -> None:
    from src.assembled_core.adapters.outbound.event_bus_inprocess import (
        InProcessEventBus,
    )

    bus = InProcessEventBus()
    good_seen: list[int] = []

    def bad(_e):  # type: ignore[no-untyped-def]
        raise RuntimeError("boom")

    def good(e):  # type: ignore[no-untyped-def]
        good_seen.append(int(e["x"]))

    bus.subscribe("topic", bad)
    bus.subscribe("topic", good)
    bus.publish("topic", {"x": 1})
    bus.publish("topic", {"x": 2})
    assert good_seen == [1, 2]
    assert bus.dispatch_errors == 2


def test_event_bus_no_subscribers_no_crash() -> None:
    from src.assembled_core.adapters.outbound.event_bus_inprocess import (
        InProcessEventBus,
    )

    bus = InProcessEventBus()
    bus.publish("nobody", {"x": 1})
    assert bus.publish_count == 1


def test_event_bus_implements_port() -> None:
    from src.assembled_core.adapters.outbound.event_bus_inprocess import (
        InProcessEventBus,
    )
    from src.assembled_core.ports.event_bus import EventBus

    assert isinstance(InProcessEventBus(), EventBus)


# ===========================================================================
# Adaptive Conformal Inference
# ===========================================================================


def test_aci_init_validates_inputs() -> None:
    from src.assembled_core.qa.conformal_adaptive import init_aci

    with pytest.raises(ValueError):
        init_aci(np.array([1.0]), target_alpha=0.0)
    with pytest.raises(ValueError):
        init_aci(np.array([1.0]), gamma=0.0)
    with pytest.raises(ValueError):
        init_aci(np.array([]))


def test_aci_updates_alpha_when_missing() -> None:
    from src.assembled_core.qa.conformal_adaptive import (
        current_half_width,
        init_aci,
        update_aci,
    )

    cal = np.linspace(0.1, 1.0, 50)
    state = init_aci(cal, target_alpha=0.1, gamma=0.1)
    initial_alpha = state.current_alpha
    initial_hw = current_half_width(state)
    # Force a clear miss (score 5 ≫ initial half-width).
    update_aci(state, y_true=5.0, y_pred=0.0)
    assert state.current_alpha < initial_alpha  # widened interval
    assert current_half_width(state) >= initial_hw


def test_aci_alpha_clamps_to_unit_interval() -> None:
    from src.assembled_core.qa.conformal_adaptive import init_aci, update_aci

    cal = np.linspace(0.1, 1.0, 50)
    state = init_aci(cal, target_alpha=0.5, gamma=0.4)
    for _ in range(50):
        update_aci(state, y_true=10.0, y_pred=0.0)  # always misses
    assert 0.0 < state.current_alpha < 1.0


def test_aci_converges_to_target_under_stationarity() -> None:
    from src.assembled_core.qa.conformal_adaptive import init_aci, update_aci

    rng = np.random.default_rng(0)
    # Calibrate on N(0,1) errors.
    cal = np.abs(rng.normal(size=200))
    state = init_aci(cal, target_alpha=0.1, gamma=0.01)
    # Stream from the same distribution.
    for _ in range(500):
        y = rng.normal()
        update_aci(state, y_true=y, y_pred=0.0)
    miss_rate = state.empirical_miss_rate()
    # Should be near 0.1 — give a generous band.
    assert 0.03 < miss_rate < 0.20


# ===========================================================================
# CQR
# ===========================================================================


class _MockQR:
    """A trivial quantile regressor with fixed asymmetric bands."""

    def __init__(self, lo: float = -1.0, hi: float = 0.5) -> None:
        self._lo = lo
        self._hi = hi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MockQR":
        return self

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        return np.full(n, self._lo), np.full(n, self._hi)


def test_cqr_predict_intervals_smoke() -> None:
    from src.assembled_core.qa.conformal_quantile import (
        fit_cqr,
        predict_with_intervals,
    )

    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 1))
    y = rng.normal(size=n) * 0.5
    model, cal_scores = fit_cqr(_MockQR(lo=-1.0, hi=0.5), X, y, calibration_frac=0.3)
    X_test = rng.normal(size=(50, 1))
    intervals = predict_with_intervals(model, cal_scores, X_test, alpha=0.1)
    assert intervals.lower.shape == intervals.upper.shape == (50,)
    assert intervals.Q >= 0.0
    assert (intervals.upper >= intervals.lower).all()


def test_cqr_rejects_bad_alpha() -> None:
    from src.assembled_core.qa.conformal_quantile import (
        fit_cqr,
        predict_with_intervals,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    model, cal = fit_cqr(_MockQR(), X, y, calibration_frac=0.3)
    with pytest.raises(ValueError):
        predict_with_intervals(model, cal, X[:5], alpha=0.0)


def test_cqr_rejects_bad_calibration_frac() -> None:
    from src.assembled_core.qa.conformal_quantile import fit_cqr

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = rng.normal(size=100)
    with pytest.raises(ValueError):
        fit_cqr(_MockQR(), X, y, calibration_frac=0.0)
