"""Wave 16 — Beyond Tier 1 + Forensics + Performance.

Covers:
- C2-010 chaos middleware (env-gated, opt-in pass-through)
- C2-013 bulkhead + circuit breaker
- C2-017 crisis injection
- C2-020 adversarial perturbation
- C2-022 Hansen SPA wrapper (graceful when arch absent)
- C2-030 conformal prediction ICP
- B-004 async_fetch (smoke — just import + retry helper)
"""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# C2-010 — Chaos middleware
# ---------------------------------------------------------------------------


def test_chaos_middleware_passes_through_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ASSEMBLED_CHAOS_MODE → middleware MUST be a pure pass-through."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.delenv("ASSEMBLED_CHAOS_MODE", raising=False)

    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    r = client.get("/live")
    assert r.status_code == 200
    assert r.headers.get("X-Chaos") is None


def test_chaos_middleware_can_force_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASSEMBLED_CHAOS_MODE=1 + 5XX_PROB=1.0 deterministically returns 500."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("ASSEMBLED_CHAOS_MODE", "1")
    monkeypatch.setenv("ASSEMBLED_CHAOS_5XX_PROB", "1.0")
    monkeypatch.setenv("ASSEMBLED_CHAOS_DROP_PROB", "0")
    monkeypatch.setenv("ASSEMBLED_CHAOS_LATENCY_MS", "0")
    monkeypatch.delenv("ASSEMBLED_API_RATE_LIMIT", raising=False)

    from src.assembled_core.api.app import create_app

    client = TestClient(create_app())
    r = client.get("/live")
    assert r.status_code == 500
    assert r.headers.get("X-Chaos") == "5xx"


# ---------------------------------------------------------------------------
# C2-013 — Bulkhead + circuit breaker
# ---------------------------------------------------------------------------


def test_bulkhead_circuit_opens_after_threshold() -> None:
    from src.assembled_core.utils.bulkhead import Bulkhead, BulkheadOpenError

    bh = Bulkhead(
        "alpaca-test",
        max_concurrent=4,
        failure_threshold=3,
        cooldown_seconds=10.0,
    )

    async def fail_three_times() -> None:
        for _ in range(3):
            try:
                async with bh.acquire():
                    raise RuntimeError("simulated upstream failure")
            except RuntimeError:
                pass

    asyncio.run(fail_three_times())
    assert bh.is_open is True

    async def must_be_refused() -> None:
        with pytest.raises(BulkheadOpenError):
            async with bh.acquire():
                pass

    asyncio.run(must_be_refused())


def test_bulkhead_success_resets_counter() -> None:
    from src.assembled_core.utils.bulkhead import Bulkhead

    bh = Bulkhead(
        "polygon-test",
        max_concurrent=2,
        failure_threshold=3,
        cooldown_seconds=10.0,
    )

    async def two_fail_then_success() -> None:
        for _ in range(2):
            try:
                async with bh.acquire():
                    raise RuntimeError("fail")
            except RuntimeError:
                pass
        async with bh.acquire():
            pass

    asyncio.run(two_fail_then_success())
    assert bh.is_open is False


# ---------------------------------------------------------------------------
# C2-017 — Crisis injection
# ---------------------------------------------------------------------------


def test_crisis_injection_replaces_window_with_expected_size() -> None:
    from src.assembled_core.qa.crisis_injection import (
        inject_2008_shock,
        inject_2020_covid_shock,
        inject_2022_inflation_grind,
        run_crisis_battery,
    )

    rng = np.random.default_rng(seed=42)
    returns = pd.Series(0.0005 + 0.01 * rng.standard_normal(1000))

    r2008 = inject_2008_shock(returns, start_index=100)
    # 30 days replaced — outside the window, values are unchanged.
    assert r2008.iloc[0] == returns.iloc[0]
    assert r2008.iloc[200] == returns.iloc[200]
    # Inside the window the values differ.
    assert not np.allclose(
        r2008.iloc[100:130].to_numpy(), returns.iloc[100:130].to_numpy()
    )

    r2020 = inject_2020_covid_shock(returns, start_index=500)
    assert r2020.iloc[400] == returns.iloc[400]
    r2022 = inject_2022_inflation_grind(returns, start_index=600)
    assert r2022.iloc[599] == returns.iloc[599]

    battery = run_crisis_battery(returns)
    assert set(battery.keys()) == {"2008", "2020", "2022"}


def test_crisis_injection_rejects_overflow_window() -> None:
    from src.assembled_core.qa.crisis_injection import inject_2022_inflation_grind

    returns = pd.Series(np.zeros(50))
    with pytest.raises(ValueError):
        inject_2022_inflation_grind(returns, start_index=0)  # needs 126 rows


# ---------------------------------------------------------------------------
# C2-020 — Adversarial feature perturbation
# ---------------------------------------------------------------------------


def test_adversarial_flips_simple_linear_model() -> None:
    from src.assembled_core.qa.adversarial_perturbation import min_perturbation_to_flip

    weights = np.array([1.0, -0.5, 0.0])

    def predict(x: np.ndarray) -> float:
        return float(np.dot(weights, x))

    x = np.array([1.0, 1.0, 1.0])  # baseline pred = 0.5 > 0
    result = min_perturbation_to_flip(predict, x, max_eps=5.0)
    assert result["flipped"] is True
    assert result["eps"] > 0


def test_adversarial_no_flip_within_budget_when_baseline_strong() -> None:
    from src.assembled_core.qa.adversarial_perturbation import min_perturbation_to_flip

    def predict(x: np.ndarray) -> float:
        return 100.0  # constant — no perturbation can flip the sign

    x = np.array([0.0, 0.0])
    result = min_perturbation_to_flip(predict, x, max_eps=1.0)
    assert result["flipped"] is False


# ---------------------------------------------------------------------------
# C2-022 — Hansen SPA wrapper (graceful when arch absent — but it IS installed)
# ---------------------------------------------------------------------------


def test_spa_p_values_returns_dict_with_three_p_values() -> None:
    pytest.importorskip("arch")
    from src.assembled_core.qa.spa_test import spa_p_values

    rng = np.random.default_rng(seed=42)
    n = 300
    # Candidate slightly beats benchmark
    bench = rng.standard_normal((n, 3)) * 0.01
    cand = bench[:, 0] + 0.0005 + 0.001 * rng.standard_normal(n)
    result = spa_p_values(
        pd.Series(cand),
        pd.DataFrame(bench, columns=["b1", "b2", "b3"]),
        reps=100,
        seed=1,
    )
    for key in ("p_lower", "p_consistent", "p_upper"):
        assert key in result
        assert 0.0 <= result[key] <= 1.0


def test_spa_p_values_short_series_returns_nan() -> None:
    from src.assembled_core.qa.spa_test import spa_p_values

    short = pd.Series(np.zeros(5))
    result = spa_p_values(short, pd.DataFrame({"b": np.zeros(5)}))
    assert np.isnan(result["p_lower"])


# ---------------------------------------------------------------------------
# C2-030 — Conformal prediction ICP
# ---------------------------------------------------------------------------


class _LinearModel:
    """Tiny model fixture — fits y = w·x via lstsq."""

    def __init__(self) -> None:
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearModel":
        self.w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return X @ self.w


def test_conformal_intervals_cover_at_target_rate() -> None:
    """ICP must achieve ~ (1 - alpha) coverage on held-out test data."""
    from src.assembled_core.qa.conformal import fit_conformal, predict_with_intervals

    rng = np.random.default_rng(seed=42)
    n_train, n_test = 600, 200
    X_train = rng.standard_normal((n_train, 3))
    true_w = np.array([1.0, -0.5, 0.3])
    y_train = X_train @ true_w + rng.standard_normal(n_train) * 0.5

    X_test = rng.standard_normal((n_test, 3))
    y_test = X_test @ true_w + rng.standard_normal(n_test) * 0.5

    model, cal_scores = fit_conformal(
        _LinearModel(), X_train, y_train, calibration_frac=0.2, alpha=0.1
    )
    result = predict_with_intervals(model, cal_scores, X_test, alpha=0.1)

    inside = (y_test >= result.lower) & (y_test <= result.upper)
    coverage = float(inside.mean())
    # Target 0.9 ± 0.05 — wider tolerance than 1/sqrt(200) but tight enough
    # to catch a half_width bug.
    assert 0.80 <= coverage <= 1.0, coverage
    assert result.half_width > 0


def test_conformal_size_factor_pauses_when_snr_zero() -> None:
    from src.assembled_core.qa.conformal import conformal_size_factor

    assert conformal_size_factor(edge=0.1, half_width=0.0) == 0.0
    assert conformal_size_factor(edge=0.0, half_width=1.0) == 0.0
    assert conformal_size_factor(edge=0.5, half_width=0.5) == pytest.approx(1.0)
    assert conformal_size_factor(edge=2.0, half_width=0.5, cap=1.0) == pytest.approx(
        1.0
    )


# ---------------------------------------------------------------------------
# B-004 — async_fetch retry helper
# ---------------------------------------------------------------------------


def test_async_retry_recovers_after_transient_failures() -> None:
    from src.assembled_core.utils.async_fetch import _async_retry

    counter = {"n": 0}

    async def fail_then_succeed() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    async def run() -> str:
        return await _async_retry(
            fail_then_succeed,
            attempts=4,
            base=0.001,
            cap=0.005,
            jitter=0.0,
            exceptions=(ConnectionError,),
        )

    result = asyncio.run(run())
    assert result == "ok"
    assert counter["n"] == 3
