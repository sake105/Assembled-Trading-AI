"""Tests for portfolio/hrp_sizing.py (Sprint 3 / Plan W10)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# HRP uses scipy linkage; skip when missing.
pytest.importorskip("scipy")

from src.assembled_core.portfolio.hrp_sizing import (  # noqa: E402
    apply_hrp_sizing,
    apply_hrp_sizing_from_policy,
    blend_hrp_with_score,
    compute_hrp_target_weights,
)


def _panel(symbols: list[str], n_bars: int = 120, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    driver = rng.normal(0.0, 0.01, size=n_bars)
    for i, sym in enumerate(symbols):
        idio = rng.normal(0.0, 0.005 + 0.001 * i, size=n_bars)
        rets = driver + idio
        px = 100.0 * np.cumprod(1.0 + rets)
        for b in range(n_bars):
            rows.append({
                "timestamp": base + pd.Timedelta(days=b),
                "symbol": sym,
                "close": float(px[b]),
            })
    return pd.DataFrame(rows)


def test_empty_scores_return_empty() -> None:
    adjusted, reasons = apply_hrp_sizing({}, pd.DataFrame())
    assert adjusted == {}
    assert reasons == []


def test_insufficient_prices_falls_back_to_score() -> None:
    scores = {"AAA": 0.4, "BBB": 0.2}
    # No price panel at all → fallback to score, scaled to target_invested_pct=1
    adjusted, reasons = apply_hrp_sizing(scores, pd.DataFrame())
    assert set(adjusted.keys()) == {"AAA", "BBB"}
    total = sum(adjusted.values())
    assert abs(total - 1.0) < 1e-9
    assert any("insufficient" in r for r in reasons)


def test_hrp_blend_produces_valid_weights() -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}

    adjusted, reasons = apply_hrp_sizing(
        scores,
        prices,
        lookback_days=100,
        blend=0.7,
        target_invested_pct=0.8,
    )
    assert set(adjusted.keys()) == set(symbols)
    total = sum(adjusted.values())
    assert abs(total - 0.8) < 1e-6
    for w in adjusted.values():
        assert w >= 0.0
    assert any("blended HRP" in r for r in reasons)


def test_blend_zero_equals_score_only() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}

    adjusted, _ = apply_hrp_sizing(
        scores,
        prices,
        blend=0.0,
        target_invested_pct=1.0,
    )
    # With blend=0 we should recover normalized score weights
    for s in symbols:
        assert abs(adjusted[s] - scores[s]) < 1e-6


def test_from_policy_disabled_returns_copy() -> None:
    scores = {"AAA": 0.5, "BBB": 0.5}
    adjusted, reasons = apply_hrp_sizing_from_policy(
        scores, pd.DataFrame(), {"hrp_sizing": {"enabled": False}}
    )
    assert adjusted == scores
    assert adjusted is not scores
    assert reasons == []


def test_from_policy_enabled_applies_blend() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
    policy = {
        "hrp_sizing": {
            "enabled": True,
            "lookback_days": 100,
            "blend": 0.5,
            "target_invested_pct": 0.9,
        }
    }
    adjusted, reasons = apply_hrp_sizing_from_policy(scores, prices, policy)
    total = sum(adjusted.values())
    assert abs(total - 0.9) < 1e-6
    assert any("blended HRP" in r for r in reasons)


# ---------------------------------------------------------------------------
# W10 sidecar: compute_hrp_target_weights + blend_hrp_with_score
# ---------------------------------------------------------------------------


def _wide_returns_panel(n_days: int = 90, n_symbols: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0.0005, scale=0.01, size=(n_days, n_symbols))
    cols = [f"SYM{i}" for i in range(n_symbols)]
    idx = pd.date_range("2024-01-02", periods=n_days, freq="B")
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.mark.phase12
def test_compute_hrp_target_weights_sum_to_target_gross() -> None:
    panel = _wide_returns_panel()
    w = compute_hrp_target_weights(panel, target_gross=0.80)
    assert isinstance(w, pd.Series)
    assert w.name == "hrp_weight"
    assert w.sum() == pytest.approx(0.80, abs=1e-6)


@pytest.mark.phase12
def test_compute_hrp_target_weights_non_negative() -> None:
    panel = _wide_returns_panel()
    w = compute_hrp_target_weights(panel)
    assert (w >= 0).all()


@pytest.mark.phase12
def test_compute_hrp_target_weights_custom_gross() -> None:
    panel = _wide_returns_panel()
    w = compute_hrp_target_weights(panel, target_gross=1.25)
    assert w.sum() == pytest.approx(1.25, abs=1e-6)


@pytest.mark.phase12
def test_compute_hrp_target_weights_too_short_raises() -> None:
    panel = _wide_returns_panel(n_days=10)
    with pytest.raises(ValueError, match="insufficient history"):
        compute_hrp_target_weights(panel, min_history=30)


@pytest.mark.phase12
def test_compute_hrp_target_weights_single_symbol_raises() -> None:
    panel = _wide_returns_panel(n_symbols=1)
    with pytest.raises(ValueError, match="at least 2 symbols"):
        compute_hrp_target_weights(panel)


@pytest.mark.phase12
def test_compute_hrp_target_weights_invalid_gross_raises() -> None:
    panel = _wide_returns_panel()
    with pytest.raises(ValueError, match="target_gross"):
        compute_hrp_target_weights(panel, target_gross=0.0)


@pytest.mark.phase12
def test_compute_hrp_target_weights_not_dataframe_raises() -> None:
    with pytest.raises(ValueError, match="DataFrame"):
        compute_hrp_target_weights([1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.phase12
def test_blend_alpha_one_returns_hrp() -> None:
    hrp = pd.Series({"A": 0.4, "B": 0.3, "C": 0.1}, name="hrp_weight")
    score = pd.Series({"A": 0.2, "B": 0.2, "C": 0.4}, name="score")
    out = blend_hrp_with_score(hrp, score, hrp_alpha=1.0)
    for sym in ["A", "B", "C"]:
        assert out[sym] == pytest.approx(hrp[sym], abs=1e-9)


@pytest.mark.phase12
def test_blend_alpha_zero_returns_score() -> None:
    hrp = pd.Series({"A": 0.4, "B": 0.3, "C": 0.1}, name="hrp_weight")
    score = pd.Series({"A": 0.2, "B": 0.2, "C": 0.4}, name="score")
    out = blend_hrp_with_score(hrp, score, hrp_alpha=0.0)
    for sym in ["A", "B", "C"]:
        assert out[sym] == pytest.approx(score[sym], abs=1e-9)


@pytest.mark.phase12
def test_blend_alpha_half_midpoint_same_gross() -> None:
    hrp = pd.Series({"A": 0.5, "B": 0.5}, name="hrp_weight")
    score = pd.Series({"A": 0.5, "B": 0.5}, name="score")
    out = blend_hrp_with_score(hrp, score, hrp_alpha=0.5)
    assert out["A"] == pytest.approx(0.5, abs=1e-9)
    assert out["B"] == pytest.approx(0.5, abs=1e-9)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.phase12
def test_blend_alpha_out_of_range_raises() -> None:
    hrp = pd.Series({"A": 0.5, "B": 0.5}, name="hrp_weight")
    score = pd.Series({"A": 0.5, "B": 0.5}, name="score")
    with pytest.raises(ValueError, match="hrp_alpha"):
        blend_hrp_with_score(hrp, score, hrp_alpha=1.5)
    with pytest.raises(ValueError, match="hrp_alpha"):
        blend_hrp_with_score(hrp, score, hrp_alpha=-0.1)


@pytest.mark.phase12
def test_blend_handles_disjoint_symbols() -> None:
    hrp = pd.Series({"A": 0.4, "B": 0.3, "C": 0.1}, name="hrp_weight")
    score = pd.Series({"A": 0.3, "B": 0.2, "D": 0.3}, name="score")
    out = blend_hrp_with_score(hrp, score, hrp_alpha=0.5)
    assert set(out.index) == {"A", "B", "C", "D"}
    target = max(hrp.sum(), score.sum())
    assert out.sum() == pytest.approx(target, abs=1e-9)
    assert out["C"] > 0
    assert out["D"] > 0


@pytest.mark.phase12
def test_blend_gross_does_not_inflate() -> None:
    hrp = pd.Series({"A": 0.6, "B": 0.2}, name="hrp_weight")
    score = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4}, name="score")
    out = blend_hrp_with_score(hrp, score, hrp_alpha=0.7)
    assert out.sum() == pytest.approx(1.0, abs=1e-9)
