"""Tests for portfolio/bl_sizing.py (Sprint 3 / Plan W11)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip('src.assembled_core.portfolio.bl_sizing')
from src.assembled_core.portfolio.bl_sizing import (  # noqa: E402
    apply_bl_sizing,
    apply_bl_sizing_from_policy,
    blend_bl_with_score,
    compute_bl_target_weights,
)

# BL optimiser needs scipy.optimize.
pytest.importorskip("scipy")


def _panel(symbols: list[str], n_bars: int = 120, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []
    for i, sym in enumerate(symbols):
        rets = rng.normal(0.0005, 0.012 + 0.002 * i, size=n_bars)
        px = 100.0 * np.cumprod(1.0 + rets)
        for b in range(n_bars):
            rows.append({
                "timestamp": base + pd.Timedelta(days=b),
                "symbol": sym,
                "close": float(px[b]),
            })
    return pd.DataFrame(rows)


def test_empty_scores_return_empty() -> None:
    adjusted, reasons = apply_bl_sizing({}, pd.DataFrame())
    assert adjusted == {}
    assert reasons == []


def test_fallback_to_score_on_missing_prices() -> None:
    scores = {"AAA": 0.4, "BBB": 0.2}
    adjusted, reasons = apply_bl_sizing(scores, pd.DataFrame(), target_invested_pct=1.0)
    assert set(adjusted.keys()) == {"AAA", "BBB"}
    total = sum(adjusted.values())
    assert abs(total - 1.0) < 1e-9
    assert any("insufficient" in r or "falling back" in r for r in reasons)


def test_bl_sizing_produces_scaled_weights() -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.3, "CCC": 0.2, "DDD": 0.1}

    adjusted, reasons = apply_bl_sizing(
        scores,
        prices,
        lookback_days=100,
        target_invested_pct=0.75,
        max_position=0.35,
    )
    assert set(adjusted.keys()) == set(symbols)
    total = sum(abs(w) for w in adjusted.values())
    assert abs(total - 0.75) < 1e-6
    assert any("BL posterior" in r for r in reasons)


def test_from_policy_disabled_returns_copy() -> None:
    scores = {"AAA": 0.5, "BBB": 0.5}
    adjusted, reasons = apply_bl_sizing_from_policy(
        scores, pd.DataFrame(), {"bl_sizing": {"enabled": False}}
    )
    assert adjusted == scores
    assert adjusted is not scores
    assert reasons == []


def test_from_policy_enabled_applies() -> None:
    symbols = ["AAA", "BBB", "CCC"]
    prices = _panel(symbols, n_bars=120)
    scores = {"AAA": 0.4, "BBB": 0.4, "CCC": 0.2}
    policy = {
        "bl_sizing": {
            "enabled": True,
            "lookback_days": 100,
            "target_invested_pct": 0.9,
            "max_position": 0.5,
        }
    }
    adjusted, reasons = apply_bl_sizing_from_policy(scores, prices, policy)
    total = sum(adjusted.values())
    assert abs(total - 0.9) < 1e-6
    assert any("BL posterior" in r for r in reasons)


# ---------------------------------------------------------------------------
# W11 sidecar: compute_bl_target_weights + blend_bl_with_score
# ---------------------------------------------------------------------------


def _returns_panel(n_bars: int = 120, n_syms: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    symbols = [f"SYM{i}" for i in range(n_syms)]
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")
    data = rng.normal(0.0005, 0.012, size=(n_bars, n_syms))
    return pd.DataFrame(data, index=dates, columns=symbols)


@pytest.mark.phase12
def test_compute_bl_target_weights_sums_to_target_gross() -> None:
    panel = _returns_panel()
    views = pd.Series(
        {"SYM0": 1.0, "SYM1": 0.5, "SYM2": 0.0, "SYM3": -0.5, "SYM4": -1.0}
    )
    w = compute_bl_target_weights(panel, views, target_gross=0.80)
    assert abs(float(w.sum()) - 0.80) < 1e-6
    assert w.name == "bl_weight"
    assert list(w.index) == list(panel.columns)


@pytest.mark.phase12
def test_compute_bl_target_weights_long_only() -> None:
    panel = _returns_panel()
    views = pd.Series(
        {"SYM0": 2.0, "SYM1": 1.0, "SYM2": 0.0, "SYM3": -1.0, "SYM4": -2.0}
    )
    w = compute_bl_target_weights(panel, views, target_gross=1.0)
    assert (w >= -1e-12).all(), f"expected long-only weights, got {w.to_dict()}"


@pytest.mark.phase12
def test_compute_bl_target_weights_positive_view_beats_negative() -> None:
    # Build a panel with identical-ish variance across symbols so the view
    # is the dominant differentiator. Use a small seed with low-noise draws.
    rng = np.random.default_rng(11)
    n = 150
    symbols = ["POS", "NEG"]
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = rng.normal(0.0005, 0.01, size=(n, 2))
    panel = pd.DataFrame(data, index=dates, columns=symbols)

    views = pd.Series({"POS": 1.5, "NEG": -1.5})
    w = compute_bl_target_weights(panel, views, target_gross=1.0)
    assert w["POS"] > w["NEG"], (
        f"positive view should dominate: POS={w['POS']}, NEG={w['NEG']}"
    )


@pytest.mark.phase12
def test_compute_bl_target_weights_short_history_raises() -> None:
    panel = _returns_panel(n_bars=20)
    views = pd.Series({f"SYM{i}": 0.0 for i in range(5)})
    with pytest.raises(ValueError, match="insufficient history"):
        compute_bl_target_weights(panel, views)


@pytest.mark.phase12
def test_compute_bl_target_weights_mismatched_views_raises() -> None:
    panel = _returns_panel()
    views = pd.Series({"SYM0": 1.0, "SYM1": 0.5})  # missing SYM2..SYM4
    with pytest.raises(ValueError, match="missing symbols"):
        compute_bl_target_weights(panel, views)


@pytest.mark.phase12
def test_compute_bl_target_weights_equal_weight_prior_baseline() -> None:
    panel = _returns_panel()
    views = pd.Series({f"SYM{i}": 0.0 for i in range(5)})
    w = compute_bl_target_weights(
        panel, views, target_gross=0.6, equal_weight_prior=True
    )
    assert abs(float(w.sum()) - 0.6) < 1e-6
    # With zero views and equal-weight prior, posterior should roughly track
    # the prior. Weights should all be meaningfully > 0.
    assert (w > 0).all()


@pytest.mark.phase12
def test_blend_bl_with_score_alpha_one_returns_bl() -> None:
    bl = pd.Series({"A": 0.6, "B": 0.4}, name="bl_weight")
    score = pd.Series({"A": 0.3, "B": 0.7}, name="score")
    out = blend_bl_with_score(bl, score, bl_alpha=1.0)
    # With identical gross sums (both 1.0) and alpha=1.0, result equals bl.
    assert abs(out["A"] - 0.6) < 1e-12
    assert abs(out["B"] - 0.4) < 1e-12


@pytest.mark.phase12
def test_blend_bl_with_score_alpha_zero_returns_score() -> None:
    bl = pd.Series({"A": 0.6, "B": 0.4}, name="bl_weight")
    score = pd.Series({"A": 0.3, "B": 0.7}, name="score")
    out = blend_bl_with_score(bl, score, bl_alpha=0.0)
    assert abs(out["A"] - 0.3) < 1e-12
    assert abs(out["B"] - 0.7) < 1e-12


@pytest.mark.phase12
def test_blend_bl_with_score_rejects_alpha_out_of_range() -> None:
    bl = pd.Series({"A": 0.5, "B": 0.5})
    score = pd.Series({"A": 0.5, "B": 0.5})
    with pytest.raises(ValueError, match="bl_alpha"):
        blend_bl_with_score(bl, score, bl_alpha=-0.1)
    with pytest.raises(ValueError, match="bl_alpha"):
        blend_bl_with_score(bl, score, bl_alpha=1.1)
