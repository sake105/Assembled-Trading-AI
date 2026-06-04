# tests/portfolio/test_position_sizing_parity.py
"""GUARDRAIL — emit-time target_qty / target_notional parity.

Pins FACT A (the harmless half): every position_sizing EMIT path writes BOTH
``target_qty`` AND ``target_notional`` as the SAME value at emit time
(= ``target_weight * total_capital`` notional dollars — emit does NOT round;
the 2-dp rounding happens later in the ``_tc_sizing`` overlays). Emit is the
ONLY place the two columns coexist and are equal-by-construction; the live
pipeline overlays (``_tc_sizing``) subsequently mutate only ``target_qty`` and
leave ``target_notional`` stale — so post-overlay drift between the two is
BY DESIGN and is NOT asserted here (see
``tests/pipeline/test_tc_sizing_target_qty_notional_drift.py`` for the
companion guardrail).

Source pinned (src/assembled_core/portfolio/position_sizing.py, current code):
- compute_target_positions                       lines 144-145
- compute_kelly_weights                          lines 306-307
- compute_risk_parity_weights                    lines 421-422
- compute_vol_scaled_weights                     lines 534-535
- compute_target_positions_with_smoothing        lines 913-915 (base-fn path)
- compute_target_positions_from_trend_signals    delegates to compute_target_positions
- apply_news_sentiment_weight_adjustment         lines 826-832 (cap-renorm path:
  recomputes target_notional, then re-syncs target_qty = target_notional)

Each emit assigns ``target_notional = target_weight * total_capital`` and then
``target_qty = target_notional`` (value-identical alias). The module is
READ-ONLY here — these tests pin existing behaviour, no src/ file is edited.

DISCRIMINATION: every test asserts EXACT column-wise equality immediately after
emit. A future edit that set the two columns to different values, scaled only
one of them, or dropped one column would fail loudly here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Repo root: tests/portfolio/<file> -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.portfolio.position_sizing import (  # noqa: E402
    apply_news_sentiment_weight_adjustment,
    compute_kelly_weights,
    compute_risk_parity_weights,
    compute_target_positions,
    compute_target_positions_from_trend_signals,
    compute_target_positions_with_smoothing,
    compute_vol_scaled_weights,
)

pytestmark = pytest.mark.fast

# A non-trivial capital so target_notional != target_weight (would mask a bug
# where one column accidentally held the weight instead of the notional).
CAPITAL = 100_000.0


def _long_signals() -> pd.DataFrame:
    """Five LONG signals with varied scores (and one FLAT, which must be dropped)."""
    return pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "direction": ["LONG", "LONG", "LONG", "LONG", "LONG", "FLAT"],
            "score": [0.9, 0.7, 0.5, 0.3, 0.1, 0.8],
        }
    )


def _vols() -> dict[str, float]:
    return {"AAA": 0.10, "BBB": 0.20, "CCC": 0.30, "DDD": 0.40, "EEE": 0.50}


def _assert_emit_parity(result: pd.DataFrame, label: str) -> None:
    """Both columns present, non-empty, and EXACTLY equal element-wise at emit."""
    assert not result.empty, f"{label}: emit produced an empty frame"
    assert "target_qty" in result.columns, f"{label}: target_qty column missing"
    assert "target_notional" in result.columns, (
        f"{label}: target_notional column missing"
    )
    # The load-bearing invariant: value-identical alias at emit time.
    assert (result["target_qty"] == result["target_notional"]).all(), (
        f"{label}: target_qty != target_notional at emit "
        f"(qty={result['target_qty'].tolist()} "
        f"notional={result['target_notional'].tolist()})"
    )
    # And both equal target_weight * CAPITAL (notional dollars, NOT shares / NOT weight).
    expected = result["target_weight"] * CAPITAL
    assert ((result["target_notional"] - expected).abs() < 1e-6).all(), (
        f"{label}: target_notional != target_weight * total_capital"
    )


# Each entry: (label, callable producing an emit frame from the shared inputs).
EMIT_PATHS = [
    (
        "compute_target_positions[equal_weight]",
        lambda: compute_target_positions(
            _long_signals(), total_capital=CAPITAL, equal_weight=True
        ),
    ),
    (
        "compute_target_positions[score_weight]",
        lambda: compute_target_positions(
            _long_signals(), total_capital=CAPITAL, equal_weight=False
        ),
    ),
    (
        "compute_target_positions_from_trend_signals",
        lambda: compute_target_positions_from_trend_signals(
            _long_signals(), total_capital=CAPITAL, min_score=0.0
        ),
    ),
    (
        "compute_kelly_weights",
        lambda: compute_kelly_weights(_long_signals(), total_capital=CAPITAL),
    ),
    (
        "compute_risk_parity_weights",
        lambda: compute_risk_parity_weights(
            _long_signals(), volatilities=_vols(), total_capital=CAPITAL
        ),
    ),
    (
        "compute_vol_scaled_weights",
        lambda: compute_vol_scaled_weights(
            _long_signals(), volatilities=_vols(), total_capital=CAPITAL
        ),
    ),
    (
        "compute_target_positions_with_smoothing[base_path]",
        # previous_positions=None -> returns the base compute_target_positions
        # frame, exercising the base-fn notional/qty assignment (lines 913-915).
        lambda: compute_target_positions_with_smoothing(
            _long_signals(),
            previous_positions=None,
            total_capital=CAPITAL,
            equal_weight=True,
        ),
    ),
]


@pytest.mark.parametrize("label,emit_fn", EMIT_PATHS, ids=[p[0] for p in EMIT_PATHS])
def test_emit_paths_have_qty_notional_parity(label, emit_fn):
    """Every emit path writes target_qty == target_notional, exactly, at emit."""
    result = emit_fn()
    _assert_emit_parity(result, label)


def test_smoothing_path_with_previous_positions_keeps_parity():
    """compute_target_positions_with_smoothing: even when smoothing rewrites
    target_weight, the base-fn re-scale (lines 913-915) keeps target_notional and
    target_qty value-identical. Pins that the smoothing branch does not drift the
    two columns apart."""
    prev = {"AAA": 0.05, "BBB": 0.05, "CCC": 0.05, "DDD": 0.05, "EEE": 0.05}
    result = compute_target_positions_with_smoothing(
        _long_signals(),
        previous_positions=prev,
        total_capital=CAPITAL,
        equal_weight=True,
        smoothing_alpha=0.3,
    )
    _assert_emit_parity(result, "compute_target_positions_with_smoothing[smoothed]")


def test_news_sentiment_cap_renorm_resyncs_qty_to_notional():
    """apply_news_sentiment_weight_adjustment (non-shadow) recomputes
    target_notional from the renormalized weights, then RE-SYNCS
    target_qty = target_notional (lines 826-832). This is the one place in
    position_sizing.py that maintains the alias AFTER an adjustment — pin that
    it keeps them equal (so a regression that desynced them here would fail)."""

    class _IdentityLinker:
        def link(self, entity):  # noqa: D401 - trivial stub
            return entity

    emit = compute_target_positions(
        _long_signals(), total_capital=CAPITAL, equal_weight=False
    )
    news = pd.DataFrame({"entity": ["AAA", "BBB"], "sentiment_score": [0.8, -0.4]})
    adjusted = apply_news_sentiment_weight_adjustment(
        emit,
        news,
        entity_linker=_IdentityLinker(),
        shadow_only=False,
    )
    # Weights actually changed (otherwise the re-sync branch never ran).
    assert not emit["target_weight"].equals(adjusted["target_weight"]), (
        "news-sentiment adjustment did not change any weight (branch not exercised)"
    )
    # And the alias is kept exactly in sync by lines 831-832.
    assert (adjusted["target_qty"] == adjusted["target_notional"]).all(), (
        "cap-renorm path desynced target_qty from target_notional"
    )


def test_discriminates_on_broken_alias():
    """Sanity: the parity assertion is strict — a frame whose target_qty was
    perturbed away from target_notional MUST fail _assert_emit_parity. This
    proves the guardrail is discriminating, not vacuous."""
    good = compute_target_positions(
        _long_signals(), total_capital=CAPITAL, equal_weight=True
    )
    broken = good.copy()
    broken.loc[broken.index[0], "target_qty"] = (
        broken.loc[broken.index[0], "target_qty"] + 1.0
    )
    with pytest.raises(AssertionError):
        _assert_emit_parity(broken, "deliberately-broken")
