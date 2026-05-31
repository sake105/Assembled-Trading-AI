"""STR-001 regression: forward-looking LABEL columns can never be
cross-sectionally ranked into a live ``*_xrank`` feature.

Background: ``ta_factors_core._add_multi_horizon_returns`` builds forward returns
with ``shift(-N)`` — ``returns_1m/3m/6m/12m`` and ``momentum_12m_excl_1m``. The
last one reads like a trailing momentum factor (name collision with the causal
``trailing_momentum_12m_excl_1m`` twin) and used to sit in the default
``rank_cols`` of ``_tc_features.build_features``. When enhanced enrichment ran on
a non-precomputed (live/eod/paper) panel it produced ``momentum_12m_excl_1m_xrank``
— a forward-looking value carried as a feature. Contained at audit time (no named
consumer, precomputed backtests skip enrichment) but a latent PIT leak.

This pins three things:
1. the denylist ``FORWARD_LOOKING_LABEL_COLS`` stays in sync with what the
   producer actually emits;
2. ``_strip_forward_label_cols`` drops every forward label (loudly) and keeps the
   causal ``trailing_*`` twins;
3. the leak is REAL without the guard and GONE with it (end-to-end mechanics).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.cross_sectional import rank_cross_sectional
from src.assembled_core.features.ta_factors_core import (
    FORWARD_LOOKING_LABEL_COLS,
    build_core_ta_factors,
)
from src.assembled_core.pipeline._tc_features import _strip_forward_label_cols


def _panel(n_symbols: int = 6, n_days: int = 320, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="UTC")
    frames = []
    for i in range(n_symbols):
        steps = rng.normal(0.0005, 0.02, size=n_days)
        close = 100.0 * np.exp(np.cumsum(steps))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": f"SYM{i:02d}",
                    "close": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "volume": rng.integers(1_000, 10_000, size=n_days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 1. denylist <-> producer sync
# --------------------------------------------------------------------------- #


def test_denylist_matches_producer_forward_columns() -> None:
    """Every name in the denylist must actually be produced by
    build_core_ta_factors — otherwise the guard silently protects nothing."""
    enriched = build_core_ta_factors(_panel())
    for col in FORWARD_LOOKING_LABEL_COLS:
        assert col in enriched.columns, f"{col} not produced — denylist drift"
    # the collision-prone one is explicitly covered
    assert "momentum_12m_excl_1m" in FORWARD_LOOKING_LABEL_COLS


def test_causal_trailing_twins_are_not_in_denylist() -> None:
    # The guard must not strip the genuinely causal trailing factors.
    assert "trailing_momentum_12m_excl_1m" not in FORWARD_LOOKING_LABEL_COLS
    assert "trailing_returns_12m" not in FORWARD_LOOKING_LABEL_COLS


# --------------------------------------------------------------------------- #
# 2. _strip_forward_label_cols behaviour
# --------------------------------------------------------------------------- #


def test_strip_drops_forward_labels_preserves_order() -> None:
    cols = [
        "trend_strength_20",
        "momentum_12m_excl_1m",
        "quality_score",
        "returns_12m",
        "trend_strength_50",
    ]
    out = _strip_forward_label_cols(cols)
    assert out == ["trend_strength_20", "quality_score", "trend_strength_50"]


def test_strip_noop_when_clean_returns_copy() -> None:
    cols = ["trend_strength_20", "quality_score"]
    out = _strip_forward_label_cols(cols)
    assert out == cols
    assert out is not cols  # defensive copy, not the same object


def test_strip_warns_loudly_on_drop(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _strip_forward_label_cols(["quality_score", "momentum_12m_excl_1m"])
    assert any(
        "PIT guard" in r.message and "momentum_12m_excl_1m" in str(r.args)
        for r in caplog.records
    )


def test_strip_silent_when_no_drop(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _strip_forward_label_cols(["quality_score", "trend_strength_20"])
    assert not [r for r in caplog.records if "PIT guard" in r.message]


# --------------------------------------------------------------------------- #
# 3. end-to-end leak mechanics: real without guard, gone with guard
# --------------------------------------------------------------------------- #


def test_leak_is_real_without_guard() -> None:
    """Without the guard, ranking the forward label DOES materialise an
    ``*_xrank`` feature column — proves the guard is non-vacuous."""
    enriched = build_core_ta_factors(_panel())
    ranked = rank_cross_sectional(
        enriched, feature_cols=["momentum_12m_excl_1m"], normalize_to="symmetric"
    )
    assert "momentum_12m_excl_1m_xrank" in ranked.columns


def test_guard_prevents_forward_xrank_column() -> None:
    """With the guard applied first, no forward-looking ``*_xrank`` column is
    created even though the forward column is present in the panel."""
    enriched = build_core_ta_factors(_panel())
    requested = ["trend_strength_20", "momentum_12m_excl_1m"]
    safe_cols = _strip_forward_label_cols(requested)
    ranked = rank_cross_sectional(
        enriched, feature_cols=safe_cols, normalize_to="symmetric"
    )
    assert "momentum_12m_excl_1m_xrank" not in ranked.columns
    assert "trend_strength_20_xrank" in ranked.columns
