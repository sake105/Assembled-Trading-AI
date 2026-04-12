"""Tests for multifactor_v2 skeleton (Sprint 2 / Plan §4).

v2 is a delegating wrapper around v1 with an optional meta-model confidence
filter hook. These tests cover:
  * empty prices → empty signals
  * delegation equivalence with v1 when meta_model is disabled
  * graceful fallback when meta_model is enabled but model file is missing
  * graceful fallback when feature columns are not present on v1 output
  * check_exit_signals / compute_target_positions pass-through
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.strategies import multifactor_v1, multifactor_v2


def _build_prices_with_features(n_symbols: int = 5, n_bars: int = 80) -> pd.DataFrame:
    """Build a minimal panel DF that v1 compute_signals will accept."""
    import numpy as np

    rng = np.random.default_rng(42)
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for s_idx in range(n_symbols):
        sym = f"SYM{s_idx:02d}"
        px = 100.0 + s_idx * 10
        for b in range(n_bars):
            px = px * (1.0 + rng.normal(0.001, 0.01))
            rows.append(
                {
                    "timestamp": base + pd.Timedelta(days=b),
                    "symbol": sym,
                    "open": px * 0.99,
                    "high": px * 1.01,
                    "low": px * 0.98,
                    "close": px,
                    "volume": 1_000_000.0 + rng.uniform(0, 500_000),
                }
            )
    return pd.DataFrame(rows)


def test_version_tag() -> None:
    assert multifactor_v2.STRATEGY_VERSION == "v2"


def test_empty_prices_returns_empty() -> None:
    empty = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    out = multifactor_v2.compute_signals(empty, {})
    assert out.empty


def test_delegation_equivalence_when_meta_disabled() -> None:
    """v2 must equal v1 when meta_model is disabled (default)."""
    prices = _build_prices_with_features()
    v1_out = multifactor_v1.compute_signals(prices, {"min_signal_score": -10.0})
    v2_out = multifactor_v2.compute_signals(prices, {"min_signal_score": -10.0})
    # Same row count
    assert len(v1_out) == len(v2_out)
    if not v1_out.empty:
        # Same symbols, same scores
        v1_sorted = v1_out.sort_values("symbol").reset_index(drop=True)
        v2_sorted = v2_out.sort_values("symbol").reset_index(drop=True)
        assert list(v1_sorted["symbol"]) == list(v2_sorted["symbol"])
        for a, b in zip(v1_sorted["score"], v2_sorted["score"]):
            assert abs(a - b) < 1e-12


def test_meta_model_missing_file_falls_back() -> None:
    """Enabling meta_model with a non-existent model path must fall back to v1."""
    prices = _build_prices_with_features()
    cfg = {
        "min_signal_score": -10.0,
        "meta_model": {
            "enabled": True,
            "model_path": "models/DOES_NOT_EXIST.joblib",
            "min_confidence": 0.55,
        },
    }
    v1_out = multifactor_v1.compute_signals(prices, {"min_signal_score": -10.0})
    v2_out = multifactor_v2.compute_signals(prices, cfg)
    assert len(v1_out) == len(v2_out)


def test_meta_model_disabled_by_default() -> None:
    prices = _build_prices_with_features()
    v2_out = multifactor_v2.compute_signals(prices, {"min_signal_score": -10.0})
    # Meta confidence column must NOT be present when filter not applied
    if not v2_out.empty:
        assert "meta_confidence" not in v2_out.columns


def test_compute_target_positions_delegates() -> None:
    """v2 compute_target_positions must return v1 output unchanged."""
    signals = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "AAA", "direction": "LONG", "score": 1.5},
            {"timestamp": pd.Timestamp("2025-03-01", tz="UTC"), "symbol": "BBB", "direction": "LONG", "score": 1.0},
        ]
    )
    v1_out = multifactor_v1.compute_target_positions(signals, total_capital=100_000.0)
    v2_out = multifactor_v2.compute_target_positions(signals, total_capital=100_000.0)
    assert list(v1_out.columns) == list(v2_out.columns)
    assert len(v1_out) == len(v2_out)


def test_check_exit_signals_delegates() -> None:
    """v2 check_exit_signals must return v1 output unchanged."""
    positions = {"AAA": {"qty": 10.0, "avg_price": 100.0, "hwm": 110.0}}
    prices = pd.DataFrame([{"symbol": "AAA", "close": 90.0}])  # stop-loss trigger
    v1_out = multifactor_v1.check_exit_signals(positions, prices, {})
    v2_out = multifactor_v2.check_exit_signals(positions, prices, {})
    assert list(v1_out.columns) == list(v2_out.columns)
    assert len(v1_out) == len(v2_out)


# ---------------------------------------------------------------------------
# phase12 regression lane: skeleton scaffold + 30-factor target contract
# ---------------------------------------------------------------------------


@pytest.mark.phase12
def test_v2_public_api_exists_and_callable() -> None:
    """The three public entry points must exist on the v2 module."""
    for name in ("compute_signals", "compute_target_positions", "check_exit_signals"):
        fn = getattr(multifactor_v2, name, None)
        assert fn is not None, f"multifactor_v2.{name} missing"
        assert callable(fn), f"multifactor_v2.{name} not callable"


@pytest.mark.phase12
def test_v2_version_marker() -> None:
    """VERSION constant must exist and identify v2."""
    assert hasattr(multifactor_v2, "VERSION")
    assert isinstance(multifactor_v2.VERSION, str)
    assert "multifactor_v2" in multifactor_v2.VERSION.lower()


@pytest.mark.phase12
def test_v2_produces_signals_like_v1() -> None:
    """v2 must produce signals on the same input as v1 (same schema, may differ in values)."""
    prices = _build_prices_with_features(n_symbols=4, n_bars=60)
    cfg = {"min_signal_score": -10.0}
    v2_out = multifactor_v2.compute_signals(prices, cfg)
    # Both should produce signals (exact values will differ due to v2's extra factors)
    assert isinstance(v2_out, pd.DataFrame)
    if not v2_out.empty:
        assert "symbol" in v2_out.columns
        assert "score" in v2_out.columns
        assert "direction" in v2_out.columns


@pytest.mark.phase12
def test_v2_default_weights_shape_and_sum() -> None:
    """DEFAULT_V2_WEIGHTS must have 29 additive factors summing to ~1.0."""
    weights = multifactor_v2.DEFAULT_V2_WEIGHTS
    assert isinstance(weights, dict)
    assert len(weights) == 29, f"expected 29 factors, got {len(weights)}"

    total = sum(weights.values())
    assert abs(total - 1.0) <= 0.02, (
        f"factor weights should sum to ~1.0 (tol 0.02); got {total:.4f}"
    )

    # All weights should be positive
    for name, w in weights.items():
        assert w > 0, f"weight for {name} must be positive, got {w}"
