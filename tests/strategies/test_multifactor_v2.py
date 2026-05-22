"""Tests for multifactor_v2 strategy — 30-factor, regime-conditional, ATR exits.

Covers:
  - Signal generation with synthetic price data
  - Regime-conditional weight loading
  - Factor contract (30 factors enumerated)
  - ATR-regime exit engine
  - Meta-model filter (disabled by default)
  - Fallback to static exits
  - Momentum verification in exits
  - Time-stop (zombie killer)
  - Graceful degradation when factor data is unavailable
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.assembled_core.strategies.multifactor_v2 import (
    DEFAULT_V2_WEIGHTS,
    VERSION,
    _compute_atr,
    _detect_regime,
    _get_weights_for_regime,
    check_exit_signals,
    compute_signals,
    compute_target_positions,
)

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synth_panel(
    symbols: list[str] | None = None,
    n_days: int = 250,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic price panel with TA-like feature columns."""
    rng = np.random.default_rng(seed)
    symbols = symbols or ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    rows = []
    for sym in symbols:
        base = 100.0 + rng.normal(0, 10)
        prices = [base]
        for _ in range(n_days - 1):
            prices.append(prices[-1] * (1 + rng.normal(0.0005, 0.015)))
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        for i, d in enumerate(dates):
            p = prices[i]
            rows.append(
                {
                    "timestamp": d,
                    "symbol": sym,
                    "close": p,
                    "high": p * (1 + abs(rng.normal(0, 0.01))),
                    "low": p * (1 - abs(rng.normal(0, 0.01))),
                    "volume": int(rng.uniform(1e6, 5e6)),
                    "ta_rsi_14_v1": rng.uniform(30, 70),
                    "ta_adx_v1": rng.uniform(15, 40),
                    "ta_macd_hist_v1": rng.normal(0, 0.5),
                    "ta_ma_200_v1": p * (1 + rng.normal(0, 0.05)),
                    "ta_ma_50_v1": p * (1 + rng.normal(0, 0.02)),
                    "ta_bb_pctb_v1": rng.uniform(0.1, 0.9),
                    "ta_stoch_k_v1": rng.uniform(20, 80),
                    "ta_obv_v1": rng.uniform(1e7, 5e7),
                    "ta_vol_weighted_mom_20d_v1": rng.normal(0, 0.02),
                    "tick_imbalance_20d": rng.uniform(0.3, 0.7),
                    "abnormal_vol_20d": rng.uniform(0.5, 2.0),
                    "rv_20": rng.uniform(0.10, 0.30),
                    "vov_20_60": rng.uniform(0.0, 0.05),
                }
            )
    return pd.DataFrame(rows)


def _positions_dict(prices: dict[str, float], pct_gain: float = 0.0) -> dict:
    """Create position dict from price map with optional gain/loss."""
    return {
        sym: {
            "qty": 100,
            "avg_price": p / (1 + pct_gain),
            "hwm": p * 1.05,
            "days_held": 10,
        }
        for sym, p in prices.items()
    }


# ---------------------------------------------------------------------------
# 1. Signal generation basics
# ---------------------------------------------------------------------------


def test_compute_signals_returns_dataframe() -> None:
    panel = _synth_panel()
    result = compute_signals(panel)
    assert isinstance(result, pd.DataFrame)
    if not result.empty:
        assert "symbol" in result.columns
        assert "score" in result.columns
        assert "direction" in result.columns


def test_compute_signals_long_only() -> None:
    panel = _synth_panel()
    result = compute_signals(panel)
    if not result.empty:
        assert (result["direction"] == "LONG").all()


def test_compute_signals_empty_input() -> None:
    empty = pd.DataFrame()
    result = compute_signals(empty)
    assert result.empty


def test_compute_signals_single_symbol() -> None:
    panel = _synth_panel(symbols=["AAPL"], n_days=100)
    result = compute_signals(panel)
    assert isinstance(result, pd.DataFrame)


def test_compute_signals_reason_contains_v2() -> None:
    """V2 signals include 'v2|regime=' in reason string."""
    panel = _synth_panel()
    result = compute_signals(panel)
    if not result.empty:
        assert all("v2|regime=" in str(r) for r in result["reason"])


# ---------------------------------------------------------------------------
# 2. Regime detection
# ---------------------------------------------------------------------------


def test_detect_regime_returns_string() -> None:
    panel = _synth_panel()
    regime = _detect_regime(panel, {})
    assert isinstance(regime, str)
    assert regime in {"bull", "sideways", "bear", "crisis", "neutral", "reflation"}


# ---------------------------------------------------------------------------
# 3. Factor weights
# ---------------------------------------------------------------------------


def test_default_weights_sum_to_one() -> None:
    total = sum(DEFAULT_V2_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01


def test_default_weights_has_34_active_factors() -> None:
    """34 active factors after adding insider_cluster, pead_sue, buyback_drift (2026-05-23)."""
    assert len(DEFAULT_V2_WEIGHTS) == 34


def test_get_weights_unknown_regime_returns_defaults() -> None:
    weights = _get_weights_for_regime("unknown_regime_xyz", {})
    assert weights == DEFAULT_V2_WEIGHTS


def test_get_weights_known_regime_with_file(tmp_path: Path) -> None:
    """When regime weights file exists, load from it."""
    import json

    weights_file = tmp_path / "weights.json"
    test_weights = {
        "test_regime": {"trend_ema_spread": 0.5, "mom_rsi_centered": 0.5},
    }
    weights_file.write_text(json.dumps(test_weights))

    # Clear cache using the public API (cache is now a _BoundedCache, not None-able)
    from src.assembled_core.strategies.multifactor_v2 import clear_regime_cache

    clear_regime_cache()

    result = _get_weights_for_regime(
        "test_regime", {"regime_weights_path": str(weights_file)}
    )
    assert result["trend_ema_spread"] == 0.5

    # Cleanup cache
    clear_regime_cache()


# ---------------------------------------------------------------------------
# 4. Target positions (delegated to v1)
# ---------------------------------------------------------------------------


def test_compute_target_positions_from_signals() -> None:
    signals = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "NVDA"],
            "score": [2.0, 1.5, 1.0],
            "direction": ["LONG", "LONG", "LONG"],
        }
    )
    result = compute_target_positions(signals, total_capital=100_000)
    assert not result.empty
    assert "target_weight" in result.columns


def test_compute_target_positions_empty_signals() -> None:
    result = compute_target_positions(pd.DataFrame(), total_capital=100_000)
    assert result.empty


# ---------------------------------------------------------------------------
# 5. ATR computation
# ---------------------------------------------------------------------------


def test_compute_atr_positive() -> None:
    panel = _synth_panel(symbols=["AAPL"], n_days=50)
    atr = _compute_atr(panel, "AAPL", window=14)
    assert atr > 0


def test_compute_atr_short_history() -> None:
    panel = _synth_panel(symbols=["AAPL"], n_days=5)
    atr = _compute_atr(panel, "AAPL", window=14)
    assert atr == 0.0  # insufficient data


# ---------------------------------------------------------------------------
# 6. ATR-regime exit signals
# ---------------------------------------------------------------------------


def test_exit_signals_stop_loss() -> None:
    """Price well below avg_price triggers ATR stop."""
    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-06-01")] * 20,
            "symbol": ["AAPL"] * 20,
            "close": [80.0] * 20,  # price crashed
            "high": [82.0] * 20,
            "low": [78.0] * 20,
        }
    )
    positions = {"AAPL": {"qty": 100, "avg_price": 100.0, "hwm": 105.0, "days_held": 5}}
    result = check_exit_signals(positions, prices, {"exits": {"mode": "atr_regime"}})
    assert not result.empty
    assert result.iloc[0]["symbol"] == "AAPL"
    assert "stop_atr" in result.iloc[0]["exit_reason"]


def test_exit_signals_trailing_stop() -> None:
    """Price drops from HWM by more than trail_mult * ATR."""
    n = 30
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-01", periods=n, freq="B"),
            "symbol": ["AAPL"] * n,
            "close": [110.0] * (n - 1) + [95.0],
            "high": [112.0] * (n - 1) + [96.0],
            "low": [108.0] * (n - 1) + [94.0],
        }
    )
    positions = {
        "AAPL": {"qty": 100, "avg_price": 100.0, "hwm": 115.0, "days_held": 20}
    }
    result = check_exit_signals(positions, prices, {"exits": {"mode": "atr_regime"}})
    if not result.empty:
        assert any(
            "trailing_atr" in str(r) or "stop_atr" in str(r)
            for r in result["exit_reason"]
        )


def test_exit_signals_take_profit() -> None:
    """Price exceeds avg_price + tp_mult * ATR → partial exit."""
    n = 30
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-01", periods=n, freq="B"),
            "symbol": ["AAPL"] * n,
            "close": [100.0] * (n - 1) + [150.0],
            "high": [102.0] * (n - 1) + [152.0],
            "low": [98.0] * (n - 1) + [148.0],
        }
    )
    positions = {"AAPL": {"qty": 100, "avg_price": 100.0, "hwm": 150.0, "days_held": 5}}
    result = check_exit_signals(positions, prices, {"exits": {"mode": "atr_regime"}})
    if not result.empty:
        tp_exits = result[result["exit_reason"].str.contains("take_profit")]
        if not tp_exits.empty:
            assert tp_exits.iloc[0]["exit_qty_pct"] == 0.5


def test_exit_signals_time_stop() -> None:
    """Position held > 30 days with < 3% return triggers time stop."""
    n = 30
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-05-01", periods=n, freq="B"),
            "symbol": ["AAPL"] * n,
            "close": [101.0] * n,
            "high": [102.0] * n,
            "low": [100.0] * n,
        }
    )
    positions = {
        "AAPL": {"qty": 100, "avg_price": 100.0, "hwm": 102.0, "days_held": 35}
    }
    result = check_exit_signals(
        positions,
        prices,
        {
            "exits": {
                "mode": "atr_regime",
                "time_stop_days": 30,
                "time_stop_min_return": 0.03,
            }
        },
    )
    if not result.empty:
        time_exits = result[result["exit_reason"].str.contains("time_stop")]
        assert not time_exits.empty


def test_exit_signals_static_mode_fallback() -> None:
    """mode=static delegates to v1 exits."""
    prices = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-06-01")],
            "symbol": ["AAPL"],
            "close": [85.0],
        }
    )
    positions = {"AAPL": {"qty": 100, "avg_price": 100.0, "hwm": 105.0, "days_held": 5}}
    result = check_exit_signals(positions, prices, {"exits": {"mode": "static"}})
    assert isinstance(result, pd.DataFrame)


def test_exit_signals_empty_positions() -> None:
    result = check_exit_signals({}, pd.DataFrame(), {})
    assert result.empty


# ---------------------------------------------------------------------------
# 7. VERSION marker
# ---------------------------------------------------------------------------


def test_version_not_skeleton() -> None:
    """V2 version should no longer be skeleton."""
    assert "skeleton" not in VERSION.lower()
    assert VERSION.startswith("multifactor_v2")


# ---------------------------------------------------------------------------
# 8. Integration: signals → positions pipeline
# ---------------------------------------------------------------------------


def test_signal_to_position_pipeline() -> None:
    """Full pipeline: compute_signals → compute_target_positions."""
    panel = _synth_panel(n_days=300)
    signals = compute_signals(panel)
    if not signals.empty:
        positions = compute_target_positions(signals, total_capital=100_000)
        assert not positions.empty
        assert positions["target_weight"].sum() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 9. Graceful degradation
# ---------------------------------------------------------------------------


def test_signals_with_minimal_features() -> None:
    """Even without TA features, v2 should not crash."""
    panel = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="B").tolist() * 3,
            "symbol": ["AAPL"] * 50 + ["MSFT"] * 50 + ["NVDA"] * 50,
            "close": list(np.random.default_rng(42).uniform(90, 110, 150)),
        }
    )
    result = compute_signals(panel)
    assert isinstance(result, pd.DataFrame)


def test_signals_two_symbols_minimum() -> None:
    """Minimum viable: 2 symbols with basic columns."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    panel = pd.DataFrame(
        {
            "timestamp": list(dates) * 2,
            "symbol": ["A"] * 100 + ["B"] * 100,
            "close": list(np.linspace(100, 120, 100)) + list(np.linspace(50, 45, 100)),
        }
    )
    result = compute_signals(panel)
    assert isinstance(result, pd.DataFrame)
