"""Point-in-time safety tests for vol_target_overlay strategy.

Core claim: vol-target weights at timestamp T must be identical whether computed
from a price panel ending at T or from a panel that also includes bars after T
(even with extreme manipulated values).

All signal computations use strictly causal rolling windows:
  - pct_change(): uses close[i] / close[i-1] — no future data
  - rolling(window, min_periods=window).std(): backward-looking window only
  - rolling(window, min_periods=window).mean(): backward-looking window only

These tests verify that no non-causal computation is inadvertently introduced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.assembled_core.strategies.vol_target_overlay import (
    compute_signals,
    generate_vol_target_signals_from_prices,
)

# ---------------------------------------------------------------------------
# Parameters — use short windows so tests run on a compact synthetic panel
# ---------------------------------------------------------------------------
_SMA = 50
_VOL_LB = 10
_N_BARS = _SMA + 120  # 170 bars: 50 warmup + 120 valid signal bars


def _make_spy_prices(n_bars: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic SPY-only price panel (long format)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_bars, freq="B", tz="UTC")
    returns = rng.standard_normal(n_bars) * 0.012 + 0.0003
    close = 300.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame({"timestamp": dates, "symbol": "SPY", "close": close})


def _pick_asof(sigs: pd.DataFrame, idx: int = -1) -> pd.Timestamp:
    """Pick a signal timestamp at ``idx`` (default = last valid bar)."""
    sig_ts = sorted(sigs["timestamp"].unique())
    assert len(sig_ts) >= 10, f"Need ≥10 signal bars; got {len(sig_ts)}"
    return sig_ts[idx]


# ---------------------------------------------------------------------------
# Test 1 — core PIT: future bars ×5 must not change weight at as_of
# ---------------------------------------------------------------------------


def test_pit_no_lookahead_future_multiplied():
    """Weight at as_of bar is identical whether future bars are ×5 or unchanged."""
    prices = _make_spy_prices(_N_BARS)

    # Step 1: baseline — pick as_of from ACTUAL signal timestamps (middle bar)
    baseline = generate_vol_target_signals_from_prices(
        prices, vol_lookback=_VOL_LB, sma_window=_SMA
    )
    assert not baseline.empty, "No baseline signals generated"
    as_of_ts = _pick_asof(baseline, len(sorted(baseline["timestamp"].unique())) // 2)

    baseline_at = baseline[baseline["timestamp"] == as_of_ts].set_index("symbol")
    assert "SPY" in baseline_at.index, f"No SPY signal at {as_of_ts}"
    assert "IEF" in baseline_at.index, f"No IEF signal at {as_of_ts}"

    # Step 2: manipulate future bars
    prices_manip = prices.copy()
    future_mask = prices_manip["timestamp"] > as_of_ts
    assert future_mask.sum() > 0, "No future bars to manipulate — pick an earlier as_of"
    prices_manip.loc[future_mask, "close"] *= 5.0

    # Step 3: re-compute signals
    modified = generate_vol_target_signals_from_prices(
        prices_manip, vol_lookback=_VOL_LB, sma_window=_SMA
    )
    modified_at = modified[modified["timestamp"] == as_of_ts].set_index("symbol")

    # Step 4: scores at as_of must be identical
    for sym in ["SPY", "IEF"]:
        b = float(baseline_at.loc[sym, "score"])
        m = float(modified_at.loc[sym, "score"])
        assert abs(b - m) < 1e-10, (
            f"{sym}: score changed from {b:.10f} to {m:.10f} "
            "when future bars ×5 — look-ahead suspected"
        )
        assert baseline_at.loc[sym, "direction"] == modified_at.loc[sym, "direction"]


# ---------------------------------------------------------------------------
# Test 2 — PIT: future bars zeroed
# ---------------------------------------------------------------------------


def test_pit_no_lookahead_future_zeroed():
    """Weight at as_of bar unchanged when future bars are set to near-zero."""
    prices = _make_spy_prices(_N_BARS, seed=77)

    baseline = generate_vol_target_signals_from_prices(
        prices, vol_lookback=_VOL_LB, sma_window=_SMA
    )
    assert not baseline.empty
    as_of_ts = _pick_asof(baseline, len(sorted(baseline["timestamp"].unique())) // 3)

    baseline_at = baseline[baseline["timestamp"] == as_of_ts].set_index("symbol")

    prices_zero = prices.copy()
    prices_zero.loc[prices_zero["timestamp"] > as_of_ts, "close"] = 0.01

    modified = generate_vol_target_signals_from_prices(
        prices_zero, vol_lookback=_VOL_LB, sma_window=_SMA
    )
    modified_at = modified[modified["timestamp"] == as_of_ts].set_index("symbol")

    for sym in ["SPY", "IEF"]:
        b = float(baseline_at.loc[sym, "score"])
        m = float(modified_at.loc[sym, "score"])
        assert abs(b - m) < 1e-10, (
            f"{sym}: score changed {b:.10f} → {m:.10f} under zero-future manipulation"
        )


# ---------------------------------------------------------------------------
# Test 3 — classical PIT: signal from prices[:T] == signal at T from prices[:T+N]
# ---------------------------------------------------------------------------


def test_pit_slice_equals_full_panel_at_asof():
    """Weight at bar T from a T-bar slice equals weight at bar T from a longer panel.

    This is the canonical PIT check: if rolling MA or std uses any future data
    (e.g. center=True or expanding windows), this test fails.
    """
    prices = _make_spy_prices(_N_BARS)

    # Generate from full panel to find a valid as_of timestamp
    sig_full = generate_vol_target_signals_from_prices(
        prices, vol_lookback=_VOL_LB, sma_window=_SMA
    )
    assert not sig_full.empty
    as_of_ts = _pick_asof(sig_full, len(sorted(sig_full["timestamp"].unique())) // 2)

    # Slice prices to as_of
    prices_slice = prices[prices["timestamp"] <= as_of_ts].copy()

    # Signal from the slice (as_of is the latest bar)
    sig_slice = generate_vol_target_signals_from_prices(
        prices_slice, vol_lookback=_VOL_LB, sma_window=_SMA
    )
    slice_at = sig_slice[sig_slice["timestamp"] == as_of_ts].set_index("symbol")
    assert "SPY" in slice_at.index, "No SPY signal in slice at as_of"

    # Signal from the full panel at as_of
    full_at = sig_full[sig_full["timestamp"] == as_of_ts].set_index("symbol")
    assert "SPY" in full_at.index

    assert set(slice_at.index) == set(full_at.index), (
        "Symbol sets differ between slice and full panel"
    )
    for sym in ["SPY", "IEF"]:
        diff = abs(float(slice_at.loc[sym, "score"]) - float(full_at.loc[sym, "score"]))
        assert diff < 1e-10, (
            f"{sym}: score from slice={float(slice_at.loc[sym, 'score']):.10f} "
            f"vs full={float(full_at.loc[sym, 'score']):.10f} — rolling is not causal"
        )


# ---------------------------------------------------------------------------
# Test 4 — determinism: two calls with identical input must yield identical output
# ---------------------------------------------------------------------------


def test_compute_signals_deterministic():
    """compute_signals must return identical output on two calls with the same input."""
    prices = _make_spy_prices(_N_BARS)

    result_a = compute_signals(prices, vol_lookback=_VOL_LB, sma_window=_SMA)
    result_b = compute_signals(prices, vol_lookback=_VOL_LB, sma_window=_SMA)

    assert list(result_a.columns) == list(result_b.columns)
    pd.testing.assert_frame_equal(
        result_a.reset_index(drop=True),
        result_b.reset_index(drop=True),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Test 5 — output schema and value constraints
# ---------------------------------------------------------------------------


def test_output_schema_and_value_constraints():
    """Weights must sum to 1.0, be in [0,1], direction always LONG."""
    prices = _make_spy_prices(_N_BARS)

    sigs = generate_vol_target_signals_from_prices(
        prices, vol_lookback=_VOL_LB, sma_window=_SMA
    )

    assert not sigs.empty, "No signals produced — check warmup arithmetic"
    assert set(sigs.columns) >= {"timestamp", "symbol", "direction", "score"}
    assert sigs["direction"].eq("LONG").all(), "direction must always be LONG"
    assert sigs["score"].between(0.0, 1.0 + 1e-9).all(), "scores must be in [0, 1]"
    assert not sigs["score"].isna().any(), "NaN scores in output"

    # Weights must sum to 1.0 at every timestamp
    by_ts = sigs.groupby("timestamp")["score"].sum()
    assert (by_ts - 1.0).abs().max() < 1e-9, (
        f"Weights do not sum to 1.0; max deviation: {(by_ts - 1.0).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Test 6 — trend filter: below-SMA halves SPY weight
# ---------------------------------------------------------------------------


def _make_spy_with_guaranteed_below_sma(sma: int, vol_lb: int) -> pd.DataFrame:
    """Price series with a guaranteed below-SMA block after warmup.

    Structure: rise 10% over sma+vol_lb bars (warmup),
    then crash 25% in a single step, then flat (below SMA for many bars).
    """
    n_warmup = sma + vol_lb + 10
    dates = pd.date_range("2020-01-02", periods=n_warmup + 60, freq="B", tz="UTC")
    # Rising phase: prices drift up gently
    prices_up = 300.0 * (1.0 + 0.001) ** np.arange(n_warmup)
    # Crash: single -25% drop, then flat
    crash_price = prices_up[-1] * 0.75
    prices_flat = np.full(60, crash_price)
    close = np.concatenate([prices_up, prices_flat])
    return pd.DataFrame({"timestamp": dates, "symbol": "SPY", "close": close})


def test_trend_filter_halves_spy_weight():
    """When SPY close < SMA at a given bar, SPY weight equals 0.5 × unconstrained weight.

    Uses a deterministic fixture (rise → crash → flat) to guarantee below-SMA bars
    are always present — no random seed, no pytest.skip.
    """
    prices = _make_spy_with_guaranteed_below_sma(_SMA, _VOL_LB)

    sigs = generate_vol_target_signals_from_prices(
        prices, vol_lookback=_VOL_LB, sma_window=_SMA, target_vol=0.12
    )
    assert not sigs.empty, "No signals generated — check warmup arithmetic"

    # Reproduce indicator computation on the same prices
    spy = prices.sort_values("timestamp").copy().reset_index(drop=True)
    spy["_ret"] = spy["close"].pct_change()
    spy["_rvol"] = spy["_ret"].rolling(_VOL_LB, min_periods=_VOL_LB).std() * np.sqrt(
        252
    )
    spy["_sma"] = spy["close"].rolling(_SMA, min_periods=_SMA).mean()
    spy["_raw_w"] = np.minimum(1.0, 0.12 / spy["_rvol"].clip(lower=1e-9))
    spy["_expected_w"] = np.where(
        spy["close"] < spy["_sma"],
        spy["_raw_w"] * 0.5,
        spy["_raw_w"],
    )

    # Find valid below-SMA bars that also have signals
    sig_ts_set = set(sigs["timestamp"].unique())
    valid_spy = spy.dropna(subset=["_rvol", "_sma"])
    below_sma_rows = valid_spy[
        (valid_spy["close"] < valid_spy["_sma"])
        & valid_spy["timestamp"].isin(sig_ts_set)
    ]

    assert not below_sma_rows.empty, (
        "Deterministic fixture produced no below-SMA bars — fixture construction error"
    )

    # Check first below-SMA row
    row = below_sma_rows.iloc[0]
    ts_below = row["timestamp"]
    expected_w = float(row["_expected_w"])

    spy_at = sigs[(sigs["timestamp"] == ts_below) & (sigs["symbol"] == "SPY")]
    assert not spy_at.empty, f"No SPY signal at below-SMA bar {ts_below}"

    actual_w = float(spy_at["score"].iloc[0])
    assert abs(actual_w - expected_w) < 1e-9, (
        f"Trend filter: expected SPY weight {expected_w:.8f}, got {actual_w:.8f} at {ts_below}"
    )
