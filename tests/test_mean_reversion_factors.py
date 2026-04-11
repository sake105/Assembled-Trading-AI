"""Tests for mean-reversion factor sidecar (plan item B2.1).

Deterministic fixtures only — no hypothesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.mean_reversion_factors import (
    compute_mean_reversion_factors,
)

pytestmark = pytest.mark.phase12

FACTOR_COLS = [
    "mr_zscore_reversal_3d",
    "mr_rsi_extreme_uptrend",
    "mr_bollinger_squeeze_break",
]


def _make_prices(close: np.ndarray, symbol: str = "AAA", start: str = "2020-01-01") -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=len(close), freq="D")
    return pd.DataFrame(
        {"timestamp": ts, "symbol": symbol, "close": close.astype(float)}
    )


def _linear(n: int, start: float = 100.0, step: float = 0.5) -> np.ndarray:
    return start + step * np.arange(n, dtype=float)


# ---------------------------------------------------------------------------
# 1. Insufficient history
# ---------------------------------------------------------------------------
def test_insufficient_history_all_nan():
    df = _make_prices(_linear(10))
    out = compute_mean_reversion_factors(df)
    assert len(out) == 10
    # The two rolling(60)-dependent factors must be fully NaN with 10 rows.
    assert out["mr_zscore_reversal_3d"].isna().all()
    assert out["mr_bollinger_squeeze_break"].isna().all()
    # The RSI-gated factor uses only EMAs, so it may produce 0.0 when
    # uptrend_flag is False. It must never produce a positive signal here
    # because 10 rows of a flat-ish series cannot give RSI<30 inside a
    # genuine uptrend. Accept NaN or 0.
    rsi_vals = out["mr_rsi_extreme_uptrend"]
    assert ((rsi_vals.isna()) | (rsi_vals == 0.0)).all()


# ---------------------------------------------------------------------------
# 2. / 3. / 4. zscore_reversal_3d sign + clipping
# ---------------------------------------------------------------------------
def test_zscore_reversal_crash_gives_positive_signal():
    close = _linear(80, start=100.0, step=0.1)
    # Sharp crash at the end
    close[-1] = close[-4] * 0.8
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    val = out["mr_zscore_reversal_3d"].iloc[-1]
    assert not np.isnan(val)
    assert val > 0.0, f"expected positive signal after crash, got {val}"


def test_zscore_reversal_rally_gives_negative_signal():
    close = _linear(80, start=100.0, step=0.1)
    # Sharp rally at the end
    close[-1] = close[-4] * 1.2
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    val = out["mr_zscore_reversal_3d"].iloc[-1]
    assert not np.isnan(val)
    assert val < 0.0, f"expected negative signal after rally, got {val}"


def test_zscore_reversal_clipped_to_pm_3():
    close = _linear(80, start=100.0, step=0.05)
    # Cataclysmic crash
    close[-1] = close[-4] * 0.1
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    assert (out["mr_zscore_reversal_3d"].dropna() <= 3.0 + 1e-9).all()
    assert (out["mr_zscore_reversal_3d"].dropna() >= -3.0 - 1e-9).all()
    # And the tail should sit at or very near the clip
    assert out["mr_zscore_reversal_3d"].iloc[-1] >= 2.9


# ---------------------------------------------------------------------------
# 5. / 6. RSI extreme uptrend gating
# ---------------------------------------------------------------------------
def test_rsi_extreme_oversold_in_uptrend_positive():
    # Long gentle uptrend so EMA50 > EMA200, then a short sharp dip.
    n = 260
    close = _linear(n, start=100.0, step=0.4)
    # Inject a sharp multi-day drop near the end to push RSI<30
    for i in range(1, 10):
        close[-i] = close[-11] * (1.0 - 0.04 * (10 - i))
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    tail = out["mr_rsi_extreme_uptrend"].iloc[-1]
    assert not np.isnan(tail)
    assert tail > 0.0, f"expected oversold-in-uptrend > 0, got {tail}"


def test_rsi_extreme_oversold_in_downtrend_zero():
    # Gentle downtrend so EMA50 < EMA200.
    n = 260
    close = _linear(n, start=200.0, step=-0.4)
    # Sharp additional drop at the end → RSI<30
    for i in range(1, 10):
        close[-i] = close[-11] * (1.0 - 0.04 * (10 - i))
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    tail = out["mr_rsi_extreme_uptrend"].iloc[-1]
    # downtrend flag → 0 regardless of RSI level
    assert tail == 0.0, f"expected 0 in downtrend, got {tail}"


# ---------------------------------------------------------------------------
# 7. / 8. Bollinger squeeze break band position
# ---------------------------------------------------------------------------
def test_bollinger_lower_band_in_squeeze_positive():
    # Very tight range for a long time → squeeze; final bar tags lower band.
    n = 80
    rng = np.random.default_rng(0)
    close = 100.0 + rng.normal(0.0, 0.05, size=n)
    # Force a clean close at ~lower band on the last bar
    close[-1] = 100.0 - 0.3
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    val = out["mr_bollinger_squeeze_break"].iloc[-1]
    assert not np.isnan(val)
    assert val > 0.0


def test_bollinger_upper_band_near_zero():
    n = 80
    rng = np.random.default_rng(1)
    close = 100.0 + rng.normal(0.0, 0.05, size=n)
    close[-1] = 100.0 + 0.3  # tag upper band
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    val = out["mr_bollinger_squeeze_break"].iloc[-1]
    assert not np.isnan(val)
    assert val < 0.05, f"expected ~0 at upper band, got {val}"


# ---------------------------------------------------------------------------
# 9. Multi-symbol independence
# ---------------------------------------------------------------------------
def test_multi_symbol_no_cross_contamination():
    close_a = _linear(80, start=100.0, step=0.1)
    close_b = _linear(80, start=50.0, step=-0.1)
    close_a[-1] = close_a[-4] * 0.8  # crash A only
    df_a = _make_prices(close_a, symbol="AAA")
    df_b = _make_prices(close_b, symbol="BBB")
    combined = pd.concat([df_a, df_b], ignore_index=True)

    out = compute_mean_reversion_factors(combined)

    out_a = out[out["symbol"] == "AAA"].reset_index(drop=True)

    # Reference: compute A alone
    ref_a = compute_mean_reversion_factors(df_a).reset_index(drop=True)
    for col in FACTOR_COLS:
        pd.testing.assert_series_equal(
            out_a[col].reset_index(drop=True),
            ref_a[col].reset_index(drop=True),
            check_names=False,
        )

    # A should show crash signal (cross-contamination would dilute this).
    assert out_a["mr_zscore_reversal_3d"].iloc[-1] > 0.0


# ---------------------------------------------------------------------------
# 10. Idempotent
# ---------------------------------------------------------------------------
def test_idempotent_repeated_call():
    close = _linear(120, start=100.0, step=0.2)
    close[-1] *= 0.9
    df = _make_prices(close)
    out1 = compute_mean_reversion_factors(df)
    out2 = compute_mean_reversion_factors(df)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# 11. Output schema
# ---------------------------------------------------------------------------
def test_output_schema_and_timestamps_preserved():
    close = _linear(70, start=100.0)
    df = _make_prices(close)
    out = compute_mean_reversion_factors(df)
    assert list(out.columns) == ["timestamp", "symbol", *FACTOR_COLS]
    assert len(out) == len(df)
    # Timestamps preserved (though possibly re-sorted by symbol/timestamp)
    assert set(out["timestamp"]) == set(df["timestamp"])


# ---------------------------------------------------------------------------
# 12. No raise on all-NaN input
# ---------------------------------------------------------------------------
def test_all_nan_close_no_raise():
    ts = pd.date_range(start="2020-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {"timestamp": ts, "symbol": "AAA", "close": np.full(80, np.nan)}
    )
    out = compute_mean_reversion_factors(df)
    assert len(out) == 80
    for col in FACTOR_COLS:
        assert out[col].isna().all()
