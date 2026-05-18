"""PIT (Point-In-Time) property tests for strategy-critical TA features.

Audit §8.7 "Look-Ahead-Bias": the existing ``test_property_fsm_pit.py``
covered only ``rolling_mean`` and ``pct_change``. Strategy-specific features
(RSI, MACD, Bollinger, ATR, log-returns, moving averages) were unpinned.

This module fills that gap. The PIT property is:

    For any PIT-safe feature ``f`` and any prefix length ``k``::

        f(prices[:k]) == f(prices)[:k]

i.e. computing the feature on a prefix and computing it on the full series
then slicing to the same prefix must yield identical values. A leak from
``prices[k:]`` into ``f(prices)[:k]`` is exactly the Look-Ahead-Bias the
audit prohibits.

Markers:
- All tests use ``@pytest.mark.fast`` so they run in the CI fast lane.
- Property tests use Hypothesis (already in dev extras).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.assembled_core.features.ta_features import (
    add_atr,
    add_bollinger_bands,
    add_log_returns,
    add_macd,
    add_moving_averages,
    add_rsi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(prices: list[float]) -> pd.DataFrame:
    """Build a minimal single-symbol DataFrame for the add_* helpers.

    add_* functions expect columns: symbol + price columns. ATR additionally
    needs high/low/close — synthesise from a single price by widening to
    ±0.5%.
    """
    n = len(prices)
    arr = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "symbol": ["TST"] * n,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "open": arr,
            "high": arr * 1.005,
            "low": arr * 0.995,
            "close": arr,
            "volume": np.full(n, 1_000_000.0),
        }
    )


def _assert_pit_safe(
    prices: list[float],
    cut: int,
    add_feature_fn,
    feature_cols: list[str],
) -> None:
    """Generic PIT-safety harness: f(prices[:cut]) == f(prices)[:cut]."""
    df_full = _make_df(prices)
    df_prefix = _make_df(prices[:cut])

    result_full = add_feature_fn(df_full)
    result_prefix = add_feature_fn(df_prefix)

    for col in feature_cols:
        assert col in result_full.columns, f"missing feature column: {col}"
        assert col in result_prefix.columns, f"missing feature column: {col}"
        full_vals = (
            result_full.sort_values("timestamp")[col].iloc[:cut].reset_index(drop=True)
        )
        prefix_vals = result_prefix.sort_values("timestamp")[col].reset_index(drop=True)
        pd.testing.assert_series_equal(
            full_vals,
            prefix_vals,
            check_dtype=False,
            atol=1e-9,
            rtol=0,
            obj=f"PIT violation in {col}",
        )


# ---------------------------------------------------------------------------
# Strategy-specific feature PIT tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestPITLogReturns:
    @given(
        prices=st.lists(
            st.floats(min_value=1.0, max_value=500.0, allow_nan=False),
            min_size=20,
            max_size=80,
        ),
        cut_frac=st.floats(min_value=0.3, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_log_returns_pit_safe(self, prices: list[float], cut_frac: float) -> None:
        cut = max(2, int(len(prices) * cut_frac))
        cut = min(cut, len(prices) - 1)
        _assert_pit_safe(
            prices, cut, add_log_returns, feature_cols=["ta_log_return_v1"]
        )


@pytest.mark.fast
class TestPITMovingAverages:
    @given(
        prices=st.lists(
            st.floats(min_value=10.0, max_value=300.0, allow_nan=False),
            min_size=60,
            max_size=120,
        ),
        cut_frac=st.floats(min_value=0.5, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_moving_averages_pit_safe(
        self, prices: list[float], cut_frac: float
    ) -> None:
        cut = max(30, int(len(prices) * cut_frac))
        cut = min(cut, len(prices) - 1)
        df = _make_df(prices)
        df_prefix = _make_df(prices[:cut])
        full = add_moving_averages(df)
        prefix = add_moving_averages(df_prefix)
        # add_moving_averages emits at least sma_20 / ema_20 namespaced columns
        for col in full.columns:
            if col.startswith(("ta_sma_", "ta_ema_", "sma_", "ema_")):
                full_vals = (
                    full.sort_values("timestamp")[col].iloc[:cut].reset_index(drop=True)
                )
                prefix_vals = prefix.sort_values("timestamp")[col].reset_index(
                    drop=True
                )
                pd.testing.assert_series_equal(
                    full_vals,
                    prefix_vals,
                    check_dtype=False,
                    atol=1e-9,
                    rtol=0,
                    obj=f"PIT violation in {col}",
                )


@pytest.mark.fast
class TestPITATR:
    @given(
        prices=st.lists(
            st.floats(min_value=10.0, max_value=300.0, allow_nan=False),
            min_size=40,
            max_size=100,
        ),
        cut_frac=st.floats(min_value=0.5, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_atr_pit_safe(self, prices: list[float], cut_frac: float) -> None:
        cut = max(20, int(len(prices) * cut_frac))
        cut = min(cut, len(prices) - 1)
        df = _make_df(prices)
        df_prefix = _make_df(prices[:cut])
        full = add_atr(df)
        prefix = add_atr(df_prefix)
        atr_cols = [c for c in full.columns if c.startswith(("ta_atr_", "atr_"))]
        assert atr_cols, "no ATR column emitted"
        for col in atr_cols:
            if col not in prefix.columns:
                continue
            full_vals = (
                full.sort_values("timestamp")[col].iloc[:cut].reset_index(drop=True)
            )
            prefix_vals = prefix.sort_values("timestamp")[col].reset_index(drop=True)
            pd.testing.assert_series_equal(
                full_vals,
                prefix_vals,
                check_dtype=False,
                atol=1e-6,  # Wilder smoothing has stronger init transient
                rtol=0,
                obj=f"PIT violation in {col}",
            )


@pytest.mark.fast
class TestPITRSI:
    @given(
        prices=st.lists(
            st.floats(min_value=10.0, max_value=300.0, allow_nan=False),
            min_size=50,
            max_size=120,
        ),
        cut_frac=st.floats(min_value=0.5, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_rsi_pit_safe(self, prices: list[float], cut_frac: float) -> None:
        cut = max(30, int(len(prices) * cut_frac))
        cut = min(cut, len(prices) - 1)
        df = _make_df(prices)
        df_prefix = _make_df(prices[:cut])
        full = add_rsi(df)
        prefix = add_rsi(df_prefix)
        rsi_cols = [c for c in full.columns if c.startswith(("ta_rsi_", "rsi_"))]
        assert rsi_cols, "no RSI column emitted"
        for col in rsi_cols:
            if col not in prefix.columns:
                continue
            full_vals = (
                full.sort_values("timestamp")[col].iloc[:cut].reset_index(drop=True)
            )
            prefix_vals = prefix.sort_values("timestamp")[col].reset_index(drop=True)
            pd.testing.assert_series_equal(
                full_vals,
                prefix_vals,
                check_dtype=False,
                atol=1e-6,
                rtol=0,
                obj=f"PIT violation in {col}",
            )


@pytest.mark.fast
class TestPITMACD:
    @given(
        prices=st.lists(
            st.floats(min_value=10.0, max_value=300.0, allow_nan=False),
            min_size=60,
            max_size=120,
        ),
        cut_frac=st.floats(min_value=0.5, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_macd_pit_safe(self, prices: list[float], cut_frac: float) -> None:
        cut = max(35, int(len(prices) * cut_frac))
        cut = min(cut, len(prices) - 1)
        df = _make_df(prices)
        df_prefix = _make_df(prices[:cut])
        full = add_macd(df)
        prefix = add_macd(df_prefix)
        macd_cols = [c for c in full.columns if "macd" in c.lower()]
        assert macd_cols, "no MACD column emitted"
        for col in macd_cols:
            if col not in prefix.columns:
                continue
            full_vals = (
                full.sort_values("timestamp")[col].iloc[:cut].reset_index(drop=True)
            )
            prefix_vals = prefix.sort_values("timestamp")[col].reset_index(drop=True)
            pd.testing.assert_series_equal(
                full_vals,
                prefix_vals,
                check_dtype=False,
                atol=1e-6,
                rtol=0,
                obj=f"PIT violation in {col}",
            )


@pytest.mark.fast
class TestPITBollinger:
    @given(
        prices=st.lists(
            st.floats(min_value=10.0, max_value=300.0, allow_nan=False),
            min_size=50,
            max_size=120,
        ),
        cut_frac=st.floats(min_value=0.5, max_value=0.9, allow_nan=False),
    )
    @settings(max_examples=15, suppress_health_check=[HealthCheck.too_slow])
    def test_bollinger_pit_safe(self, prices: list[float], cut_frac: float) -> None:
        cut = max(25, int(len(prices) * cut_frac))
        cut = min(cut, len(prices) - 1)
        df = _make_df(prices)
        df_prefix = _make_df(prices[:cut])
        full = add_bollinger_bands(df)
        prefix = add_bollinger_bands(df_prefix)
        bb_cols = [
            c
            for c in full.columns
            if c.startswith(("ta_bb_", "bb_")) and c in prefix.columns
        ]
        assert bb_cols, "no Bollinger column emitted"
        for col in bb_cols:
            full_vals = (
                full.sort_values("timestamp")[col].iloc[:cut].reset_index(drop=True)
            )
            prefix_vals = prefix.sort_values("timestamp")[col].reset_index(drop=True)
            pd.testing.assert_series_equal(
                full_vals,
                prefix_vals,
                check_dtype=False,
                atol=1e-9,
                rtol=0,
                obj=f"PIT violation in {col}",
            )


# ---------------------------------------------------------------------------
# Negative control: a deliberately leaky feature MUST fail the PIT property
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_pit_negative_control_detects_leakage() -> None:
    """Sanity check: if we construct a leaky feature (uses future values),
    the PIT-safety harness MUST detect the violation. Without this negative
    control, a buggy harness could silently pass everything.
    """
    prices = [100.0, 101.0, 99.0, 102.0, 103.0, 98.0, 105.0, 107.0, 104.0, 110.0]
    df_full = _make_df(prices)
    df_prefix = _make_df(prices[:6])

    # Build a deliberately leaky feature on each frame: use the LAST value
    # of the series as a "constant" — this is the canonical look-ahead.
    df_full = df_full.sort_values("timestamp").copy()
    df_prefix = df_prefix.sort_values("timestamp").copy()
    df_full["leaky"] = df_full["close"].iloc[-1]  # 110.0 for the full series
    df_prefix["leaky"] = df_prefix["close"].iloc[-1]  # 98.0 for the prefix

    # The harness must detect the leak (different "constant" values for
    # the same prefix indexes).
    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(
            df_full["leaky"].iloc[:6].reset_index(drop=True),
            df_prefix["leaky"].reset_index(drop=True),
            check_dtype=False,
            atol=1e-9,
            rtol=0,
        )


@pytest.mark.fast
def test_pit_assert_pit_safe_helper_detects_real_leak() -> None:
    """F-senior-6: end-to-end negative control for the `_assert_pit_safe`
    helper itself. The earlier negative test only verified
    `pd.testing.assert_series_equal` catches differing values — but did not
    exercise `_assert_pit_safe` with a leaky `add_feature_fn`. This test
    plugs a deliberately-leaky feature builder INTO the harness and asserts
    the harness raises. Without this, a future refactor of `_assert_pit_safe`
    could silently break the actual production negative-control path.
    """

    def _leaky_add_last_close(df: pd.DataFrame) -> pd.DataFrame:
        """Inject a column whose value depends on the FUTURE — the future-most
        ``close`` is broadcast to every row. PIT-violating by construction."""
        out = df.sort_values("timestamp").copy()
        out["leaky_future_constant"] = float(out["close"].iloc[-1])
        return out

    prices = [100.0, 102.0, 99.0, 105.0, 103.0, 110.0, 108.0, 112.0]
    cut = 5
    with pytest.raises(AssertionError, match="PIT violation"):
        _assert_pit_safe(
            prices,
            cut=cut,
            add_feature_fn=_leaky_add_last_close,
            feature_cols=["leaky_future_constant"],
        )
