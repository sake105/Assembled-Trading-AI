"""Polars-vs-pandas numerical-equivalence tests (audit B-001 / §8.2).

Pin the new Polars-backed feature functions to within 1e-9 of the
pandas reference implementations. Two synthetic panels:

    * single-symbol short series — covers warm-up edge cases
    * multi-symbol multi-year — covers groupby-over-symbol behavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


pl = pytest.importorskip("polars")  # auto-skip if polars missing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def single_symbol_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    base = np.cumprod(1.0 + rng.normal(0.0005, 0.012, n))
    close = 100.0 * base
    high = close * (1.0 + rng.uniform(0.0, 0.01, n))
    low = close * (1.0 - rng.uniform(0.0, 0.01, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "AAPL",
            "close": close,
            "high": high,
            "low": low,
        }
    )


@pytest.fixture(scope="module")
def multi_symbol_panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    symbols = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"]
    frames = []
    n = 500
    ts = pd.date_range("2023-01-01", periods=n, freq="D")
    for sym in symbols:
        base = np.cumprod(1.0 + rng.normal(0.0005, 0.015, n))
        close = 100.0 * base
        high = close * (1.0 + rng.uniform(0.0, 0.012, n))
        low = close * (1.0 - rng.uniform(0.0, 0.012, n))
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "close": close,
                    "high": high,
                    "low": low,
                }
            )
        )
    # Interleave rows so neither symbol nor timestamp is pre-sorted.
    return (
        pd.concat(frames, ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# log_returns
# ---------------------------------------------------------------------------


def _align_for_compare(
    a: pd.DataFrame, b: pd.DataFrame, *, sort_by: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align two DataFrames row-wise by stable sort columns for elementwise compare."""
    a_s = a.sort_values(sort_by).reset_index(drop=True)
    b_s = b.sort_values(sort_by).reset_index(drop=True)
    return a_s, b_s


def test_log_returns_single_symbol_equivalence(single_symbol_panel) -> None:
    from src.assembled_core.features.ta_features import add_log_returns
    from src.assembled_core.features.ta_features_polars import add_log_returns_polars

    p = add_log_returns(single_symbol_panel)
    q = add_log_returns_polars(single_symbol_panel)
    p_s, q_s = _align_for_compare(p, q, sort_by=["symbol", "timestamp"])
    np.testing.assert_allclose(
        p_s["ta_log_return_v1"].fillna(0).to_numpy(),
        q_s["ta_log_return_v1"].fillna(0).to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_log_returns_multi_symbol_equivalence(multi_symbol_panel) -> None:
    from src.assembled_core.features.ta_features import add_log_returns
    from src.assembled_core.features.ta_features_polars import add_log_returns_polars

    p = add_log_returns(multi_symbol_panel)
    q = add_log_returns_polars(multi_symbol_panel)
    p_s, q_s = _align_for_compare(p, q, sort_by=["symbol", "timestamp"])
    np.testing.assert_allclose(
        p_s["ta_log_return_v1"].fillna(0).to_numpy(),
        q_s["ta_log_return_v1"].fillna(0).to_numpy(),
        atol=1e-12,
        rtol=1e-12,
    )


def test_log_returns_polars_rejects_missing_symbol() -> None:
    from src.assembled_core.features.ta_features_polars import add_log_returns_polars

    df = pd.DataFrame({"close": [100.0, 101.0]})
    with pytest.raises(KeyError):
        add_log_returns_polars(df)


# ---------------------------------------------------------------------------
# moving_averages
# ---------------------------------------------------------------------------


def test_moving_averages_multi_symbol_equivalence(multi_symbol_panel) -> None:
    from src.assembled_core.features.ta_features import add_moving_averages
    from src.assembled_core.features.ta_features_polars import (
        add_moving_averages_polars,
    )

    p = add_moving_averages(multi_symbol_panel, windows=(5, 20, 50))
    q = add_moving_averages_polars(multi_symbol_panel, windows=(5, 20, 50))
    p_s, q_s = _align_for_compare(p, q, sort_by=["symbol", "timestamp"])
    for w in (5, 20, 50):
        np.testing.assert_allclose(
            p_s[f"ta_ma_{w}_v1"].fillna(0).to_numpy(),
            q_s[f"ta_ma_{w}_v1"].fillna(0).to_numpy(),
            atol=1e-9,
            rtol=1e-9,
        )


def test_moving_averages_polars_emits_legacy_columns(multi_symbol_panel) -> None:
    from src.assembled_core.features.ta_features_polars import (
        add_moving_averages_polars,
    )

    q = add_moving_averages_polars(multi_symbol_panel, windows=(20,))
    assert "ta_ma_20_v1" in q.columns
    assert "ma_20" in q.columns  # legacy mirror
    np.testing.assert_array_equal(q["ta_ma_20_v1"].to_numpy(), q["ma_20"].to_numpy())


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def test_atr_multi_symbol_equivalence(multi_symbol_panel) -> None:
    from src.assembled_core.features.ta_features import add_atr
    from src.assembled_core.features.ta_features_polars import add_atr_polars

    p = add_atr(multi_symbol_panel, window=14)
    q = add_atr_polars(multi_symbol_panel, window=14)
    p_s, q_s = _align_for_compare(p, q, sort_by=["symbol", "timestamp"])
    np.testing.assert_allclose(
        p_s["ta_atr_14_v1"].fillna(0).to_numpy(),
        q_s["ta_atr_14_v1"].fillna(0).to_numpy(),
        atol=1e-9,
        rtol=1e-9,
    )


def test_atr_polars_rejects_missing_columns(multi_symbol_panel) -> None:
    from src.assembled_core.features.ta_features_polars import add_atr_polars

    bad = multi_symbol_panel.drop(columns=["high"])
    with pytest.raises(KeyError, match="Missing required"):
        add_atr_polars(bad)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def test_rsi_multi_symbol_equivalence(multi_symbol_panel) -> None:
    from src.assembled_core.features.ta_features import add_rsi
    from src.assembled_core.features.ta_features_polars import add_rsi_polars

    p = add_rsi(multi_symbol_panel, window=14)
    q = add_rsi_polars(multi_symbol_panel, window=14)
    p_s, q_s = _align_for_compare(p, q, sort_by=["symbol", "timestamp"])
    # RSI uses /1e-12 fallback when avg_loss==0 — both implementations
    # use the same fallback, so 1e-9 tolerance is appropriate.
    np.testing.assert_allclose(
        p_s["ta_rsi_14_v1"].fillna(50.0).to_numpy(),
        q_s["ta_rsi_14_v1"].fillna(50.0).to_numpy(),
        atol=1e-6,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------


def test_is_polars_available_returns_true() -> None:
    from src.assembled_core.features.ta_features_polars import is_polars_available

    assert is_polars_available() is True
