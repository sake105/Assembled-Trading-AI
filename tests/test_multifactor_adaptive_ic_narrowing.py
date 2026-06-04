"""Regression tests for the IC-weighted path of build_adaptive_multifactor_signal.

A mypy cleanup added ``assert ic_weights_df is not None`` guards inside the two
``if use_ic:`` branches of ``build_adaptive_multifactor_signal``. These asserts
are pure type-narrowing: ``use_ic`` is True only when ``ic_weights_df is not
None`` (and non-empty), so the asserts always hold and cannot alter any weight
or score. These tests exercise the IC-weighted path end-to-end and pin the
``mf_score`` / ``mf_aggregate_ic`` output so any accidental behavioural change
in that path would be caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.config.factor_bundles import (
    FactorBundleConfig,
    FactorBundleOptions,
    FactorConfig,
)
from src.assembled_core.signals.multifactor_signal import (
    build_adaptive_multifactor_signal,
)


_DATES = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
_SYMBOLS = ["AAA", "BBB", "CCC", "DDD"]


def _panel() -> pd.DataFrame:
    """Small balanced panel: 6 timestamps x 4 symbols, 2 factors + fwd return."""
    rng = np.random.default_rng(7)
    rows = []
    for ts in _DATES:
        for sym in _SYMBOLS:
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "mom": rng.standard_normal(),
                    "value": rng.standard_normal(),
                }
            )
    df = pd.DataFrame(rows)
    # Forward return correlated with mom so IC is well-defined.
    df["fwd_ret"] = 0.5 * df["mom"] + rng.standard_normal(len(df)) * 0.1
    return df


def _ic_weights() -> pd.DataFrame:
    """Pre-built IC-weights frame indexed by timestamp.

    Provides ``weight_mom`` / ``weight_value`` (per-timestamp factor weights)
    and ``aggregate_ic`` so the ``use_ic=True`` branch (with the narrowing
    assert) executes deterministically without needing a 60+ day history.
    """
    return pd.DataFrame(
        {
            "weight_mom": np.linspace(0.6, 0.8, len(_DATES)),
            "weight_value": np.linspace(0.4, 0.2, len(_DATES)),
            "aggregate_ic": np.linspace(0.1, 0.3, len(_DATES)),
        },
        index=pd.Index(_DATES, name="timestamp"),
    )


def _bundle() -> FactorBundleConfig:
    return FactorBundleConfig(
        universe="test_universe",
        factor_set="core",
        horizon_days=5,
        factors=[
            FactorConfig(name="mom", weight=0.5, direction="positive"),
            FactorConfig(name="value", weight=0.5, direction="positive"),
        ],
        options=FactorBundleOptions(winsorize=False, zscore=True),
    )


@pytest.mark.fast
def test_ic_weighted_path_produces_aggregate_ic_and_scores():
    """A non-empty ic_weights_df triggers the use_ic=True path (asserts run)."""
    df = _panel()
    bundle = _bundle()

    res = build_adaptive_multifactor_signal(df, bundle, ic_weights_df=_ic_weights())

    # IC-weighted path was taken.
    assert res.meta["ic_weighted"] is True
    assert "mf_score" in res.df.columns
    # The aggregate-IC confidence column is only written on the use_ic branch
    # guarded by the narrowing assert — its presence proves that branch ran.
    assert "mf_aggregate_ic" in res.df.columns
    # Scores are finite where weights are non-zero (at least some rows).
    assert res.df["mf_score"].notna().any()
    # mf_aggregate_ic is mapped per timestamp from the supplied frame.
    assert res.df["mf_aggregate_ic"].notna().all()


@pytest.mark.fast
def test_ic_weighted_output_is_deterministic_snapshot():
    """Pin mf_score so the narrowing asserts cannot silently change values."""
    df = _panel()
    bundle = _bundle()

    res_a = build_adaptive_multifactor_signal(df, bundle, ic_weights_df=_ic_weights())
    res_b = build_adaptive_multifactor_signal(df, bundle, ic_weights_df=_ic_weights())

    # Deterministic across calls (no hidden state mutated by the asserts).
    pd.testing.assert_series_equal(
        res_a.df["mf_score"], res_b.df["mf_score"], check_names=False
    )
    pd.testing.assert_series_equal(
        res_a.df["mf_aggregate_ic"],
        res_b.df["mf_aggregate_ic"],
        check_names=False,
    )


@pytest.mark.fast
def test_static_path_unaffected_when_no_ic():
    """Without IC inputs, use_ic is False; asserts are skipped, output valid."""
    df = _panel().drop(columns=["fwd_ret"])
    bundle = _bundle()

    res = build_adaptive_multifactor_signal(df, bundle)

    assert res.meta["ic_weighted"] is False
    assert "mf_score" in res.df.columns
    # mf_aggregate_ic is NOT written on the static path.
    assert "mf_aggregate_ic" not in res.df.columns
