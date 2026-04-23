"""Tests for wave-8 module wiring into trading_cycle.py.

Covers:
  Step 3.55 — signals.multifactor_signal (build_multifactor_signal + load_factor_bundle)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.config.factor_bundles import load_factor_bundle, FactorBundleConfig
from src.assembled_core.signals.multifactor_signal import (
    build_multifactor_signal,
    MultiFactorSignalResult,
)


def _make_factor_df(
    n_symbols: int = 3,
    n_days: int = 20,
    factor_cols: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factor_cols = factor_cols or ["momentum_12m_excl_1m", "returns_12m"]
    rows = []
    for sym in [f"S{i}" for i in range(n_symbols)]:
        for d in range(n_days):
            row = {
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "symbol": sym,
            }
            for fc in factor_cols:
                row[fc] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _minimal_bundle() -> FactorBundleConfig:
    from src.assembled_core.config.factor_bundles import FactorConfig, FactorBundleOptions
    return FactorBundleConfig(
        universe="test",
        factor_set="core",
        horizon_days=5,
        factors=[
            FactorConfig(name="momentum_12m_excl_1m", weight=0.6, direction="positive"),
            FactorConfig(name="returns_12m", weight=0.4, direction="positive"),
        ],
        options=FactorBundleOptions(winsorize=False, zscore=True),
    )


# ---------------------------------------------------------------------------
# load_factor_bundle
# ---------------------------------------------------------------------------

def test_load_factor_bundle_macro_world():
    bundle = load_factor_bundle("config/factor_bundles/macro_world_etfs_core_bundle.yaml")
    assert bundle.universe == "macro_world_etfs"
    assert len(bundle.factors) > 0
    # Weights must sum to ~1.0
    total_weight = sum(f.weight for f in bundle.factors)
    assert abs(total_weight - 1.0) < 0.01


def test_load_factor_bundle_raises_for_missing_path(tmp_path):
    with pytest.raises(Exception):
        load_factor_bundle(tmp_path / "nonexistent_bundle.yaml")


# ---------------------------------------------------------------------------
# build_multifactor_signal
# ---------------------------------------------------------------------------

def test_build_multifactor_signal_returns_result():
    df = _make_factor_df()
    bundle = _minimal_bundle()
    result = build_multifactor_signal(df, bundle)
    assert isinstance(result, MultiFactorSignalResult)
    assert "mf_score" in result.df.columns


def test_build_multifactor_signal_has_meta():
    df = _make_factor_df()
    bundle = _minimal_bundle()
    result = build_multifactor_signal(df, bundle)
    assert "used_factors" in result.meta
    assert "missing_factors" in result.meta
    assert len(result.meta["used_factors"]) > 0


def test_build_multifactor_signal_missing_factors_reported():
    df = _make_factor_df(factor_cols=["momentum_12m_excl_1m"])  # missing returns_12m
    bundle = _minimal_bundle()
    result = build_multifactor_signal(df, bundle)
    assert "returns_12m" in result.meta["missing_factors"]


def test_build_multifactor_signal_score_per_symbol():
    df = _make_factor_df(n_symbols=5, n_days=15)
    bundle = _minimal_bundle()
    result = build_multifactor_signal(df, bundle)
    # Each symbol should have non-NaN scores for at least some rows
    latest = result.df.groupby("symbol")["mf_score"].last()
    assert len(latest) == 5


def test_build_multifactor_signal_empty_df_raises():
    empty = pd.DataFrame(columns=["timestamp", "symbol", "momentum_12m_excl_1m"])
    bundle = _minimal_bundle()
    with pytest.raises((KeyError, ValueError)):
        build_multifactor_signal(empty, bundle)


def test_build_multifactor_with_real_bundle():
    bundle = load_factor_bundle("config/factor_bundles/macro_world_etfs_core_bundle.yaml")
    df = _make_factor_df(
        n_symbols=4,
        n_days=20,
        factor_cols=["momentum_12m_excl_1m", "trend_strength_50", "returns_12m"],
    )
    result = build_multifactor_signal(df, bundle)
    # Some factors used, some missing — should not crash
    assert "mf_score" in result.df.columns
    assert len(result.meta["used_factors"]) >= 2
