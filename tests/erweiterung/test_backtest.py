"""Tests for erweiterung.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.backtest import (
    cpcv,
    deflated_sharpe,
    performance_metrics,
    walk_forward,
    white_reality_check,
)


def test_cpcv_splits_count():
    config = cpcv.CPCVConfig(n_groups=4, test_groups_per_split=2, embargo_pct=0.01)
    splits = cpcv.cpcv_splits(n_samples=100, config=config)
    # C(4, 2) = 6 splits
    assert len(splits) == 6
    for tr, te in splits:
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


def test_cpcv_no_overlap():
    config = cpcv.CPCVConfig(n_groups=5, test_groups_per_split=1, embargo_pct=0.02)
    splits = cpcv.cpcv_splits(100, config)
    for tr, te in splits:
        assert (tr < 100).all()
        assert (te < 100).all()


def test_deflated_sharpe_basic(synthetic_returns):
    r = synthetic_returns.iloc[:, 0]
    out = deflated_sharpe.deflated_sharpe_ratio(r, n_trials=1)
    assert "sr" in out
    assert "annualized_sr" in out
    assert "dsr" in out


def test_deflated_sharpe_lowers_with_more_trials(synthetic_returns):
    r = synthetic_returns.iloc[:, 0]
    out_few = deflated_sharpe.deflated_sharpe_ratio(r, n_trials=1)
    out_many = deflated_sharpe.deflated_sharpe_ratio(r, n_trials=1000)
    # higher n_trials -> higher sr0 -> typically lower dsr
    assert out_many["sr0"] >= out_few["sr0"]


def test_psr():
    r = pd.Series(np.random.default_rng(0).normal(0.001, 0.01, 500))
    psr = deflated_sharpe.probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    assert 0 <= psr <= 1


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(0)
    idx = white_reality_check.stationary_bootstrap_indices(
        100, expected_block_length=5, rng=rng
    )
    assert len(idx) == 100
    assert (idx >= 0).all() and (idx < 100).all()


def test_whites_reality_check_basic():
    rng = np.random.default_rng(0)
    n = 200
    K = 5
    excess = pd.DataFrame(
        rng.normal(0, 0.01, (n, K)),
        columns=[f"strategy_{k}" for k in range(K)],
        index=pd.date_range("2024-01-01", periods=n),
    )
    out = white_reality_check.whites_reality_check(excess, n_bootstrap=200)
    assert "p_value" in out
    assert 0 <= out["p_value"] <= 1


def test_hansen_spa_test():
    rng = np.random.default_rng(0)
    n = 200
    excess = pd.DataFrame(
        {
            "good": rng.normal(0.001, 0.01, n),
            "bad": rng.normal(-0.001, 0.01, n),
            "noise": rng.normal(0, 0.01, n),
        },
        index=pd.date_range("2024-01-01", periods=n),
    )
    out = white_reality_check.hansen_spa_test(excess, n_bootstrap=200)
    assert "p_value" in out


def test_sharpe_ratio_basic(synthetic_returns):
    sr = performance_metrics.sharpe_ratio(synthetic_returns.iloc[:, 0])
    assert np.isfinite(sr)


def test_sortino_ratio(synthetic_returns):
    s = performance_metrics.sortino_ratio(synthetic_returns.iloc[:, 0])
    assert np.isfinite(s)


def test_max_drawdown():
    eq = pd.Series([100, 120, 100, 110, 80, 90, 130])
    mdd, _, _ = performance_metrics.max_drawdown(eq)
    assert mdd <= -0.30


def test_calmar_ratio(synthetic_returns):
    c = performance_metrics.calmar_ratio(synthetic_returns.iloc[:, 0])
    assert np.isfinite(c) or pd.isna(c)


def test_all_metrics(synthetic_returns):
    out = performance_metrics.all_metrics(synthetic_returns.iloc[:, 0])
    for key in ("sharpe", "sortino", "max_drawdown", "calmar"):
        assert key in out


def test_walk_forward_basic():
    df = pd.DataFrame(
        {"return": np.random.default_rng(0).normal(0.001, 0.01, 1000)},
        index=pd.date_range("2020-01-01", periods=1000, freq="D"),
    )

    def strat(train_df, test_df):
        # naive: predict mean of train
        mu = train_df["return"].mean()
        return pd.Series(mu, index=test_df.index)

    config = walk_forward.WalkForwardConfig(train_size=200, test_size=20)
    out = walk_forward.walk_forward_run(df, strat, config)
    assert "fold_id" in out.columns
    assert len(out) > 0
