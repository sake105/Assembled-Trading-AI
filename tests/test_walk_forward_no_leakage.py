"""A9: Walk-Forward purge/embargo gaps prevent train-test label leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    generate_walk_forward_splits,
)


@pytest.mark.fast
def test_purge_days_creates_gap_between_train_and_test():
    """With purge_days=10, test_start must be >= train_end + 10 days."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-12-31", tz="UTC"),
        train_window_days=252,
        test_window_days=63,
        mode="rolling",
        step_size_days=63,
        min_train_periods=200,
        min_test_periods=50,
        purge_days=10,
    )
    splits = generate_walk_forward_splits(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2024-12-31", tz="UTC"),
        config,
    )
    assert len(splits) > 0
    for s in splits:
        gap = (s.test_start - s.train_end).days
        assert gap >= 10, (
            f"Split {s.split_index}: gap between train_end and test_start is {gap} days, "
            f"must be >= purge_days=10"
        )


@pytest.mark.fast
def test_embargo_days_creates_gap_between_splits():
    """With embargo_days=5, consecutive splits must have train_end[i+1] >= test_end[i] + 5 days."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-12-31", tz="UTC"),
        train_window_days=252,
        test_window_days=63,
        mode="rolling",
        step_size_days=63,
        min_train_periods=200,
        min_test_periods=50,
        embargo_days=5,
    )
    splits = generate_walk_forward_splits(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2024-12-31", tz="UTC"),
        config,
    )
    assert len(splits) > 1
    for i in range(len(splits) - 1):
        embargo_gap = (splits[i + 1].train_end - splits[i].test_end).days
        assert embargo_gap >= 5, (
            f"Between splits {i} and {i + 1}: gap is {embargo_gap} days, "
            f"must be >= embargo_days=5"
        )


@pytest.mark.fast
def test_purge_label_horizon_validation():
    """purge_days < max_label_horizon must raise ValueError."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-12-31", tz="UTC"),
        train_window_days=252,
        test_window_days=63,
        mode="rolling",
        step_size_days=63,
        min_train_periods=200,
        min_test_periods=50,
        purge_days=5,
        max_label_horizon=10,  # purge_days (5) < max_label_horizon (10) → must raise
    )
    with pytest.raises(ValueError, match="purge_days.*max_label_horizon"):
        generate_walk_forward_splits(
            pd.Timestamp("2022-01-01", tz="UTC"),
            pd.Timestamp("2024-12-31", tz="UTC"),
            config,
        )


@pytest.mark.fast
def test_no_purge_embargo_backward_compatible():
    """purge_days=0 and embargo_days=0 (defaults) must produce same structure as before."""
    config = WalkForwardConfig(
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-12-31", tz="UTC"),
        train_window_days=252,
        test_window_days=63,
        mode="rolling",
        step_size_days=63,
        min_train_periods=200,
        min_test_periods=50,
    )
    splits = generate_walk_forward_splits(
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2024-12-31", tz="UTC"),
        config,
    )
    assert len(splits) > 0
    for s in splits:
        # With purge=0, test_start should equal train_end (no gap)
        assert s.test_start == s.train_end, (
            f"Split {s.split_index}: with purge_days=0, test_start must equal train_end"
        )


@pytest.mark.fast
def test_walkforward_config_has_embargo_and_purge_fields():
    """WalkForwardConfig must have purge_days and embargo_days fields."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(WalkForwardConfig)}
    assert "purge_days" in fields, "WalkForwardConfig must have purge_days"
    assert "embargo_days" in fields, "WalkForwardConfig must have embargo_days"
    assert "max_label_horizon" in fields, (
        "WalkForwardConfig must have max_label_horizon"
    )

    # Defaults must be 0 (backward compatible)
    cfg = WalkForwardConfig(
        start_date=pd.Timestamp("2022-01-01", tz="UTC"),
        end_date=pd.Timestamp("2024-01-01", tz="UTC"),
        train_window_days=252,
        test_window_days=63,
    )
    assert cfg.purge_days == 0
    assert cfg.embargo_days == 0
    assert cfg.max_label_horizon is None


# ---------------------------------------------------------------------------
# Regression tests — production config (trend_baseline 2018-2025, 252d WF)
# Follow-up 2 of GO_LIVE B2: CI-enforced regression guard so fold-count and
# boundary changes are caught automatically (script-only coverage is not enough).
# ---------------------------------------------------------------------------

_PROD_CONFIG = dict(
    start_date=pd.Timestamp("2018-01-01", tz="UTC"),
    end_date=pd.Timestamp("2025-12-31", tz="UTC"),
    train_window_days=252,
    test_window_days=252,
    mode="rolling",
    step_size_days=252,
    min_train_periods=200,
    min_test_periods=63,  # WalkForwardConfig default — explicit for regression stability
)
_PROD_START = pd.Timestamp("2018-01-01", tz="UTC")
_PROD_END = pd.Timestamp("2025-12-31", tz="UTC")


def _prod_splits():
    cfg = WalkForwardConfig(**_PROD_CONFIG)
    return generate_walk_forward_splits(_PROD_START, _PROD_END, cfg)


@pytest.mark.fast
def test_production_config_10_folds():
    """Production config (2018-2025, 252/252/252) must yield exactly 10 folds."""
    splits = _prod_splits()
    assert len(splits) == 10, f"Expected 10 folds, got {len(splits)}"


@pytest.mark.fast
def test_production_config_fold_1_boundaries():
    """Fold 1: train_start=2018-01-01, test_start==train_end (purge_days=0)."""
    splits = _prod_splits()
    f = splits[0]
    assert f.train_start == _PROD_START, f"Fold 1 train_start: {f.train_start}"
    assert f.test_start == f.train_end, (
        f"Fold 1 test_start ({f.test_start}) != train_end ({f.train_end})"
    )


@pytest.mark.fast
def test_production_config_no_train_test_overlap():
    """All 10 folds: test_start >= train_end (no overlap, purge_days=0)."""
    for s in _prod_splits():
        assert s.test_start >= s.train_end, (
            f"Fold {s.split_index}: test_start {s.test_start} < train_end {s.train_end}"
        )


@pytest.mark.fast
def test_production_config_cross_fold_test_independence():
    """All fold test windows are non-overlapping (step_size == test_window)."""
    splits = _prod_splits()
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            assert not (a.test_start < b.test_end and b.test_start < a.test_end), (
                f"Folds {a.split_index} and {b.split_index}: test windows overlap"
            )


# ---------------------------------------------------------------------------
# Follow-up 1 of GO_LIVE B2: demonstrate purge/embargo on production config.
# trend_baseline is rule-based (not ML), so CPCV is the walk-forward analogue
# with purge_days > 0. This test verifies the purge mechanism works on the
# same 2018-2025 universe, not just on generic synthetic configs.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_production_config_with_purge_creates_gap():
    """Production config + purge_days=20 must create >=20-day gap in all folds."""
    cfg = WalkForwardConfig(
        **{**_PROD_CONFIG, "purge_days": 20},
    )
    splits = generate_walk_forward_splits(_PROD_START, _PROD_END, cfg)
    assert len(splits) > 0, "Expected folds with purge_days=20"
    for s in splits:
        gap = (s.test_start - s.train_end).days
        assert gap >= 20, f"Fold {s.split_index}: purge gap {gap}d < purge_days=20"


@pytest.mark.fast
def test_production_config_purge_does_not_increase_fold_count():
    """purge_days=20 must not increase fold count relative to purge_days=0."""
    splits_no_purge = _prod_splits()
    cfg_purge = WalkForwardConfig(**{**_PROD_CONFIG, "purge_days": 20})
    splits_purge = generate_walk_forward_splits(_PROD_START, _PROD_END, cfg_purge)
    assert len(splits_purge) <= len(splits_no_purge), (
        f"Purge unexpectedly increased fold count: {len(splits_no_purge)} → {len(splits_purge)}"
    )
