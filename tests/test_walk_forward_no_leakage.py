"""A9: Walk-Forward purge/embargo gaps prevent train-test label leakage."""
from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.qa.walk_forward import WalkForwardConfig, generate_walk_forward_splits


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
            f"Between splits {i} and {i+1}: gap is {embargo_gap} days, "
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
    assert "max_label_horizon" in fields, "WalkForwardConfig must have max_label_horizon"

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
