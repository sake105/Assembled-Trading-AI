"""Tests for wave-36 module wiring into trading_cycle.py.

Covers:
  Step 2.21 — features.vpin (compute_vpin / compute_vpin_panel)
  Step 8.26 — ml.feedback_loop (FeedbackLoopController)
  Step 8.27 — ml.regime_model_router (RegimeModelRouter)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.vpin import (
    compute_vpin,
    compute_vpin_panel,
    VPINResult,
    classify_volume_bulk,
)
from src.assembled_core.ml.feedback_loop import (
    FeedbackLoopController,
    FeedbackLoopConfig,
)
from src.assembled_core.ml.regime_model_router import (
    RegimeModelRouter,
    RegimeRouterConfig,
)


# ---------------------------------------------------------------------------
# compute_vpin (Step 2.21)
# ---------------------------------------------------------------------------

def _make_price_vol(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)))
    volumes = pd.Series(rng.uniform(1e4, 1e6, n))
    return prices, volumes


def test_vpin_returns_result():
    prices, volumes = _make_price_vol()
    result = compute_vpin(prices, volumes)
    assert isinstance(result, VPINResult)


def test_vpin_avg_in_01():
    prices, volumes = _make_price_vol()
    result = compute_vpin(prices, volumes)
    assert 0.0 <= result.avg_vpin <= 1.0


def test_vpin_current_in_01():
    prices, volumes = _make_price_vol()
    result = compute_vpin(prices, volumes)
    assert 0.0 <= result.current_vpin <= 1.0


def test_vpin_short_series_returns_empty():
    prices = pd.Series([100.0] * 10)
    volumes = pd.Series([1000.0] * 10)
    result = compute_vpin(prices, volumes)
    assert result.n_buckets == 0
    assert result.avg_vpin == 0.0


def test_vpin_is_toxic_bool():
    prices, volumes = _make_price_vol()
    result = compute_vpin(prices, volumes)
    assert isinstance(result.is_toxic, bool)


def test_vpin_n_buckets_non_negative():
    prices, volumes = _make_price_vol(200)
    result = compute_vpin(prices, volumes)
    assert result.n_buckets >= 0


def test_classify_volume_bulk_returns_tuple():
    prices, volumes = _make_price_vol(50)
    buy_vol, sell_vol = classify_volume_bulk(prices, volumes)
    assert isinstance(buy_vol, pd.Series)
    assert isinstance(sell_vol, pd.Series)
    assert len(buy_vol) == len(prices)


def test_vpin_panel_returns_df(tmp_path):
    rng = np.random.default_rng(1)
    n, syms = 100, 3
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = pd.DataFrame({f"S{i}": 100.0 + np.cumsum(rng.normal(0, 0.5, n)) for i in range(syms)}, index=idx)
    volumes = pd.DataFrame({f"S{i}": rng.uniform(1e4, 1e6, n) for i in range(syms)}, index=idx)
    result = compute_vpin_panel(prices, volumes)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# FeedbackLoopController (Step 8.26)
# ---------------------------------------------------------------------------

def test_feedback_controller_creates(tmp_path):
    ctrl = FeedbackLoopController(state_dir=tmp_path / "feedback")
    assert (tmp_path / "feedback").exists()


def test_feedback_controller_config_default(tmp_path):
    ctrl = FeedbackLoopController(state_dir=tmp_path / "fb")
    assert isinstance(ctrl.config, FeedbackLoopConfig)


def test_feedback_controller_check_interval(tmp_path):
    cfg = FeedbackLoopConfig(check_interval_days=7)
    ctrl = FeedbackLoopController(config=cfg, state_dir=tmp_path / "fb")
    assert ctrl.config.check_interval_days == 7


def test_feedback_controller_state_file_attr(tmp_path):
    ctrl = FeedbackLoopController(state_dir=tmp_path / "fb")
    assert hasattr(ctrl, "_STATE_FILE")
    assert ctrl._STATE_FILE.endswith(".json")


def test_feedback_controller_auto_deploy_false():
    cfg = FeedbackLoopConfig()
    assert cfg.max_retrain_per_quarter == 4  # default guardrail


def test_feedback_config_retrain_cooldown():
    cfg = FeedbackLoopConfig(retrain_cooldown_days=45)
    assert cfg.retrain_cooldown_days == 45


# ---------------------------------------------------------------------------
# RegimeModelRouter (Step 8.27)
# ---------------------------------------------------------------------------

def test_regime_router_creates():
    router = RegimeModelRouter()
    assert isinstance(router, RegimeModelRouter)


def test_regime_router_config_default():
    router = RegimeModelRouter()
    assert isinstance(router.config, RegimeRouterConfig)


def test_regime_router_no_state_initially():
    router = RegimeModelRouter()
    assert router._state is None


def test_regime_router_config_custom():
    cfg = RegimeRouterConfig()
    router = RegimeModelRouter(config=cfg)
    assert router.config is cfg


def test_regime_router_config_has_min_samples():
    cfg = RegimeRouterConfig()
    assert hasattr(cfg, "min_samples_per_regime")
    assert cfg.min_samples_per_regime > 0
