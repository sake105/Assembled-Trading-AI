"""Tests for wave-39 module wiring into trading_cycle.py.

Covers:
  Step 7.69 — ops.alert_manager (AlertManager)
  Step 8.30 — qa.backtest_overfit (compute_pbo)
  Step 8.31 — qa.ml_evaluation (evaluate_meta_model)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ops.alert_manager import AlertManager, Alert
from src.assembled_core.qa.backtest_overfit import compute_pbo, PBOResult, performance_degradation
from src.assembled_core.qa.ml_evaluation import evaluate_meta_model


# ---------------------------------------------------------------------------
# AlertManager (Step 7.69)
# ---------------------------------------------------------------------------

def test_alert_manager_creates(tmp_path):
    am = AlertManager(rate_limit_seconds=0, output_dir=str(tmp_path / "alerts"))
    assert isinstance(am, AlertManager)


def test_alert_fires_first_time(tmp_path):
    am = AlertManager(rate_limit_seconds=0, output_dir=str(tmp_path))
    fired = am.alert("WARNING", "test", "Test alert message")
    assert fired is True


def test_alert_rate_limited(tmp_path):
    am = AlertManager(rate_limit_seconds=9999, output_dir=str(tmp_path))
    am.alert("WARNING", "test", "Test alert")
    fired = am.alert("WARNING", "test", "Test alert")  # Same message, rate-limited
    assert fired is False


def test_alert_pending_count(tmp_path):
    am = AlertManager(rate_limit_seconds=0, output_dir=str(tmp_path))
    assert am.pending_count == 0
    am.alert("INFO", "src", "msg1")
    am.alert("WARNING", "src", "msg2")
    assert am.pending_count == 2


def test_alert_flush_to_json_creates_file(tmp_path):
    am = AlertManager(rate_limit_seconds=0, output_dir=str(tmp_path / "alerts"))
    am.alert("INFO", "test", "flush test")
    result = am.flush_to_json()
    assert result is not None
    assert (tmp_path / "alerts").exists()


def test_alert_flush_empty_returns_none(tmp_path):
    am = AlertManager(rate_limit_seconds=0, output_dir=str(tmp_path))
    result = am.flush_to_json()
    assert result is None


def test_alert_clears_after_flush(tmp_path):
    am = AlertManager(rate_limit_seconds=0, output_dir=str(tmp_path / "a"))
    am.alert("INFO", "test", "msg")
    am.flush_to_json()
    assert am.pending_count == 0


def test_alert_levels():
    am = AlertManager(rate_limit_seconds=0)
    for level in ["INFO", "WARNING", "CRITICAL"]:
        fired = am.alert(level, "test", f"level_{level}")
        assert fired is True


# ---------------------------------------------------------------------------
# compute_pbo (Step 8.30)
# ---------------------------------------------------------------------------

def _make_strategy_returns(n: int = 50, n_strats: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"strat_{i}": rng.normal(0, 0.01, n) for i in range(n_strats)}
    )


def test_pbo_returns_result():
    df = _make_strategy_returns(n=50, n_strats=4)
    result = compute_pbo(df, n_splits=16)
    assert isinstance(result, PBOResult)


def test_pbo_value_in_01():
    df = _make_strategy_returns(n=50, n_strats=4)
    result = compute_pbo(df, n_splits=16)
    assert 0.0 <= result.pbo <= 1.0


def test_pbo_n_splits_reported():
    df = _make_strategy_returns(n=50, n_strats=4)
    result = compute_pbo(df, n_splits=8)
    assert result.n_splits > 0


def test_pbo_requires_two_strategies():
    df = pd.DataFrame({"strat_0": [0.01, -0.01, 0.02, -0.02]})
    with pytest.raises(ValueError):
        compute_pbo(df)


def test_pbo_requires_four_periods():
    df = pd.DataFrame({"s0": [0.01, -0.01, 0.02], "s1": [0.0, 0.01, -0.01]})
    with pytest.raises(ValueError):
        compute_pbo(df)


def test_pbo_two_col_df():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 0.01, 40), "b": rng.normal(0, 0.01, 40)})
    result = compute_pbo(df, n_splits=8)
    assert isinstance(result, PBOResult)


# ---------------------------------------------------------------------------
# evaluate_meta_model (Step 8.31)
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn", reason="scikit-learn required for ml_evaluation")


def _make_binary_predictions(n: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = pd.Series((rng.random(n) > 0.5).astype(float))
    y_prob = pd.Series(rng.uniform(0.1, 0.9, n))
    return y_true, y_prob


def test_evaluate_meta_model_returns_dict():
    y_true, y_prob = _make_binary_predictions()
    result = evaluate_meta_model(y_true, y_prob)
    assert isinstance(result, dict)


def test_evaluate_meta_model_has_roc_auc():
    y_true, y_prob = _make_binary_predictions()
    result = evaluate_meta_model(y_true, y_prob)
    assert "roc_auc" in result


def test_evaluate_meta_model_has_brier_score():
    y_true, y_prob = _make_binary_predictions()
    result = evaluate_meta_model(y_true, y_prob)
    assert "brier_score" in result


def test_evaluate_meta_model_roc_in_01():
    y_true, y_prob = _make_binary_predictions()
    result = evaluate_meta_model(y_true, y_prob)
    roc = result.get("roc_auc")
    if roc is not None and not np.isnan(roc):
        assert 0.0 <= roc <= 1.0


def test_evaluate_meta_model_brier_non_negative():
    y_true, y_prob = _make_binary_predictions()
    result = evaluate_meta_model(y_true, y_prob)
    brier = result.get("brier_score")
    assert brier is not None
    assert brier >= 0.0


def test_evaluate_meta_model_wrong_labels_raises():
    y_true = pd.Series([0.0, 0.5, 1.0, 2.0])  # 2.0 is invalid
    y_prob = pd.Series([0.1, 0.4, 0.6, 0.9])
    with pytest.raises(ValueError):
        evaluate_meta_model(y_true, y_prob)


def test_evaluate_meta_model_length_mismatch_raises():
    y_true = pd.Series([0.0, 1.0, 0.0])
    y_prob = pd.Series([0.1, 0.9])
    with pytest.raises(ValueError):
        evaluate_meta_model(y_true, y_prob)
