"""Tests for wave-51 module wiring into trading_cycle.py.

Covers:
  Step 3.92 — signals.ml_integration (MLSignalPipeline)
  Step 3.93 — signals.rules_event_insider_shipping (generate_event_signals)
  Step 7.72 — ops.paper_summary (build_paper_summary / write_paper_summary)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.signals.ml_integration import MLSignalPipeline, MLPipelineOutput
from src.assembled_core.signals.rules_event_insider_shipping import generate_event_signals
from src.assembled_core.ops.paper_summary import build_paper_summary, write_paper_summary


# ---------------------------------------------------------------------------
# MLSignalPipeline (Step 3.92)
# ---------------------------------------------------------------------------

def test_ml_pipeline_creates():
    pipe = MLSignalPipeline()
    assert isinstance(pipe, MLSignalPipeline)


def test_ml_pipeline_no_models_by_default():
    pipe = MLSignalPipeline()
    assert pipe.primary_model is None
    assert pipe.regime_router is None


def test_ml_pipeline_run_empty_features():
    pipe = MLSignalPipeline()
    features = pd.DataFrame()
    result = pipe.run(features)
    assert isinstance(result, MLPipelineOutput)


def test_ml_pipeline_run_with_features():
    rng = np.random.default_rng(0)
    pipe = MLSignalPipeline()
    features = pd.DataFrame(rng.normal(0, 1, (10, 4)), columns=["f1", "f2", "f3", "f4"])
    result = pipe.run(features)
    assert isinstance(result, MLPipelineOutput)


def test_ml_pipeline_output_has_primary_signal():
    rng = np.random.default_rng(0)
    pipe = MLSignalPipeline()
    features = pd.DataFrame(rng.normal(0, 1, (10, 3)), columns=["f1", "f2", "f3"])
    output = pipe.run(features)
    assert hasattr(output, "primary_signal") or hasattr(output, "final_position")


def test_ml_pipeline_with_primary_model():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 3))
    y = rng.normal(0, 1, 50)
    model = Ridge().fit(X, y)
    pipe = MLSignalPipeline(primary_model=model, feature_cols=["f1", "f2", "f3"])
    features = pd.DataFrame(rng.normal(0, 1, (10, 3)), columns=["f1", "f2", "f3"])
    output = pipe.run(features)
    assert isinstance(output, MLPipelineOutput)


# ---------------------------------------------------------------------------
# generate_event_signals (Step 3.93)
# ---------------------------------------------------------------------------

def _make_prices_with_features(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": "AAPL",
        "close": 150.0 + np.cumsum(rng.normal(0, 0.5, n)),
        "insider_net_buy_20d": rng.normal(0, 500, n),
        "shipping_congestion_score_7d": rng.uniform(20, 80, n),
    })


def test_generate_event_signals_returns_df():
    prices = _make_prices_with_features()
    result = generate_event_signals(prices)
    assert isinstance(result, pd.DataFrame)


def test_generate_event_signals_has_direction():
    prices = _make_prices_with_features()
    result = generate_event_signals(prices)
    assert "direction" in result.columns


def test_generate_event_signals_has_score():
    prices = _make_prices_with_features()
    result = generate_event_signals(prices)
    assert "score" in result.columns


def test_generate_event_signals_valid_directions():
    prices = _make_prices_with_features()
    result = generate_event_signals(prices)
    assert set(result["direction"].unique()).issubset({"LONG", "FLAT", "SHORT"})


def test_generate_event_signals_missing_required_raises():
    prices = pd.DataFrame({"timestamp": [], "symbol": []})
    with pytest.raises(KeyError):
        generate_event_signals(prices)


def test_generate_event_signals_zero_features_flat():
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": ts,
        "symbol": "AAPL",
        "close": 150.0 + rng.normal(0, 0.5, 10),
        "insider_net_buy_20d": 0.0,
        "shipping_congestion_score_7d": 50.0,
    })
    result = generate_event_signals(prices)
    assert (result["direction"] == "FLAT").all()


# ---------------------------------------------------------------------------
# paper_summary (Step 7.72)
# ---------------------------------------------------------------------------

def test_build_paper_summary_empty_dates():
    result = build_paper_summary("/nonexistent/path", dates=[])
    assert isinstance(result, dict)


def test_build_paper_summary_has_schema_version():
    result = build_paper_summary("/nonexistent/path", dates=[])
    assert "schema_version" in result


def test_build_paper_summary_no_equity_none(tmp_path):
    result = build_paper_summary(str(tmp_path), dates=["2024-01-15"])
    assert result.get("total_return") is None


def test_write_paper_summary_creates_file(tmp_path):
    summary = {"schema_version": "paper.summary.v1", "total_return": 0.1}
    path = write_paper_summary(tmp_path, "2024-01-01", "2024-01-31", summary)
    assert path.exists()


def test_write_paper_summary_valid_json(tmp_path):
    summary = {"schema_version": "paper.summary.v1", "total_return": 0.12}
    path = write_paper_summary(tmp_path, "2024-01-01", "2024-01-31", summary)
    data = json.loads(path.read_text())
    assert data["total_return"] == 0.12
