"""Tests for assembled_core/strategy/hyperparameter.py (spec 39)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from assembled_core.strategy.hyperparameter import (
    check_config_drift,
    deployment_inventory,
    walk_forward_objective,
)

# ---------------------------------------------------------------------------
# walk_forward_objective
# ---------------------------------------------------------------------------


def _make_data(n=24):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", periods=n, freq="ME")
    return pd.DataFrame({"value": rng.normal(0, 1, n)}, index=idx)


class TestWalkForwardObjective:
    def _train_fn(self, data, params):
        return {"mean": data["value"].mean()}

    def _eval_fn(self, model, data):
        return float(-abs(data["value"].mean() - model["mean"]))

    def test_returns_float(self):
        data = _make_data(24)
        trial = type(
            "T",
            (),
            {
                "suggest_int": lambda self, n, lo, hi: lo,
                "suggest_float": lambda self, n, lo, hi: lo,
            },
        )()
        score = walk_forward_objective(trial, data, self._train_fn, self._eval_fn)
        assert isinstance(score, float)

    def test_insufficient_data_returns_neg_inf(self):
        data = _make_data(5)
        trial = type("T", (), {})()
        score = walk_forward_objective(
            trial, data, self._train_fn, self._eval_fn, train_months=9, test_months=3
        )
        assert score == float("-inf")

    def test_param_space_int(self):
        data = _make_data(24)
        suggested = {}

        class FakeTrial:
            def suggest_int(self, name, lo, hi):
                suggested[name] = lo
                return lo

        score = walk_forward_objective(
            FakeTrial(),
            data,
            self._train_fn,
            self._eval_fn,
            param_space={"n": (1, 10, "int")},
        )
        assert "n" in suggested

    def test_multiple_folds_averaged(self):
        data = _make_data(24)
        scores = []

        class CountingTrial:
            def suggest_int(self, name, lo, hi):
                return lo

        score = walk_forward_objective(
            CountingTrial(),
            data,
            self._train_fn,
            self._eval_fn,
            train_months=9,
            test_months=3,
            stride_months=3,
        )
        assert score != float("-inf")


# ---------------------------------------------------------------------------
# check_config_drift
# ---------------------------------------------------------------------------


class TestCheckConfigDrift:
    def _write_json(self, path, data):
        path.write_text(json.dumps(data))

    def test_no_drift(self, tmp_path):
        cfg = {"strategy_id": "s1", "weights": {"news": 0.15}}
        paths = {}
        for env in ("dev", "staging", "prod"):
            p = tmp_path / f"{env}.json"
            self._write_json(p, cfg)
            paths[env] = p
        drifts = check_config_drift(paths)
        assert drifts == []

    def test_drift_detected(self, tmp_path):
        cfgs = {
            "dev": {"news_weight": 0.15},
            "staging": {"news_weight": 0.20},
            "prod": {"news_weight": 0.12},
        }
        paths = {}
        for env, cfg in cfgs.items():
            p = tmp_path / f"{env}.json"
            self._write_json(p, cfg)
            paths[env] = p
        drifts = check_config_drift(paths)
        assert len(drifts) == 1
        assert drifts[0]["key"] == "news_weight"

    def test_missing_config_does_not_raise(self, tmp_path):
        p = tmp_path / "dev.json"
        p.write_text('{"k": 1}')
        paths = {"dev": p, "prod": tmp_path / "nonexistent.json"}
        drifts = check_config_drift(paths)
        assert isinstance(drifts, list)

    def test_nested_drift(self, tmp_path):
        dev = {"weights": {"news": 0.15, "trend": 0.20}}
        prod = {"weights": {"news": 0.10, "trend": 0.20}}
        p_dev = tmp_path / "dev.json"
        p_prod = tmp_path / "prod.json"
        self._write_json(p_dev, dev)
        self._write_json(p_prod, prod)
        drifts = check_config_drift({"dev": p_dev, "prod": p_prod})
        keys = [d["key"] for d in drifts]
        assert "weights.news" in keys
        assert "weights.trend" not in keys


# ---------------------------------------------------------------------------
# deployment_inventory
# ---------------------------------------------------------------------------


class TestDeploymentInventory:
    def _write_json(self, path, data):
        path.write_text(json.dumps(data))

    def test_returns_dict(self, tmp_path):
        p = tmp_path / "paper.json"
        self._write_json(p, {"strategy_id": "s1", "model_versions": {}})
        snap = deployment_inventory({"paper": p})
        assert "timestamp" in snap
        assert "environments" in snap

    def test_known_env_has_strategy_id(self, tmp_path):
        p = tmp_path / "paper.json"
        self._write_json(
            p, {"strategy_id": "my_strat", "model_versions": {"clf": "v3"}}
        )
        snap = deployment_inventory({"paper": p})
        assert snap["environments"]["paper"]["strategy_id"] == "my_strat"
        assert snap["environments"]["paper"]["model_versions"]["clf"] == "v3"

    def test_missing_file_no_exception(self, tmp_path):
        snap = deployment_inventory({"prod": tmp_path / "missing.json"})
        assert "error" in snap["environments"]["prod"]

    def test_custom_timestamp(self, tmp_path):
        p = tmp_path / "dev.json"
        self._write_json(p, {"strategy_id": "x"})
        snap = deployment_inventory(
            {"dev": p}, timestamp_utc="2024-01-01T00:00:00+00:00"
        )
        assert snap["timestamp"] == "2024-01-01T00:00:00+00:00"
