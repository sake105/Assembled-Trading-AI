"""Robustness tests for ML model registry + regime HMM observability.

Covers Diagnostik findings A16, A17, A18, A43, A46. Each assertion targets the
SPECIFIC new behaviour introduced by the fix so the test would fail pre-fix.

- A16: registry.json writes are atomic (tmp + os.replace), no leftover .tmp.
- A43: register_model() resets BOTH _registry_cache and _registry_mtime.
- A17: RegimeHMM.partial_update exposes last_partial_update_ok (False on fit fail).
- A18: MultiFeatureRegimeHMM.predict_regime exposes last_predict_degraded on fallback.
- A46: verify_model_hash fail-open on empty registry emits a WARNING.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml import model_registry as mr
from src.assembled_core.ml.regime_hmm import (
    HMMLEARN_AVAILABLE,
    MultiFeatureRegimeHMM,
    RegimeHMM,
)


# ---------------------------------------------------------------------------
# A16 — atomic registry.json write (no truncation on crash, no leftover .tmp)
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestA16AtomicWrite:
    def test_save_meta_uses_os_replace(self, tmp_path, monkeypatch):
        """_save_meta must route through os.replace (atomic rename), not a
        direct truncating write."""
        registry = mr.ModelRegistry(base_dir=tmp_path)

        calls: list[tuple] = []
        real_replace = os.replace

        def _spy_replace(src, dst, *a, **k):
            calls.append((str(src), str(dst)))
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(mr.os, "replace", _spy_replace)

        mv = mr.ModelVersion(
            model_id="m1",
            version=1,
            status="candidate",
            metrics={"sharpe": 1.0},
            path=tmp_path / "m1" / "v1.joblib",
            sha256="deadbeef",
        )
        registry._save_meta("m1", [mv])

        assert calls, "expected os.replace to be used for an atomic write"
        # The destination must be the registry.json meta file.
        assert calls[-1][1].endswith("registry.json")

    def test_save_meta_roundtrips_and_leaves_no_tmp(self, tmp_path):
        registry = mr.ModelRegistry(base_dir=tmp_path)
        mv = mr.ModelVersion(
            model_id="m2",
            version=3,
            status="approved",
            metrics={},
            path=tmp_path / "m2" / "v3.joblib",
            sha256="abc123",
        )
        registry._save_meta("m2", [mv])

        meta = tmp_path / "m2" / "registry.json"
        assert meta.exists()
        data = json.loads(meta.read_text(encoding="utf-8"))
        assert data["versions"][0]["version"] == 3
        # A16: no .tmp artifact must survive a completed write.
        leftovers = list((tmp_path / "m2").glob("*.tmp"))
        assert leftovers == [], f"unexpected leftover tmp file(s): {leftovers}"

    def test_module_register_model_atomic_no_tmp(self, tmp_path, monkeypatch):
        """The module-level register_model() must also write atomically."""
        reg_path = tmp_path / "models" / "registry.json"
        monkeypatch.setattr(mr, "_REGISTRY_PATH", reg_path)
        monkeypatch.setattr(mr, "_registry_cache", None)
        monkeypatch.setattr(mr, "_registry_mtime", None)

        model_file = tmp_path / "model_a.joblib"
        model_file.write_bytes(b"dummy-model-bytes")

        replace_calls: list = []
        real_replace = os.replace

        def _spy_replace(src, dst, *a, **k):
            replace_calls.append((str(src), str(dst)))
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(mr.os, "replace", _spy_replace)

        mr.register_model(model_file)

        assert reg_path.exists()
        assert replace_calls, "register_model must write registry.json atomically"
        assert list((tmp_path / "models").glob("*.tmp")) == []


# ---------------------------------------------------------------------------
# A43 — register_model resets BOTH cache and mtime
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestA43CacheInvalidation:
    def test_register_model_resets_cache_and_mtime(self, tmp_path, monkeypatch):
        reg_path = tmp_path / "models" / "registry.json"
        monkeypatch.setattr(mr, "_REGISTRY_PATH", reg_path)
        # Pre-seed a stale cache + mtime as if a prior load had happened.
        monkeypatch.setattr(mr, "_registry_cache", {"stale.joblib": {"sha256": "x"}})
        monkeypatch.setattr(mr, "_registry_mtime", 12345.0)

        model_file = tmp_path / "model_b.joblib"
        model_file.write_bytes(b"another-dummy-model")

        mr.register_model(model_file)

        # A43: BOTH must be reset to None so the next load re-reads from disk;
        # pre-fix only _registry_cache was cleared, leaving a stale mtime that
        # could serve the stale cache after a fast write.
        assert mr._registry_cache is None
        assert mr._registry_mtime is None


# ---------------------------------------------------------------------------
# A46 — verify_model_hash fail-open WARNING on empty/missing registry
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestA46FailOpenObservable:
    def test_empty_registry_returns_true_and_warns(self, tmp_path, monkeypatch, caplog):
        # Point the registry at a non-existent path → empty registry.
        reg_path = tmp_path / "models" / "registry.json"
        monkeypatch.setattr(mr, "_REGISTRY_PATH", reg_path)
        monkeypatch.setattr(mr, "_registry_cache", None)
        monkeypatch.setattr(mr, "_registry_mtime", None)

        model_file = tmp_path / "some_model.joblib"
        model_file.write_bytes(b"x")

        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.ml.model_registry"
        ):
            result = mr.verify_model_hash(model_file)

        # Cold-start contract preserved (fail-open) ...
        assert result is True
        # ... but the inactive posture must be observable at WARNING level.
        warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "INACTIVE" in r.getMessage()
        ]
        assert warnings, (
            "expected a WARNING that the integrity check is INACTIVE; "
            f"got records: {[r.getMessage() for r in caplog.records]}"
        )


# ---------------------------------------------------------------------------
# A17 — partial_update observability flag
# ---------------------------------------------------------------------------


def _fit_baseline_hmm() -> RegimeHMM:
    rng = np.random.default_rng(0)
    baseline = pd.Series(rng.normal(0, 0.01, 400))
    model = RegimeHMM(n_regimes=2)
    model.fit(baseline)
    return model


@pytest.mark.fast
@pytest.mark.skipif(not HMMLEARN_AVAILABLE, reason="hmmlearn not installed")
class TestA17PartialUpdateFlag:
    def test_success_sets_flag_true_and_returns_self(self):
        model = _fit_baseline_hmm()
        rng = np.random.default_rng(1)
        new_data = pd.Series(rng.normal(0, 0.01, 60))
        result = model.partial_update(new_data, min_samples=20)
        # Chaining contract preserved.
        assert result is model
        assert model.last_partial_update_ok is True

    def test_fit_exception_sets_flag_false_and_keeps_model(self, monkeypatch):
        model = _fit_baseline_hmm()
        old_model = model._model

        # Force the warm-start fit to raise so the except path runs.
        import src.assembled_core.ml.regime_hmm as rh

        def _boom(*a, **k):
            raise RuntimeError("forced warm-start failure")

        monkeypatch.setattr(rh.GaussianHMM, "fit", _boom)

        rng = np.random.default_rng(2)
        new_data = pd.Series(rng.normal(0, 0.01, 60))
        result = model.partial_update(new_data, min_samples=20)

        # return self preserved, but failure is now observable.
        assert result is model
        assert model.last_partial_update_ok is False
        # Stale model deliberately kept.
        assert model._model is old_model


# ---------------------------------------------------------------------------
# A18 — predict_regime degraded marker on fallback
# ---------------------------------------------------------------------------


def _fit_multifeature() -> MultiFeatureRegimeHMM:
    rng = np.random.default_rng(0)
    n = 300
    ret = rng.normal(0, 0.01, n)
    vol = pd.Series(ret).rolling(20, min_periods=10).std().bfill().values
    df = pd.DataFrame(
        {"daily_return": ret, "realized_vol": vol},
        index=pd.bdate_range("2020-01-01", periods=n),
    )
    model = MultiFeatureRegimeHMM(n_regimes=2, n_iter=20, n_seeds=2)
    model.fit(df)
    return model


@pytest.mark.fast
@pytest.mark.skipif(not HMMLEARN_AVAILABLE, reason="hmmlearn not installed")
class TestA18PredictDegradedMarker:
    def test_normal_path_not_degraded(self):
        model = _fit_multifeature()
        assert model._fitted, "fixture model failed to fit"
        rng = np.random.default_rng(3)
        n = 60
        ret = rng.normal(0, 0.01, n)
        vol = pd.Series(ret).rolling(20, min_periods=10).std().bfill().values
        df = pd.DataFrame(
            {"daily_return": ret, "realized_vol": vol},
            index=pd.bdate_range("2021-01-01", periods=n),
        )
        out = model.predict_regime(df)
        assert isinstance(out, pd.Series)
        assert model.last_predict_degraded is False

    def test_fallback_sets_degraded_true(self, monkeypatch):
        model = _fit_multifeature()
        assert model._fitted, "fixture model failed to fit"

        # Force the primary Viterbi predict to raise → fallback path.
        def _boom(*a, **k):
            raise RuntimeError("forced predict failure")

        monkeypatch.setattr(model._model, "predict", _boom)

        rng = np.random.default_rng(4)
        n = 60
        ret = rng.normal(0, 0.01, n)
        vol = pd.Series(ret).rolling(20, min_periods=10).std().bfill().values
        df = pd.DataFrame(
            {"daily_return": ret, "realized_vol": vol},
            index=pd.bdate_range("2022-01-01", periods=n),
        )
        out = model.predict_regime(df)

        # Return contract preserved (still a Series) ...
        assert isinstance(out, pd.Series)
        # ... and the silent vol-proxy substitution is now observable.
        assert model.last_predict_degraded is True

    def test_early_return_resets_stale_degraded_flag(self):
        # F-auditor-1/F-senior-1: a prior degraded call leaves the flag True;
        # a later not-fitted / empty-data early-return must NOT read that stale
        # value. The entry-reset clears it back to False. Without the reset this
        # asserts would fail (the early-return path would leave the stale True).
        model = MultiFeatureRegimeHMM(n_regimes=2, n_iter=20, n_seeds=2)
        assert not model._fitted, "expected an unfitted model for this case"
        model.last_predict_degraded = True  # simulate prior degraded call

        df = pd.DataFrame(
            {"daily_return": [0.0, 0.01], "realized_vol": [0.01, 0.02]},
            index=pd.bdate_range("2023-01-01", periods=2),
        )
        out = model.predict_regime(df)  # hits the not-fitted early-return guard

        assert isinstance(out, pd.Series)
        assert model.last_predict_degraded is False
