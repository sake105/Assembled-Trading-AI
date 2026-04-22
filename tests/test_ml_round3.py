"""Tests für Round-3 ML-Erweiterungen.

- Adversarial Validation
- Feature Clustering
- Model Registry
- Bayesian Ensemble
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# Adversarial Validation
# ---------------------------------------------------------------------------

def test_adversarial_no_shift():
    """Identische Verteilungen → AUC nahe 0.5."""
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.adversarial_validation import run_adversarial_validation

    rng = np.random.default_rng(42)
    n = 400
    X1 = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    X2 = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})

    result = run_adversarial_validation(X1, X2)
    assert result.n_train == n
    assert result.n_test == n
    # Same distribution → AUC should be near 0.5
    assert 0.40 < result.auc < 0.65


def test_adversarial_strong_shift():
    """Stark verschiedene Verteilungen → hohe AUC."""
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.adversarial_validation import run_adversarial_validation

    rng = np.random.default_rng(42)
    n = 400
    X1 = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    X2 = pd.DataFrame({
        "a": rng.standard_normal(n) + 3.0,  # massiver Shift
        "b": rng.standard_normal(n) * 3.0,
    })
    result = run_adversarial_validation(X1, X2)
    assert result.auc > 0.85
    assert "SHIFT" in result.interpret()


def test_adversarial_sample_weights():
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.adversarial_validation import sample_weight_from_adversarial

    rng = np.random.default_rng(7)
    n = 200
    X_train = pd.DataFrame({"a": rng.standard_normal(n)})
    X_test = pd.DataFrame({"a": rng.standard_normal(n) + 1.5})

    weights = sample_weight_from_adversarial(X_train, X_test, max_weight=5.0)
    assert len(weights) == n
    assert (weights >= 0).all()
    assert (weights <= 5.0).all()


# ---------------------------------------------------------------------------
# Feature Clustering
# ---------------------------------------------------------------------------

def test_cluster_by_correlation_separates_groups():
    from src.assembled_core.ml.feature_clustering import cluster_features_by_correlation

    pytest.importorskip("scipy")

    rng = np.random.default_rng(1)
    n = 200
    # Group 1: stark korreliert
    base1 = rng.standard_normal(n)
    g1a = base1 + 0.01 * rng.standard_normal(n)
    g1b = base1 + 0.01 * rng.standard_normal(n)
    # Group 2: stark korreliert, aber unabhängig von Group 1
    base2 = rng.standard_normal(n)
    g2a = base2 + 0.01 * rng.standard_normal(n)
    g2b = base2 + 0.01 * rng.standard_normal(n)

    X = pd.DataFrame({"g1a": g1a, "g1b": g1b, "g2a": g2a, "g2b": g2b})
    result = cluster_features_by_correlation(X, distance_threshold=0.3)

    # Sollte 2 Cluster erkennen
    assert result.n_clusters == 2
    cluster_members = {frozenset(v) for v in result.clusters.values()}
    assert {frozenset(["g1a", "g1b"]), frozenset(["g2a", "g2b"])} == cluster_members


def test_cluster_ic_selection():
    from src.assembled_core.ml.feature_clustering import (
        cluster_features_by_correlation,
        select_features_by_cluster_ic,
    )
    pytest.importorskip("scipy")

    rng = np.random.default_rng(3)
    n = 300
    # IC: weak=0, strong=0.5 (high IC)
    noise = rng.standard_normal(n)
    y = pd.Series(rng.standard_normal(n))
    weak = 0.05 * y + noise
    strong = 0.7 * y + 0.3 * rng.standard_normal(n)
    # Stark korrelierter Cluster mit weak + strong
    X = pd.DataFrame({"weak": weak, "strong": strong})

    cluster_result = cluster_features_by_correlation(X, distance_threshold=0.9)
    cluster_result = select_features_by_cluster_ic(X, y, cluster_result)

    # Wenn sie in einem Cluster sind, sollte 'strong' gewählt werden
    if cluster_result.n_clusters == 1:
        assert list(cluster_result.representatives.values())[0] == "strong"


def test_clustered_mda():
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from sklearn.ensemble import RandomForestRegressor

    from src.assembled_core.ml.feature_clustering import (
        cluster_features_by_correlation,
        clustered_mda,
    )

    rng = np.random.default_rng(5)
    n = 300
    X = pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.standard_normal(n),
    })
    y = pd.Series(2.0 * X["a"] + rng.normal(0, 0.5, n))

    model = RandomForestRegressor(n_estimators=30, random_state=0)
    model.fit(X.values, y.values)

    cluster_result = cluster_features_by_correlation(X, distance_threshold=0.3)
    mda = clustered_mda(model, X, y, cluster_result, n_repeats=2)
    assert len(mda) == cluster_result.n_clusters
    # Mindestens ein Cluster sollte positive MDA haben
    assert any(v > 0 for v in mda.values())


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def test_registry_register_and_list(tmp_path):
    pytest.importorskip("joblib")
    from sklearn.linear_model import Ridge

    from src.assembled_core.ml.model_registry import ModelRegistry

    registry = ModelRegistry(base_dir=tmp_path)
    model1 = Ridge()
    model2 = Ridge(alpha=2.0)

    r1 = registry.register(model1, model_id="test_model", metrics={"ic": 0.1})
    r2 = registry.register(model2, model_id="test_model", metrics={"ic": 0.12})

    assert r1.version == 1
    assert r2.version == 2
    assert r1.status == "candidate"
    assert r2.status == "candidate"

    versions = registry.list_versions("test_model")
    assert len(versions) == 2


def test_registry_approval_deployment_workflow(tmp_path):
    pytest.importorskip("joblib")
    from sklearn.linear_model import Ridge

    from src.assembled_core.ml.model_registry import ModelRegistry

    registry = ModelRegistry(base_dir=tmp_path)
    registry.register(Ridge(), model_id="m", metrics={"ic": 0.1})
    registry.register(Ridge(alpha=2.0), model_id="m", metrics={"ic": 0.15})

    # v2 deployen ohne Approval → Fehler
    with pytest.raises(ValueError, match="nicht approved"):
        registry.promote_to_deployed("m", 2)

    # Approve + Deploy
    registry.approve("m", 2, approver="test_user")
    registry.promote_to_deployed("m", 2)

    # deployed.joblib sollte existieren
    assert (tmp_path / "m" / "deployed.joblib").exists()

    # load_deployed funktioniert
    model = registry.load_deployed("m")
    assert model is not None

    # v1 deployen (second promote): v2 wird archived
    registry.approve("m", 1)
    registry.promote_to_deployed("m", 1)
    versions = registry.list_versions("m")
    by_ver = {v.version: v for v in versions}
    assert by_ver[2].status == "archived"
    assert by_ver[1].status == "deployed"


def test_registry_rollback(tmp_path):
    pytest.importorskip("joblib")
    from sklearn.linear_model import Ridge
    from src.assembled_core.ml.model_registry import ModelRegistry

    registry = ModelRegistry(base_dir=tmp_path)
    registry.register(Ridge(), model_id="m", metrics={"ic": 0.1})
    registry.register(Ridge(alpha=2.0), model_id="m", metrics={"ic": 0.12})
    registry.approve("m", 2)
    registry.promote_to_deployed("m", 2)

    # Rollback zu v1
    registry.rollback("m", 1)
    deployed = registry.load_deployed("m")
    assert deployed is not None

    by_ver = {v.version: v for v in registry.list_versions("m")}
    assert by_ver[1].status == "deployed"
    assert by_ver[2].status == "archived"


# ---------------------------------------------------------------------------
# Bayesian Ensemble
# ---------------------------------------------------------------------------

def test_bma_weights_softmax():
    from src.assembled_core.ml.bayesian_ensemble import compute_bma_weights

    scores = {"a": -0.1, "b": -0.5, "c": -2.0}  # neg_mse: higher = better
    weights = compute_bma_weights(scores, temperature=1.0, score_type="neg_mse")
    assert sum(weights.values()) == pytest.approx(1.0)
    # Best model should get highest weight
    assert weights["a"] > weights["b"] > weights["c"]


def test_bma_weights_temperature():
    from src.assembled_core.ml.bayesian_ensemble import compute_bma_weights

    scores = {"a": 0.9, "b": 0.3, "c": -0.5}
    # Niedrige Temperatur → Winner-takes-all
    w_low = compute_bma_weights(scores, temperature=0.1, score_type="neg_mse")
    # Hohe Temperatur → flacher
    w_high = compute_bma_weights(scores, temperature=10.0, score_type="neg_mse")
    assert w_low["a"] > w_high["a"]


def test_bma_training_and_predict():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Lasso, Ridge

    from src.assembled_core.ml.bayesian_ensemble import run_bayesian_ensemble

    rng = np.random.default_rng(11)
    n = 400
    X = pd.DataFrame({"f1": rng.standard_normal(n), "f2": rng.standard_normal(n)})
    y = pd.Series(X["f1"] + 0.5 * X["f2"] + rng.normal(0, 0.3, n))

    result = run_bayesian_ensemble(
        X_train=X.iloc[:200], y_train=y.iloc[:200],
        X_val=X.iloc[200:300], y_val=y.iloc[200:300],
        model_factories={
            "ridge": lambda: Ridge(),
            "lasso": lambda: Lasso(alpha=0.1),
        },
        score_type="neg_mse",
    )

    assert len(result.fitted_models) == 2
    assert sum(result.model_weights.values()) == pytest.approx(1.0)

    preds = result.predict(X.iloc[300:])
    assert len(preds) == 100
