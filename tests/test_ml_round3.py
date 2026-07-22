"""Tests für Round-3 ML-Erweiterungen.

- Adversarial Validation
- Feature Clustering
- Model Registry
- Bayesian Ensemble
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Adversarial Validation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature Clustering
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


def test_registry_register_and_list(tmp_path):
    import pytest

    pytest.importorskip("src.assembled_core.ml.model_registry")
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
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
    import pytest

    pytest.importorskip("src.assembled_core.ml.model_registry")
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
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
    import pytest

    pytest.importorskip("src.assembled_core.ml.model_registry")
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
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
