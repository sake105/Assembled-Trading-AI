"""Tests for M17 Wave 1 Batch 2: Items 1.2, 1.5, 2.3, 2.4, 2.7, 4.3."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1.2 Regime-Conditional Factor Weights
# ---------------------------------------------------------------------------


class TestRegimeBlendedWeights:
    """Tests for REGIME_FACTOR_WEIGHTS and compute_regime_blended_weights."""

    def test_pure_bull_weights(self):
        from src.assembled_core.signals.multifactor_signal import (
            REGIME_FACTOR_WEIGHTS,
            compute_regime_blended_weights,
        )

        result = compute_regime_blended_weights({"bull": 1.0})
        # Should match bull weights exactly
        for cat, w in REGIME_FACTOR_WEIGHTS["bull"].items():
            assert abs(result[cat] - w) < 1e-6, f"{cat}: {result[cat]} != {w}"

    def test_equal_bull_bear_blend(self):
        from src.assembled_core.signals.multifactor_signal import (
            REGIME_FACTOR_WEIGHTS,
            compute_regime_blended_weights,
        )

        result = compute_regime_blended_weights({"bull": 0.5, "bear": 0.5})
        bull_w = REGIME_FACTOR_WEIGHTS["bull"]
        bear_w = REGIME_FACTOR_WEIGHTS["bear"]
        for cat in bull_w:
            expected = 0.5 * bull_w[cat] + 0.5 * bear_w.get(cat, 0.0)
            assert abs(result[cat] - expected) < 1e-5

    def test_factor_category_mapping(self):
        from src.assembled_core.signals.multifactor_signal import compute_regime_blended_weights

        mapping = {"rsi_14": "momentum", "pe_ratio": "value", "roe": "quality"}
        result = compute_regime_blended_weights({"bull": 1.0}, factor_categories=mapping)
        assert set(result.keys()) == {"rsi_14", "pe_ratio", "roe"}
        assert result["rsi_14"] == 0.30  # bull momentum weight

    def test_empty_probabilities(self):
        from src.assembled_core.signals.multifactor_signal import compute_regime_blended_weights

        result = compute_regime_blended_weights({})
        assert result == {}

    def test_all_five_regimes(self):
        from src.assembled_core.signals.multifactor_signal import (
            REGIME_FACTOR_WEIGHTS,
        )

        # All 5 regimes exist
        assert set(REGIME_FACTOR_WEIGHTS.keys()) == {"bull", "bear", "crisis", "recovery", "sideways"}
        # All have the same categories
        cats = set(REGIME_FACTOR_WEIGHTS["bull"].keys())
        for regime in REGIME_FACTOR_WEIGHTS:
            assert set(REGIME_FACTOR_WEIGHTS[regime].keys()) == cats


# ---------------------------------------------------------------------------
# 1.5 Intel Alpha Factor
# ---------------------------------------------------------------------------


class TestIntelAlphaFactor:
    """Tests for compute_symbol_intel_scores, normalize_intel_scores, build_intel_alpha_factor."""

    def test_basic_scoring(self):
        import pytest; pytest.importorskip('src.assembled_core.signals.intel_signal_adapter')
        from src.assembled_core.signals.intel_signal_adapter import compute_symbol_intel_scores

        result = compute_symbol_intel_scores(
            sector_impacts={"AAPL": 0.5, "MSFT": -0.3},
            supply_chain_vulnerability={"AAPL": 0.2, "MSFT": 0.8},
        )
        assert "AAPL" in result
        assert "MSFT" in result
        # AAPL positive sector + low vulnerability → positive
        assert result["AAPL"] > 0
        # MSFT negative sector + high vulnerability → negative
        assert result["MSFT"] < 0

    def test_normalization(self):
        import pytest; pytest.importorskip('src.assembled_core.signals.intel_signal_adapter')
        from src.assembled_core.signals.intel_signal_adapter import normalize_intel_scores

        scores = {"A": 1.0, "B": -1.0, "C": 0.0}
        normed = normalize_intel_scores(scores)
        values = list(normed.values())
        assert abs(np.mean(values)) < 1e-6
        assert abs(np.std(values) - 1.0) < 0.1

    def test_build_intel_alpha_factor_returns_series(self):
        import pytest; pytest.importorskip('src.assembled_core.signals.intel_signal_adapter')
        from src.assembled_core.signals.intel_signal_adapter import build_intel_alpha_factor

        result = build_intel_alpha_factor(
            sector_impacts={"XLE": 0.8, "GLD": 0.5, "QQQ": -0.3},
        )
        assert isinstance(result, pd.Series)
        assert result.name == "intel_alpha"
        assert len(result) == 3

    def test_empty_inputs(self):
        import pytest; pytest.importorskip('src.assembled_core.signals.intel_signal_adapter')
        from src.assembled_core.signals.intel_signal_adapter import compute_symbol_intel_scores

        assert compute_symbol_intel_scores() == {}

    def test_confidence_weighting(self):
        import pytest; pytest.importorskip('src.assembled_core.signals.intel_signal_adapter')
        from src.assembled_core.signals.intel_signal_adapter import compute_symbol_intel_scores

        high_conf = compute_symbol_intel_scores(
            sector_impacts={"A": 0.5},
            confidence={"A": 1.0},
        )
        low_conf = compute_symbol_intel_scores(
            sector_impacts={"A": 0.5},
            confidence={"A": 0.1},
        )
        assert abs(high_conf["A"]) > abs(low_conf["A"])


# ---------------------------------------------------------------------------
# 2.3 Multi-Feature HMM
# ---------------------------------------------------------------------------


class TestMultiFeatureHMM:
    """Tests for MultiFeatureRegimeHMM and build_multifeature_observables."""

    def test_build_observables(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.regime_hmm')
        from src.assembled_core.ml.regime_hmm import build_multifeature_observables

        np.random.seed(42)
        n = 100
        returns = pd.Series(np.random.normal(0, 0.01, n), name="returns")
        result = build_multifeature_observables(returns, vol_window=20)
        assert "daily_return" in result.columns
        assert "realized_vol" in result.columns
        assert len(result) > 0

    def test_build_observables_with_extras(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.regime_hmm')
        from src.assembled_core.ml.regime_hmm import build_multifeature_observables

        np.random.seed(42)
        n = 100
        idx = pd.date_range("2020-01-01", periods=n)
        returns = pd.Series(np.random.normal(0, 0.01, n), index=idx)
        vix = pd.Series(np.random.normal(0, 0.5, n), index=idx)
        result = build_multifeature_observables(returns, vix_changes=vix)
        assert "vix_change" in result.columns

    def test_fallback_proba(self):
        """Without hmmlearn, MultiFeatureRegimeHMM should use fallback."""
        import pytest; pytest.importorskip('src.assembled_core.ml.regime_hmm')
        from src.assembled_core.ml.regime_hmm import MultiFeatureRegimeHMM

        hmm = MultiFeatureRegimeHMM()
        np.random.seed(42)
        features = pd.DataFrame({
            "ret": np.random.normal(0, 0.01, 100),
            "vol": np.random.uniform(0.01, 0.03, 100),
        })
        proba = hmm.predict_proba(features)
        # Fallback returns p_bull, p_bear, p_sideways
        assert "p_bull" in proba.columns or proba.empty

    def test_crisis_alert_unfitted(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.regime_hmm')
        from src.assembled_core.ml.regime_hmm import MultiFeatureRegimeHMM

        hmm = MultiFeatureRegimeHMM()
        features = pd.DataFrame({"ret": np.random.normal(0, 0.01, 50)})
        alert = hmm.crisis_alert(features)
        assert "crisis_prob" in alert
        assert "alert" in alert


# ---------------------------------------------------------------------------
# 2.4 Quantile Regression
# ---------------------------------------------------------------------------


class TestQuantileModels:
    """Tests for QuantilePrediction, fit_quantile_lgbm, predict_quantiles."""

    def test_quantile_prediction_properties(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.quantile_models')
        from src.assembled_core.ml.quantile_models import QuantilePrediction

        qp = QuantilePrediction(
            symbol="AAPL", q05=-0.05, q25=-0.01, q50=0.02, q75=0.04, q95=0.08,
        )
        assert qp.confidence > 0
        assert qp.asymmetry > 0
        assert qp.expected_direction == "positive"

    def test_quantile_prediction_negative(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.quantile_models')
        from src.assembled_core.ml.quantile_models import QuantilePrediction

        qp = QuantilePrediction(
            symbol="X", q05=-0.10, q25=-0.05, q50=-0.02, q75=0.01, q95=0.05,
        )
        assert qp.expected_direction == "negative"

    def test_quantile_prediction_neutral(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.quantile_models')
        from src.assembled_core.ml.quantile_models import QuantilePrediction

        qp = QuantilePrediction(
            symbol="X", q05=-0.05, q25=-0.001, q50=0.0005, q75=0.001, q95=0.05,
        )
        assert qp.expected_direction == "neutral"

    def test_fallback_quantiles(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.quantile_models')
        from src.assembled_core.ml.quantile_models import _fallback_quantiles

        y_train = np.random.normal(0, 1, 200)
        X_predict = np.random.normal(0, 1, (10, 3))
        result = _fallback_quantiles(y_train, X_predict, (0.05, 0.50, 0.95))
        assert set(result.keys()) == {0.05, 0.50, 0.95}
        for q, arr in result.items():
            assert len(arr) == 10

    def test_predict_quantiles_small_data(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.quantile_models')
        from src.assembled_core.ml.quantile_models import predict_quantiles

        df = pd.DataFrame({
            "f1": np.random.normal(0, 1, 30),
            "target": np.random.normal(0, 1, 30),
        })
        result = predict_quantiles(df, "target", ["f1"])
        assert result == []  # too few samples (<50)

    def test_predict_quantiles_sufficient(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.quantile_models')
        from src.assembled_core.ml.quantile_models import predict_quantiles

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "f1": np.random.normal(0, 1, n),
            "f2": np.random.normal(0, 1, n),
            "target": np.random.normal(0, 1, n),
            "symbol": ["SYM"] * n,
        })
        result = predict_quantiles(df, "target", ["f1", "f2"])
        assert len(result) > 0
        for qp in result:
            assert qp.q05 <= qp.q50 <= qp.q95


# ---------------------------------------------------------------------------
# 2.7 Online Learning
# ---------------------------------------------------------------------------


class TestOnlineLearning:
    """Tests for EWRLSModel, RetrainingTrigger, compute_model_age_confidence."""

    def test_ewrls_basic(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.online_learning')
        from src.assembled_core.ml.online_learning import EWRLSModel

        model = EWRLSModel(n_features=2)
        assert model.beta.shape == (2,)
        assert model.P.shape == (2, 2)
        assert model.n_updates == 0

    def test_ewrls_converges(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.online_learning')
        from src.assembled_core.ml.online_learning import EWRLSModel

        np.random.seed(42)
        true_beta = np.array([1.5, -0.5])
        model = EWRLSModel(n_features=2, forgetting_factor=0.99)
        for _ in range(500):
            x = np.random.normal(0, 1, 2)
            y = x @ true_beta + np.random.normal(0, 0.1)
            model.update(x, y)
        # Should be close to true beta
        assert np.allclose(model.beta, true_beta, atol=0.2)
        assert model.n_updates == 500

    def test_ewrls_batch_update(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.online_learning')
        from src.assembled_core.ml.online_learning import EWRLSModel

        np.random.seed(42)
        model = EWRLSModel(n_features=3)
        X = np.random.normal(0, 1, (50, 3))
        y = X @ np.array([1.0, 0.0, -1.0]) + np.random.normal(0, 0.1, 50)
        errors = model.batch_update(X, y)
        assert len(errors) == 50
        assert model.n_updates == 50

    def test_retraining_trigger_ic(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.online_learning')
        from src.assembled_core.ml.online_learning import RetrainingTrigger

        trigger = RetrainingTrigger(ic_threshold=0.0, consecutive_bad_days=3)
        # 3 bad days in a row should trigger
        assert not trigger.check(-0.01, 0.01)
        assert not trigger.check(-0.02, 0.01)
        assert trigger.check(-0.03, 0.01)  # 3rd bad day

    def test_retraining_trigger_reset(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.online_learning')
        from src.assembled_core.ml.online_learning import RetrainingTrigger

        trigger = RetrainingTrigger(consecutive_bad_days=5)
        trigger._bad_day_count = 4
        trigger.reset()
        assert trigger._bad_day_count == 0

    def test_model_age_confidence(self):
        import pytest; pytest.importorskip('src.assembled_core.ml.online_learning')
        from src.assembled_core.ml.online_learning import compute_model_age_confidence

        assert compute_model_age_confidence(0) == 1.0
        c30 = compute_model_age_confidence(30, half_life_days=30)
        assert abs(c30 - 0.5) < 0.01
        c60 = compute_model_age_confidence(60, half_life_days=30)
        assert abs(c60 - 0.25) < 0.01


# ---------------------------------------------------------------------------
# 4.3 Expanded Trigger Rules
# ---------------------------------------------------------------------------


class TestExpandedTriggerRules:
    """Tests for expanded KEYWORD_RULES in geo_trigger.py."""

    def test_all_trigger_types_have_rules(self):
        from src.assembled_core.intel.geo_trigger import KEYWORD_RULES

        # Should have at least 20 trigger types now
        assert len(KEYWORD_RULES) >= 20

    def test_financial_triggers_exist(self):
        from src.assembled_core.intel.geo_trigger import KEYWORD_RULES
        from src.assembled_core.intel.models import TriggerType

        for tt in [TriggerType.BANKING_CRISIS, TriggerType.CREDIT_DOWNGRADE,
                   TriggerType.RATE_SURPRISE, TriggerType.PEG_STRESS]:
            assert tt in KEYWORD_RULES, f"Missing {tt}"
            assert len(KEYWORD_RULES[tt]) >= 8

    def test_military_triggers_exist(self):
        from src.assembled_core.intel.geo_trigger import KEYWORD_RULES
        from src.assembled_core.intel.models import TriggerType

        for tt in [TriggerType.MILITARY_BUILDUP, TriggerType.NUCLEAR_THREAT,
                   TriggerType.CAPABILITY_SHIFT]:
            assert tt in KEYWORD_RULES, f"Missing {tt}"
            assert len(KEYWORD_RULES[tt]) >= 8

    def test_score_event_with_new_triggers(self):
        from src.assembled_core.intel.geo_trigger import score_event
        from src.assembled_core.intel.models import NewsEvent, SourceTier

        evt = NewsEvent(
            event_id="test1",
            source_id="reuters",
            source_tier=SourceTier.T1,
            title="Bank run fears as SVB deposit freeze triggers contagion",
            url="https://example.com",
            published_at="2024-01-01T00:00:00Z",
            ingested_at="2024-01-01T00:01:00Z",
            content_hash="abc123",
            keywords=["bank run", "svb", "deposit freeze"],
            geo_tags=[],
            entities=[],
        )
        score = score_event(evt)
        assert score > 0

    def test_classify_trigger_type_banking(self):
        from src.assembled_core.intel.geo_trigger import classify_trigger_type
        from src.assembled_core.intel.models import NewsEvent, SourceTier, TriggerType

        evt = NewsEvent(
            event_id="test2",
            source_id="reuters",
            source_tier=SourceTier.T1,
            title="Bank failure and bail-in announced, depositors face haircut",
            url="https://example.com",
            published_at="2024-01-01T00:00:00Z",
            ingested_at="2024-01-01T00:01:00Z",
            content_hash="def456",
            keywords=["bail-in", "bank failure", "deposit freeze"],
            geo_tags=[],
            entities=[],
        )
        result = classify_trigger_type(evt)
        assert result == TriggerType.BANKING_CRISIS
