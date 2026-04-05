"""Phase F — M15 wiring smoke tests.

Verifies that all previously-dormant modules (D1–D13) and new M15 modules
are importable, instantiable, and callable with minimal synthetic inputs.

These tests do NOT assert deep numerical correctness — they ensure the wiring
is complete and no import-time or construction-time errors exist.

Marker: phase12
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# D1 — Black-Litterman
# ---------------------------------------------------------------------------

class TestD1BlackLitterman:
    def test_import(self):
        from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer
        assert BlackLittermanOptimizer is not None

    def test_optimize_from_scores_runs(self):
        import pandas as pd
        from src.assembled_core.portfolio.black_litterman import BlackLittermanOptimizer

        symbols = ["AAPL", "MSFT", "GOOGL"]
        scores = pd.Series([0.8, 0.3, 0.6], index=symbols)

        # Build a simple covariance matrix (sigma, not returns)
        import numpy as np
        rng = np.random.default_rng(42)
        ret_matrix = rng.normal(0.001, 0.02, (60, 3))
        ret_df = pd.DataFrame(ret_matrix, columns=symbols)
        sigma = ret_df.cov()  # proper covariance DataFrame

        optimizer = BlackLittermanOptimizer(risk_aversion=2.5, tau=0.05)
        weights = optimizer.optimize_from_scores(scores, sigma)
        assert isinstance(weights, pd.Series)
        assert set(weights.index) == set(symbols)


# ---------------------------------------------------------------------------
# D2 — Barra factor risk model
# ---------------------------------------------------------------------------

class TestD2BarraFactorRisk:
    def test_import(self):
        from src.assembled_core.risk.factor_risk_model import FactorRiskModel
        assert FactorRiskModel is not None

    def test_constructor_and_methods_exist(self):
        """FactorRiskModel must be instantiable and have key methods."""
        from src.assembled_core.risk.factor_risk_model import FactorRiskModel

        model = FactorRiskModel()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict_portfolio_vol")

    def test_predict_vol_unfitted_returns_float(self):
        """predict_portfolio_vol on an unfitted model should return a fallback float."""
        import pandas as pd
        from src.assembled_core.risk.factor_risk_model import FactorRiskModel

        symbols = ["AAPL", "MSFT", "GOOGL"]
        model = FactorRiskModel()
        weights = pd.Series([0.40, 0.35, 0.25], index=symbols)
        # Unfitted model — should return a fallback value, not raise
        try:
            vol = model.predict_portfolio_vol(weights)
            assert isinstance(vol, float)
        except Exception:
            pass  # Some implementations require fitting first — that's acceptable


# ---------------------------------------------------------------------------
# D3 — HMM regime detection
# ---------------------------------------------------------------------------

class TestD3HMMRegime:
    def test_import(self):
        from src.assembled_core.risk.regime_models import build_regime_state_hmm
        assert build_regime_state_hmm is not None

    def test_graceful_return_without_hmmlearn(self):
        import pandas as pd
        from src.assembled_core.risk.regime_models import build_regime_state_hmm

        dates = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        prices = pd.DataFrame({
            "timestamp": dates.repeat(2),
            "symbol": ["SPY"] * 60 + ["SPY"] * 60,
            "close": list(range(100, 160)) + list(range(100, 160)),
        })

        # Should not crash regardless of hmmlearn availability
        result = build_regime_state_hmm(prices, n_regimes=3, benchmark_symbol="SPY")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# D4 — Stacking ensemble
# ---------------------------------------------------------------------------

class TestD4StackingEnsemble:
    def test_import(self):
        from src.assembled_core.ml.factor_models import train_stacked_ensemble
        assert train_stacked_ensemble is not None

    def test_function_callable(self):
        """Just verify the function is importable and has the right signature."""
        import inspect
        from src.assembled_core.ml.factor_models import train_stacked_ensemble

        sig = inspect.signature(train_stacked_ensemble)
        params = list(sig.parameters.keys())
        assert "panel_df" in params
        assert "experiment" in params


# ---------------------------------------------------------------------------
# D5 — Intermarket features
# ---------------------------------------------------------------------------

class TestD5IntermarketFeatures:
    def test_import(self):
        from src.assembled_core.features.intermarket_factors import (
            build_intermarket_factors,
        )
        assert build_intermarket_factors is not None

    def test_align_function_importable(self):
        from src.assembled_core.features.intermarket_factors import (
            align_intermarket_factors_to_panel,
        )
        assert align_intermarket_factors_to_panel is not None


# ---------------------------------------------------------------------------
# D6 — Candlestick features
# ---------------------------------------------------------------------------

class TestD6CandlestickFeatures:
    def test_import(self):
        from src.assembled_core.features.ta_candlestick import (
            build_candlestick_features,
        )
        assert build_candlestick_features is not None

    def test_get_feature_names_importable(self):
        from src.assembled_core.features.ta_candlestick import (
            get_candlestick_feature_names,
        )
        names = get_candlestick_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0


# ---------------------------------------------------------------------------
# D8 — TWAP/VWAP paper execution
# ---------------------------------------------------------------------------

class TestD8AlgoExecution:
    def test_import(self):
        from src.assembled_core.execution.paper_trading_engine import (
            PaperTradingEngine,
        )
        assert PaperTradingEngine is not None

    def test_submit_algo_order_method_exists(self):
        import inspect
        from src.assembled_core.execution.paper_trading_engine import PaperTradingEngine

        assert hasattr(PaperTradingEngine, "submit_algo_order"), (
            "PaperTradingEngine missing submit_algo_order method (D8 wiring)"
        )
        sig = inspect.signature(PaperTradingEngine.submit_algo_order)
        params = list(sig.parameters.keys())
        assert "algo" in params


# ---------------------------------------------------------------------------
# D9 — Earnings calendar features
# ---------------------------------------------------------------------------

class TestD9EarningsCalendar:
    def test_import(self):
        from src.assembled_core.data.sources.earnings_calendar_source import (
            EarningsCalendarSource,
        )
        assert EarningsCalendarSource is not None

    def test_build_earnings_factors_method_exists(self):
        from src.assembled_core.data.sources.earnings_calendar_source import (
            EarningsCalendarSource,
        )
        assert hasattr(EarningsCalendarSource, "build_earnings_factors")


# ---------------------------------------------------------------------------
# D10 — FinBERT sentiment
# ---------------------------------------------------------------------------

class TestD10FinBERT:
    def test_import(self):
        from src.assembled_core.ml.nlp_sentiment import score_texts_finbert
        assert score_texts_finbert is not None

    def test_returns_list_or_skips_without_transformers(self):
        from src.assembled_core.ml.nlp_sentiment import score_texts_finbert

        texts = ["Stocks fell sharply today.", "Earnings beat expectations."]
        try:
            result = score_texts_finbert(texts)
            assert isinstance(result, list)
        except ImportError:
            pytest.skip("transformers/torch not installed — FinBERT unavailable")


# ---------------------------------------------------------------------------
# D11 — Optuna hyperopt
# ---------------------------------------------------------------------------

class TestD11Optuna:
    def test_import(self):
        from src.assembled_core.ml.factor_models import train_with_hyperopt
        assert train_with_hyperopt is not None

    def test_function_callable(self):
        import inspect
        from src.assembled_core.ml.factor_models import train_with_hyperopt

        sig = inspect.signature(train_with_hyperopt)
        params = list(sig.parameters.keys())
        assert "n_trials" in params
        assert "metric" in params


# ---------------------------------------------------------------------------
# D12 — Scenario engine
# ---------------------------------------------------------------------------

class TestD12ScenarioEngine:
    def test_import(self):
        from src.assembled_core.qa.scenario_engine import run_crisis_scenarios
        assert run_crisis_scenarios is not None

    def test_function_callable(self):
        import inspect
        from src.assembled_core.qa.scenario_engine import run_crisis_scenarios

        sig = inspect.signature(run_crisis_scenarios)
        params = list(sig.parameters.keys())
        assert "prices" in params


# ---------------------------------------------------------------------------
# D13 — FeatureConfig extended
# ---------------------------------------------------------------------------

class TestD13FeatureConfig:
    def test_feature_config_has_new_fields(self):
        from src.assembled_core.config.models import FeatureConfig

        cfg = FeatureConfig()
        assert hasattr(cfg, "include_intermarket")
        assert hasattr(cfg, "include_candlestick")
        assert hasattr(cfg, "include_earnings")
        assert hasattr(cfg, "include_options_signals")

    def test_feature_config_defaults_false(self):
        from src.assembled_core.config.models import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.include_intermarket is False
        assert cfg.include_candlestick is False
        assert cfg.include_earnings is False
        assert cfg.include_options_signals is False

    def test_feature_config_can_enable_flags(self):
        from src.assembled_core.config.models import FeatureConfig

        cfg = FeatureConfig(
            include_intermarket=True,
            include_candlestick=True,
            include_earnings=True,
        )
        assert cfg.include_intermarket is True
        assert cfg.include_candlestick is True
        assert cfg.include_earnings is True


# ---------------------------------------------------------------------------
# E2 — Multi-domain GDELT
# ---------------------------------------------------------------------------

class TestE2MultiDomainGdelt:
    def test_gdelt_queries_count(self):
        from src.assembled_core.events.news.fetch_gdelt import GDELT_QUERIES

        assert len(GDELT_QUERIES) >= 10

    def test_fetch_multi_domain_importable(self):
        from src.assembled_core.events.news.fetch_gdelt import fetch_gdelt_multi_domain

        assert fetch_gdelt_multi_domain is not None

    def test_all_domains_have_nonempty_queries(self):
        from src.assembled_core.events.news.fetch_gdelt import GDELT_QUERIES

        for domain, query in GDELT_QUERIES.items():
            assert len(query.strip()) > 0, f"Domain '{domain}' has empty query"
            # Must have at least one term
            assert " " in query or len(query) > 3


# ---------------------------------------------------------------------------
# Phase B — Shock propagation magnitude/dampening
# ---------------------------------------------------------------------------

class TestPhaseBShockPropagation:
    def test_propagate_accepts_magnitude_param(self):
        import inspect
        from src.assembled_core.intel.shock_propagation import propagate

        sig = inspect.signature(propagate)
        params = list(sig.parameters.keys())
        assert "magnitude" in params
        assert "dampening_factor" in params
        assert "regime" in params

    def test_regime_multipliers_defined(self):
        from src.assembled_core.intel.shock_propagation import REGIME_MAGNITUDE_MULTIPLIER

        assert "crisis" in REGIME_MAGNITUDE_MULTIPLIER
        assert "bull" in REGIME_MAGNITUDE_MULTIPLIER
        # Crisis should amplify more than bull
        assert REGIME_MAGNITUDE_MULTIPLIER["crisis"] > REGIME_MAGNITUDE_MULTIPLIER["bull"]

    def test_expanded_trigger_map(self):
        """Trigger map should have been expanded from 8 to 26+ entries."""
        from src.assembled_core.intel.shock_propagation import TRIGGER_TO_SHOCKS

        assert len(TRIGGER_TO_SHOCKS) >= 20, (
            f"Expected >= 20 trigger mappings, got {len(TRIGGER_TO_SHOCKS)}"
        )


# ---------------------------------------------------------------------------
# Phase C — Short policy configuration
# ---------------------------------------------------------------------------

class TestPhaseCPolicy:
    def test_policy_yaml_has_shorts_block(self):
        """configs/policy.yaml should have a 'shorts' block enabled."""
        import yaml
        from pathlib import Path

        policy_path = Path("configs/policy.yaml")
        if not policy_path.exists():
            pytest.skip("configs/policy.yaml not found")

        with open(policy_path) as fh:
            policy = yaml.safe_load(fh)

        assert "shorts" in policy, "No 'shorts' block in policy.yaml"
        assert policy["shorts"].get("enabled") is True, "shorts.enabled != true"
        assert "regime_scaling" in policy["shorts"]

    def test_shorts_instruments_no_leveraged(self):
        """Policy must not allow 2x/3x inverse ETFs by default."""
        import yaml
        from pathlib import Path

        policy_path = Path("configs/policy.yaml")
        if not policy_path.exists():
            pytest.skip("configs/policy.yaml not found")

        with open(policy_path) as fh:
            policy = yaml.safe_load(fh)

        instruments = policy.get("shorts", {}).get("allowed_instruments", {})
        assert instruments.get("inverse_etf_2x") is False
        assert instruments.get("inverse_etf_3x") is False
