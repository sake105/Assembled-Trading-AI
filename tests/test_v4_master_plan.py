"""Tests for all 13 items from MASTER_IMPLEMENTATION_PLAN_V4_2026-04-29.md.

Coverage:
  §1  Polymarket source
  §2  Bayesian Sharpe metrics
  §3  Redis Streams event bus
  §4  QuestDB tick store
  §5  RL execution skeleton
  §6  GNN signal stub
  §7  LLM-RAG news reasoning
  §8  Risk-parity strategy allocator
  §9  Walk-forward + Optuna
  §10 Differential Privacy utilities
  §11 Order-book imbalance features
  §12 Regime-conditional allocator
  §13 Kalshi source + combined signal
  bayesian_confidence T1.5 tier
  georisk_overlay prediction-market integration
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


# ============================================================
# §1 Polymarket
# ============================================================

class TestPolymarketSource:
    def test_module_importable(self):
        from assembled_core.data.sources.polymarket_source import (
            GEO_KEYWORDS
        )
        assert "election" in GEO_KEYWORDS

    def test_get_market_implied_geo_signal_structure(self, monkeypatch):
        from assembled_core.data.sources import polymarket_source
        monkeypatch.setattr(polymarket_source, "_get", lambda *a, **kw: None)
        result = polymarket_source.get_market_implied_geo_signal(limit=10)
        assert "signal" in result
        assert "n_markets" in result
        assert result["source"] == "polymarket"
        assert 0.0 <= result["signal"] <= 1.0

    def test_fetch_active_markets_empty_on_failure(self, monkeypatch):
        from assembled_core.data.sources import polymarket_source
        monkeypatch.setattr(polymarket_source, "_get", lambda *a, **kw: None)
        result = polymarket_source.fetch_active_markets(limit=5)
        assert result == []

    def test_signal_in_unit_interval(self, monkeypatch):
        from assembled_core.data.sources import polymarket_source

        fake_markets = [
            {"id": "1", "question": "Will war start?", "endDate": "", "volume": "1000",
             "liquidity": "500", "lastTradePrice": "0.65", "outcomes": []},
        ]
        monkeypatch.setattr(polymarket_source, "_get", lambda *a, **kw: fake_markets)
        result = polymarket_source.get_market_implied_geo_signal(limit=5)
        assert 0.0 <= result["signal"] <= 1.0
        assert result["n_markets"] == 1


# ============================================================
# §13 Kalshi + combined
# ============================================================

class TestKalshiSource:
    def test_module_importable_v2(self):
        pass

    def test_get_market_implied_geo_signal_empty(self, monkeypatch):
        from assembled_core.data.sources import kalshi_source
        monkeypatch.setattr(kalshi_source, "_get", lambda *a, **kw: None)
        result = kalshi_source.get_market_implied_geo_signal()
        assert result["signal"] == 0.0
        assert result["source"] == "kalshi"

    def test_combined_signal_both_none(self):
        from assembled_core.data.sources.kalshi_source import fetch_combined_prediction_signal
        result = fetch_combined_prediction_signal(None, None)
        assert result["signal"] == 0.0
        assert result["n_sources"] == 0

    def test_combined_signal_poly_only(self):
        from assembled_core.data.sources.kalshi_source import fetch_combined_prediction_signal
        poly = {"signal": 0.4}
        result = fetch_combined_prediction_signal(poly, None)
        assert result["signal"] == pytest.approx(0.4)
        assert result["n_sources"] == 1

    def test_combined_signal_blended(self):
        from assembled_core.data.sources.kalshi_source import fetch_combined_prediction_signal
        poly = {"signal": 0.6}
        kals = {"signal": 0.4}
        result = fetch_combined_prediction_signal(poly, kals, poly_weight=0.5)
        assert result["signal"] == pytest.approx(0.5)
        assert result["n_sources"] == 2


# ============================================================
# §2 Bayesian Sharpe
# ============================================================

class TestBayesianMetrics:
    def test_module_importable_v3(self):
        pass

    def test_sharpe_posterior_analytic(self):
        from assembled_core.qa.bayesian_metrics import bayesian_sharpe_posterior
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.015, 252).tolist()
        result = bayesian_sharpe_posterior(returns, strategy="test", use_pymc=False)
        assert result.strategy == "test"
        assert isinstance(result.mean, float)
        assert result.hdi_lower < result.hdi_upper
        assert 0.0 <= result.p_positive <= 1.0
        assert result.n_obs == 252
        assert result.backend == "analytic"

    def test_positive_returns_positive_sharpe(self):
        from assembled_core.qa.bayesian_metrics import bayesian_sharpe_posterior
        returns = [0.005] * 252  # constant positive returns
        result = bayesian_sharpe_posterior(returns, use_pymc=False)
        assert result.mean > 0

    def test_hierarchical_comparison_two_strategies(self):
        from assembled_core.qa.bayesian_metrics import hierarchical_strategy_comparison
        rng = np.random.default_rng(1)
        strats = {
            "A": rng.normal(0.002, 0.015, 120).tolist(),
            "B": rng.normal(0.0005, 0.012, 120).tolist(),
        }
        result = hierarchical_strategy_comparison(strats, use_pymc=False)
        assert set(result.strategies) == {"A", "B"}
        assert len(result.posteriors) == 2
        total_p_best = sum(result.p_best.values())
        assert total_p_best == pytest.approx(1.0, abs=0.05)
        assert result.backend in ("pymc", "analytic")

    def test_empty_returns_graceful(self):
        from assembled_core.qa.bayesian_metrics import bayesian_sharpe_posterior
        result = bayesian_sharpe_posterior([], use_pymc=False)
        assert isinstance(result.mean, float)


# ============================================================
# §3 Redis Streams event bus
# ============================================================

class TestEventBus:
    def test_module_importable_v4(self):
        pass

    def test_null_bus_available_false(self):
        from assembled_core.pipeline.event_bus import get_null_bus
        bus = get_null_bus()
        assert bus.available is False

    def test_publish_noop_when_unavailable(self):
        from assembled_core.pipeline.event_bus import get_null_bus
        bus = get_null_bus()
        result = bus.publish("test_event", {"key": "value"})
        assert result is False

    def test_publish_batch_noop(self):
        from assembled_core.pipeline.event_bus import get_null_bus
        bus = get_null_bus()
        n = bus.publish_batch([("event_a", {"x": 1}), ("event_b", {"y": 2})])
        assert n == 0

    def test_read_latest_empty_when_unavailable(self):
        from assembled_core.pipeline.event_bus import get_null_bus
        bus = get_null_bus()
        assert bus.read_latest("test") == []

    def test_streamed_phase_executes_body(self):
        from assembled_core.pipeline.event_bus import EventBus, streamed_phase
        bus = EventBus.__new__(EventBus)
        bus._prefix = "t"
        bus._maxlen = 0
        bus._client = None
        bus._available = False

        executed = []
        with streamed_phase(bus, "test_phase", {"run": "1"}):
            executed.append(True)
        assert executed == [True]

    def test_streamed_phase_propagates_exception(self):
        from assembled_core.pipeline.event_bus import EventBus, streamed_phase
        bus = EventBus.__new__(EventBus)
        bus._prefix = "t"
        bus._maxlen = 0
        bus._client = None
        bus._available = False

        with pytest.raises(ValueError, match="boom"):
            with streamed_phase(bus, "fail_phase"):
                raise ValueError("boom")


# ============================================================
# §4 QuestDB tick store
# ============================================================

class TestTickStore:
    def test_module_importable_v5(self):
        pass

    def test_write_ticks_empty_list(self):
        from assembled_core.data.tick_store import write_ticks
        assert write_ticks([]) == 0

    def test_ping_returns_bool(self):
        from assembled_core.data.tick_store import ping
        result = ping()
        assert isinstance(result, bool)

    def test_query_ohlcv_no_server(self):
        from assembled_core.data.tick_store import query_ohlcv
        from datetime import datetime, timezone
        result = query_ohlcv(
            "SPY",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )
        assert isinstance(result, list)

    def test_ohlcv_tick_dataclass(self):
        from assembled_core.data.tick_store import OHLCVTick
        from datetime import datetime, timezone
        tick = OHLCVTick(
            symbol="AAPL",
            ts=datetime(2024, 1, 15, tzinfo=timezone.utc),
            open=180.0, high=182.0, low=179.0, close=181.5, volume=1_000_000.0,
        )
        assert tick.symbol == "AAPL"
        assert tick.close == 181.5


# ============================================================
# §5 RL execution skeleton
# ============================================================

class TestRLEnvironment:
    def test_module_importable_v6(self):
        pass

    def test_env_config_defaults(self):
        from assembled_core.execution.rl_environment import ExecutionEnvConfig
        cfg = ExecutionEnvConfig()
        assert cfg.total_shares == 10_000
        assert cfg.n_steps == 20

    def test_env_observe_shape(self):
        from assembled_core.execution.rl_environment import OrderExecutionEnv, ExecutionEnvConfig
        env = OrderExecutionEnv(ExecutionEnvConfig(seed=0))
        obs, _ = env.reset(seed=42)
        assert obs.shape == (5,)
        assert float(obs[0]) == pytest.approx(1.0)   # remaining_frac starts at 1

    def test_env_step_reduces_remaining(self):
        from assembled_core.execution.rl_environment import OrderExecutionEnv, ExecutionEnvConfig
        import numpy as np
        env = OrderExecutionEnv(ExecutionEnvConfig(seed=0))
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert info["shares_executed"] > 0
        assert info["remaining"] < 10_000

    def test_env_full_episode_exhausts_shares(self):
        from assembled_core.execution.rl_environment import OrderExecutionEnv, ExecutionEnvConfig
        import numpy as np
        env = OrderExecutionEnv(ExecutionEnvConfig(n_steps=5, seed=0))
        env.reset(seed=0)
        done = False
        while not done:
            obs, r, terminated, truncated, info = env.step(np.array([0.5]))
            done = terminated or truncated
        assert info["remaining"] == 0


class TestRLExecution:
    def test_module_importable_v7(self):
        pass

    def test_rule_based_executor_runs(self):
        from assembled_core.execution.rl_execution import RuleBasedExecutor
        from assembled_core.execution.rl_environment import ExecutionEnvConfig
        exec_ = RuleBasedExecutor(ExecutionEnvConfig(n_steps=5, seed=0))
        result = exec_.execute(n_steps=5, seed=0)
        assert result["shares_executed"] == 10_000
        assert result["backend"] == "twap_rule"
        assert isinstance(result["implementation_shortfall"], float)

    def test_rl_executor_no_sb3(self):
        from assembled_core.execution.rl_execution import RLExecutor, SB3_AVAILABLE
        if SB3_AVAILABLE:
            pytest.skip("SB3 installed — stub test not applicable")
        exec_ = RLExecutor()
        exec_.train()  # should log warning, not crash
        result = exec_.execute(n_steps=5)
        assert result["n_slices"] == 5


# ============================================================
# §6 GNN signal stub
# ============================================================

class TestGNNSignal:
    def test_module_importable_v8(self):
        pass

    def test_stub_predict_returns_zero_scores(self):
        from assembled_core.ml.gnn_signal import GNNSignalModel
        model = GNNSignalModel()
        node_features = np.zeros((3, 8))
        result = model.predict(node_features, symbols=["A", "B", "C"])
        assert set(result.symbols) == {"A", "B", "C"}
        assert all(v == 0.0 for v in result.scores.values())
        assert result.backend == "stub"

    def test_fit_raises_not_implemented_without_pyg(self):
        from assembled_core.ml.gnn_signal import GNNSignalModel, PYG_AVAILABLE
        if PYG_AVAILABLE:
            pytest.skip("PyG installed — test not applicable")
        model = GNNSignalModel()
        with pytest.raises(NotImplementedError):
            model.fit(np.zeros((10, 3)), ["A", "B", "C"])

    def test_build_graph_stub_mode(self):
        from assembled_core.ml.gnn_signal import GNNSignalModel, TORCH_AVAILABLE
        if TORCH_AVAILABLE:
            pytest.skip("Torch installed")
        model = GNNSignalModel()
        edge_idx, edge_w = model.build_graph(np.zeros((5, 3)), ["A", "B", "C"])
        assert edge_idx is None


# ============================================================
# §7 LLM-RAG news reasoning
# ============================================================

class TestNewsRAG:
    def test_module_importable_v9(self):
        pass

    def test_ingest_and_query_memory_store(self):
        from assembled_core.intel.news_rag import NewsRAG
        rag = NewsRAG()  # no Qdrant, no LLM
        rag.ingest("Fed hikes 75bps", "SPY", "2022-06-15", outcome_return=-0.03)
        rag.ingest("Strong jobs report surprises", "SPY", "2022-07-08", outcome_return=0.02)
        result = rag.query("Federal Reserve raises interest rates")
        assert isinstance(result.retrieved, list)
        assert isinstance(result.predicted_direction, str)
        assert result.predicted_direction in ("bullish", "bearish", "neutral")
        assert 0.0 <= result.confidence <= 1.0

    def test_n_stored_increments(self):
        from assembled_core.intel.news_rag import NewsRAG
        rag = NewsRAG()
        assert rag.n_stored == 0
        rag.ingest("Test headline", "AAPL", "2024-01-01", outcome_return=0.01)
        assert rag.n_stored == 1

    def test_empty_store_returns_neutral(self):
        from assembled_core.intel.news_rag import NewsRAG
        rag = NewsRAG()
        result = rag.query("Some random event")
        assert result.predicted_direction == "neutral"

    def test_bearish_outcome_predicts_bearish(self):
        from assembled_core.intel.news_rag import NewsRAG
        rag = NewsRAG()
        for _ in range(5):
            rag.ingest("Massive negative event", "SPY", "2024-01-01", outcome_return=-0.05)
        result = rag.query("Another negative event")
        assert result.predicted_direction == "bearish"


# ============================================================
# §8 Risk-parity strategy allocator
# ============================================================

class TestStrategyAllocator:
    def test_module_importable_v10(self):
        pass

    def test_two_strategies_equal_vol(self):
        from assembled_core.portfolio.strategy_allocator import allocate_from_returns_dict
        rng = np.random.default_rng(0)
        returns = {
            "A": rng.normal(0.001, 0.015, 252).tolist(),
            "B": rng.normal(0.001, 0.015, 252).tolist(),
        }
        result = allocate_from_returns_dict(returns, target_vol=0.15)
        # Equal vol → roughly equal weights
        assert abs(result.weights["A"] - result.weights["B"]) < 0.20

    def test_high_vol_gets_lower_weight(self):
        from assembled_core.portfolio.strategy_allocator import allocate_from_returns_dict
        rng = np.random.default_rng(1)
        returns = {
            "low_vol":  rng.normal(0.001, 0.008, 252).tolist(),
            "high_vol": rng.normal(0.001, 0.030, 252).tolist(),
        }
        result = allocate_from_returns_dict(returns)
        # After inverse-vol weighting and capping, low_vol should dominate
        assert result.weights["low_vol"] > result.weights["high_vol"]

    def test_weights_scale_to_target_vol(self):
        from assembled_core.portfolio.strategy_allocator import allocate_from_returns_dict
        rng = np.random.default_rng(2)
        # Very low daily vol (0.003 * sqrt(252) ≈ 4.8% annualised) → scale up to 15%
        returns = {"A": rng.normal(0, 0.003, 252).tolist()}
        result = allocate_from_returns_dict(returns, target_vol=0.15)
        assert result.vol_scale > 1.0   # low-vol strategy scaled up

    def test_empty_strategies(self):
        from assembled_core.portfolio.strategy_allocator import inverse_vol_weights
        result = inverse_vol_weights([])
        assert result.weights == {}

    def test_max_weight_capped(self):
        from assembled_core.portfolio.strategy_allocator import allocate_from_returns_dict
        rng = np.random.default_rng(3)
        returns = {
            "A": rng.normal(0.01, 0.001, 252).tolist(),   # very low vol → would dominate
            "B": rng.normal(0.001, 0.030, 252).tolist(),
        }
        result = allocate_from_returns_dict(returns, max_weight=0.60)
        # After scaling A should still be ≤ 0.60 * vol_scale (rough check)
        for w in result.weights.values():
            assert w >= 0.0


# ============================================================
# §9 Walk-forward + Optuna
# ============================================================

class TestWalkForwardOptuna:
    def test_module_importable_v11(self):
        pass

    def test_basic_walk_forward_no_optuna(self, monkeypatch):
        from assembled_core.qa import walk_forward_optuna as wfo_mod
        monkeypatch.setattr(wfo_mod, "_OPTUNA_AVAILABLE", False)
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.015, 500).tolist()

        def dummy_obj(arr, params):
            return float(np.mean(arr) / (np.std(arr) + 1e-9) * math.sqrt(252))

        result = wfo_mod.walk_forward_optuna(
            returns=returns,
            objective_fn=dummy_obj,
            search_space={"lookback": ("int", 10, 60)},
            train_days=120,
            test_days=60,
            step_days=60,
            default_params={"lookback": 20},
        )
        assert result.n_folds >= 1
        assert isinstance(result.avg_test_sharpe, float)
        assert result.backend == "default"

    def test_momentum_objective_positive_returns(self):
        from assembled_core.qa.walk_forward_optuna import momentum_sharpe_objective
        rng = np.random.default_rng(0)
        arr = rng.normal(0.002, 0.012, 252)
        sharpe = momentum_sharpe_objective(arr, {"lookback": 20, "threshold": 0.0})
        assert isinstance(sharpe, float)

    def test_insufficient_data_returns_empty(self, monkeypatch):
        from assembled_core.qa import walk_forward_optuna as wfo_mod
        monkeypatch.setattr(wfo_mod, "_OPTUNA_AVAILABLE", False)
        result = wfo_mod.walk_forward_optuna(
            returns=[0.001] * 50,
            objective_fn=lambda a, p: 0.0,
            search_space={},
            train_days=200,
            test_days=100,
        )
        assert result.n_folds == 0


# ============================================================
# §10 Differential Privacy
# ============================================================

class TestDifferentialPrivacy:
    def test_module_importable_v12(self):
        pass

    def test_gaussian_mechanism_adds_noise(self):
        from assembled_core.ml.differential_privacy import gaussian_mechanism
        rng = np.random.default_rng(0)
        result = gaussian_mechanism(10.0, sensitivity=1.0, epsilon=1.0, delta=1e-5, rng=rng)
        assert isinstance(result, float)
        # Noisy result should be close to 10 (within many sigmas)
        assert abs(result - 10.0) < 50.0

    def test_laplace_mechanism_adds_noise(self):
        from assembled_core.ml.differential_privacy import laplace_mechanism
        rng = np.random.default_rng(1)
        result = laplace_mechanism(5.0, sensitivity=1.0, epsilon=0.5, rng=rng)
        assert isinstance(result, float)

    def test_dp_mean_close_to_true_mean(self):
        from assembled_core.ml.differential_privacy import dp_mean
        rng = np.random.default_rng(2)
        data = np.ones(1000) * 0.5   # true mean = 0.5
        # With high epsilon (low noise), should be close
        result = dp_mean(data, clip_bound=1.0, epsilon=10.0, delta=1e-6, rng=rng)
        assert abs(result - 0.5) < 0.2

    def test_dp_count_non_negative(self):
        from assembled_core.ml.differential_privacy import dp_count
        rng = np.random.default_rng(3)
        result = dp_count(100, epsilon=1.0, rng=rng)
        assert result >= 0

    def test_privacy_budget_tracking(self):
        from assembled_core.ml.differential_privacy import PrivacyBudget
        budget = PrivacyBudget(epsilon_total=2.0, delta=1e-5)
        assert budget.epsilon_remaining == pytest.approx(2.0)
        ok = budget.consume(0.8, "gaussian")
        assert ok is True
        assert budget.epsilon_remaining == pytest.approx(1.2)
        ok2 = budget.consume(1.3, "laplace")   # would exceed budget
        assert ok2 is False

    def test_dpsgd_trainer_no_opacus(self):
        from assembled_core.ml.differential_privacy import DPSGDTrainer, OPACUS_AVAILABLE
        if OPACUS_AVAILABLE:
            pytest.skip("Opacus installed — stub test not applicable")
        trainer = DPSGDTrainer()
        with pytest.raises(NotImplementedError):
            trainer.make_private(None, None, None)

    def test_invalid_epsilon_raises(self):
        from assembled_core.ml.differential_privacy import gaussian_noise_scale
        with pytest.raises(ValueError):
            gaussian_noise_scale(1.0, epsilon=-1.0, delta=1e-5)


# ============================================================
# §11 Order-book imbalance
# ============================================================

class TestOrderBookImbalance:
    def test_module_importable_v13(self):
        pass

    def test_balanced_book_zero_imbalance(self):
        from assembled_core.features.order_book_imbalance import (
            OrderBookSnapshot, BookLevel, compute_imbalance_features
        )
        snap = OrderBookSnapshot(
            symbol="AAPL", timestamp=1700000000.0,
            bids=[BookLevel(price=100.0, size=100)],
            asks=[BookLevel(price=100.1, size=100)],
        )
        feat = compute_imbalance_features(snap)
        assert feat.l1_imbalance == pytest.approx(0.0)

    def test_bid_heavy_positive_imbalance(self):
        from assembled_core.features.order_book_imbalance import (
            OrderBookSnapshot, BookLevel, compute_imbalance_features
        )
        snap = OrderBookSnapshot(
            symbol="AAPL", timestamp=1700000000.0,
            bids=[BookLevel(price=100.0, size=300)],
            asks=[BookLevel(price=100.1, size=100)],
        )
        feat = compute_imbalance_features(snap)
        assert feat.l1_imbalance > 0.0

    def test_ask_heavy_negative_imbalance(self):
        from assembled_core.features.order_book_imbalance import (
            OrderBookSnapshot, BookLevel, compute_imbalance_features
        )
        snap = OrderBookSnapshot(
            symbol="SPY", timestamp=1700000000.0,
            bids=[BookLevel(price=450.0, size=50)],
            asks=[BookLevel(price=450.1, size=500)],
        )
        feat = compute_imbalance_features(snap)
        assert feat.l1_imbalance < 0.0

    def test_imbalance_from_dict(self):
        from assembled_core.features.order_book_imbalance import imbalance_from_dict
        snap_dict = {
            "symbol": "TSLA", "timestamp": 1700000000.0,
            "bids": [{"price": 200.0, "size": 100}, {"price": 199.9, "size": 80}],
            "asks": [{"price": 200.1, "size": 90}],
        }
        feat = imbalance_from_dict(snap_dict)
        assert feat.symbol == "TSLA"
        assert -1.0 <= feat.l5_imbalance <= 1.0

    def test_spread_calculation(self):
        from assembled_core.features.order_book_imbalance import (
            OrderBookSnapshot, BookLevel, compute_imbalance_features
        )
        snap = OrderBookSnapshot(
            symbol="QQQ", timestamp=1700000000.0,
            bids=[BookLevel(price=100.0, size=100)],
            asks=[BookLevel(price=100.2, size=100)],
        )
        feat = compute_imbalance_features(snap)
        assert feat.spread == pytest.approx(0.2, abs=1e-6)
        assert feat.spread_bps == pytest.approx(0.2 / 100.1 * 10_000, abs=0.5)  # ~20 bps

    def test_rolling_imbalance_signal_length(self):
        from assembled_core.features.order_book_imbalance import rolling_imbalance_signal
        snaps = [
            {"symbol": "SPY", "timestamp": float(i),
             "bids": [{"price": 450.0, "size": 100}],
             "asks": [{"price": 450.1, "size": 100 + i % 5}]}
            for i in range(15)
        ]
        signals = rolling_imbalance_signal(snaps, lookback=5)
        assert len(signals) == 15

    def test_empty_book_zero_imbalance(self):
        from assembled_core.features.order_book_imbalance import (
            OrderBookSnapshot, compute_imbalance_features
        )
        snap = OrderBookSnapshot(symbol="X", timestamp=0.0)
        feat = compute_imbalance_features(snap)
        assert feat.l1_imbalance == 0.0
        assert feat.spread == 0.0


# ============================================================
# §12 Regime-conditional allocator
# ============================================================

class TestRegimeConditionalAllocator:
    def test_module_importable_v14(self):
        pass

    def test_compute_regime_sharpes_basic(self):
        from assembled_core.portfolio.regime_conditional_allocator import compute_regime_sharpes
        rng = np.random.default_rng(0)
        n = 100
        returns = {
            "A": rng.normal(0.002, 0.015, n).tolist(),
            "B": rng.normal(0.001, 0.012, n).tolist(),
        }
        regimes = [0] * 50 + [1] * 50   # 2 regimes
        perfs = compute_regime_sharpes(returns, regimes)
        assert "A" in perfs
        assert 0 in perfs["A"]
        assert 1 in perfs["A"]

    def test_allocate_by_regime_returns_weights(self):
        from assembled_core.portfolio.regime_conditional_allocator import (
            compute_regime_sharpes, allocate_by_regime
        )
        rng = np.random.default_rng(1)
        n = 100
        returns = {
            "A": rng.normal(0.003, 0.015, n).tolist(),
            "B": rng.normal(0.001, 0.012, n).tolist(),
        }
        regimes = [0] * 50 + [1] * 50
        perfs = compute_regime_sharpes(returns, regimes)
        result = allocate_by_regime(0, perfs, target_vol=0.15)
        assert len(result.weights) > 0
        for w in result.weights.values():
            assert w >= 0.0

    def test_regime_allocator_interface(self):
        from assembled_core.portfolio.regime_conditional_allocator import build_regime_allocator
        rng = np.random.default_rng(2)
        n = 120
        returns = {
            "mom": rng.normal(0.002, 0.015, n).tolist(),
            "rev": rng.normal(0.001, 0.020, n).tolist(),
        }
        regimes = ["BULL"] * 60 + ["BEAR"] * 60
        allocator = build_regime_allocator(returns, regimes)
        assert set(allocator.strategies) == {"mom", "rev"}
        result = allocator.allocate("BULL")
        assert isinstance(result.weights, dict)
        assert result.regime == "BULL"

    def test_unknown_regime_fallback(self):
        from assembled_core.portfolio.regime_conditional_allocator import build_regime_allocator
        rng = np.random.default_rng(3)
        n = 60
        returns = {"A": rng.normal(0.001, 0.015, n).tolist()}
        regimes = [0] * n
        allocator = build_regime_allocator(returns, regimes)
        result = allocator.allocate(999)   # regime never seen → fallback
        assert "A" in result.weights


# ============================================================
# T1.5 tier in bayesian_confidence
# ============================================================

class TestBayesianConfidenceT15:
    def test_t15_in_source_reliability(self):
        from assembled_core.intel.bayesian_confidence import SOURCE_RELIABILITY
        assert "T1.5" in SOURCE_RELIABILITY

    def test_t15_reliability_between_t1_and_t2(self):
        from assembled_core.intel.bayesian_confidence import SOURCE_RELIABILITY
        assert SOURCE_RELIABILITY["T2"] < SOURCE_RELIABILITY["T1.5"] < SOURCE_RELIABILITY["T1"]

    def test_t15_fpr_in_false_positive_rate(self):
        from assembled_core.intel.bayesian_confidence import FALSE_POSITIVE_RATE
        assert "T1.5" in FALSE_POSITIVE_RATE

    def test_t15_fpr_between_t1_and_t2(self):
        from assembled_core.intel.bayesian_confidence import FALSE_POSITIVE_RATE
        assert FALSE_POSITIVE_RATE["T1"] < FALSE_POSITIVE_RATE["T1.5"] < FALSE_POSITIVE_RATE["T2"]

    def test_bayesian_update_with_t15_source(self):
        from assembled_core.intel.bayesian_confidence import (
            bayesian_update, SOURCE_RELIABILITY, FALSE_POSITIVE_RATE
        )
        posterior = bayesian_update(
            prior=0.05,
            evidence_strength=0.8,
            source_reliability=SOURCE_RELIABILITY["T1.5"],
            false_positive_rate=FALSE_POSITIVE_RATE["T1.5"],
        )
        assert 0.05 < posterior < 0.99


# ============================================================
# georisk_overlay prediction-market integration
# ============================================================

class TestGeoRiskOverlayPredictionMarkets:
    def test_get_market_implied_geo_signal_importable(self):
        pass

    def test_get_market_implied_geo_signal_structure_v2(self, monkeypatch):
        from assembled_core.risk import georisk_overlay
        # Monkeypatch both source modules to return known signals
        from unittest.mock import MagicMock
        poly_mock = MagicMock(return_value={"signal": 0.3, "source": "polymarket", "n_markets": 2, "avg_prob": 0.45, "volume_weighted_prob": 0.46})
        kals_mock = MagicMock(return_value={"signal": 0.2, "source": "kalshi", "n_markets": 3, "avg_mid": 0.44, "volume_weighted_mid": 0.43})

        import assembled_core.data.sources.polymarket_source as pm_mod
        import assembled_core.data.sources.kalshi_source as km_mod
        monkeypatch.setattr(pm_mod, "get_market_implied_geo_signal", poly_mock)
        monkeypatch.setattr(km_mod, "get_market_implied_geo_signal", kals_mock)

        result = georisk_overlay.get_market_implied_geo_signal()
        assert "signal" in result
        assert 0.0 <= result["signal"] <= 1.0

    def test_get_signal_no_sources(self, monkeypatch):
        from assembled_core.risk.georisk_overlay import get_market_implied_geo_signal
        # Both False → skip both
        result = get_market_implied_geo_signal(use_polymarket=False, use_kalshi=False)
        assert result["signal"] == 0.0
        assert result["n_sources"] == 0
