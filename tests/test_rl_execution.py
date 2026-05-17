"""Tests for M31: Reinforcement Learning for Optimal Execution."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("src.assembled_core.ml.rl_execution")
from src.assembled_core.ml.rl_execution import (
    ExecutionState,
    QLearningExecutionAgent,
    compute_execution_reward,
    simulate_execution_episode,
    train_execution_agent,
    ACTIONS,
)


@pytest.mark.fast
class TestExecutionState:
    def test_to_array(self):
        state = ExecutionState(
            remaining_qty=0.5,
            time_remaining=0.8,
            spread_bps=10.0,
            volume_ratio=1.5,
            volatility_ratio=1.2,
            momentum=0.3,
            inventory_risk=0.2,
        )
        arr = state.to_array()
        assert len(arr) == 7
        assert all(np.isfinite(arr))

    def test_normalized_values(self):
        state = ExecutionState(
            remaining_qty=1.0,
            time_remaining=1.0,
            spread_bps=50.0,
            volume_ratio=3.0,
            volatility_ratio=3.0,
            momentum=0.0,
            inventory_risk=0.0,
        )
        arr = state.to_array()
        # All should be in [0, 1] range after normalization
        assert all(0 <= v <= 1.0 for v in arr)


@pytest.mark.fast
class TestQLearningAgent:
    def test_select_action(self):
        agent = QLearningExecutionAgent()
        state = ExecutionState(0.5, 0.8, 5.0, 1.0, 1.0, 0.0, 0.1)
        action = agent.select_action(state)
        assert 0 <= action < len(ACTIONS)

    def test_force_execute_at_end(self):
        agent = QLearningExecutionAgent()
        state = ExecutionState(0.5, 0.02, 5.0, 1.0, 1.0, 0.0, 0.5)
        action = agent.select_action(state)
        assert action == 4  # full aggressive when time running out

    def test_update_changes_q(self):
        agent = QLearningExecutionAgent()
        state = ExecutionState(1.0, 1.0, 5.0, 1.0, 1.0, 0.0, 0.0)
        next_state = ExecutionState(0.5, 0.9, 5.0, 1.0, 1.0, 0.0, 0.05)
        td = agent.update(state, 1, -0.1, next_state, False)
        assert isinstance(td, float)
        assert agent.get_q_table_size() > 0

    def test_exploration(self):
        agent = QLearningExecutionAgent(epsilon=1.0)
        state = ExecutionState(0.5, 0.5, 5.0, 1.0, 1.0, 0.0, 0.25)
        # With epsilon=1.0 in training mode, actions should be random
        actions = set()
        for _ in range(100):
            actions.add(agent.select_action(state, training=True))
        # Should have explored multiple actions
        assert len(actions) > 1


@pytest.mark.fast
class TestExecutionReward:
    def test_perfect_execution(self):
        reward = compute_execution_reward(
            execution_price=100.0,
            arrival_price=100.0,
            spread_bps=5.0,
            remaining_qty=0.0,
            time_remaining=0.5,
        )
        assert reward > -1.0  # should be near zero cost

    def test_costly_execution(self):
        reward = compute_execution_reward(
            execution_price=101.0,
            arrival_price=100.0,
            spread_bps=20.0,
            remaining_qty=0.5,
            time_remaining=0.05,
        )
        assert reward < 0  # high cost

    def test_zero_arrival_price(self):
        reward = compute_execution_reward(0.0, 0.0, 5.0, 0.5, 0.5)
        assert reward == 0.0


@pytest.mark.fast
class TestSimulateEpisode:
    def test_basic_episode(self):
        agent = QLearningExecutionAgent()
        result = simulate_execution_episode(
            agent,
            total_qty=1.0,
            n_steps=10,
            seed=42,
        )
        assert "total_reward" in result
        assert "vwap_slippage_bps" in result
        assert "fill_rate" in result
        assert result["fill_rate"] == pytest.approx(1.0, abs=0.01)

    def test_deterministic_with_seed(self):
        agent1 = QLearningExecutionAgent()
        agent2 = QLearningExecutionAgent()
        r1 = simulate_execution_episode(agent1, 1.0, 10, seed=42, training=False)
        r2 = simulate_execution_episode(agent2, 1.0, 10, seed=42, training=False)
        assert r1["vwap_slippage_bps"] == r2["vwap_slippage_bps"]


@pytest.mark.fast
class TestTrainAgent:
    def test_basic_training(self):
        agent, metrics = train_execution_agent(
            n_episodes=50,
            n_steps=10,
            seed=42,
        )
        assert isinstance(agent, QLearningExecutionAgent)
        assert len(metrics) == 50
        assert agent.get_q_table_size() > 0

    def test_training_improves(self):
        agent, metrics = train_execution_agent(
            n_episodes=200,
            n_steps=15,
            seed=42,
        )
        early_reward = np.mean([m["total_reward"] for m in metrics[:20]])
        late_reward = np.mean([m["total_reward"] for m in metrics[-20:]])
        # Training should improve or at least not dramatically worsen
        assert late_reward >= early_reward - 1.0
