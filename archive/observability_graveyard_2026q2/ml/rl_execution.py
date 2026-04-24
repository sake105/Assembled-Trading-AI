"""Reinforcement Learning for Optimal Execution (M31).

Implements RL-based order execution that learns optimal timing and sizing:
  1. Q-Learning agent for discrete execution decisions
  2. State representation: market microstructure features
  3. Reward: negative implementation shortfall
  4. Action space: execute now, wait, split order

The RL agent learns to minimize execution costs by adapting to
real-time market conditions (spread, volume, volatility).

Reference:
    Nevmyvaka, Y. et al. (2006). "Reinforcement Learning for Optimized
    Trade Execution."
    Hendricks, D. & Wilcox, D. (2014). "A Reinforcement Learning Extension
    to the Almgren-Chriss Model."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionState:
    """State representation for the RL execution agent.

    Attributes:
        remaining_qty: Fraction of order remaining (0 to 1).
        time_remaining: Fraction of time horizon remaining (0 to 1).
        spread_bps: Current bid-ask spread in basis points.
        volume_ratio: Current volume / average volume.
        volatility_ratio: Current vol / average vol.
        momentum: Recent price momentum (normalized).
        inventory_risk: Current inventory risk level.
    """

    remaining_qty: float
    time_remaining: float
    spread_bps: float
    volume_ratio: float
    volatility_ratio: float
    momentum: float
    inventory_risk: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.remaining_qty,
            self.time_remaining,
            self.spread_bps / 50.0,  # normalize
            min(self.volume_ratio, 3.0) / 3.0,
            min(self.volatility_ratio, 3.0) / 3.0,
            np.clip(self.momentum, -1, 1),
            np.clip(self.inventory_risk, 0, 1),
        ])


@dataclass
class ExecutionAction:
    """Action taken by the RL agent.

    Attributes:
        action_type: "execute", "wait", or "split".
        execute_pct: Fraction of remaining to execute (0-1).
        urgency: How aggressively to execute (affects limit price).
    """

    action_type: str
    execute_pct: float
    urgency: float


# Discrete action space
ACTIONS = [
    ExecutionAction("wait", 0.0, 0.0),            # 0: wait
    ExecutionAction("execute", 0.10, 0.3),         # 1: small passive
    ExecutionAction("execute", 0.25, 0.5),         # 2: medium
    ExecutionAction("execute", 0.50, 0.7),         # 3: large
    ExecutionAction("execute", 1.00, 1.0),         # 4: full aggressive
    ExecutionAction("split", 0.33, 0.5),           # 5: split into thirds
]

N_ACTIONS = len(ACTIONS)
STATE_DIM = 7


class QLearningExecutionAgent:
    """Tabular Q-Learning agent for order execution.

    Uses discretized state space and epsilon-greedy exploration.

    Attributes:
        n_bins: Number of bins per state dimension.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate.
        q_table: Q-value table.
    """

    def __init__(
        self,
        n_bins: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: discretized state -> action values
        # Using a dict for sparse representation
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self._rng = np.random.default_rng(42)

    def _discretize_state(self, state: ExecutionState) -> tuple[int, ...]:
        arr = state.to_array()
        bins = np.clip(
            (arr * self.n_bins).astype(int),
            0, self.n_bins - 1,
        )
        return tuple(bins)

    def _get_q_values(self, state_key: tuple[int, ...]) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(N_ACTIONS)
        return self.q_table[state_key]

    def select_action(
        self,
        state: ExecutionState,
        training: bool = False,
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current execution state.
            training: Whether in training mode (enables exploration).

        Returns:
            Action index.
        """
        # Force execute if very little time remaining
        if state.time_remaining < 0.05 and state.remaining_qty > 0.01:
            return 4  # full aggressive

        state_key = self._discretize_state(state)
        q_values = self._get_q_values(state_key)

        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, N_ACTIONS))

        return int(np.argmax(q_values))

    def update(
        self,
        state: ExecutionState,
        action: int,
        reward: float,
        next_state: ExecutionState,
        done: bool,
    ) -> float:
        """Update Q-values from experience.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            done: Whether episode is done.

        Returns:
            TD error.
        """
        state_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state)

        q_current = self._get_q_values(state_key)
        q_next = self._get_q_values(next_key)

        target = reward + (0.0 if done else self.gamma * np.max(q_next))
        td_error = target - q_current[action]
        q_current[action] += self.alpha * td_error

        return float(td_error)

    def get_q_table_size(self) -> int:
        return len(self.q_table)


def compute_execution_reward(
    execution_price: float,
    arrival_price: float,
    spread_bps: float,
    remaining_qty: float,
    time_remaining: float,
) -> float:
    """Compute reward for an execution step.

    Reward = negative implementation shortfall + penalties.

    Args:
        execution_price: Price at which execution occurred.
        arrival_price: Price at decision time (benchmark).
        spread_bps: Current spread in basis points.
        remaining_qty: Fraction remaining after execution.
        time_remaining: Fraction of time left.

    Returns:
        Scalar reward.
    """
    if arrival_price <= 0:
        return 0.0

    # Implementation shortfall (negative = good, we want to minimize cost)
    shortfall_bps = (execution_price - arrival_price) / arrival_price * 10000
    cost_reward = -abs(shortfall_bps) / 100.0  # scale

    # Penalty for holding inventory (risk)
    inventory_penalty = -remaining_qty * 0.1

    # Urgency penalty: not finishing on time
    time_penalty = 0.0
    if time_remaining < 0.1 and remaining_qty > 0.1:
        time_penalty = -remaining_qty * 0.5

    # Spread cost
    spread_penalty = -spread_bps / 10000.0

    return float(cost_reward + inventory_penalty + time_penalty + spread_penalty)


def simulate_execution_episode(
    agent: QLearningExecutionAgent,
    total_qty: float,
    n_steps: int = 20,
    base_price: float = 100.0,
    avg_spread_bps: float = 5.0,
    volatility: float = 0.02,
    training: bool = True,
    seed: int | None = None,
) -> dict[str, float]:
    """Simulate one execution episode for training or evaluation.

    Args:
        agent: The RL agent.
        total_qty: Total quantity to execute.
        n_steps: Number of time steps in the episode.
        base_price: Starting price.
        avg_spread_bps: Average spread in basis points.
        volatility: Price volatility per step.
        training: Whether to train the agent.
        seed: Random seed.

    Returns:
        Dict with episode metrics: total_cost_bps, vwap_slippage, fill_rate, n_trades.
    """
    rng = np.random.default_rng(seed)
    price = base_price
    arrival_price = base_price
    remaining = 1.0  # fraction remaining
    total_reward = 0.0
    total_filled = 0.0
    n_trades = 0
    executed_prices = []
    executed_sizes = []

    for step in range(n_steps):
        # Market dynamics
        price_change = rng.normal(0, volatility) * price
        price += price_change
        spread = avg_spread_bps * (1.0 + 0.5 * rng.random())
        volume_ratio = 0.5 + 1.5 * rng.random()
        vol_ratio = 0.7 + 0.6 * rng.random()
        momentum = price_change / (base_price * volatility + 1e-10)

        time_remaining = 1.0 - (step + 1) / n_steps
        inventory_risk = remaining * (1.0 - time_remaining)

        state = ExecutionState(
            remaining_qty=remaining,
            time_remaining=time_remaining,
            spread_bps=spread,
            volume_ratio=volume_ratio,
            volatility_ratio=vol_ratio,
            momentum=float(np.clip(momentum, -1, 1)),
            inventory_risk=inventory_risk,
        )

        action_idx = agent.select_action(state, training=training)
        action = ACTIONS[action_idx]

        # Execute
        fill_pct = 0.0
        exec_price = price
        if action.action_type in ("execute", "split") and remaining > 0.001:
            fill_pct = min(action.execute_pct, remaining)
            # Execution cost: half spread + impact
            impact_bps = fill_pct * 2.0  # larger orders have more impact
            exec_price = price * (1.0 + (spread / 2 + impact_bps) / 10000.0)
            remaining -= fill_pct
            total_filled += fill_pct
            n_trades += 1
            executed_prices.append(exec_price)
            executed_sizes.append(fill_pct)

        reward = compute_execution_reward(
            exec_price, arrival_price, spread, remaining, time_remaining,
        )

        done = remaining < 0.001 or step == n_steps - 1

        # Force fill at end
        if done and remaining > 0.001:
            fill_pct = remaining
            exec_price = price * (1.0 + spread / 10000.0)
            remaining = 0.0
            total_filled += fill_pct
            n_trades += 1
            executed_prices.append(exec_price)
            executed_sizes.append(fill_pct)
            reward -= 0.5  # penalty for forced fill

        next_state = ExecutionState(
            remaining_qty=remaining,
            time_remaining=max(0, 1.0 - (step + 2) / n_steps),
            spread_bps=spread,
            volume_ratio=volume_ratio,
            volatility_ratio=vol_ratio,
            momentum=float(np.clip(momentum, -1, 1)),
            inventory_risk=remaining * max(0, 1.0 - (step + 2) / n_steps),
        )

        if training:
            agent.update(state, action_idx, reward, next_state, done)

        total_reward += reward

        if done:
            break

    # Compute VWAP
    if executed_prices and executed_sizes:
        sizes = np.array(executed_sizes)
        prices = np.array(executed_prices)
        vwap = float(np.sum(prices * sizes) / np.sum(sizes))
        vwap_slippage = (vwap - arrival_price) / arrival_price * 10000
    else:
        vwap_slippage = 0.0

    return {
        "total_reward": round(total_reward, 4),
        "vwap_slippage_bps": round(float(vwap_slippage), 2),
        "fill_rate": round(float(total_filled), 4),
        "n_trades": n_trades,
        "q_table_size": agent.get_q_table_size(),
    }


def train_execution_agent(
    n_episodes: int = 500,
    n_steps: int = 20,
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> tuple[QLearningExecutionAgent, list[dict[str, float]]]:
    """Train an RL execution agent over multiple episodes.

    Args:
        n_episodes: Number of training episodes.
        n_steps: Steps per episode.
        base_price: Starting price.
        volatility: Price volatility.
        seed: Random seed.

    Returns:
        Tuple of (trained_agent, episode_metrics).
    """
    agent = QLearningExecutionAgent(
        n_bins=10, alpha=0.1, gamma=0.99, epsilon=0.15,
    )
    metrics = []

    for ep in range(n_episodes):
        # Decay epsilon
        agent.epsilon = max(0.01, 0.15 * (1 - ep / n_episodes))

        result = simulate_execution_episode(
            agent, total_qty=1.0, n_steps=n_steps,
            base_price=base_price, volatility=volatility,
            training=True, seed=seed + ep,
        )
        metrics.append(result)

    logger.info(
        "[RLExecution] Trained %d episodes, final avg reward=%.3f, "
        "avg slippage=%.1f bps, Q-table size=%d",
        n_episodes,
        np.mean([m["total_reward"] for m in metrics[-50:]]),
        np.mean([m["vwap_slippage_bps"] for m in metrics[-50:]]),
        agent.get_q_table_size(),
    )

    return agent, metrics


__all__ = [
    "ExecutionState",
    "ExecutionAction",
    "QLearningExecutionAgent",
    "compute_execution_reward",
    "simulate_execution_episode",
    "train_execution_agent",
    "ACTIONS",
]
