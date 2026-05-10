"""Multi-Armed-Bandit für adaptive Strategy-Selection.

Algorithms
----------
- **ε-greedy**: random exploration mit Wahrscheinlichkeit ε
- **UCB1** (Auer et al. 2002): mean + √(2 ln t / n_i) upper-confidence-bound
- **Thompson Sampling** (Thompson 1933): Beta-Bernoulli oder Normal-Normal Posterior

Anwendung
---------
- Adaptive Strategy-Allocation: bei mehreren Strategien lerne welche aktuell
  funktioniert. Vorteil ggü. Hedge: explizit explorativ.
- Multi-Stock-Trading: jedes Asset = Arm, reward = realized Sharpe over window.

Reference
---------
- Lattimore & Szepesvari (2020). *Bandit Algorithms*. Cambridge.
- Auer, Cesa-Bianchi & Fischer (2002). Finite-time Analysis of the Multiarmed
  Bandit Problem. *Machine Learning* 47.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class BanditState:
    n_arms: int
    counts: np.ndarray = field(default=None)  # n_i
    rewards_sum: np.ndarray = field(default=None)  # Σ r
    rewards_sq_sum: np.ndarray = field(default=None)  # Σ r²
    t: int = 0

    def __post_init__(self):
        if self.counts is None:
            self.counts = np.zeros(self.n_arms)
        if self.rewards_sum is None:
            self.rewards_sum = np.zeros(self.n_arms)
        if self.rewards_sq_sum is None:
            self.rewards_sq_sum = np.zeros(self.n_arms)

    def mean_reward(self, arm: int) -> float:
        return self.rewards_sum[arm] / self.counts[arm] if self.counts[arm] > 0 else 0.0

    def reward_std(self, arm: int) -> float:
        if self.counts[arm] < 2:
            return 1.0
        mean = self.mean_reward(arm)
        return float(np.sqrt(self.rewards_sq_sum[arm] / self.counts[arm] - mean**2))


class EpsilonGreedy:
    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 42):
        self.state = BanditState(n_arms=n_arms)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self) -> int:
        if self.rng.random() < self.epsilon or (self.state.counts == 0).any():
            # Explore — pick an unselected arm first, else random
            zero_idx = np.where(self.state.counts == 0)[0]
            if len(zero_idx) > 0:
                return int(zero_idx[0])
            return int(self.rng.integers(0, self.state.n_arms))
        # Exploit
        means = np.array([self.state.mean_reward(i) for i in range(self.state.n_arms)])
        return int(np.argmax(means))

    def update(self, arm: int, reward: float) -> None:
        self.state.counts[arm] += 1
        self.state.rewards_sum[arm] += reward
        self.state.rewards_sq_sum[arm] += reward**2
        self.state.t += 1


class UCB1:
    """UCB1 (Auer et al. 2002)."""

    def __init__(self, n_arms: int, c: float = 1.0):
        self.state = BanditState(n_arms=n_arms)
        self.c = c

    def select(self) -> int:
        # Force-play each arm once
        zero_idx = np.where(self.state.counts == 0)[0]
        if len(zero_idx) > 0:
            return int(zero_idx[0])
        # UCB
        t = self.state.t
        ucbs = np.zeros(self.state.n_arms)
        for i in range(self.state.n_arms):
            mean = self.state.mean_reward(i)
            bonus = self.c * np.sqrt(2 * np.log(t + 1) / self.state.counts[i])
            ucbs[i] = mean + bonus
        return int(np.argmax(ucbs))

    def update(self, arm: int, reward: float) -> None:
        self.state.counts[arm] += 1
        self.state.rewards_sum[arm] += reward
        self.state.rewards_sq_sum[arm] += reward**2
        self.state.t += 1


class ThompsonGaussian:
    """Thompson Sampling für Normal-Normal-Conjugate Bandit.

    Prior μ ~ N(0, 1), Likelihood r_t | μ ~ N(μ, σ²) (σ² known/estimated).
    Posterior: μ | r_1...r_n ~ N(μ_n, σ_n²) — standard.
    """

    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        obs_std: float = 1.0,
        seed: int = 42,
    ):
        self.state = BanditState(n_arms=n_arms)
        self.prior_mean = prior_mean
        self.prior_var = prior_std**2
        self.obs_var = obs_std**2
        self.rng = np.random.default_rng(seed)

    def _posterior(self, arm: int) -> tuple[float, float]:
        n = self.state.counts[arm]
        if n == 0:
            return self.prior_mean, self.prior_var
        sample_mean = self.state.mean_reward(arm)
        post_var = 1.0 / (1.0 / self.prior_var + n / self.obs_var)
        post_mean = post_var * (
            self.prior_mean / self.prior_var + n * sample_mean / self.obs_var
        )
        return post_mean, post_var

    def select(self) -> int:
        samples = np.zeros(self.state.n_arms)
        for i in range(self.state.n_arms):
            mu, var = self._posterior(i)
            samples[i] = self.rng.normal(mu, np.sqrt(var))
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        self.state.counts[arm] += 1
        self.state.rewards_sum[arm] += reward
        self.state.rewards_sq_sum[arm] += reward**2
        self.state.t += 1


def run_bandit_on_strategy_returns(
    strategy_returns: "pd.DataFrame",
    algorithm: str = "ucb1",
    window: int = 21,
):
    """Run a bandit over time on strategy daily returns.

    Args:
        strategy_returns: DataFrame T × K of daily returns.
        algorithm: 'epsilon', 'ucb1', 'thompson'.
        window: window for reward = trailing-Sharpe.

    Returns:
        DataFrame T × K — binary "selected" plus combined PnL series.
    """
    import pandas as pd

    K = strategy_returns.shape[1]
    if algorithm == "epsilon":
        bandit = EpsilonGreedy(K, epsilon=0.15)
    elif algorithm == "ucb1":
        bandit = UCB1(K, c=2.0)
    elif algorithm == "thompson":
        bandit = ThompsonGaussian(K)
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")

    selected_history = []
    combined = []
    for t in range(len(strategy_returns)):
        chosen = bandit.select()
        ret_today = float(strategy_returns.iloc[t, chosen])
        combined.append(ret_today)
        selected_history.append(chosen)
        # reward = trailing Sharpe of chosen arm
        if t >= window:
            sub = strategy_returns.iloc[t - window : t, chosen]
            if sub.std() > 0:
                reward = float(sub.mean() / sub.std())
            else:
                reward = 0.0
        else:
            reward = ret_today * 100  # raw return scaled
        bandit.update(chosen, reward)

    return pd.DataFrame(
        {
            "chosen": selected_history,
            "return": combined,
        },
        index=strategy_returns.index,
    )


__all__ = [
    "BanditState",
    "EpsilonGreedy",
    "UCB1",
    "ThompsonGaussian",
    "run_bandit_on_strategy_returns",
]
