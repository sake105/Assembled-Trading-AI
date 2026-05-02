"""RL execution environment — Gymnasium-compatible order execution env (skeleton).

Models the order execution problem as a Markov Decision Process:
  - State: remaining shares, time elapsed, current mid-price, spread, imbalance.
  - Action: fraction of remaining shares to execute in this step (continuous [0,1]).
  - Reward: negative implementation shortfall (price impact + opportunity cost).

This is a skeleton / proof-of-concept. The environment is correct enough to
train a basic PPO agent via stable-baselines3, but the market simulation is
intentionally simplified (linear permanent impact, no LOB dynamics).

Requires: gymnasium (or gym), numpy.
stable-baselines3 is optional (only needed for training, not for the env itself).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import gymnasium as gym  # type: ignore[import]
    from gymnasium import spaces
    _GYM_AVAILABLE = True
    _GYM_BACKEND = "gymnasium"
except ImportError:
    try:
        import gym  # type: ignore[import]  # noqa: F401
        from gym import spaces  # type: ignore[import]
        _GYM_AVAILABLE = True
        _GYM_BACKEND = "gym"
    except ImportError:
        _GYM_AVAILABLE = False
        _GYM_BACKEND = None
        # Provide minimal stubs so the module can be imported without gym
        class spaces:  # type: ignore[no-redef]
            class Box:
                def __init__(self, *a: Any, **kw: Any) -> None: ...
            class Env:
                pass


@dataclass
class ExecutionEnvConfig:
    """Configuration for the RL execution environment."""
    total_shares: int = 10_000       # parent order size
    n_steps: int = 20                # time slices available
    arrival_price: float = 100.0     # price at order arrival
    sigma_daily: float = 0.015       # daily vol (σ)
    eta: float = 0.10                # temporary impact coefficient
    gamma_perm: float = 0.05         # permanent impact coefficient
    lambda_risk: float = 1e-5        # risk aversion (variance penalty)
    bid_ask_spread: float = 0.02     # fixed spread (simplification)
    seed: int = 42


class OrderExecutionEnv:
    """Gymnasium-compatible execution environment.

    Observation (5-dim):
        [0] remaining_frac   — shares remaining / total_shares  ∈ [0, 1]
        [1] time_frac        — steps remaining / n_steps        ∈ [0, 1]
        [2] price_frac       — current mid / arrival_price - 1  ∈ [-0.2, 0.2]
        [3] spread_frac      — bid-ask / mid                    ∈ [0, 0.02]
        [4] imbalance        — fake order-book imbalance signal  ∈ [-1, 1]

    Action (1-dim continuous):
        fraction of remaining shares to execute in this step ∈ [0, 1]

    Reward:
        -implementation_shortfall per step (normalised by arrival_price * total_shares)
        Shortfall = permanent impact (already paid) + temporary impact (this slice)

    Terminal reward includes opportunity cost from unsold shares.
    """

    def __init__(self, config: ExecutionEnvConfig | None = None) -> None:
        self.config = config or ExecutionEnvConfig()
        self._rng = np.random.default_rng(self.config.seed)

        obs_low  = np.array([0.0, 0.0, -0.20, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0,  0.20, 0.02, 1.0], dtype=np.float32)

        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=obs_low, high=obs_high, dtype=np.float32
            )
            self.action_space = spaces.Box(
                low=np.float32(0.0), high=np.float32(1.0),
                shape=(1,), dtype=np.float32,
            )

        self._reset_state()

    def _reset_state(self) -> None:
        cfg = self.config
        self._remaining = cfg.total_shares
        self._step_idx = 0
        self._price = cfg.arrival_price
        self._perm_impact_acc = 0.0   # accumulated permanent impact (price drift)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        obs = self._observe()
        return obs, {}

    def step(
        self,
        action: np.ndarray | float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        if hasattr(action, '__len__'):
            frac = float(np.clip(float(action[0]), 0.0, 1.0))
        else:
            frac = float(np.clip(float(action), 0.0, 1.0))

        shares_this_step = max(0, round(frac * self._remaining))

        # On last step, execute everything remaining
        if self._step_idx == cfg.n_steps - 1:
            shares_this_step = self._remaining

        shares_this_step = min(shares_this_step, self._remaining)

        # Temporary impact: slippage = eta * qty * sigma^2 / price
        temp_impact_frac = cfg.eta * shares_this_step * cfg.sigma_daily ** 2 / max(self._price, 1e-6)
        execution_price = self._price * (1.0 + temp_impact_frac + cfg.bid_ask_spread / 2)

        # Permanent impact: price drifts by gamma * qty * sigma^2 / price
        perm_impact_frac = cfg.gamma_perm * shares_this_step * cfg.sigma_daily ** 2 / max(self._price, 1e-6)
        self._price *= (1.0 + perm_impact_frac)

        # Simulate next-step price return (GBM)
        dt = 1.0 / cfg.n_steps
        gbm_ret = self._rng.normal(0, cfg.sigma_daily * math.sqrt(dt))
        self._price *= (1.0 + gbm_ret)

        # Implementation shortfall for this slice
        shortfall = (execution_price - cfg.arrival_price) * shares_this_step
        # Variance penalty
        variance_penalty = cfg.lambda_risk * (self._remaining ** 2) * cfg.sigma_daily ** 2

        reward = -(shortfall + variance_penalty) / max(cfg.arrival_price * cfg.total_shares, 1.0)

        self._remaining -= shares_this_step
        self._step_idx += 1

        terminated = (self._remaining <= 0) or (self._step_idx >= cfg.n_steps)
        truncated = False

        # Opportunity cost for unsold shares at terminal step
        if terminated and self._remaining > 0:
            opp_cost = (self._price - cfg.arrival_price) * self._remaining
            reward -= opp_cost / max(cfg.arrival_price * cfg.total_shares, 1.0)

        info = {
            "shares_executed": shares_this_step,
            "remaining": self._remaining,
            "execution_price": execution_price,
            "current_mid": self._price,
            "step": self._step_idx,
        }

        return self._observe(), float(reward), terminated, truncated, info

    def _observe(self) -> np.ndarray:
        cfg = self.config
        remaining_frac = self._remaining / max(cfg.total_shares, 1)
        time_frac = max(0.0, (cfg.n_steps - self._step_idx) / cfg.n_steps)
        price_frac = float(np.clip((self._price / cfg.arrival_price) - 1.0, -0.20, 0.20))
        spread_frac = float(np.clip(cfg.bid_ask_spread / max(self._price, 1e-6), 0.0, 0.02))
        # Synthetic imbalance: random walk bounded in [-1,1]
        imbalance = float(np.clip(self._rng.normal(0, 0.3), -1.0, 1.0))
        return np.array([remaining_frac, time_frac, price_frac, spread_frac, imbalance],
                        dtype=np.float32)


GYM_AVAILABLE = _GYM_AVAILABLE
GYM_BACKEND = _GYM_BACKEND
