"""RL execution agent — PPO policy wrapper for order execution (skeleton).

Wraps stable-baselines3 PPO to train and deploy a continuous-action
execution agent on OrderExecutionEnv.

When stable-baselines3 is not installed, the module exposes a fallback
RuleBasedExecutor that mimics the interface (TWAP-like).

This is intentionally a skeleton: the environment dynamics are simplified
and the agent is not production-ready. Wire into the full backtesting
pipeline after thorough simulation validation.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.assembled_core.execution.rl_environment import (
    GYM_AVAILABLE,
    ExecutionEnvConfig,
    OrderExecutionEnv,
)

logger = logging.getLogger(__name__)

_SB3_AVAILABLE = False
try:
    from stable_baselines3 import PPO  # type: ignore[import]
    from stable_baselines3.common.env_checker import check_env  # type: ignore[import]  # noqa: F401

    _SB3_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# PPO-based executor
# ---------------------------------------------------------------------------


class RLExecutor:
    """Train and run a PPO agent for order execution.

    Args:
        config: Environment configuration.
        model_path: Path to save/load trained model (None = in-memory only).
        total_timesteps: Training steps (default 50_000 for quick demo).
    """

    def __init__(
        self,
        config: ExecutionEnvConfig | None = None,
        model_path: str | None = None,
        total_timesteps: int = 50_000,
    ) -> None:
        self.config = config or ExecutionEnvConfig()
        self._model_path = model_path
        self._total_timesteps = total_timesteps
        self._model: Any = None
        self._backend = "ppo" if (_SB3_AVAILABLE and GYM_AVAILABLE) else "rule_based"

    def train(self) -> None:
        """Train the PPO agent on the execution environment."""
        if not (_SB3_AVAILABLE and GYM_AVAILABLE):
            logger.warning(
                "[RLExecutor] stable-baselines3 or gymnasium not installed; "
                "training is a no-op. Fallback to TWAP rule."
            )
            return

        env = OrderExecutionEnv(self.config)
        self._model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=256,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            ent_coef=0.01,
            seed=self.config.seed,
        )
        logger.info("[RLExecutor] training PPO for %d timesteps", self._total_timesteps)
        self._model.learn(total_timesteps=self._total_timesteps, progress_bar=False)

        if self._model_path:
            self._model.save(self._model_path)
            logger.info("[RLExecutor] model saved to %s", self._model_path)

    def load(self, path: str | None = None) -> None:
        """Load a previously trained model."""
        if not (_SB3_AVAILABLE and GYM_AVAILABLE):
            return
        p = path or self._model_path
        if p and Path(p).exists():
            self._model = PPO.load(p)
            logger.info("[RLExecutor] model loaded from %s", p)
        else:
            logger.warning("[RLExecutor] model path not found: %s", p)

    def execute(
        self,
        n_steps: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Run one execution episode and return diagnostics.

        Args:
            n_steps: Override config n_steps for this episode.
            seed: Random seed for price path.

        Returns:
            Dict with avg_execution_price, implementation_shortfall, total_reward, slices.
        """
        cfg_override = ExecutionEnvConfig(
            total_shares=self.config.total_shares,
            n_steps=n_steps or self.config.n_steps,
            arrival_price=self.config.arrival_price,
            sigma_daily=self.config.sigma_daily,
            eta=self.config.eta,
            gamma_perm=self.config.gamma_perm,
            lambda_risk=self.config.lambda_risk,
            bid_ask_spread=self.config.bid_ask_spread,
            seed=seed or self.config.seed,
        )
        env = OrderExecutionEnv(cfg_override)
        obs, _ = env.reset(seed=seed)

        total_reward = 0.0
        slices: list[dict[str, Any]] = []
        done = False

        while not done:
            if self._model is not None:
                action, _ = self._model.predict(obs, deterministic=True)
            else:
                # TWAP fallback: uniform fraction
                _remaining = float(obs[0]) * cfg_override.total_shares
                steps_left = max(1, round(float(obs[1]) * cfg_override.n_steps))
                action = np.array([1.0 / steps_left], dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            slices.append(info)
            done = terminated or truncated

        total_executed = sum(s["shares_executed"] for s in slices)
        if slices:
            avg_price = float(
                np.mean(
                    [s["execution_price"] for s in slices if s["shares_executed"] > 0]
                )
            )
        else:
            avg_price = cfg_override.arrival_price

        shortfall = (avg_price - cfg_override.arrival_price) * total_executed

        return {
            "avg_execution_price": round(avg_price, 4),
            "arrival_price": cfg_override.arrival_price,
            "implementation_shortfall": round(shortfall, 2),
            "shortfall_bps": round(
                shortfall
                / (cfg_override.arrival_price * cfg_override.total_shares)
                * 10_000,
                2,
            ),
            "total_reward": round(float(total_reward), 6),
            "shares_executed": total_executed,
            "n_slices": len(slices),
            "backend": self._backend,
        }


# ---------------------------------------------------------------------------
# Rule-based fallback (TWAP)
# ---------------------------------------------------------------------------


class RuleBasedExecutor:
    """TWAP-like executor with the same interface as RLExecutor."""

    def __init__(self, config: ExecutionEnvConfig | None = None) -> None:
        self.config = config or ExecutionEnvConfig()
        self._backend = "twap_rule"

    def train(self) -> None:
        pass  # no training needed

    def execute(
        self, n_steps: int | None = None, seed: int | None = None
    ) -> dict[str, Any]:
        cfg = self.config
        n = n_steps or cfg.n_steps
        slice_qty = cfg.total_shares // n
        remainder = cfg.total_shares % n

        rng = np.random.default_rng(seed or cfg.seed)
        prices = []
        for i in range(n):
            qty = slice_qty + (1 if i < remainder else 0)
            temp_impact = cfg.eta * qty * cfg.sigma_daily**2 / cfg.arrival_price
            exec_price = cfg.arrival_price * (
                1.0 + temp_impact + rng.normal(0, cfg.sigma_daily / math.sqrt(n))
            )
            prices.append(exec_price)

        avg_price = float(np.mean(prices))
        shortfall = (avg_price - cfg.arrival_price) * cfg.total_shares
        return {
            "avg_execution_price": round(avg_price, 4),
            "arrival_price": cfg.arrival_price,
            "implementation_shortfall": round(shortfall, 2),
            "shortfall_bps": round(
                shortfall / (cfg.arrival_price * cfg.total_shares) * 10_000, 2
            ),
            "total_reward": 0.0,
            "shares_executed": cfg.total_shares,
            "n_slices": n,
            "backend": self._backend,
        }


SB3_AVAILABLE = _SB3_AVAILABLE
