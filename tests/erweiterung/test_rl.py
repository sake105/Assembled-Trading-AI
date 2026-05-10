"""Tests for erweiterung.rl (offline, no torch required for env)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.rl.portfolio_env import PortfolioEnv, PortfolioEnvConfig


def _toy_returns():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, (200, 3)),
        index=pd.date_range("2024-01-01", periods=200, freq="B"),
        columns=["A", "B", "C"],
    )


def test_env_reset_and_step():
    env = PortfolioEnv(_toy_returns(), PortfolioEnvConfig(lookback=10))
    obs = env.reset()
    assert obs.shape[0] == 3 + 10 * 3  # weights + lookback*assets
    obs2, r, done, info = env.step(np.array([0.4, 0.4, 0.2]))
    assert isinstance(r, float)
    assert "equity" in info
    assert obs2.shape == obs.shape


def test_env_long_only_constraint():
    env = PortfolioEnv(_toy_returns(), PortfolioEnvConfig(lookback=5, long_only=True))
    env.reset()
    _, _, _, _ = env.step(np.array([-0.5, 0.5, 0.0]))
    assert (env.weights >= 0).all()
    assert abs(env.weights.sum() - 1.0) < 1e-9


def test_env_runs_to_completion():
    env = PortfolioEnv(_toy_returns(), PortfolioEnvConfig(lookback=5))
    env.reset()
    done = False
    n_steps = 0
    while not done and n_steps < 500:
        _, _, done, _ = env.step(np.array([0.3, 0.4, 0.3]))
        n_steps += 1
    assert done
    assert env.equity > 0
