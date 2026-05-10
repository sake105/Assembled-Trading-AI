"""Portfolio-Allocation Gym-style Environment.

Observation
-----------
- Aktuelle Gewichte (n_assets,)
- Trailing Returns (lookback × n_assets)
- Optional: Volatility, Momentum, Faktor-Features

Action
------
Continuous (n_assets,) — neue Ziel-Gewichte. Constraints: Σ = 1, w ∈ [0, 1].

Reward
------
Differential-Sharpe-Ratio (Moody/Saffell 2001) — exponentially-weighted moving
mean / std mit incremental updates.  Penaliziert Turnover via Transaction-Costs.

Reference
---------
- Moody, J. & Saffell, M. (2001). Learning to Trade via Direct Reinforcement.
  *IEEE Trans. Neural Networks* 12(4).
- Jiang, Z., Xu, D. & Liang, J. (2017). A Deep RL Framework for Financial
  Portfolio Management. *arXiv 1706.10059*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioEnvConfig:
    lookback: int = 30
    transaction_cost_bps: float = 10.0  # 10 bps roundtrip
    initial_cash: float = 1.0
    eta: float = 0.01  # ewma rate for differential sharpe
    long_only: bool = True
    max_per_asset: float = 0.4


class PortfolioEnv:
    """Pseudo-Gym Env (no gym dependency)."""

    def __init__(
        self,
        returns: pd.DataFrame,
        config: Optional[PortfolioEnvConfig] = None,
    ):
        self.returns = returns.dropna(how="all").fillna(0).values
        self.dates = returns.index
        self.n_assets = self.returns.shape[1]
        self.config = config or PortfolioEnvConfig()
        self.t = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.equity = self.config.initial_cash
        self.equity_curve: list[float] = []
        self._a = 0.0  # ewma mean
        self._b = 0.0  # ewma sq

    def reset(self) -> np.ndarray:
        self.t = self.config.lookback
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.equity = self.config.initial_cash
        self.equity_curve = []
        self._a = 0.0
        self._b = 0.0
        return self._observe()

    def _observe(self) -> np.ndarray:
        win = self.returns[self.t - self.config.lookback : self.t]
        return np.concatenate([self.weights, win.flatten()])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.asarray(action, dtype=float)
        if self.config.long_only:
            a = np.clip(a, 0, self.config.max_per_asset)
        else:
            a = np.clip(a, -self.config.max_per_asset, self.config.max_per_asset)
        if a.sum() <= 0:
            a = np.ones_like(a) / len(a)
        else:
            a = a / a.sum()

        # Transaction cost
        turnover = float(np.abs(a - self.weights).sum())
        tc = turnover * self.config.transaction_cost_bps / 10000.0

        # PnL today
        r_t = float(self.weights @ self.returns[self.t]) - tc
        self.equity *= 1 + r_t
        self.equity_curve.append(self.equity)
        self.weights = a

        # Differential Sharpe (Moody-Saffell)
        eta = self.config.eta
        diff_a = r_t - self._a
        diff_b = r_t**2 - self._b
        if self._b - self._a**2 > 1e-12:
            d_sharpe = (self._b * diff_a - 0.5 * self._a * diff_b) / (
                self._b - self._a**2
            ) ** 1.5
        else:
            d_sharpe = r_t
        self._a = self._a + eta * diff_a
        self._b = self._b + eta * diff_b
        reward = float(d_sharpe)

        self.t += 1
        done = self.t >= len(self.returns)
        return (
            self._observe(),
            reward,
            done,
            {"equity": self.equity, "turnover": turnover},
        )


__all__ = ["PortfolioEnvConfig", "PortfolioEnv"]
