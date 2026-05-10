"""Lightweight PPO-Agent for Portfolio-Allocation.

Reference
---------
Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv 1707.06347.

Implementation
--------------
- Actor-Network: state → softmax-weights (long-only) bzw. tanh + projection.
- Critic-Network: state → value.
- Clipped-Surrogate-Loss (PPO-Clip).

Hinweis: Diese Implementierung ist **didaktisch**. Für ernsthaftes Training
empfehle ich ``stable-baselines3`` — siehe ``rl/__init__.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    hidden_dim: int = 64
    n_epochs: int = 5
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_iterations: int = 50
    rollout_steps: int = 256


def _import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore

        return torch, nn, F
    except ImportError as e:
        raise RuntimeError("torch required") from e


def make_actor_critic(obs_dim: int, action_dim: int, config: PPOConfig):
    torch, nn, _ = _import_torch()

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
            )
            self.mean_head = nn.Linear(config.hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        def forward(self, x):
            h = self.net(x)
            mean = self.mean_head(h)
            std = self.log_std.exp().expand_as(mean)
            return mean, std

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Tanh(),
                nn.Linear(config.hidden_dim, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return Actor(), Critic()


def train_ppo(env, config: PPOConfig | None = None, verbose: bool = False):
    """Trainiere PPO-Agent auf einem PortfolioEnv."""
    torch, nn, F = _import_torch()
    config = config or PPOConfig()

    obs0 = env.reset()
    obs_dim = len(obs0)
    action_dim = env.n_assets
    actor, critic = make_actor_critic(obs_dim, action_dim, config)
    opt_a = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    opt_c = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    rewards_log = []
    for it in range(config.n_iterations):
        # rollout
        observations, actions, log_probs, rewards, values, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        obs = env.reset()
        for _ in range(config.rollout_steps):
            obs_t = torch.from_numpy(obs.astype(np.float32))
            with torch.no_grad():
                mean, std = actor(obs_t)
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample()
                logp = dist.log_prob(a).sum()
                v = critic(obs_t)
            # softmax-ish projection: clip + sum=1 (long-only)
            a_np = a.numpy()
            a_pos = np.clip(a_np, 0, None) + 1e-9
            a_pos = a_pos / a_pos.sum()
            obs_next, r, done, _info = env.step(a_pos)

            observations.append(obs)
            actions.append(a_np)
            log_probs.append(logp.item())
            rewards.append(r)
            values.append(v.item())
            dones.append(float(done))
            obs = obs_next
            if done:
                obs = env.reset()

        # GAE
        values.append(0.0)
        adv = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t] + config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            )
            gae = delta + config.gamma * config.lam * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + np.array(values[:-1])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO updates
        obs_t = torch.from_numpy(np.stack(observations).astype(np.float32))
        act_t = torch.from_numpy(np.stack(actions).astype(np.float32))
        logp_old = torch.from_numpy(np.array(log_probs).astype(np.float32))
        adv_t = torch.from_numpy(adv)
        ret_t = torch.from_numpy(returns.astype(np.float32))
        for _ in range(config.n_epochs):
            mean, std = actor(obs_t)
            dist = torch.distributions.Normal(mean, std)
            logp = dist.log_prob(act_t).sum(dim=-1)
            ratio = (logp - logp_old).exp()
            unclipped = ratio * adv_t
            clipped = (
                torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv_t
            )
            actor_loss = -torch.min(unclipped, clipped).mean()
            opt_a.zero_grad()
            actor_loss.backward()
            opt_a.step()

            v_pred = critic(obs_t)
            critic_loss = F.mse_loss(v_pred, ret_t)
            opt_c.zero_grad()
            critic_loss.backward()
            opt_c.step()

        avg_r = float(np.mean(rewards))
        rewards_log.append(avg_r)
        if verbose:
            logger.info("[ppo iter %d] avg_r=%.5f", it, avg_r)

    return actor, critic, rewards_log


__all__ = ["PPOConfig", "make_actor_critic", "train_ppo"]
