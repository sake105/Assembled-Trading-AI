"""Train an RL order-execution agent using PPO on the synthetic execution environment.

Usage:
    python scripts/train_rl_executor.py \
        [--total-shares 10000] \
        [--n-steps 20] \
        [--timesteps 100000] \
        [--out models/rl_executor.zip]

Requires: stable-baselines3, gymnasium (or gym).
Falls back to TWAP baseline evaluation when SB3 is not installed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL execution agent")
    parser.add_argument("--total-shares", type=int, default=10_000)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--arrival-price", type=float, default=100.0)
    parser.add_argument("--sigma-daily", type=float, default=0.015)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--out", default="models/rl_executor", help="Model save path (no extension)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10, help="Episodes to evaluate after training")
    args = parser.parse_args()

    from src.assembled_core.execution.rl_environment import ExecutionEnvConfig, GYM_AVAILABLE
    from src.assembled_core.execution.rl_execution import RLExecutor, RuleBasedExecutor, SB3_AVAILABLE

    env_cfg = ExecutionEnvConfig(
        total_shares=args.total_shares,
        n_steps=args.n_steps,
        arrival_price=args.arrival_price,
        sigma_daily=args.sigma_daily,
        seed=args.seed,
    )

    if not (SB3_AVAILABLE and GYM_AVAILABLE):
        log.warning("stable-baselines3 or gymnasium not installed — running TWAP baseline only")
        baseline = RuleBasedExecutor(config=env_cfg)
        shortfalls = []
        for ep in range(args.eval_episodes):
            res = baseline.execute(n_steps=args.n_steps, seed=args.seed + ep)
            shortfalls.append(res["shortfall_bps"])
            log.info("[TWAP ep=%d] shortfall=%.2f bps", ep, res["shortfall_bps"])
        avg = sum(shortfalls) / max(len(shortfalls), 1)
        log.info("[TWAP baseline] avg shortfall=%.2f bps over %d episodes", avg, len(shortfalls))
        return

    # Ensure output dir exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    executor = RLExecutor(config=env_cfg, model_path=args.out, total_timesteps=args.timesteps)
    log.info("Training PPO for %d timesteps (total_shares=%d, n_steps=%d)...", args.timesteps, args.total_shares, args.n_steps)
    executor.train()

    # Evaluate trained agent vs TWAP baseline
    log.info("Evaluating trained agent (%d episodes)...", args.eval_episodes)
    rl_shortfalls = []
    twap_shortfalls = []
    baseline = RuleBasedExecutor(config=env_cfg)

    for ep in range(args.eval_episodes):
        rl_res = executor.execute(n_steps=args.n_steps, seed=1000 + ep)
        twap_res = baseline.execute(n_steps=args.n_steps, seed=1000 + ep)
        rl_shortfalls.append(rl_res["shortfall_bps"])
        twap_shortfalls.append(twap_res["shortfall_bps"])

    rl_avg = sum(rl_shortfalls) / max(len(rl_shortfalls), 1)
    twap_avg = sum(twap_shortfalls) / max(len(twap_shortfalls), 1)
    improvement_bps = twap_avg - rl_avg

    summary = {
        "model_path": args.out,
        "timesteps_trained": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "rl_avg_shortfall_bps": round(rl_avg, 3),
        "twap_avg_shortfall_bps": round(twap_avg, 3),
        "improvement_bps": round(improvement_bps, 3),
        "backend": "ppo",
    }
    log.info("[RESULT] RL shortfall=%.2f bps, TWAP=%.2f bps, improvement=%.2f bps", rl_avg, twap_avg, improvement_bps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
