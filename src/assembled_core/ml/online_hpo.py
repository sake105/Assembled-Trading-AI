"""Online Hyperparameter Adaptation via Thompson Sampling (Multi-Armed Bandit).

Statt fixer Hyperparameter: kleiner Satz von Param-Kombis als "Arme". Nach jedem
Retrain wird der gemessene IC als Reward verwendet. Thompson Sampling zieht den
nächsten Arm proportional zur posterior Erfolgswahrscheinlichkeit.

Vorteile gegenüber Optuna WF-HPO (Round 3):
- läuft dauerhaft online — kein separater Optimization-Run nötig
- adaptiert an Regime-Wechsel: Arme, die historisch gut waren, verlieren
  Gewicht wenn sie jetzt schlecht performen
- klein und schnell: 4-8 Arme reichen typischerweise

PIT-Invariante: Rewards werden aus realisierter IC berechnet, nicht geleakt.

State-Persistenz: `output/ml/online_hpo_state.json`
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Statistik für ein Arm (Param-Kombi).

    n_pulls ist `float`, damit OnlineHyperparamAdapter optional einen
    exponentiellen Discount anwenden kann (effektive Sample-Size für
    regime-adaptive Bandits). Bei discount_factor=1 bleibt n_pulls ganzzahlig.
    """

    arm_id: str
    params: dict
    n_pulls: float = 0.0
    sum_reward: float = 0.0
    sum_reward_sq: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.sum_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    @property
    def var_reward(self) -> float:
        if self.n_pulls < 2:
            return 1.0
        mean = self.mean_reward
        return max(
            1e-4,
            (self.sum_reward_sq - self.n_pulls * mean ** 2) / max(1.0, self.n_pulls - 1),
        )


class OnlineHyperparamAdapter:
    """Thompson-Sampling über kleine Menge von Hyperparameter-Kombinationen.

    Rewards: Normal-Verteilung (für kontinuierliche Rewards wie IC).
    Prior: N(0, 1). Posterior: N(mean_reward, var_reward / n_pulls).

    Usage:
        adapter = OnlineHyperparamAdapter.with_default_arms()
        # Vor Retrain:
        chosen_arm = adapter.select_arm()
        params = chosen_arm.params
        # Nach Retrain:
        adapter.observe_reward(chosen_arm.arm_id, reward=measured_ic)
        adapter.save()
    """

    DEFAULT_LGBM_ARMS = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4},
        {"n_estimators": 500, "learning_rate": 0.02, "max_depth": 8},
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    ]

    DEFAULT_SKLEARN_GB_ARMS = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 4},
        {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 2},
    ]

    DEFAULT_RIDGE_ARMS = [
        {"alpha": 0.01},
        {"alpha": 0.1},
        {"alpha": 1.0},
        {"alpha": 10.0},
    ]

    def __init__(
        self,
        arms: list[dict] | None = None,
        state_path: Path | None = None,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        seed: int = 42,
        discount_factor: float = 1.0,
    ) -> None:
        """Args:
            arms: Liste von Param-Dicts. Default: 6 LGBM-Arme.
            state_path: Pfad für Persistenz.
            prior_mean, prior_std: Prior für Bayesian-Updates.
            seed: RNG-Seed.
            discount_factor: In (0, 1]. < 1 bedeutet, dass alte Beobachtungen
                bei jedem neuen Reward exponentiell abklingen. Default 1.0 (kein
                Discount; klassisches Thompson Sampling). Typisch für regime-
                adaptive Bandits: 0.95–0.99.
        """
        self.arms: dict[str, ArmStats] = {}
        if arms is None:
            arms = self.DEFAULT_LGBM_ARMS
        for i, params in enumerate(arms):
            aid = f"arm_{i}"
            self.arms[aid] = ArmStats(arm_id=aid, params=dict(params))

        self.state_path = state_path or Path("output/ml/online_hpo_state.json")
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self._rng = np.random.default_rng(seed)
        if not (0.0 < discount_factor <= 1.0):
            raise ValueError("discount_factor must be in (0, 1]")
        self.discount_factor = discount_factor
        self._load()

    @classmethod
    def with_default_arms(cls, state_path: Path | None = None) -> "OnlineHyperparamAdapter":
        return cls(state_path=state_path)

    @classmethod
    def with_sklearn_gb_arms(cls, state_path: Path | None = None) -> "OnlineHyperparamAdapter":
        """Preset für sklearn GradientBoosting (kein LightGBM nötig)."""
        return cls(arms=cls.DEFAULT_SKLEARN_GB_ARMS, state_path=state_path)

    @classmethod
    def with_ridge_arms(cls, state_path: Path | None = None) -> "OnlineHyperparamAdapter":
        """Preset für Ridge/Linear-Modelle (nur alpha)."""
        return cls(arms=cls.DEFAULT_RIDGE_ARMS, state_path=state_path)

    @classmethod
    def from_param_grid(
        cls,
        param_grid: dict[str, list],
        state_path: Path | None = None,
        seed: int = 42,
    ) -> "OnlineHyperparamAdapter":
        """Baut Arme als kartesisches Produkt aus einem Param-Grid.

        Beispiel:
            param_grid = {"alpha": [0.1, 1.0], "l1_ratio": [0.0, 0.5]}
            → 4 Arme mit allen Kombinationen.

        Für Modelle mit vielen Parametern sollte der Grid klein bleiben
        (≤ 8 Arme ist typisch), damit Thompson Sampling schnell konvergiert.
        """
        from itertools import product
        keys = list(param_grid.keys())
        if not keys:
            return cls(arms=[], state_path=state_path, seed=seed)
        value_lists = [param_grid[k] for k in keys]
        arms = [dict(zip(keys, combo)) for combo in product(*value_lists)]
        return cls(arms=arms, state_path=state_path, seed=seed)

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            for arm_dict in data.get("arms", []):
                aid = arm_dict["arm_id"]
                if aid in self.arms:
                    self.arms[aid].n_pulls = float(arm_dict.get("n_pulls", 0.0))
                    self.arms[aid].sum_reward = float(arm_dict.get("sum_reward", 0.0))
                    self.arms[aid].sum_reward_sq = float(arm_dict.get("sum_reward_sq", 0.0))
        except Exception as exc:
            logger.warning("[OnlineHPO] Load failed: %s", exc)

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "arms": [
                {
                    "arm_id": a.arm_id,
                    "params": a.params,
                    "n_pulls": a.n_pulls,
                    "sum_reward": a.sum_reward,
                    "sum_reward_sq": a.sum_reward_sq,
                }
                for a in self.arms.values()
            ]
        }
        self.state_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def select_arm(self) -> ArmStats:
        """Thompson Sampling: ziehe pro Arm aus Posterior, wähle argmax."""
        best_arm = None
        best_sample = -np.inf
        for arm in self.arms.values():
            if arm.n_pulls == 0:
                # Prior: N(prior_mean, prior_std)
                sample = self.prior_mean + self._rng.normal(0, self.prior_std)
            else:
                std = np.sqrt(arm.var_reward / arm.n_pulls)
                sample = arm.mean_reward + self._rng.normal(0, std)
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        if best_arm is None:
            best_arm = next(iter(self.arms.values()))
        logger.info(
            "[OnlineHPO] Selected %s (mean=%.4f, n=%d): %s",
            best_arm.arm_id, best_arm.mean_reward, best_arm.n_pulls, best_arm.params,
        )
        return best_arm

    def observe_reward(self, arm_id: str, reward: float) -> None:
        if arm_id not in self.arms:
            logger.warning("[OnlineHPO] Unknown arm %s — reward ignored", arm_id)
            return
        arm = self.arms[arm_id]
        df = self.discount_factor
        if df < 1.0:
            # Exponential discount: jeder neue Reward gewichtet alten Ballast um df ab.
            # Das hält effektive Sample-Size endlich und erlaubt Regime-Adaption.
            arm.n_pulls = df * arm.n_pulls + 1
            arm.sum_reward = df * arm.sum_reward + float(reward)
            arm.sum_reward_sq = df * arm.sum_reward_sq + float(reward) ** 2
        else:
            arm.n_pulls += 1
            arm.sum_reward += float(reward)
            arm.sum_reward_sq += float(reward) ** 2
        logger.info(
            "[OnlineHPO] %s reward=%.4f → n=%.2f mean=%.4f",
            arm_id, reward, arm.n_pulls, arm.mean_reward,
        )

    def best_arm(self) -> ArmStats | None:
        """Arm mit höchstem mean_reward (bei ≥ 1 Pull)."""
        candidates = [a for a in self.arms.values() if a.n_pulls > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.mean_reward)

    def summary(self) -> dict:
        return {
            "n_arms": len(self.arms),
            "arms": [
                {
                    "arm_id": a.arm_id,
                    "params": a.params,
                    "n_pulls": a.n_pulls,
                    "mean_reward": round(a.mean_reward, 4),
                }
                for a in self.arms.values()
            ],
        }


__all__ = [
    "ArmStats",
    "OnlineHyperparamAdapter",
]
