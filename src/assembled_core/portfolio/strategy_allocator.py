"""Strategy-level capital allocation.

Provides two independent allocation mechanisms:

1. **StrategyAllocator** — signal-level ensemble combiner.
   Merges signals from multiple Strategy objects via weighted-average,
   majority-vote, or regime-conditional methods.

2. **Risk-parity vol-targeting** — capital-level allocator.
   Weights strategy buckets by inverse-volatility and scales to a
   target portfolio volatility via ``inverse_vol_weights()`` /
   ``allocate_from_returns_dict()``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Observed performance statistics for one strategy bucket."""
    name: str
    returns: list[float]              # daily return series
    realized_vol: float = 0.0        # annualised vol (filled in by allocator)
    weight: float = 0.0              # target weight (filled in by allocator)


@dataclass
class AllocationResult:
    """Output of the strategy allocator."""
    weights: dict[str, float]          # strategy_name → target weight (sum ≈ 1)
    vol_scale: float                   # portfolio-level volatility scalar applied
    estimated_portfolio_vol: float     # annualised vol before scaling (%)
    target_vol: float                  # target annualised vol
    strategy_vols: dict[str, float]    # per-strategy annualised vol


def _annualised_vol(returns: list[float], min_obs: int = 10) -> float:
    """Compute annualised volatility from a daily return series.

    Returns 0 if the series is too short, so the caller can decide how to handle it.
    """
    if len(returns) < min_obs:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = float(np.std(arr, ddof=1))
    return std * math.sqrt(252)


def inverse_vol_weights(
    strategies: list[StrategyStats],
    target_vol: float = 0.15,
    min_vol_floor: float = 0.02,
    max_weight: float = 0.70,
) -> AllocationResult:
    """Compute inverse-volatility risk-parity weights and vol-scale to target.

    Algorithm:
    1. Compute annualised volatility per strategy.
    2. Replace any strategy with vol < min_vol_floor with the floor (prevents
       near-zero strategies from dominating).
    3. Inverse-vol weights: w_i = (1/vol_i) / sum(1/vol_j).
    4. Cap individual weights at max_weight and renormalise.
    5. Estimate blended portfolio vol = sqrt(sum(w_i^2 * vol_i^2))
       (assumes zero cross-correlations — conservative).
    6. Scale weights so that portfolio vol equals target_vol.

    Args:
        strategies: List of StrategyStats with populated .returns.
        target_vol: Desired annualised portfolio volatility (decimal, e.g. 0.15 = 15%).
        min_vol_floor: Minimum vol assigned to any strategy (prevents blow-up in weighting).
        max_weight: Maximum weight for a single strategy.

    Returns:
        AllocationResult with normalised weights and diagnostics.
    """
    if not strategies:
        return AllocationResult(
            weights={}, vol_scale=1.0,
            estimated_portfolio_vol=0.0,
            target_vol=target_vol,
            strategy_vols={},
        )

    # Step 1-2: compute vols, apply floor
    vols: dict[str, float] = {}
    for s in strategies:
        v = _annualised_vol(s.returns)
        vols[s.name] = max(v, min_vol_floor)

    # Step 3: inverse-vol weights
    inv_vol_sum = sum(1.0 / v for v in vols.values())
    raw_weights = {name: (1.0 / v) / inv_vol_sum for name, v in vols.items()}

    # Step 4: cap and renormalise
    capped: dict[str, float] = {}
    for name, w in raw_weights.items():
        capped[name] = min(w, max_weight)
    total = sum(capped.values())
    if total < 1e-9:
        # Edge case: return equal weights
        n = len(strategies)
        capped = {s.name: 1.0 / n for s in strategies}
        total = 1.0
    weights = {name: w / total for name, w in capped.items()}

    # Step 5: estimate portfolio vol (no-correlation assumption)
    port_var = sum(weights[s.name] ** 2 * vols[s.name] ** 2 for s in strategies)
    port_vol = math.sqrt(max(port_var, 1e-12))

    # Step 6: vol scale
    vol_scale = target_vol / max(port_vol, 1e-9)
    # Cap scale to avoid over-leveraging
    vol_scale = min(vol_scale, 2.0)

    scaled_weights = {name: w * vol_scale for name, w in weights.items()}

    # Update StrategyStats in place
    for s in strategies:
        s.realized_vol = vols[s.name]
        s.weight = scaled_weights.get(s.name, 0.0)

    return AllocationResult(
        weights=scaled_weights,
        vol_scale=vol_scale,
        estimated_portfolio_vol=port_vol,
        target_vol=target_vol,
        strategy_vols=vols,
    )


def allocate_from_returns_dict(
    returns_dict: dict[str, list[float]],
    target_vol: float = 0.15,
    min_vol_floor: float = 0.02,
    max_weight: float = 0.70,
) -> AllocationResult:
    """Convenience wrapper: accepts {strategy_name: [daily_returns]} dict.

    Args:
        returns_dict: Mapping from strategy name to daily return series.
        target_vol: Target portfolio annualised vol.
        min_vol_floor: Vol floor per strategy.
        max_weight: Max weight cap per strategy.

    Returns:
        AllocationResult.
    """
    strategies = [
        StrategyStats(name=name, returns=rets)
        for name, rets in returns_dict.items()
    ]
    return inverse_vol_weights(
        strategies=strategies,
        target_vol=target_vol,
        min_vol_floor=min_vol_floor,
        max_weight=max_weight,
    )


# =============================================================================
# Signal-level ensemble combiner (StrategyAllocator / AllocationConfig)
# =============================================================================

try:
    import pandas as pd  # type: ignore[import]
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


@dataclass
class AllocationConfig:
    """Configuration for the signal-level StrategyAllocator ensemble."""
    method: str = "weighted_average"     # "weighted_average" | "majority_vote" | "regime_conditional"
    weights: dict[str, float] = field(default_factory=dict)
    regime_weights: dict[str, dict[str, float]] = field(default_factory=dict)
    score_normalization: bool = True
    min_strategies_required: int = 1

    def __post_init__(self) -> None:
        # Normalise weights so they sum to 1
        if self.weights:
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}


@dataclass
class EnsembleResult:
    """Output of StrategyAllocator.generate_combined_signals()."""
    combined_signals: Any          # pd.DataFrame
    per_strategy_signals: dict[str, Any]
    strategy_contributions: dict[str, float]
    metadata: dict[str, Any]


class StrategyAllocator:
    """Ensemble combiner for multiple Strategy signal sources.

    Combines signals from a dict of Strategy objects using configurable
    weighting or voting methods. Handles strategy failures gracefully —
    a crashing strategy is simply excluded from the ensemble.

    Args:
        strategies: Dict mapping name → Strategy instance.
        config: AllocationConfig controlling combination method and weights.
    """

    def __init__(
        self,
        strategies: dict[str, Any],
        config: AllocationConfig | None = None,
    ) -> None:
        self._strategies = strategies
        self._config = config or AllocationConfig()

        # Fill in equal weights for any strategy not explicitly weighted
        if not self._config.weights:
            n = len(strategies)
            self._config.weights = {name: 1.0 / n for name in strategies} if n else {}

    @property
    def strategy_names(self) -> list[str]:
        return list(self._strategies.keys())

    def generate_combined_signals(
        self,
        prices: Any,
        regime: str | None = None,
        **kwargs: Any,
    ) -> EnsembleResult:
        """Generate combined signals by running all strategies and merging results.

        Args:
            prices: Price DataFrame passed to each strategy's generate_signals().
            regime: Current regime label (used for regime_conditional method).
            **kwargs: Extra kwargs forwarded to generate_signals().

        Returns:
            EnsembleResult with combined signals and diagnostics.
        """
        if not _PANDAS_AVAILABLE:
            return EnsembleResult(
                combined_signals=None,
                per_strategy_signals={},
                strategy_contributions={},
                metadata={"error": "pandas_not_available"},
            )

        per_signals: dict[str, Any] = {}
        for name, strat in self._strategies.items():
            try:
                result = strat.generate_signals(prices, **kwargs)
                df = getattr(result, "signals", result)
                if df is not None and not df.empty:
                    per_signals[name] = df
            except Exception as exc:
                logger.debug("[StrategyAllocator] strategy %s failed: %s", name, exc)

        n_active = len(per_signals)
        if n_active < self._config.min_strategies_required:
            return EnsembleResult(
                combined_signals=pd.DataFrame(),
                per_strategy_signals=per_signals,
                strategy_contributions={},
                metadata={"error": "insufficient_strategies", "n_active": n_active},
            )

        # Determine effective weights
        weights = self._resolve_weights(regime, per_signals)

        if self._config.method == "majority_vote":
            combined = self._majority_vote(per_signals)
        else:
            # weighted_average and regime_conditional use the same aggregation path
            combined = self._weighted_average(per_signals, weights)

        contributions = {name: weights.get(name, 0.0) for name in per_signals}

        return EnsembleResult(
            combined_signals=combined,
            per_strategy_signals=per_signals,
            strategy_contributions=contributions,
            metadata={
                "method": self._config.method,
                "n_strategies_active": n_active,
                "regime": regime,
            },
        )

    def _resolve_weights(
        self,
        regime: str | None,
        active: dict[str, Any],
    ) -> dict[str, float]:
        if (
            self._config.method == "regime_conditional"
            and regime
            and regime in self._config.regime_weights
        ):
            rw = self._config.regime_weights[regime]
            total = sum(rw.get(name, 0.0) for name in active)
            if total > 0:
                return {name: rw.get(name, 0.0) / total for name in active}

        # Normalise base weights over active strategies only
        active_weights = {name: self._config.weights.get(name, 0.0) for name in active}
        total = sum(active_weights.values())
        if total <= 0:
            n = len(active)
            return {name: 1.0 / n for name in active} if n else {}
        return {name: w / total for name, w in active_weights.items()}

    def _weighted_average(
        self,
        per_signals: dict[str, Any],
        weights: dict[str, float],
    ) -> Any:
        all_dfs = []
        for name, df in per_signals.items():
            w = weights.get(name, 0.0)
            tmp = df.copy()
            if "score" in tmp.columns and self._config.score_normalization:
                max_score = tmp["score"].abs().max()
                if max_score > 0:
                    tmp = tmp.copy()
                    tmp["score"] = tmp["score"] / max_score
            tmp["_weight"] = w
            tmp["_name"] = name
            all_dfs.append(tmp)

        if not all_dfs:
            return pd.DataFrame()

        combined_raw = pd.concat(all_dfs, ignore_index=True)

        if "symbol" not in combined_raw.columns or "direction" not in combined_raw.columns:
            return combined_raw

        # Aggregate per symbol: weighted score, majority direction
        def _agg(grp: Any) -> Any:
            if "score" in grp.columns:
                wscore = (grp["score"] * grp["_weight"]).sum()
            else:
                wscore = 0.0
            # Weighted direction vote
            dir_weights: dict[str, float] = {}
            for _, row in grp.iterrows():
                d = str(row.get("direction", "NEUTRAL"))
                dir_weights[d] = dir_weights.get(d, 0.0) + float(row.get("_weight", 1.0))
            best_dir = max(dir_weights, key=dir_weights.get)  # type: ignore[arg-type]
            return pd.Series({"score": wscore, "direction": best_dir})

        result = combined_raw.groupby("symbol", group_keys=False).apply(_agg).reset_index()
        return result

    def _majority_vote(self, per_signals: dict[str, Any]) -> Any:
        all_dfs = []
        for name, df in per_signals.items():
            tmp = df.copy()
            tmp["_name"] = name
            all_dfs.append(tmp)

        if not all_dfs:
            return pd.DataFrame()

        combined_raw = pd.concat(all_dfs, ignore_index=True)

        if "symbol" not in combined_raw.columns or "direction" not in combined_raw.columns:
            return combined_raw

        def _vote(grp: Any) -> Any:
            counts = grp["direction"].value_counts()
            direction = counts.idxmax()
            score = float(grp["score"].mean()) if "score" in grp.columns else 0.0
            return pd.Series({"direction": direction, "score": score})

        result = combined_raw.groupby("symbol", group_keys=False).apply(_vote).reset_index()
        return result
