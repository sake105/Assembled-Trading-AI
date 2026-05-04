"""Regime-conditional strategy allocation.

Adjusts strategy weights based on the current market regime.
Each strategy can have different historical Sharpe ratios per regime;
the allocator weights strategies proportionally to their regime-conditional
Sharpe, then applies a vol-targeting scalar.

Regime labels follow the existing convention in ml/regime_hmm.py and
risk/regime_models.py: integer indices or string labels like "BULL" / "BEAR".

This module has no I/O — it accepts plain Python data structures and
returns allocation dicts.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegimePerformance:
    """Historical performance of a strategy in a specific regime."""

    strategy: str
    regime: str | int
    sharpe: float  # historical Sharpe in this regime
    n_obs: int = 0  # number of days observed
    avg_return: float = 0.0
    vol: float = 0.0


@dataclass
class RegimeAllocationResult:
    """Output of the regime-conditional allocator."""

    regime: str | int
    weights: dict[str, float]  # strategy → weight (may not sum to 1 if vol-scaled)
    vol_scale: float
    regime_sharpes: dict[str, float]  # strategy → regime-specific Sharpe used
    n_active_strategies: int


def compute_regime_sharpes(
    strategy_returns: dict[str, list[float]],
    regime_series: list[int | str],
) -> dict[str, dict[str | int, RegimePerformance]]:
    """Compute per-regime Sharpe for each strategy from aligned return / regime series.

    Args:
        strategy_returns: {strategy_name: [daily_returns]} — all series must have
                          the same length as regime_series.
        regime_series: Sequence of regime labels (one per day), same length as returns.

    Returns:
        Nested dict: {strategy_name: {regime_label: RegimePerformance}}.
    """
    result: dict[str, dict[str | int, RegimePerformance]] = {}
    regimes_unique = sorted(set(regime_series), key=str)

    for name, returns in strategy_returns.items():
        arr = np.asarray(returns, dtype=float)
        if len(arr) != len(regime_series):
            logger.warning(
                "[regime_alloc] length mismatch for %s (%d vs %d) — skipping",
                name,
                len(arr),
                len(regime_series),
            )
            continue

        reg_arr = np.array(regime_series)
        perf_by_regime: dict[str | int, RegimePerformance] = {}

        for r in regimes_unique:
            mask = reg_arr == r
            r_returns = arr[mask]
            if len(r_returns) < 5:
                perf_by_regime[r] = RegimePerformance(
                    strategy=name,
                    regime=r,
                    sharpe=0.0,
                    n_obs=int(mask.sum()),
                )
                continue
            mu = float(np.mean(r_returns))
            sigma = float(np.std(r_returns, ddof=1))
            sharpe = mu / max(sigma, 1e-9) * math.sqrt(252)
            perf_by_regime[r] = RegimePerformance(
                strategy=name,
                regime=r,
                sharpe=round(sharpe, 4),
                n_obs=int(mask.sum()),
                avg_return=round(mu, 6),
                vol=round(sigma, 6),
            )

        result[name] = perf_by_regime

    return result


def allocate_by_regime(
    current_regime: str | int,
    regime_performances: dict[str, dict[str | int, RegimePerformance]],
    target_vol: float = 0.15,
    min_sharpe_threshold: float = 0.0,
    max_weight: float = 0.60,
    vol_scale_cap: float = 2.0,
) -> RegimeAllocationResult:
    """Compute strategy weights for the current regime.

    Algorithm:
    1. For each strategy, extract the Sharpe in current_regime.
    2. Drop strategies with Sharpe < min_sharpe_threshold.
    3. Normalise positive Sharpes to weights (Sharpe-proportional).
    4. Cap individual weights at max_weight and renormalise.
    5. Apply vol-targeting scalar.

    Args:
        current_regime: The detected current regime label.
        regime_performances: Output of compute_regime_sharpes().
        target_vol: Target annualised portfolio vol.
        min_sharpe_threshold: Exclude strategies with regime Sharpe below this.
        max_weight: Maximum weight per strategy.
        vol_scale_cap: Maximum vol-scaling multiplier.

    Returns:
        RegimeAllocationResult.
    """
    # Collect regime-specific Sharpes
    sharpes: dict[str, float] = {}
    vols: dict[str, float] = {}

    for strategy, regime_map in regime_performances.items():
        perf = regime_map.get(current_regime)
        if perf is None:
            # Fall back to overall-best regime if current not seen
            if regime_map:
                best_perf = max(regime_map.values(), key=lambda p: p.sharpe)
                perf = best_perf
                logger.debug(
                    "[regime_alloc] regime %s not found for %s; using best-regime fallback (%s Sharpe=%.2f)",
                    current_regime,
                    strategy,
                    best_perf.regime,
                    best_perf.sharpe,
                )
            else:
                continue

        if perf.sharpe < min_sharpe_threshold:
            continue

        sharpes[strategy] = perf.sharpe
        vols[strategy] = max(perf.vol * math.sqrt(252), 0.02)  # annualised

    if not sharpes:
        logger.warning(
            "[regime_alloc] no strategies pass threshold in regime %s; equal-weight fallback",
            current_regime,
        )
        n = len(regime_performances)
        if n == 0:
            return RegimeAllocationResult(
                regime=current_regime,
                weights={},
                vol_scale=1.0,
                regime_sharpes={},
                n_active_strategies=0,
            )
        fallback_w = 1.0 / n
        return RegimeAllocationResult(
            regime=current_regime,
            weights={s: fallback_w for s in regime_performances},
            vol_scale=1.0,
            regime_sharpes={s: 0.0 for s in regime_performances},
            n_active_strategies=n,
        )

    # Normalise positive Sharpes to weights
    total_sharpe = sum(max(s, 0.0) for s in sharpes.values())
    if total_sharpe < 1e-9:
        # All negative — equal weight among selected
        raw_weights = {s: 1.0 / len(sharpes) for s in sharpes}
    else:
        raw_weights = {s: max(sh, 0.0) / total_sharpe for s, sh in sharpes.items()}

    # Cap and renormalise
    capped = {s: min(w, max_weight) for s, w in raw_weights.items()}
    total = sum(capped.values())
    if total < 1e-9:
        total = 1.0
    weights = {s: w / total for s, w in capped.items()}

    # Estimate blended portfolio vol
    port_var = sum(weights.get(s, 0.0) ** 2 * vols.get(s, 0.15) ** 2 for s in weights)
    port_vol = math.sqrt(max(port_var, 1e-12))

    vol_scale = min(target_vol / max(port_vol, 1e-9), vol_scale_cap)
    scaled_weights = {s: w * vol_scale for s, w in weights.items()}

    return RegimeAllocationResult(
        regime=current_regime,
        weights=scaled_weights,
        vol_scale=round(vol_scale, 4),
        regime_sharpes=sharpes,
        n_active_strategies=len(sharpes),
    )


def build_regime_allocator(
    strategy_returns: dict[str, list[float]],
    regime_series: list[int | str],
    target_vol: float = 0.15,
    min_sharpe_threshold: float = 0.0,
    max_weight: float = 0.60,
) -> "RegimeAllocator":
    """Convenience factory: fits regime performances and returns a ready allocator."""
    perfs = compute_regime_sharpes(strategy_returns, regime_series)
    return RegimeAllocator(
        regime_performances=perfs,
        target_vol=target_vol,
        min_sharpe_threshold=min_sharpe_threshold,
        max_weight=max_weight,
    )


class RegimeAllocator:
    """Stateful allocator — call .allocate(regime) to get weights for any regime."""

    def __init__(
        self,
        regime_performances: dict[str, dict[str | int, RegimePerformance]],
        target_vol: float = 0.15,
        min_sharpe_threshold: float = 0.0,
        max_weight: float = 0.60,
    ) -> None:
        self._perfs = regime_performances
        self._target_vol = target_vol
        self._min_sharpe = min_sharpe_threshold
        self._max_weight = max_weight

    def allocate(self, current_regime: str | int) -> RegimeAllocationResult:
        return allocate_by_regime(
            current_regime=current_regime,
            regime_performances=self._perfs,
            target_vol=self._target_vol,
            min_sharpe_threshold=self._min_sharpe,
            max_weight=self._max_weight,
        )

    @property
    def strategies(self) -> list[str]:
        return list(self._perfs.keys())

    @property
    def regimes(self) -> list[str | int]:
        regimes: set[str | int] = set()
        for perf_map in self._perfs.values():
            regimes.update(perf_map.keys())
        return sorted(regimes, key=str)
