"""Turnover-Penalty-Wrapper für Positions-Smoothing.

Standard-Sizer erzeugen Target-Positionen ohne Memory. Bei volatilen Signalen
führt das zu exzessivem Turnover und hohen Transaktionskosten.

Lösung: EMA-Smoothing zwischen previous und target Positionen.

    smoothed_t = alpha × target_t + (1 - alpha) × smoothed_{t-1}

Niedriges alpha (z.B. 0.2) = starkes Smoothing, träger.
Hohes alpha (z.B. 0.8) = minimales Smoothing, reagiert schnell.

Komplement zur bestehenden `apply_tc_penalized_rebalancing` in position_sizing.py:
- `apply_tc_penalized_rebalancing`: verändert Sizes basierend auf TC-Threshold
- Turnover-Smoothing hier: EMA-basierte Glättung, orthogonal

PIT-Invariante: Nur previous + target Zustand, keine Zukunftsdaten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_turnover_smoothing(
    target: pd.Series | dict,
    previous: pd.Series | dict | None,
    alpha: float = 0.3,
) -> pd.Series:
    """EMA-Smoothing zwischen target und previous Positionen.

    Args:
        target: Ziel-Positionen (symbol → weight oder pd.Series)
        previous: Letzte Positionen (None = voll target)
        alpha: EMA-Koeffizient [0, 1]. Niedrig = mehr Smoothing.

    Returns:
        pd.Series mit geglätteten Weights.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha muss in [0, 1] sein, bekam {alpha}")

    tgt = target if isinstance(target, pd.Series) else pd.Series(target)

    if previous is None:
        return tgt.copy()

    prev = previous if isinstance(previous, pd.Series) else pd.Series(previous)

    # Union of symbols (neue + alte)
    all_syms = tgt.index.union(prev.index)
    tgt = tgt.reindex(all_syms, fill_value=0.0)
    prev = prev.reindex(all_syms, fill_value=0.0)

    smoothed = alpha * tgt + (1.0 - alpha) * prev
    return smoothed


def compute_turnover(
    target: pd.Series | dict,
    previous: pd.Series | dict | None,
) -> float:
    """One-way Turnover: Σ|target - previous| / 2.

    Typische Interpretation: wenn target = previous × 2 → turnover = 50%.
    """
    tgt = target if isinstance(target, pd.Series) else pd.Series(target)
    if previous is None:
        return float(tgt.abs().sum() / 2.0)
    prev = previous if isinstance(previous, pd.Series) else pd.Series(previous)
    all_syms = tgt.index.union(prev.index)
    tgt = tgt.reindex(all_syms, fill_value=0.0)
    prev = prev.reindex(all_syms, fill_value=0.0)
    return float((tgt - prev).abs().sum() / 2.0)


def compute_turnover_cost(
    target: pd.Series | dict,
    previous: pd.Series | dict | None,
    tc_bps: float = 10.0,
) -> float:
    """Erwartete TC-Kosten in bps bei gegebenem Turnover.

    Args:
        tc_bps: Round-trip Kosten in Basispunkten (default 10 bps).
    """
    turnover = compute_turnover(target, previous)
    # Turnover in [0, 1] bedeutet 100% one-way; TC wird pro one-way berechnet
    return float(turnover * tc_bps / 10000.0)


def enforce_turnover_budget(
    target: pd.Series | dict,
    previous: pd.Series | dict | None,
    max_turnover: float = 0.3,
) -> pd.Series:
    """Skaliert Target linear Richtung Previous wenn Turnover-Budget überschritten.

    Ergebnis: Turnover <= max_turnover.
    """
    tgt = target if isinstance(target, pd.Series) else pd.Series(target)
    if previous is None:
        return tgt.copy()
    prev = previous if isinstance(previous, pd.Series) else pd.Series(previous)

    all_syms = tgt.index.union(prev.index)
    tgt = tgt.reindex(all_syms, fill_value=0.0)
    prev = prev.reindex(all_syms, fill_value=0.0)

    raw_turnover = float((tgt - prev).abs().sum() / 2.0)
    if raw_turnover <= max_turnover or raw_turnover < 1e-9:
        return tgt

    # Linear interpolate: y = prev + t × (tgt - prev), choose t so turnover = max_turnover
    t = max_turnover / raw_turnover
    capped = prev + t * (tgt - prev)
    logger.info(
        "[Turnover] Budget overrun: %.3f → %.3f (cap=%.3f, t=%.2f)",
        raw_turnover, float((capped - prev).abs().sum() / 2.0), max_turnover, t,
    )
    return capped


@dataclass
class TurnoverPenaltyConfig:
    enabled: bool = True
    ema_alpha: float = 0.3
    max_turnover_per_period: float = 0.3
    tc_bps: float = 10.0


class TurnoverConstrainedSizer:
    """Wrapper für bestehende Sizer mit Smoothing + Budget-Constraint."""

    def __init__(self, config: TurnoverPenaltyConfig | None = None) -> None:
        self.config = config or TurnoverPenaltyConfig()
        self._previous: pd.Series | None = None

    def process(self, target: pd.Series | dict) -> pd.Series:
        """Apply EMA smoothing + turnover budget + remember state."""
        if not self.config.enabled:
            result = target if isinstance(target, pd.Series) else pd.Series(target)
            self._previous = result
            return result

        smoothed = apply_turnover_smoothing(target, self._previous, alpha=self.config.ema_alpha)
        capped = enforce_turnover_budget(
            smoothed, self._previous, max_turnover=self.config.max_turnover_per_period,
        )
        self._previous = capped
        return capped

    @property
    def previous_positions(self) -> pd.Series | None:
        return self._previous

    def set_previous(self, previous: pd.Series | dict | None) -> None:
        if previous is None:
            self._previous = None
        else:
            self._previous = previous if isinstance(previous, pd.Series) else pd.Series(previous)


__all__ = [
    "TurnoverPenaltyConfig",
    "TurnoverConstrainedSizer",
    "apply_turnover_smoothing",
    "compute_turnover",
    "compute_turnover_cost",
    "enforce_turnover_budget",
]
