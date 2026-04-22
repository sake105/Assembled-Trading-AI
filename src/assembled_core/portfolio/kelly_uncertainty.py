"""Kelly Criterion mit Uncertainty-Penalty.

Standard-Kelly maximiert log-Wealth unter Annahme perfekt bekannter
win_prob + payoff. In der Praxis sind beide geschätzt mit Fehlerband.

Lösung: Kelly × (1 - uncertainty_penalty), wo penalty aus Conformal-Half-Width
oder Bootstrap-Intervall kommt.

    kelly_fraction = edge / variance       (Standard)
    uncertainty_scale = 1 - clip(cw / ref_cw, 0, 1)
    final = kelly_fraction × uncertainty_scale × fractional_kelly

Fractional-Kelly (default 0.5) ist zusätzliche Konservativitätsstufe.

Ergänzt `compute_kelly_weights` in position_sizing.py (bleibt unverändert).

PIT-Invariante: Uncertainty kommt aus Conformal-Calibration auf historischen Daten.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_kelly_with_uncertainty(
    edge: float,
    variance: float,
    conformal_half_width: float | None = None,
    reference_half_width: float | None = None,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.25,
) -> float:
    """Kelly-Fraction mit Uncertainty-Discount.

    Args:
        edge: Erwarteter Edge (e.g. E[return] - risk_free).
        variance: Varianz des Returns.
        conformal_half_width: Aktuelles Prediction-Intervall (niedrig = sicher).
        reference_half_width: Historisches "normales" Intervall; None = kein scaling.
        fractional_kelly: Globaler Discount (default 0.5).
        max_fraction: Harte Obergrenze.

    Returns:
        Position-Fraction in [0, max_fraction].
    """
    if variance <= 1e-12:
        return 0.0

    kelly = edge / variance

    # Uncertainty scale: 1.0 = volle Sicherheit, 0.0 = maximale Unsicherheit
    if conformal_half_width is None or reference_half_width is None or reference_half_width <= 1e-12:
        uncertainty_scale = 1.0
    else:
        relative_uncertainty = conformal_half_width / reference_half_width
        uncertainty_scale = float(max(0.0, 1.0 - min(1.0, relative_uncertainty - 1.0 if relative_uncertainty > 1.0 else 0.0)))

    final = kelly * uncertainty_scale * fractional_kelly
    final = float(np.clip(final, -max_fraction, max_fraction))
    return final


def compute_kelly_weights_with_uncertainty(
    edges: pd.Series,
    variances: pd.Series,
    conformal_half_widths: pd.Series | None = None,
    reference_half_width: float | None = None,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.25,
    normalize: bool = True,
) -> pd.Series:
    """Batch-Variante: Kelly-Weights mit per-Symbol-Uncertainty.

    Args:
        edges: Edge pro Symbol
        variances: Varianz pro Symbol
        conformal_half_widths: Optional pro Symbol; None = kein Uncertainty-Discount
        reference_half_width: Gemeinsamer Referenz-Wert
        fractional_kelly: 0.5 = half-Kelly
        max_fraction: Cap pro Position
        normalize: Sum(|weights|) = 1 normalisieren

    Returns:
        pd.Series mit Weights pro Symbol.
    """
    if variances.min() <= 1e-12:
        variances = variances.clip(lower=1e-6)

    weights = pd.Series(0.0, index=edges.index, name="kelly_weight")
    for sym in edges.index:
        cw = conformal_half_widths[sym] if (conformal_half_widths is not None and sym in conformal_half_widths.index) else None
        w = compute_kelly_with_uncertainty(
            edge=float(edges[sym]),
            variance=float(variances[sym]),
            conformal_half_width=cw,
            reference_half_width=reference_half_width,
            fractional_kelly=fractional_kelly,
            max_fraction=max_fraction,
        )
        weights[sym] = w

    if normalize:
        total = weights.abs().sum()
        if total > 1e-9:
            weights = weights / total

    logger.info(
        "[KellyUnc] %d Weights berechnet, mean|w|=%.4f",
        len(weights), float(weights.abs().mean()),
    )
    return weights


__all__ = [
    "compute_kelly_with_uncertainty",
    "compute_kelly_weights_with_uncertainty",
]
