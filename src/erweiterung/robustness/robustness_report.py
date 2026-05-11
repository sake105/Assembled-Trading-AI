"""Robustness-Report — aggregiert Sub-Period + Regime-Conditional + Sensitivity
in einer Composite-Robustness-Score-Tabelle.

Liefert pro Strategy einen einzigen "Robustness-Score" ∈ [0, 1]:
- 25% Sub-Period-Consistency (alle Epochen-Sharpe stable)
- 25% Worst-Epoch-Performance (Sharpe in schlechtester Epoche > 0)
- 25% Parameter-Stability (Sharpe um optimum smooth)
- 25% Regime-Coverage (Sharpe positiv in ≥ 60% der Regimes)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.robustness.parameter_sensitivity import stability_score
from erweiterung.robustness.regime_conditional import returns_by_regime
from erweiterung.robustness.sub_period import (
    consistency_score,
    sub_period_metrics,
    worst_period_sharpe,
)


def robustness_score(
    returns: pd.Series,
    regime: pd.Series | None = None,
    parameter_sweep_df: pd.DataFrame | None = None,
) -> dict:
    """Composite-Score 0..1 with sub-scores.

    Args:
        returns: Series of strategy returns.
        regime: optional regime-label-Series.
        parameter_sweep_df: optional from parameter_sensitivity.parameter_sweep.

    Returns:
        Dict with composite_score + subscores.
    """
    sub_df = sub_period_metrics(returns)
    sub_consistency = consistency_score(returns)
    worst = worst_period_sharpe(returns)
    worst_sharpe = worst.get("sharpe", float("nan"))

    # Sub-Period consistency normalized → [0,1]
    sp_score = (
        float(np.clip((sub_consistency + 1) / 2, 0, 1))
        if np.isfinite(sub_consistency)
        else 0.5
    )
    # Worst-Epoch: tanh-mapping so Sharpe > 0 = > 0.5
    we_score = (
        float(0.5 * (1 + np.tanh(worst_sharpe))) if np.isfinite(worst_sharpe) else 0.0
    )

    # Parameter stability
    if parameter_sweep_df is not None and not parameter_sweep_df.empty:
        stab = stability_score(parameter_sweep_df)
        ps_score = float(np.clip(stab, 0, 1)) if np.isfinite(stab) else 0.5
    else:
        ps_score = 0.5

    # Regime coverage
    if regime is not None:
        regime_df = returns_by_regime(returns, regime)
        if not regime_df.empty:
            pos_fraction = float((regime_df["sharpe_ann"] > 0).mean())
            rg_score = float(np.clip(pos_fraction, 0, 1))
        else:
            rg_score = 0.5
    else:
        rg_score = 0.5

    composite = 0.25 * sp_score + 0.25 * we_score + 0.25 * ps_score + 0.25 * rg_score

    return {
        "composite_score": float(composite),
        "sub_period_consistency_score": sp_score,
        "worst_period_score": we_score,
        "parameter_stability_score": ps_score,
        "regime_coverage_score": rg_score,
        "sub_period_table": sub_df.to_dict("records"),
        "worst_epoch": worst,
    }


__all__ = ["robustness_score"]
