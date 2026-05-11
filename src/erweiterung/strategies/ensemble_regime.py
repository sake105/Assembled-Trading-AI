"""Ensemble-Regime-Detector — kombiniert mehrere Regime-Signals via Voting.

Idee
----
Die drei vorhandenen Regime-Detectors haben **unterschiedliche Stärken**:

- ``regime_conditional_allocator`` (Drawdown-Only): bestes MDD, slow, lag-heavy.
- ``multi_signal_regime`` (Drawdown + RV + Dispersion + News): bestes Sharpe,
  reactive.
- ``macro_stress_signals`` (VIX + Yield-Curve + HY-Spread + Real-Yield):
  längste Lead-Time, aber 2021-2026 wenig Trennschärfe.

Statt einen "besten" Trigger zu wählen, kombiniert dieser Detector alle drei
über drei Voting-Schemata:

1. **MAJORITY**: ≥ 2 von 3 Detectors sagen stress → stress.
2. **CONSERVATIVE**: alle 3 Detectors müssen einig sein → stress.
3. **ANY**: einer reicht (max-Sensitivität, niedrigste MDD-Toleranz).

Jeder Detector kommt mit einer Confidence-Series (composite_score in [0, 1]).
Der Ensemble-Score ist Mean (oder Max bei `ANY`-Mode) der Confidences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class EnsembleConfig:
    voting_scheme: Literal["majority", "conservative", "any", "weighted_mean"] = (
        "weighted_mean"
    )
    threshold: float = 0.50
    smoothing_days: int = 3
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "drawdown": 0.40,
            "multi_signal": 0.30,
            "macro": 0.30,
        }
    )


def _smooth_regime(
    regime: pd.Series, smoothing_days: int = 3, calm_label: str = "calm"
) -> pd.Series:
    if smoothing_days <= 1 or regime.empty:
        return regime
    last = regime.iloc[0]
    run = 1
    cleaned = [last]
    for t in range(1, len(regime)):
        cur = regime.iloc[t]
        if cur == last:
            run += 1
            cleaned.append(cur)
        else:
            if run < smoothing_days:
                cleaned.append(last)
            else:
                cleaned.append(cur)
                last = cur
                run = 1
    return pd.Series(cleaned, index=regime.index, name=regime.name)


def ensemble_regime(
    drawdown_regime: pd.Series | None = None,
    multi_signal_regime_in: pd.Series | None = None,
    macro_regime_in: pd.Series | None = None,
    drawdown_score: pd.Series | None = None,
    multi_signal_score: pd.Series | None = None,
    macro_score: pd.Series | None = None,
    config: EnsembleConfig | None = None,
) -> pd.DataFrame:
    """Ensemble aus mehreren Regime-Detector-Outputs.

    Args:
        drawdown_regime/multi_signal_regime_in/macro_regime_in: optionale
            regime-labels (Series mit "calm"/"stress").
        drawdown_score/multi_signal_score/macro_score: optionale composite-
            scores in [0, 1] (für weighted_mean).
        config: EnsembleConfig.

    Returns:
        DataFrame [ensemble_score, regime, n_voting].
    """
    cfg = config or EnsembleConfig()

    # Stelle sicher, dass wir min. einen Input haben
    label_inputs: dict[str, pd.Series] = {}
    score_inputs: dict[str, pd.Series] = {}
    if drawdown_regime is not None:
        label_inputs["drawdown"] = drawdown_regime
    if multi_signal_regime_in is not None:
        label_inputs["multi_signal"] = multi_signal_regime_in
    if macro_regime_in is not None:
        label_inputs["macro"] = macro_regime_in
    if drawdown_score is not None:
        score_inputs["drawdown"] = drawdown_score
    if multi_signal_score is not None:
        score_inputs["multi_signal"] = multi_signal_score
    if macro_score is not None:
        score_inputs["macro"] = macro_score

    if not label_inputs and not score_inputs:
        return pd.DataFrame()

    # Align all on a common index
    all_index = None
    for s in list(label_inputs.values()) + list(score_inputs.values()):
        idx = s.index if hasattr(s, "index") else None
        if idx is not None:
            all_index = idx if all_index is None else all_index.union(idx)

    if all_index is None:
        return pd.DataFrame()

    if cfg.voting_scheme == "weighted_mean":
        out = pd.DataFrame(index=all_index)
        cols = []
        for k in ("drawdown", "multi_signal", "macro"):
            if k in score_inputs:
                out[f"{k}_score"] = score_inputs[k].reindex(all_index)
                cols.append(f"{k}_score")
            elif k in label_inputs:
                # convert label to {0, 1}
                lab = label_inputs[k].reindex(all_index)
                out[f"{k}_score"] = (lab == "stress").astype(float)
                cols.append(f"{k}_score")
        if not cols:
            return pd.DataFrame()
        weight_map = {
            f"{k}_score": cfg.weights.get(k, 0.0)
            for k in ("drawdown", "multi_signal", "macro")
        }
        weight_arr = np.array([weight_map.get(c, 0.0) for c in cols])
        valid_mask = out[cols].notna().values
        weights_per_row = valid_mask * weight_arr[None, :]
        sum_weights = weights_per_row.sum(axis=1)
        sum_weights = np.where(sum_weights == 0, 1.0, sum_weights)
        ensemble = (out[cols].fillna(0).values * weights_per_row).sum(
            axis=1
        ) / sum_weights
        out["ensemble_score"] = ensemble
        out["regime"] = np.where(ensemble >= cfg.threshold, "stress", "calm")
        out["regime"] = _smooth_regime(out["regime"], cfg.smoothing_days)
        return out

    # Majority / Conservative / Any
    votes = pd.DataFrame(index=all_index)
    for k, lab in label_inputs.items():
        votes[k] = (lab.reindex(all_index) == "stress").astype(float)
    if votes.empty:
        return pd.DataFrame()

    n_voting = votes.notna().sum(axis=1)
    stress_count = votes.sum(axis=1)

    if cfg.voting_scheme == "majority":
        out_regime = np.where(stress_count >= np.ceil(n_voting / 2), "stress", "calm")
    elif cfg.voting_scheme == "conservative":
        out_regime = np.where(stress_count >= n_voting, "stress", "calm")
    elif cfg.voting_scheme == "any":
        out_regime = np.where(stress_count >= 1, "stress", "calm")
    else:
        raise ValueError(f"Unknown voting scheme: {cfg.voting_scheme}")

    out = pd.DataFrame(
        {
            "ensemble_score": stress_count / n_voting.replace(0, np.nan),
            "regime": out_regime,
            "n_voting": n_voting,
        },
        index=all_index,
    )
    out["regime"] = _smooth_regime(out["regime"], cfg.smoothing_days)
    return out


__all__ = ["EnsembleConfig", "ensemble_regime"]
