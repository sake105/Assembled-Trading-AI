"""Tail-Risk-Hedging via VIX-Spike-Trigger.

Idee
----
Master-Allocator-Default ist always-on. Tail-Risk-Hedging reduziert die
Exposure (oder schaltet auf Cash), wenn VIX einen Stress-Spike zeigt.
Da volle Options-Daten zu sparse sind, nutzen wir VIX-Z-Score als
Proxy für Tail-Stress.

Trigger-Definitionen
--------------------
1. **VIX-Z-Spike**: VIX > 252-day-Mean + spike_threshold × Std
   → exposure_during_stress (Default 50 %)
2. **VIX-Level**: VIX > absolute_threshold (z. B. > 30)
   → exposure_during_stress

Backtest-Methodik
-----------------
- Trailing-VIX-Stats: kein Look-Ahead
- t-1-Lag: Trigger-Entscheidung von gestern wirkt heute
- Re-Engage: bei VIX < re_engage_threshold (typisch < mean), volle Exposure

Historische Performance-Erwartung
---------------------------------
- 2008 GFC: VIX bis 82, Trigger feuert klar
- 2020 COVID: VIX bis 82, Trigger feuert
- 2022 Inflation: VIX nur bis 35-40, Trigger feuert teilweise
- Trade-off: AnnRet verloren (de-risk im falschen Moment), aber
  MDD-Reduktion in Tail-Phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TailHedgeConfig:
    vix_window: int = 252
    spike_zscore_threshold: float = 1.5
    vix_absolute_threshold: float = 30.0
    use_zscore: bool = True  # True = z-score-trigger, False = absolute-level
    exposure_during_stress: float = 0.50
    exposure_normal: float = 1.0
    re_engage_zscore: float = 0.5
    re_engage_absolute: float = 22.0
    smoothing_days: int = 3


def vix_stress_trigger(
    vix: pd.Series, config: TailHedgeConfig | None = None
) -> pd.Series:
    """Stress-Trigger basierend auf VIX (z-score oder absolute level).

    Args:
        vix: pd.Series mit VIX-Level (date-indexed).
        config: TailHedgeConfig.

    Returns:
        Series mit Werten "stress" / "normal" (date-indexed).
    """
    cfg = config or TailHedgeConfig()
    if vix.empty:
        return pd.Series(dtype=str)

    if cfg.use_zscore:
        mean = vix.rolling(cfg.vix_window, min_periods=20).mean()
        std = vix.rolling(cfg.vix_window, min_periods=20).std()
        z = (vix - mean) / std.replace(0, np.nan)
        # Stateful: stay stress until z < re_engage_zscore
        state = "normal"
        states = []
        for v in z:
            if pd.isna(v):
                states.append(state)
                continue
            if state == "normal" and v > cfg.spike_zscore_threshold:
                state = "stress"
            elif state == "stress" and v < cfg.re_engage_zscore:
                state = "normal"
            states.append(state)
        out = pd.Series(states, index=vix.index, name="trigger")
    else:
        state = "normal"
        states = []
        for v in vix:
            if pd.isna(v):
                states.append(state)
                continue
            if state == "normal" and v > cfg.vix_absolute_threshold:
                state = "stress"
            elif state == "stress" and v < cfg.re_engage_absolute:
                state = "normal"
            states.append(state)
        out = pd.Series(states, index=vix.index, name="trigger")

    # Smoothing
    if cfg.smoothing_days > 1:
        last = out.iloc[0]
        run = 1
        cleaned = [last]
        for t in range(1, len(out)):
            cur = out.iloc[t]
            if cur == last:
                run += 1
                cleaned.append(cur)
            else:
                if run < cfg.smoothing_days:
                    cleaned.append(last)
                else:
                    cleaned.append(cur)
                    last = cur
                    run = 1
        out = pd.Series(cleaned, index=out.index, name="trigger")

    return out


def apply_tail_hedge(
    portfolio_returns: pd.Series,
    vix: pd.Series,
    config: TailHedgeConfig | None = None,
) -> pd.DataFrame:
    """Wende VIX-Stress-Trigger auf Portfolio-Returns an.

    Args:
        portfolio_returns: Daily returns des Master-Allocators (z. B.).
        vix: VIX-Series, gleicher Date-Index.
        config: TailHedgeConfig.

    Returns:
        DataFrame [trigger, exposure, raw_return, hedged_return].
    """
    cfg = config or TailHedgeConfig()
    aligned = pd.concat({"r": portfolio_returns, "v": vix}, axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()
    trigger = vix_stress_trigger(aligned["v"], cfg)
    # t-1 lag: today's exposure based on yesterday's trigger
    trigger_lag = trigger.shift(1)
    exposure = np.where(
        trigger_lag == "stress", cfg.exposure_during_stress, cfg.exposure_normal
    )
    hedged = aligned["r"] * exposure
    out = pd.DataFrame(
        {
            "trigger": trigger,
            "exposure": exposure,
            "raw_return": aligned["r"],
            "hedged_return": hedged,
        }
    )
    return out


__all__ = ["TailHedgeConfig", "vix_stress_trigger", "apply_tail_hedge"]
