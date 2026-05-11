"""Regime-Conditional Allocator — schaltet zwischen Equal-Weight und Factor-Tilt.

Idee
----
Aus dem Expanded-Universe-Backtest folgt (siehe
``docs/erweiterung/EXPANDED_UNIVERSE_BACKTEST.md``):

- Im **Inflation/Bear-Regime** liefert Factor-Tilt-Long-Only (Momentum +
  Residual-Mom) ≈ 11-14 pp Outperformance gegen Equal-Weight.
- Im **Trend/Recovery-Regime** ist Equal-Weight kompetitiv.

Eine simple Trailing-Regime-Detection erlaubt es, die Allokation zwischen den
beiden Modi zu switchen, ohne Cherry-Picking ex-post (die Regime-Klassifikation
basiert nur auf Vergangenheitsdaten, t−1).

Wichtig: das ist eine **regelbasierte** Allokator-Schicht, kein neues
Signal-Modell. Sie verwendet existierende Strategie-Returns als Inputs.

Definition Regime
-----------------
Wir benutzen den 60-Tage-Trailing-Drawdown des Equal-Weight-Benchmarks:
- Drawdown > 8 % → "stress" → aktiviere Factor-Tilt
- sonst → "calm" → Equal-Weight

8 % ist konservativ, kein Hyperparameter-Tuning; siehe Sub-Period-Daten
(Inflation_2022 hatte Benchmark-MDD ≈ 23 %, also clear über Threshold).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RegimeConfig:
    drawdown_threshold: float = 0.08
    lookback_days: int = 60
    smoothing_days: int = 3
    """Smoothing-Days verhindern Tagesflicker."""


def detect_regime(
    benchmark_returns: pd.Series, config: RegimeConfig | None = None
) -> pd.Series:
    """Bestimme Regime-Label pro Tag aus Benchmark-Drawdown.

    Args:
        benchmark_returns: Tages-Returns der Referenzstrategie (z. B. Equal-Weight).
        config: RegimeConfig.

    Returns:
        Series of {"calm", "stress"}, indexed wie benchmark_returns.
    """
    cfg = config or RegimeConfig()
    eq = (1 + benchmark_returns.fillna(0)).cumprod()
    rolling_max = eq.rolling(cfg.lookback_days, min_periods=1).max()
    dd = (eq / rolling_max - 1).abs()
    raw = np.where(dd > cfg.drawdown_threshold, "stress", "calm")
    out = pd.Series(raw, index=benchmark_returns.index, name="regime")
    if cfg.smoothing_days > 1:
        # Mode-Smoothing: Block min smoothing_days bevor regime-flip greift
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
                    # keep run counter — small flips überschrieben
                else:
                    cleaned.append(cur)
                    last = cur
                    run = 1
        out = pd.Series(cleaned, index=out.index, name="regime")
    return out


def allocate_regime_conditional(
    equal_weight_returns: pd.Series,
    factor_tilt_returns: pd.Series,
    config: RegimeConfig | None = None,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Schalte zwischen zwei Strategie-Streams basierend auf Regime.

    Args:
        equal_weight_returns: "calm"-Mode-Returns (Tages-Returns).
        factor_tilt_returns: "stress"-Mode-Returns.
        config: RegimeConfig.
        lag_days: Regime-Klassifikation wird um lag_days nach hinten geshiftet
            (vermeidet Look-Ahead; Default 1).

    Returns:
        DataFrame [regime, allocated_return, calm_return, stress_return].
    """
    cfg = config or RegimeConfig()
    aligned = pd.concat(
        {"calm": equal_weight_returns, "stress": factor_tilt_returns}, axis=1
    ).dropna()
    if aligned.empty:
        return pd.DataFrame()

    regime = detect_regime(aligned["calm"], cfg)
    regime_lagged = regime.shift(lag_days)
    out = aligned.copy()
    out["regime"] = regime_lagged
    out["allocated_return"] = np.where(
        out["regime"] == "stress", out["stress"], out["calm"]
    )
    return out.rename(columns={"calm": "calm_return", "stress": "stress_return"})[
        ["regime", "calm_return", "stress_return", "allocated_return"]
    ]


def regime_metrics(allocation: pd.DataFrame) -> dict:
    """Aggregate Performance-Diagnostik nach Regime."""
    if allocation.empty or "allocated_return" not in allocation.columns:
        return {}
    out: dict = {}
    for label in ("calm", "stress", "all"):
        if label == "all":
            sub = allocation["allocated_return"].dropna()
        else:
            sub = allocation.loc[
                allocation["regime"] == label, "allocated_return"
            ].dropna()
        if sub.empty or sub.std() == 0:
            out[label] = {
                "n_days": int(len(sub)),
                "ann_return": None,
                "sharpe": None,
                "max_dd": None,
            }
            continue
        ann_ret = (1 + sub).prod() ** (252 / len(sub)) - 1
        ann_vol = sub.std() * np.sqrt(252)
        eq = (1 + sub).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        out[label] = {
            "n_days": int(len(sub)),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else None,
            "max_dd": float(dd),
        }
    return out


__all__ = [
    "RegimeConfig",
    "detect_regime",
    "allocate_regime_conditional",
    "regime_metrics",
]
