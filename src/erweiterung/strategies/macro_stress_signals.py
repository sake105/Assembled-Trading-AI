"""Makro-Stress-Signale aus FRED-/macro.parquet-Daten.

Idee
----
Markt-basierte Stress-Signale (Drawdown, Realized-Vol) sind **reaktiv**: sie
zeigen Stress an, nachdem er schon eingesetzt hat. Makro-Variablen wie VIX,
Yield-Curve-Inversion und High-Yield-Spreads haben dagegen oft Lead-Time von
Wochen bis Monaten.

Dieses Modul bietet zwei orthogonale Makro-Trigger:

1. **VIX-Spike**: rolling-z-Score des VIX gegen 252-Tage-Trail.
2. **Yield-Curve-Stress**: 10y−2y-Spread inverted oder fällt scharf.
3. **HY-Spread-Widening**: HY-Credit-Spread > rolling-90-Tage-Mean × 1.3.
4. **Real-Yield-Spike**: nominale 10y minus T10YIE (Breakeven-Inflation).

Jedes Signal in [0, 1] normalisiert, mit Caller-konfigurierbaren Gewichten
zu einem Macro-Stress-Composite gemittelt.

Limitierung
-----------
- HY-Spread ist im macro.parquet nur 56 % befüllt (43.5 % NaN) — Signal wird
  bei Missing-Wert ausgeschlossen, ohne den Composite zu verfälschen.
- Real-Yield-Signal nutzt FRED T10YIE — siehe ``output/macro_fred.parquet``.

Daten-Pfade
-----------
- ``output/macro.parquet``: VIX, yield_curve_spread, hy_spread (Mainline)
- ``output/macro_fred.parquet``: DGS10, DGS2, FEDFUNDS, T10YIE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MacroStressConfig:
    vix_zscore_window: int = 252
    vix_zscore_threshold: float = 1.0
    yc_inversion_threshold: float = 0.0  # 10y-2y < 0 = inverted
    yc_zscore_window: int = 252
    hy_spread_baseline_window: int = 90
    hy_spread_alarm_ratio: float = 1.3
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "vix_spike": 0.35,
            "yield_curve_stress": 0.30,
            "hy_spread_widening": 0.20,
            "real_yield_spike": 0.15,
        }
    )
    stress_threshold: float = 0.55


def _ensure_dt_index(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], utc=True)
        return df.set_index(col).sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    raise ValueError(f"DataFrame needs '{col}' column or DatetimeIndex")


def load_macro_panel(
    main_path: str | Path = "output/macro.parquet",
    fred_path: str | Path = "output/macro_fred.parquet",
) -> pd.DataFrame:
    """Lade & merge die zwei Makro-Panels.

    Returns:
        DataFrame mit DatetimeIndex (UTC, daily) und den verfügbaren Spalten.
    """
    main_path = Path(main_path)
    fred_path = Path(fred_path)
    main = pd.read_parquet(main_path) if main_path.exists() else pd.DataFrame()
    fred = pd.read_parquet(fred_path) if fred_path.exists() else pd.DataFrame()
    if not main.empty:
        main = _ensure_dt_index(main)
    if not fred.empty:
        fred = _ensure_dt_index(fred)
    if main.empty and fred.empty:
        return pd.DataFrame()
    if main.empty:
        return fred
    if fred.empty:
        return main
    return main.join(fred, how="outer", rsuffix="_fred")


def vix_spike_signal(
    vix: pd.Series, window: int = 252, threshold: float = 1.0
) -> pd.Series:
    """Z-Score-basierte VIX-Spike-Detektion → [0, 1]."""
    mean = vix.rolling(window, min_periods=20).mean()
    std = vix.rolling(window, min_periods=20).std()
    z = (vix - mean) / std.replace(0, np.nan)
    # Map z=threshold -> 0.5, z=threshold+2 -> 1.0, z<0 -> 0
    signal = ((z - threshold) / 2.0 + 0.5).clip(0, 1)
    return signal.fillna(0)


def yield_curve_stress_signal(
    yc_spread: pd.Series,
    inversion_threshold: float = 0.0,
    window: int = 252,
) -> pd.Series:
    """Yield-Curve-Inversion + scharfer Rückgang → [0, 1].

    Hohe Werte bei (1) Inversion oder (2) Rapid Flattening (z-Score < -1).
    """
    inverted = (yc_spread < inversion_threshold).astype(float)
    # Trailing z-score: starkes Flattening (z < -1) zählt auch
    mean = yc_spread.rolling(window, min_periods=20).mean()
    std = yc_spread.rolling(window, min_periods=20).std()
    z = (yc_spread - mean) / std.replace(0, np.nan)
    flattening = (-z / 2.0).clip(0, 1)  # z=-1 -> 0.5, z=-2 -> 1.0
    # Take element-wise max (NaN-safe)
    combined = pd.concat([inverted, flattening], axis=1).max(axis=1)
    return combined.fillna(0)


def hy_spread_widening_signal(
    hy_spread: pd.Series,
    baseline_window: int = 90,
    alarm_ratio: float = 1.3,
) -> pd.Series:
    """HY-Credit-Spread-Widening → [0, 1].

    Wert ≥ baseline × alarm_ratio markiert Stress. Bei NaN wird 0 zurückgegeben.
    """
    if hy_spread.dropna().empty:
        return pd.Series(np.nan, index=hy_spread.index)
    baseline = hy_spread.rolling(baseline_window, min_periods=10).mean()
    ratio = hy_spread / baseline.replace(0, np.nan)
    # ratio=1 -> 0; ratio=alarm_ratio -> 0.5; ratio=alarm_ratio*1.5 -> 1.0
    signal = ((ratio - 1.0) / (alarm_ratio - 1.0)).clip(0, 1)
    return signal


def real_yield_spike_signal(
    nominal_10y: pd.Series, breakeven_10y: pd.Series, window: int = 252
) -> pd.Series:
    """Real-Yield-Spike (10y nominal − 10y breakeven inflation) → [0, 1]."""
    aligned = pd.concat({"n": nominal_10y, "b": breakeven_10y}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(np.nan, index=nominal_10y.index)
    real = aligned["n"] - aligned["b"]
    mean = real.rolling(window, min_periods=20).mean()
    std = real.rolling(window, min_periods=20).std()
    z = (real - mean) / std.replace(0, np.nan)
    # Spike = sehr hoher Real-Yield ggü. trailing
    signal = ((z - 1.0) / 2.0 + 0.5).clip(0, 1)
    return signal.reindex(nominal_10y.index).fillna(0)


def macro_stress_composite(
    macro_panel: pd.DataFrame,
    config: MacroStressConfig | None = None,
) -> pd.DataFrame:
    """Aggregiere die Makro-Stress-Signale → composite + regime label.

    Args:
        macro_panel: DataFrame mit DatetimeIndex. Erforderlich min:
            ``vix`` (oder ``vix_close``), ``yield_curve_spread``.
            Optional: ``hy_spread``, ``treasury_10y`` (nominal), und
            ``T10YIE`` (breakeven inflation, aus FRED).
        config: MacroStressConfig.

    Returns:
        DataFrame [vix_spike, yc_stress, hy_widening, real_yield, composite, regime].
    """
    cfg = config or MacroStressConfig()
    out = pd.DataFrame(index=macro_panel.index)

    # VIX
    vix_col = "vix_close" if "vix_close" in macro_panel.columns else "vix"
    if vix_col in macro_panel.columns:
        out["vix_spike"] = vix_spike_signal(
            macro_panel[vix_col], cfg.vix_zscore_window, cfg.vix_zscore_threshold
        )
    else:
        out["vix_spike"] = np.nan

    # Yield-Curve
    yc_col = (
        "yield_curve_spread" if "yield_curve_spread" in macro_panel.columns else None
    )
    if yc_col:
        out["yield_curve_stress"] = yield_curve_stress_signal(
            macro_panel[yc_col],
            cfg.yc_inversion_threshold,
            cfg.yc_zscore_window,
        )
    else:
        out["yield_curve_stress"] = np.nan

    # HY-Spread
    if "hy_spread" in macro_panel.columns:
        out["hy_spread_widening"] = hy_spread_widening_signal(
            macro_panel["hy_spread"],
            cfg.hy_spread_baseline_window,
            cfg.hy_spread_alarm_ratio,
        )
    else:
        out["hy_spread_widening"] = np.nan

    # Real-Yield
    nom_col = (
        "treasury_10y"
        if "treasury_10y" in macro_panel.columns
        else ("DGS10" if "DGS10" in macro_panel.columns else None)
    )
    if nom_col and "T10YIE" in macro_panel.columns:
        out["real_yield_spike"] = real_yield_spike_signal(
            macro_panel[nom_col], macro_panel["T10YIE"]
        )
    else:
        out["real_yield_spike"] = np.nan

    # Composite
    cols = ["vix_spike", "yield_curve_stress", "hy_spread_widening", "real_yield_spike"]
    weight_arr = np.array([cfg.weights.get(c, 0.0) for c in cols])
    valid_mask = out[cols].notna().values
    weights_per_row = valid_mask * weight_arr[None, :]
    sum_weights = weights_per_row.sum(axis=1)
    sum_weights = np.where(sum_weights == 0, 1.0, sum_weights)
    composite = (out[cols].fillna(0).values * weights_per_row).sum(axis=1) / sum_weights
    out["composite_score"] = composite
    out["regime"] = np.where(composite >= cfg.stress_threshold, "stress", "calm")
    return out


__all__ = [
    "MacroStressConfig",
    "load_macro_panel",
    "vix_spike_signal",
    "yield_curve_stress_signal",
    "hy_spread_widening_signal",
    "real_yield_spike_signal",
    "macro_stress_composite",
]
