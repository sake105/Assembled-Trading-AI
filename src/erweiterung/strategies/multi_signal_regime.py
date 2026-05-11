"""Multi-Signal Regime-Detector — Aggregat aus mehreren Stress-Indikatoren.

Idee
----
Der `regime_conditional_allocator` nutzt nur Trailing-Drawdown als
Stress-Signal. Das ist robust, aber lag-behaftet: bis ein 8-%-MDD aufgebaut
ist, vergehen mehrere Wochen.

Dieser Detector aggregiert vier orthogonale Stress-Signale, um Regime-Wechsel
früher und mit weniger False-Positives zu erkennen:

1. **Drawdown-Signal** — Trailing-MDD des Marktes (slow, lag-heavy).
2. **Realized-Vol-Signal** — kurze RV / lange RV (fast, sensitiv).
3. **Cross-Section-Dispersion** — Cross-Asset-Vol-of-Returns (Crisis-Signal:
   in Krisen explodiert die Dispersion).
4. **News-Anomaly-Plug** — optionaler Input aus News-Sentiment-Volume oder
   anderen externen Signalen.

Jedes Signal wird auf [0, 1] normalisiert (Trailing-Percentile),
mit Caller-konfigurierbaren Gewichten zu einem Composite-Score gemittelt.
Schwelle > 0.6 → Stress-Regime.

Referenzen
----------
- Cross-Section-Vol als Crisis-Indicator: Connor & Korajczyk (1993).
- Realized-Vol-Ratio als Stress-Signal: Standard in Vol-Targeting-Literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MultiSignalConfig:
    drawdown_window: int = 60
    rv_short_window: int = 5
    rv_long_window: int = 60
    dispersion_window: int = 21
    percentile_window: int = 252
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "drawdown": 0.30,
            "realized_vol": 0.30,
            "dispersion": 0.30,
            "news_anomaly": 0.10,
        }
    )
    stress_threshold: float = 0.60
    smoothing_days: int = 3


def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile-rank in [0, 1]."""

    def _last_rank(x: np.ndarray) -> float:
        if len(x) < 2:
            return np.nan
        v = x[-1]
        return float((x <= v).sum() / len(x))

    return series.rolling(window, min_periods=10).apply(_last_rank, raw=True)


def drawdown_signal(market_returns: pd.Series, window: int = 60) -> pd.Series:
    """Trailing-Drawdown als [0, 1]-Signal."""
    eq = (1 + market_returns.fillna(0)).cumprod()
    roll_max = eq.rolling(window, min_periods=1).max()
    dd_abs = (1 - eq / roll_max).abs()
    return dd_abs.clip(0, 0.50) / 0.50  # 0..1 mit Cap bei 50 % DD


def realized_vol_signal(
    market_returns: pd.Series, short: int = 5, long: int = 60
) -> pd.Series:
    """Kurz-/Lang-RV-Ratio als Stress-Signal."""
    rv_s = market_returns.rolling(short, min_periods=2).std()
    rv_l = market_returns.rolling(long, min_periods=10).std()
    ratio = (rv_s / rv_l.replace(0, np.nan)).clip(0, 5.0)
    # Ratio 1 = normal; > 2 = sehr stressig
    return ((ratio - 0.5) / 1.5).clip(0, 1)


def dispersion_signal(panel_returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """Cross-Section-Standardabweichung der Returns — geglättet → [0, 1].

    Args:
        panel_returns: DataFrame Date × Symbol mit Tages-Returns.
    """
    if panel_returns.shape[1] < 5:
        return pd.Series(np.nan, index=panel_returns.index)
    daily_disp = panel_returns.std(axis=1)
    smoothed = daily_disp.rolling(window, min_periods=5).mean()
    # auf rolling-percentile normalisieren
    return _percentile_rank(smoothed, 252).fillna(0)


def news_anomaly_signal(
    sentiment_panel: pd.DataFrame | None,
    expected_baseline_count: float = 5.0,
) -> pd.Series | None:
    """News-Volume-Anomalie-Signal aus optionalem Sentiment-Panel.

    Args:
        sentiment_panel: DataFrame mit timestamp-Index und 'count'- oder
            'sentiment_volume'-Spalte. None → Signal wird übersprungen.
        expected_baseline_count: Normalisierungs-Faktor.

    Returns:
        Series in [0, 1] oder None wenn nicht verfügbar.
    """
    if sentiment_panel is None or sentiment_panel.empty:
        return None
    vol_col = (
        "sentiment_volume" if "sentiment_volume" in sentiment_panel.columns else "count"
    )
    if vol_col not in sentiment_panel.columns:
        return None
    daily = sentiment_panel.groupby(sentiment_panel["timestamp"].dt.normalize())[
        vol_col
    ].sum()
    daily.index = pd.DatetimeIndex(daily.index, tz="UTC")
    excess = (daily / expected_baseline_count).clip(0, 5.0)
    return ((excess - 0.5) / 1.5).clip(0, 1)


def composite_stress_score(
    market_returns: pd.Series,
    panel_returns: pd.DataFrame,
    sentiment_panel: pd.DataFrame | None = None,
    config: MultiSignalConfig | None = None,
) -> pd.DataFrame:
    """Aggregiere Signale → Composite-Score + Regime-Label.

    Returns:
        DataFrame [drawdown, realized_vol, dispersion, news_anomaly,
        composite_score, regime].
    """
    cfg = config or MultiSignalConfig()
    out = pd.DataFrame(index=market_returns.index)
    out["drawdown"] = drawdown_signal(market_returns, cfg.drawdown_window)
    out["realized_vol"] = realized_vol_signal(
        market_returns, cfg.rv_short_window, cfg.rv_long_window
    )
    out["dispersion"] = dispersion_signal(panel_returns, cfg.dispersion_window).reindex(
        out.index
    )
    news = news_anomaly_signal(sentiment_panel)
    out["news_anomaly"] = news.reindex(out.index) if news is not None else np.nan

    # Renormalisiere Gewichte über nur-vorhandene Signale pro Zeile
    w = cfg.weights.copy()
    cols = ["drawdown", "realized_vol", "dispersion", "news_anomaly"]
    valid_mask = out[cols].notna()
    weight_arr = np.array([w.get(c, 0.0) for c in cols])
    weights_per_row = valid_mask.values * weight_arr[None, :]
    sum_weights = weights_per_row.sum(axis=1)
    sum_weights = np.where(sum_weights == 0, 1.0, sum_weights)
    weighted_vals = (out[cols].fillna(0).values * weights_per_row).sum(
        axis=1
    ) / sum_weights
    out["composite_score"] = weighted_vals

    raw_regime = np.where(
        out["composite_score"] >= cfg.stress_threshold, "stress", "calm"
    )
    out["regime"] = raw_regime

    # Smoothing
    if cfg.smoothing_days > 1:
        last = out["regime"].iloc[0]
        run = 1
        cleaned = [last]
        for t in range(1, len(out)):
            cur = out["regime"].iloc[t]
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
        out["regime"] = cleaned
    return out


__all__ = [
    "MultiSignalConfig",
    "drawdown_signal",
    "realized_vol_signal",
    "dispersion_signal",
    "news_anomaly_signal",
    "composite_stress_score",
]
