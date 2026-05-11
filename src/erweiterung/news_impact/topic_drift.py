"""Topic-Drift Detection in News-Flow.

Theorie
-------
News-Topic-Mix ändert sich im Zeitverlauf:
- Vor 2020: M&A, Earnings, Product-Launches dominant.
- COVID: Health, Stimulus, Lockdown.
- 2022: Inflation, Energy, Geopolitik.

**Topic-Drift** = Change in topic-distribution = Regime-Indicator.

Methodik
--------
1. **Jensen-Shannon-Divergenz** zwischen aktueller und historischer Topic-Verteilung.
2. **Spike** in JS-Divergenz = neuer Themen-Cluster auftauchend.
3. **Topic-Persistence** (autocorrelation der Topic-Distributionen): hoch = stabiles
   Regime, niedrig = turbulent.

Anwendung
---------
- Crisis-Detection: Topic-Drift-Spike + Sentiment-Drop = early warning.
- Strategy-Switching: bei Topic-Regime-Change → re-train ML models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """JS-Divergence between two discrete distributions.

    Symmetric, bounded ∈ [0, ln(2)].
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("shape mismatch")
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    eps = 1e-12

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * (kl(p, m) + kl(q, m))


def topic_distribution_per_window(
    topic_panel: pd.DataFrame,
    window: int = 30,
    date_col: str = "date",
) -> pd.DataFrame:
    """Aggregate topic-counts per rolling window.

    Args:
        topic_panel: DataFrame with [date, topic_id] OR with topic-distribution-Columns
            ``topic_0``, ``topic_1``, ...
        window: rolling-window in days.

    Returns:
        DataFrame indexed by window-end-date, columns = topic-Ids, values = proportions.
    """
    df = topic_panel.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values(date_col)

    # Detect format
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if topic_cols:
        # Already topic-distribution per article — aggregate via mean
        df_indexed = df.set_index(date_col)[topic_cols]
        rolling_mean = df_indexed.rolling(f"{window}D", min_periods=10).mean()
        return rolling_mean
    if "topic_id" in df.columns:
        # Count topics per window
        df_indexed = df.set_index(date_col)
        # OneHot
        dummies = pd.get_dummies(df_indexed["topic_id"], prefix="topic")
        rolling_count = dummies.rolling(f"{window}D", min_periods=10).sum()
        # Normalize
        row_sums = rolling_count.sum(axis=1).replace(0, np.nan)
        return rolling_count.div(row_sums, axis=0).fillna(0)
    raise ValueError("topic_panel must have 'topic_id' or 'topic_*' columns")


def topic_drift_signal(
    topic_distributions: pd.DataFrame, baseline_window: int = 90
) -> pd.Series:
    """Compute JS-divergence between current and trailing-baseline topic-distribution.

    Args:
        topic_distributions: DataFrame from topic_distribution_per_window.
        baseline_window: how many days back to compare with.

    Returns:
        Series of JS-divergence values (higher = bigger topic-drift).
    """
    if topic_distributions.empty:
        return pd.Series(dtype=float)
    out = pd.Series(np.nan, index=topic_distributions.index)
    for i, t in enumerate(topic_distributions.index):
        if i < baseline_window // 2:
            continue
        current = topic_distributions.iloc[i].values
        # Baseline: mean over previous baseline_window observations
        start = max(0, i - baseline_window)
        baseline = topic_distributions.iloc[start:i].mean(axis=0).values
        if baseline.sum() <= 0:
            continue
        out.iloc[i] = jensen_shannon_divergence(current, baseline)
    return out


def topic_persistence(topic_distributions: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Topic-Distribution-Autokorrelation (lag-1 ähnlichkeit).

    Hoch = stabiles Regime, niedrig = turbulent.

    Returns:
        Series of 1 − JS-divergence(t, t-lag).
    """
    out = pd.Series(np.nan, index=topic_distributions.index)
    for i in range(lag, len(topic_distributions)):
        a = topic_distributions.iloc[i].values
        b = topic_distributions.iloc[i - lag].values
        if a.sum() <= 0 or b.sum() <= 0:
            continue
        js = jensen_shannon_divergence(a, b)
        out.iloc[i] = 1.0 - js / np.log(2)  # normalize
    return out


def detect_topic_change_points(
    drift_signal: pd.Series, threshold_quantile: float = 0.95
) -> list[pd.Timestamp]:
    """Identify dates with topic-drift exceeding rolling-quantile threshold."""
    s = pd.Series(drift_signal).dropna()
    if s.empty:
        return []
    rolling_thresh = s.rolling(60, min_periods=20).quantile(threshold_quantile)
    spikes = s[s > rolling_thresh.shift(1).fillna(0)]
    return list(spikes.index)


__all__ = [
    "jensen_shannon_divergence",
    "topic_distribution_per_window",
    "topic_drift_signal",
    "topic_persistence",
    "detect_topic_change_points",
]
