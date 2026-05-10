"""Triple-Barrier Labeling + Meta-Labeling (Lopez de Prado 2018).

Theorie
-------
Klassisches "fester Horizont"-Labeling (z. B. ``y = sign(r_{t+5})``) ist
problematisch:
- ignoriert Stop-Loss / Take-Profit-Realität
- zerstört Stationarität
- behandelt späte und schnelle Bewegungen gleich

Triple-Barrier
--------------
Für jeden Sample t:
- Take-Profit-Barrier (oben): pt_t = price_t × (1 + tp_pct)
- Stop-Loss-Barrier   (unten): sl_t = price_t × (1 - sl_pct)
- Time-Barrier (rechts):       t + horizon

Welche Barrier zuerst getroffen wird, definiert das Label:
+1 (TP), -1 (SL), 0 (Time-Out, wenn config so).

Meta-Labeling
-------------
Zwei-Stage-Modell:
1. Primary-Model: gibt Side (long/short) vor.
2. Meta-Model: lernt, ob Side richtig sein WIRD (binary: take trade yes/no).

Vorteil: Trennt Direction-Prediction von Risk-Prediction. Stabilere
Performance laut Lopez de Prado.

Sample Weights
--------------
Wegen Overlap im Triple-Barrier (Label_t und Label_{t+1} können den gleichen
Pfad teilen) sind klassische CV-Splits invalide. Sample-Weights nach
"average uniqueness" sind nötig (s. ``sample_uniqueness``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TripleBarrierConfig:
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.02
    horizon_days: int = 10
    timeout_label: int = 0  # 0 = neutral; alternativ +1 oder -1


def triple_barrier_labels(
    prices: pd.Series,
    config: Optional[TripleBarrierConfig] = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Berechne Triple-Barrier-Labels.

    Args:
        prices: Series indexed by date.
        config: TripleBarrierConfig.
        side: Optional Side-Vorhersagen (long=+1, short=-1) je date. Wenn
            gegeben, werden TP/SL relativ zur Side definiert (für long: TP oben,
            SL unten; für short: vice versa). Ohne side: simulate long-only.

    Returns:
        DataFrame [date, t1 (Touch-Date), label, return_at_t1].
    """
    config = config or TripleBarrierConfig()
    if prices.empty:
        return pd.DataFrame()

    rows = []
    horizon = pd.Timedelta(days=config.horizon_days)
    for t0, p0 in prices.items():
        if not np.isfinite(p0):
            continue
        s = 1 if side is None else int(side.get(t0, 1))
        if s == 0:
            continue
        tp_level = p0 * (1 + s * config.take_profit_pct)
        sl_level = p0 * (1 - s * config.stop_loss_pct)
        future = prices[(prices.index > t0) & (prices.index <= t0 + horizon)]
        if future.empty:
            continue
        # find first touch
        if s == 1:
            tp_hit = future[future >= tp_level]
            sl_hit = future[future <= sl_level]
        else:
            tp_hit = future[future <= tp_level]
            sl_hit = future[future >= sl_level]
        tp_t = tp_hit.index[0] if not tp_hit.empty else None
        sl_t = sl_hit.index[0] if not sl_hit.empty else None
        if tp_t is None and sl_t is None:
            t1 = future.index[-1]
            label = config.timeout_label
        elif tp_t is None:
            t1 = sl_t
            label = -1
        elif sl_t is None:
            t1 = tp_t
            label = +1
        else:
            t1 = min(tp_t, sl_t)
            label = +1 if t1 == tp_t else -1
        ret = float(prices.loc[t1] / p0 - 1)
        rows.append({"date": t0, "t1": t1, "label": label, "return_at_t1": ret})
    return pd.DataFrame(rows)


def sample_uniqueness(
    barriers_df: pd.DataFrame,
) -> pd.Series:
    """Lopez de Prado Average-Uniqueness pro Sample.

    Idee: Wenn label_t über [t, t1] berechnet wird, "verwendet" es jeden Tag in
    diesem Intervall. Tage, die von vielen Labels verwendet werden, machen
    überlappende Samples redundant. Uniqueness_i = 1 / (avg # concurrent labels).

    Returns:
        Series Index = barriers_df['date'], Values = uniqueness ∈ (0, 1].
    """
    if barriers_df.empty:
        return pd.Series(dtype=float)
    df = barriers_df.dropna(subset=["t1"]).copy()
    df["t1"] = pd.to_datetime(df["t1"])
    df["t0"] = pd.to_datetime(df["date"])
    # build concurrent count series — sample is active on [t0, t1] inclusive,
    # so end events are placed at t1 + 1 day (after-close).
    starts = pd.Series(1, index=df["t0"])
    ends = pd.Series(-1, index=df["t1"] + pd.Timedelta(days=1))
    events = pd.concat([starts, ends]).groupby(level=0).sum().sort_index()
    cumulative = events.cumsum()
    # daily concurrent count
    full_idx = pd.date_range(df["t0"].min(), df["t1"].max(), freq="D")
    daily = cumulative.reindex(full_idx, method="ffill").fillna(0)

    # for each sample, average concurrent count over [t0, t1]
    out = []
    for _, r in df.iterrows():
        t0, t1 = r["t0"], r["t1"]
        sub = daily.loc[t0:t1]
        avg = float(sub.mean()) if not sub.empty else 1.0
        out.append(1.0 / avg if avg > 0 else 1.0)
    return pd.Series(out, index=df["t0"].values)


def make_meta_labels(
    side_predictions: pd.Series,
    triple_barrier_labels_: pd.Series,
) -> pd.Series:
    """Meta-Label: 1 wenn primary-side die richtige Richtung vorhersagt.

    Args:
        side_predictions: Series mit primary-model side (+1 / -1) per date.
        triple_barrier_labels_: Series mit Triple-Barrier labels per date.

    Returns:
        Series mit binary labels (1 = take trade, 0 = skip).
    """
    aligned = pd.concat(
        [side_predictions.rename("side"), triple_barrier_labels_.rename("tb_label")],
        axis=1,
    ).dropna()
    return (aligned["side"] == aligned["tb_label"]).astype(int)


__all__ = [
    "TripleBarrierConfig",
    "triple_barrier_labels",
    "sample_uniqueness",
    "make_meta_labels",
]
