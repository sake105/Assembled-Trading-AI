"""Lead-Lag-Network — Granger-Causality-basierte Signalweitergabe.

Theorie
-------
Manche Aktien führen / lagen anderen systematisch:
- Größere Firmen lagen kleineren (Lo/MacKinlay 1990, *RFS*).
- Liquider Sektor-Leader führt Sektor-Mitglieder.
- Cross-Asset: Halbleiter-Index (SOX) führt einzelnen Halbleiter-Aktien um Tage.

Wir bauen pro Periode ein Lead-Lag-Netzwerk und propagieren Signale entlang
dieser Kanten.

Methodik
--------
1. Granger-Causality-Test mit Lag-1 zwischen Symbolpaaren.
2. Top-K-Edges je Symbol (k Lead-Kandidaten).
3. Signal: ``r_{lead, t}`` als Vorhersage für ``r_{follower, t+1}``.

Achtung: Granger-F-Statistik benötigt mindestens ~120 Beobachtungen für
stabile Schätzungen. Multi-Test-Korrektur (Bonferroni) wird empfohlen.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def granger_causality_lag1(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Approximative Granger-Causality x → y mit Lag 1.

    Returns:
        (F_statistic, p_value). p_value ist nur grob (basiert auf F(1, n-3)).
    """
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 30:
        return (np.nan, np.nan)
    s.columns = ["x", "y"]
    s["x_lag"] = s["x"].shift(1)
    s["y_lag"] = s["y"].shift(1)
    s = s.dropna()
    n = len(s)
    if n < 30:
        return (np.nan, np.nan)
    # Restricted: y ~ y_lag
    yv = s["y"].values
    Xr = np.column_stack([np.ones(n), s["y_lag"].values])
    br, _resr, _, _ = np.linalg.lstsq(Xr, yv, rcond=None)
    rss_r = float(((yv - Xr @ br) ** 2).sum())
    # Unrestricted: y ~ y_lag + x_lag
    Xu = np.column_stack([np.ones(n), s["y_lag"].values, s["x_lag"].values])
    bu, _resu, _, _ = np.linalg.lstsq(Xu, yv, rcond=None)
    rss_u = float(((yv - Xu @ bu) ** 2).sum())
    # F = (RSS_r - RSS_u) / 1 / (RSS_u / (n - 3))
    if rss_u <= 0:
        return (np.nan, np.nan)
    F = (rss_r - rss_u) / (rss_u / (n - 3))
    if F < 0:
        F = 0.0
    # Approx p-value via F-distribution (lazy, aber konservativ ohne scipy)
    # Wir geben (F, NaN) zurück, p_value aus scipy.stats falls verfügbar.
    try:
        from scipy.stats import f as f_dist  # type: ignore

        p = 1.0 - f_dist.cdf(F, 1, n - 3)
        return (float(F), float(p))
    except ImportError:
        return (float(F), np.nan)


def build_leadlag_network(
    returns_panel: pd.DataFrame,
    window: int = 252,
    end_date: pd.Timestamp | None = None,
    f_threshold: float = 5.0,
    max_pairs: int = 50,
    min_volume_filter: int = 30,
) -> pd.DataFrame:
    """Baue ein Lead-Lag-Netzwerk basierend auf Granger-Causality.

    Args:
        returns_panel: DataFrame [date, symbol, return].
        window: Rolling-Fenster.
        end_date: Endzeitpunkt für die Schätzung (PIT-sicher).
        f_threshold: Mindest-F-Statistik für eine Kante.
        max_pairs: Maximale Kanten pro Lead-Symbol.
        min_volume_filter: Mindestens N gültige Beobachtungen pro Pair.

    Returns:
        DataFrame [lead, follower, f_stat, p_value, n_obs].
    """
    if returns_panel.empty:
        return pd.DataFrame()
    end = end_date if end_date is not None else returns_panel["date"].max()
    start = end - pd.Timedelta(days=window * 2)
    sub = returns_panel[
        (returns_panel["date"] >= start) & (returns_panel["date"] <= end)
    ]
    pivot = sub.pivot_table(index="date", columns="symbol", values="return")
    syms = list(pivot.columns)

    rows = []
    for i, lead in enumerate(syms):
        x = pivot[lead].tail(window)
        if x.notna().sum() < min_volume_filter:
            continue
        for j, follower in enumerate(syms):
            if i == j:
                continue
            y = pivot[follower].tail(window)
            if y.notna().sum() < min_volume_filter:
                continue
            F, p = granger_causality_lag1(x, y)
            if not np.isfinite(F) or F < f_threshold:
                continue
            rows.append(
                {
                    "lead": lead,
                    "follower": follower,
                    "f_stat": F,
                    "p_value": p,
                    "n_obs": min(x.notna().sum(), y.notna().sum()),
                }
            )
    if not rows:
        return pd.DataFrame()
    edges = pd.DataFrame(rows)
    # Top-K per Lead
    edges = (
        edges.sort_values(["lead", "f_stat"], ascending=[True, False])
        .groupby("lead")
        .head(max_pairs)
    )
    return edges.reset_index(drop=True)


def propagate_lead_signal(
    returns_panel: pd.DataFrame,
    network: pd.DataFrame,
    decay: float = 0.5,
) -> pd.DataFrame:
    """Propagiere Lead-Returns zu Followern (PIT-sicher: ``r_{lead, t-1}`` -> ``signal_{follower, t}``).

    Args:
        returns_panel: DataFrame [date, symbol, return].
        network: Output von ``build_leadlag_network``.
        decay: Gewichtung des Signals via F-Statistik decay.

    Returns:
        DataFrame [date, symbol, lead_signal] — averaged over all leads.
    """
    if returns_panel.empty or network.empty:
        return pd.DataFrame()
    pivot = returns_panel.pivot_table(index="date", columns="symbol", values="return")

    out_rows = []
    for follower, edges in network.groupby("follower"):
        leads = edges["lead"].tolist()
        weights = edges["f_stat"].values ** decay
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        if not leads:
            continue
        leads_in = [c for c in leads if c in pivot.columns]
        if not leads_in:
            continue
        weighted = (pivot[leads_in].shift(1) * weights[: len(leads_in)]).sum(axis=1)
        for d, val in weighted.items():
            if pd.notna(val):
                out_rows.append({"date": d, "symbol": follower, "lead_signal": val})
    return pd.DataFrame(out_rows)


__all__ = [
    "granger_causality_lag1",
    "build_leadlag_network",
    "propagate_lead_signal",
]
