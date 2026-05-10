"""Options-implied Faktoren als Cross-Section-Signal.

Theorie
-------
Options-Daten enthalten **forward-looking** Information, die in Spot-Preisen
nicht direkt sichtbar ist:

1. **IV-Skew** (Put-Skew): Hohes Skew = teurer OTM-Put = Crash-Furcht.
   Akademisch: Cremers/Weinbaum (2010, *JFE*) "Deviations from put-call parity".

2. **IV-Term-Structure**: Inversion (front > back) = akute Stress-Erwartung.

3. **Put/Call-Ratio**: Sentimentindikator. Extreme PCRs sind kontrarisch.

4. **Realized vs Implied Vol Spread**: rv > iv = Markt unterschätzt Risiko.

5. **Variance Risk Premium** (VRP) = E[RV] − IV². Stabile akademische Renditequelle.

Implementierung
---------------
Output-Series sind alle so designed, dass größere Werte = höhere Wahrscheinlichkeit
positiver Forward-Returns (für Long-Signale) bzw. negativ vorzeichenkohärent.

PIT
---
Yahoo-Snapshots sind real-time. Für Backtests muss historisches IV/Skew
gesondert beschafft werden (siehe ``PAID_DATA_WISHLIST.md``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def vrp_signal(
    iv_history: pd.DataFrame, realized_vol: pd.DataFrame, lookback: int = 30
) -> pd.DataFrame:
    """Variance Risk Premium Signal.

    Args:
        iv_history: DataFrame [date, symbol, iv_30d_atm] (annualisiert).
        realized_vol: DataFrame [date, symbol, rv_30d] (annualisiert).
        lookback: Rolling Z-Score-Fenster.

    Returns:
        DataFrame mit ``vrp`` und ``vrp_z`` (vrp_t = iv_t² − rv_t²).

    Interpretation
    --------------
    VRP > 0 (typisch): Versicherer verlangen Prämie für Vola-Verkauf —
    historisch positive Volatility-Risk-Premium-Strategie.
    Hoher VRP-Z = Vola-Mean-Reversion-Signal.
    """
    if iv_history.empty or realized_vol.empty:
        return pd.DataFrame()

    df = iv_history.merge(realized_vol, on=["date", "symbol"], how="inner")
    df["vrp"] = df["iv_30d_atm"] ** 2 - df["rv_30d"] ** 2

    df = df.sort_values(["symbol", "date"])
    grp = df.groupby("symbol", group_keys=False)
    df["vrp_pit"] = grp["vrp"].shift(1)
    df["vrp_mean"] = grp["vrp_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback // 2).mean()
    )
    df["vrp_std"] = grp["vrp_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback // 2).std()
    )
    df["vrp_z"] = (df["vrp_pit"] - df["vrp_mean"]) / df["vrp_std"]
    return df


def skew_signal(skew_history: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Put-Skew als Crash-Risk-Indikator.

    Hohes Skew = teurer OTM-Put = Marktteilnehmer fürchten Crash.
    -> negative Korrelation mit Forward-Returns auf 1-3-Monats-Horizont.

    Args:
        skew_history: DataFrame [date, symbol, skew_25d].
        lookback: Z-Score-Fenster.

    Returns:
        DataFrame mit ``skew_z`` (höhere Werte = bärischer Indikator).
    """
    if skew_history.empty:
        return skew_history.assign(skew_z=pd.Series(dtype=float))

    out = skew_history.sort_values(["symbol", "date"]).copy()
    grp = out.groupby("symbol", group_keys=False)
    out["skew_pit"] = grp["skew_25d"].shift(1)
    out["mean"] = grp["skew_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback // 2).mean()
    )
    out["std"] = grp["skew_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback // 2).std()
    )
    out["skew_z"] = (out["skew_pit"] - out["mean"]) / out["std"]
    return out


def realized_vol(
    returns_panel: pd.DataFrame, window: int = 21, annualize: bool = True
) -> pd.DataFrame:
    """Realized Volatility (annualisiert).

    Args:
        returns_panel: DataFrame [date, symbol, return].
        window: Anzahl Tage.
        annualize: ob mit √252 multipliziert.

    Returns:
        DataFrame [date, symbol, rv_X_d].
    """
    if returns_panel.empty:
        return returns_panel
    out = returns_panel.sort_values(["symbol", "date"]).copy()
    grp = out.groupby("symbol", group_keys=False)
    rolling = grp["return"].transform(
        lambda s: s.rolling(window, min_periods=window // 2).std()
    )
    if annualize:
        rolling = rolling * np.sqrt(252)
    out[f"rv_{window}_d"] = rolling
    return out


def garman_klass_volatility(ohlc_panel: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Garman-Klass-Volatility — präziser als Close-to-Close-RV.

    GK ist ~7× effizienter als Close-to-Close bei gleicher Sample-Größe.
    Setzt OHLC voraus.

    Args:
        ohlc_panel: DataFrame [date, symbol, open, high, low, close].
        window: Rolling-Fenster (Tage).

    Returns:
        DataFrame [date, symbol, gk_vol].
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(ohlc_panel.columns):
        raise ValueError(f"missing columns: {required - set(ohlc_panel.columns)}")

    df = ohlc_panel.copy()
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    df["gk_term"] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    out = df.sort_values(["symbol", "date"])
    grp = out.groupby("symbol", group_keys=False)
    out["gk_vol"] = (
        grp["gk_term"]
        .transform(lambda s: s.rolling(window, min_periods=window // 2).mean())
        .clip(lower=0)
    ) ** 0.5 * np.sqrt(252)
    return out[["date", "symbol", "gk_vol"]]


__all__ = [
    "vrp_signal",
    "skew_signal",
    "realized_vol",
    "garman_klass_volatility",
]
