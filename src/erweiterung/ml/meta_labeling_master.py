"""Meta-Labeling für Master-Allocator (Lopez de Prado 2018, §3.6).

Idee
----
Master-Allocator liefert eine konstante Allokation. Meta-Labeling fügt
einen "Gate"-Klassifikator hinzu: gehe nur LONG, wenn das Modell glaubt,
der Trade wird profitabel sein. Sonst halte Cash.

Pipeline
--------
1. Primary-Signal: Master-Allocator-Return-Series (immer "long", side = +1).
2. Triple-Barrier-Labeling auf Master-Equity-Curve.
3. Features berechnen: trailing vol, trailing sharpe, drawdown-state,
   regime-indicators, macro-state (VIX, yield-curve).
4. Meta-Klassifikator (RandomForest / Logistic) lernt: profitable = 1, loss = 0.
5. OOS-Test: trade nur an Tagen, an denen Meta-Modell prob > threshold.

Theorie
-------
Lopez de Prado argumentiert: trennen von Direction (Primary) und Magnitude/
Confidence (Meta) liefert stabilere OOS-Performance als ein gemeinsames Modell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetaLabelingConfig:
    take_profit_pct: float = 0.025
    stop_loss_pct: float = 0.025
    horizon_days: int = 21
    feature_vol_window: int = 21
    feature_sharpe_window: int = 60
    feature_drawdown_window: int = 60
    train_window: int = 504  # 2 years
    test_window: int = 126  # 6 months
    min_train_samples: int = 100
    meta_threshold: float = 0.50
    """Probabilitätsschwelle für 'trade go-Signal'."""


def triple_barrier_simple(
    returns: pd.Series, config: MetaLabelingConfig | None = None
) -> pd.Series:
    """Triple-Barrier-Labels auf Cumulative-Returns.

    Args:
        returns: Daily-Returns des Primary-Signals.
        config: MetaLabelingConfig.

    Returns:
        Series mit Labels:
        +1 = Take-Profit traf
        -1 = Stop-Loss traf
         0 = Time-Out (keiner traf bis horizon)
    """
    cfg = config or MetaLabelingConfig()
    eq = (1 + returns.fillna(0)).cumprod()
    n = len(eq)
    labels = pd.Series(np.nan, index=returns.index)
    for i in range(n):
        end = min(i + cfg.horizon_days, n)
        if end <= i + 1:
            continue
        path = eq.iloc[i:end] / eq.iloc[i] - 1.0
        tp_hit = (path >= cfg.take_profit_pct).any()
        sl_hit = (path <= -cfg.stop_loss_pct).any()
        if tp_hit and not sl_hit:
            labels.iloc[i] = 1.0
        elif sl_hit and not tp_hit:
            labels.iloc[i] = -1.0
        elif tp_hit and sl_hit:
            # Beide getroffen: gewinner = erste
            tp_idx = path[path >= cfg.take_profit_pct].index[0]
            sl_idx = path[path <= -cfg.stop_loss_pct].index[0]
            labels.iloc[i] = 1.0 if tp_idx <= sl_idx else -1.0
        else:
            labels.iloc[i] = 0.0
    return labels


def build_features(
    returns: pd.Series,
    macro_panel: pd.DataFrame | None = None,
    config: MetaLabelingConfig | None = None,
) -> pd.DataFrame:
    """Baue Features für Meta-Modell.

    Args:
        returns: Daily-Returns des Primary-Signals.
        macro_panel: optionaler DataFrame mit VIX, yield-curve-spread etc.
            Spalten ``vix_close``, ``yield_curve_spread``, ``hy_spread`` werden genutzt.
        config: MetaLabelingConfig.

    Returns:
        DataFrame mit Features (alle t-1 gelagt).
    """
    cfg = config or MetaLabelingConfig()
    feat = pd.DataFrame(index=returns.index)
    feat["trailing_vol"] = returns.rolling(
        cfg.feature_vol_window, min_periods=5
    ).std() * np.sqrt(252)
    feat["trailing_sharpe"] = (
        returns.rolling(cfg.feature_sharpe_window, min_periods=10).mean()
        * 252
        / (
            returns.rolling(cfg.feature_sharpe_window, min_periods=10).std()
            * np.sqrt(252)
        ).replace(0, np.nan)
    )
    eq = (1 + returns.fillna(0)).cumprod()
    rolling_max = eq.rolling(cfg.feature_drawdown_window, min_periods=1).max()
    feat["drawdown_pct"] = (eq / rolling_max - 1.0).abs()
    feat["return_lag1"] = returns.shift(1)
    feat["return_5d"] = returns.rolling(5, min_periods=1).sum()
    feat["return_21d"] = returns.rolling(21, min_periods=5).sum()
    feat["vol_of_vol"] = feat["trailing_vol"].rolling(21, min_periods=5).std()

    if macro_panel is not None and not macro_panel.empty:
        m_aligned = macro_panel.reindex(returns.index, method="ffill")
        if "vix_close" in m_aligned.columns:
            feat["vix"] = m_aligned["vix_close"]
            feat["vix_zscore"] = (
                feat["vix"] - feat["vix"].rolling(252, min_periods=20).mean()
            ) / feat["vix"].rolling(252, min_periods=20).std().replace(0, np.nan)
        if "yield_curve_spread" in m_aligned.columns:
            feat["yc_spread"] = m_aligned["yield_curve_spread"]
            feat["yc_inverted"] = (m_aligned["yield_curve_spread"] < 0).astype(float)
        if "hy_spread" in m_aligned.columns:
            feat["hy_spread"] = m_aligned["hy_spread"]

    return feat.shift(1)  # t-1 lag, kein lookahead


def walk_forward_meta_predictions(
    features: pd.DataFrame,
    labels: pd.Series,
    config: MetaLabelingConfig | None = None,
    model_type: str = "logistic",
) -> pd.DataFrame:
    """Walk-Forward Out-of-Sample Meta-Modell-Vorhersagen.

    Args:
        features: DataFrame (Date × Feature).
        labels: Series mit Triple-Barrier-Labels in {-1, 0, +1}.
        config: MetaLabelingConfig.
        model_type: 'logistic' oder 'rf'.

    Returns:
        DataFrame mit Spalten [proba, predicted, actual_label].
        proba = Wahrscheinlichkeit für Klasse "profitable" (label=1).
    """
    cfg = config or MetaLabelingConfig()
    # Binär-Labels: 1 wenn TP traf, 0 sonst
    binary_labels = (labels == 1.0).astype(int)
    aligned = pd.concat([features, binary_labels.rename("y")], axis=1).dropna()
    if len(aligned) < cfg.train_window + cfg.test_window:
        return pd.DataFrame()

    out_rows = []
    start = 0
    while start + cfg.train_window + cfg.test_window <= len(aligned):
        train = aligned.iloc[start : start + cfg.train_window]
        test = aligned.iloc[
            start + cfg.train_window : start + cfg.train_window + cfg.test_window
        ]

        X_train = train.drop(columns=["y"])
        y_train = train["y"]
        X_test = test.drop(columns=["y"])
        y_test = test["y"]

        if y_train.sum() < 10 or (len(y_train) - y_train.sum()) < 10:
            # Imbalance zu extrem
            start += cfg.test_window
            continue

        if model_type == "rf":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42
            )
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                ]
            )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= cfg.meta_threshold).astype(int)
        for i, idx in enumerate(test.index):
            out_rows.append(
                {
                    "date": idx,
                    "proba": float(proba[i]),
                    "predicted": int(pred[i]),
                    "actual_label": int(y_test.iloc[i]),
                }
            )
        start += cfg.test_window

    return pd.DataFrame(out_rows).set_index("date")


def apply_meta_gate(
    primary_returns: pd.Series, meta_predictions: pd.DataFrame
) -> pd.Series:
    """Wende Meta-Gate auf Primary-Returns an: trade nur bei predicted=1."""
    aligned = pd.concat(
        {"r": primary_returns, "p": meta_predictions["predicted"]}, axis=1
    ).dropna()
    gated = aligned["r"] * aligned["p"]
    return gated


__all__ = [
    "MetaLabelingConfig",
    "triple_barrier_simple",
    "build_features",
    "walk_forward_meta_predictions",
    "apply_meta_gate",
]
