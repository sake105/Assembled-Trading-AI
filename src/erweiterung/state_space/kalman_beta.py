"""Kalman-Filter für Time-Varying-Beta-Estimation.

Theorie
-------
Klassisches OLS-β nimmt **konstantes** β an. In der Realität ändert sich β
über Zeit (Marktstruktur ändert sich, Geschäftsmodell ändert sich, etc.).

Ein State-Space-Modell:
    Beobachtungsgleichung: y_t = β_t · x_t + v_t,  v_t ~ N(0, R)
    Zustandsgleichung:     β_t = β_{t-1} + w_t,    w_t ~ N(0, Q)

Kalman-Filter liefert posterior-mean und posterior-variance von β_t.

Anwendung
---------
- Time-varying Market-Beta in Faktor-Modellen
- Adaptive Hedge-Ratios (Pairs-Trading)
- "Beta-Fade" Detection (β decreasing toward 1)

Reference
---------
- Harvey, A. (1989). *Forecasting, Structural Time Series Models and the
  Kalman Filter*. Cambridge.
- Hamilton, J. (1994). *Time Series Analysis*. Princeton.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KalmanBetaConfig:
    process_variance: float = 1e-4  # Q — wie schnell β driftet
    observation_variance: float = 1e-2  # R — Measurement-Noise
    initial_beta: float = 1.0
    initial_variance: float = 1.0
    include_alpha: bool = True


def kalman_filter_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    config: KalmanBetaConfig | None = None,
) -> pd.DataFrame:
    """Run Kalman-Filter on (asset, market) zur Schätzung time-varying β.

    Args:
        asset_returns, market_returns: aligned pd.Series.
        config: KalmanBetaConfig.

    Returns:
        DataFrame [date, beta, alpha, beta_var, alpha_var, residual].
    """
    cfg = config or KalmanBetaConfig()
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    df.columns = ["a", "m"]
    n = len(df)
    if n < 10:
        raise ValueError("need >= 10 obs")

    # State: [alpha, beta] if include_alpha else [beta]
    if cfg.include_alpha:
        state = np.array([0.0, cfg.initial_beta])
        P = np.eye(2) * cfg.initial_variance
        Q = np.eye(2) * cfg.process_variance
    else:
        state = np.array([cfg.initial_beta])
        P = np.array([[cfg.initial_variance]])
        Q = np.array([[cfg.process_variance]])

    R = np.array([[cfg.observation_variance]])

    out_rows = []
    for t in range(n):
        m_t = float(df["m"].iloc[t])
        y_t = float(df["a"].iloc[t])

        # Predict (state transition is identity)
        state_pred = state.copy()
        P_pred = P + Q

        # Observation matrix H
        if cfg.include_alpha:
            H = np.array([[1.0, m_t]])
        else:
            H = np.array([[m_t]])

        # Innovation
        y_pred = float((H @ state_pred).item())
        innov = y_t - y_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + R

        # Kalman gain
        K = (P_pred @ H.T) / float(S.item())

        # Update
        state = state_pred + (K * innov).flatten()
        P = (np.eye(P.shape[0]) - K @ H) @ P_pred

        if cfg.include_alpha:
            out_rows.append(
                {
                    "date": df.index[t],
                    "alpha": float(state[0]),
                    "beta": float(state[1]),
                    "alpha_var": float(P[0, 0]),
                    "beta_var": float(P[1, 1]),
                    "residual": float(innov),
                }
            )
        else:
            out_rows.append(
                {
                    "date": df.index[t],
                    "alpha": 0.0,
                    "beta": float(state[0]),
                    "alpha_var": 0.0,
                    "beta_var": float(P[0, 0]),
                    "residual": float(innov),
                }
            )

    return pd.DataFrame(out_rows).set_index("date")


def kalman_pairs_hedge_ratio(
    y: pd.Series, x: pd.Series, config: KalmanBetaConfig | None = None
) -> pd.DataFrame:
    """Time-varying hedge ratio for pairs trading via Kalman.

    Returns:
        DataFrame [date, hedge_ratio, spread, spread_var].
    """
    cfg = config or KalmanBetaConfig(include_alpha=False)
    res = kalman_filter_beta(y, x, cfg)
    res["hedge_ratio"] = res["beta"]
    res["spread"] = res["residual"]
    res["spread_var"] = res["beta_var"]
    return res[["hedge_ratio", "spread", "spread_var"]]


__all__ = ["KalmanBetaConfig", "kalman_filter_beta", "kalman_pairs_hedge_ratio"]
