"""Fractional Differentiation (Lopez de Prado 2018).

Theorie
-------
Klassisches D=1 (first-difference) entfernt zu viel Information — die
resultierende Series hat nur sehr kurzes Memory.

Lopez de Prado: D ∈ (0, 1) — fractional differentiation entfernt non-stationarity
während ein Großteil des "memory" erhalten bleibt:

    Δ^d X_t = Σ_{k=0}^{∞} ω_k · X_{t-k}, mit ω_k = (-1)^k Γ(d+1) / (Γ(k+1) Γ(d-k+1))

Vorteil
-------
- ADF-Test passes mit kleinerem D (z. B. D=0.4) statt D=1.
- Mehr signal in der differenzierten Series.

Implementation
--------------
"Fixed-window" version (Lopez de Prado 2018 §5.5) — finite weights, fast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Fixed-Width Fractional Differentiation Weights.

    Stops adding weights when |w_k| < threshold.

    Args:
        d: fractional differentiation parameter ∈ (0, 1).
        threshold: cut-off for small weights.

    Returns:
        Array of weights w_0, w_1, w_2, ... (shortest first).
    """
    weights = [1.0]
    k = 1
    while True:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
        k += 1
    return np.array(weights)


def fractional_diff_ffd(
    series: pd.Series, d: float, threshold: float = 1e-5
) -> pd.Series:
    """Apply fixed-window fractional differentiation.

    Args:
        series: pandas Series indexed by date.
        d: fractional order.
        threshold: weight cutoff.

    Returns:
        Differentiated series (initial NaN tail).
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)
    s = pd.Series(series).dropna()
    out = pd.Series(np.nan, index=s.index)
    for i in range(width - 1, len(s)):
        window = s.iloc[i - width + 1 : i + 1].values
        out.iloc[i] = float(weights[::-1] @ window)
    return out


def find_min_d_for_stationarity(
    series: pd.Series, d_range: tuple[float, float] = (0.0, 1.0), step: float = 0.05
) -> dict:
    """Suche das kleinste d, bei dem die differenzierte Series stationär ist (ADF-p < 0.05).

    Args:
        series: input series.
        d_range, step: search grid.

    Returns:
        Dict with best_d, adf_pvalue, n_samples_lost.
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except ImportError:
        return {"error": "statsmodels required"}

    out_rows = []
    for d in np.arange(d_range[0], d_range[1] + 1e-9, step):
        try:
            diff = fractional_diff_ffd(series, d).dropna()
            if len(diff) < 30:
                continue
            adf = adfuller(diff.values, regression="c", autolag="AIC")
            out_rows.append({"d": d, "adf_p": adf[1], "n_obs": len(diff)})
        except Exception:  # noqa: BLE001
            continue
    if not out_rows:
        return {"error": "no successful fit"}
    df = pd.DataFrame(out_rows)
    stationary = df[df["adf_p"] < 0.05]
    if stationary.empty:
        # fallback: smallest p-value
        idx = df["adf_p"].idxmin()
        best = df.loc[idx]
    else:
        idx = stationary["d"].idxmin()
        best = stationary.loc[idx]
    return {
        "best_d": float(best["d"]),
        "adf_pvalue": float(best["adf_p"]),
        "n_obs_kept": int(best["n_obs"]),
        "search_grid": df.to_dict("records"),
    }


__all__ = ["get_weights_ffd", "fractional_diff_ffd", "find_min_d_for_stationarity"]
