"""LASSO/Elastic-Net Faktor-Selection (Tibshirani 1996).

Theorie
-------
LASSO: min ||y - Xβ||² + λ ||β||_1.
- L1-Penalty erzwingt **sparse** β: viele Koeffizienten exakt 0.
- Automatische Faktor-Selection.

Elastic-Net: min ||y - Xβ||² + λ_1 ||β||_1 + λ_2 ||β||_2².
- Mixt L1 + L2.
- Stabiler bei korrelierten Features.

Anwendung
---------
- Faktor-Modell-Auswahl: aus 50 Kandidat-Signalen → top-5 sparsam ausgewählt.
- Regularisierte Faktor-Loadings für Time-Varying-Beta-Modelle.

Implementation
--------------
Wir verwenden sklearn falls vorhanden — Lasso/ElasticNet sind tabellarisch
Standard. Fallback: Coordinate-Descent in NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LassoResult:
    coefficients: np.ndarray
    intercept: float
    selected_features: list[str]
    alpha: float
    r_squared: float


def _lasso_coord_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-7,
) -> np.ndarray:
    """Numpy coordinate-descent LASSO."""
    n, p = X.shape
    beta = np.zeros(p)
    X_col_sq = (X**2).sum(axis=0)
    X_col_sq = np.where(X_col_sq == 0, 1, X_col_sq)
    for _ in range(max_iter):
        beta_prev = beta.copy()
        for j in range(p):
            r_partial = y - X @ beta + X[:, j] * beta[j]
            rho_j = float(X[:, j] @ r_partial)
            # soft-thresholding
            if rho_j > alpha * n / 2:
                beta[j] = (rho_j - alpha * n / 2) / X_col_sq[j]
            elif rho_j < -alpha * n / 2:
                beta[j] = (rho_j + alpha * n / 2) / X_col_sq[j]
            else:
                beta[j] = 0.0
        if np.linalg.norm(beta - beta_prev) < tol:
            break
    return beta


def lasso_factor_selection(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float = 0.01,
    standardize: bool = True,
) -> LassoResult:
    """LASSO-Faktor-Selection.

    Args:
        X: DataFrame mit potenziellen Faktoren (T × K).
        y: response Series.
        alpha: L1-penalty strength.
        standardize: ob features standardisiert werden.

    Returns:
        LassoResult mit gewählten Faktoren.
    """
    df = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    X_v = df[X.columns].values.astype(float)
    y_v = df["__y__"].values.astype(float)
    if len(y_v) < 30:
        raise ValueError("need >= 30 obs")
    feature_names = list(X.columns)

    # Standardize
    if standardize:
        x_mean = X_v.mean(axis=0)
        x_std = X_v.std(axis=0, ddof=0)
        x_std = np.where(x_std == 0, 1, x_std)
        X_v = (X_v - x_mean) / x_std
    y_mean = y_v.mean()
    y_centered = y_v - y_mean

    # Try sklearn first
    try:
        from sklearn.linear_model import Lasso  # type: ignore

        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=2000)
        model.fit(X_v, y_centered)
        beta = model.coef_
    except ImportError:
        beta = _lasso_coord_descent(X_v, y_centered, alpha=alpha)

    if standardize:
        beta_orig = beta / x_std
        intercept = float(y_mean - x_mean @ beta_orig)
    else:
        beta_orig = beta
        intercept = float(y_mean)

    selected = [feature_names[i] for i, b in enumerate(beta) if abs(b) > 1e-9]
    # R²
    y_pred = X_v @ beta + y_mean
    ss_res = float(((y_v - y_pred) ** 2).sum())
    ss_tot = float(((y_v - y_mean) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return LassoResult(
        coefficients=beta_orig,
        intercept=intercept,
        selected_features=selected,
        alpha=alpha,
        r_squared=r2,
    )


def lasso_path(
    X: pd.DataFrame, y: pd.Series, alphas: list[float] | None = None
) -> pd.DataFrame:
    """LASSO regularization path — coefficients vs α.

    Returns:
        DataFrame (n_alphas, n_features + 1) with "alpha" + features.
    """
    if alphas is None:
        alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    rows = []
    for a in alphas:
        try:
            res = lasso_factor_selection(X, y, alpha=a)
            row = {"alpha": a, "r_squared": res.r_squared}
            for i, c in enumerate(X.columns):
                row[c] = res.coefficients[i]
            rows.append(row)
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def cv_optimal_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: list[float] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Cross-validation for LASSO α-selection.

    Time-series-aware: forward-chaining splits.
    """
    if alphas is None:
        alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
    df = pd.concat([X, y.rename("__y__")], axis=1).dropna()
    n = len(df)
    fold_size = n // n_folds
    cv_scores: dict[float, list[float]] = {a: [] for a in alphas}

    for k in range(1, n_folds):
        end_train = k * fold_size
        end_test = (k + 1) * fold_size if k < n_folds - 1 else n
        X_tr = df.iloc[:end_train][X.columns]
        y_tr = df.iloc[:end_train]["__y__"]
        X_te = df.iloc[end_train:end_test][X.columns]
        y_te = df.iloc[end_train:end_test]["__y__"]
        for a in alphas:
            try:
                res = lasso_factor_selection(X_tr, y_tr, alpha=a)
                # Predict
                y_pred = X_te.values @ res.coefficients + res.intercept
                mse = float(((y_te.values - y_pred) ** 2).mean())
                cv_scores[a].append(mse)
            except Exception:  # noqa: BLE001
                continue

    mean_scores = {
        a: float(np.mean(v)) if v else float("nan") for a, v in cv_scores.items()
    }
    if not mean_scores:
        return {"error": "no folds succeeded"}
    valid = {a: s for a, s in mean_scores.items() if np.isfinite(s)}
    if not valid:
        return {"error": "all NaN"}
    best_alpha = min(valid, key=valid.get)
    return {
        "best_alpha": best_alpha,
        "cv_mse": valid[best_alpha],
        "all_scores": mean_scores,
    }


__all__ = ["LassoResult", "lasso_factor_selection", "lasso_path", "cv_optimal_alpha"]
