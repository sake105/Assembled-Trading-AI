"""Barra-Style Multi-Factor Risk Model.

Decomposes portfolio risk into factor contributions using a cross-sectional
regression approach (Fama-MacBeth style). The model estimates:

    Total Covariance: Sigma = B @ F @ B.T + D

Where:
    B: Factor exposure matrix (n_assets × n_factors)
    F: Factor covariance matrix (n_factors × n_factors)
    D: Diagonal idiosyncratic variance matrix (n_assets × n_assets)

The six standard Barra-style factors implemented:
    1. Market   — SPY beta (systematic equity risk)
    2. Size     — log(market_cap) — small vs. large cap exposure
    3. Value    — book-to-market ratio (value vs. growth)
    4. Momentum — 12-month return excl. last month
    5. Quality  — return on equity (profitability)
    6. LowVol   — realized volatility (low-vol anomaly)

Usage:
    from src.assembled_core.risk.factor_risk_model import FactorRiskModel

    model = FactorRiskModel()
    model.fit(factor_exposures_df, returns_df)
    vol = model.predict_portfolio_vol(weights)
    contrib = model.predict_factor_contributions(weights)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard Barra-style factor names
BARRA_FACTORS = ["market", "size", "value", "momentum", "quality", "low_vol"]

# Mapping from factor name to expected column in factor exposure DataFrame
_FACTOR_COLUMN_MAP = {
    "market": "beta_market",
    "size": "log_market_cap",
    "value": "book_to_market",
    "momentum": "momentum_12m_excl_1m",
    "quality": "roe",
    "low_vol": "rv_20",
}

try:
    from sklearn.linear_model import LinearRegression  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    LinearRegression = None  # type: ignore


@dataclass
class FactorRiskModel:
    """Barra-style cross-sectional factor risk model.

    Attributes:
        factors: List of factor names to include (default: all 6 standard factors).
        factor_col_map: Dict mapping factor name → column name in exposure DataFrame.
        min_assets: Minimum assets per cross-section for regression (default: 20).
        cov_window: Rolling window for factor return covariance (default: 252).
    """

    factors: list[str] = field(default_factory=lambda: list(BARRA_FACTORS))
    factor_col_map: dict[str, str] = field(default_factory=lambda: dict(_FACTOR_COLUMN_MAP))
    min_assets: int = 20
    cov_window: int = 252

    # State set during fit
    _factor_cov: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _idio_var: Optional[pd.Series] = field(default=None, init=False, repr=False)
    _B: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _symbols: list[str] = field(default_factory=list, init=False, repr=False)
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not SKLEARN_AVAILABLE:
            logger.warning("[FactorRisk] sklearn not available — fit() will use numpy OLS")

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        factor_exposures: pd.DataFrame,
        returns: pd.DataFrame,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
    ) -> "FactorRiskModel":
        """Fit the factor risk model via Fama-MacBeth cross-sectional regressions.

        For each date t, regress cross-sectional returns onto factor exposures
        to estimate factor returns. Then compute factor covariance from the
        time-series of factor returns, and idiosyncratic variance from residuals.

        Args:
            factor_exposures: Panel DataFrame with symbol, timestamp, and factor
                columns (e.g., beta_market, log_market_cap, rv_20, ...).
            returns: Panel DataFrame with symbol, timestamp, and 'return' column
                (next-period returns, PIT-safe). Or wide format (symbols as columns).
            symbol_col: Symbol column name.
            timestamp_col: Timestamp column name.

        Returns:
            self (fitted)
        """
        # Determine which factor columns are present
        available_factors = [
            f for f in self.factors
            if self.factor_col_map.get(f, f) in factor_exposures.columns
        ]
        if not available_factors:
            raise ValueError(
                f"None of the factor columns found in factor_exposures. "
                f"Expected: {[self.factor_col_map.get(f, f) for f in self.factors]}"
            )

        factor_cols = [self.factor_col_map.get(f, f) for f in available_factors]
        logger.info("[FactorRisk] Fitting with factors: %s", available_factors)

        # Prepare returns in long format
        if "return" not in returns.columns and symbol_col in returns.columns:
            # Assume it's already long with a 'return' or 'ret' column
            ret_col = next(
                (c for c in ["return", "ret", "returns", "close"] if c in returns.columns),
                returns.columns[-1],
            )
        else:
            ret_col = "return"

        timestamps = sorted(factor_exposures[timestamp_col].unique())
        factor_return_rows = []

        for ts in timestamps:
            exp_t = factor_exposures[factor_exposures[timestamp_col] == ts]
            if symbol_col not in exp_t.columns:
                continue
            # Get matching returns
            if timestamp_col in returns.columns:
                ret_t = returns[returns[timestamp_col] == ts]
                syms_common = list(set(exp_t[symbol_col]) & set(ret_t[symbol_col]))
            else:
                # Wide format
                syms_common = [s for s in exp_t[symbol_col] if s in returns.columns and ts in returns.index]

            if len(syms_common) < self.min_assets:
                continue

            exp_sub = exp_t[exp_t[symbol_col].isin(syms_common)].set_index(symbol_col)
            X = exp_sub[factor_cols].reindex(syms_common).fillna(0).values

            if timestamp_col in returns.columns:
                ret_sub = ret_t[ret_t[symbol_col].isin(syms_common)].set_index(symbol_col)
                y = ret_sub[ret_col].reindex(syms_common).fillna(0).values
            else:
                y = returns.loc[ts, syms_common].fillna(0).values

            # Cross-sectional OLS: y_t = X_t * f_t + epsilon_t
            try:
                if SKLEARN_AVAILABLE:
                    reg = LinearRegression(fit_intercept=True)
                    reg.fit(X, y)
                    f_t = reg.coef_
                    _residuals = y - reg.predict(X)  # noqa: F841 — kept for future idio-vol
                else:
                    X_aug = np.column_stack([np.ones(len(X)), X])
                    coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                    f_t = coef[1:]  # skip intercept
                    _residuals = y - X_aug @ coef  # noqa: F841 — kept for future idio-vol
            except Exception as exc:
                # Audit-trail only (no behavior change): the existing
                # logic continues past failed timestamps, but previously
                # it did so silently. A single-date regression error
                # (singular matrix, numerical blow-up, etc.) would be
                # invisible in ops and could accumulate into material
                # coverage gaps in _B before the final raise at line 189.
                logger.warning(
                    "[FactorRisk] skipped timestamp %s regression: %s: %s",
                    ts, type(exc).__name__, exc,
                )
                continue

            row = {timestamp_col: ts}
            for fname, fval in zip(available_factors, f_t):
                row[fname] = float(fval)
            factor_return_rows.append(row)

        if not factor_return_rows:
            raise ValueError("[FactorRisk] No cross-sections had enough assets for regression")

        factor_returns_df = pd.DataFrame(factor_return_rows).set_index(timestamp_col)
        factor_returns_df.index = pd.to_datetime(factor_returns_df.index)
        factor_returns_df = factor_returns_df.sort_index()

        # Factor covariance from time series of factor returns
        f_cov = factor_returns_df.cov() * self.cov_window  # annualize
        self._factor_cov = f_cov

        # Build final exposure matrix on most recent date
        last_ts = timestamps[-1]
        exp_last = factor_exposures[factor_exposures[timestamp_col] == last_ts]
        exp_last = exp_last.set_index(symbol_col)[factor_cols].fillna(0)
        exp_last.columns = available_factors
        self._B = exp_last
        self._symbols = list(exp_last.index)

        # Idiosyncratic variance: average residual variance per symbol
        # Approximate via total return variance minus factor-explained variance
        # Use diagonal of (Sigma_total - B @ F @ B.T)
        if timestamp_col in returns.columns:
            ret_wide = returns.pivot_table(index=timestamp_col, columns=symbol_col, values=ret_col)
        else:
            ret_wide = returns
        ret_wide.index = pd.to_datetime(ret_wide.index)
        symbol_vols = ret_wide[self._symbols].std() ** 2 * self.cov_window  # annualize

        B_arr = self._B.values
        F_arr = self._factor_cov.values
        factor_var = pd.Series(
            np.diag(B_arr @ F_arr @ B_arr.T),
            index=self._symbols,
        )
        idio_var = (symbol_vols - factor_var).clip(lower=1e-6)
        self._idio_var = idio_var

        self._is_fitted = True
        logger.info(
            "[FactorRisk] Fitted on %d symbols, %d factor-return dates",
            len(self._symbols),
            len(factor_return_rows),
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_portfolio_vol(
        self,
        weights: pd.Series,
        annualized: bool = True,
    ) -> float:
        """Predict annualized portfolio volatility.

        Args:
            weights: Portfolio weights (index = symbols).
            annualized: Return annualized vol (default: True).

        Returns:
            Portfolio volatility (standard deviation).
        """
        self._check_fitted()
        sigma = self._build_full_cov()
        syms = [s for s in weights.index if s in sigma.index]
        if not syms:
            return float("nan")
        w = weights.reindex(syms).fillna(0).values
        S = sigma.reindex(index=syms, columns=syms).values
        var = float(w @ S @ w)
        vol = float(np.sqrt(max(var, 0)))
        return vol if annualized else vol / np.sqrt(252)

    def predict_factor_contributions(
        self,
        weights: pd.Series,
    ) -> pd.DataFrame:
        """Decompose portfolio variance into factor contributions.

        Returns:
            DataFrame with columns: factor, contribution_pct, contribution_vol.
            Rows sum to total portfolio variance.
        """
        self._check_fitted()
        syms = [s for s in weights.index if s in self._B.index]
        if not syms:
            return pd.DataFrame(columns=["factor", "contribution_pct", "contribution_vol"])

        w = weights.reindex(syms).fillna(0).values
        B = self._B.reindex(syms).fillna(0).values  # (n × k)
        F = self._factor_cov.values                  # (k × k)
        D = np.diag(self._idio_var.reindex(syms).fillna(0).values)  # (n × n)

        total_var = float(w @ (B @ F @ B.T + D) @ w)
        if total_var <= 0:
            return pd.DataFrame(columns=["factor", "contribution_pct", "contribution_vol"])

        # Factor contributions: w' B F_i B' w where F_i is contribution of factor i
        rows = []
        factor_names = list(self._factor_cov.columns)
        Bw = B.T @ w  # (k,) factor portfolio exposures

        for i, fname in enumerate(factor_names):
            # Marginal contribution of factor i
            F_i = np.zeros_like(F)
            F_i[i, :] = F[i, :]
            F_i[:, i] = F[:, i]
            F_i[i, i] = F[i, i]
            contrib = float(Bw @ F_i @ Bw) / total_var
            rows.append({
                "factor": fname,
                "contribution_pct": contrib * 100,
                "contribution_vol": float(np.sqrt(max(contrib * total_var, 0))),
            })

        # Idiosyncratic contribution
        idio_contrib = float(w @ D @ w) / total_var
        rows.append({
            "factor": "idiosyncratic",
            "contribution_pct": idio_contrib * 100,
            "contribution_vol": float(np.sqrt(max(idio_contrib * total_var, 0))),
        })

        return pd.DataFrame(rows).sort_values("contribution_pct", ascending=False).reset_index(drop=True)

    def _build_full_cov(self) -> pd.DataFrame:
        """Build full covariance matrix: Sigma = B @ F @ B.T + D."""
        B = self._B.values
        F = self._factor_cov.values
        D = np.diag(self._idio_var.reindex(self._symbols).fillna(0).values)
        sigma = B @ F @ B.T + D
        return pd.DataFrame(sigma, index=self._symbols, columns=self._symbols)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("FactorRiskModel must be fitted before calling predict*")


# ---------------------------------------------------------------------------
# 7.6  Factor Exposure Limits
# ---------------------------------------------------------------------------

def check_factor_exposure_limits(
    weights: pd.Series,
    factor_exposures: pd.DataFrame,
    max_factor_exposure: float = 0.5,
) -> list[dict]:
    """Check if portfolio factor exposures exceed limits.

    Args:
        weights: Symbol → weight.
        factor_exposures: DataFrame with symbol index and factor columns.
        max_factor_exposure: Maximum absolute factor exposure.

    Returns:
        List of violation dicts with factor, exposure, limit.
    """
    syms = [s for s in weights.index if s in factor_exposures.index]
    if not syms:
        return []

    w = weights.reindex(syms).fillna(0).values
    B = factor_exposures.reindex(syms).fillna(0).values

    port_exposures = B.T @ w  # (k,)

    violations = []
    for i, factor_name in enumerate(factor_exposures.columns):
        exposure = float(port_exposures[i])
        if abs(exposure) > max_factor_exposure:
            violations.append({
                "factor": factor_name,
                "exposure": round(exposure, 4),
                "limit": max_factor_exposure,
                "breach": round(abs(exposure) - max_factor_exposure, 4),
            })

    return violations
