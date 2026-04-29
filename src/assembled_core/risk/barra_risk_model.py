"""Barra-style factor risk decomposition — pure numpy/pandas implementation.

Decomposes portfolio variance into:
  market / sector / style (momentum, size, value) / idiosyncratic

Optionally uses `toraniko` for factor return estimation if installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BarraRiskModel:
    """Estimate factor exposures and decompose portfolio risk.

    Usage
    -----
    model = BarraRiskModel(returns, fundamentals)
    model.fit()
    decomp = model.decompose_portfolio_risk(portfolio_weights)
    """

    STYLE_FACTORS = ("momentum", "size", "value")

    def __init__(self, returns: pd.DataFrame, fundamentals: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        returns:
            Wide DataFrame of daily returns (index=date, columns=symbol).
        fundamentals:
            DataFrame with columns including ``market_cap`` and ``book_to_price``,
            indexed by symbol (or MultiIndex date×symbol).
        """
        self.returns = returns
        self.fundamentals = fundamentals
        self._factor_returns: pd.DataFrame | None = None
        self._factor_loadings: pd.DataFrame | None = None
        self._residuals: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> "BarraRiskModel":
        """Estimate factor loadings and factor returns via cross-sectional regression."""
        style_scores = self._build_style_scores()
        sector_dummies = self._build_sector_dummies()
        market_dummy = pd.DataFrame(
            1.0, index=style_scores.index, columns=["market"]
        )
        X = pd.concat([market_dummy, sector_dummies, style_scores], axis=1).fillna(0)
        self._factor_loadings = X

        # Time-series of factor returns (cross-sectional WLS per day)
        factor_ret_rows: list[pd.Series] = []
        residual_rows: list[pd.Series] = []
        for date in self.returns.index:
            y = self.returns.loc[date].dropna()
            if len(y) < X.shape[1] + 2:
                continue
            X_day = X.loc[y.index].fillna(0)
            try:
                coef, resid, *_ = np.linalg.lstsq(X_day.values, y.values, rcond=None)
                f_ret = pd.Series(coef, index=X_day.columns, name=date)
                r_vec = pd.Series(y.values - X_day.values @ coef, index=y.index, name=date)
                factor_ret_rows.append(f_ret)
                residual_rows.append(r_vec)
            except Exception:
                continue

        self._factor_returns = pd.DataFrame(factor_ret_rows)
        self._residuals = pd.DataFrame(residual_rows).T
        return self

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose_portfolio_risk(
        self, portfolio_weights: pd.Series | pd.DataFrame
    ) -> dict[str, float]:
        """Decompose portfolio variance into factor/idio components.

        Parameters
        ----------
        portfolio_weights:
            Series or single-column DataFrame of symbol → weight (sum to 1).

        Returns
        -------
        Dict with keys: market_var_pct, sector_var_pct, style_var_pct, idio_var_pct.
        """
        if self._factor_returns is None or self._factor_loadings is None:
            self.fit()

        w = portfolio_weights
        if isinstance(w, pd.DataFrame):
            w = w.iloc[:, 0]
        w = w.dropna()

        # Factor covariance
        F = self._factor_returns.cov()
        X = self._factor_loadings.loc[w.index].fillna(0)
        w_vec = w.values

        port_factor_exposure = X.T.values @ w_vec  # shape (n_factors,)
        total_factor_var = float(port_factor_exposure @ F.values @ port_factor_exposure)

        # Idiosyncratic variance
        if self._residuals is not None:
            resid_common = self._residuals.loc[
                self._residuals.index.isin(w.index)
            ]
            resid_var = resid_common.var(axis=1).reindex(w.index).fillna(0)
            idio_var = float((w_vec**2) @ resid_var.values)
        else:
            idio_var = 0.0

        total_var = total_factor_var + idio_var
        if total_var == 0:
            return {
                "market_var_pct": 0.0,
                "sector_var_pct": 0.0,
                "style_var_pct": 0.0,
                "idio_var_pct": 0.0,
                "total_variance": 0.0,
            }

        # Attribution by factor group
        factor_cols = list(self._factor_returns.columns)
        market_idx = [i for i, c in enumerate(factor_cols) if c == "market"]
        sector_idx = [i for i, c in enumerate(factor_cols)
                      if c not in ("market",) and c not in self.STYLE_FACTORS]
        style_idx = [i for i, c in enumerate(factor_cols) if c in self.STYLE_FACTORS]

        def _group_var(idxs: list[int]) -> float:
            if not idxs:
                return 0.0
            sub_exp = port_factor_exposure[idxs]
            sub_F = F.values[np.ix_(idxs, idxs)]
            return float(sub_exp @ sub_F @ sub_exp)

        return {
            "market_var_pct": _group_var(market_idx) / total_var,
            "sector_var_pct": _group_var(sector_idx) / total_var,
            "style_var_pct": _group_var(style_idx) / total_var,
            "idio_var_pct": idio_var / total_var,
            "total_variance": total_var,
        }

    def factor_exposures(self, symbol: str) -> pd.Series | None:
        """Return factor loadings for a single symbol."""
        if self._factor_loadings is None:
            self.fit()
        if symbol not in self._factor_loadings.index:
            return None
        return self._factor_loadings.loc[symbol]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_style_scores(self) -> pd.DataFrame:
        """Compute cross-sectionally standardised momentum, size, value loadings."""
        symbols = self.returns.columns.tolist()

        # Momentum: 12M-1M return
        mom_252 = self.returns.iloc[-252:].mean() * 252 if len(self.returns) >= 252 else self.returns.mean()
        mom_21 = self.returns.iloc[-21:].mean() * 21 if len(self.returns) >= 21 else self.returns.mean()
        momentum = mom_252 - mom_21

        # Fundamentals-based: expect fundamentals indexed by symbol
        fund = self.fundamentals
        if isinstance(fund.index, pd.MultiIndex):
            fund = fund.xs(fund.index.get_level_values(0)[-1], level=0)

        mcap = fund["market_cap"].reindex(symbols) if "market_cap" in fund.columns else pd.Series(np.nan, index=symbols)
        b2p = fund["book_to_price"].reindex(symbols) if "book_to_price" in fund.columns else pd.Series(np.nan, index=symbols)

        size = -np.log(mcap.clip(lower=1))  # smaller market cap → positive size score

        scores = pd.DataFrame({
            "momentum": momentum,
            "size": size,
            "value": b2p,
        }, index=symbols)

        # Cross-sectional standardisation
        return scores.apply(lambda col: (col - col.mean()) / (col.std() + 1e-9))

    def _build_sector_dummies(self) -> pd.DataFrame:
        """Build sector dummy matrix if sector column present in fundamentals."""
        symbols = self.returns.columns.tolist()
        fund = self.fundamentals
        if isinstance(fund.index, pd.MultiIndex):
            fund = fund.xs(fund.index.get_level_values(0)[-1], level=0)

        if "sector" not in fund.columns:
            return pd.DataFrame(index=symbols)

        sectors = fund["sector"].reindex(symbols).fillna("Unknown")
        return pd.get_dummies(sectors, prefix="sector", dtype=float)
