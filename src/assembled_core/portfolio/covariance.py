from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def returns_from_prices(prices: pd.DataFrame, log_returns: bool = True) -> pd.DataFrame:
    """Convert long-format price DataFrame to wide-format returns.

    Expects columns: timestamp, symbol, close.
    Returns wide DataFrame (index=timestamp, columns=symbols).
    """
    if prices is None or prices.empty:
        return pd.DataFrame()

    if "symbol" in prices.columns and "close" in prices.columns:
        ts_col = "timestamp" if "timestamp" in prices.columns else prices.columns[0]
        wide = prices.pivot_table(
            index=ts_col, columns="symbol", values="close", aggfunc="last"
        )
    else:
        wide = prices.copy()

    wide = wide.sort_index()
    if log_returns:
        ret = np.log(wide / wide.shift(1)).iloc[1:]
    else:
        ret = wide.pct_change().iloc[1:]

    return ret.dropna(how="all")


def _ewm_covariance(
    ret: pd.DataFrame, halflife: int = 30, min_periods: int | None = None
) -> np.ndarray:
    """Compute EWM covariance matrix from returns DataFrame.

    Returns zero matrix if fewer than min_periods rows available.
    """
    if ret is None or ret.empty:
        n = ret.shape[1] if ret is not None else 0
        return np.zeros((n, n))

    mp = min_periods if min_periods is not None else max(1, halflife // 2)
    n = ret.shape[1]

    if len(ret) < mp:
        return np.zeros((n, n))

    alpha = 1.0 - np.exp(-np.log(2) / halflife)
    weights = np.array([(1 - alpha) ** i for i in range(len(ret) - 1, -1, -1)])
    weights /= weights.sum()

    X = ret.values
    mean = (weights[:, None] * X).sum(axis=0)
    X_c = X - mean
    cov: np.ndarray = (weights[:, None] * X_c).T @ X_c
    return cov


def _ensure_psd(cov: pd.DataFrame, min_eigenval: float = 1e-8) -> pd.DataFrame:
    """Clip negative eigenvalues to min_eigenval to enforce positive definiteness."""
    if cov is None or cov.empty:
        return cov

    vals, vecs = np.linalg.eigh(cov.values)
    vals_clipped = np.maximum(vals, min_eigenval)
    psd = vecs @ np.diag(vals_clipped) @ vecs.T
    return pd.DataFrame(psd, index=cov.index, columns=cov.columns)


def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
    ewm_halflife: int = 30,
    annualize: bool = False,
) -> pd.DataFrame:
    """Estimate covariance matrix from returns DataFrame.

    method: "ledoit_wolf" | "oas" | "ewm" | "dcc_garch" | "cdcc" | "sample"
        dcc_garch: Engle (2002) Dynamic Conditional Correlation, real impl since C4-072 (2026-05-17).
        cdcc:      Aielli (2013) bias-corrected DCC.
        Both return the most recent conditional covariance H_T.
    ewm_halflife: halflife in periods, used when method="ewm"
    annualize: multiply result by 252 (daily→annual)
    """
    if returns is None or returns.empty or returns.shape[1] < 2:
        logger.debug(
            "[SKIP] estimate_covariance: insufficient data (shape=%s)",
            getattr(returns, "shape", None),
        )
        return pd.DataFrame()

    clean = returns.dropna(axis=1, how="all").fillna(0.0)
    if clean.empty or clean.shape[1] < 1:
        return pd.DataFrame()

    cols = clean.columns
    X = clean.values

    try:
        result_df: pd.DataFrame

        if method == "ewm":
            cov_arr = _ewm_covariance(clean, halflife=ewm_halflife)
            result_df = pd.DataFrame(cov_arr, index=cols, columns=cols)
        elif method in ("ledoit_wolf", "oas"):
            try:
                if method == "ledoit_wolf":
                    from sklearn.covariance import LedoitWolf

                    cov = LedoitWolf().fit(X).covariance_
                else:
                    from sklearn.covariance import OAS

                    cov = OAS().fit(X).covariance_
                result_df = pd.DataFrame(cov, index=cols, columns=cols)
            except ImportError:
                logger.debug(
                    "[WARN] sklearn not available, falling back to sample covariance"
                )
                cov = np.cov(X, rowvar=False)
                result_df = pd.DataFrame(cov, index=cols, columns=cols)
        elif method in ("dcc_garch", "cdcc"):
            # C4-072 (2026-05-17): real DCC-GARCH (Engle 2002) / cDCC (Aielli 2013)
            # via src.assembled_core.risk.dcc_garch — replaces the previous
            # silent fall-through to sample covariance (§7.4 dummy violation).
            from src.assembled_core.risk.dcc_garch import (
                current_covariance,
                fit_dcc_garch,
            )

            dcc_method: Literal["dcc", "cdcc"] = "cdcc" if method == "cdcc" else "dcc"
            dcc_result = fit_dcc_garch(clean, method=dcc_method)
            if dcc_result is None:
                logger.warning(
                    "[WARN] DCC-GARCH unavailable (missing arch/scipy or fit failed); "
                    "falling back to sample covariance"
                )
                cov = np.cov(X, rowvar=False)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                result_df = pd.DataFrame(cov, index=cols, columns=cols)
            else:
                result_df = current_covariance(dcc_result)
        else:
            # sample (default fallback for unknown method names)
            cov = np.cov(X, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            result_df = pd.DataFrame(cov, index=cols, columns=cols)

        if annualize:
            result_df = result_df * 252.0
        return result_df
    except Exception as exc:
        logger.debug("[ERROR] estimate_covariance failed: %s", exc)
        return pd.DataFrame()
