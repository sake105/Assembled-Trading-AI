"""Causal Inference for Factor–Return Relationships (M29).

Implements causal estimation methods to distinguish genuine alpha factors
from spurious correlations:
  1. Propensity Score Matching: estimate causal effect of factor exposure
  2. Instrumental Variable (IV) estimation: use lagged values as instruments
  3. Difference-in-Differences: before/after event impact estimation
  4. Granger Causality: temporal causal ordering between factors and returns

The key insight: correlation between a factor and returns does not imply
the factor *causes* returns. Causal methods help prune the factor zoo
by identifying factors with genuine predictive *mechanism*.

Reference:
    Pearl, J. (2009). "Causality."
    Angrist, J. & Pischke, J.S. (2009). "Mostly Harmless Econometrics."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CausalEffectResult:
    """Result of causal effect estimation.

    Attributes:
        factor_name: Name of the factor being tested.
        method: Estimation method used.
        ate: Average Treatment Effect (causal effect estimate).
        ate_stderr: Standard error of the ATE.
        t_statistic: t-statistic for H0: ATE=0.
        p_value: p-value for H0: ATE=0.
        n_treated: Number of treated observations.
        n_control: Number of control observations.
        is_significant: Whether the effect is significant at 5%.
    """

    factor_name: str
    method: str
    ate: float
    ate_stderr: float
    t_statistic: float
    p_value: float
    n_treated: int
    n_control: int
    is_significant: bool


@dataclass
class GrangerResult:
    """Result of Granger causality test.

    Attributes:
        cause_variable: The potential causal variable.
        effect_variable: The potential effect variable.
        f_statistic: F-statistic for the Granger test.
        p_value: p-value.
        optimal_lag: Best lag order selected.
        is_causal: Whether Granger causality is detected (p < 0.05).
    """

    cause_variable: str
    effect_variable: str
    f_statistic: float
    p_value: float
    optimal_lag: int
    is_causal: bool


def estimate_propensity_score(
    covariates: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    """Estimate propensity scores via logistic regression.

    P(treatment=1 | covariates) using a simple logistic model.

    Args:
        covariates: (n, k) array of confounding variables.
        treatment: Binary treatment indicator (0/1).

    Returns:
        Array of propensity scores in (0, 1).
    """
    X = np.asarray(covariates, dtype=float)
    t = np.asarray(treatment, dtype=float).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-10] = 1.0
    X_std = (X - mu) / sd

    # Add intercept
    n = len(t)
    X_aug = np.column_stack([np.ones(n), X_std])

    # Logistic regression via IRLS (iteratively reweighted least squares)
    beta = np.zeros(X_aug.shape[1])
    for _ in range(25):
        p = 1.0 / (1.0 + np.exp(-X_aug @ beta))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        w = p * (1 - p)
        W = np.diag(w)
        z = X_aug @ beta + (t - p) / w
        try:
            beta = np.linalg.solve(X_aug.T @ W @ X_aug + 1e-6 * np.eye(X_aug.shape[1]),
                                   X_aug.T @ W @ z)
        except np.linalg.LinAlgError:
            break

    scores = 1.0 / (1.0 + np.exp(-X_aug @ beta))
    return np.clip(scores, 0.01, 0.99)


def propensity_score_matching(
    factor_values: np.ndarray,
    returns: np.ndarray,
    covariates: np.ndarray | None = None,
    n_quantiles: int = 5,
) -> CausalEffectResult:
    """Estimate causal effect via propensity score matching.

    Splits factor into high (treated) vs low (control) groups,
    matches on propensity scores, and estimates the ATE.

    Args:
        factor_values: Factor exposure values.
        returns: Forward returns.
        covariates: Confounding variables to control for.
            If None, uses lagged factor values.
        n_quantiles: Number of propensity score strata.

    Returns:
        CausalEffectResult with ATE and significance.
    """
    f = np.asarray(factor_values, dtype=float).ravel()
    r = np.asarray(returns, dtype=float).ravel()

    # Remove NaN
    mask = np.isfinite(f) & np.isfinite(r)
    if covariates is not None:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov.reshape(-1, 1)
        mask &= np.all(np.isfinite(cov), axis=1)
        cov = cov[mask]
    else:
        # Use lagged factor as covariate (proxy for confounders)
        cov = np.roll(f, 1).reshape(-1, 1)
        cov[0] = f.mean()
        mask &= np.isfinite(cov.ravel())
        cov = cov[mask]

    f = f[mask]
    r = r[mask]

    if len(f) < 20:
        return CausalEffectResult(
            factor_name="unknown", method="propensity_score_matching",
            ate=0.0, ate_stderr=1.0, t_statistic=0.0, p_value=1.0,
            n_treated=0, n_control=0, is_significant=False,
        )

    # Define treatment: top quintile vs bottom quintile
    q_high = np.percentile(f, 80)
    q_low = np.percentile(f, 20)
    treatment = np.where(f >= q_high, 1, np.where(f <= q_low, 0, -1))
    valid = treatment >= 0
    treatment = treatment[valid].astype(float)
    r_valid = r[valid]
    cov_valid = cov[valid] if cov.ndim > 1 else cov[valid].reshape(-1, 1)

    if treatment.sum() < 5 or (1 - treatment).sum() < 5:
        return CausalEffectResult(
            factor_name="unknown", method="propensity_score_matching",
            ate=0.0, ate_stderr=1.0, t_statistic=0.0, p_value=1.0,
            n_treated=int(treatment.sum()), n_control=int((1 - treatment).sum()),
            is_significant=False,
        )

    # Estimate propensity scores
    ps = estimate_propensity_score(cov_valid, treatment)

    # Stratified ATE estimation
    strata_effects = []
    strata_counts = []
    quantile_bounds = np.linspace(0, 1, n_quantiles + 1)

    for i in range(n_quantiles):
        lo = np.percentile(ps, quantile_bounds[i] * 100)
        hi = np.percentile(ps, quantile_bounds[i + 1] * 100)
        in_stratum = (ps >= lo) & (ps <= hi)
        treated_in = in_stratum & (treatment == 1)
        control_in = in_stratum & (treatment == 0)

        if treated_in.sum() >= 2 and control_in.sum() >= 2:
            effect = r_valid[treated_in].mean() - r_valid[control_in].mean()
            strata_effects.append(effect)
            strata_counts.append(int(in_stratum.sum()))

    if not strata_effects:
        return CausalEffectResult(
            factor_name="unknown", method="propensity_score_matching",
            ate=0.0, ate_stderr=1.0, t_statistic=0.0, p_value=1.0,
            n_treated=int(treatment.sum()), n_control=int((1 - treatment).sum()),
            is_significant=False,
        )

    # Weighted average ATE
    total_count = sum(strata_counts)
    ate = sum(e * c / total_count for e, c in zip(strata_effects, strata_counts))

    # Standard error via bootstrap-like variance across strata
    if len(strata_effects) > 1:
        ate_stderr = float(np.std(strata_effects) / np.sqrt(len(strata_effects)))
    else:
        ate_stderr = abs(ate) * 0.5 + 1e-6

    t_stat = ate / ate_stderr if ate_stderr > 1e-10 else 0.0

    # Approximate p-value from t-distribution (normal approx for large n)
    from math import erfc, sqrt
    p_value = erfc(abs(t_stat) / sqrt(2))

    return CausalEffectResult(
        factor_name="unknown",
        method="propensity_score_matching",
        ate=round(float(ate), 6),
        ate_stderr=round(float(ate_stderr), 6),
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_value), 4),
        n_treated=int(treatment.sum()),
        n_control=int((1 - treatment).sum()),
        is_significant=p_value < 0.05,
    )


def iv_two_stage_least_squares(
    factor_values: np.ndarray,
    returns: np.ndarray,
    instrument: np.ndarray,
) -> CausalEffectResult:
    """Estimate causal effect via two-stage least squares (2SLS).

    Uses an instrumental variable to isolate the causal component
    of the factor-return relationship.

    Stage 1: factor = alpha + gamma * instrument + epsilon
    Stage 2: returns = beta_0 + beta_1 * factor_hat + u

    Args:
        factor_values: Endogenous factor exposure.
        returns: Forward returns (outcome).
        instrument: Instrumental variable (e.g., lagged factor).

    Returns:
        CausalEffectResult with IV estimate.
    """
    f = np.asarray(factor_values, dtype=float).ravel()
    r = np.asarray(returns, dtype=float).ravel()
    z = np.asarray(instrument, dtype=float).ravel()

    mask = np.isfinite(f) & np.isfinite(r) & np.isfinite(z)
    f, r, z = f[mask], r[mask], z[mask]

    n = len(f)
    if n < 20:
        return CausalEffectResult(
            factor_name="unknown", method="iv_2sls",
            ate=0.0, ate_stderr=1.0, t_statistic=0.0, p_value=1.0,
            n_treated=n // 2, n_control=n // 2, is_significant=False,
        )

    # Stage 1: regress factor on instrument
    Z = np.column_stack([np.ones(n), z])
    gamma = np.linalg.lstsq(Z, f, rcond=None)[0]
    f_hat = Z @ gamma

    # Stage 2: regress returns on predicted factor
    X = np.column_stack([np.ones(n), f_hat])
    beta = np.linalg.lstsq(X, r, rcond=None)[0]

    # IV estimate is beta[1]
    iv_estimate = beta[1]

    # Standard error
    residuals = r - X @ beta
    sigma2 = float(np.dot(residuals, residuals) / (n - 2))
    try:
        cov_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = float(np.sqrt(max(cov_beta[1, 1], 1e-12)))
    except np.linalg.LinAlgError:
        se = abs(iv_estimate) * 0.5 + 1e-6

    t_stat = iv_estimate / se if se > 1e-10 else 0.0
    from math import erfc, sqrt
    p_value = erfc(abs(t_stat) / sqrt(2))

    return CausalEffectResult(
        factor_name="unknown",
        method="iv_2sls",
        ate=round(float(iv_estimate), 6),
        ate_stderr=round(float(se), 6),
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_value), 4),
        n_treated=n // 2,
        n_control=n - n // 2,
        is_significant=p_value < 0.05,
    )


def difference_in_differences(
    returns_treated: np.ndarray,
    returns_control: np.ndarray,
    pre_periods: int,
) -> CausalEffectResult:
    """Estimate causal effect via Difference-in-Differences.

    Compares the change in returns for treated vs control groups
    around an event (e.g., factor exposure change).

    DiD = (treated_post - treated_pre) - (control_post - control_pre)

    Args:
        returns_treated: Full return series for treated group.
        returns_control: Full return series for control group.
        pre_periods: Number of pre-event periods.

    Returns:
        CausalEffectResult with DiD estimate.
    """
    rt = np.asarray(returns_treated, dtype=float)
    rc = np.asarray(returns_control, dtype=float)

    min_len = min(len(rt), len(rc))
    if min_len < pre_periods + 5 or pre_periods < 5:
        return CausalEffectResult(
            factor_name="unknown", method="difference_in_differences",
            ate=0.0, ate_stderr=1.0, t_statistic=0.0, p_value=1.0,
            n_treated=len(rt), n_control=len(rc), is_significant=False,
        )

    rt = rt[:min_len]
    rc = rc[:min_len]

    treated_pre = rt[:pre_periods].mean()
    treated_post = rt[pre_periods:].mean()
    control_pre = rc[:pre_periods].mean()
    control_post = rc[pre_periods:].mean()

    did = (treated_post - treated_pre) - (control_post - control_pre)

    # Standard error via pooled variance
    n_post = min_len - pre_periods
    var_treated = rt[pre_periods:].var() / n_post
    var_control = rc[pre_periods:].var() / n_post
    se = float(np.sqrt(var_treated + var_control + 1e-12))

    t_stat = did / se if se > 1e-10 else 0.0
    from math import erfc, sqrt
    p_value = erfc(abs(t_stat) / sqrt(2))

    return CausalEffectResult(
        factor_name="unknown",
        method="difference_in_differences",
        ate=round(float(did), 6),
        ate_stderr=round(float(se), 6),
        t_statistic=round(float(t_stat), 4),
        p_value=round(float(p_value), 4),
        n_treated=len(rt),
        n_control=len(rc),
        is_significant=p_value < 0.05,
    )


def granger_causality_test(
    cause: np.ndarray,
    effect: np.ndarray,
    max_lag: int = 5,
) -> GrangerResult:
    """Test Granger causality: does `cause` help predict `effect`?

    Compares restricted model (effect ~ own lags) vs unrestricted
    model (effect ~ own lags + cause lags) using F-test.

    Args:
        cause: Potential causal time series.
        effect: Potential effect time series.
        max_lag: Maximum lag order to test.

    Returns:
        GrangerResult with F-statistic and significance.
    """
    x = np.asarray(cause, dtype=float).ravel()
    y = np.asarray(effect, dtype=float).ravel()

    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    n = len(y)
    if n < max_lag + 20:
        return GrangerResult(
            cause_variable="cause", effect_variable="effect",
            f_statistic=0.0, p_value=1.0, optimal_lag=1, is_causal=False,
        )

    best_f = 0.0
    best_p = 1.0
    best_lag = 1

    for lag in range(1, max_lag + 1):
        # Build lagged matrices
        y_dep = y[lag:]
        n_obs = len(y_dep)

        # Restricted model: y_t ~ y_{t-1} + ... + y_{t-lag}
        Y_lags = np.column_stack([y[lag - i - 1:n - i - 1] for i in range(lag)])
        X_restricted = np.column_stack([np.ones(n_obs), Y_lags])

        # Unrestricted model: add x lags
        X_lags = np.column_stack([x[lag - i - 1:n - i - 1] for i in range(lag)])
        X_unrestricted = np.column_stack([X_restricted, X_lags])

        # OLS
        try:
            beta_r = np.linalg.lstsq(X_restricted, y_dep, rcond=None)[0]
            beta_u = np.linalg.lstsq(X_unrestricted, y_dep, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        rss_r = float(np.sum((y_dep - X_restricted @ beta_r) ** 2))
        rss_u = float(np.sum((y_dep - X_unrestricted @ beta_u) ** 2))

        df_diff = lag
        df_resid = n_obs - X_unrestricted.shape[1]

        if df_resid <= 0 or rss_u < 1e-15:
            continue

        f_stat = ((rss_r - rss_u) / df_diff) / (rss_u / df_resid)

        # F-distribution p-value approximation
        # Using the relationship between F and Beta distributions
        try:
            d1, d2 = float(df_diff), float(df_resid)
            # Use normal approximation for large df
            if d2 > 30:
                # Approximate F p-value via chi-squared
                chi2 = f_stat * d1
                # Chi-squared survival via normal approximation
                z = (chi2 / d1) ** (1 / 3) - (1 - 2 / (9 * d1))
                z /= (2 / (9 * d1)) ** 0.5
                from math import erfc, sqrt
                p_val = 0.5 * erfc(z / sqrt(2))
            else:
                p_val = 0.5  # conservative fallback
        except (ValueError, OverflowError):
            p_val = 0.5

        if f_stat > best_f:
            best_f = f_stat
            best_p = p_val
            best_lag = lag

    return GrangerResult(
        cause_variable="cause",
        effect_variable="effect",
        f_statistic=round(best_f, 4),
        p_value=round(best_p, 4),
        optimal_lag=best_lag,
        is_causal=best_p < 0.05,
    )


def screen_factors_causal(
    factor_df: pd.DataFrame,
    returns: pd.Series,
    methods: list[str] | None = None,
) -> list[CausalEffectResult]:
    """Screen multiple factors for causal relationships with returns.

    Args:
        factor_df: DataFrame with factors as columns.
        returns: Forward returns series (aligned with factor_df index).
        methods: List of methods to use. Default: ["propensity_score_matching"].

    Returns:
        List of CausalEffectResult, one per factor.
    """
    if methods is None:
        methods = ["propensity_score_matching"]

    r = returns.values
    results = []

    for col in factor_df.columns:
        f = factor_df[col].values

        for method in methods:
            if method == "propensity_score_matching":
                result = propensity_score_matching(f, r)
            elif method == "iv_2sls":
                instrument = np.roll(f, 1)
                instrument[0] = f.mean()
                result = iv_two_stage_least_squares(f, r, instrument)
            else:
                continue

            result.factor_name = col
            results.append(result)

    # Sort by p-value
    results.sort(key=lambda x: x.p_value)

    logger.info(
        "[CausalInference] Screened %d factors, %d significant (p<0.05)",
        len(results),
        sum(1 for r in results if r.is_significant),
    )

    return results


__all__ = [
    "CausalEffectResult",
    "GrangerResult",
    "estimate_propensity_score",
    "propensity_score_matching",
    "iv_two_stage_least_squares",
    "difference_in_differences",
    "granger_causality_test",
    "screen_factors_causal",
]
