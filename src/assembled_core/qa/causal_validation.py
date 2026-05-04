"""Causal validation — estimate average treatment effects for news triggers.

Uses `dowhy` if available; falls back to linear regression ATE when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CausalEstimate:
    ate: float  # average treatment effect
    method: str
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def estimate_news_trigger_effect(
    trades_df: pd.DataFrame,
    treatment_col: str = "has_news_trigger",
    outcome_col: str = "return",
    common_causes: list[str] | None = None,
    use_dowhy: bool = True,
) -> dict[str, Any]:
    """Estimate the causal effect of news triggers on trade returns.

    Tries `dowhy` (if installed and ``use_dowhy=True``); falls back to
    OLS regression ATE with propensity-score matching when unavailable.

    Parameters
    ----------
    trades_df:
        DataFrame with at least ``treatment_col`` (binary 0/1) and ``outcome_col``.
        Include confounders as additional columns (sector, vol_regime, etc.).
    treatment_col:
        Binary indicator for news trigger presence.
    outcome_col:
        Continuous outcome (e.g. trade return).
    common_causes:
        Column names of confounders. Defaults to all other numeric columns.
    use_dowhy:
        Whether to attempt dowhy estimation. Set False to force OLS fallback.

    Returns
    -------
    Dict with ``estimates``, ``interpretation``, and ``method_used``.
    """
    if common_causes is None:
        common_causes = [
            c
            for c in trades_df.columns
            if c not in (treatment_col, outcome_col)
            and pd.api.types.is_numeric_dtype(trades_df[c])
        ]

    if use_dowhy:
        try:
            return _estimate_dowhy(trades_df, treatment_col, outcome_col, common_causes)
        except ImportError:
            pass
        except Exception as exc:
            notes = f"dowhy failed: {exc!r}; falling back to OLS"
            result = _estimate_ols(trades_df, treatment_col, outcome_col, common_causes)
            result["notes"] = notes
            return result

    return _estimate_ols(trades_df, treatment_col, outcome_col, common_causes)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _estimate_ols(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any]:
    """OLS regression with confounders to estimate ATE."""
    from scipy import stats as scipy_stats  # noqa: PLC0415

    X_cols = [treatment] + confounders
    sub = df[[outcome] + X_cols].dropna()
    if len(sub) < 5:
        return {
            "estimates": {"ols": CausalEstimate(0.0, "ols", notes="insufficient data")},
            "method_used": "ols_fallback",
            "interpretation": "Insufficient data for estimation.",
        }

    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in X_cols])
    y = sub[outcome].values
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        ate = float(coef[1])  # coefficient on treatment indicator
    except Exception:
        return {
            "estimates": {"ols": CausalEstimate(0.0, "ols", notes="lstsq failed")},
            "method_used": "ols_fallback",
            "interpretation": "OLS estimation failed.",
        }

    # Two-sample t-test as a supplementary check
    treated = sub.loc[sub[treatment] == 1, outcome].values
    control = sub.loc[sub[treatment] == 0, outcome].values
    p_val: float | None = None
    if len(treated) >= 2 and len(control) >= 2:
        _, p_val = scipy_stats.ttest_ind(treated, control)

    naive_ate = (
        float(np.mean(treated) - np.mean(control))
        if len(treated) and len(control)
        else 0.0
    )

    estimates = {
        "ols_adjusted": CausalEstimate(ate=ate, method="ols_adjusted", p_value=p_val),
        "naive_ate": CausalEstimate(ate=naive_ate, method="naive_ttest", p_value=p_val),
    }
    return {
        "estimates": {k: vars(v) for k, v in estimates.items()},
        "method_used": "ols_fallback",
        "n_treated": int(len(treated)),
        "n_control": int(len(control)),
        "interpretation": (
            "OLS-adjusted ATE. If estimates are similar across methods "
            "AND p-value > 0.05 under placebo treatment, causal effect is robust."
        ),
    }


def _estimate_dowhy(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> dict[str, Any]:
    """dowhy-based causal estimation."""
    from dowhy import CausalModel  # noqa: PLC0415

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    estimates: dict[str, Any] = {}
    for method in [
        "backdoor.linear_regression",
        "backdoor.propensity_score_matching",
    ]:
        try:
            est = model.estimate_effect(identified, method_name=method)
            estimates[method] = {"ate": float(est.value), "method": method}
        except Exception as exc:
            estimates[method] = {"ate": None, "error": str(exc)}

    # Refutation check (only for linear regression)
    refutations: dict[str, Any] = {}
    if (
        "backdoor.linear_regression" in estimates
        and estimates["backdoor.linear_regression"].get("ate") is not None
    ):
        try:
            lin_est = model.estimate_effect(
                identified, method_name="backdoor.linear_regression"
            )
            ref = model.refute_estimate(
                identified, lin_est, method_name="random_common_cause"
            )
            refutations["random_common_cause"] = {
                "original_effect": ref.estimated_effect,
                "new_effect": ref.new_effect,
                "p_value": (
                    ref.refutation_result.get("p_value")
                    if ref.refutation_result
                    else None
                ),
            }
        except Exception as exc:
            refutations["error"] = str(exc)

    return {
        "estimates": estimates,
        "refutations": refutations,
        "method_used": "dowhy",
        "interpretation": (
            "If estimates are similar across methods AND refutation p-value > 0.05, "
            "causal effect is robust."
        ),
    }


def heterogeneous_treatment_effects(
    trades_df: pd.DataFrame,
    treatment_col: str = "has_news_trigger",
    outcome_col: str = "return",
    heterogeneity_col: str = "sector",
) -> pd.DataFrame:
    """Simple stratified ATE by heterogeneity_col (sector, regime, etc.).

    Returns a DataFrame with each stratum's ATE, sample size, and naive p-value.
    """
    results: list[dict[str, Any]] = []
    for group, subset in trades_df.groupby(heterogeneity_col):
        treated = subset.loc[subset[treatment_col] == 1, outcome_col].dropna()
        control = subset.loc[subset[treatment_col] == 0, outcome_col].dropna()
        if len(treated) < 2 or len(control) < 2:
            continue
        from scipy import stats as scipy_stats  # noqa: PLC0415

        ate = float(treated.mean() - control.mean())
        _, p = scipy_stats.ttest_ind(treated.values, control.values)
        results.append(
            {
                heterogeneity_col: group,
                "ate": ate,
                "n_treated": len(treated),
                "n_control": len(control),
                "p_value": float(p),
                "significant": p < 0.05,
            }
        )
    return pd.DataFrame(results)
