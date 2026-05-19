"""Synthetic Control Method (Abadie-Diamond-Hainmueller 2003/2010).

Audit C2-027 showcase: estimate counterfactual outcome for a single treated
unit by constructing a weighted combination of untreated donor units that
matches the treated unit's pre-treatment trajectory.

The method:

1. **Pre-treatment fit:** find weights ``w = (w_2, ..., w_N)`` over donor
   units such that the **synthetic control** ``Y_syn(t) = Σ_j w_j Y_j(t)``
   approximates the treated unit's outcome ``Y_1(t)`` over the pre-treatment
   periods ``1..T_0``. Constraints:

   .. math::

       w_j \\geq 0, \\quad \\sum_j w_j = 1

   The constrained least-squares problem is solved via scipy SLSQP.

2. **Treatment effect:** for ``t > T_0``::

       TE(t) = Y_1(t) - Y_syn(t)

3. **Placebo inference:** apply the same method to every donor unit (treating
   each donor as if IT were the treated unit, with the remaining donors as
   the new donor pool) and compute placebo treatment effects. The original
   treatment effect's significance is the rank of the original effect's
   magnitude in the distribution of placebo effects.

Public API:
- :func:`fit_synthetic_control` — fit weights for a single treated unit.
- :func:`compute_treatment_effect` — observed - synthetic per period.
- :func:`placebo_test` — distribution of placebo effects + percentile rank.

References:
- Abadie, A., & Gardeazabal, J. (2003). The Economic Costs of Conflict: A
  Case Study of the Basque Country. American Economic Review, 93(1), 113-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). Synthetic Control
  Methods for Comparative Case Studies: Estimating the Effect of California's
  Tobacco Control Program. JASA, 105(490), 493-505.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class SyntheticControlResult:
    """Result of fitting a synthetic control for one treated unit.

    Attributes:
        weights: pd.Series of donor weights (index = donor names, sum to 1).
        synthetic_series: pd.Series, synthetic outcome over ALL periods
            (pre + post). Length matches the input outcome series.
        pre_treatment_rmse: Root-mean-squared error of the synthetic fit
            on pre-treatment periods (lower = better pre-treatment match).
        treated_name: Name of the treated unit (for reporting).
        n_donors: Number of donor units used.
        n_pre: Number of pre-treatment periods.
        n_post: Number of post-treatment periods.
        converged: Whether SLSQP reported success.
    """

    weights: pd.Series
    synthetic_series: pd.Series
    pre_treatment_rmse: float
    treated_name: str
    n_donors: int
    n_pre: int
    n_post: int
    converged: bool


def fit_synthetic_control(
    treated: pd.Series,
    donor_pool: pd.DataFrame,
    treatment_period: int,
) -> SyntheticControlResult:
    """Fit synthetic control weights for a single treated unit.

    Args:
        treated: pd.Series of the treated unit's outcome (index = time).
        donor_pool: pd.DataFrame, columns = donor units, rows = time.
            MUST share the same index as ``treated``.
        treatment_period: Integer index marking the FIRST post-treatment
            period. Pre-treatment periods are ``0..treatment_period-1``;
            post-treatment are ``treatment_period..len(treated)-1``.

    Returns:
        :class:`SyntheticControlResult`.

    Raises:
        ValueError: If shapes don't match, treatment_period out of range,
            donor_pool has fewer than 2 columns, or treated has NaN.
    """
    if not isinstance(treated, pd.Series):
        raise ValueError(f"treated must be pd.Series, got {type(treated)}")
    if not isinstance(donor_pool, pd.DataFrame):
        raise ValueError(f"donor_pool must be pd.DataFrame, got {type(donor_pool)}")
    if len(treated) != len(donor_pool):
        raise ValueError(
            f"treated has {len(treated)} rows, donor_pool has {len(donor_pool)} "
            "— must match"
        )
    if donor_pool.shape[1] < 2:
        raise ValueError(f"donor_pool needs ≥2 donors, got {donor_pool.shape[1]}")
    if treated.isna().any():
        raise ValueError("treated contains NaN")
    if donor_pool.isna().any().any():
        raise ValueError("donor_pool contains NaN")
    n_periods = len(treated)
    if not (0 < treatment_period < n_periods):
        raise ValueError(
            f"treatment_period={treatment_period} must be in (0, {n_periods})"
        )

    treated_arr = treated.to_numpy(dtype=float)
    donor_arr = donor_pool.to_numpy(dtype=float)
    pre_treated = treated_arr[:treatment_period]
    pre_donor = donor_arr[:treatment_period]
    n_donors = donor_pool.shape[1]

    # Objective: minimise ||pre_treated - pre_donor @ w||²
    def objective(w: np.ndarray) -> float:
        residual = pre_treated - pre_donor @ w
        return float(residual @ residual)

    # Constraint: sum(w) = 1
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    # Bounds: 0 ≤ w_j ≤ 1
    bounds = [(0.0, 1.0)] * n_donors
    # Initial guess: uniform weights
    x0 = np.full(n_donors, 1.0 / n_donors)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = result.x
    # Normalise against numerical drift from SLSQP
    w_sum = float(w.sum())
    if w_sum > 1e-12:
        w = w / w_sum

    # Build synthetic series over ALL periods
    synthetic = donor_arr @ w
    synthetic_series = pd.Series(synthetic, index=treated.index)

    # Pre-treatment RMSE
    pre_synthetic = synthetic[:treatment_period]
    rmse = float(np.sqrt(((pre_treated - pre_synthetic) ** 2).mean()))

    return SyntheticControlResult(
        weights=pd.Series(w, index=donor_pool.columns),
        synthetic_series=synthetic_series,
        pre_treatment_rmse=rmse,
        treated_name=str(treated.name) if treated.name is not None else "treated",
        n_donors=n_donors,
        n_pre=int(treatment_period),
        n_post=int(n_periods - treatment_period),
        converged=bool(result.success),
    )


def compute_treatment_effect(
    result: SyntheticControlResult,
    treated: pd.Series,
) -> pd.Series:
    """Observed minus synthetic per period.

    Args:
        result: SyntheticControlResult from :func:`fit_synthetic_control`.
        treated: The original treated series (must match the index used).

    Returns:
        pd.Series of length len(treated). Pre-treatment values are the
        fit residuals; post-treatment values are the treatment effects.
    """
    if len(treated) != len(result.synthetic_series):
        raise ValueError("treated and synthetic_series must have same length")
    return treated - result.synthetic_series


def placebo_test(
    treated: pd.Series,
    donor_pool: pd.DataFrame,
    treatment_period: int,
    rmse_filter_ratio: float = 5.0,
) -> dict:
    """Placebo inference: re-run the method on every donor as if IT were
    treated, computing the average post-treatment effect for each.

    Following ADH 2010 §3.2, donors whose pre-treatment fit is much worse
    than the original treated unit's fit are excluded from the placebo
    distribution (the rmse_filter_ratio threshold). The treatment effect's
    p-value is the rank of the original effect's magnitude among the
    remaining placebos.

    Args:
        treated: Treated unit's outcome series.
        donor_pool: Donor pool DataFrame.
        treatment_period: First post-treatment index.
        rmse_filter_ratio: Exclude placebos whose pre-treatment RMSE exceeds
            ``rmse_filter_ratio × original_pre_treatment_rmse``. Default 5
            per ADH 2010 (Section "Placebo Studies").

    Returns:
        Dict with:
            - "original_avg_post_effect": treated unit's average post-effect
            - "original_pre_rmse": treated unit's pre-treatment RMSE
            - "placebo_effects": list of (donor_name, avg_post_effect, pre_rmse)
            - "n_placebos_total": int — donors attempted
            - "n_placebos_used": int — donors after RMSE filter
            - "p_value": fraction of placebos with |effect| ≥ |original|
              (two-sided), among RMSE-filtered placebos. NaN if 0 used.
    """
    # Original fit
    original = fit_synthetic_control(treated, donor_pool, treatment_period)
    original_te = compute_treatment_effect(original, treated)
    original_avg_post = float(original_te.iloc[treatment_period:].mean())

    placebo_records: list[tuple[str, float, float]] = []
    donor_names = list(donor_pool.columns)

    for placebo_name in donor_names:
        # Use the placebo donor as the "treated" unit, all OTHER donors as pool
        placebo_treated = donor_pool[placebo_name].rename("placebo")
        new_pool = donor_pool.drop(columns=placebo_name)
        if new_pool.shape[1] < 2:
            continue
        try:
            placebo_result = fit_synthetic_control(
                placebo_treated, new_pool, treatment_period
            )
        except ValueError:
            continue
        placebo_te = compute_treatment_effect(placebo_result, placebo_treated)
        placebo_avg_post = float(placebo_te.iloc[treatment_period:].mean())
        placebo_records.append(
            (placebo_name, placebo_avg_post, placebo_result.pre_treatment_rmse)
        )

    # Filter by RMSE
    if rmse_filter_ratio > 0 and original.pre_treatment_rmse > 0:
        threshold = original.pre_treatment_rmse * rmse_filter_ratio
        used = [r for r in placebo_records if r[2] <= threshold]
    else:
        used = list(placebo_records)

    # Two-sided p-value: fraction with |effect| >= |original|
    if used:
        magnitudes = np.array([abs(r[1]) for r in used])
        original_mag = abs(original_avg_post)
        p_value = float((magnitudes >= original_mag).mean())
    else:
        p_value = float("nan")

    return {
        "original_avg_post_effect": original_avg_post,
        "original_pre_rmse": original.pre_treatment_rmse,
        "placebo_effects": placebo_records,
        "n_placebos_total": len(placebo_records),
        "n_placebos_used": len(used),
        "rmse_filter_ratio": float(rmse_filter_ratio),
        "p_value": p_value,
    }


def in_time_placebo_test(
    treated: pd.Series,
    donor_pool: pd.DataFrame,
    true_treatment_period: int,
    min_pre_periods: int = 20,
) -> dict:
    """In-time placebo: shift the treatment-period to earlier pre-periods,
    fit the synthetic control, and compute the placebo treatment effect at
    each fake date. A real treatment effect should be larger than any of
    the fake-date effects within the pre-treatment window.

    Following ADH 2010 §3.3: this is complementary to the in-space placebo
    in :func:`placebo_test` — it tests whether the original effect size
    is unusual given the strategy's own pre-treatment variability, not
    given the donor pool's variability.

    Args:
        treated: Treated unit's outcome series.
        donor_pool: Donor pool DataFrame.
        true_treatment_period: The ACTUAL treatment date index.
        min_pre_periods: Minimum pre-treatment periods required for each
            fake-treatment fit (so the synthetic control has enough data).
            Fake periods range over ``[min_pre_periods..true_treatment_period-1]``.

    Returns:
        Dict with:
            - "original_avg_post_effect": effect at the true treatment date
            - "fake_effects": list of (fake_period, avg_post_effect)
            - "n_fake_periods_tried": int
            - "p_value": fraction of fake effects with |effect| ≥ |original|
              (NaN if no fake periods could be fit)
    """
    n = len(treated)
    if true_treatment_period >= n:
        raise ValueError(
            f"true_treatment_period={true_treatment_period} ≥ len(treated)={n}"
        )
    if true_treatment_period <= min_pre_periods:
        return {
            "error": (
                f"true_treatment_period={true_treatment_period} ≤ "
                f"min_pre_periods={min_pre_periods} — no room for fake dates"
            ),
            "n_fake_periods_tried": 0,
        }

    # Original effect at the true treatment date
    original = fit_synthetic_control(treated, donor_pool, true_treatment_period)
    original_te = compute_treatment_effect(original, treated)
    original_avg_post = float(original_te.iloc[true_treatment_period:].mean())

    # Sweep fake treatment dates over [min_pre_periods, true_treatment_period)
    fake_effects: list[tuple[int, float]] = []
    for fake_period in range(min_pre_periods, true_treatment_period):
        try:
            fake_result = fit_synthetic_control(treated, donor_pool, fake_period)
        except ValueError:
            continue
        fake_te = compute_treatment_effect(fake_result, treated)
        # Only consider the [fake_period, true_treatment_period) window
        # as the "fake post" — beyond that is the REAL treatment zone.
        fake_post_slice = fake_te.iloc[fake_period:true_treatment_period]
        if len(fake_post_slice) < 1:
            continue
        fake_avg = float(fake_post_slice.mean())
        fake_effects.append((int(fake_period), fake_avg))

    if not fake_effects:
        p_value = float("nan")
    else:
        magnitudes = np.array([abs(e[1]) for e in fake_effects])
        original_mag = abs(original_avg_post)
        p_value = float((magnitudes >= original_mag).mean())

    return {
        "original_avg_post_effect": original_avg_post,
        "true_treatment_period": int(true_treatment_period),
        "fake_effects": fake_effects,
        "n_fake_periods_tried": len(fake_effects),
        "p_value": p_value,
        "min_pre_periods": int(min_pre_periods),
    }


def export_att_chart(
    result: SyntheticControlResult,
    treated: pd.Series,
    output_path: str | pathlib.Path,
    figsize: tuple[float, float] = (10.0, 6.0),
) -> pathlib.Path:
    """Export ATT (Average Treatment effect on the Treated) chart to PNG.

    Plots treated vs synthetic counterfactual over time, with the post-treatment
    gap shaded. Companion to ``output/qa/`` artefacts produced by the forensic
    showcase scripts (parity with equity_curve_audit, survivorship_bias_check,
    out_of_regime_test, etc.).

    Args:
        result: SyntheticControlResult from :func:`fit_synthetic_control`.
        treated: Original treated series (must match result.synthetic_series
            length and index).
        output_path: PNG file path. Parent directories are created.
        figsize: (width, height) inches.

    Returns:
        Resolved pathlib.Path of the saved file.

    Raises:
        ValueError: If treated and synthetic_series lengths differ.
        ImportError: If matplotlib is not installed.
    """
    if len(treated) != len(result.synthetic_series):
        raise ValueError(
            f"treated has {len(treated)} rows, synthetic_series has "
            f"{len(result.synthetic_series)} — must match"
        )
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for ATT chart export. "
            "Install with: pip install matplotlib"
        ) from e

    # F-senior-1 (F-1): Don't hijack the backend if one is already initialised.
    # Headless CI defaults to Agg; this only matters for interactive sessions.
    if matplotlib.get_backend().lower() not in {"agg"}:
        try:
            matplotlib.use("Agg", force=False)
        except Exception:
            # If a backend is already locked, fall through and use whatever is set.
            pass

    treatment_idx = result.n_pre
    treated_arr = treated.to_numpy(dtype=float)
    x = np.arange(len(treated))

    # F-senior-1 (F-4): reuse compute_treatment_effect for ATT (single source of truth).
    te = compute_treatment_effect(result, treated)
    post_te = te.iloc[treatment_idx:]
    att = float(post_te.mean()) if len(post_te) > 0 else float("nan")
    synthetic_arr = result.synthetic_series.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    try:
        ax.plot(
            x,
            treated_arr,
            label=f"Treated ({result.treated_name})",
            linewidth=2,
            color="#c0392b",
        )
        ax.plot(
            x,
            synthetic_arr,
            label="Synthetic Control",
            linewidth=2,
            color="#2c3e50",
            linestyle="--",
        )
        if treatment_idx < len(treated):
            ax.fill_between(
                x[treatment_idx:],
                treated_arr[treatment_idx:],
                synthetic_arr[treatment_idx:],
                alpha=0.25,
                color="#f39c12",
                label="ATT gap",
            )
        ax.axvline(
            treatment_idx,
            color="black",
            linestyle=":",
            alpha=0.7,
            label=f"Treatment (t={treatment_idx})",
        )
        ax.set_xlabel("Period", fontsize=12)
        ax.set_ylabel("Outcome", fontsize=12)
        ax.set_title(
            f"Synthetic Control: {result.treated_name}  "
            f"ATT={att:.4f}  pre-RMSE={result.pre_treatment_rmse:.4f}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        # F-senior-1 (F-2): ensure figure is closed even if savefig raises.
        plt.close(fig)
    return output_path.resolve()


__all__ = [
    "SyntheticControlResult",
    "compute_treatment_effect",
    "export_att_chart",
    "fit_synthetic_control",
    "in_time_placebo_test",
    "placebo_test",
]
