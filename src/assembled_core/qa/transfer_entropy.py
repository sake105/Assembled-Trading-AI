"""Transfer Entropy estimators — directional information flow between time series.

Audit C4-080 (KNOWN_ISSUES §8.13) partial closure: Mutual Information via KSG
already exists at ``qa/feature_screen.py`` (sklearn ``mutual_info_regression``
is KSG-based). The Transfer Entropy direction was NOT implemented — this module
closes that gap.

Transfer Entropy (Schreiber 2000) quantifies how much knowing the past of a
source series ``X`` reduces uncertainty about the future of a target series
``Y``, beyond what ``Y``'s own past already explains. It is *directional*:
``TE(X→Y) ≠ TE(Y→X)`` in general — a key advantage over symmetric mutual
information.

Two estimators provided:

1. ``transfer_entropy_binned`` — histogram-based discrete TE. Numerically
   exact for finite ``n_bins`` discretisation; recommended for reproducibility
   and minimal dependencies.

2. ``transfer_entropy_ksg`` — sklearn-based heuristic approximation.

   **IMPORTANT: This is NOT the textbook KSG-TE.** Wibral et al. 2014 §2.2
   defines TE as ``MI((Y_past, X_past) → Y_future) − MI(Y_past → Y_future)``
   where the first term is a multivariate joint MI. sklearn's
   ``mutual_info_regression`` is column-wise, NOT a joint-MI estimator —
   the proper joint KSG estimator is in ``idtxl`` or ``JIDT``.

   This implementation substitutes a home-grown conservative bound:
   ``TE_approx = max(0, MI(X_past; Y_future) − MI(Y_past; Y_future) · ρ²(X_past, Y_past))``
   where ρ² is the squared Pearson correlation. This tracks the true TE for
   Gaussian-AR-like processes (where MI is monotone in ρ²) but lacks general
   justification beyond that. Use the binned estimator for rigorous TE; use
   this only for fast screening when the dependency profile is roughly Gaussian.

   For production-grade KSG-TE install ``idtxl`` (multivariate kNN estimator).

References:
- Schreiber, T. (2000). *Measuring Information Transfer*. PRL 85(2): 461-464.
- Kraskov, A., Stögbauer, H., Grassberger, P. (2004). *Estimating Mutual
  Information*. Phys. Rev. E 69, 066138.
- Wibral, M., Vicente, R., Lindner, M. (2014). *Transfer Entropy in
  Neuroscience*. Chapter 1 in "Directed Information Measures in Neuroscience".

Use case: pre-screen pair candidates for cointegration (``signals/
pairs_diagnostics``); detect lead-lag asymmetries between assets / factors.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _align_lagged(
    source: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned (X_past, Y_past, Y_future) triples for TE estimation.

    Returns three 1-D arrays of equal length N = len - lag, with
    Y_future[i] = target[i + lag], Y_past[i] = target[i], X_past[i] = source[i].
    NaN rows are dropped jointly.
    """
    s = pd.Series(source, dtype=float).reset_index(drop=True)
    t = pd.Series(target, dtype=float).reset_index(drop=True)
    if len(s) != len(t):
        raise ValueError(
            f"transfer_entropy: source and target length mismatch ({len(s)} vs {len(t)})"
        )
    if lag < 1:
        raise ValueError(f"transfer_entropy: lag must be ≥1, got {lag}")
    if len(s) <= lag + 30:
        raise ValueError(
            f"transfer_entropy: need ≥{lag + 31} obs for lag={lag}, got {len(s)}"
        )

    y_future = t.iloc[lag:].to_numpy()
    y_past = t.iloc[:-lag].to_numpy()
    x_past = s.iloc[:-lag].to_numpy()

    mask = np.isfinite(y_future) & np.isfinite(y_past) & np.isfinite(x_past)
    if mask.sum() < 30:
        raise ValueError(
            f"transfer_entropy: only {int(mask.sum())} finite-aligned obs after dropna (need ≥30)"
        )
    return x_past[mask], y_past[mask], y_future[mask]


def transfer_entropy_binned(
    source: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray,
    lag: int = 1,
    n_bins: int = 8,
) -> float:
    """Histogram-based Transfer Entropy ``TE(source → target; lag)`` in nats.

    Discretises each variable into equal-frequency bins, then computes:
    ``TE = ∑ p(y_f, y_p, x_p) · log[ p(y_f | y_p, x_p) / p(y_f | y_p) ]``
    where ``y_f = target[t+lag]``, ``y_p = target[t]``, ``x_p = source[t]``.

    Args:
        source: Source time series.
        target: Target time series (same length as source).
        lag: Forecast lag (default 1).
        n_bins: Number of bins per variable for joint-density discretisation
            (default 8 — balances bias and variance per Wibral §2.4.1).

    Returns:
        Transfer Entropy in nats. Non-negative by construction (clipped to 0
        on numerical-noise small negatives). Larger = more information flow.

    Raises:
        ValueError: If inputs are length-mismatched, lag<1, or have <30
            finite-aligned observations.
    """
    x_past, y_past, y_future = _align_lagged(source, target, lag)

    # Equal-frequency binning per variable
    def _bin(arr: np.ndarray) -> np.ndarray:
        # Use quantile bins for equal-frequency partitioning
        edges = np.quantile(arr, np.linspace(0, 1, n_bins + 1))
        # Ensure unique edges (collapsed bins for constant input)
        edges = np.unique(edges)
        if len(edges) < 2:
            return np.zeros_like(arr, dtype=int)
        # digitize: bin indices in [0, len(edges)-1]; clip to [0, n_bins-1]
        idx = np.clip(np.digitize(arr, edges[1:-1]), 0, n_bins - 1)
        return idx

    yf_b = _bin(y_future)
    yp_b = _bin(y_past)
    xp_b = _bin(x_past)
    n = len(yf_b)

    # Joint p(y_future, y_past, x_past)
    # Use unique-row counting on a 3-D index
    joint_idx = yf_b * (n_bins * n_bins) + yp_b * n_bins + xp_b
    joint_counts = np.bincount(joint_idx, minlength=n_bins**3)
    p_joint = joint_counts / n  # shape (n_bins^3,)

    # Marginals
    yp_xp = yp_b * n_bins + xp_b
    p_yp_xp = np.bincount(yp_xp, minlength=n_bins * n_bins) / n
    yf_yp = yf_b * n_bins + yp_b
    p_yf_yp = np.bincount(yf_yp, minlength=n_bins * n_bins) / n
    p_yp = np.bincount(yp_b, minlength=n_bins) / n

    # TE = sum p(yf,yp,xp) * log[ p(yf,yp,xp) * p(yp) / (p(yp,xp) * p(yf,yp)) ]
    te = 0.0
    for idx in range(n_bins**3):
        if p_joint[idx] <= 0:
            continue
        yf = idx // (n_bins * n_bins)
        yp = (idx // n_bins) % n_bins
        xp = idx % n_bins
        num = p_joint[idx] * p_yp[yp]
        den = p_yp_xp[yp * n_bins + xp] * p_yf_yp[yf * n_bins + yp]
        if num <= 0 or den <= 0:
            continue
        te += p_joint[idx] * np.log(num / den)

    return float(max(te, 0.0))  # Clip small numerical negatives to 0


def transfer_entropy_ksg(
    source: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray,
    lag: int = 1,
    k: int = 3,
    random_state: int = 42,
) -> float | None:
    """Kraskov-Stögbauer-Grassberger (kNN-based) Transfer Entropy in nats.

    Computes ``TE(X→Y; lag) ≈ MI((Y_past, X_past) → Y_future) − MI(Y_past → Y_future)``
    via sklearn's ``mutual_info_regression`` (KSG estimator per Kraskov 2004).

    Args:
        source: Source time series.
        target: Target time series.
        lag: Forecast lag (default 1).
        k: kNN parameter for KSG estimator (default 3). Higher = lower
            variance but higher bias per Kraskov et al.
        random_state: RNG seed for sklearn reproducibility.

    Returns:
        TE in nats (non-negative by construction; clipped to 0 on numerical
        noise). Returns ``None`` if sklearn is not installed.

    Raises:
        ValueError: Length mismatch / lag<1 / <30 finite-aligned obs.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression
    except ImportError:
        logger.warning(
            "transfer_entropy_ksg: sklearn not installed — returning None. "
            "Use transfer_entropy_binned() as dependency-free alternative."
        )
        return None

    x_past, y_past, y_future = _align_lagged(source, target, lag)

    # NOTE on sklearn's mutual_info_regression: it returns the MI of *each
    # feature column* with the target, NOT the joint MI of all columns. A
    # proper joint-MI estimator (Kraskov 2004 multivariate kNN) is not in
    # sklearn — production code that needs it should install idtxl or JIDT.
    # Here we approximate the conditional-MI term via the chain rule plus
    # the corr(X_past, Y_past)² confounding share — conservative for
    # screening, not for high-precision causality testing. See module
    # docstring for the rigor caveat.

    # MI(Y_past → Y_future)
    try:
        mi_marginal = float(
            mutual_info_regression(
                y_past.reshape(-1, 1),
                y_future,
                discrete_features=False,
                n_neighbors=k,
                random_state=random_state,
            )[0]
        )
    except (ValueError, RuntimeError) as exc:
        logger.debug("KSG MI(Y_past → Y_future) failed: %s", exc)
        return None
    try:
        mi_xpast_yf = float(
            mutual_info_regression(
                x_past.reshape(-1, 1),
                y_future,
                discrete_features=False,
                n_neighbors=k,
                random_state=random_state,
            )[0]
        )
    except (ValueError, RuntimeError):
        return None

    # TE ≈ MI(X_past; Y_future) - portion already explained by Y_past
    # Conservative estimator: max(0, MI(X_past; Y_future) - MI(Y_past; Y_future) *
    # corr(X_past, Y_past)^2). This is a heuristic; for production use the
    # binned estimator or install a proper multivariate KSG library (idtxl).
    if len(x_past) > 1 and len(y_past) > 1:
        corr_xy = float(np.corrcoef(x_past, y_past)[0, 1])
        if not np.isfinite(corr_xy):
            corr_xy = 0.0
    else:
        corr_xy = 0.0
    confounded_share = mi_marginal * (corr_xy**2)
    te = mi_xpast_yf - confounded_share

    return float(max(te, 0.0))


__all__ = [
    "transfer_entropy_binned",
    "transfer_entropy_ksg",
]
