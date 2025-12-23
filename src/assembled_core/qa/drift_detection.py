"""Drift detection for features, labels, and performance.

This module provides functions to detect distribution drift in features, labels,
and performance metrics over time. It uses Population Stability Index (PSI) for
distribution comparison and rolling statistics for performance drift detection.

Key features:
- Population Stability Index (PSI) computation for distribution comparison
- Feature drift detection across multiple features
- Label drift detection for binary and multi-class labels
- Performance drift detection using rolling statistics
- Robust handling of edge cases (small samples, constant values, missing data)

Usage:
    >>> from src.assembled_core.qa.drift_detection import (
    ...     compute_psi,
    ...     detect_feature_drift,
    ...     detect_label_drift,
    ...     compute_performance_drift
    ... )
    >>> import pandas as pd
    >>>
    >>> # Feature drift
    >>> base_df = pd.DataFrame({"feature1": [1, 2, 3, 4, 5], "feature2": [10, 20, 30, 40, 50]})
    >>> current_df = pd.DataFrame({"feature1": [10, 20, 30, 40, 50], "feature2": [10, 20, 30, 40, 50]})
    >>> drift_results = detect_feature_drift(base_df, current_df)
    >>>
    >>> # Label drift
    >>> base_labels = pd.Series([0, 0, 0, 1, 1])
    >>> current_labels = pd.Series([0, 1, 1, 1, 1])
    >>> label_drift = detect_label_drift(base_labels, current_labels)
    >>>
    >>> # Performance drift
    >>> equity = pd.Series([10000, 10100, 10200, 10300, 10400, 10500])
    >>> perf_drift = compute_performance_drift(equity, window=3)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_psi(base: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI measures how much a distribution has shifted between a base period
    and a current period. Higher PSI values indicate greater drift.

    PSI interpretation:
    - PSI < 0.1: No significant drift
    - 0.1 <= PSI < 0.2: Moderate drift
    - PSI >= 0.2: Significant drift

    Args:
        base: Base period distribution (reference)
        current: Current period distribution (to compare)
        bins: Number of bins for discretization (default: 10)

    Returns:
        PSI value (float, >= 0). Higher values indicate more drift.

    Example:
        >>> import pandas as pd
        >>> base = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> current = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> psi = compute_psi(base, current)
        >>> assert abs(psi) < 0.01  # Identical distributions → PSI ≈ 0
        >>>
        >>> # Different distributions
        >>> current2 = pd.Series([50, 60, 70, 80, 90, 100, 110, 120, 130, 140])
        >>> psi2 = compute_psi(base, current2)
        >>> assert psi2 > 0.2  # Significant drift
    """
    # Remove NaN values
    base_clean = base.dropna()
    current_clean = current.dropna()

    # Handle edge cases
    if len(base_clean) == 0 or len(current_clean) == 0:
        # If either series is empty, return high PSI (indicates problem)
        return 1.0

    # Check if both series are constant (same value)
    if base_clean.nunique() == 1 and current_clean.nunique() == 1:
        if base_clean.iloc[0] == current_clean.iloc[0]:
            return 0.0  # Same constant value → no drift
        else:
            # Different constant values → maximum drift
            return 1.0

    # Check if one series is constant but the other is not
    if base_clean.nunique() == 1 or current_clean.nunique() == 1:
        # One is constant, other is not → significant drift
        return 0.5  # Moderate-high drift indicator

    # Determine bin edges based on combined distribution
    # Use quantile-based binning for robustness, but include both distributions
    try:
        # Combine both distributions to determine bin edges
        combined = pd.concat([base_clean, current_clean])
        # Create bins based on combined distribution percentiles
        bin_edges = np.quantile(combined, np.linspace(0, 1, bins + 1))
        # Ensure unique bin edges (handle edge case where many values are identical)
        bin_edges = np.unique(bin_edges)

        # If we have fewer unique edges than expected, reduce bins
        if len(bin_edges) < 2:
            # Fallback: use min/max of combined distribution
            bin_edges = np.array([combined.min(), combined.max() + 1e-10])

        # Ensure last edge includes maximum value
        bin_edges[-1] = combined.max() + 1e-10
    except Exception:
        # Fallback: use min/max with equal-width bins (combined range)
        combined = pd.concat([base_clean, current_clean])
        bin_edges = np.linspace(combined.min(), combined.max() + 1e-10, bins + 1)

    # Compute histograms
    base_counts, _ = np.histogram(base_clean, bins=bin_edges)
    current_counts, _ = np.histogram(current_clean, bins=bin_edges)

    # Convert to relative frequencies
    base_freq = base_counts / len(base_clean)
    current_freq = current_counts / len(current_clean)

    # Compute PSI: sum over bins of (current_freq - base_freq) * ln(current_freq / base_freq)
    # Use original frequencies consistently, with epsilon protection only where needed for division by zero
    # This maintains mathematical consistency: all bins use the same frequency basis
    epsilon = 1e-10
    psi = 0.0
    for i in range(len(base_freq)):
        base_is_zero = base_freq[i] == 0
        current_is_zero = current_freq[i] == 0

        if base_is_zero and current_is_zero:
            # Both zero: skip this bin (no contribution to PSI)
            continue
        elif base_is_zero:
            # All current in this bin, none in base → high drift
            # Formula: current_freq * ln(current_freq / epsilon)
            # Use epsilon in denominator to avoid ln(infinity), but use original current_freq
            psi += current_freq[i] * np.log(current_freq[i] / epsilon)
        elif current_is_zero:
            # All base in this bin, none in current → high drift
            # Formula: base_freq * ln(epsilon / base_freq)
            # Use epsilon in numerator, but use original base_freq
            # We want positive contribution, so use absolute value
            psi += base_freq[i] * abs(np.log(epsilon / base_freq[i]))
        else:
            # Both non-zero: standard PSI formula using original frequencies
            psi += (current_freq[i] - base_freq[i]) * np.log(
                current_freq[i] / base_freq[i]
            )

    return max(0.0, psi)  # Ensure non-negative


def detect_feature_drift(
    base_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = 0.2,
    severe_threshold: float = 0.3,
) -> pd.DataFrame:
    """Detect feature drift between base and current feature DataFrames.

    Computes PSI for each common feature column and flags drift severity.

    Args:
        base_df: Base period feature DataFrame (reference)
        current_df: Current period feature DataFrame (to compare)
        psi_threshold: PSI threshold for moderate drift (default: 0.2)
        severe_threshold: PSI threshold for severe drift (default: 0.3)

    Returns:
        DataFrame with columns:
        - feature: Feature name
        - psi: PSI value
        - drift_flag: "NONE", "MODERATE", or "SEVERE"

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> base_df = pd.DataFrame({
        ...     "feature1": np.random.normal(0, 1, 100),
        ...     "feature2": np.random.normal(0, 1, 100)
        ... })
        >>> current_df = pd.DataFrame({
        ...     "feature1": np.random.normal(5, 1, 100),  # Shifted distribution
        ...     "feature2": np.random.normal(0, 1, 100)  # Same distribution
        ... })
        >>> drift_results = detect_feature_drift(base_df, current_df)
        >>> assert len(drift_results) == 2
        >>> assert drift_results[drift_results["feature"] == "feature1"]["drift_flag"].iloc[0] == "SEVERE"
    """
    # Find common columns
    common_cols = set(base_df.columns) & set(current_df.columns)

    if len(common_cols) == 0:
        return pd.DataFrame(columns=["feature", "psi", "drift_flag"])

    results = []

    for col in sorted(common_cols):
        base_series = base_df[col]
        current_series = current_df[col]

        # Compute PSI
        try:
            psi = compute_psi(base_series, current_series)
        except Exception:
            # If PSI computation fails, mark as severe drift
            psi = 1.0

        # Determine drift flag
        if psi >= severe_threshold:
            drift_flag = "SEVERE"
        elif psi >= psi_threshold:
            drift_flag = "MODERATE"
        else:
            drift_flag = "NONE"

        results.append({"feature": col, "psi": psi, "drift_flag": drift_flag})

    return pd.DataFrame(results)


def detect_label_drift(
    base_labels: pd.Series, current_labels: pd.Series, psi_threshold: float = 0.2
) -> dict[str, Any]:
    """Detect label drift between base and current label distributions.

    Compares the distribution of labels (e.g., binary 0/1 or multi-class)
    between base and current periods using PSI.

    Args:
        base_labels: Base period labels (reference)
        current_labels: Current period labels (to compare)
        psi_threshold: PSI threshold for drift detection (default: 0.2)

    Returns:
        Dictionary with:
        - psi: PSI value
        - base_mean: Mean of base labels (for binary: proportion of positive class)
        - current_mean: Mean of current labels
        - mean_shift: Difference in means (current - base)
        - drift_detected: bool (True if PSI >= psi_threshold)
        - drift_severity: "NONE", "MODERATE", or "SEVERE" (based on PSI)

    Example:
        >>> import pandas as pd
        >>>
        >>> # Binary labels with drift
        >>> base_labels = pd.Series([0, 0, 0, 0, 1, 1])  # 33% positive
        >>> current_labels = pd.Series([0, 1, 1, 1, 1, 1])  # 83% positive
        >>> drift = detect_label_drift(base_labels, current_labels)
        >>> assert drift["drift_detected"] is True
        >>> assert drift["mean_shift"] > 0  # More positive labels
    """
    # Remove NaN values
    base_clean = base_labels.dropna()
    current_clean = current_labels.dropna()

    if len(base_clean) == 0 or len(current_clean) == 0:
        return {
            "psi": 1.0,
            "base_mean": None,
            "current_mean": None,
            "mean_shift": None,
            "drift_detected": True,
            "drift_severity": "SEVERE",
        }

    # Compute PSI
    try:
        psi = compute_psi(base_clean, current_clean)
    except Exception:
        psi = 1.0

    # Compute statistics
    base_mean = float(base_clean.mean())
    current_mean = float(current_clean.mean())
    mean_shift = current_mean - base_mean

    # Determine drift severity
    if psi >= 0.3:
        drift_severity = "SEVERE"
    elif psi >= psi_threshold:
        drift_severity = "MODERATE"
    else:
        drift_severity = "NONE"

    drift_detected = psi >= psi_threshold

    return {
        "psi": float(psi),
        "base_mean": base_mean,
        "current_mean": current_mean,
        "mean_shift": mean_shift,
        "drift_detected": drift_detected,
        "drift_severity": drift_severity,
    }


def compute_performance_drift(equity: pd.Series, window: int = 63) -> dict[str, Any]:
    """Detect performance drift in equity curve using rolling statistics.

    Compares performance metrics (rolling Sharpe, average return) between
    the first half and second half of the equity series to detect degradation.

    Args:
        equity: Equity curve series
        window: Rolling window size for Sharpe calculation (default: 63, ~1 quarter)

    Returns:
        Dictionary with:
        - reference_sharpe: Rolling Sharpe in reference period (first half)
        - current_sharpe: Rolling Sharpe in current period (second half)
        - reference_avg_return: Average return in reference period
        - current_avg_return: Average return in current period
        - performance_degrading: bool (True if current performance worse than reference)
        - sharpe_degradation: Difference in Sharpe (current - reference)

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Equity with degrading performance
        >>> np.random.seed(42)
        >>> returns1 = np.random.normal(0.01, 0.01, 100)  # Good: 1% daily return
        >>> returns2 = np.random.normal(-0.005, 0.01, 100)  # Bad: -0.5% daily return
        >>> equity = pd.Series([10000.0])
        >>> for r in list(returns1) + list(returns2):
        ...     equity = pd.concat([equity, pd.Series([equity.iloc[-1] * (1 + r)])])
        >>>
        >>> drift = compute_performance_drift(equity, window=20)
        >>> assert drift["performance_degrading"] is True
    """
    if len(equity) < 2:
        return {
            "reference_sharpe": None,
            "current_sharpe": None,
            "reference_avg_return": None,
            "current_avg_return": None,
            "performance_degrading": False,
            "sharpe_degradation": None,
        }

    # Compute returns
    returns = equity.pct_change().dropna()

    if len(returns) < window:
        # Not enough data for rolling window, use simple mean/std
        if len(returns) < 2:
            return {
                "reference_sharpe": None,
                "current_sharpe": None,
                "reference_avg_return": None,
                "current_avg_return": None,
                "performance_degrading": False,
                "sharpe_degradation": None,
            }

        # Use simple statistics
        reference_avg_return = float(returns.iloc[: len(returns) // 2].mean())
        current_avg_return = float(returns.iloc[len(returns) // 2 :].mean())

        reference_std = float(returns.iloc[: len(returns) // 2].std())
        current_std = float(returns.iloc[len(returns) // 2 :].std())

        reference_sharpe = (
            reference_avg_return / reference_std if reference_std > 0 else 0.0
        )
        current_sharpe = current_avg_return / current_std if current_std > 0 else 0.0

        sharpe_degradation = current_sharpe - reference_sharpe
        performance_degrading = (
            sharpe_degradation < -0.1
        )  # Threshold: -0.1 Sharpe degradation

        return {
            "reference_sharpe": reference_sharpe,
            "current_sharpe": current_sharpe,
            "reference_avg_return": reference_avg_return,
            "current_avg_return": current_avg_return,
            "performance_degrading": performance_degrading,
            "sharpe_degradation": sharpe_degradation,
        }

    # Split into reference (first half) and current (second half)
    mid_point = len(returns) // 2
    reference_returns = returns.iloc[:mid_point]
    current_returns = returns.iloc[mid_point:]

    # Compute rolling Sharpe for each period
    def _rolling_sharpe(returns_series: pd.Series, window_size: int) -> float:
        """Compute average rolling Sharpe ratio."""
        if len(returns_series) < window_size:
            # Fallback: simple Sharpe
            mean_ret = returns_series.mean()
            std_ret = returns_series.std()
            return mean_ret / std_ret if std_ret > 0 else 0.0

        rolling_sharpes = []
        for i in range(window_size, len(returns_series) + 1):
            window_returns = returns_series.iloc[i - window_size : i]
            mean_ret = window_returns.mean()
            std_ret = window_returns.std()
            if std_ret > 0:
                rolling_sharpes.append(mean_ret / std_ret)

        return float(np.mean(rolling_sharpes)) if rolling_sharpes else 0.0

    reference_sharpe = _rolling_sharpe(
        reference_returns, min(window, len(reference_returns))
    )
    current_sharpe = _rolling_sharpe(current_returns, min(window, len(current_returns)))

    # Compute average returns
    reference_avg_return = float(reference_returns.mean())
    current_avg_return = float(current_returns.mean())

    # Determine if performance is degrading
    sharpe_degradation = current_sharpe - reference_sharpe
    # Performance is degrading if Sharpe decreased by more than 0.1 or avg return decreased
    performance_degrading = (
        sharpe_degradation < -0.1
        or current_avg_return < reference_avg_return - 0.001  # 0.1% threshold
    )

    return {
        "reference_sharpe": reference_sharpe,
        "current_sharpe": current_sharpe,
        "reference_avg_return": reference_avg_return,
        "current_avg_return": current_avg_return,
        "performance_degrading": performance_degrading,
        "sharpe_degradation": sharpe_degradation,
    }
