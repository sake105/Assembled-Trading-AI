"""Standardized Unexpected Earnings (SUE) with explicit expected-EPS source choice.

Audit C4-083 (KNOWN_ISSUES §8.13) closure: the existing earnings-surprise
factors (`features/altdata_earnings_insider_factors.py`) consume already-
reported `eps_surprise` percentages without exposing the expected-EPS model.
The audit asked: IBES analyst consensus vs Random Walk vs Seasonal RW —
which one is the expected-EPS baseline?

This module makes the choice **explicit and parametrised** — callers select
the expected-EPS model and the SUE result records it:

    SUE_t = (actual_EPS_t − expected_EPS_t) / σ(forecast_error)

Models implemented:

- ``random_walk`` (naive): E[EPS_t] = EPS_{t-1}
- ``seasonal_rw`` (default): E[EPS_t] = EPS_{t-s} where s = seasonality
  (s=4 for quarterly data). Most-cited PEAD baseline (Bernard-Thomas 1989).
- ``foster`` (Foster 1977): E[EPS_t] = EPS_{t-s} + drift, where drift is the
  trailing average of year-over-year quarterly EPS changes. Captures a slow
  growth trend that pure seasonal RW misses.
- ``external``: caller provides `expected_eps` directly (e.g. from IBES
  consensus). Bypasses in-module expectation; just standardises.

IBES analyst-consensus EPS is the academic gold standard for SUE but
requires paid data (Refinitiv / I/B/E/S). When available, pass it via
``compute_sue_from_expected(actual, expected_eps_ibes)``.

**Important on σ (forecast-error standard deviation):** This module computes
σ as the **full-sample standard deviation of forecast errors within a single
input series** (i.e. per firm, non-rolling). Classical Bernard-Thomas (1989)
SUE uses a **rolling 8-quarter per-firm σ** estimated only on PAST forecast
errors to avoid look-ahead. The full-sample σ here is appropriate for
*ex-post research analysis*; for *PIT-safe trading-signal generation* callers
should pre-standardise externally (compute rolling-σ per firm from past
forecast errors only) and feed the standardised series into a downstream
ranking layer rather than relying on this module's σ.

References:
- Bernard, V. L., Thomas, J. K. (1989). *Post-Earnings-Announcement Drift:
  Delayed Price Response or Risk Premium?* JAR 27 Supplement.
- Foster, G. (1977). *Quarterly Accounting Data: Time-Series Properties and
  Predictive-Ability Results*. AR 52(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


ExpectedEpsMethod = Literal["random_walk", "seasonal_rw", "foster", "external"]


@dataclass
class SueResult:
    """Result of a Standardized Unexpected Earnings computation.

    Attributes:
        sue: Series of SUE values per event (= forecast_error / sigma_fe).
            High |SUE| = strong surprise. Index matches input.
        expected_eps: The E[EPS_t] used per event (NaN where insufficient history).
        forecast_error: actual_EPS − expected_EPS per event.
        sigma_forecast_error: Sample std of forecast errors over the full
            input series (per-firm, full-sample, NON-rolling). Single scalar.
            NOTE: classical Bernard-Thomas (1989) SUE uses a rolling per-firm
            σ on PAST forecast errors only (PIT-safe). The full-sample σ here
            is for ex-post research; for PIT-safe signal generation, pre-
            standardise externally and use compute_sue_from_expected.
        n_events: Number of non-NaN events in the result.
        method: Which expected-EPS model was used.
    """

    sue: pd.Series
    expected_eps: pd.Series
    forecast_error: pd.Series
    sigma_forecast_error: float
    n_events: int
    method: ExpectedEpsMethod


def compute_expected_eps_random_walk(eps_series: pd.Series) -> pd.Series:
    """Random-walk expectation: E[EPS_t] = EPS_{t-1}.

    Args:
        eps_series: Reported EPS, indexed by event timestamp (sorted ascending).

    Returns:
        Series of expected EPS, same index. First obs is NaN.
    """
    s = pd.Series(eps_series, dtype=float)
    return s.shift(1).rename("expected_eps_rw")


def compute_expected_eps_seasonal_rw(
    eps_series: pd.Series,
    seasonality: int = 4,
) -> pd.Series:
    """Seasonal random walk: E[EPS_t] = EPS_{t-seasonality}.

    Args:
        eps_series: Reported EPS, indexed by event timestamp (sorted ascending).
        seasonality: Lag in periods (default 4 = same quarter last year for
            quarterly reporters).

    Returns:
        Series of expected EPS, same index. First ``seasonality`` obs are NaN.

    Raises:
        ValueError: If seasonality < 1.
    """
    if seasonality < 1:
        raise ValueError(f"seasonality must be ≥1, got {seasonality}")
    s = pd.Series(eps_series, dtype=float)
    return s.shift(seasonality).rename("expected_eps_seasonal_rw")


def compute_expected_eps_foster(
    eps_series: pd.Series,
    seasonality: int = 4,
    drift_window: int = 4,
) -> pd.Series:
    """Foster (1977) seasonal RW with drift.

    ``E[EPS_t] = EPS_{t-s} + δ_t``  where  ``δ_t = (1/n) Σ_{i=0..n-1} (EPS_{t-1-i} − EPS_{t-s-1-i})``

    The drift δ_t is the trailing average of year-over-year quarterly EPS
    changes — captures slow growth/decline trends that pure seasonal RW
    misses (per Foster 1977 §III, the dominant time-series specification
    in pre-IBES PEAD studies).

    Args:
        eps_series: Reported EPS, indexed ascending.
        seasonality: Year lag (default 4 for quarterly).
        drift_window: Number of past year-over-year diffs to average for
            drift (default 4 ≈ last 4 quarters of YoY change).

    Returns:
        Series of expected EPS. First (seasonality + drift_window) obs are NaN.
    """
    if seasonality < 1:
        raise ValueError(f"seasonality must be ≥1, got {seasonality}")
    if drift_window < 1:
        raise ValueError(f"drift_window must be ≥1, got {drift_window}")
    s = pd.Series(eps_series, dtype=float)
    # YoY diff at each point: EPS_t − EPS_{t-seasonality}
    yoy = s - s.shift(seasonality)
    # Trailing mean of past YoY diffs (excluding current, hence shift(1))
    drift = yoy.shift(1).rolling(drift_window, min_periods=drift_window).mean()
    expected = s.shift(seasonality) + drift
    return expected.rename("expected_eps_foster")


def compute_sue(
    eps_series: pd.Series,
    method: ExpectedEpsMethod = "seasonal_rw",
    seasonality: int = 4,
    drift_window: int = 4,
) -> SueResult:
    """Compute SUE using one of the in-module expected-EPS models.

    Args:
        eps_series: Reported EPS, indexed ascending by event timestamp.
        method: ``"random_walk"`` | ``"seasonal_rw"`` (default) | ``"foster"``.
            Use ``compute_sue_from_expected`` for ``"external"`` (e.g. IBES).
        seasonality: Period lag for seasonal/Foster (default 4 for quarterly).
        drift_window: Foster drift averaging window (default 4).

    Returns:
        SueResult with sue, expected_eps, forecast_error, sigma_forecast_error,
        n_events, method.

    Raises:
        ValueError: If method is not one of the in-module options, or input
            has fewer than ``seasonality + 2`` observations.
    """
    if method == "external":
        raise ValueError(
            "compute_sue: method='external' requires pre-computed expected_eps; "
            "use compute_sue_from_expected(eps_series, expected_eps_external)."
        )
    s = pd.Series(eps_series, dtype=float).dropna()
    if len(s) < seasonality + 2:
        raise ValueError(
            f"compute_sue: need ≥{seasonality + 2} non-NaN obs, got {len(s)}"
        )

    if method == "random_walk":
        expected = compute_expected_eps_random_walk(s)
    elif method == "seasonal_rw":
        expected = compute_expected_eps_seasonal_rw(s, seasonality=seasonality)
    elif method == "foster":
        expected = compute_expected_eps_foster(
            s, seasonality=seasonality, drift_window=drift_window
        )
    else:
        raise ValueError(
            f"compute_sue: unknown method '{method}'. "
            "Use 'random_walk' | 'seasonal_rw' | 'foster' | (external via separate fn)"
        )

    forecast_error = s - expected
    fe_clean = forecast_error.dropna()
    sigma_fe = float(fe_clean.std(ddof=1)) if len(fe_clean) > 1 else float("nan")
    if not np.isfinite(sigma_fe) or sigma_fe <= 0:
        logger.warning(
            "compute_sue: degenerate sigma_forecast_error=%s — returning NaN SUEs",
            sigma_fe,
        )
        sue = pd.Series(np.nan, index=forecast_error.index, name="sue")
    else:
        sue = (forecast_error / sigma_fe).rename("sue")

    return SueResult(
        sue=sue,
        expected_eps=expected,
        forecast_error=forecast_error.rename("forecast_error"),
        sigma_forecast_error=sigma_fe,
        n_events=int(fe_clean.notna().sum()),
        method=method,
    )


def compute_sue_from_expected(
    actual_eps: pd.Series,
    expected_eps: pd.Series,
) -> SueResult:
    """Compute SUE when expected EPS comes from an external source (e.g. IBES consensus).

    Bypasses the in-module expectation models — caller has already computed
    expected EPS from analyst consensus or another external benchmark.
    SUE = (actual − expected) / σ(forecast_error).

    Args:
        actual_eps: Reported EPS, indexed ascending.
        expected_eps: External expected EPS, same index as actual_eps. NaN
            rows in expected are dropped (no forecast available).

    Returns:
        SueResult with method='external'.

    Raises:
        ValueError: If indices mismatch or <2 aligned non-NaN pairs available.
    """
    a = pd.Series(actual_eps, dtype=float)
    e = pd.Series(expected_eps, dtype=float)
    # Align on intersection
    common = a.index.intersection(e.index)
    if len(common) == 0:
        raise ValueError(
            "compute_sue_from_expected: actual_eps and expected_eps share no index"
        )
    a_aligned = a.loc[common]
    e_aligned = e.loc[common]
    mask = a_aligned.notna() & e_aligned.notna()
    if mask.sum() < 2:
        raise ValueError(
            f"compute_sue_from_expected: need ≥2 non-NaN aligned obs, got {int(mask.sum())}"
        )

    forecast_error = (a_aligned - e_aligned).rename("forecast_error")
    fe_clean = forecast_error.dropna()
    sigma_fe = float(fe_clean.std(ddof=1)) if len(fe_clean) > 1 else float("nan")
    if not np.isfinite(sigma_fe) or sigma_fe <= 0:
        sue = pd.Series(np.nan, index=forecast_error.index, name="sue")
    else:
        sue = (forecast_error / sigma_fe).rename("sue")

    return SueResult(
        sue=sue,
        expected_eps=e_aligned.rename("expected_eps"),
        forecast_error=forecast_error,
        sigma_forecast_error=sigma_fe,
        n_events=int(fe_clean.notna().sum()),
        method="external",
    )


__all__ = [
    "ExpectedEpsMethod",
    "SueResult",
    "compute_expected_eps_random_walk",
    "compute_expected_eps_seasonal_rw",
    "compute_expected_eps_foster",
    "compute_sue",
    "compute_sue_from_expected",
]
