"""Constants for multifactor_v2 strategy.

Extracted from multifactor_v2.py to avoid magic numbers inline and
to make tuning decisions visible and auditable.

Rule: each constant carries a brief "why" comment so future readers
understand the intent, not just the value.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Temporal / calendar constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR: int = 252
# Standard US equity trading days per calendar year; used for annualisation
# and rolling-window sizing.

CALENDAR_DAYS_PER_YEAR: int = 365
# Calendar days — used when converting between calendar and trading-day counts.

# ---------------------------------------------------------------------------
# Cross-sectional z-score normalisation
# ---------------------------------------------------------------------------

FACTOR_CLIP_MIN: float = -3.0
# Lower bound of the z-score clip applied to each factor.
# ±3 keeps factors within 3 standard deviations, preventing extreme
# outliers from dominating the weighted composite.

FACTOR_CLIP_MAX: float = 3.0
# Upper bound, symmetric with FACTOR_CLIP_MIN.

SMALL_UNIVERSE_THRESHOLD: int = 5
# Universe size below which cross-sectional z-scoring is statistically
# meaningless; rank-normalisation is used instead.

RANK_NORM_SCALE: float = 2.0
# Scale factor in rank-normalisation: maps [0,1] rank → [-1, +1].
# Applied as (rank/n - 0.5) * RANK_NORM_SCALE, consistent with z-score range.

FACTOR_ZERO_VARIANCE_EPS: float = 1e-10
# Any factor with abs-sum below this is treated as zero-variance (dead factor)
# and excluded from the weighted composite to avoid dilution.

STD_ZERO_REPLACE: float = 1e-10
# Threshold below which a factor's cross-sectional std is replaced with NaN
# before dividing, so zero-variance columns yield 0 instead of inf/NaN.

# ---------------------------------------------------------------------------
# Drawdown damper defaults
# ---------------------------------------------------------------------------

DD_MDD_THRESHOLD: float = 0.12
# Maximum drawdown threshold that triggers the damper: 12% from peak.
# Set conservatively — this is a hard exposure limiter, not a soft nudge.

DD_DAMPER_DAYS: int = 30
# Number of trading days the damper stays active after trigger.
# 30 days gives time for volatility to settle; reviewed against 2022 drawdown.

DD_DAMPER_FACTOR: float = 0.5
# Exposure multiplier applied while damper is active.
# 0.5 halves gross exposure, consistent with stress-test targets.

# ---------------------------------------------------------------------------
# Regime weights cache
# ---------------------------------------------------------------------------

REGIME_CACHE_MAX_CONFIGS: int = 4
# Maximum number of distinct config-path entries to keep in the regime cache.
# Kept small: in normal operation there is exactly 1 config path per process.

# ---------------------------------------------------------------------------
# HMM market-return cache
# ---------------------------------------------------------------------------

HMM_CACHE_MAXSIZE: int = 10
# Maximum number of distinct (model_path, symbol_tuple) entries in the
# bounded LRU cache for HMM regime predictions.  Prevents unbounded growth
# in long-running processes that cycle through many universes.

# ---------------------------------------------------------------------------
# VIX exposure caps (4-tier)
# ---------------------------------------------------------------------------

VIX_CAP_EXTREME: float = 0.25  # VIX > 40 — GFC / COVID peak panic
VIX_CAP_CRISIS: float = 0.40  # VIX > 30 — crisis (GFC, COVID March)
VIX_CAP_ELEVATED: float = 0.55  # VIX > 22 — elevated stress (2022 inflation)
VIX_CAP_MILD: float = 0.75  # VIX ≥ 18 — mild caution
# Thresholds match stress-test calibration from 2026-05-05 pre-pilot hardening.

VIX_THRESHOLD_EXTREME: float = 40.0
VIX_THRESHOLD_CRISIS: float = 30.0
VIX_THRESHOLD_ELEVATED: float = 22.0
VIX_THRESHOLD_MILD: float = 18.0

# ---------------------------------------------------------------------------
# Yield-curve inversion cap
# ---------------------------------------------------------------------------

YIELD_CURVE_CAP_DEFAULT: float = 0.60
# Default exposure cap when yield curve is persistently inverted.
# Addresses slow low-VIX bear markets (e.g. 2022) that VIX-cap misses.

YIELD_CURVE_INVERSION_MIN_HISTORY: int = 15
# Minimum number of recent dates needed to evaluate inversion persistence.

YIELD_CURVE_INVERSION_FRACTION: float = 0.65
# Fraction of recent dates that must show inversion (slope < 0) to confirm
# persistent inversion; avoids reacting to transient dips.

YIELD_CURVE_LOOKBACK_DAYS: int = 30
# Rolling window (in trading dates) for persistence check.

# ---------------------------------------------------------------------------
# Geo composite normalisation
# ---------------------------------------------------------------------------

GPR_BASELINE_NORM: float = 50.0
# Rough normalisation for the Caldara-Iacoviello GPR index:
# index ≈ 100 at baseline → z ≈ 2 after dividing by 50.
# Kept simple — no dynamic rescaling to avoid look-ahead bias.

# ---------------------------------------------------------------------------
# Congress factor
# ---------------------------------------------------------------------------

CONGRESS_LOG_AMOUNT_SCALE: float = 10.0
# Divisor applied to log(1 + |amount|) when building the congress activity
# composite.  Prevents large dollar amounts from overwhelming trade count.

# ---------------------------------------------------------------------------
# Safe-divide default
# ---------------------------------------------------------------------------

SAFE_DIVIDE_DEFAULT: float = 0.0
# Default value returned by safe_divide when denominator is effectively zero.
# Zero treats the ratio as neutral, consistent with the fillna(0) policy.

SAFE_DIVIDE_EPS: float = 1e-12
# Denominator below this absolute value is treated as zero in safe_divide.
