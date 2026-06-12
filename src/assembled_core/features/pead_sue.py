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

The parametrised model framing above applies to :func:`compute_sue` /
:func:`compute_sue_from_expected`. The XBRL-fed live path
(:func:`latest_sue_from_xbrl`) does NOT select a model: it hard-wires a TRUE
``(fp, fy-1)`` fiscal-label join (:func:`quarterly_seasonal_expected`) and bypasses
the positional shift entirely.

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


_PANEL_COLS = ["period_end", "fy", "fp", "eps"]


def build_quarterly_eps_panel(xbrl_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """PIT-correct quarterly diluted-EPS PANEL for one symbol (with fiscal labels).

    From the tall SEC-XBRL frame (output of
    ``altdata_loader.load_fundamentals_xbrl``): coalesce diluted EPS, fall back to
    ``NetIncomeLoss / WeightedAvgDilutedShares`` where the EPS tag is absent for a
    quarter, keep only ~3-month (quarterly) durations for Q1-Q4, and DERIVE
    ``Q4 = FY - (Q1+Q2+Q3)`` when only the FY annual figure is tagged (NEVER
    fabricated when the FY is missing).

    Returns columns ``[period_end, fy, fp, eps]`` (one row per fiscal quarter,
    ascending by ``period_end``). The ``(fy, fp)`` labels let downstream code build
    a TRUE ``(fp, fy-1)`` seasonal comparable (see :func:`quarterly_seasonal_expected`)
    that is robust to gaps — unlike a positional ``shift(4)``.
    """
    # Lazy import keeps the data->features dependency one-directional and avoids
    # any import-order fragility (Rule 50: reuse coalesce_field, don't duplicate).
    from src.assembled_core.data.fundamentals_xbrl_ingest import (  # noqa: PLC0415
        coalesce_field,
    )

    empty = pd.DataFrame(columns=_PANEL_COLS)
    if xbrl_df is None or xbrl_df.empty:
        return empty
    sym = str(symbol).strip().upper()
    sub = xbrl_df[xbrl_df["symbol"].astype(str).str.upper() == sym]
    if sub.empty:
        return empty

    eps = coalesce_field(sub, "eps_diluted")
    ni = coalesce_field(sub, "net_income")
    sh = coalesce_field(sub, "weighted_diluted_shares")

    key_cols = ["period_start", "period_end", "fp", "fy"]
    parts = [d[key_cols] for d in (eps, ni, sh) if not d.empty]
    if not parts:
        return empty
    base = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["period_end"])
        .drop_duplicates(subset=["period_start", "period_end"])
        .reset_index(drop=True)
    )
    base = base.merge(
        eps[["period_start", "period_end", "eps_diluted"]],
        on=["period_start", "period_end"],
        how="left",
    )
    base = base.merge(
        ni[["period_start", "period_end", "net_income"]],
        on=["period_start", "period_end"],
        how="left",
    )
    base = base.merge(
        sh[["period_start", "period_end", "weighted_diluted_shares"]],
        on=["period_start", "period_end"],
        how="left",
    )

    # EPS-from-NetIncome fallback only where the EPS tag is absent for the period.
    # Divide by shares.where(!=0) so a zero/absent denominator yields NaN (never
    # inf) — avoids the pandas `replace`-downcasting FutureWarning entirely.
    eps_v = pd.to_numeric(base["eps_diluted"], errors="coerce")
    ni_v = pd.to_numeric(base["net_income"], errors="coerce")
    sh_v = pd.to_numeric(base["weighted_diluted_shares"], errors="coerce")
    fallback = ni_v / sh_v.where(sh_v != 0)
    base["eps_val"] = eps_v.where(eps_v.notna(), fallback)
    base = base.dropna(subset=["eps_val"])
    if base.empty:
        return empty

    base["period_start"] = pd.to_datetime(base["period_start"], errors="coerce")
    base["period_end"] = pd.to_datetime(base["period_end"], errors="coerce")
    base["_dur"] = (base["period_end"] - base["period_start"]).dt.days
    base["_fp"] = base["fp"].astype(str).str.upper()

    quarterly = base[base["_dur"] <= 100]
    annual = base[base["_dur"] >= 350]

    recs: list[dict] = []
    for _, r in quarterly.iterrows():
        recs.append(
            {
                "period_end": r["period_end"],
                "fy": r["fy"],
                "fp": r["_fp"],
                "eps": float(r["eps_val"]),
            }
        )

    # Derive Q4 = FY - (Q1+Q2+Q3) per fiscal year ONLY when Q4 is not directly
    # tagged AND all three earlier quarters are present (never fabricate).
    for _, fy_row in annual.iterrows():
        fy = fy_row["fy"]
        q = quarterly[quarterly["fy"] == fy]
        fps = set(q["_fp"])
        if "Q4" in fps:
            continue
        if {"Q1", "Q2", "Q3"}.issubset(fps):
            q123 = q[q["_fp"].isin(["Q1", "Q2", "Q3"])]["eps_val"].astype(float).sum()
            recs.append(
                {
                    "period_end": fy_row["period_end"],
                    "fy": fy,
                    "fp": "Q4",
                    "eps": float(fy_row["eps_val"]) - q123,
                }
            )

    if not recs:
        return empty
    panel = pd.DataFrame(recs)
    panel["period_end"] = pd.to_datetime(panel["period_end"])
    panel = (
        panel.drop_duplicates(subset=["period_end"], keep="first")
        .sort_values("period_end")
        .reset_index(drop=True)
    )
    return panel[_PANEL_COLS]


def build_quarterly_eps_series(xbrl_df: pd.DataFrame, symbol: str) -> pd.Series:
    """Quarterly diluted-EPS as a ``period_end``-indexed Series (thin view of the
    panel from :func:`build_quarterly_eps_panel`). Carries the value only; use the
    panel when fiscal ``(fy, fp)`` labels are needed for seasonal alignment."""
    panel = build_quarterly_eps_panel(xbrl_df, symbol)
    if panel.empty:
        return pd.Series(dtype=float, name="eps_diluted")
    s = pd.Series(
        panel["eps"].astype(float).values,
        index=pd.to_datetime(panel["period_end"]),
        name="eps_diluted",
    )
    return s.sort_index()


def quarterly_seasonal_expected(panel: pd.DataFrame) -> pd.Series:
    """TRUE ``(fp, fy-1)`` year-ago same-quarter expected EPS, indexed by period_end.

    Joins on the fiscal labels: ``expected[(fy, fp)] = eps[(fy-1, fp)]``. This is the
    seasonal-RW PEAD baseline done correctly — robust to a missing/extra quarter,
    unlike a positional ``shift(4)`` which silently misaligns when the per-firm
    series has any gap. Observations without a prior-year same-quarter value get
    ``NaN`` (and are dropped by the downstream standardisation).
    """
    if panel is None or panel.empty:
        return pd.Series(dtype=float, name="expected_eps")
    p = panel.dropna(subset=["fy", "fp"]).copy()
    if p.empty:
        return pd.Series(dtype=float, name="expected_eps")
    # Guard the standalone API: keep the output index period_end-unique so it
    # aligns 1:1 in compute_sue_from_expected (the live caller already dedups).
    p = p.drop_duplicates(subset=["period_end"], keep="last")
    prior = p.drop_duplicates(["fy", "fp"], keep="last").set_index(["fy", "fp"])["eps"]
    vals: list[float] = []
    for _, r in p.iterrows():
        try:
            vals.append(float(prior.get((int(r["fy"]) - 1, r["fp"]), float("nan"))))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return pd.Series(vals, index=pd.to_datetime(p["period_end"]), name="expected_eps")


def latest_sue_from_xbrl(
    xbrl_df: pd.DataFrame,
    symbols: list[str],
    *,
    min_quarters: int = 6,
) -> pd.Series:
    """Latest PIT SUE per symbol from the XBRL quarterly-EPS panel.

    Expected EPS is the TRUE ``(fp, fy-1)`` year-ago same-quarter value (seasonal
    RW, joined on fiscal labels via :func:`quarterly_seasonal_expected`), then
    standardised by :func:`compute_sue_from_expected` — NOT a positional
    ``shift(4)`` (which would misalign on any gap). Returns a Series indexed by the
    input ``symbols`` (NaN where history < ``min_quarters``, fewer than 2 aligned
    year-over-year pairs, or σ degenerate). XBRL-fed data path for
    ``_compute_pead_sue_factor`` (live weight is held at 0 pending an OOS backtest).

    NOTE (OOS precondition): :func:`compute_sue_from_expected` standardises with a
    FULL-SAMPLE σ (including the latest event's own forecast error) — acceptable
    for this shadow/logging use, but the eventual OOS backtest MUST standardise
    with an expanding, strictly-past-only σ to avoid a standardiser look-ahead.
    """
    out: dict[str, float] = {}
    for sym in symbols:
        panel = build_quarterly_eps_panel(xbrl_df, sym)
        panel = panel.dropna(subset=["eps", "fy", "fp"]).sort_values("period_end")
        if len(panel) < min_quarters:
            out[sym] = float("nan")
            continue
        actual = pd.Series(
            panel["eps"].astype(float).values,
            index=pd.to_datetime(panel["period_end"]),
        )
        expected = quarterly_seasonal_expected(panel)
        if expected.notna().sum() < 2:
            out[sym] = float("nan")
            continue
        try:
            res = compute_sue_from_expected(actual, expected)
        except ValueError:
            out[sym] = float("nan")
            continue
        sue_clean = res.sue.dropna()
        out[sym] = float(sue_clean.iloc[-1]) if not sue_clean.empty else float("nan")
    return pd.Series(out, name="pead_sue_score", dtype=float)


__all__ = [
    "ExpectedEpsMethod",
    "SueResult",
    "compute_expected_eps_random_walk",
    "compute_expected_eps_seasonal_rw",
    "compute_expected_eps_foster",
    "compute_sue",
    "compute_sue_from_expected",
    "build_quarterly_eps_panel",
    "build_quarterly_eps_series",
    "quarterly_seasonal_expected",
    "latest_sue_from_xbrl",
]
