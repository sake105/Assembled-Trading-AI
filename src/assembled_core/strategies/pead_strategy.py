"""PEAD (Post-Earnings Announcement Drift) Strategy.

Implements the Bernard & Thomas (1989) PEAD strategy for signal generation:
- On earnings release, compute Standardized Unexpected Earnings (SUE)
- Long top-quintile SUE stocks, short (or avoid) bottom-quintile SUE stocks
- Hold for drift_window calendar days

SUE normalization: uses compute_sue_from_expected (sigma of forecast errors)
when both actual and estimate columns are present (>= 2 common pairs).
Falls back to (actual - estimate) / |estimate| when only one data point exists.
Note: the fallback normalization is on a different scale than the primary path.
Cross-sectional ranking is applied within each normalization path separately
before mixing, so relative ordering is preserved. For research use only — not
suitable for live production without a rolling-sigma PIT-safe variant.

References:
    Bernard, V. L., Thomas, J. K. (1989). Post-Earnings-Announcement Drift:
    Delayed Price Response or Risk Premium? JAR 27 Supplement.

Audit: C2-060
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from assembled_core.features.pead_sue import compute_sue, compute_sue_from_expected

logger = logging.getLogger(__name__)

__all__ = [
    "PEADConfig",
    "generate_pead_signals",
]

_SIGNAL_COLS = [
    "symbol",
    "signal",
    "sue_score",
    "earnings_date",
    "expected_exit_date",
    "confidence",
]

_EMPTY_SIGNALS = pd.DataFrame(columns=_SIGNAL_COLS)


@dataclass
class PEADConfig:
    """Configuration for PEAD signal generation.

    Attributes:
        drift_window: Calendar days to hold post-earnings position (default 60).
        top_quintile_pct: Fraction of symbols with highest SUE to long (default 0.20).
        bottom_quintile_pct: Fraction of symbols with lowest SUE to short/avoid (default 0.20).
        min_sue_abs: Minimum |SUE| below which signals are treated as noise (default 0.5).
        long_only: If True, bottom-quintile results in signal=0, not -1 (default True).
        sue_method: EPS-expectation model passed to compute_sue (default "seasonal_rw").
    """

    drift_window: int = 60
    top_quintile_pct: float = 0.20
    bottom_quintile_pct: float = 0.20
    min_sue_abs: float = 0.5
    long_only: bool = True
    sue_method: str = "seasonal_rw"
    _extra: dict = field(default_factory=dict, init=False, repr=False)


def _compute_sue_for_symbol(
    sym_df: pd.DataFrame,
    config: PEADConfig,
) -> pd.Series:
    """Return a Series mapping earnings_date → sue_score for one symbol.

    Requires the FULL PIT-safe earnings history for the symbol (not just the
    drift-window slice) so that sigma estimation uses sufficient observations.

    Uses external estimate if eps_estimate is available; else falls back to
    compute_sue() on eps_actual history.  Returns empty Series on failure.
    """
    sym_df = sym_df.sort_values("earnings_date")

    # If we have both actual and external estimate, prefer external path
    has_estimate = (
        "eps_estimate" in sym_df.columns and sym_df["eps_estimate"].notna().any()
    )
    has_actual = "eps_actual" in sym_df.columns and sym_df["eps_actual"].notna().any()

    if has_actual and has_estimate:
        actual = sym_df.set_index("earnings_date")["eps_actual"].dropna()
        estimate = sym_df.set_index("earnings_date")["eps_estimate"].reindex(
            actual.index
        )
        common_mask = actual.notna() & estimate.notna()
        if common_mask.sum() >= 2:
            try:
                result = compute_sue_from_expected(
                    actual[common_mask], estimate[common_mask]
                )
                return result.sue
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "[PEAD] compute_sue_from_expected failed for symbol — %s", exc
                )

    # Fallback: in-module SUE on eps_actual history
    if has_actual:
        actual = sym_df.set_index("earnings_date")["eps_actual"].dropna()
        # Need enough history for the chosen method
        min_obs = 6  # seasonal_rw needs ≥ seasonality+2 = 6 for default seasonality=4
        if len(actual) >= min_obs:
            try:
                result = compute_sue(actual, method=config.sue_method)  # type: ignore[arg-type]
                return result.sue
            except Exception as exc:  # noqa: BLE001
                logger.debug("[PEAD] compute_sue fallback failed — %s", exc)

    # Single-event shortcut: use eps_estimate if only one row
    # SUE = (actual - estimate) / |estimate| (simple normalization — different scale
    # from the sigma-normalized path above; only comparable within this path)
    if has_actual and has_estimate:
        merged = sym_df.dropna(subset=["eps_actual", "eps_estimate"])
        if len(merged) >= 1:
            sue_scores: list[tuple[pd.Timestamp, float]] = []
            for _, row in merged.iterrows():
                est = float(row["eps_estimate"])
                act = float(row["eps_actual"])
                denom = abs(est) if abs(est) > 1e-9 else 1.0
                sue_scores.append((row["earnings_date"], (act - est) / denom))
            if sue_scores:
                dates, values = zip(*sue_scores)
                return pd.Series(
                    list(values), index=pd.DatetimeIndex(list(dates)), name="sue"
                )

    return pd.Series(dtype=float, name="sue")


def generate_pead_signals(
    earnings_df: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    config: PEADConfig | None = None,
) -> pd.DataFrame:
    """Generate PEAD long/short signals from earnings surprises.

    PIT-safe: only uses earnings with earnings_date <= as_of.
    Signal is generated for earnings events within the drift window:
        as_of - drift_window days <= earnings_date <= as_of

    SUE computation uses the full PIT-safe history per symbol (not just the
    drift-window slice) so that sigma estimation is numerically valid.

    Args:
        earnings_df: Must contain columns ``symbol``, ``earnings_date``, ``eps_actual``.
            Optional: ``eps_estimate``.
        prices: Daily price panel with columns ``date``, ``symbol``, ``close``.
            Currently used to verify the drift-window context; not required for signal logic.
        as_of: PIT cutoff timestamp.
        config: PEAD strategy configuration.  Defaults to PEADConfig() if None.

    Returns:
        DataFrame with columns:
            symbol, signal (1 or -1), sue_score (float),
            earnings_date (pd.Timestamp), expected_exit_date (pd.Timestamp),
            confidence (float in [0, 1]).
        Only rows with a non-zero signal are returned.
        Empty DataFrame with correct columns when no valid signals are found.
    """
    if config is None:
        config = PEADConfig()

    # --- guard empty input ---
    if earnings_df is None or earnings_df.empty:
        return _EMPTY_SIGNALS.copy()

    required_cols = {"symbol", "earnings_date", "eps_actual"}
    missing = required_cols - set(earnings_df.columns)
    if missing:
        logger.warning("[PEAD] Missing required columns: %s", missing)
        return _EMPTY_SIGNALS.copy()

    df = earnings_df.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    # PIT-safe: exclude future earnings (full history for SUE computation)
    as_of_ts = pd.Timestamp(as_of)
    pit_safe_df = df[df["earnings_date"] <= as_of_ts]

    # Identify which symbols have earnings within the drift window
    window_start = as_of_ts - pd.Timedelta(days=config.drift_window)
    windowed_symbols = set(
        pit_safe_df.loc[pit_safe_df["earnings_date"] >= window_start, "symbol"].unique()
    )

    if not windowed_symbols:
        return _EMPTY_SIGNALS.copy()

    # --- Compute SUE per symbol using full PIT-safe history ---
    # Then select the most-recent windowed event's SUE value for ranking.
    sue_rows: list[dict] = []

    for symbol, sym_df in pit_safe_df.groupby("symbol", sort=False):
        if symbol not in windowed_symbols:
            continue

        sue_series = _compute_sue_for_symbol(sym_df, config)

        # Most-recent windowed event date for this symbol
        sym_windowed = sym_df[sym_df["earnings_date"] >= window_start]
        if sym_windowed.empty:
            continue
        latest_date = sym_windowed["earnings_date"].max()

        if sue_series.empty or latest_date not in sue_series.index:
            # No SUE available — drop this symbol (no fallback NaN rows)
            logger.debug("[PEAD] No SUE computed for %s at %s", symbol, latest_date)
            continue

        sue_val = float(sue_series.loc[latest_date])
        sue_rows.append(
            {
                "symbol": symbol,
                "earnings_date": latest_date,
                "sue_score": sue_val,
            }
        )

    if not sue_rows:
        return _EMPTY_SIGNALS.copy()

    signals_df = pd.DataFrame(sue_rows)
    signals_df = signals_df.dropna(subset=["sue_score"])

    # Warn when rows from different normalization paths are pooled.
    # Path 1 (sigma-normalized, compute_sue_from_expected) and Path 3
    # (raw (actual-estimate)/|estimate|) produce SUE values on different
    # scales. Cross-sectional ranking across mixed paths may produce
    # non-comparable relative ordering. Research-tier only.
    if "sue_path" in signals_df.columns:
        n_paths = signals_df["sue_path"].nunique()
        if n_paths > 1:
            logger.warning(
                "[PEAD] Cross-sectional ranking pools %d different SUE "
                "normalization paths — relative ordering may be unreliable.",
                n_paths,
            )

    if signals_df.empty:
        return _EMPTY_SIGNALS.copy()

    # --- Minimum |SUE| filter ---
    signals_df = signals_df[signals_df["sue_score"].abs() >= config.min_sue_abs]

    if signals_df.empty:
        return _EMPTY_SIGNALS.copy()

    # --- Cross-sectional ranking and quintile assignment ---
    # Thin-universe guard: if fewer symbols than needed for quintile split,
    # use absolute SUE sign instead of percentile ranking to avoid spurious signals.
    n_sym = len(signals_df)
    min_for_quintile = max(2, math.ceil(1.0 / config.top_quintile_pct))

    signals_df["signal"] = 0

    if n_sym >= min_for_quintile:
        ranks = signals_df["sue_score"].rank(pct=True)
        top_mask = ranks >= (1.0 - config.top_quintile_pct)
        bot_mask = ranks <= config.bottom_quintile_pct
    else:
        logger.warning(
            "[PEAD] Only %d symbols qualify (min for quintile: %d) — "
            "using absolute SUE direction instead of percentile ranking.",
            n_sym,
            min_for_quintile,
        )
        top_mask = signals_df["sue_score"] > 0
        bot_mask = signals_df["sue_score"] < 0

    signals_df.loc[top_mask, "signal"] = 1
    if not config.long_only:
        signals_df.loc[bot_mask, "signal"] = -1

    # --- Confidence: |SUE| normalised to [0, 1] ---
    sue_vals = signals_df["sue_score"].values
    max_abs_sue = float(np.abs(sue_vals).max())
    if max_abs_sue > 0:
        signals_df["confidence"] = (signals_df["sue_score"].abs() / max_abs_sue).clip(
            0.0, 1.0
        )
    else:
        signals_df["confidence"] = 0.0

    # --- Expected exit date ---
    signals_df["expected_exit_date"] = signals_df["earnings_date"] + pd.Timedelta(
        days=config.drift_window
    )

    # Return only rows with a non-zero signal
    result = signals_df[signals_df["signal"] != 0][_SIGNAL_COLS].copy()
    result = result.sort_values("sue_score", ascending=False).reset_index(drop=True)
    return result
