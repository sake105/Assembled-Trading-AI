"""Earnings calendar integration for signal pipeline (V18).

Provides:
- Pre-earnings signal suppression: mute mean-reversion signals 3 days before earnings
- PEAD (Post-Earnings Announcement Drift): trade in direction of surprise for 20-60 days
- Earnings concentration check: warn if too many positions have upcoming earnings

Reference: Ball & Brown (1968), Bernard & Thomas (1989) — PEAD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class EarningsIntegrationResult:
    """Result of earnings-aware signal adjustments."""

    suppressed_symbols: list[str]  # Signals muted due to upcoming earnings
    pead_signals: dict[str, float]  # Symbol -> PEAD direction (+1/-1 scaled by surprise)
    concentration_warning: bool  # True if too many positions near earnings
    pct_near_earnings: float  # Fraction of portfolio near earnings


def get_days_to_earnings(
    symbols: list[str],
    as_of: pd.Timestamp,
    earnings_calendar: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "report_date",
) -> dict[str, int | None]:
    """Compute days until next earnings for each symbol.

    Args:
        symbols: List of symbols to check.
        as_of: Current date.
        earnings_calendar: DataFrame with symbol and report_date columns.
        symbol_col: Column name for symbol.
        date_col: Column name for earnings date.

    Returns:
        Symbol -> days until next earnings (None if no upcoming earnings known).
    """
    result: dict[str, int | None] = {}
    if earnings_calendar.empty:
        return {s: None for s in symbols}

    cal = earnings_calendar.copy()
    cal[date_col] = pd.to_datetime(cal[date_col])
    as_of_ts = pd.to_datetime(as_of)

    for sym in symbols:
        sym_cal = cal[cal[symbol_col] == sym]
        future = sym_cal[sym_cal[date_col] >= as_of_ts]
        if future.empty:
            result[sym] = None
        else:
            next_date = future[date_col].min()
            result[sym] = int((next_date - as_of_ts).days)

    return result


def suppress_pre_earnings_signals(
    signals: pd.DataFrame,
    days_to_earnings: dict[str, int | None],
    suppress_window: int = 3,
    signal_types_to_suppress: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Zero out mean-reversion signals for symbols near earnings.

    Args:
        signals: Signal DataFrame with symbol and score columns.
        days_to_earnings: Symbol -> days to next earnings.
        suppress_window: Days before earnings to suppress (default: 3).
        signal_types_to_suppress: Signal types to suppress. If None,
            suppresses all signals for near-earnings symbols.

    Returns:
        (adjusted_signals, suppressed_symbols)
    """
    suppressed = []
    result = signals.copy()

    if "symbol" not in result.columns:
        return result, suppressed

    for sym, days in days_to_earnings.items():
        if days is not None and 0 <= days <= suppress_window:
            mask = result["symbol"] == sym
            if mask.any():
                result.loc[mask, "score"] = 0.0
                suppressed.append(sym)

    if suppressed:
        _log.info("PRE-EARNINGS: suppressed signals for %s (within %dd)", suppressed, suppress_window)

    return result, suppressed


def compute_pead_signals(
    symbols: list[str],
    earnings_events: pd.DataFrame,
    as_of: pd.Timestamp,
    pead_window_days: int = 60,
    min_surprise_pct: float = 5.0,
    symbol_col: str = "symbol",
    date_col: str = "timestamp",
) -> dict[str, float]:
    """Compute PEAD direction signals from recent earnings surprises.

    For each symbol with a recent earnings event (within pead_window_days),
    returns a signal in [-1, 1] based on the surprise direction and magnitude.

    Args:
        symbols: Symbols to check.
        earnings_events: Earnings events with eps_actual, eps_estimate.
        as_of: Current date.
        pead_window_days: Days after earnings to maintain PEAD signal.
        min_surprise_pct: Minimum surprise % to generate signal.

    Returns:
        Symbol -> PEAD signal (positive = positive surprise drift).
    """
    pead: dict[str, float] = {}

    if earnings_events.empty:
        return pead

    events = earnings_events.copy()
    events[date_col] = pd.to_datetime(events[date_col])
    as_of_ts = pd.to_datetime(as_of)

    for sym in symbols:
        sym_events = events[events[symbol_col] == sym]
        # Recent earnings within PEAD window
        recent = sym_events[
            (sym_events[date_col] <= as_of_ts)
            & (sym_events[date_col] >= as_of_ts - pd.Timedelta(days=pead_window_days))
        ]

        if recent.empty:
            continue

        # Use most recent earnings
        latest = recent.sort_values(date_col).iloc[-1]

        eps_actual = latest.get("eps_actual")
        eps_estimate = latest.get("eps_estimate")

        if pd.isna(eps_actual) or pd.isna(eps_estimate) or eps_estimate == 0:
            continue

        surprise_pct = (float(eps_actual) - float(eps_estimate)) / abs(float(eps_estimate)) * 100

        if abs(surprise_pct) < min_surprise_pct:
            continue

        # Decay: signal decays linearly over PEAD window
        days_since = (as_of_ts - pd.to_datetime(latest[date_col])).days
        decay = max(0.0, 1.0 - days_since / pead_window_days)

        # Signal: direction * decay * capped magnitude
        magnitude = min(abs(surprise_pct) / 100.0, 1.0)  # Cap at 100% surprise
        direction = 1.0 if surprise_pct > 0 else -1.0

        pead[sym] = round(direction * magnitude * decay, 4)

    if pead:
        _log.info("PEAD signals: %d symbols (pos=%d, neg=%d)",
                   len(pead),
                   sum(1 for v in pead.values() if v > 0),
                   sum(1 for v in pead.values() if v < 0))

    return pead


def check_earnings_concentration(
    portfolio_symbols: list[str],
    days_to_earnings: dict[str, int | None],
    warning_threshold_pct: float = 30.0,
    near_window: int = 5,
) -> tuple[bool, float]:
    """Check if too many portfolio positions have upcoming earnings.

    Args:
        portfolio_symbols: Symbols in portfolio.
        days_to_earnings: Symbol -> days to earnings.
        warning_threshold_pct: Warn if this % of portfolio is near earnings.
        near_window: Days considered "near" earnings.

    Returns:
        (is_concentrated, pct_near_earnings)
    """
    if not portfolio_symbols:
        return False, 0.0

    near_count = sum(
        1 for s in portfolio_symbols
        if days_to_earnings.get(s) is not None and 0 <= days_to_earnings[s] <= near_window
    )
    pct = near_count / len(portfolio_symbols) * 100

    is_concentrated = pct >= warning_threshold_pct
    if is_concentrated:
        _log.warning(
            "EARNINGS CONCENTRATION: %.1f%% of portfolio (%d/%d) within %dd of earnings",
            pct, near_count, len(portfolio_symbols), near_window,
        )

    return is_concentrated, round(pct, 1)


def apply_earnings_integration(
    signals: pd.DataFrame,
    earnings_calendar: pd.DataFrame | None = None,
    earnings_events: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    suppress_window: int = 3,
    pead_window_days: int = 60,
    pead_weight: float = 0.15,
) -> tuple[pd.DataFrame, EarningsIntegrationResult]:
    """Full earnings integration: suppression + PEAD + concentration check.

    Args:
        signals: Signal DataFrame with symbol, score columns.
        earnings_calendar: Future earnings dates.
        earnings_events: Historical earnings with surprises.
        as_of: Current date.
        suppress_window: Days before earnings to suppress.
        pead_window_days: PEAD signal duration.
        pead_weight: Weight of PEAD signal added to score.

    Returns:
        (adjusted_signals, EarningsIntegrationResult)
    """
    if as_of is None:
        as_of = pd.Timestamp.now("UTC")

    symbols = list(signals["symbol"].unique()) if "symbol" in signals.columns else []
    result_info = EarningsIntegrationResult(
        suppressed_symbols=[], pead_signals={},
        concentration_warning=False, pct_near_earnings=0.0,
    )

    if not symbols:
        return signals, result_info

    # Days to earnings
    dte: dict[str, int | None] = {}
    if earnings_calendar is not None and not earnings_calendar.empty:
        dte = get_days_to_earnings(symbols, as_of, earnings_calendar)

    # Pre-earnings suppression
    adjusted, suppressed = suppress_pre_earnings_signals(signals, dte, suppress_window)
    result_info.suppressed_symbols = suppressed

    # PEAD signals
    if earnings_events is not None and not earnings_events.empty:
        pead = compute_pead_signals(symbols, earnings_events, as_of, pead_window_days)
        result_info.pead_signals = pead

        # Add PEAD to scores
        if pead and "score" in adjusted.columns:
            for sym, pead_val in pead.items():
                mask = adjusted["symbol"] == sym
                if mask.any():
                    adjusted.loc[mask, "score"] += pead_weight * pead_val

    # Concentration check
    is_conc, pct = check_earnings_concentration(symbols, dte)
    result_info.concentration_warning = is_conc
    result_info.pct_near_earnings = pct

    return adjusted, result_info


__all__ = [
    "EarningsIntegrationResult",
    "get_days_to_earnings",
    "suppress_pre_earnings_signals",
    "compute_pead_signals",
    "check_earnings_concentration",
    "apply_earnings_integration",
]
