"""Part B wiring: populate TradingContext intel attrs from artifacts.

The trading_cycle intel signal_layer + bayesian_confidence paths read optional
ctx attributes (intel_active_shocks, intel_sector_impacts, signal_historical_scores,
sector_rotation_scores, earnings_calendar). Paper_runner previously never set these,
so the flipped flags silent-skipped.

This helper wires these paths:

1. ``ctx.intel_active_shocks`` — built from news trigger topic_ids via a
   curated topic_id → shock_type map.
2. ``ctx.sector_rotation_scores`` — computed inline from prices (SPDR sector
   ETFs + SPY) when sufficient history exists.
3. ``ctx.earnings_calendar`` — loaded from cached parquet if present.
4. ``ctx.signal_historical_scores`` — loaded from a JSONL rolling cache so
   Bayesian prior uses real history instead of current-cross-section fallback.

Not wired (remains X2 PARK — see docs/intel/T4.5-signal_layer-investigation.md):
- intel_sector_impacts
- intel_supply_vulnerability
- intel_sanctions_beneficiary
- intel_chokepoint_exposure
- intel_confidence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# news trigger topic_id → list of ShockType keys used by SHOCK_BENEFICIARY_MAP
# in intel_signal_adapter. Curated, high-confidence only.
TOPIC_TO_SHOCKS: dict[str, list[str]] = {
    "geopolitical_conflict": ["defense_demand_surge", "global_risk_off"],
    "sanctions_trade": ["global_risk_off", "inflation_spike"],
    "shipping_disruption": ["shipping_cost_risk", "oil_supply_risk"],
    "taiwan_strait": ["semiconductor_supply_risk", "defense_demand_surge", "global_risk_off"],
    "energy_crisis": ["oil_supply_risk", "energy_price_spike"],
    "market_crash": ["global_risk_off"],
    "central_bank": ["rate_shock"],
    "nuclear_risk": ["nuclear_escalation_risk", "global_risk_off"],
}

# Minimum trigger severity to count as an active shock (1 = WATCH, 2 = ACTIVE)
MIN_SHOCK_SEVERITY = 2

# Minimum price history (trading days) required to compute sector scores
MIN_SECTOR_HISTORY_DAYS = 126  # ~6 months

# Max days to keep in rolling historical scores cache
HISTORICAL_SCORES_WINDOW_DAYS = 90

_DEFAULT_EARNINGS_CACHE = "output/intel/earnings/calendar_latest.parquet"
_DEFAULT_HISTORICAL_SCORES = "output/intel/signals/historical_scores.jsonl"


def active_shocks_from_triggers(
    items: list[dict[str, Any]],
    *,
    min_severity: int = MIN_SHOCK_SEVERITY,
) -> list[str]:
    """Extract active shock types from a news triggers list.

    Args:
        items: Raw list of trigger dicts (triggers_latest.json -> items).
        min_severity: Skip triggers below this severity.

    Returns:
        De-duplicated list of shock types in SHOCK_BENEFICIARY_MAP form.
    """
    if not items:
        return []

    shocks: set[str] = set()
    for t in items:
        try:
            sev = int(t.get("severity", 0))
        except (TypeError, ValueError):
            continue
        if sev < min_severity:
            continue
        topic_id = str(t.get("topic_id", "")).strip().lower()
        if not topic_id:
            continue
        mapped = TOPIC_TO_SHOCKS.get(topic_id)
        if mapped:
            shocks.update(mapped)

    return sorted(shocks)


def _populate_active_shocks(ctx: Any, root: Path, news_triggers_path: str | None) -> None:
    triggers_path = (
        Path(news_triggers_path)
        if news_triggers_path
        else root / "output" / "intel" / "news" / "triggers_latest.json"
    )

    active_shocks: list[str] = []
    try:
        if triggers_path.exists():
            data = json.loads(triggers_path.read_text(encoding="utf-8"))
            items = data.get("items", []) if isinstance(data, dict) else []
            active_shocks = active_shocks_from_triggers(items)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "[INTEL-CTX] failed to load news triggers from %s: %s",
            triggers_path,
            exc,
        )

    if active_shocks:
        ctx.intel_active_shocks = active_shocks
        log.info(
            "[INTEL-CTX] populated intel_active_shocks (%d): %s",
            len(active_shocks),
            active_shocks,
        )
    else:
        log.debug("[INTEL-CTX] no active shocks from triggers (empty or low severity)")


def _populate_sector_rotation_scores(ctx: Any) -> None:
    """Compute sector rotation composite scores from ctx.prices and attach
    the latest row at/before as_of to ``ctx.sector_rotation_scores``.

    Silent no-op when:
    - prices missing or empty
    - fewer than MIN_SECTOR_HISTORY_DAYS of data
    - SPY or sector ETFs absent from the universe
    """
    prices = getattr(ctx, "prices", None)
    if prices is None or prices.empty:
        log.debug("[INTEL-CTX] sector_rotation_scores: no prices on ctx")
        return
    if "symbol" not in prices.columns or "close" not in prices.columns:
        log.debug("[INTEL-CTX] sector_rotation_scores: prices missing symbol/close")
        return

    try:
        from src.assembled_core.signals.sector_rotation import (
            SECTOR_ETFS,
            compute_sector_scores,
        )
    except Exception as exc:
        log.debug("[INTEL-CTX] sector_rotation import failed: %s", exc)
        return

    universe = set(prices["symbol"].astype(str).unique())
    available = [e for e in SECTOR_ETFS if e in universe]
    if len(available) < 3 or "SPY" not in universe:
        log.debug(
            "[INTEL-CTX] sector_rotation_scores: insufficient ETF coverage (%d ETFs, SPY=%s)",
            len(available), "SPY" in universe,
        )
        return

    ts_col = "timestamp" if "timestamp" in prices.columns else None
    if ts_col is None:
        log.debug("[INTEL-CTX] sector_rotation_scores: prices missing timestamp")
        return

    sector_df = prices[prices["symbol"].isin(available)].copy()
    spy_df = prices[prices["symbol"] == "SPY"].copy()
    if sector_df.empty or spy_df.empty:
        return

    # Need at least 6 months of data per ETF
    counts = sector_df.groupby("symbol").size()
    if counts.max() < MIN_SECTOR_HISTORY_DAYS:
        log.debug(
            "[INTEL-CTX] sector_rotation_scores: insufficient history (max %d < %d)",
            int(counts.max()), MIN_SECTOR_HISTORY_DAYS,
        )
        return

    try:
        scores_df = compute_sector_scores(sector_df, spy_df)
    except Exception as exc:
        log.debug("[INTEL-CTX] compute_sector_scores failed: %s", exc)
        return

    if scores_df is None or scores_df.empty:
        return

    as_of = getattr(ctx, "as_of", None)
    if as_of is not None and ts_col in scores_df.columns:
        _ts_series = pd.to_datetime(scores_df[ts_col], utc=True, errors="coerce")
        _as_of_ts = pd.Timestamp(as_of)
        if _as_of_ts.tzinfo is None:
            _as_of_ts = _as_of_ts.tz_localize("UTC")
        cut = scores_df[_ts_series <= _as_of_ts]
        if cut.empty:
            return
        last_row = cut.iloc[-1]
    else:
        last_row = scores_df.iloc[-1]

    ctx.sector_rotation_scores = last_row
    score_keys = [k for k in last_row.index if k.endswith("_score")]
    log.info(
        "[INTEL-CTX] populated sector_rotation_scores (%d ETF scores at %s)",
        len(score_keys),
        last_row.get(ts_col, "unknown"),
    )


def _populate_earnings_calendar(
    ctx: Any,
    root: Path,
    earnings_cache_path: str | None,
) -> None:
    """Load earnings calendar from cached parquet if present.

    Cache is refreshed out-of-band via scripts/fetch_earnings_calendar.py.
    Silent no-op if cache missing or unreadable.
    """
    cache_path = (
        Path(earnings_cache_path)
        if earnings_cache_path
        else root / _DEFAULT_EARNINGS_CACHE
    )
    if not cache_path.exists():
        log.debug("[INTEL-CTX] earnings_calendar cache missing: %s", cache_path)
        return

    try:
        cal = pd.read_parquet(cache_path)
    except Exception as exc:
        log.warning("[INTEL-CTX] earnings_calendar load failed: %s", exc)
        return

    if cal is None or cal.empty:
        log.debug("[INTEL-CTX] earnings_calendar cache empty")
        return

    ctx.earnings_calendar = cal
    log.info(
        "[INTEL-CTX] populated earnings_calendar (%d rows, %d symbols)",
        len(cal),
        cal["symbol"].nunique() if "symbol" in cal.columns else 0,
    )


def _populate_historical_scores(
    ctx: Any,
    root: Path,
    historical_scores_path: str | None,
) -> None:
    """Load a rolling cross-sectional mean series from JSONL cache."""
    cache_path = (
        Path(historical_scores_path)
        if historical_scores_path
        else root / _DEFAULT_HISTORICAL_SCORES
    )
    if not cache_path.exists():
        log.debug("[INTEL-CTX] historical_scores cache missing: %s", cache_path)
        return

    try:
        rows: list[dict[str, Any]] = []
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as exc:
        log.warning("[INTEL-CTX] historical_scores read failed: %s", exc)
        return

    if not rows:
        log.debug("[INTEL-CTX] historical_scores cache empty")
        return

    # Each row: {"ts": iso, "mean": float, "n": int}
    # Bayesian needs a Series of cross-sectional means for prior estimation.
    try:
        series = pd.Series(
            [float(r["mean"]) for r in rows if "mean" in r],
            index=pd.to_datetime([r.get("ts") for r in rows if "mean" in r], utc=True, errors="coerce"),
        ).dropna()
    except Exception as exc:
        log.warning("[INTEL-CTX] historical_scores parse failed: %s", exc)
        return

    if series.empty:
        return

    ctx.signal_historical_scores = series
    log.info("[INTEL-CTX] populated signal_historical_scores (%d points)", len(series))


def populate_ctx_from_artifacts(
    ctx: Any,
    root: Path,
    *,
    news_triggers_path: str | None = None,
    earnings_cache_path: str | None = None,
    historical_scores_path: str | None = None,
) -> None:
    """Populate ctx intel attributes from on-disk artifacts.

    Each sub-populator is independently defensive — a failure in one path
    does not block the others. Downstream code gates on ``getattr(ctx, ..., None)``.
    """
    _populate_active_shocks(ctx, root, news_triggers_path)
    _populate_sector_rotation_scores(ctx)
    _populate_earnings_calendar(ctx, root, earnings_cache_path)
    _populate_historical_scores(ctx, root, historical_scores_path)


def persist_historical_scores(
    scores: pd.Series,
    root: Path,
    *,
    historical_scores_path: str | None = None,
    window_days: int = HISTORICAL_SCORES_WINDOW_DAYS,
) -> None:
    """Append current cross-section mean to the rolling JSONL cache.

    Trims entries older than ``window_days`` on each append.
    """
    if scores is None or scores.empty:
        return

    cache_path = (
        Path(historical_scores_path)
        if historical_scores_path
        else root / _DEFAULT_HISTORICAL_SCORES
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mean = float(scores.dropna().mean())
        n = int(scores.dropna().size)
    except Exception as exc:
        log.debug("[INTEL-CTX] historical_scores append skipped: %s", exc)
        return

    now = pd.Timestamp.now("UTC")
    cutoff = now - pd.Timedelta(days=window_days)

    kept: list[dict[str, Any]] = []
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = pd.to_datetime(rec.get("ts"), utc=True, errors="coerce")
                    if pd.isna(ts) or ts < cutoff:
                        continue
                    kept.append(rec)
        except OSError as exc:
            log.warning("[INTEL-CTX] historical_scores rewrite failed: %s", exc)
            return

    kept.append({"ts": now.isoformat(), "mean": mean, "n": n})

    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            for rec in kept:
                fh.write(json.dumps(rec) + "\n")
    except OSError as exc:
        log.warning("[INTEL-CTX] historical_scores write failed: %s", exc)
        return

    log.debug(
        "[INTEL-CTX] historical_scores appended (window=%d, total=%d)",
        window_days,
        len(kept),
    )


__all__ = [
    "TOPIC_TO_SHOCKS",
    "MIN_SHOCK_SEVERITY",
    "MIN_SECTOR_HISTORY_DAYS",
    "HISTORICAL_SCORES_WINDOW_DAYS",
    "active_shocks_from_triggers",
    "populate_ctx_from_artifacts",
    "persist_historical_scores",
]
