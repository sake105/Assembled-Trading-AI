"""Crisis-Alpha episode labeler using López de Prado triple-barrier method.

Provides `label_crisis_alpha_episodes()` — a QA/research utility that:
1. Reads persisted CrisisStateRecord history (JSONL state log, or a list of records).
2. Extracts ACTIVE-episode intervals (entered_at_utc → next non-ACTIVE record).
3. Runs triple-barrier labeling on each defensive basket symbol for every episode.
4. Returns a tidy DataFrame of labeled crisis episodes for meta-model training.

This module is NOT part of the live trading pipeline; it is called from:
- QA scripts / notebooks
- `scripts/label_crisis_alpha_episodes.py`
- Falsification backtests that need to evaluate whether crisis entries actually helped.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PT_SL = (2.0, 1.0)  # profit-take / stop-loss multiplier of daily vol
DEFAULT_VERTICAL_DAYS = 10  # max holding period aligns with exit_rules max_hold_hours=8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def label_crisis_alpha_episodes(
    prices: pd.DataFrame,
    state_records: list[dict[str, Any]] | None = None,
    state_log_path: Path | str | None = None,
    symbols: list[str] | None = None,
    pt_sl: tuple[float, float] = DEFAULT_PT_SL,
    vertical_barrier_days: int = DEFAULT_VERTICAL_DAYS,
) -> pd.DataFrame:
    """Label historical crisis-alpha ACTIVE episodes with triple-barrier outcomes.

    Args:
        prices: OHLCV DataFrame with DatetimeIndex and symbol columns (wide format,
            or multi-level columns). Must contain at least close prices.
        state_records: Optional list of dicts with keys ``state``, ``entered_at_utc``,
            ``last_evaluated_utc``, ``geo_score_at_entry``, ``reason``.
            Provide this OR ``state_log_path``.
        state_log_path: Path to JSONL file produced by ``save_crisis_state`` (one JSON
            record per line). Alternative to ``state_records``.
        symbols: Which symbols to label. Defaults to defensive basket symbols
            (GLD, TLT, SH, VIXY).
        pt_sl: (profit_take, stop_loss) multipliers for vol-based barriers.
        vertical_barrier_days: Vertical barrier — max days held before forced exit.

    Returns:
        DataFrame with columns:
            episode_id, symbol, entry_time, exit_time, ret, bin (label),
            geo_score_at_entry, reason, episode_duration_days.

        ``bin`` values: +1 (profit-take hit), -1 (stop-loss hit), 0 (time stop).
        Returns empty DataFrame if no ACTIVE episodes found.
    """
    from src.assembled_core.events.crisis_alpha.baskets import get_basket_symbols

    # ------------------------------------------------------------------
    # 1. Load state records
    # ------------------------------------------------------------------
    records = _load_records(state_records, state_log_path)
    if not records:
        log.warning("[CA-LABEL] No state records found — returning empty DataFrame.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 2. Extract ACTIVE episode intervals
    # ------------------------------------------------------------------
    episodes = _extract_active_episodes(records)
    if not episodes:
        log.info("[CA-LABEL] No ACTIVE episodes in records.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 3. Determine symbols to label
    # ------------------------------------------------------------------
    if symbols is None:
        symbols = list(get_basket_symbols())  # all defensive basket symbols

    # ------------------------------------------------------------------
    # 4. Build close-price series per symbol and run triple-barrier
    # ------------------------------------------------------------------
    close_prices = _extract_close_prices(prices, symbols)
    if not close_prices:
        log.warning("[CA-LABEL] No usable price columns found for %s.", symbols)
        return pd.DataFrame()

    from src.assembled_core.features.triple_barrier import triple_barrier_labels

    rows: list[dict[str, Any]] = []

    for ep_id, ep in enumerate(episodes):
        entry_time: datetime = ep["entry_time"]
        exit_time: datetime | None = ep[
            "exit_time"
        ]  # None = still ACTIVE / no exit record
        geo_score = ep.get("geo_score_at_entry", float("nan"))
        reason = ep.get("reason", "")

        for sym, close in close_prices.items():
            # Slice prices from entry through the vertical window
            sym_close = close[close.index >= pd.Timestamp(entry_time)]
            if sym_close.empty:
                continue

            # Use CUSUM event at the exact entry point
            entry_ts = sym_close.index[0]
            events = pd.DatetimeIndex([entry_ts])

            try:
                labeled = triple_barrier_labels(
                    sym_close,
                    events,
                    pt_sl=pt_sl,
                    vertical_barrier_days=vertical_barrier_days,
                )
            except Exception as exc:
                log.debug(
                    "[CA-LABEL] triple_barrier_labels failed for %s @ %s: %s",
                    sym,
                    entry_ts,
                    exc,
                )
                continue

            if labeled.empty:
                continue

            # triple_barrier_labels returns a DataFrame; iloc[0] is a Series.
            row = labeled.iloc[0]
            exit_ts = row.get("t1", pd.NaT)
            ret_val = row.get("ret", float("nan"))
            bin_val = row.get("bin", 0)

            rows.append(
                {
                    "episode_id": ep_id,
                    "symbol": sym,
                    "entry_time": entry_ts,
                    "exit_time": exit_ts,
                    "ret": ret_val,
                    "bin": bin_val,
                    "geo_score_at_entry": geo_score,
                    "reason": reason,
                    "episode_duration_days": (
                        (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)).days
                        if exit_time is not None
                        else None
                    ),
                }
            )

    if not rows:
        log.info(
            "[CA-LABEL] No labeled rows produced (episodes may predate price data)."
        )
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_records(
    state_records: list[dict[str, Any]] | None,
    state_log_path: Path | str | None,
) -> list[dict[str, Any]]:
    if state_records is not None:
        return list(state_records)

    if state_log_path is not None:
        p = Path(state_log_path)
        if not p.exists():
            log.warning("[CA-LABEL] state_log_path does not exist: %s", p)
            return []
        records: list[dict[str, Any]] = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    log.debug("[CA-LABEL] Skipping malformed JSONL line: %s", exc)
        return records

    return []


def _parse_utc(ts_str: str) -> datetime | None:
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _extract_active_episodes(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return list of episode dicts for each ACTIVE interval."""
    episodes: list[dict[str, Any]] = []

    sorted_records = sorted(
        records,
        key=lambda r: r.get("entered_at_utc", "") or r.get("last_evaluated_utc", ""),
    )

    active_start: datetime | None = None
    active_meta: dict[str, Any] = {}

    for rec in sorted_records:
        state = str(rec.get("state", "")).upper()
        evaluated_at = _parse_utc(rec.get("last_evaluated_utc", ""))
        entered_at = _parse_utc(rec.get("entered_at_utc", ""))

        if state == "ACTIVE":
            if active_start is None:
                # New episode begins; geo_score and reason captured at entry only
                # (subsequent ACTIVE evaluations update the state but not the episode meta).
                active_start = entered_at or evaluated_at
                active_meta = {
                    "geo_score_at_entry": rec.get("geo_score_at_entry", float("nan")),
                    "reason": rec.get("reason", ""),
                }
        else:
            if active_start is not None:
                # Episode ended
                episodes.append(
                    {
                        "entry_time": active_start,
                        "exit_time": evaluated_at or entered_at,
                        **active_meta,
                    }
                )
                active_start = None
                active_meta = {}

    # If still ACTIVE at end of records, mark as open episode (exit_time=None)
    if active_start is not None:
        episodes.append({"entry_time": active_start, "exit_time": None, **active_meta})

    return episodes


def _extract_close_prices(
    prices: pd.DataFrame,
    symbols: list[str],
) -> dict[str, pd.Series]:
    """Extract per-symbol close Series from wide or multi-level price DataFrame."""
    result: dict[str, pd.Series] = {}

    if isinstance(prices.columns, pd.MultiIndex):
        # Multi-level: (field, symbol) or (symbol, field)
        for sym in symbols:
            for level_order in [
                ("close", sym),
                (sym, "close"),
                ("Close", sym),
                (sym, "Close"),
            ]:
                if level_order in prices.columns:
                    result[sym] = prices[level_order].dropna()
                    break
    else:
        # Wide format: either symbol columns directly or "SYMBOL_close" columns
        cols_lower = {c.lower(): c for c in prices.columns}
        for sym in symbols:
            if sym in prices.columns:
                result[sym] = prices[sym].dropna()
            elif f"{sym.lower()}_close" in cols_lower:
                result[sym] = prices[cols_lower[f"{sym.lower()}_close"]].dropna()
            elif sym.lower() in cols_lower:
                result[sym] = prices[cols_lower[sym.lower()]].dropna()

    return result
