"""Caldara-Iacoviello GPR (Geopolitical Risk) index merge helpers.

Merges the monthly GPR series into a daily per-symbol panel so that
``_compute_geo_risk_composite`` (multifactor_v2 factor 31) can read
``gpr_index`` from the live ``latest`` DataFrame (Path 1) instead of
falling through to zero-fill.

Source:
    Caldara & Iacoviello, "Measuring Geopolitical Risk".
    https://www.matteoiacoviello.com/gpr.htm — free, monthly, updated
    around the 1st of each month with prior-month values.

Producer:
    ``scripts/ops/fetch_caldara_iacoviello_gpr.py`` — writes a tidy
    parquet to ``output/macro_gpr.parquet`` with at least the columns
    ``timestamp`` (month-start, UTC-aware) and ``gpr_index`` (float).

PIT semantics:
    GPR for month *t* is timestamped at month-start of *t* in the
    Caldara-Iacoviello parquet, but the actual public release happens
    during month *t+1* (typically early-mid month). A naive
    ``merge_asof(direction="backward")`` therefore leaks future data:
    a backtest bar dated 2024-02-01 would see the 2024-02-01 GPR value
    which is not public until mid-March 2024.

    Defense: ``merge_gpr_index_into_panel`` accepts ``release_lag_days``
    (default 32) which shifts each GPR timestamp forward by that delta
    before the asof-merge. With the default, the Feb-2024 value
    (stamped 2024-02-01) becomes "publishable" at 2024-03-04 and is
    invisible to any panel row dated before that. Backfill-style
    callers (e.g. parity tests or research notebooks that knowingly
    use raw month-start values) can pass ``release_lag_days=0``.

See §9.9 in ``KNOWN_ISSUES.md`` and §6.1 (sensitive zone notes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_GPR_COLUMN = "gpr_index"
_TIMESTAMP_COLUMN = "timestamp"


def load_gpr_series(gpr_path: Path | str) -> pd.DataFrame | None:
    """Read the GPR parquet and return a minimal sorted ``[timestamp, gpr_index]`` frame.

    Returns ``None`` if the file does not exist or lacks the required
    columns. Logs at ``debug`` when the file is missing (the trading
    cycle calls this every bar and we don't want spam) and at
    ``warning`` when the file is present but malformed.
    """
    p = Path(gpr_path)
    if not p.exists():
        logger.debug("[MACRO-GPR] %s not found — skipping merge", p)
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as exc:
        logger.warning("[MACRO-GPR] failed to read %s: %s", p, exc)
        return None

    required = {_TIMESTAMP_COLUMN, _GPR_COLUMN}
    missing = required - set(df.columns)
    if missing:
        logger.warning(
            "[MACRO-GPR] %s missing required columns %s — skipping merge",
            p,
            sorted(missing),
        )
        return None

    out = df[[_TIMESTAMP_COLUMN, _GPR_COLUMN]].copy()
    out[_TIMESTAMP_COLUMN] = pd.to_datetime(out[_TIMESTAMP_COLUMN], utc=True)
    out = (
        out.dropna(subset=[_TIMESTAMP_COLUMN])
        .sort_values(_TIMESTAMP_COLUMN)
        .drop_duplicates(_TIMESTAMP_COLUMN, keep="last")
        .reset_index(drop=True)
    )
    return out


def merge_gpr_index_into_panel(
    panel: pd.DataFrame,
    gpr_path: Path | str = "output/macro_gpr.parquet",
    release_lag_days: int = 32,
) -> pd.DataFrame:
    """Add the ``gpr_index`` column to a per-symbol price/feature panel.

    Args:
        panel: Per-symbol panel with at least a ``timestamp`` column. Row
            order is preserved (multi-symbol panels typically arrive
            sorted by ``(symbol, timestamp)``; we do not assume that).
        gpr_path: Path to the parquet produced by the Caldara-Iacoviello
            feeder. Default ``output/macro_gpr.parquet``.
        release_lag_days: Number of days to shift each GPR timestamp
            forward before the asof-merge, to model the
            Caldara-Iacoviello publication delay (month *t* values
            release during month *t+1*). Default 32 is a conservative
            upper bound. Pass 0 only for parity tests / research
            notebooks that knowingly use raw month-start values.

    Returns:
        ``panel`` (a copy) with ``gpr_index`` attached. If the GPR file
        is missing, malformed, or the panel already carries ``gpr_index``,
        the panel is returned unchanged. Panels lacking a ``timestamp``
        column or containing only NaT timestamps are also returned
        unchanged (with a warning log in the latter case).

    PIT-safety: see module docstring. ``release_lag_days`` defaults
    high enough to prevent the most common look-ahead failure mode.
    """
    if panel is None or panel.empty or _TIMESTAMP_COLUMN not in panel.columns:
        return panel

    # Don't clobber if the column is already present (e.g. another upstream
    # step already merged GPR). Caller can drop the column to force re-merge.
    if _GPR_COLUMN in panel.columns:
        nan_pct = panel[_GPR_COLUMN].isna().mean() * 100.0
        logger.debug(
            "[MACRO-GPR] panel already has %s (nan%%=%.1f) — skipping merge",
            _GPR_COLUMN,
            nan_pct,
        )
        return panel

    gpr = load_gpr_series(gpr_path)
    if gpr is None or gpr.empty:
        return panel

    # Apply the PIT release-lag shift BEFORE the merge so the asof
    # operation sees publishable timestamps, not raw month-start ones.
    if release_lag_days:
        gpr = gpr.copy()
        gpr[_TIMESTAMP_COLUMN] = gpr[_TIMESTAMP_COLUMN] + pd.Timedelta(
            days=release_lag_days
        )

    # Guard merge_asof against NaT in the panel timestamp column —
    # merge_asof raises ValueError on null left keys. Caller (Step 2.18
    # in _tc_features) would catch via _warn_once_feature_skip, but a
    # clean degradation here keeps the failure mode observable.
    panel_ts = pd.to_datetime(panel[_TIMESTAMP_COLUMN], utc=True)
    if panel_ts.isna().any():
        logger.warning(
            "[MACRO-GPR] panel timestamp column contains %d NaT rows — "
            "skipping merge (caller should filter NaT upstream)",
            int(panel_ts.isna().sum()),
        )
        return panel

    # Sort-key DataFrame preserves the panel's original row order via
    # the _row_idx column. merge_asof requires the left key sorted.
    left = (
        pd.DataFrame({_TIMESTAMP_COLUMN: panel_ts, "_row_idx": range(len(panel))})
        .sort_values(_TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )
    merged = pd.merge_asof(
        left,
        gpr,
        on=_TIMESTAMP_COLUMN,
        direction="backward",
    )
    # Restore original panel ordering via _row_idx.
    merged = merged.sort_values("_row_idx").reset_index(drop=True)

    out = panel.copy()
    out[_GPR_COLUMN] = merged[_GPR_COLUMN].values

    nan_pct = out[_GPR_COLUMN].isna().mean() * 100.0
    logger.debug(
        "[MACRO-GPR] merged %s into panel (rows=%d, nan%%=%.1f, lag_days=%d)",
        _GPR_COLUMN,
        len(out),
        nan_pct,
        release_lag_days,
    )
    return out
