"""Attribution drilldowns for paper-engine runs.

Provides small, pandas-native helpers that split a fills frame and/or a
ledger frame into interpretable P&L components:

- ``compute_cost_attribution(fills)``: per-category bps and cash buckets,
  aggregated overall and per symbol.
- ``compute_regime_attribution(fills, regime_history)``: assigns each fill to
  the regime that was active on its date.
- ``compute_factor_attribution(fills, factor_scores)``: when orders carry a
  per-fill dominant factor label, groups realised cost by that factor.

The helpers deliberately accept plain DataFrames so they work both on a live
``UnifiedPaperEngine`` run output and on back-test frames with matching
columns. They never mutate inputs.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

_COST_COMPONENTS = (
    "spread_cost_bps",
    "impact_cost_bps",
    "adversarial_cost_bps",
    "sor_cost_bps",
)


def _safe_col(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return frame[col].astype(float).fillna(default)
    return pd.Series([default] * len(frame), index=frame.index, dtype=float)


def _per_fill_notional(frame: pd.DataFrame) -> pd.Series:
    """Notional per fill: |fill_qty| * fill_price when available, else |qty|*price."""
    if "fill_qty" in frame.columns and "fill_price" in frame.columns:
        return _safe_col(frame, "fill_qty").abs() * _safe_col(frame, "fill_price").abs()
    return _safe_col(frame, "qty").abs() * _safe_col(frame, "price").abs()


def compute_cost_attribution(fills: pd.DataFrame) -> dict:
    """Break fills into per-category cost buckets (bps and cash).

    Returns a dict with:

    - ``per_symbol`` (DataFrame): one row per symbol with notional, per-component
      cost in bps (share of the symbol's notional), and the cash equivalent.
    - ``total`` (dict): aggregate ``notional``, per-component bps, cash totals.

    Bps are computed as a notional-weighted average so they are comparable
    across symbols.
    """
    cols = ["symbol", "notional"] + list(_COST_COMPONENTS) + [
        c.replace("_bps", "_cash") for c in _COST_COMPONENTS
    ] + ["total_cost_bps", "total_cost_cash", "n_fills"]

    if fills is None or fills.empty:
        return {"per_symbol": pd.DataFrame(columns=cols),
                "total": {"notional": 0.0, "n_fills": 0}}

    frame = fills.copy()
    frame["notional"] = _per_fill_notional(frame)
    for component in _COST_COMPONENTS:
        bps = _safe_col(frame, component)
        cash = bps / 10_000.0 * frame["notional"]
        frame[component.replace("_bps", "_cash")] = cash
    total_bps = _safe_col(frame, "total_cost_bps")
    frame["total_cost_cash"] = total_bps / 10_000.0 * frame["notional"]

    agg_dict = {"notional": "sum"}
    for component in _COST_COMPONENTS:
        agg_dict[component.replace("_bps", "_cash")] = "sum"
    agg_dict["total_cost_cash"] = "sum"

    if "symbol" not in frame.columns:
        frame["symbol"] = "__unknown__"

    per_symbol = frame.groupby("symbol", as_index=False).agg(agg_dict)
    # Re-derive per-symbol bps from aggregated notional+cash (notional-weighted).
    for component in _COST_COMPONENTS:
        cash_col = component.replace("_bps", "_cash")
        per_symbol[component] = (
            per_symbol[cash_col] / per_symbol["notional"].replace(0, pd.NA) * 10_000.0
        ).fillna(0.0)
    per_symbol["total_cost_bps"] = (
        per_symbol["total_cost_cash"] / per_symbol["notional"].replace(0, pd.NA)
        * 10_000.0
    ).fillna(0.0)
    per_symbol["n_fills"] = (
        frame.groupby("symbol")["symbol"].count().reindex(per_symbol["symbol"]).values
    )

    per_symbol = per_symbol.sort_values("symbol", kind="mergesort").reset_index(drop=True)

    total_notional = float(per_symbol["notional"].sum())
    total: dict = {"notional": total_notional, "n_fills": int(per_symbol["n_fills"].sum())}
    for component in _COST_COMPONENTS:
        cash_col = component.replace("_bps", "_cash")
        total[cash_col] = float(per_symbol[cash_col].sum())
        total[component] = (
            total[cash_col] / total_notional * 10_000.0 if total_notional > 0 else 0.0
        )
    total["total_cost_cash"] = float(per_symbol["total_cost_cash"].sum())
    total["total_cost_bps"] = (
        total["total_cost_cash"] / total_notional * 10_000.0
        if total_notional > 0 else 0.0
    )

    return {"per_symbol": per_symbol, "total": total}


def compute_regime_attribution(
    fills: pd.DataFrame,
    regime_history: Iterable[dict],
) -> pd.DataFrame:
    """Assign each fill to the regime active on its ``date`` and aggregate.

    Args:
        fills: DataFrame with at least columns ``date``, ``symbol`` and any of
            ``total_cost_bps``, ``fill_qty``, ``fill_price``.
        regime_history: Iterable of ``{"date": str, "regime": str}`` mappings.
            Missing dates fall back to ``regime="unknown"``.

    Returns:
        DataFrame with one row per regime: ``regime``, ``notional``,
        ``n_fills``, ``total_cost_cash``, ``total_cost_bps``.
    """
    cols = ["regime", "notional", "n_fills", "total_cost_cash", "total_cost_bps"]
    if fills is None or fills.empty:
        return pd.DataFrame(columns=cols)

    regime_map: dict[str, str] = {}
    n_skipped = 0
    for entry in regime_history:
        try:
            regime_map[str(entry["date"])] = str(entry["regime"])
        except Exception:
            n_skipped += 1
            continue
    if n_skipped > 0:
        import logging
        log = logging.getLogger(__name__)
        if not regime_map:
            # Every entry malformed → downstream maps every fill to
            # regime="unknown" and the aggregate looks like one plausible
            # "unknown" bucket. That biases regime-cost calibration and
            # masks a schema drift. Warn loudly in this case.
            log.warning(
                "[Attribution] regime_history had %d entries but all failed "
                "schema — every fill will attribute to regime='unknown'; "
                "regime-cost calibration will be biased",
                n_skipped,
            )
        else:
            log.warning(
                "[Attribution] %d regime_history entries dropped due to "
                "schema errors (%d parsed cleanly)",
                n_skipped, len(regime_map),
            )

    frame = fills.copy()
    frame["regime"] = frame.get("date", pd.Series([""] * len(frame))).astype(str)
    frame["regime"] = frame["regime"].map(lambda d: regime_map.get(d, "unknown"))
    frame["notional"] = _per_fill_notional(frame)
    frame["total_cost_cash"] = (
        _safe_col(frame, "total_cost_bps") / 10_000.0 * frame["notional"]
    )

    agg = (
        frame.groupby("regime", as_index=False)
        .agg(notional=("notional", "sum"),
             n_fills=("regime", "count"),
             total_cost_cash=("total_cost_cash", "sum"))
    )
    agg["total_cost_bps"] = (
        agg["total_cost_cash"] / agg["notional"].replace(0, pd.NA) * 10_000.0
    ).fillna(0.0)
    return agg.sort_values("regime", kind="mergesort").reset_index(drop=True)


def compute_factor_attribution(
    fills: pd.DataFrame,
    factor_column: str = "dominant_factor",
) -> pd.DataFrame:
    """Aggregate cost by a ``dominant_factor`` tag carried on each fill.

    If the column is missing, the function returns an empty frame; that signals
    to callers that factor attribution is unavailable for this run.
    """
    cols = ["factor", "notional", "n_fills", "total_cost_cash", "total_cost_bps"]
    if fills is None or fills.empty or factor_column not in fills.columns:
        return pd.DataFrame(columns=cols)

    frame = fills.copy()
    frame["factor"] = frame[factor_column].astype(str).fillna("unknown")
    frame["notional"] = _per_fill_notional(frame)
    frame["total_cost_cash"] = (
        _safe_col(frame, "total_cost_bps") / 10_000.0 * frame["notional"]
    )
    agg = (
        frame.groupby("factor", as_index=False)
        .agg(notional=("notional", "sum"),
             n_fills=("factor", "count"),
             total_cost_cash=("total_cost_cash", "sum"))
    )
    agg["total_cost_bps"] = (
        agg["total_cost_cash"] / agg["notional"].replace(0, pd.NA) * 10_000.0
    ).fillna(0.0)
    return agg.sort_values("factor", kind="mergesort").reset_index(drop=True)


__all__ = [
    "compute_cost_attribution",
    "compute_regime_attribution",
    "compute_factor_attribution",
]
