"""Corporate actions: splits, dividends, total return adjustment."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_corporate_actions(path: str | None = None) -> pd.DataFrame:
    """Load corporate actions from CSV.

    Contract:
    - ``path is None`` → return empty frame (explicit "no source configured").
    - ``path`` provided but file missing → raise ``FileNotFoundError``.
      Silently returning an empty frame for an explicit path has masked
      missing corporate-actions files in prior incidents (D3 risk class).
    """
    if path is None:
        return pd.DataFrame(columns=["symbol", "date", "action_type", "factor"])
    return pd.read_csv(path, dtype={"symbol": "string", "action_type": "string"})


def apply_splits_for_research_prices(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Add close_research column with split-adjusted prices for research (returns/features).

    Prices before a split date are multiplied by 1/split_ratio; close is unchanged for trading.
    """
    result = prices.copy()
    if "close" not in result.columns:
        result["close_research"] = pd.Series(dtype=float)
        return result

    if actions.empty:
        result["close_research"] = result["close"]
        return result

    required = {"symbol", "action_type", "effective_date", "split_ratio"}
    missing = required - set(actions.columns)
    if missing:
        raise ValueError(f"actions missing required columns: {sorted(missing)}")
    if (actions["action_type"] != "SPLIT").any():
        raise ValueError("actions must contain only SPLIT actions")

    # Normalize timestamps for comparison
    ts_col = (
        "timestamp"
        if "timestamp" in result.columns
        else (result.columns[0] if len(result.columns) > 0 else "timestamp")
    )
    result_ts = pd.to_datetime(result[ts_col], utc=True)
    actions = actions.copy()
    actions["effective_date"] = pd.to_datetime(actions["effective_date"], utc=True)

    # Vectorized split-adjustment: for each price row, multiply by the product of
    # 1/split_ratio for all future splits of that symbol.
    # Pre-group splits by symbol so we avoid repeated O(n_actions) scans per row.
    splits_by_symbol: dict[str, pd.DataFrame] = {
        sym: grp.sort_values("effective_date") for sym, grp in actions.groupby("symbol")
    }

    result = result.copy()
    result_ts = result_ts.reset_index(drop=True)
    result = result.reset_index(drop=True)

    # Build the cumulative adjustment factor per row in one vectorized pass per symbol
    adj_factors = pd.Series(1.0, index=result.index, dtype=float)

    for sym, sym_splits in splits_by_symbol.items():
        sym_mask = result["symbol"] == sym
        if not sym_mask.any():
            continue
        sym_idx = result.index[sym_mask]
        sym_ts = result_ts[sym_idx]

        for split_row in sym_splits.itertuples(index=False):
            ratio = float(split_row.split_ratio)
            if ratio <= 0:
                continue
            # Rows before the split effective date get divided by split_ratio
            pre_split = sym_ts < split_row.effective_date
            adj_factors.loc[sym_idx[pre_split]] /= ratio

    result["close_research"] = result["close"] * adj_factors
    return result


def compute_dividend_cashflows(
    positions: pd.DataFrame,
    actions: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute dividend cashflows for positions (ledger-ready events)."""
    required_pos = {"symbol", "qty"}
    required_act = {"symbol", "action_type", "effective_date", "dividend_cash"}
    if required_pos - set(positions.columns):
        raise ValueError(
            f"positions missing required columns: {sorted(required_pos - set(positions.columns))}"
        )
    if actions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "cashflow_type", "amount"])
    if required_act - set(actions.columns):
        raise ValueError(
            f"actions missing required columns: {sorted(required_act - set(actions.columns))}"
        )
    if (actions["action_type"] != "DIVIDEND").any():
        raise ValueError("actions must contain only DIVIDEND actions")

    actions = actions.copy()
    actions["effective_date"] = pd.to_datetime(actions["effective_date"], utc=True)
    if as_of is not None:
        a = pd.Timestamp(as_of)
        as_of_utc = (
            a.tz_convert("UTC") if a.tzinfo is not None else a.tz_localize("UTC")
        )
        actions = actions[actions["effective_date"] <= as_of_utc]

    pos_df = positions[["symbol", "qty"]].copy()
    pos_df["qty"] = pd.to_numeric(pos_df["qty"], errors="coerce").fillna(0.0)
    merged = pos_df.merge(
        actions[["symbol", "effective_date", "dividend_cash"]],
        on="symbol",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "cashflow_type", "amount"])
    merged["amount"] = merged["qty"] * pd.to_numeric(
        merged["dividend_cash"], errors="coerce"
    ).fillna(0.0)
    merged["cashflow_type"] = "DIVIDEND"
    out = merged.rename(columns={"effective_date": "timestamp"})[
        ["timestamp", "symbol", "cashflow_type", "amount"]
    ]
    return out.sort_values("timestamp").reset_index(drop=True)


def adjust_prices_for_splits(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Adjust the close column for stock splits (backward adjustment).

    Prices recorded *before* a split date are multiplied by ``1/split_ratio``
    so that the adjusted series is continuous across the split event.

    Requires actions DataFrame with columns:
        ``symbol``, ``action_type`` (must be "SPLIT"), ``effective_date``, ``split_ratio``

    Requires prices DataFrame with columns:
        ``symbol``, ``close``, and a timestamp column (``timestamp`` preferred,
        otherwise the first column is used as the time axis).

    Args:
        prices: Price DataFrame. Returns a copy with ``close`` adjusted.
        actions: Corporate actions DataFrame. Only SPLIT rows are processed.

    Returns:
        Copy of *prices* with ``close`` backward-adjusted for all splits found
        in *actions*. Returns *prices* unchanged (not a copy) if *actions* is empty.
        Returns a copy unchanged if required columns are missing in *actions*.

    Note:
        Only modifies the ``close`` column. Other price columns (open, high, low,
        volume) are preserved without adjustment.
    """
    if actions.empty:
        return prices

    required = {"symbol", "action_type", "effective_date", "split_ratio"}
    missing = required - set(actions.columns)
    if missing:
        # A malformed corporate-actions file (e.g. schema drift renaming
        # ``split_ratio`` → ``ratio``) used to silently return unadjusted
        # prices, so backtests computed returns across split boundaries
        # (a 10:1 split became a -90% "return") without any log line. The
        # sibling ``apply_splits_for_research_prices`` already raises on
        # schema gaps; aligning both paths eliminates the silent fork.
        raise ValueError(
            f"adjust_prices_for_splits: actions missing columns {sorted(missing)}; "
            f"available columns: {sorted(actions.columns)}"
        )

    split_actions = actions[actions["action_type"] == "SPLIT"].copy()
    if split_actions.empty:
        return prices.copy()

    if "close" not in prices.columns:
        return prices.copy()

    result = prices.copy()

    # Determine timestamp column
    ts_col = (
        "timestamp"
        if "timestamp" in result.columns
        else (result.columns[0] if len(result.columns) > 0 else "timestamp")
    )
    result_ts = pd.to_datetime(result[ts_col], utc=True, errors="coerce")
    split_actions["effective_date"] = pd.to_datetime(
        split_actions["effective_date"], utc=True, errors="coerce"
    )

    # Apply each split: for rows of the same symbol before the split date,
    # multiply close by 1/split_ratio (backward adjustment).
    for s in split_actions.itertuples(index=False):
        ratio = float(s.split_ratio)
        if ratio <= 0:
            continue
        apply_mask = (result["symbol"] == s.symbol) & (result_ts < s.effective_date)
        result.loc[apply_mask, "close"] = result.loc[apply_mask, "close"] / ratio

    return result


def compute_total_return_index(
    prices: pd.DataFrame,
    dividend_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Compute total return (price + reinvested dividends) per symbol.

    Adds a 'close_total_return' column that accounts for dividend reinvestment.
    This corrects the ~2% p.a. bias from ignoring dividends in return computation.

    Args:
        prices: DataFrame with columns: symbol, timestamp (or date), close.
            Must be sorted by (symbol, timestamp).
        dividend_actions: DataFrame with columns: symbol, effective_date,
            dividend_cash, action_type='DIVIDEND'.

    Returns:
        Copy of prices with added 'close_total_return' column.
    """
    result = prices.copy()
    result["close_total_return"] = result["close"].astype(float).copy()

    # The sibling `adjust_prices_for_splits` raises on schema gaps; this path
    # previously returned un-adjusted prices silently on any of four early
    # exits. Downstream PnL then under-reports returns by ~div yield p.a.
    # without any visible signal. Emit a WARN on each silent-no-op so a
    # missing/misshaped dividend feed becomes observable instead of becoming
    # a 2%-p.a. attribution blind spot.
    if dividend_actions.empty:
        logger.warning(
            "[CorpActions] compute_total_return_index: dividend_actions is "
            "empty — returning close_total_return = close (no dividend adjustment)"
        )
        return result

    required = {"symbol", "effective_date", "dividend_cash"}
    missing = required - set(dividend_actions.columns)
    if missing:
        logger.warning(
            "[CorpActions] compute_total_return_index: dividend_actions "
            "missing columns %s — returning close_total_return = close",
            missing,
        )
        return result

    divs = dividend_actions.copy()
    if "action_type" in divs.columns:
        divs = divs[divs["action_type"] == "DIVIDEND"]
    if divs.empty:
        logger.warning(
            "[CorpActions] compute_total_return_index: no DIVIDEND rows after "
            "action_type filter — returning close_total_return = close"
        )
        return result

    ts_col = "timestamp" if "timestamp" in result.columns else "date"
    if ts_col not in result.columns:
        logger.warning(
            "[CorpActions] compute_total_return_index: prices frame has "
            "neither 'timestamp' nor 'date' column — returning close_total_return = close"
        )
        return result

    divs["effective_date"] = pd.to_datetime(divs["effective_date"], utc=True)
    result[ts_col] = pd.to_datetime(result[ts_col], utc=True)

    # Pre-group divs by symbol to avoid O(N*M) per-symbol filter
    _divs_by_sym = {
        sym: grp.sort_values("effective_date")
        for sym, grp in divs.groupby("symbol", sort=False)
    }

    # For each symbol, compute cumulative dividend adjustment
    for sym, sym_prices in result.groupby("symbol", sort=False):
        sym_divs = _divs_by_sym.get(sym, pd.DataFrame())
        if sym_divs.empty:
            continue

        sym_idx = sym_prices.index
        sym_ts = sym_prices[ts_col]
        sym_close = sym_prices["close"].astype(float)

        # Build cumulative reinvestment factor per row
        cum_factor = pd.Series(1.0, index=sym_idx, dtype=float)
        for div_row in sym_divs.itertuples(index=False):
            ex_date = div_row.effective_date
            div_cash = float(div_row.dividend_cash)
            # Find the close price on or just before ex-date for reinvestment ratio
            pre_ex = sym_ts[sym_ts < ex_date]
            if pre_ex.empty:
                continue
            last_pre_idx = pre_ex.index[-1]
            close_at = float(sym_close.loc[last_pre_idx])
            if close_at <= 0:
                continue
            reinvest_ratio = 1.0 + div_cash / close_at
            # All rows on or after ex-date get multiplied
            post_mask = sym_ts >= ex_date
            cum_factor.loc[sym_idx[post_mask]] *= reinvest_ratio

        result.loc[sym_idx, "close_total_return"] = sym_close.values * cum_factor.values

    return result


# ---------------------------------------------------------------------------
# Delisting: forced position closure (Sprint 5 / C2)
# ---------------------------------------------------------------------------


def apply_delisting_exits(
    positions: pd.DataFrame,
    actions: pd.DataFrame,
    prices: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate forced-exit events for delisted symbols.

    When a symbol reaches its delisting effective_date, any open position is
    closed at the last available price.  The result is a ledger-ready DataFrame
    with ``exit_type='DELIST_EXIT'``.

    Args:
        positions: Open positions with columns ``symbol``, ``qty``.
        actions: Corporate actions with ``action_type='DELISTING'``,
            ``symbol``, ``effective_date``.
        prices: Price panel with ``symbol``, ``timestamp`` (or first col),
            ``close``.
        as_of: Optional cutoff — only delistings on or before *as_of* are
            processed.

    Returns:
        DataFrame with columns ``timestamp``, ``symbol``, ``exit_type``,
        ``exit_price``, ``qty``.  Empty if no delistings apply.
    """
    out_cols = ["timestamp", "symbol", "exit_type", "exit_price", "qty"]
    if positions is None or positions.empty:
        return pd.DataFrame(columns=out_cols)
    if actions is None or actions.empty:
        return pd.DataFrame(columns=out_cols)

    required = {"symbol", "action_type", "effective_date"}
    missing = required - set(actions.columns)
    if missing:
        # Silent no-op on schema drift would let delistings invisibly
        # accumulate as stale positions forever. Warn so a rename like
        # effective_date→eff_date becomes observable.
        logger.warning(
            "[CorpActions] apply_delisting_exits: actions missing columns %s "
            "— skipped; open positions for delisted symbols will NOT be "
            "forcibly exited",
            missing,
        )
        return pd.DataFrame(columns=out_cols)

    delistings = actions[actions["action_type"] == "DELISTING"].copy()
    if delistings.empty:
        return pd.DataFrame(columns=out_cols)

    delistings["effective_date"] = pd.to_datetime(
        delistings["effective_date"], utc=True, errors="coerce"
    )
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        delistings = delistings[delistings["effective_date"] <= cutoff]

    if delistings.empty:
        return pd.DataFrame(columns=out_cols)

    ts_col = (
        "timestamp"
        if "timestamp" in prices.columns
        else (prices.columns[0] if len(prices.columns) > 0 else "timestamp")
    )
    pos_qty_map = positions.groupby("symbol")["qty"].first().to_dict()
    sym_prices_map = {sym: grp for sym, grp in prices.groupby("symbol")}
    rows: list[dict] = []
    for dl in delistings.itertuples(index=False):
        sym = dl.symbol
        eff = dl.effective_date
        qty = float(pos_qty_map.get(sym, 0.0))
        if qty == 0:
            continue
        sym_prices = sym_prices_map.get(sym)
        if sym_prices is None or sym_prices.empty:
            continue
        sym_prices = sym_prices.copy()
        sym_prices[ts_col] = pd.to_datetime(sym_prices[ts_col], utc=True)
        before = sym_prices[sym_prices[ts_col] <= eff]
        if before.empty:
            last_price = float(sym_prices["close"].iloc[-1])
        else:
            last_price = float(before["close"].iloc[-1])
        rows.append(
            {
                "timestamp": eff,
                "symbol": sym,
                "exit_type": "DELIST_EXIT",
                "exit_price": last_price,
                "qty": qty,
            }
        )

    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Spin-off: position split into parent + child (Sprint 5 / C2)
# ---------------------------------------------------------------------------


def apply_spinoff(
    positions: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Apply spin-off actions: split a position into parent + child.

    A spin-off action distributes shares of a new entity (``child_symbol``)
    to holders of the parent at a given ``spinoff_ratio`` (child shares per
    parent share).  The parent position qty is unchanged; a new row for the
    child is created.

    Required action columns: ``symbol`` (parent), ``action_type`` (``'SPINOFF'``),
    ``effective_date``, ``child_symbol``, ``spinoff_ratio``.

    Args:
        positions: Current positions with ``symbol``, ``qty``,
            ``avg_price`` (optional).
        actions: Corporate actions.

    Returns:
        Updated positions DataFrame with new child rows appended.
        Original rows are preserved.
    """
    if positions is None or positions.empty:
        return positions.copy() if positions is not None else pd.DataFrame()
    if actions is None or actions.empty:
        return positions.copy()

    required = {
        "symbol",
        "action_type",
        "effective_date",
        "child_symbol",
        "spinoff_ratio",
    }
    missing = required - set(actions.columns)
    if missing:
        # A spinoff with a renamed column (spinoff_ratio → ratio) previously
        # returned positions unchanged — child position never created, qty
        # accounting silently drifts. Warn so schema drift is observable.
        logger.warning(
            "[CorpActions] apply_spinoff: actions missing columns %s — "
            "skipped; child positions will NOT be created",
            missing,
        )
        return positions.copy()

    spinoffs = actions[actions["action_type"] == "SPINOFF"].copy()
    if spinoffs.empty:
        return positions.copy()

    result = positions.copy()
    new_rows: list[dict] = []
    for sp in spinoffs.itertuples(index=False):
        parent = sp.symbol
        child = sp.child_symbol
        ratio = float(sp.spinoff_ratio)
        if ratio <= 0:
            continue
        parent_mask = result["symbol"] == parent
        if not parent_mask.any():
            continue
        parent_qty = float(result.loc[parent_mask, "qty"].iloc[0])
        child_qty = parent_qty * ratio
        row = {"symbol": child, "qty": child_qty}
        if "avg_price" in result.columns:
            row["avg_price"] = 0.0  # cost basis TBD by accounting
        # Copy any other columns from parent row with sensible defaults
        parent_row = result.loc[parent_mask].iloc[0]
        for col in result.columns:
            if col not in row:
                row[col] = parent_row[col]
        row["symbol"] = child
        row["qty"] = child_qty
        new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        result = pd.concat([result, new_df], ignore_index=True)

    return result
