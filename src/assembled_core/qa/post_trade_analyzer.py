"""Post-Trade Analyzer — M11: Post-Trade Learning Loop.

Analyzes completed trades against actual price outcomes to compute signal quality metrics
and feed learning records into the learning store.

PIT-safe: only uses price data available after trade close.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default forward-return horizon in calendar days
DEFAULT_HORIZON_DAYS = 5


def compute_forward_returns(
    prices_df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.DataFrame:
    """Compute N-day forward returns for each symbol from a price DataFrame.

    Args:
        prices_df: DataFrame with columns: timestamp (UTC), symbol, close
        horizon_days: Number of calendar days to look forward.

    Returns:
        DataFrame with columns: timestamp, symbol, close, forward_return
        forward_return = close_t+horizon / close_t - 1 (NaN if future price missing)
    """
    required = {"timestamp", "symbol", "close"}
    if not required.issubset(prices_df.columns):
        missing = required - set(prices_df.columns)
        raise ValueError(f"prices_df missing columns: {missing}")

    df = prices_df[["timestamp", "symbol", "close"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    chunks = []
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("timestamp").copy()
        grp["future_ts"] = grp["timestamp"] + pd.Timedelta(days=horizon_days)
        ts_arr = grp["timestamp"].values
        close_arr = grp["close"].to_numpy(dtype=float)
        future_arr = grp["future_ts"].values

        # Binary search: O(n log n) instead of O(n²) filter per row
        idxs = np.searchsorted(ts_arr, future_arr, side="left")
        valid = idxs < len(close_arr)
        fwd_closes = np.where(
            valid, close_arr[np.minimum(idxs, len(close_arr) - 1)], np.nan
        )
        fwd_returns = np.where(valid, fwd_closes / close_arr - 1.0, np.nan)

        sub = grp[["timestamp", "close"]].copy()
        sub["symbol"] = symbol
        sub["forward_return"] = fwd_returns
        chunks.append(sub)

    if not chunks:
        return pd.DataFrame(columns=["timestamp", "symbol", "close", "forward_return"])
    return pd.concat(chunks, ignore_index=True)[
        ["timestamp", "symbol", "close", "forward_return"]
    ]


def compute_signal_hit_rate(
    trades_df: pd.DataFrame,
    forward_returns_df: pd.DataFrame,
    *,
    tolerance_days: int = 2,
) -> pd.DataFrame:
    """Match trades to forward returns and compute hit rate per symbol.

    A trade is a "hit" if:
    - BUY and forward_return > 0
    - SELL and forward_return < 0

    Args:
        trades_df: DataFrame with columns: symbol, side (BUY/SELL), event_ts, qty, price
        forward_returns_df: Output from compute_forward_returns().
        tolerance_days: Max days difference for timestamp matching.

    Returns:
        DataFrame with columns: symbol, total_trades, hits, hit_rate, avg_forward_return
    """
    if trades_df.empty or forward_returns_df.empty:
        return pd.DataFrame(
            columns=["symbol", "total_trades", "hits", "hit_rate", "avg_forward_return"]
        )

    trades = trades_df.copy()
    trades["event_ts"] = pd.to_datetime(trades["event_ts"], utc=True)

    fwd = forward_returns_df.copy()
    fwd["timestamp"] = pd.to_datetime(fwd["timestamp"], utc=True)

    _fwd_by_sym = {
        sym: grp.set_index("timestamp")["forward_return"]
        for sym, grp in fwd.groupby("symbol", sort=False)
    }

    results = []
    for symbol, sym_trades in trades.groupby("symbol", sort=False):
        sym_fwd_raw = _fwd_by_sym.get(symbol, pd.Series(dtype=float))
        if sym_fwd_raw.empty:
            continue
        sym_fwd = sym_fwd_raw.sort_index()
        sym_idx = pd.DatetimeIndex(sym_fwd.index)

        hits = 0
        total = 0
        fwd_returns = []

        for trade in sym_trades.itertuples(index=False):
            ts = pd.Timestamp(trade.event_ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            side = str(getattr(trade, "side", "")).upper()
            if not side:
                qty = float(getattr(trade, "qty", 0))
                side = "BUY" if qty > 0 else "SELL"

            # get_indexer avoids int64 unit mismatch between ns/us pandas builds
            locs = sym_idx.get_indexer([ts], method="nearest")
            closest_pos = int(locs[0])
            if closest_pos == -1:
                continue
            if (
                abs((sym_idx[closest_pos] - ts).total_seconds())
                > tolerance_days * 86400
            ):
                continue

            fwd_ret = float(sym_fwd.iloc[closest_pos])
            if np.isnan(fwd_ret):
                continue

            fwd_returns.append(fwd_ret)
            total += 1
            if (side == "BUY" and fwd_ret > 0) or (side == "SELL" and fwd_ret < 0):
                hits += 1

        if total > 0:
            results.append(
                {
                    "symbol": symbol,
                    "total_trades": total,
                    "hits": hits,
                    "hit_rate": hits / total,
                    "avg_forward_return": (
                        sum(fwd_returns) / len(fwd_returns) if fwd_returns else None
                    ),
                }
            )

    if not results:
        return pd.DataFrame(
            columns=["symbol", "total_trades", "hits", "hit_rate", "avg_forward_return"]
        )
    return pd.DataFrame(results).sort_values("symbol").reset_index(drop=True)


def build_learning_record(
    run_id: str,
    analysis_date: str,
    hit_rate_df: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured learning record for storage.

    Args:
        run_id: Unique run identifier.
        analysis_date: ISO date string (YYYY-MM-DD).
        hit_rate_df: Output from compute_signal_hit_rate().
        horizon_days: Forward horizon used.
        extra: Optional extra fields.

    Returns:
        Dict suitable for JSON serialization.
    """
    overall_hits = int(hit_rate_df["hits"].sum()) if not hit_rate_df.empty else 0
    overall_total = (
        int(hit_rate_df["total_trades"].sum()) if not hit_rate_df.empty else 0
    )
    overall_hit_rate = overall_hits / overall_total if overall_total > 0 else 0.0

    record: dict[str, Any] = {
        "run_id": run_id,
        "analysis_date": analysis_date,
        "horizon_days": horizon_days,
        "overall_hit_rate": round(overall_hit_rate, 4),
        "overall_total_trades": overall_total,
        "per_symbol": [],
    }

    if not hit_rate_df.empty:
        record["per_symbol"] = [
            {
                "symbol": row.symbol,
                "total_trades": int(row.total_trades),
                "hits": int(row.hits),
                "hit_rate": round(float(row.hit_rate), 4),
                "avg_forward_return": round(float(row.avg_forward_return), 4),
            }
            for row in hit_rate_df.itertuples(index=False)
        ]

    if extra:
        record.update(extra)

    return record


def analyze_and_learn(
    trades_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    factor_panel_df: pd.DataFrame | None = None,
    regime_state_df: pd.DataFrame | None = None,
    learning_store_path: Path | None = None,
) -> dict:
    """Extended post-trade analysis with factor attribution and regime context.

    Extends compute_signal_hit_rate() with:
    - Factor attribution per trade (if factor_panel_df provided)
    - Regime context at entry (if regime_state_df provided)
    - Signal-strength vs outcome binning (weak/medium/strong)
    - Writes enriched records to learning store JSONL

    Log prefix: [POST-TRADE]

    Args:
        trades_df: DataFrame with columns: event_ts, symbol, side/direction,
            and optionally pnl, score.
        prices_df: Price DataFrame with columns: timestamp, symbol, close.
        factor_panel_df: Optional DataFrame with factor values per
            (timestamp, symbol). If provided, factor columns (all non-index
            columns except timestamp/symbol) are used for attribution.
        regime_state_df: Optional DataFrame with columns: timestamp,
            regime (or regime_label). Joined to trades by nearest timestamp.
        learning_store_path: Path to JSONL store. If None, uses
            DEFAULT_LEARNING_STORE_PATH from learning_store module.

    Returns:
        Summary dict with keys: hit_rate_summary, factor_attribution,
        regime_breakdown, strength_bins, n_trades_analyzed, store_path.
    """
    from pathlib import Path as _Path

    logger.info(
        "[POST-TRADE] analyze_and_learn called with %d trades",
        len(trades_df) if trades_df is not None else 0,
    )

    result: dict[str, Any] = {
        "hit_rate_summary": {},
        "factor_attribution": [],
        "regime_breakdown": {},
        "strength_bins": {},
        "n_trades_analyzed": 0,
        "store_path": None,
    }

    # Guard: need at least some data
    if trades_df is None or trades_df.empty:
        logger.warning("[POST-TRADE] trades_df is empty — nothing to analyze")
        return result
    if prices_df is None or prices_df.empty:
        logger.warning("[POST-TRADE] prices_df is empty — skipping analysis")
        return result

    # --- Step 1: Compute basic signal hit rate ---
    try:
        hit_df = compute_signal_hit_rate(trades_df, prices_df)
        if not hit_df.empty:
            total = int(hit_df["total_trades"].sum())
            hits = int(hit_df["hits"].sum())
            overall_hr = hits / total if total > 0 else 0.0
            result["hit_rate_summary"] = {
                "overall_hit_rate": round(overall_hr, 4),
                "total_trades": total,
                "hits": hits,
                "per_symbol": hit_df.to_dict(orient="records"),
            }
            result["n_trades_analyzed"] = total
            logger.info("[POST-TRADE] hit rate: %.3f (%d/%d)", overall_hr, hits, total)
    except Exception as exc:
        logger.warning("[POST-TRADE] compute_signal_hit_rate failed: %s", exc)

    # --- Step 2: Build enriched trade records (merge factor/regime info) ---
    enriched = trades_df.copy()

    # Normalise timestamp column name
    ts_col = (
        "event_ts"
        if "event_ts" in enriched.columns
        else ("timestamp" if "timestamp" in enriched.columns else None)
    )

    # Attach factor values at entry
    if factor_panel_df is not None and not factor_panel_df.empty and ts_col is not None:
        try:
            fp = factor_panel_df.copy()
            # Identify factor columns (everything except timestamp/symbol)
            meta_cols = {"timestamp", "symbol", "date"}
            factor_cols = [c for c in fp.columns if c.lower() not in meta_cols]
            if factor_cols:
                fp_ts = "timestamp" if "timestamp" in fp.columns else fp.columns[0]
                fp["_merge_ts"] = pd.to_datetime(fp[fp_ts], errors="coerce")
                enriched["_merge_ts"] = pd.to_datetime(
                    enriched[ts_col], errors="coerce"
                )

                # Merge on nearest timestamp + symbol if possible
                sym_col_fp = "symbol" if "symbol" in fp.columns else None
                sym_col_tr = "symbol" if "symbol" in enriched.columns else None

                if sym_col_fp and sym_col_tr:
                    merged = pd.merge_asof(
                        enriched.sort_values("_merge_ts"),
                        fp[["_merge_ts", sym_col_fp] + factor_cols].sort_values(
                            "_merge_ts"
                        ),
                        on="_merge_ts",
                        by=sym_col_fp,
                        direction="backward",
                        tolerance=pd.Timedelta(days=5),
                    )
                    enriched = merged
                logger.info(
                    "[POST-TRADE] factor_panel merged, %d factor cols", len(factor_cols)
                )
        except Exception as exc:
            logger.warning("[POST-TRADE] factor_panel merge failed: %s", exc)
            factor_cols = []
    else:
        factor_cols = []

    # Attach regime label at entry
    if regime_state_df is not None and not regime_state_df.empty and ts_col is not None:
        try:
            rdf = regime_state_df.copy()
            regime_ts_col = (
                "timestamp" if "timestamp" in rdf.columns else rdf.columns[0]
            )
            regime_label_col = next(
                (c for c in rdf.columns if "regime" in c.lower()), None
            )
            if regime_label_col:
                rdf["_regime_ts"] = pd.to_datetime(rdf[regime_ts_col], errors="coerce")
                enriched["_regime_ts"] = pd.to_datetime(
                    enriched[ts_col], errors="coerce"
                )
                rdf_slim = rdf[["_regime_ts", regime_label_col]].sort_values(
                    "_regime_ts"
                )
                merged_r = pd.merge_asof(
                    enriched.sort_values("_regime_ts"),
                    rdf_slim,
                    on="_regime_ts",
                    direction="backward",
                    tolerance=pd.Timedelta(days=5),
                )
                enriched = merged_r
                logger.info("[POST-TRADE] regime column '%s' merged", regime_label_col)
        except Exception as exc:
            logger.warning("[POST-TRADE] regime merge failed: %s", exc)

    # --- Step 3: Factor attribution ---
    if factor_cols:
        try:
            from src.assembled_core.qa.learning_store import compute_factor_attribution

            attr_df = compute_factor_attribution(enriched, factor_cols)
            result["factor_attribution"] = attr_df.to_dict(orient="records")
        except Exception as exc:
            logger.warning("[POST-TRADE] factor_attribution failed: %s", exc)

    # --- Step 4: Regime breakdown ---
    try:
        regime_col = next(
            (
                c
                for c in enriched.columns
                if "regime" in c.lower() and c != "_regime_ts"
            ),
            None,
        )
        if regime_col:
            pnl_col = "pnl" if "pnl" in enriched.columns else None
            if pnl_col:
                rbreakdown = (
                    enriched.groupby(regime_col)[pnl_col]
                    .agg(["mean", "count", "sum"])
                    .rename(
                        columns={
                            "mean": "avg_pnl",
                            "count": "n_trades",
                            "sum": "total_pnl",
                        }
                    )
                )
                result["regime_breakdown"] = rbreakdown.to_dict(orient="index")
    except Exception as exc:
        logger.warning("[POST-TRADE] regime breakdown failed: %s", exc)

    # --- Step 5: Signal-strength binning (weak/medium/strong) ---
    try:
        score_col = "score" if "score" in enriched.columns else None
        pnl_col = "pnl" if "pnl" in enriched.columns else None
        if score_col and pnl_col:
            scores = enriched[score_col].dropna()
            if not scores.empty:
                q33 = float(scores.quantile(0.33))
                q66 = float(scores.quantile(0.66))

                def _strength_bin(s: float) -> str:
                    try:
                        if s <= q33:
                            return "weak"
                        elif s <= q66:
                            return "medium"
                        else:
                            return "strong"
                    except Exception:
                        return "unknown"

                enriched = enriched.copy()
                enriched["_strength_bin"] = enriched[score_col].apply(
                    lambda x: _strength_bin(float(x)) if pd.notna(x) else "unknown"
                )
                bins_summary = (
                    enriched.groupby("_strength_bin")[pnl_col]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "avg_pnl", "count": "n_trades"})
                )
                result["strength_bins"] = bins_summary.to_dict(orient="index")
    except Exception as exc:
        logger.warning("[POST-TRADE] strength binning failed: %s", exc)

    # --- Step 6: Write enriched records to learning store ---
    try:
        from src.assembled_core.qa.learning_store import (
            DEFAULT_LEARNING_STORE_PATH,
            append_learning_record,
        )

        store_path = (
            _Path(learning_store_path)
            if learning_store_path
            else DEFAULT_LEARNING_STORE_PATH
        )

        import uuid
        from datetime import date

        record = {
            "run_id": str(uuid.uuid4())[:8],
            "analysis_date": str(date.today()),
            "n_trades_analyzed": result["n_trades_analyzed"],
            "overall_hit_rate": result["hit_rate_summary"].get("overall_hit_rate"),
            "factor_attribution": result["factor_attribution"],
            "regime_breakdown": {
                str(k): v for k, v in result["regime_breakdown"].items()
            },
            "strength_bins": result["strength_bins"],
        }
        append_learning_record(record, store_path)
        result["store_path"] = str(store_path)
        logger.info("[POST-TRADE] enriched record written to %s", store_path)
    except Exception as exc:
        logger.warning("[POST-TRADE] writing to learning store failed: %s", exc)

    logger.info(
        "[POST-TRADE] analyze_and_learn complete: %d trades, hit_rate=%s",
        result["n_trades_analyzed"],
        result["hit_rate_summary"].get("overall_hit_rate"),
    )
    return result
