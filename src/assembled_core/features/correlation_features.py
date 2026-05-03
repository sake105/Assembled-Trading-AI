"""Cross-asset correlation and dispersion features.

Provides features that capture the correlation regime of the market:

- ``avg_pairwise_corr_20d`` / ``_60d``: Rolling average pairwise correlation
  across all symbols.  Low → diversification intact.  High → herding/crisis.
- ``return_dispersion``: Cross-sectional return dispersion (std of returns
  across symbols per day).  Low dispersion → herd behaviour, high → stock
  picking opportunity.
- ``intra_sector_dispersion`` / ``inter_sector_dispersion``: Sector-level
  decomposition — collapse of intra-sector dispersion signals sector crash risk.
- ``corr_to_vix``: Rolling correlation of each asset to VIX — assets with high
  VIX correlation are fragile in stress.
- ``corr_regime_zscore``: z-score of current avg correlation vs. 252d history —
  how unusual is the current correlation regime.

These features are critical for portfolio construction (diversification
monitoring) and regime detection (correlation breakdown signals crises).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_avg_pairwise_correlation(
    returns_wide: pd.DataFrame,
    windows: tuple[int, ...] = (20, 60),
) -> pd.DataFrame:
    """Compute rolling average pairwise correlation across assets.

    Args:
        returns_wide: Wide-format DataFrame (index=dates, columns=symbols)
            containing daily returns.
        windows: Rolling windows in trading days.

    Returns:
        DataFrame indexed by date with ``avg_pairwise_corr_{w}d`` columns.
    """
    if returns_wide.empty or returns_wide.shape[1] < 2:
        return pd.DataFrame(index=returns_wide.index)

    results = pd.DataFrame(index=returns_wide.index)
    n_sym = returns_wide.shape[1]
    n_dates = len(returns_wide)
    tri_row, tri_col = np.triu_indices(n_sym, k=1)

    for w in windows:
        col = f"avg_pairwise_corr_{w}d"
        rolling_corr = returns_wide.rolling(w, min_periods=w).corr()
        # reshape (n_dates * n_sym, n_sym) → (n_dates, n_sym, n_sym) then extract upper triangle
        arr = rolling_corr.to_numpy().reshape(n_dates, n_sym, n_sym)
        upper_vals = arr[:, tri_row, tri_col]  # (n_dates, n_pairs)
        with np.errstate(all="ignore"):
            mean_corr = np.nanmean(upper_vals, axis=1)
        results[col] = mean_corr

    return results


def compute_return_dispersion(
    returns_wide: pd.DataFrame,
) -> pd.Series:
    """Cross-sectional return dispersion per day.

    Dispersion = std of returns across all symbols on each date.
    Low dispersion → herd behaviour (all stocks move together).
    High dispersion → idiosyncratic moves (stock-picking opportunity).

    Args:
        returns_wide: Wide-format returns (dates × symbols).

    Returns:
        Series of cross-sectional standard deviation per date.
    """
    return returns_wide.std(axis=1, skipna=True).rename("return_dispersion")


def compute_sector_dispersion(
    returns_wide: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Compute intra-sector and inter-sector dispersion.

    Args:
        returns_wide: Wide-format returns (dates × symbols).
        sector_map: Dict mapping symbol → sector name.

    Returns:
        DataFrame with ``intra_sector_dispersion`` and
        ``inter_sector_dispersion`` columns.
    """
    # Map symbols to sectors
    available = [s for s in returns_wide.columns if s in sector_map]
    if len(available) < 2:
        return pd.DataFrame(
            {"intra_sector_dispersion": np.nan, "inter_sector_dispersion": np.nan},
            index=returns_wide.index,
        )

    # Vectorized: reshape to long, add sector column, groupby (date, sector)
    _stacked = returns_wide[available].stack()
    _stacked.name = "ret"
    ret_long = _stacked.reset_index()
    _date_col = ret_long.columns[0]
    ret_long.columns = [_date_col, "symbol", "ret"]
    ret_long["sector"] = ret_long["symbol"].map(sector_map)

    grp = ret_long.groupby([_date_col, "sector"])["ret"]
    sector_means_wide = grp.mean().unstack("sector")
    sector_stds_wide = grp.std(ddof=1).unstack("sector")
    sector_counts_wide = grp.count().unstack("sector")

    # Intra-sector: mean of within-sector std (only where count >= 2 symbols)
    intra_raw = sector_stds_wide.where(sector_counts_wide >= 2)
    intra_series = intra_raw.mean(axis=1, skipna=True).reindex(returns_wide.index)

    # Inter-sector: population std of sector means (only where >= 2 sectors have finite mean)
    valid_count = sector_means_wide.count(axis=1)
    inter_series = sector_means_wide.std(axis=1, ddof=0, skipna=True).where(
        valid_count >= 2
    ).reindex(returns_wide.index)

    return pd.DataFrame(
        {
            "intra_sector_dispersion": intra_series.to_numpy(),
            "inter_sector_dispersion": inter_series.to_numpy(),
        },
        index=returns_wide.index,
    )


def compute_correlation_to_benchmark(
    returns_wide: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Rolling correlation of each asset to a benchmark (e.g., VIX or SPY).

    Args:
        returns_wide: Wide-format returns (dates × symbols).
        benchmark_returns: Series of benchmark returns, same index as
            ``returns_wide``.
        window: Rolling correlation window in days.

    Returns:
        DataFrame with ``corr_to_benchmark_{symbol}`` for each symbol.
    """
    results = pd.DataFrame(index=returns_wide.index)

    for sym in returns_wide.columns:
        col_name = f"corr_to_benchmark_{sym}"
        results[col_name] = returns_wide[sym].rolling(window, min_periods=20).corr(benchmark_returns)

    return results


def compute_correlation_regime_features(
    returns_wide: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 60,
    history_window: int = 252,
) -> pd.DataFrame:
    """Compute correlation regime features.

    Produces a compact set of market-wide correlation features:

    - ``avg_corr_short``: Short-window average pairwise correlation
    - ``avg_corr_long``: Long-window average pairwise correlation
    - ``corr_regime_zscore``: z-score of short-window corr vs 252d history
    - ``corr_momentum``: 5d change in average correlation (rising = stress)

    Args:
        returns_wide: Wide-format returns (dates × symbols).
        short_window: Short rolling window.
        long_window: Long rolling window.
        history_window: History for z-scoring.

    Returns:
        DataFrame with correlation regime features.
    """
    corr_df = compute_avg_pairwise_correlation(
        returns_wide, windows=(short_window, long_window),
    )

    short_col = f"avg_pairwise_corr_{short_window}d"
    long_col = f"avg_pairwise_corr_{long_window}d"

    result = pd.DataFrame(index=returns_wide.index)
    result["avg_corr_short"] = corr_df[short_col]
    result["avg_corr_long"] = corr_df[long_col]

    # Z-score of short-window correlation vs. rolling history
    rolling_mean = result["avg_corr_short"].rolling(history_window, min_periods=60).mean()
    rolling_std = result["avg_corr_short"].rolling(history_window, min_periods=60).std()
    result["corr_regime_zscore"] = (result["avg_corr_short"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # Correlation momentum (5d change)
    result["corr_momentum"] = result["avg_corr_short"].diff(5)

    return result


def build_correlation_features_panel(
    prices_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    close_col: str = "close",
    sector_map: dict[str, str] | None = None,
    benchmark_col: str | None = None,
) -> pd.DataFrame:
    """Build correlation features from panel-format price data.

    Convenience function that converts panel prices to wide returns,
    computes all correlation features, and returns them in panel format.

    Args:
        prices_df: Panel DataFrame with timestamp, symbol, close.
        timestamp_col: Timestamp column.
        symbol_col: Symbol column.
        close_col: Close price column.
        sector_map: Optional symbol → sector mapping.
        benchmark_col: Optional symbol to use as benchmark for
            per-asset correlation (e.g., "^VIX").

    Returns:
        Panel DataFrame (timestamp, symbol + feature columns).
    """
    if prices_df is None or prices_df.empty:
        return pd.DataFrame()

    # Pivot to wide format
    wide = prices_df.pivot_table(
        index=timestamp_col, columns=symbol_col, values=close_col,
    )
    # Compute returns
    returns_wide = wide.pct_change(fill_method=None).dropna(how="all")

    if returns_wide.empty or returns_wide.shape[1] < 2:
        return pd.DataFrame()

    # Market-wide features (same for all symbols on a given date)
    regime_features = compute_correlation_regime_features(returns_wide)
    dispersion = compute_return_dispersion(returns_wide)
    regime_features["return_dispersion"] = dispersion

    # Sector dispersion (if available)
    if sector_map:
        sector_disp = compute_sector_dispersion(returns_wide, sector_map)
        regime_features = regime_features.join(sector_disp)

    # Per-asset benchmark correlation
    if benchmark_col and benchmark_col in returns_wide.columns:
        bench_ret = returns_wide[benchmark_col]
        other_cols = [c for c in returns_wide.columns if c != benchmark_col]
        bench_corr = compute_correlation_to_benchmark(
            returns_wide[other_cols], bench_ret, window=60,
        )

    # Convert market-wide features back to panel format
    rows = []
    for sym in returns_wide.columns:
        sym_regime = regime_features.copy()
        sym_regime[symbol_col] = sym
        sym_regime[timestamp_col] = sym_regime.index

        # Add per-asset benchmark correlation if computed
        if benchmark_col and benchmark_col in returns_wide.columns and sym != benchmark_col:
            corr_col = f"corr_to_benchmark_{sym}"
            if corr_col in bench_corr.columns:
                sym_regime["corr_to_benchmark"] = bench_corr[corr_col].values

        rows.append(sym_regime)

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    return result


__all__ = [
    "build_correlation_features_panel",
    "compute_avg_pairwise_correlation",
    "compute_correlation_regime_features",
    "compute_correlation_to_benchmark",
    "compute_return_dispersion",
    "compute_sector_dispersion",
]
